"""main.py — 批量视频处理主入口

功能：遍历 videos/ 目录，对每场比赛视频运行完整追踪流水线，支持断点续跑
"""
import cv2
import numpy as np
import json
import os
import time
import threading
import queue
import torch
from ultralytics import YOLO

import config_legacy as config
from court_detector import CourtDetector
from pose_tracker import PoseTracker


class BatchTennisPipeline:
    def __init__(self):
        self.input_dir = config.VIDEO_PATH
        self.output_base_dir = config.OUTPUT_DIR

        # 获取目录下所有 mp4 文件并排序，确保处理顺序一致
        self.video_files = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith('.mp4')])
        if not self.video_files:
            raise FileNotFoundError(f"在 {self.input_dir} 中未找到任何 mp4 文件！")

        self.court_detector = CourtDetector(scale=config.SCOUT_SCALE)

        torch.backends.cudnn.benchmark = True

        # 全局进度状态
        self.current_video_idx = 0
        self.current_scout_frame = 0
        self.current_task_count = 0
        self.pending_queue_data = []  # 暂存断点时的队列数据

        self._load_checkpoint()

    def _load_checkpoint(self):
        """ 加载本地存档，恢复到特定的视频和特定的帧位 """
        if os.path.exists(config.CHECKPOINT_FILE):
            with open(config.CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.current_video_idx = data.get("video_idx", 0)
            self.current_scout_frame = data.get("scout_frame", 0)
            self.current_task_count = data.get("gpu_task_count", 0)
            self.pending_queue_data = data.get("pending_queue", [])

            # 越界检查
            if self.current_video_idx >= len(self.video_files):
                print("[*] 存档显示的视频已全部处理完毕，将从头开始。")
                self.current_video_idx = 0
                self.current_scout_frame = 0
                self.current_task_count = 0
                self.pending_queue_data = []
            else:
                resume_video = self.video_files[self.current_video_idx]
                print(f"[*] 读取存档成功。准备继续处理: {resume_video}")
                print(f"[*] 进度 -> CPU 帧位: {self.current_scout_frame}, GPU 已完成: {self.current_task_count}")

    def _save_checkpoint(self):
        """ 挂起时保存跨文件全局状态 """
        pending = list(self.task_queue.queue)
        state = {
            "video_idx": self.current_video_idx,
            "scout_frame": self.current_scout_frame,
            "gpu_task_count": self.current_task_count,
            "pending_queue": pending
        }
        with open(config.CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        print(f"[*] 进度已导出至 {config.CHECKPOINT_FILE}")

    def producer_scout_thread(self, video_path, total_frames, fps, width, height):
        print(f"[CPU] 巡视器启动 -> {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_scout_frame)

        is_active = False
        rally_start_frame = 0
        hits, misses = 0, 0
        current_far_rois, current_near_rois = [], []
        frame_idx = self.current_scout_frame

        HIT_BUF, MISS_BUF = 3, 6

        while cap.isOpened():
            if self.stop_event.is_set():
                self.current_scout_frame = frame_idx
                break

            ret, frame = cap.read()
            if not ret: break

            if frame_idx % config.SCOUT_SKIP_FRAMES == 0:
                far_roi, near_roi = self.court_detector.get_rois(frame, width, height)

                if far_roi is not None:
                    hits += 1
                    misses = 0
                    current_far_rois.append(far_roi)
                    current_near_rois.append(near_roi)

                    if hits > HIT_BUF and not is_active:
                        is_active = True
                        rally_start_frame = max(0, frame_idx - (HIT_BUF * config.SCOUT_SKIP_FRAMES))
                else:
                    misses += 1
                    hits = 0
                    if misses > MISS_BUF and is_active:
                        is_active = False
                        true_end = frame_idx - (MISS_BUF * config.SCOUT_SKIP_FRAMES)
                        duration = (true_end - rally_start_frame) / fps

                        if duration >= config.MIN_RALLY_DURATION:
                            task = {
                                'start': rally_start_frame,
                                'end': true_end,
                                'duration': duration,
                                'far_roi': np.median(current_far_rois, axis=0).astype(int).tolist(),
                                'near_roi': np.median(current_near_rois, axis=0).astype(int).tolist()
                            }
                            self.task_queue.put(task)
                            print(
                                f"[CPU] 回合入队 | 时长: {duration:.1f}s | 进度: {(frame_idx / total_frames) * 100:.1f}%")

                        current_far_rois.clear()
                        current_near_rois.clear()

            frame_idx += 1

        self.current_scout_frame = frame_idx
        cap.release()
        self.scout_finished.set()

    def consumer_yolo_thread(self, video_path, video_output_dir, fps, width, height):
        print("[GPU] 提取机启动")
        model = YOLO(config.MODEL_PATH)
        model.to('cuda:0')
        tracker = PoseTracker(model)

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        task_count = self.current_task_count

        while True:
            # 优先检查停止信号
            if self.stop_event.is_set():
                self.current_task_count = task_count
                print("[GPU] 收到挂起指令，完成当前片段后安全退出")
                break

            try:
                rally = self.task_queue.get(timeout=0.5)
            except queue.Empty:
                if self.scout_finished.is_set():
                    break
                continue

            task_count += 1
            duration = rally['duration']

            # 为每个回合创建专属子文件夹
            clip_name = f"rally_{task_count:03d}_{duration:.1f}s"
            clip_dir = os.path.join(video_output_dir, clip_name)
            os.makedirs(clip_dir, exist_ok=True)

            raw_path = os.path.join(clip_dir, "raw_clip.mp4")
            ann_path = os.path.join(clip_dir, "annotated_clip.mp4")
            json_path = os.path.join(clip_dir, "pose_data.json")

            out_raw = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
            out_ann = cv2.VideoWriter(ann_path, fourcc, fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, rally['start'])
            curr_frame = rally['start']
            json_data = []

            fx1, fy1, fx2, fy2 = rally['far_roi']
            nx1, ny1, nx2, ny2 = rally['near_roi']

            h_far = {'box': None, 'kpts': None, 'miss': 0}
            h_near = {'box': None, 'kpts': None, 'miss': 0}

            print(f"[GPU] 正在标注: {clip_name}")

            while curr_frame <= rally['end']:
                ret, frame = cap.read()
                if not ret: break

                out_raw.write(frame)
                ann_frame = frame.copy()
                f_data = {"frame": curr_frame, "far_player": None, "near_player": None}

                f_data["far_player"] = tracker.process_and_smooth(
                    frame[fy1:fy2, fx1:fx2], fx1, fy1, True, h_far, ann_frame)

                f_data["near_player"] = tracker.process_and_smooth(
                    frame[ny1:ny2, nx1:nx2], nx1, ny1, False, h_near, ann_frame)

                out_ann.write(ann_frame)
                json_data.append(f_data)
                curr_frame += 1

            out_raw.release()
            out_ann.release()
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)

            torch.cuda.empty_cache()
            self.current_task_count = task_count
            self.task_queue.task_done()
            print(f"[GPU] 片段完成: {clip_name}")

        cap.release()

    def process_single_video(self, video_path):
        """ 处理单个视频的完整生命周期 """
        temp_cap = cv2.VideoCapture(video_path)
        fps = temp_cap.get(cv2.CAP_PROP_FPS)
        width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_cap.release()

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_base_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        # 重置当前视频的队列与事件
        self.task_queue = queue.Queue()
        for item in self.pending_queue_data:
            self.task_queue.put(item)
        self.pending_queue_data = []  # 装载后清空缓冲

        self.scout_finished = threading.Event()
        self.stop_event = threading.Event()

        scout_t = threading.Thread(target=self.producer_scout_thread,
                                   args=(video_path, total_frames, fps, width, height))
        yolo_t = threading.Thread(target=self.consumer_yolo_thread,
                                  args=(video_path, video_output_dir, fps, width, height))

        scout_t.start()
        yolo_t.start()

        # 监控 control.txt
        while scout_t.is_alive() or yolo_t.is_alive():
            if os.path.exists(config.CONTROL_FILE):
                with open(config.CONTROL_FILE, "r", encoding="utf-8") as f:
                    cmd = f.read().strip().lower()
                if cmd == "save":
                    print("\n[*] 接收到保存指令，正在挂起工作线程...")
                    self.stop_event.set()
                    with open(config.CONTROL_FILE, "w", encoding="utf-8") as f:
                        f.write("saved")
                    break
            time.sleep(2)

        scout_t.join()
        yolo_t.join()

        # 如果是安全退出，返回 True；如果是正常处理完，返回 False
        return self.stop_event.is_set()

    def run(self):
        s_time = time.time()

        # 遍历目录下所有视频
        for idx in range(self.current_video_idx, len(self.video_files)):
            self.current_video_idx = idx
            video_file = self.video_files[idx]
            video_path = os.path.join(self.input_dir, video_file)

            print(f"\n{'=' * 50}")
            print(f"[*] 开始处理队列 ({idx + 1}/{len(self.video_files)}): {video_file}")
            print(f"{'=' * 50}")

            # 运行当前视频，并接收是否被人工打断的信号
            is_stopped = self.process_single_video(video_path)

            if is_stopped:
                self._save_checkpoint()
                print(f"[*] 断点已成功保存。随时可以安全关闭程序。")
                return  # 打断整个批处理流程
            else:
                print(f"[*] 视频 {video_file} 处理完毕。")
                # 为下一个视频重置计数器
                self.current_scout_frame = 0
                self.current_task_count = 0

        print(f"\n[!!!] 文件夹内所有视频批量处理完成 [!!!]")
        print(f"总耗时: {(time.time() - s_time) / 60:.2f} 分钟。")
        if os.path.exists(config.CHECKPOINT_FILE):
            os.remove(config.CHECKPOINT_FILE)


if __name__ == '__main__':
    pipeline = BatchTennisPipeline()
    pipeline.run()