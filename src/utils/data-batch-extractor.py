"""data-batch-extractor.py — 批量回合数据提取流水线

功能：遍历 data/rallies_new/，对每个回合视频运行球场检测+姿态追踪，输出 JSON 标注
"""
import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_UTILS_DIR))

# =====================================================================
# 1. 配置区
# =====================================================================
DATASET_ROOT = os.path.join(_PROJECT_DIR, "data", "rallies_new")

COURT_MODEL_PATH = os.path.join(_PROJECT_DIR, "runs", "court_finetune", "court_14pts_ultimate", "weights", "best.pt")
TRACKER_MODEL_PATH = os.path.join(_PROJECT_DIR, "models", "court", "best.pt")
POSE_MODEL_PATH = os.path.join(_PROJECT_DIR, "models", "yolo", "yolo11x-pose.pt")

# 系统状态文件
PROGRESS_LOG = os.path.join(_PROJECT_DIR, "logs", "pipeline_progress.txt")
SUBPAR_LOG = os.path.join(_PROJECT_DIR, "logs", "pipeline_subpar.txt")
STATS_FILE = os.path.join(_PROJECT_DIR, "logs", "pipeline_stats.json")
ERROR_LOG = "data-tractor/pipeline_errors.txt"  # 记录打不开或损坏的视频


# =====================================================================
# 2. 状态管理器 (断点续传与动态均值)
# =====================================================================
class PipelineManager:
    def __init__(self):
        self.processed = set()
        self.global_stats = {"court_conf": 0, "player_conf": 0, "pose_conf": 0, "count": 0}
        self.load_state()

    def load_state(self):
        """读取断点续传进度"""
        if os.path.exists(PROGRESS_LOG):
            with open(PROGRESS_LOG, "r", encoding="utf-8") as f:
                self.processed = set(line.strip() for line in f if line.strip())

        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, "r", encoding="utf-8") as f:
                    self.global_stats = json.load(f)
            except:
                pass

        print(f"已加载进度: 完成 {len(self.processed)} 个片段，全局累计计算 {self.global_stats['count']} 次。")

    def mark_processed(self, clip_path):
        """记录片段已完成"""
        self.processed.add(str(clip_path))
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(f"{clip_path}\n")

    def log_error(self, clip_path, reason):
        """记录损坏视频，同时也标记为已处理(避免死循环重试)"""
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{clip_path}] - {reason}\n")
        self.mark_processed(clip_path)

    def update_and_check_stats(self, clip_path, clip_stats):
        """更新全局均值，并检查当前视频是否拉跨"""
        # 如果当前片段什么都没识别到，直接跳过均值计算
        if clip_stats["count"] == 0:
            return

        # 计算当前视频片段的平均置信度 (确保转换为原生 float)
        cur_court = float(clip_stats["court_conf"] / clip_stats["count"])
        cur_player = float(clip_stats["player_conf"] / clip_stats["count"])
        cur_pose = float(clip_stats["pose_conf"] / clip_stats["count"])

        # 获取历史全局均值 (避免除以 0)
        g_count = max(1, self.global_stats["count"])
        g_court = float(self.global_stats["court_conf"]) / g_count
        g_player = float(self.global_stats["player_conf"]) / g_count
        g_pose = float(self.global_stats["pose_conf"]) / g_count

        # 判断是否低于全局均值 (只有在有一定样本量后才开始判定)
        if self.global_stats["count"] > 5:
            is_subpar = False
            reasons = []
            if cur_court < g_court: reasons.append(f"球场线偏低({cur_court:.2f}<{g_court:.2f})")
            if cur_player < g_player: reasons.append(f"运动员偏低({cur_player:.2f}<{g_player:.2f})")
            if cur_pose < g_pose: reasons.append(f"肢体偏低({cur_pose:.2f}<{g_pose:.2f})")

            if reasons:
                with open(SUBPAR_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{clip_path} -> {' | '.join(reasons)}\n")

        # 修复点：将当前片段积分汇入全局大池子时，强制转换为原生 float 和 int
        self.global_stats["court_conf"] += float(clip_stats["court_conf"])
        self.global_stats["player_conf"] += float(clip_stats["player_conf"])
        self.global_stats["pose_conf"] += float(clip_stats["pose_conf"])
        self.global_stats["count"] += int(clip_stats["count"])

        # 实时保存全局状态
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.global_stats, f)


# =====================================================================
# 3. 核心提取器
# =====================================================================
def main():
    manager = PipelineManager()

    # 扫描所有视频
    dataset_path = Path(DATASET_ROOT)
    all_clips = list(dataset_path.rglob("raw_clip.mp4"))

    pending_clips = [c for c in all_clips if str(c) not in manager.processed]
    print(f"扫描到视频总数: {len(all_clips)}，剩余待处理: {len(pending_clips)}\n")

    if not pending_clips:
        print("所有视频已处理完毕！")
        return

    print("⏳ 正在加载模型...")
    court_model = YOLO(COURT_MODEL_PATH)
    tracker_model = YOLO(TRACKER_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

    for clip_idx, clip_path in enumerate(pending_clips):
        print(
            f"\n[{clip_idx + 1}/{len(pending_clips)}] 正在处理: {clip_path.parent.parent.name}/{clip_path.parent.name}")

        # 极端防崩溃保护层 1：文件不可读
        try:
            # 兼容中文路径的读取
            video_data = np.fromfile(str(clip_path), dtype=np.uint8)
            # OpenCV 无法直接用 imdecode 解码 mp4，这里改回常规 VideoCapture
            # 如果路径有中文导致 VideoCapture 失败，你需要把视频拷贝到英文临时目录，或者确保环境支持
            cap = cv2.VideoCapture(str(clip_path))
        except Exception as e:
            manager.log_error(clip_path, f"文件读取异常: {e}")
            continue

        if not cap.isOpened():
            manager.log_error(clip_path, "无法打开视频流 (可能是格式损坏)")
            continue

        clip_json_data = {"clip_info": str(clip_path), "frames": []}
        clip_stats = {"court_conf": 0.0, "player_conf": 0.0, "pose_conf": 0.0, "count": 0}
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_data = {"frame_id": frame_idx, "court": None, "players": []}
            h_img, w_img = frame.shape[:2]

            # 极端防崩溃保护层 2：推理过程全包裹
            try:
                # --- A. 球场提取 ---
                c_res = court_model.predict(frame, conf=0.3, verbose=False)[0]
                if c_res.keypoints is not None and len(c_res.keypoints.data) > 0:
                    kpts = c_res.keypoints.data[0].cpu().numpy().tolist()
                    frame_data["court"] = kpts  # 保存 14 个点的 [x, y, conf]
                    # 累加置信度用于均值计算
                    confs = [p[2] for p in kpts if p[2] > 0]
                    if confs: clip_stats["court_conf"] += sum(confs) / len(confs)

                # --- B. 运动员与肢体提取 ---
                t_res = tracker_model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
                if t_res.boxes is not None and t_res.boxes.id is not None:
                    ids = t_res.boxes.id.int().cpu().tolist()
                    bboxes = t_res.boxes.xyxy.cpu().numpy()
                    p_confs = t_res.boxes.conf.cpu().numpy()

                    for i, tid in enumerate(ids):
                        bx = bboxes[i].astype(int)
                        clip_stats["player_conf"] += p_confs[i]

                        player_info = {
                            "id": tid,
                            "bbox": bx.tolist(),
                            "bbox_conf": float(p_confs[i]),
                            "pose": None
                        }

                        # 50% 外扩截图
                        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
                        pad_x = int(bw * 0.25) + 10
                        pad_y = int(bh * 0.25) + 10

                        cx1, cy1 = max(0, bx[0] - pad_x), max(0, bx[1] - pad_y)
                        cx2, cy2 = min(w_img, bx[2] + pad_x), min(h_img, bx[3] + pad_y)

                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.shape[0] >= 10 and crop.shape[1] >= 10:
                            p_res = pose_model.predict(crop, imgsz=192, verbose=False)[0]
                            if p_res.keypoints is not None and len(p_res.keypoints.data) > 0:
                                kpts = p_res.keypoints.data[0].cpu().numpy().copy()

                                # 映射回全局
                                kpts[:, 0] += cx1
                                kpts[:, 1] += cy1

                                # 核心过滤：删除跑到“外扩区域”的无关肢体点
                                # 只保留在原本 BBox (bx[0]~bx[2], bx[1]~bx[3]) 内的点
                                valid_mask = (kpts[:, 0] >= bx[0]) & (kpts[:, 0] <= bx[2]) & \
                                             (kpts[:, 1] >= bx[1]) & (kpts[:, 1] <= bx[3])

                                # 将越界的点置信度强制设为 0，坐标归零
                                kpts[~valid_mask] = [0, 0, 0]

                                player_info["pose"] = kpts.tolist()

                                # 累加有效的肢体置信度
                                valid_confs = [p[2] for p in kpts if p[2] > 0]
                                if valid_confs: clip_stats["pose_conf"] += sum(valid_confs) / len(valid_confs)

                        frame_data["players"].append(player_info)

                clip_stats["count"] += 1
                clip_json_data["frames"].append(frame_data)
                frame_idx += 1

            except Exception as e:
                # 就算某一帧因为极度畸形的数据崩了，也只跳过这一帧，保全整个视频
                print(f"第 {frame_idx} 帧处理异常，已跳过。错误: {e}")
                continue

        cap.release()

        # 4. 结算当前视频片段
        # 将生成的 JSON 保存到对应视频的目录下
        json_save_path = clip_path.parent / "tracking_data.json"
        try:
            with open(json_save_path, "w", encoding="utf-8") as f:
                json.dump(clip_json_data, f, ensure_ascii=False)
        except Exception as e:
            manager.log_error(clip_path, f"JSON保存失败: {e}")
            continue

        # 更新大盘数据并检查及格线
        manager.update_and_check_stats(str(clip_path), clip_stats)

        # 写入断点进度
        manager.mark_processed(str(clip_path))
        print(f"   完成并保存。共提取 {frame_idx} 帧数据。")


if __name__ == "__main__":
    main()