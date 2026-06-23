"""inference_viewer.py — 人员分类模型推理可视化工具"""
import os
import cv2
from pathlib import Path
from ultralytics import YOLO

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_UTILS_DIR))

# ================= 配置区域 =================
MODEL_PATH = os.path.join(_PROJECT_DIR, "models", "person", "best.pt")
DATA_DIR = os.path.join(_PROJECT_DIR, "data", "rallies_new")
CONFIDENCE_THRESHOLD = 0.4


def main():
    print("⏳ 正在加载 YOLO26x 模型至显存...")
    model = YOLO(MODEL_PATH)

    # 1. 扫描并构建视频列表
    base_path = Path(DATA_DIR)
    video_files = list(base_path.rglob("raw_clip.mp4"))

    if not video_files:
        print("未找到视频文件，请检查 DATA_DIR 路径！")
        return

    print(f"成功加载 {len(video_files)} 个视频片段")
    print("-" * 30)
    print("视频列表索引:")
    for idx, vf in enumerate(video_files):
        # 提取 父文件夹/片段文件夹 的名称用于显示
        print(f"[{idx:02d}] {vf.parent.parent.name} / {vf.parent.name}")
    print("-" * 30)

    # 2. 全局状态控制
    cap = None
    is_paused = False
    update_trackbar_auto = False  # 防止进度条回调发生死循环的锁

    # 初始化窗口
    window_name = "YOLO Tennis Tracker (By 毕业设计)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)  # 初始窗口大小

    # 3. 核心交互回调函数
    def load_video(idx):
        """切换视频片段"""
        nonlocal cap, is_paused
        if cap is not None:
            cap.release()

        vid_path = str(video_files[idx])
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 动态更新进度条的最大值
        cv2.setTrackbarMax("Progress", window_name, max(1, total_frames - 1))
        cv2.setTrackbarPos("Progress", window_name, 0)
        is_paused = False
        print(f"\n▶正在播放: {video_files[idx].parent.parent.name} / {video_files[idx].parent.name}")

    def on_video_trackbar(val):
        """拖动列表条时触发"""
        load_video(val)

    def on_progress_trackbar(val):
        """拖动进度条时触发"""
        nonlocal update_trackbar_auto
        # 如果是程序自动更新的进度条，则不执行操作
        if not update_trackbar_auto and cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, val)

            # 即使在暂停状态下拖动，也要立刻推理并刷新当前帧画面
            ret, frame = cap.read()
            if ret:
                results = model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
                annotated_frame = results[0].plot()
                cv2.imshow(window_name, annotated_frame)
                # 读完一帧后，把指针退回拖动的位置，保持原状态
                cap.set(cv2.CAP_PROP_POS_FRAMES, val)

    # 4. 创建 UI 控件 (进度条和列表条)
    cv2.createTrackbar("Playlist", window_name, 0, len(video_files) - 1, on_video_trackbar)
    cv2.createTrackbar("Progress", window_name, 0, 100, on_progress_trackbar)

    # 启动加载第一个视频
    load_video(0)

    # 5. 主循环
    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                # 视频播放完毕，自动重头循环
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 获取当前帧号，并同步给进度条 UI
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            update_trackbar_auto = True  # 上锁
            cv2.setTrackbarPos("Progress", window_name, current_frame)
            update_trackbar_auto = False  # 解锁

            # ================= YOLO 推理核心 =================
            # verbose=False 防止终端一直刷屏打印检测信息
            results = model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

            # 获取绘制了检测框的图像 (黄色近景框，紫色黑衣远景框)
            annotated_frame = results[0].plot()

            # 绘制快捷键操作提示
            ui_text = "Space: Pause/Play | Drag to Seek | Q: Quit"
            cv2.putText(annotated_frame, ui_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window_name, annotated_frame)

        # 键盘事件监听 (等待 30ms 约等于 33 FPS)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:  # Q键 或 ESC
            break
        elif key == 32:  # 空格键: 切换暂停/播放状态
            is_paused = not is_paused
            state_str = "⏸暂停" if is_paused else "▶播放"
            print(f"[{state_str}]")

    # 收尾清理工作
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()