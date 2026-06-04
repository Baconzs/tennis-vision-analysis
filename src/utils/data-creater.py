"""data-creater.py — 人员分类训练数据采集工具

功能：从 data/rallies_new/ 中随机抽帧，存入 data/person_sorter/image/ 供标注
"""
import os
import cv2
import random
import numpy as np
from pathlib import Path


def extract_frames_per_video_folder(base_dir, output_dir, frames_per_video=10):
    base_path = Path(base_dir)
    out_path = Path(output_dir)

    # 确保输出目录存在
    out_path.mkdir(parents=True, exist_ok=True)

    total_videos_processed = 0
    total_images_saved = 0

    print(f"开始扫描目录: {base_path}")

    # 1. 遍历十个大视频文件夹 (如 Video_01, Video_02...)
    for video_folder in base_path.iterdir():
        if not video_folder.is_dir():
            continue

        print(f"正在处理视频大类: {video_folder.name}")

        # 收集当前大文件夹下所有的 raw_clip.mp4 路径
        # rglob 可以递归查找所有层级下的目标文件
        clip_files = list(video_folder.rglob("raw_clip.mp4"))

        if not clip_files:
            print(f"{video_folder.name} 下未找到任何 raw_clip.mp4，跳过。")
            continue

        frames_collected = 0
        attempts = 0
        max_attempts = frames_per_video * 5  # 防止全损坏导致的死循环

        # 2. 循环抽取，直到凑够 10 张或者达到最大尝试次数
        while frames_collected < frames_per_video and attempts < max_attempts:
            attempts += 1

            # 随机挑选一个片段文件
            clip_file = random.choice(clip_files)

            # 使用 OpenCV 打开视频
            cap = cv2.VideoCapture(str(clip_file))
            if not cap.isOpened():
                # 忽略损坏的视频 (如 moov atom not found)
                cap.release()
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                continue

            # 随机生成一个帧索引
            random_frame_idx = random.randint(0, total_frames - 1)

            # 定位到该帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
            ret, frame = cap.read()

            if ret and frame is not None:
                # 构造文件名: 大文件夹名_小文件夹名_frame帧号.jpg
                clip_folder_name = clip_file.parent.name
                img_name = f"{video_folder.name}_{clip_folder_name}_frame{random_frame_idx:04d}.jpg"
                img_save_path = out_path / img_name

                # ！！！关键修复！！！
                # 使用 cv2.imencode 和 numpy 写入，完美解决 Windows 下中文路径无法保存图片的问题
                is_success, im_buf_arr = cv2.imencode(".jpg", frame)
                if is_success:
                    im_buf_arr.tofile(str(img_save_path))
                    frames_collected += 1
                    total_images_saved += 1

            # 释放视频句柄，准备下一次抽取
            cap.release()

        if frames_collected < frames_per_video:
            print(f"{video_folder.name} 仅成功提取 {frames_collected} 张 (可能是可用视频片段过少或损坏过多)")
        else:
            print(f"   成功提取 {frames_collected} 张图片")

        total_videos_processed += 1

    print("-" * 30)
    print("抽帧任务完美结束！")
    print(f"共处理大视频文件夹: {total_videos_processed} 个")
    print(f"共生成并保存标注图片: {total_images_saved} 张")
    print(f"图片保存路径: {out_path.absolute()}")


# ==========================================
# 执行区域
# ==========================================
if __name__ == "__main__":
    import os as _os
    _PROJECT_DIR = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    SOURCE_DIR = _os.path.join(_PROJECT_DIR, "data", "rallies_new")
    TARGET_DIR = _os.path.join(_PROJECT_DIR, "data", "person_sorter", "image")
    extract_frames_per_video_folder(SOURCE_DIR, TARGET_DIR, frames_per_video=10)