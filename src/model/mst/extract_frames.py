"""extract_frames.py — 预提取全帧 JPEG，供 dataset.py 直接读图（跳过视频 seek）

用法: python extract_frames.py [--data_root ...]
输出: 每个回合目录下生成 frames/{000000.jpg, ...}
"""
import os
import ctypes
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def get_short_path(path):
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # 非 Windows 直接用原路径
        return path
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path


def save_jpg(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def extract_clip(clip_dir):
    frames_dir = os.path.join(clip_dir, "frames")
    video_path = os.path.join(clip_dir, "raw_clip.mp4")

    if not os.path.exists(video_path):
        return "no_video"

    # 断点续跑：目录存在且帧数与视频一致则跳过
    short_path = get_short_path(video_path)
    cap = cv2.VideoCapture(short_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total <= 0:
        return "no_frames"

    if os.path.isdir(frames_dir):
        existing = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        if existing >= total:
            return "skip"

    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(short_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # resize 到训练用分辨率，节省磁盘 & 加快读取
        frame = cv2.resize(frame, (320, 192))
        save_jpg(os.path.join(frames_dir, f"{idx:06d}.jpg"), frame)
        idx += 1
    cap.release()
    return f"ok:{idx}"


def main():
    parser = argparse.ArgumentParser()
    _mst_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(os.path.dirname(os.path.dirname(_mst_dir)))
    parser.add_argument("--data_root", default=os.path.join(_project_dir, "data", "rallies_train"))
    args = parser.parse_args()

    clips = sorted(d for d in os.listdir(args.data_root)
                   if os.path.isdir(os.path.join(args.data_root, d)))

    skipped = done = failed = 0
    for clip_name in tqdm(clips, desc="提取全帧"):
        result = extract_clip(os.path.join(args.data_root, clip_name))
        if result == "skip":
            skipped += 1
        elif result.startswith("ok"):
            done += 1
        else:
            failed += 1
            print(f"  跳过 {clip_name}: {result}")

    print(f"\n完成: {done} 个回合已提取，{skipped} 个已跳过，{failed} 个失败。")


if __name__ == "__main__":
    main()
