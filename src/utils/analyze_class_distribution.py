"""
分析数据集五类动作的帧分布。
用法:
  python src/utils/analyze_class_distribution.py
  python src/utils/analyze_class_distribution.py --root data/rallies_train_trimmed
"""

import json
import os
import argparse
from collections import Counter
from pathlib import Path

ACTION_NAMES = {0: "等待", 1: "正手", 2: "反手", 3: "发球", 4: "移动"}
FPS = 30


def analyze(data_root):
    data_root = Path(data_root)
    if not data_root.exists():
        print(f"错误：目录不存在 {data_root}")
        return

    rally_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    total_frames_per_class = Counter()
    total_segments_per_class = Counter()
    total_video_frames = 0
    per_rally = []

    for rally_dir in rally_dirs:
        ann_path = rally_dir / "annotations.json"
        if not ann_path.exists():
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        # 统计视频总帧数（从 raw_clip.mp4 获取）
        video_path = rally_dir / "raw_clip.mp4"
        if video_path.exists():
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            # 回退：取最后一段的 end_time * FPS
            if annotations:
                video_frames = int(round(annotations[-1]["end_time"] * FPS))
            else:
                video_frames = 0

        total_video_frames += video_frames

        class_frames = Counter()
        class_segments = Counter()
        for seg in annotations:
            aid = seg.get("action_id")
            dur = seg["end_time"] - seg["start_time"]
            n_frames = int(round(dur * FPS))
            class_frames[aid] += n_frames
            class_segments[aid] += 1

        total_frames_per_class += class_frames
        total_segments_per_class += class_segments

        per_rally.append({
            "name": rally_dir.name,
            "video_frames": video_frames,
            "class_frames": dict(class_frames),
        })

    # ── 打印报告 ──
    labeled_frames = sum(total_frames_per_class.values())

    print("=" * 60)
    print(f"数据集:  {data_root}")
    print(f"片段数:  {len(rally_dirs)}")
    print(f"视频总帧: {total_video_frames}")
    print(f"标注总帧: {labeled_frames} ({labeled_frames/total_video_frames*100:.1f}% 的视频帧被标注)")
    print()

    # 各类分布
    print(f"{'动作':>6}  {'帧数':>8}  {'占比':>6}  {'段数':>6}  {'均长(帧)':>8}  {'均长(s)':>8}")
    print("-" * 60)
    for aid in sorted(ACTION_NAMES):
        nf = total_frames_per_class.get(aid, 0)
        ns = total_segments_per_class.get(aid, 0)
        pct = nf / labeled_frames * 100 if labeled_frames > 0 else 0
        avg = nf / ns if ns > 0 else 0
        avg_s = avg / FPS
        print(f"{ACTION_NAMES[aid]:>6}  {nf:>8}  {pct:>5.1f}%  {ns:>6}  {avg:>8.1f}  {avg_s:>8.3f}")

    print("-" * 60)

    # 等待中来自末尾和开头的占比（粗略估计）
    print()
    wait_leading = 0
    wait_trailing = 0
    wait_middle = 0
    for r in per_rally:
        ann_path = data_root / r["name"] / "annotations.json"
        if not ann_path.exists():
            continue
        with open(ann_path, "r", encoding="utf-8") as f:
            annots = json.load(f)
        for i, seg in enumerate(annots):
            if seg.get("action_id") != 0:
                continue
            dur = seg["end_time"] - seg["start_time"]
            nf = int(round(dur * FPS))
            if i == 0:
                wait_leading += nf
            elif i == len(annots) - 1:
                # 需要检查倒数第二段是不是也是等待（连续等待末尾）
                wait_trailing += nf
            else:
                wait_middle += nf

    print("等待帧细分（按位置）:")
    print(f"  开头等待:     {wait_leading:>8} 帧 ({wait_leading/ max(labeled_frames,1)*100:.1f}%)")
    print(f"  中间等待:     {wait_middle:>8} 帧 ({wait_middle/ max(labeled_frames,1)*100:.1f}%)")
    print(f"  末尾等待:     {wait_trailing:>8} 帧 ({wait_trailing/ max(labeled_frames,1)*100:.1f}%)")

    print()
    print("每片段统计（按等待占比排序 top15）:")
    print(f"{'片段':>24}  {'总帧':>6}  {'等待帧':>6}  {'等待%':>6}  {'正手':>5}  {'反手':>5}  {'发球':>5}  {'移动':>5}")
    per_rally.sort(key=lambda r: r["class_frames"].get(0, 0) / max(sum(r["class_frames"].values()), 1), reverse=True)
    for r in per_rally[:15]:
        cf = r["class_frames"]
        total = sum(cf.values())
        wait_pct = cf.get(0, 0) / total * 100 if total > 0 else 0
        fh = cf.get(1, 0)
        bh = cf.get(2, 0)
        sv = cf.get(3, 0)
        mv = cf.get(4, 0)
        name = r["name"][:24]
        print(f"{name:>24}  {total:>6}  {cf.get(0,0):>6}  {wait_pct:>5.1f}%  {fh:>5}  {bh:>5}  {sv:>5}  {mv:>5}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析数据集五类动作帧分布")
    parser.add_argument("--root", default="data/rallies_train", help="数据集根目录")
    args = parser.parse_args()
    analyze(args.root)
