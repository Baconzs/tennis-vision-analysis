"""
hard_negative_extractor.py — 从 pipeline_subpar.txt 中提取识别率低的帧

用法：
    python src/utils/hard_negative_extractor.py
    python src/utils/hard_negative_extractor.py --frames-per-clip 3 --conf-threshold 0.4

输出：
    data/person_sorter/hard_negatives/images/  — 提取的帧图像
    data/person_sorter/hard_negatives/manifest.csv — 帧来源记录
"""

import os
import csv
import argparse
from pathlib import Path

import ctypes

import cv2
import numpy as np
from ultralytics import YOLO


def _short_path(path: str) -> str:
    """将含中文的 Windows 路径转为 8.3 短路径，供 cv2 使用"""
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # 非 Windows 直接用原路径
        return path
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path


def imwrite_unicode(path: str, img) -> bool:
    """支持中文路径的图片写入"""
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

SUBPAR_LOG = os.path.join(_PROJECT_DIR, "logs", "pipeline_subpar.txt")
MODEL_PATH = os.path.join(_PROJECT_DIR, "models", "person", "best.pt")
OUTPUT_DIR = os.path.join(_PROJECT_DIR, "data", "person_sorter", "hard_negatives", "images")
MANIFEST_PATH = os.path.join(_PROJECT_DIR, "data", "person_sorter", "hard_negatives", "manifest.csv")


def parse_subpar_log(log_path: str, problem_type: str) -> list[dict]:
    """解析 pipeline_subpar.txt，返回匹配问题类型的片段列表"""
    keyword_map = {
        "player": "运动员偏低",
        "pose": "肢体偏低",
        "court": "球场线偏低",
    }

    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" -> ", 1)
            if len(parts) != 2:
                continue
            video_path, issues_str = parts[0].strip(), parts[1].strip()

            if problem_type == "all":
                matched = True
            else:
                keyword = keyword_map.get(problem_type, "")
                matched = keyword in issues_str

            if matched:
                entries.append({"video_path": video_path, "issues": issues_str})

    return entries


def extract_worst_frames(
    video_path: str,
    model: YOLO,
    frames_per_clip: int,
    conf_threshold: float,
) -> list[dict]:
    """
    对单个视频跑推理，返回置信度最低的 N 帧信息。
    返回格式：[{"frame_id": int, "frame": ndarray, "min_conf": float}]
    """
    cap = cv2.VideoCapture(_short_path(video_path))
    if not cap.isOpened():
        print(f"  [跳过] 无法打开: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 均匀采样，最多采 60 帧避免处理太慢
    sample_step = max(1, total // 60)

    frame_scores = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % sample_step == 0:
            results = model.predict(frame, verbose=False, conf=conf_threshold)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                min_conf = float(boxes.conf.min().item())
            else:
                # 完全没检测到人，置信度记为 0
                min_conf = 0.0
            frame_scores.append({"frame_id": frame_id, "frame": frame.copy(), "min_conf": min_conf})
        frame_id += 1

    cap.release()

    if not frame_scores:
        return []

    # 按置信度升序，取最差的 N 帧
    frame_scores.sort(key=lambda x: x["min_conf"])
    return frame_scores[:frames_per_clip]


def main():
    parser = argparse.ArgumentParser(description="提取识别率低的帧用于重新标注")
    parser.add_argument("--problem-type", default="player",
                        choices=["player", "pose", "court", "all"],
                        help="过滤问题类型 (默认: player)")
    parser.add_argument("--frames-per-clip", type=int, default=5,
                        help="每个片段提取几帧 (默认: 5)")
    parser.add_argument("--conf-threshold", type=float, default=0.3,
                        help="YOLO 推理置信度阈值 (默认: 0.3)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"解析日志: {SUBPAR_LOG}")
    entries = parse_subpar_log(SUBPAR_LOG, args.problem_type)
    print(f"找到 {len(entries)} 个问题片段 (类型: {args.problem_type})")

    manifest_rows = []
    saved_count = 0

    for i, entry in enumerate(entries):
        video_path = entry["video_path"]
        # 路径可能是旧路径，尝试映射到当前项目目录
        if not os.path.exists(video_path):
            # 尝试从路径末尾提取相对部分重新拼接
            # 旧路径格式: <old_dataset>\【xx】...\rally_xxx\raw_clip.mp4
            parts = Path(video_path).parts
            try:
                # 找到比赛文件夹（以【开头）的位置
                match_idx = next(j for j, p in enumerate(parts) if p.startswith("【"))
                rel_path = os.path.join(*parts[match_idx:])
                new_path = os.path.join(_PROJECT_DIR, "data", "rallies_new", rel_path)
                if os.path.exists(new_path):
                    video_path = new_path
                else:
                    print(f"  [{i+1}/{len(entries)}] 文件不存在，跳过: {entry['video_path']}")
                    continue
            except StopIteration:
                print(f"  [{i+1}/{len(entries)}] 无法解析路径，跳过: {entry['video_path']}")
                continue

        print(f"  [{i+1}/{len(entries)}] 处理: {Path(video_path).parent.name}")
        worst_frames = extract_worst_frames(
            video_path, model, args.frames_per_clip, args.conf_threshold
        )

        # 用比赛名+回合名作为文件名前缀
        match_name = Path(video_path).parent.parent.name
        rally_name = Path(video_path).parent.name
        prefix = f"{match_name}_{rally_name}"

        for finfo in worst_frames:
            fname = f"{prefix}_frame{finfo['frame_id']:04d}.jpg"
            out_path = os.path.join(OUTPUT_DIR, fname)
            imwrite_unicode(out_path, finfo["frame"])
            manifest_rows.append({
                "filename": fname,
                "source_video": entry["video_path"],
                "frame_id": finfo["frame_id"],
                "min_conf": round(finfo["min_conf"], 4),
                "issues": entry["issues"],
            })
            saved_count += 1

    # 写 manifest.csv
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "source_video", "frame_id", "min_conf", "issues"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n完成！共提取 {saved_count} 帧")
    print(f"图像保存至: {OUTPUT_DIR}")
    print(f"清单保存至: {MANIFEST_PATH}")
    print(f"\n下一步：用 LabelImg 打开 {OUTPUT_DIR} 进行标注")
    print("标注格式选 YOLO，类别：0=player_near, 1=player_far")


if __name__ == "__main__":
    main()
