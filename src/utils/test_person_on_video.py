"""
test_person_on_video.py — 对 rallies_new 片段逐帧推理，输出带检测框的视频
用法：python src/utils/test_person_on_video.py
输出：results/video_person_test/
  - <rally_name>.mp4    带检测框的完整视频
  - summary.json        各 rally 统计（漏检帧数、误检帧数、得分）
"""
import os
import random
import json
import ctypes
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── 路径配置 ──────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent.parent
MODEL_PATH  = PROJECT_DIR / "runs" / "person_training" / "hard_neg_finetune_v12" / "weights" / "best.pt"
RALLY_DIR   = PROJECT_DIR / "data" / "rallies_new"
OUT_DIR     = PROJECT_DIR / "results" / "video_person_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_RALLIES = 10   # 随机选取的 rally 数量
CONF      = 0.4  # 推理置信度阈值

CLASS_NAMES = {0: "near", 1: "far"}
COLORS      = {0: (0, 200, 255), 1: (255, 100, 0)}  # BGR: 黄/橙


def get_short_path(path_str: str) -> str:
    try:
        buf = ctypes.create_unicode_buffer(260)
        if not hasattr(ctypes, "windll"):  # 非 Windows 直接用原路径
            return path_str
        ctypes.windll.kernel32.GetShortPathNameW(path_str, buf, 260)
        return buf.value or path_str
    except Exception:
        return path_str


def draw_detections(frame: np.ndarray, detections: list, frame_idx: int, total: int) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cls   = det["cls"]
        conf  = det["conf"]
        color = COLORS.get(cls, (200, 200, 200))
        label = f"{CLASS_NAMES.get(cls, cls)}:{conf:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 帧号 + 检测数（左下角）
    info = f"frame {frame_idx}/{total}  det={len(detections)}"
    cv2.putText(vis, info, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 无检测时红色警告
    if len(detections) == 0:
        cv2.putText(vis, "NO DETECTION", (w // 2 - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


def process_rally(model, video_path: Path, out_video_path: Path) -> dict:
    """逐帧推理整个 rally，写出带框视频，返回统计信息"""
    short = get_short_path(str(video_path))
    cap = cv2.VideoCapture(short)
    if not cap.isOpened():
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_short = get_short_path(str(out_video_path))
    writer = cv2.VideoWriter(
        out_short,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    frame_idx   = 0
    miss_frames = 0   # 完全没检测到的帧
    fp_frames   = 0   # 检测数 > 2（可能误检，网球场上通常只有2人）
    det_per_frame = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, conf=CONF, verbose=False)
        dets = []
        if results and results[0].boxes:
            for det in results[0].boxes:
                x1, y1, x2, y2 = [float(v) for v in det.xyxy[0]]
                dets.append({
                    "cls":  int(det.cls[0]),
                    "conf": float(det.conf[0]),
                    "box":  (x1, y1, x2, y2),
                })

        n = len(dets)
        det_per_frame.append(n)
        if n == 0:
            miss_frames += 1
        if n > 2:
            fp_frames += 1

        vis = draw_detections(frame, dets, frame_idx, total)
        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    miss_rate = miss_frames / frame_idx if frame_idx > 0 else 1.0
    score     = round(1.0 - miss_rate, 3)
    return {
        "total_frames": frame_idx,
        "miss_frames":  miss_frames,
        "fp_frames":    fp_frames,
        "miss_rate":    round(miss_rate, 3),
        "score":        score,
        "avg_det":      round(sum(det_per_frame) / len(det_per_frame), 2) if det_per_frame else 0,
    }


def collect_clips(rally_dir: Path) -> list[Path]:
    """递归收集所有 raw_clip.mp4"""
    clips = []
    for root, _, files in os.walk(str(rally_dir)):
        for f in files:
            if f == "raw_clip.mp4":
                clips.append(Path(root) / f)
    return clips


def main():
    clips = collect_clips(RALLY_DIR)
    if not clips:
        print(f"未找到视频，检查路径: {RALLY_DIR}")
        return

    random.seed(42)
    selected = random.sample(clips, min(N_RALLIES, len(clips)))
    print(f"共 {len(clips)} 个 rally，随机选取 {len(selected)} 个\n")

    print(f"加载模型: {MODEL_PATH}")
    model = YOLO(get_short_path(str(MODEL_PATH)))

    results = []
    for clip in selected:
        # 用 "比赛名_rally名" 作为输出文件名
        match_name = clip.parent.parent.name[:30]
        rally_name = clip.parent.name
        out_name   = f"{match_name}__{rally_name}.mp4"
        out_path   = OUT_DIR / out_name

        print(f"  {match_name} / {rally_name}")
        stats = process_rally(model, clip, out_path)
        if stats is None:
            print(f"    无法读取，跳过")
            continue

        print(f"    总帧={stats['total_frames']}, 漏检帧={stats['miss_frames']} "
              f"({stats['miss_rate']*100:.1f}%), 疑似误检帧={stats['fp_frames']}, "
              f"得分={stats['score']:.2f}")
        results.append({
            "match":    match_name,
            "rally":    rally_name,
            "output":   out_name,
            **stats,
        })

    # 排序
    results.sort(key=lambda x: x["score"], reverse=True)

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    good = [r for r in results if r["score"] >= 0.8]
    bad  = [r for r in results if r["score"] <  0.6]

    print(f"效果好（漏检率≤20%，{len(good)} 个）:")
    for r in good:
        print(f"   [{r['score']:.2f}] {r['output']}")

    print(f"\n效果差（漏检率>40%，{len(bad)} 个）:")
    for r in bad:
        print(f"   [{r['score']:.2f}] {r['output']}")

    print(f"\n视频输出: {OUT_DIR}")
    print(f"汇总:     {summary_path}")


if __name__ == "__main__":
    main()
