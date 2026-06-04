"""
在 player1/player2 裁剪图上重跑 pose 检测，低阈值，坐标映射回原始帧后用 person bbox 过滤。
覆盖 pose_data.json 中的 keypoints 字段（保留 bbox 和 court 字段）。
统计空检测帧，写入 logs/pose_rerun_stats.json。
支持断点续跑（--force 强制重跑）。
"""
import os
import json
import ctypes
import argparse
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CROP_SIZE = 320
CONF_THRESH = 0.1   # 极低阈值，尽量检出


def get_short_path(path):
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # 非 Windows 直接用原路径
        return path
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path


def _calc_win(bbox, base_win):
    """从 bbox [x1,y1,x2,y2] 和 base_win 计算裁剪窗口大小（与 extract_crops.py 一致）。"""
    box_side = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    return max(base_win, box_side)


def _crop_to_orig(x_crop, y_crop, cx, cy, win):
    """裁剪图坐标 → 原始帧坐标。"""
    x_orig = (x_crop / CROP_SIZE) * win + (cx - win / 2)
    y_orig = (y_crop / CROP_SIZE) * win + (cy - win / 2)
    return x_orig, y_orig


def _in_bbox(x, y, bbox):
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _run_pose_on_crop(crop_path, pose_model):
    """在裁剪图上跑 pose，返回 17 个关键点 [[x,y,conf],...] 或 None。"""
    if not os.path.exists(crop_path):
        return None
    raw = np.fromfile(crop_path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return None
    results = pose_model(img, verbose=False, conf=CONF_THRESH)
    if not results or results[0].keypoints is None:
        return None
    kps = results[0].keypoints
    if kps.xy is None or len(kps.xy) == 0:
        return None
    # 取置信度最高的那个人
    if kps.conf is not None and len(kps.conf) > 1:
        best = int(kps.conf.mean(dim=1).argmax())
    else:
        best = 0
    xy = kps.xy[best].cpu().numpy()
    conf = kps.conf[best].cpu().numpy() if kps.conf is not None else np.ones(len(xy))
    return [[float(xy[i, 0]), float(xy[i, 1]), float(conf[i])] for i in range(len(xy))]


def process_clip(clip_dir, pose_model, force=False):
    pose_path = os.path.join(clip_dir, "pose_data.json")
    video_path = os.path.join(clip_dir, "raw_clip.mp4")
    p1_dir = os.path.join(clip_dir, "player1")
    p2_dir = os.path.join(clip_dir, "player2")

    if not os.path.exists(pose_path):
        return None, "缺少 pose_data.json"
    if not os.path.isdir(p1_dir) or not os.path.isdir(p2_dir):
        return None, "缺少裁剪图目录"

    with open(pose_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    # 断点续跑：检查是否已处理（标记字段）
    if not force:
        first = pose_data[0] if isinstance(pose_data, list) and pose_data else None
        if first and first.get("_pose_rerun"):
            return None, "已跳过"

    # 获取视频宽度以计算 base_win
    short_path = get_short_path(video_path) if os.path.exists(video_path) else None
    base_win = 300  # 默认值
    if short_path:
        cap = cv2.VideoCapture(short_path)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        if vid_w > 0:
            base_win = round(vid_w / 6.4)

    is_list = isinstance(pose_data, list)
    entries = pose_data if is_list else list(pose_data.values())

    empty_near = empty_far = total = 0

    for entry in entries:
        if entry is None:
            continue
        fid = entry.get("frame", total)
        name = f"{fid:06d}.jpg"
        total += 1

        for role, crop_dir in [("near_player", p1_dir), ("far_player", p2_dir)]:
            player = entry.get(role)
            if player is None:
                if role == "near_player":
                    empty_near += 1
                else:
                    empty_far += 1
                continue

            bbox = player.get("bbox")
            if bbox is None:
                # bbox 为 None（插值帧），keypoints 保持空列表
                player["keypoints"] = []
                if role == "near_player":
                    empty_near += 1
                else:
                    empty_far += 1
                continue

            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            win = _calc_win(bbox, base_win)

            crop_path = os.path.join(crop_dir, name)
            raw_kps = _run_pose_on_crop(crop_path, pose_model)

            if raw_kps is None:
                # 保留原始关键点但置信度全零
                orig_kps = player.get("keypoints", [])
                new_kps = [[kp[0], kp[1], 0.0] for kp in orig_kps] if orig_kps else []
                player["keypoints"] = new_kps
                if role == "near_player":
                    empty_near += 1
                else:
                    empty_far += 1
                continue

            # 坐标映射回原始帧 + bbox 过滤
            new_kps = []
            for kp in raw_kps:
                x_orig, y_orig = _crop_to_orig(kp[0], kp[1], cx, cy, win)
                c = kp[2] if _in_bbox(x_orig, y_orig, bbox) else 0.0
                new_kps.append([x_orig, y_orig, c])

            player["keypoints"] = new_kps

        entry["_pose_rerun"] = True  # 标记已处理

    with open(pose_path, "w", encoding="utf-8") as f:
        json.dump(pose_data, f, ensure_ascii=False)

    stats = {
        "total_frames": total,
        "empty_near": empty_near,
        "empty_far": empty_far,
        "empty_near_pct": round(empty_near / total * 100, 1) if total else 0,
        "empty_far_pct": round(empty_far / total * 100, 1) if total else 0,
    }
    return stats, "完成"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(_PROJECT_DIR, "data", "rallies_annotated"))
    parser.add_argument("--model", default=os.path.join(_PROJECT_DIR, "models", "yolo", "yolo11x-pose.pt"))
    parser.add_argument("--force", action="store_true", help="强制重跑已处理的 rally")
    args = parser.parse_args()

    print(f"加载 pose 模型: {args.model}  置信度阈值: {CONF_THRESH}")
    pose_model = YOLO(args.model)

    clips = sorted(d for d in os.listdir(args.data_root)
                   if os.path.isdir(os.path.join(args.data_root, d)))

    all_stats = {}
    done = skipped = failed = 0

    for clip_name in tqdm(clips, desc="重跑 pose 检测"):
        clip_dir = os.path.join(args.data_root, clip_name)
        stats, msg = process_clip(clip_dir, pose_model, force=args.force)
        if msg == "已跳过":
            skipped += 1
        elif msg == "完成":
            done += 1
            all_stats[clip_name] = stats
        else:
            failed += 1
            tqdm.write(f"  [失败] {clip_name}: {msg}")

    # 写统计文件
    logs_dir = os.path.join(_PROJECT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    stats_path = os.path.join(logs_dir, "pose_rerun_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    # 汇总
    if all_stats:
        avg_near = np.mean([s["empty_near_pct"] for s in all_stats.values()])
        avg_far  = np.mean([s["empty_far_pct"]  for s in all_stats.values()])
        print(f"\n平均空检测率 — near_player: {avg_near:.1f}%，far_player: {avg_far:.1f}%")

    print(f"完成: {done} 个已处理，{skipped} 个已跳过，{failed} 个失败")
    print(f"统计文件: {stats_path}")


if __name__ == "__main__":
    main()
