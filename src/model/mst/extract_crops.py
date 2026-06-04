"""extract_crops.py — 预提取运动员裁剪图 + 生成 pose_data.json bbox

用法: python extract_crops.py [--data_root ...] [--model ...]
输出: 每个回合目录下生成 player1/{000000.jpg,...}、player2/{...}，并将 bbox 写入 pose_data.json
"""
import os
import json
import ctypes
import argparse
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm

CROP_SIZE = 320


def get_short_path(path):
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # 非 Windows 直接用原路径
        return path
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path


def save_jpg(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def crop_fixed_window(frame, cx, cy, win, h, w):
    """以 (cx,cy) 为中心裁出 win×win 窗口，出界补黑，resize 到 CROP_SIZE×CROP_SIZE。"""
    half = win // 2
    x1s, y1s = cx - half, cy - half   # 源坐标（可能为负）
    x2s, y2s = x1s + win, y1s + win

    # 目标画布
    canvas = np.zeros((win, win, 3), dtype=np.uint8)

    # 源/目标的有效交叉区域
    sx1, sy1 = max(0, x1s), max(0, y1s)
    sx2, sy2 = min(w, x2s), min(h, y2s)
    dx1, dy1 = sx1 - x1s, sy1 - y1s
    dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)

    if sx2 > sx1 and sy2 > sy1:
        canvas[dy1:dy2, dx1:dx2] = frame[sy1:sy2, sx1:sx2]

    return cv2.resize(canvas, (CROP_SIZE, CROP_SIZE))


def extract_clip(clip_dir, model, placeholder):
    p1_dir = os.path.join(clip_dir, "player1")
    p2_dir = os.path.join(clip_dir, "player2")
    pose_path = os.path.join(clip_dir, "pose_data.json")

    if (os.path.isdir(p1_dir) and os.listdir(p1_dir) and
            os.path.isdir(p2_dir) and os.listdir(p2_dir)):
        return "skip"

    video_path = os.path.join(clip_dir, "raw_clip.mp4")
    if not os.path.exists(video_path):
        return "no_video"

    os.makedirs(p1_dir, exist_ok=True)
    os.makedirs(p2_dir, exist_ok=True)

    short_path = get_short_path(video_path)
    cap = cv2.VideoCapture(short_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base_win = round(vid_w / 6.4)

    # 读取已有 pose_data.json（如果存在，保留 court 字段）
    existing_pose = []
    if os.path.exists(pose_path):
        with open(pose_path, "r", encoding="utf-8") as f:
            existing_pose = json.load(f)
            if not isinstance(existing_pose, list):
                existing_pose = []

    # ── 第一遍：检测，记录每帧的 (cx, cy, win, bbox) 或 None ──────────────────────
    dets = []   # list of [slot0: dict|None, slot1: dict|None]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_dets = [None, None]
        results = model(frame, verbose=False, conf=0.3)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            cls  = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            for slot, target_cls in enumerate([0, 1]):
                mask = cls == target_cls
                if not mask.any():
                    continue
                best = np.argmax(conf[mask])
                box  = xyxy[mask][best]
                cx   = int((box[0] + box[2]) / 2)
                cy   = int((box[1] + box[3]) / 2)
                box_side   = int(max(box[2] - box[0], box[3] - box[1]))
                actual_win = max(base_win, box_side)
                frame_dets[slot] = {
                    "cx": cx, "cy": cy, "win": actual_win,
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                }
        dets.append(frame_dets)
    cap.release()

    # ── 线性插值补全缺失帧（只插值 cx/cy/win，bbox 保持 None）────────────────
    for slot in [0, 1]:
        known = [(i, dets[i][slot]) for i in range(len(dets)) if dets[i][slot] is not None]
        if not known:
            continue
        for i in range(len(dets)):
            if dets[i][slot] is not None:
                continue
            prev = next((k for k in reversed(known) if k[0] < i), None)
            nxt  = next((k for k in known if k[0] > i), None)
            if prev is None:
                d = nxt[1]
                dets[i][slot] = {"cx": d["cx"], "cy": d["cy"], "win": d["win"], "bbox": None}
            elif nxt is None:
                d = prev[1]
                dets[i][slot] = {"cx": d["cx"], "cy": d["cy"], "win": d["win"], "bbox": None}
            else:
                t = (i - prev[0]) / (nxt[0] - prev[0])
                cx  = int(prev[1]["cx"] + t * (nxt[1]["cx"] - prev[1]["cx"]))
                cy  = int(prev[1]["cy"] + t * (nxt[1]["cy"] - prev[1]["cy"]))
                win = int(prev[1]["win"] + t * (nxt[1]["win"] - prev[1]["win"]))
                dets[i][slot] = {"cx": cx, "cy": cy, "win": win, "bbox": None}

    # ── 第二遍：按插值位置裁图并保存，同时构建 pose_data ─────────────────────
    cap = cv2.VideoCapture(short_path)
    pose_data = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        name = f"{frame_idx:06d}.jpg"

        # 保存裁剪图
        for slot, out_dir in enumerate([p1_dir, p2_dir]):
            det = dets[frame_idx][slot]
            if det:
                img = crop_fixed_window(frame, det["cx"], det["cy"], det["win"], h, w)
            else:
                img = placeholder
            save_jpg(os.path.join(out_dir, name), img)

        # 构建 pose_data 条目
        entry = {"frame": frame_idx}

        # 保留已有的 court 字段
        if frame_idx < len(existing_pose) and isinstance(existing_pose[frame_idx], dict):
            entry["court"] = existing_pose[frame_idx].get("court")
        else:
            entry["court"] = None

        # 写入 near_player (slot=0, cls=0) 和 far_player (slot=1, cls=1)
        near_det = dets[frame_idx][0]
        far_det = dets[frame_idx][1]

        # 即使 bbox 是 None（插值帧），也写入结构，方便 rerun_pose_detection.py 处理
        entry["near_player"] = {
            "bbox": near_det["bbox"] if near_det else None,
            "keypoints": []  # 空列表，等待 rerun_pose_detection.py 填充
        }

        entry["far_player"] = {
            "bbox": far_det["bbox"] if far_det else None,
            "keypoints": []
        }

        pose_data.append(entry)
        frame_idx += 1
    cap.release()

    # 保存 pose_data.json
    with open(pose_path, "w", encoding="utf-8") as f:
        json.dump(pose_data, f, ensure_ascii=False, indent=2)

    return f"ok:{frame_idx}"


def main():
    parser = argparse.ArgumentParser()
    _utils_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(os.path.dirname(os.path.dirname(_utils_dir)))
    parser.add_argument("--data_root", default=os.path.join(_project_dir, "data", "rallies_annotated"))
    parser.add_argument("--model", default=os.path.join(_project_dir, "models", "person", "best.pt"))
    args = parser.parse_args()

    print(f"加载 person 模型: {args.model}")
    model = YOLO(args.model)

    placeholder = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)

    clips = [d for d in os.listdir(args.data_root)
             if os.path.isdir(os.path.join(args.data_root, d))]
    clips.sort()

    skipped = done = failed = 0
    for clip_name in tqdm(clips, desc="提取裁剪图"):
        clip_dir = os.path.join(args.data_root, clip_name)
        result = extract_clip(clip_dir, model, placeholder)
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
