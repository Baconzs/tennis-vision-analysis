"""
数据质量可视化工具
用法: python src/utils/visualize_data_quality.py [--rally <rally目录>] [--frame <帧号>]
不指定 rally 时随机选片段（跨所有 rally 随机）。
输出：
  1. 主视图：原始帧叠加球场线 + 运动员框 + 肢体关键点
  2. 裁剪图：near_player / far_player 的预提取裁剪图 + bbox 区域叠加关键点
按 n/p 切换帧，r 随机跳到任意 rally 的任意帧，q 退出。
"""
import os
import json
import ctypes
import random
import argparse
import numpy as np
import cv2

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
_NEAR_COLOR  = (0, 255, 255)
_FAR_COLOR   = (255, 128, 0)
_COURT_COLOR = (0, 200, 255)
_BONE_COLOR  = (200, 200, 200)
_CONF_THRESH = 0.1


def get_short_path(path):
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # 非 Windows 直接用原路径
        return path
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path


def draw_keypoints(img, kps, kp_color):
    if not kps:
        return
    pts = []
    for kp in kps:
        x, y, c = float(kp[0]), float(kp[1]), float(kp[2])
        pts.append((int(x), int(y), c))

    for i, j in _SKELETON:
        if i < len(pts) and j < len(pts):
            xi, yi, ci = pts[i]
            xj, yj, cj = pts[j]
            if ci >= _CONF_THRESH and cj >= _CONF_THRESH:
                cv2.line(img, (xi, yi), (xj, yj), _BONE_COLOR, 2)

    for x, y, c in pts:
        if c >= _CONF_THRESH:
            cv2.circle(img, (x, y), 5, kp_color, -1)
            cv2.circle(img, (x, y), 5, (0, 0, 0), 1)


def draw_bbox(img, bbox, color, label=""):
    if not bbox:
        return
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_court(img, court_kps):
    if not court_kps:
        return
    for i, kp in enumerate(court_kps):
        if len(kp) < 3:
            continue
        x, y, c = float(kp[0]), float(kp[1]), float(kp[2])
        if c >= 0.3:
            cv2.circle(img, (int(x), int(y)), 7, _COURT_COLOR, -1)
            cv2.circle(img, (int(x), int(y)), 7, (0, 0, 0), 1)
            cv2.putText(img, str(i), (int(x) + 8, int(y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COURT_COLOR, 1)


def make_crop_panel(frame, player_data, crop_path, color, label):
    """返回 320x640 面板：左=预提取裁剪图，右=bbox区域+关键点。"""
    panel = np.zeros((320, 640, 3), dtype=np.uint8)

    # 左：预提取裁剪图
    if crop_path and os.path.exists(crop_path):
        raw = np.fromfile(crop_path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is not None:
            panel[:, :320] = cv2.resize(img, (320, 320))
    cv2.putText(panel, f"{label} 预提取", (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # 右：从原帧裁 bbox，叠加关键点
    if player_data and player_data.get("bbox") and frame is not None:
        h, w = frame.shape[:2]
        b = player_data["bbox"]
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(w, int(b[2])), min(h, int(b[3]))
        if x2 > x1 and y2 > y1:
            patch = cv2.resize(frame[y1:y2, x1:x2].copy(), (320, 320))
            ph, pw = (y2 - y1), (x2 - x1)
            kps = player_data.get("keypoints", [])
            mapped = [[(float(kp[0]) - x1) / pw * 320,
                       (float(kp[1]) - y1) / ph * 320,
                       float(kp[2])] for kp in kps]
            draw_keypoints(patch, mapped, color)
            panel[:, 320:] = patch
    cv2.putText(panel, f"{label} bbox+关键点", (324, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    return panel


def get_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


def load_entry(pose_data, idx):
    if isinstance(pose_data, list):
        return pose_data[idx] if idx < len(pose_data) else None
    return pose_data.get(str(idx))


def build_index(data_root):
    """返回 [(rally_dir, total_frames), ...] 列表。"""
    index = []
    for name in sorted(os.listdir(data_root)):
        d = os.path.join(data_root, name)
        pose_path = os.path.join(d, "pose_data.json")
        video_path = os.path.join(d, "raw_clip.mp4")
        if not os.path.exists(pose_path) or not os.path.exists(video_path):
            continue
        with open(pose_path, "r", encoding="utf-8") as f:
            pd = json.load(f)
        total = len(pd) if isinstance(pd, list) else (max(int(k) for k in pd) + 1)
        if total > 0:
            index.append((d, total))
    return index


def run(data_root, init_rally=None, init_frame=None):
    print("构建索引...")
    index = build_index(data_root)
    if not index:
        print("没有找到有效的 rally 目录")
        return
    print(f"共 {len(index)} 个 rally。操作: n=下一帧  p=上一帧  r=随机片段  q=退出")

    # 初始选择
    if init_rally:
        rally_dir = init_rally
        pose_path = os.path.join(rally_dir, "pose_data.json")
        with open(pose_path, "r", encoding="utf-8") as f:
            pose_data = json.load(f)
        total = len(pose_data) if isinstance(pose_data, list) else (max(int(k) for k in pose_data) + 1)
    else:
        rally_dir, total = random.choice(index)
        pose_path = os.path.join(rally_dir, "pose_data.json")
        with open(pose_path, "r", encoding="utf-8") as f:
            pose_data = json.load(f)

    cap = cv2.VideoCapture(get_short_path(os.path.join(rally_dir, "raw_clip.mp4")))
    p1_dir = os.path.join(rally_dir, "player1")
    p2_dir = os.path.join(rally_dir, "player2")
    frame_idx = init_frame if init_frame is not None else random.randint(0, total - 1)

    while True:
        frame_idx = max(0, min(frame_idx, total - 1))
        frame = get_frame(cap, frame_idx)

        if frame is None:
            frame_idx += 1
            continue

        vis = frame.copy()
        entry = load_entry(pose_data, frame_idx)
        near = entry.get("near_player") if entry else None
        far  = entry.get("far_player")  if entry else None

        draw_court(vis, entry.get("court") if entry else None)
        if near:
            draw_bbox(vis, near.get("bbox"), _NEAR_COLOR, "near")
            draw_keypoints(vis, near.get("keypoints", []), _NEAR_COLOR)
        if far:
            draw_bbox(vis, far.get("bbox"), _FAR_COLOR, "far")
            draw_keypoints(vis, far.get("keypoints", []), _FAR_COLOR)

        # 状态栏
        court_n = sum(1 for kp in (entry.get("court") or []) if len(kp) >= 3 and kp[2] >= 0.3) if entry else 0
        near_n  = sum(1 for kp in (near.get("keypoints") or []) if kp[2] >= _CONF_THRESH) if near else 0
        far_n   = sum(1 for kp in (far.get("keypoints")  or []) if kp[2] >= _CONF_THRESH) if far  else 0
        rerun   = "rerun:Y" if (entry or {}).get("_pose_rerun") else "rerun:N"
        info    = (f"{os.path.basename(rally_dir)}  帧{frame_idx}/{total-1}"
                   f"  court:{court_n}/14  near:{near_n}/17  far:{far_n}/17  {rerun}")
        cv2.putText(vis, info, (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, info, (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        h, w = vis.shape[:2]
        scale = min(1.0, 1280 / w, 720 / h)
        cv2.imshow("主视图  n/p=切帧  r=随机片段  q=退出",
                   cv2.resize(vis, (int(w * scale), int(h * scale))))

        # 裁剪图面板
        panels = []
        name = f"{frame_idx:06d}.jpg"
        if near:
            panels.append(make_crop_panel(
                frame, near,
                os.path.join(p1_dir, name) if os.path.isdir(p1_dir) else None,
                _NEAR_COLOR, "near"))
        if far:
            panels.append(make_crop_panel(
                frame, far,
                os.path.join(p2_dir, name) if os.path.isdir(p2_dir) else None,
                _FAR_COLOR, "far"))
        if panels:
            cv2.imshow("裁剪图  (左=预提取  右=bbox+关键点)", np.vstack(panels))

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            frame_idx += 1
        elif key == ord('p'):
            frame_idx -= 1
        elif key == ord('r'):
            cap.release()
            rally_dir, total = random.choice(index)
            pose_path = os.path.join(rally_dir, "pose_data.json")
            with open(pose_path, "r", encoding="utf-8") as f:
                pose_data = json.load(f)
            cap = cv2.VideoCapture(get_short_path(os.path.join(rally_dir, "raw_clip.mp4")))
            p1_dir = os.path.join(rally_dir, "player1")
            p2_dir = os.path.join(rally_dir, "player2")
            frame_idx = random.randint(0, total - 1)
            print(f"随机跳转: {os.path.basename(rally_dir)}  帧 {frame_idx}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rally", default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--data_root",
                        default=os.path.join(_PROJECT_DIR, "data", "rallies_annotated"))
    args = parser.parse_args()
    run(args.data_root, init_rally=args.rally, init_frame=args.frame)


if __name__ == "__main__":
    main()
