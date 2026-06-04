"""Extract a good forehand frame from rally videos for thesis figure."""
import json, os, sys, ctypes
import cv2
import numpy as np
from pathlib import Path

# --- court line definitions (14-pt to line connections) ---
COURT_LINES = [
    # 4 corners
    (0, 1), (1, 2), (2, 3), (3, 0),
    # net
    (4, 5),
    # service lines
    (6, 7), (7, 8), (8, 9), (9, 6),
    # center service lines
    (10, 11), (12, 13),
    # singles sidelines (connect corners to service line corners)
    (0, 6), (1, 7), (2, 8), (3, 9),
]

# COCO skeleton connections
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # shoulders-hips
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

def get_short_path(long_path):
    """Convert path to Windows short path for cv2 compatibility."""
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # 非 Windows 直接用原路径
        return long_path
    ctypes.windll.kernel32.GetShortPathNameW(long_path, buf, 512)
    return buf.value

def read_frame(video_path, frame_idx):
    """Read a specific frame from video using short path workaround."""
    short = get_short_path(str(video_path))
    cap = cv2.VideoCapture(short)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx}")
    return frame

def draw_court(img, court_pts, alpha=0.6):
    """Draw court keypoints and lines on image."""
    overlay = img.copy()
    h, w = img.shape[:2]

    # Draw lines
    for i, j in COURT_LINES:
        if i >= len(court_pts) or j >= len(court_pts):
            continue
        pi = court_pts[i]
        pj = court_pts[j]
        if pi[2] < 0.3 or pj[2] < 0.3:
            continue
        pt1 = (int(pi[0]), int(pi[1]))
        pt2 = (int(pj[0]), int(pj[1]))
        cv2.line(overlay, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

    # Draw keypoints
    for idx, pt in enumerate(court_pts):
        if pt[2] < 0.3:
            continue
        cx, cy = int(pt[0]), int(pt[1])
        cv2.circle(overlay, (cx, cy), 6, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), 7, (0, 100, 150), 2, cv2.LINE_AA)
        # label
        cv2.putText(overlay, str(idx), (cx + 10, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def draw_player(img, player_data, color, label):
    """Draw player bbox and skeleton on image."""
    if player_data is None:
        return img

    bbox = player_data.get('bbox')
    kps = player_data.get('keypoints', [])

    # Bbox
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        # label
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Skeleton
    if kps and len(kps) >= 17:
        pts = []
        for kp in kps[:17]:
            x, y, c = kp[0], kp[1], kp[2]
            if c > 0.1:
                pts.append((int(x), int(y), c))
            else:
                pts.append(None)

        # Draw connections
        for i, j in SKELETON:
            if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
                cv2.line(img, (pts[i][0], pts[i][1]), (pts[j][0], pts[j][1]),
                         color, 2, cv2.LINE_AA)

        # Draw keypoints
        for pt in pts:
            if pt:
                cv2.circle(img, (pt[0], pt[1]), 3, color, -1, cv2.LINE_AA)

    return img

def imwrite_cn(path, img):
    """imwrite_cn with Chinese path support."""
    _, buf = cv2.imencode('.png', img)
    with open(path, 'wb') as f:
        f.write(buf)

def extract_crop(img, bbox, padding=0.1):
    """Extract player crop from image given bbox."""
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    px, py = int(w * padding), int(h * padding)

    h_img, w_img = img.shape[:2]
    x1c = max(0, x1 - px)
    y1c = max(0, y1 - py)
    x2c = min(w_img, x2 + px)
    y2c = min(h_img, y2 + py)

    crop = img[y1c:y2c, x1c:x2c].copy()
    return crop

def main():
    base = Path("data/rallies_train")
    output_dir = Path("docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Candidate rallies with forehand segments (time_seconds, rally_name)
    candidates = [
        (12.3, 'rally_022_22.2s', 'forehand 1.3s'),
        (1.2, 'rally_047_9.7s', 'forehand 1.1s'),
        (24.3, 'rally_102_30.3s', 'forehand 0.7s'),
        (6.2, 'rally_102_30.3s', 'forehand 0.5s'),
    ]

    for mid_time, rally_name, desc in candidates:
        rally_dir = base / rally_name
        video_path = rally_dir / 'raw_clip.mp4'
        pose_path = rally_dir / 'pose_data.json'
        ann_path = rally_dir / 'annotations.json'

        if not video_path.exists():
            print(f'Skip {rally_name}: no video')
            continue

        # Load pose data
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)

        # Estimate FPS from frame count and video
        cap = cv2.VideoCapture(get_short_path(str(video_path)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        fps = len(pose_data) / (total_frames / 30.0) if total_frames > 0 else 30.0
        # Actually just use frame index directly
        frame_idx = int(mid_time * 30)  # assume ~30fps

        print(f'{rally_name}: t={mid_time}s, frame={frame_idx}, fps_est={fps:.1f}')

        # Read frame
        try:
            frame = read_frame(video_path, frame_idx)
        except Exception as e:
            print(f'  Error: {e}')
            continue

        # Find matching pose data entry
        pose_entry = None
        for p in pose_data:
            if abs(p['frame'] - frame_idx) <= 2:
                pose_entry = p
                break
        if pose_entry is None:
            print(f'  No pose data for frame {frame_idx}')
            continue

        # Draw court
        court_pts = pose_entry.get('court', [])
        if court_pts and len(court_pts) >= 14:
            frame = draw_court(frame, court_pts)

        # Draw players
        near = pose_entry.get('near_player')
        far = pose_entry.get('far_player')

        if near:
            frame = draw_player(frame, near, (80, 200, 120), 'Player 1 (Near)')
        if far:
            frame = draw_player(frame, far, (255, 140, 66), 'Player 2 (Far)')

        # Extract crops
        near_crop = extract_crop(frame, near.get('bbox') if near else None)
        far_crop = extract_crop(frame, far.get('bbox') if far else None)

        # Save annotated frame
        out_name = f'forehand_candidate_{rally_name.split("_")[1]}_{desc.replace(" ", "_")}.png'
        out_path = output_dir / out_name
        imwrite_cn(str(out_path), frame)
        print(f'  Saved: {out_path}')

        # Save crops
        if near_crop is not None and near_crop.size > 0:
            crop_path = output_dir / f'crop_near_{rally_name.split("_")[1]}.png'
            imwrite_cn(str(crop_path), near_crop)
            print(f'  Near crop: {crop_path} ({near_crop.shape})')
        if far_crop is not None and far_crop.size > 0:
            crop_path = output_dir / f'crop_far_{rally_name.split("_")[1]}.png'
            imwrite_cn(str(crop_path), far_crop)
            print(f'  Far crop: {crop_path} ({far_crop.shape})')

    print('\nDone! Check the output images in', str(output_dir))

if __name__ == '__main__':
    main()
