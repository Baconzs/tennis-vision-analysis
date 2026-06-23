"""Generate thesis Figure: annotated frame + square player crops with dashed connection lines.

Layout:
┌──────────────────────────────┬─────────────┐
│                              │  ┌───────┐  │
│     Main Annotated Frame     │  │P1 Crop │  │
│   (court lines, bbox, pose)  │  │ 320x320│  │
│                              │  └───────┘  │
│   ┌─bbox────┐  ┌─bbox──┐    │             │
│   │ Player1 │  │Player2│    │  ┌───────┐  │
│   └─────────┘  └───────┘    │  │P2 Crop │  │
│     └─── ─ ─ ─ ─ ──┘        │  │ 320x320│  │
│      dashed connector        │  └───────┘  │
└──────────────────────────────┴─────────────┘
"""
import json, ctypes
import cv2
import numpy as np
from pathlib import Path

COURT_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5),
    (6, 7), (7, 8), (8, 9), (9, 6),
    (10, 11), (12, 13),
    (0, 6), (1, 7), (2, 8), (3, 9),
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

NEAR_COLOR = (90, 210, 130)   # green-cyan
FAR_COLOR = (70, 150, 255)    # orange-blue

# --- Utilities ---
def get_short_path(long_path):
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # 非 Windows 直接用原路径
        return long_path
    ctypes.windll.kernel32.GetShortPathNameW(long_path, buf, 512)
    return buf.value

def imwrite_cn(path, img):
    _, buf = cv2.imencode('.png', img)
    with open(path, 'wb') as f:
        f.write(buf)

def read_frame(video_path, frame_idx):
    short = get_short_path(str(video_path))
    cap = cv2.VideoCapture(short)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame

# --- Drawing functions ---
def draw_court(img, court_pts):
    overlay = img.copy()
    poly = []
    for i in [0, 1, 2, 3]:
        if i < len(court_pts) and court_pts[i][2] > 0.3:
            poly.append((int(court_pts[i][0]), int(court_pts[i][1])))
    if len(poly) == 4:
        cv2.fillPoly(overlay, [np.array(poly)], (0, 80, 0))
    for i, j in COURT_LINES:
        if i >= len(court_pts) or j >= len(court_pts):
            continue
        pi, pj = court_pts[i], court_pts[j]
        if pi[2] < 0.3 or pj[2] < 0.3:
            continue
        pt1, pt2 = (int(pi[0]), int(pi[1])), (int(pj[0]), int(pj[1]))
        thk = 5 if (i, j) == (4, 5) else 3
        cv2.line(overlay, pt1, pt2, (255, 255, 255), thk, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for idx, pt in enumerate(court_pts):
        if pt[2] < 0.3:
            continue
        cx, cy = int(pt[0]), int(pt[1])
        cv2.circle(img, (cx, cy), 8, (0, 220, 220), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 9, (0, 100, 100), 2, cv2.LINE_AA)
    return img

def draw_player(img, player_data, color, label):
    if not player_data:
        return img
    bbox = player_data.get('bbox')
    kps = player_data.get('keypoints', [])
    dark_color = tuple(max(0, c - 60) for c in color)

    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), dark_color, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    if kps and len(kps) >= 17:
        pts = []
        for kp in kps[:17]:
            pts.append((int(kp[0]), int(kp[1])) if kp[2] > 0.15 else None)
        for i, j in SKELETON:
            if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
                cv2.line(img, pts[i], pts[j], color, 3, cv2.LINE_AA)
        for pt in pts:
            if pt:
                cv2.circle(img, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, pt, 4, dark_color, -1, cv2.LINE_AA)
    return img

def extract_crop(frame, bbox, padding=0.15):
    """Extract player crop — same logic as dataset preparation."""
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    px, py = int(w * padding), int(h * padding)
    h_img, w_img = frame.shape[:2]
    x1c = max(0, x1 - px)
    y1c = max(0, y1 - py)
    x2c = min(w_img, x2 + px)
    y2c = min(h_img, y2 + py)
    if x2c <= x1c or y2c <= y1c:
        return None
    return frame[y1c:y2c, x1c:x2c].copy()

def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=10, gap_len=6):
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx*dx + dy*dy)
    if dist < 2:
        return
    n_segs = max(1, int(dist / (dash_len + gap_len)))
    for i in range(n_segs):
        t0 = i / n_segs
        t1 = (i * (dash_len + gap_len) + dash_len) / (n_segs * (dash_len + gap_len))
        t1 = min(t1, 1.0)
        p0 = (int(x1 + dx * t0), int(y1 + dy * t0))
        p1 = (int(x1 + dx * t1), int(y1 + dy * t1))
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)

# --- Main ---
def main():
    base = Path("data/rallies_train")
    out_dir = Path("docs/figures")

    # rally_047 f36: best forehand frame
    rally_name = 'rally_047_9.7s'
    frame_idx = 36

    rally_dir = base / rally_name
    with open(rally_dir / 'pose_data.json', 'r') as f:
        pose_data = json.load(f)

    entry = None
    for p in pose_data:
        if abs(p['frame'] - frame_idx) <= 2:
            entry = p
            break

    frame = read_frame(rally_dir / 'raw_clip.mp4', frame_idx)
    print(f'Frame: {frame.shape}')

    near = entry.get('near_player')
    far = entry.get('far_player')
    court_pts = entry.get('court', [])

    # Extract crops from CLEAN frame (before annotations)
    near_crop_raw = extract_crop(frame, near.get('bbox') if near else None)
    far_crop_raw = extract_crop(frame, far.get('bbox') if far else None)

    # Build annotated main frame
    annotated = frame.copy()
    if court_pts:
        annotated = draw_court(annotated, court_pts)
    if near:
        annotated = draw_player(annotated, near, NEAR_COLOR, 'Player 1 (Near)')
    if far:
        annotated = draw_player(annotated, far, FAR_COLOR, 'Player 2 (Far)')

    # === Layout construction ===
    MAIN_W = 1100
    scale = MAIN_W / annotated.shape[1]
    MAIN_H = int(annotated.shape[0] * scale)
    main_img = cv2.resize(annotated, (MAIN_W, MAIN_H), interpolation=cv2.INTER_AREA)

    CROP_SIZE = 280  # display size of each crop in the figure
    GAP = 20
    RIGHT_W = CROP_SIZE + 40  # panel width
    RIGHT_H = CROP_SIZE * 2 + GAP + 80  # two crops + labels + gap

    # Make crop squares
    def make_square_crop(crop_img, target_size=320):
        """Resize crop to square (target_size x target_size) maintaining aspect with padding."""
        if crop_img is None:
            return None
        h, w = crop_img.shape[:2]
        # Pad to square
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_w = max_dim - w
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l
        square = cv2.copyMakeBorder(crop_img, pad_t, pad_b, pad_l, pad_r,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)

    near_sq = make_square_crop(near_crop_raw)
    far_sq = make_square_crop(far_crop_raw)

    # Resize square crops to display size
    near_disp = cv2.resize(near_sq, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA) if near_sq is not None else None
    far_disp = cv2.resize(far_sq, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA) if far_sq is not None else None

    # Determine canvas size
    canvas_h = max(MAIN_H, RIGHT_H)
    canvas_w = MAIN_W + RIGHT_W
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40  # dark bg

    # Place main image
    y_off = (canvas_h - MAIN_H) // 2
    canvas[y_off:y_off + MAIN_H, 0:MAIN_W] = main_img

    # Place crops on the right
    right_x = MAIN_W + 20
    crop_y1 = (canvas_h - (CROP_SIZE * 2 + GAP)) // 2
    crop_y2 = crop_y1 + CROP_SIZE + GAP

    if near_disp is not None:
        # label
        cv2.putText(canvas, 'Player 1', (right_x + 5, crop_y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, NEAR_COLOR, 2, cv2.LINE_AA)
        canvas[crop_y1:crop_y1 + CROP_SIZE, right_x:right_x + CROP_SIZE] = near_disp
        # border
        cv2.rectangle(canvas, (right_x, crop_y1), (right_x + CROP_SIZE, crop_y1 + CROP_SIZE),
                      NEAR_COLOR, 4)

    if far_disp is not None:
        cv2.putText(canvas, 'Player 2', (right_x + 5, crop_y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, FAR_COLOR, 2, cv2.LINE_AA)
        canvas[crop_y2:crop_y2 + CROP_SIZE, right_x:right_x + CROP_SIZE] = far_disp
        cv2.rectangle(canvas, (right_x, crop_y2), (right_x + CROP_SIZE, crop_y2 + CROP_SIZE),
                      FAR_COLOR, 4)

    # === Draw dashed connector lines ===
    # We need to connect from the player bbox corners in the main image to the crop panel
    # P1 (near) → top crop, P2 (far) → bottom crop

    def draw_connectors(canvas, bbox, crop_x, crop_y, crop_sz, color, scale, main_y_off):
        """Draw dashed lines from bbox area in main frame to crop on the right."""
        if not bbox or len(bbox) != 4:
            return
        x1, y1, x2, y2 = [int(v * scale) for v in bbox]
        y1 += main_y_off
        y2 += main_y_off
        # bbox right edge center
        bbox_right_pt = (x2, (y1 + y2) // 2)

        # crop left edge center
        crop_left_pt = (crop_x, crop_y + crop_sz // 2)

        # Draw dashed line
        draw_dashed_line(canvas, bbox_right_pt, crop_left_pt, color,
                         thickness=2, dash_len=12, gap_len=8)

        # Also draw a small indicator on the bbox
        cv2.circle(canvas, bbox_right_pt, 5, color, -1, cv2.LINE_AA)

    if near:
        draw_connectors(canvas, near['bbox'], right_x, crop_y1, CROP_SIZE,
                        NEAR_COLOR, scale, y_off)
    if far:
        draw_connectors(canvas, far['bbox'], right_x, crop_y2, CROP_SIZE,
                        FAR_COLOR, scale, y_off)

    # Save
    out_path = out_dir / 'fig_court_pose_with_crops.png'
    imwrite_cn(str(out_path), canvas)
    print(f'Saved: {out_path} ({canvas.shape[1]}x{canvas.shape[0]})')

    # Also save just the annotated main frame separately (for other uses)
    imwrite_cn(str(out_dir / 'fig_annotated_main.png'), main_img)

    print('Done!')

if __name__ == '__main__':
    main()
