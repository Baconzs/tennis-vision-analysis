"""inference.py — 推理线程：person YOLO + pose YOLO 实时检测，送入 MSTFormer"""
import os
import sys
import json

# Windows 下 torch DLL 需要提前注册搜索路径，否则直接 import app.py 时会失败
if sys.platform == "win32":
    import importlib.util
    _torch_spec = importlib.util.find_spec("torch")
    if _torch_spec:
        _torch_lib = os.path.join(os.path.dirname(_torch_spec.origin), "lib")
        if os.path.isdir(_torch_lib):
            os.add_dll_directory(_torch_lib)

import torch
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_DEMO_DIR)
_MST_DIR = os.path.join(_SRC_DIR, "model", "mst")
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _MST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_main import MSTFormer  # noqa
from dataset import _build_pose_vec, _get_short_path, _resize_uint8  # noqa
from config import load_config  # noqa

POSE_DIM = 125
CROP_SIZE = 320

# COCO 17 关键点骨架连接
_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]
_KP_COLOR  = (0, 255, 128)
_BOX_COLORS = [(0, 200, 255), (255, 160, 0)]   # p1=青, p2=橙


def _draw_person(frame, box, kpts, color):
    """在 frame 上绘制 bbox + 骨架关键点。"""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if kpts is None or len(kpts) < 17:
        return
    pts = [(int(kp[0]), int(kp[1])) for kp in kpts]
    for a, b in _SKELETON:
        if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:
            cv2.line(frame, pts[a], pts[b], color, 2)
    for i, (px, py) in enumerate(pts):
        if kpts[i][2] > 0.3:
            cv2.circle(frame, (px, py), 4, _KP_COLOR, -1)


def _crop_fixed(frame, cx, cy, win, h, w):
    half = win // 2
    x1s, y1s = cx - half, cy - half
    canvas = np.zeros((win, win, 3), dtype=np.uint8)
    sx1, sy1 = max(0, x1s), max(0, y1s)
    sx2, sy2 = min(w, x1s + win), min(h, y1s + win)
    dx1, dy1 = sx1 - x1s, sy1 - y1s
    if sx2 > sx1 and sy2 > sy1:
        canvas[dy1:dy1+(sy2-sy1), dx1:dx1+(sx2-sx1)] = frame[sy1:sy2, sx1:sx2]
    return cv2.resize(canvas, (CROP_SIZE, CROP_SIZE))


class InferenceThread(QThread):
    progress = pyqtSignal(int)
    result   = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, rally_dir, config_path, weights_path,
                 person_model_path=None, pose_model_path=None, parent=None):
        super().__init__(parent)
        self.rally_dir         = rally_dir
        self.config_path       = config_path
        self.weights_path      = weights_path
        self.person_model_path = person_model_path
        self.pose_model_path   = pose_model_path

    def run(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _run(self):
        cfg    = load_config(self.config_path)
        device = cfg["device"]

        anno_path = os.path.join(self.rally_dir, "annotations.json")
        anno_json = json.load(open(anno_path, encoding="utf-8")) if os.path.exists(anno_path) else []

        # 读视频
        short = _get_short_path(os.path.join(self.rally_dir, "raw_clip.mp4"))
        cap   = cv2.VideoCapture(short)
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps     = raw_fps if raw_fps > 0 else 30.0
        vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        raw_bgr = []
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            raw_bgr.append(frm)
        cap.release()
        total_frames = len(raw_bgr)

        if total_frames == 0:
            self.error.emit("视频读取失败或帧数为 0")
            return

        self.progress.emit(5)

        use_person = bool(self.person_model_path)
        use_pose   = bool(self.person_model_path and self.pose_model_path)

        if use_person:
            self._run_yolo(cfg, device, raw_bgr, vid_w, vid_h, fps, anno_json, use_pose=use_pose)
        else:
            self._run_json(cfg, device, raw_bgr, fps, anno_json)

    def _run_yolo(self, cfg, device, raw_bgr, vid_w, vid_h, fps, anno_json, use_pose=True):
        from ultralytics import YOLO
        total_frames = len(raw_bgr)

        # 加载 YOLO 模型
        person_yolo = YOLO(self.person_model_path)
        pose_yolo   = YOLO(self.pose_model_path) if use_pose else None

        base_win = round(vid_w / 6.4)

        # ── 第一遍：person YOLO 检测，记录每帧两个运动员的 bbox ──────────────
        dets = []   # list of [slot0: (cx,cy,win,box)|None, slot1: ...]
        for gi, frm in enumerate(raw_bgr):
            frame_dets = [None, None]
            results = person_yolo(frm, verbose=False, conf=0.3)
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
                    side = int(max(box[2]-box[0], box[3]-box[1]))
                    win  = max(base_win, side)
                    frame_dets[slot] = (cx, cy, win, box)
            dets.append(frame_dets)
            if gi % 30 == 0:
                self.progress.emit(5 + int(gi / total_frames * 20))

        # 线性插值补全缺失帧
        for slot in range(2):
            known = [(i, dets[i][slot]) for i in range(len(dets)) if dets[i][slot] is not None]
            if not known:
                continue
            for i in range(len(dets)):
                if dets[i][slot] is not None:
                    continue
                prev = next((k for k in reversed(known) if k[0] < i), None)
                nxt  = next((k for k in known if k[0] > i), None)
                if prev is None:
                    dets[i][slot] = nxt[1]
                elif nxt is None:
                    dets[i][slot] = prev[1]
                else:
                    t  = (i - prev[0]) / (nxt[0] - prev[0])
                    cx = int(prev[1][0] + t*(nxt[1][0]-prev[1][0]))
                    cy = int(prev[1][1] + t*(nxt[1][1]-prev[1][1]))
                    wn = int(prev[1][2] + t*(nxt[1][2]-prev[1][2]))
                    # 插值 box
                    pb, nb = prev[1][3], nxt[1][3]
                    box = pb + t*(nb - pb)
                    dets[i][slot] = (cx, cy, wn, box)

        self.progress.emit(25)

        # ── 第二遍：pose YOLO + 构建 pose 向量 + 裁剪图 + 标注帧 ─────────────
        all_pose      = np.zeros((total_frames, POSE_DIM), dtype=np.float32)
        annotated_rgb = []   # 每帧带标注的 RGB ndarray
        crops_p1      = []   # (CROP_SIZE, CROP_SIZE, 3) uint8 RGB
        crops_p2      = []

        cx_p = cy_p = vx_p = vy_p = None

        for gi, frm in enumerate(raw_bgr):
            ann = frm.copy()
            kpts_near = None

            # 对两个运动员分别跑 pose YOLO
            for slot in range(2):
                det = dets[gi][slot]
                if det is None:
                    crops_p1.append(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)) if slot == 0 \
                        else crops_p2.append(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8))
                    continue

                cx, cy, win, box = det
                crop_bgr = _crop_fixed(frm, cx, cy, win, vid_h, vid_w)

                # pose 推理
                kpts_raw = None
                if pose_yolo is not None:
                    pose_res = pose_yolo(crop_bgr, verbose=False, conf=0.3, classes=[0])
                    if (pose_res[0].keypoints is not None and
                            len(pose_res[0].keypoints.data) > 0):
                        kp_data = pose_res[0].keypoints.data[0].cpu().numpy()  # (17,3)
                        # 坐标从裁剪图空间映射回原图空间
                        half = win // 2
                        ox, oy = cx - half, cy - half
                        kpts_raw = []
                        for kp in kp_data:
                            kx = kp[0] / CROP_SIZE * win + ox
                            ky = kp[1] / CROP_SIZE * win + oy
                            kpts_raw.append([kx, ky, float(kp[2])])

                # 绘制标注
                _draw_person(ann, box, kpts_raw, _BOX_COLORS[slot])

                # 裁剪图（RGB）
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                if slot == 0:
                    crops_p1.append(crop_rgb)
                    # near_player 用于 pose 向量
                    if kpts_raw is not None:
                        kpts_near = kpts_raw
                else:
                    crops_p2.append(crop_rgb)

            # 构建 pose 向量（用 near_player，即 slot=0）
            player_dict = None
            if kpts_near is not None and dets[gi][0] is not None:
                bx = dets[gi][0][3]
                player_dict = {
                    "cx": (bx[0]+bx[2])/2,
                    "cy": (bx[1]+bx[3])/2,
                    "kps": kpts_near,
                }
            vec, cx_p, cy_p, vx_p, vy_p = _build_pose_vec(
                player_dict, cx_p, cy_p,
                vx_p if vx_p is not None else 0.0,
                vy_p if vy_p is not None else 0.0,
            )
            all_pose[gi] = vec
            annotated_rgb.append(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))

            if gi % 30 == 0:
                self.progress.emit(25 + int(gi / total_frames * 40))

        self.progress.emit(65)

        # ── 构建 tensor ───────────────────────────────────────────────────────
        pose_t   = torch.from_numpy(all_pose).unsqueeze(0)  # (1,T,125)
        packed_t = torch.zeros(1, total_frames, 3, CROP_SIZE, CROP_SIZE*3, dtype=torch.uint8)

        use_visual = cfg.get("use_visual", True)
        if use_visual:
            # 全帧
            frames_dir = os.path.join(self.rally_dir, "frames")
            has_frames = os.path.isdir(frames_dir)
            for gi, frm in enumerate(raw_bgr):
                if has_frames:
                    fp = os.path.join(frames_dir, f"{gi:06d}.jpg")
                    if os.path.exists(fp):
                        raw = np.fromfile(fp, dtype=np.uint8)
                        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                        if img is not None:
                            frm = img
                rgb = _resize_uint8(frm, CROP_SIZE, CROP_SIZE)
                packed_t[0, gi, :, :, :CROP_SIZE] = torch.from_numpy(rgb.transpose(2,0,1))
            # 裁剪图
            if cfg.get("use_player_crops", True):
                for gi in range(total_frames):
                    if gi < len(crops_p1):
                        p1 = cv2.resize(crops_p1[gi], (CROP_SIZE, CROP_SIZE))
                        packed_t[0, gi, :, :, CROP_SIZE:CROP_SIZE*2] = torch.from_numpy(p1.transpose(2,0,1))
                    if gi < len(crops_p2):
                        p2 = cv2.resize(crops_p2[gi], (CROP_SIZE, CROP_SIZE))
                        packed_t[0, gi, :, :, CROP_SIZE*2:CROP_SIZE*3] = torch.from_numpy(p2.transpose(2,0,1))

        self.progress.emit(70)
        _emit_result(self, raw_bgr, all_pose, crops_p1, crops_p2,
                     annotated_rgb, anno_json, fps, cfg, device)

    def _run_json(self, cfg, device, raw_bgr, fps, anno_json):
        total_frames = len(raw_bgr)
        rally_dir    = self.rally_dir

        with open(os.path.join(rally_dir, "pose_data.json"), encoding="utf-8") as f:
            pose_json = json.load(f)

        all_pose = np.zeros((total_frames, POSE_DIM), dtype=np.float32)
        cx_p = cy_p = vx_p = vy_p = None
        for i in range(total_frames):
            fd = pose_json.get(str(i)) if isinstance(pose_json, dict) else (
                pose_json[i] if i < len(pose_json) else None)
            from dataset import _parse_player
            player = _parse_player(fd, "near_player") if fd else None
            vec, cx_p, cy_p, vx_p, vy_p = _build_pose_vec(
                player, cx_p, cy_p,
                vx_p if vx_p is not None else 0.0,
                vy_p if vy_p is not None else 0.0,
            )
            all_pose[i] = vec

        p1_dir = os.path.join(rally_dir, "player1")
        p2_dir = os.path.join(rally_dir, "player2")
        has_crops = os.path.isdir(p1_dir) and os.path.isdir(p2_dir)

        crops_p1, crops_p2 = [], []
        for gi in range(total_frames):
            name = f"{gi:06d}.jpg"
            if has_crops:
                p1p = os.path.join(p1_dir, name)
                p2p = os.path.join(p2_dir, name)
                crops_p1.append(_read_crop(p1p) if os.path.exists(p1p) else
                                 np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8))
                crops_p2.append(_read_crop(p2p) if os.path.exists(p2p) else
                                 np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8))
            else:
                crops_p1.append(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8))
                crops_p2.append(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8))
            if gi % 50 == 0:
                self.progress.emit(5 + int(gi / total_frames * 65))

        annotated_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in raw_bgr]
        self.progress.emit(70)
        _emit_result(self, raw_bgr, all_pose, crops_p1, crops_p2,
                     annotated_rgb, anno_json, fps, cfg, device)


def _build_gt_labels(anno_json, total_frames, fps):
    labels = [0] * total_frames
    if not isinstance(anno_json, list):
        return labels
    for seg in anno_json:
        s   = round(seg.get("start_time", 0) * fps)
        e   = round(seg.get("end_time",   0) * fps)
        aid = seg.get("action_id", 0)
        for f in range(max(0, s), min(e+1, total_frames)):
            labels[f] = aid
    return labels


def _emit_result(thread, raw_bgr, all_pose, crops_p1, crops_p2,
                 annotated_rgb, anno_json, fps, cfg, device):
    """构建 tensor、跑 MSTFormer、发射结果信号。"""
    total_frames = len(raw_bgr)
    use_visual   = cfg.get("use_visual", True)
    num_classes  = cfg.get("num_classes", 5)
    keyframe_only = cfg.get("keyframe_only", False)

    model = MSTFormer(cfg).to(device)
    import torch as _torch
    state = _torch.load(thread.weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    pose_t   = _torch.from_numpy(all_pose).unsqueeze(0).to(device)
    packed_t = _torch.zeros(1, total_frames, 3, 320, 960, dtype=_torch.uint8)

    if use_visual:
        for gi in range(total_frames):
            # 全帧
            frm_rgb = cv2.cvtColor(raw_bgr[gi], cv2.COLOR_BGR2RGB)
            packed_t[0, gi, :, :, :320] = _torch.from_numpy(
                cv2.resize(frm_rgb, (320, 320)).transpose(2, 0, 1))
            # 裁剪图
            if gi < len(crops_p1):
                packed_t[0, gi, :, :, 320:640] = _torch.from_numpy(crops_p1[gi].transpose(2, 0, 1))
            if gi < len(crops_p2):
                packed_t[0, gi, :, :, 640:960] = _torch.from_numpy(crops_p2[gi].transpose(2, 0, 1))

    thread.progress.emit(80)
    packed_t = packed_t.to(device)

    ctx = _torch.amp.autocast("cuda") if device.type == "cuda" else _nullctx()
    with _torch.no_grad(), ctx:
        if keyframe_only:
            kf_out  = model(pose_t, packed_t)
            act_out = _torch.zeros(1, total_frames, num_classes, device=device)
        else:
            act_out, kf_out = model(pose_t, packed_t)

    thread.progress.emit(95)

    per_frame_preds = act_out[0].argmax(-1).cpu().tolist()
    per_frame_kf    = kf_out[0].argmax(-1).cpu().tolist()
    gt_labels = _build_gt_labels(anno_json, total_frames, fps)
    accuracy  = sum(p == g for p, g in zip(per_frame_preds, gt_labels)) / total_frames

    thread.result.emit({
        "total_frames":    total_frames,
        "fps":             fps,
        "anno_json":       anno_json,
        "per_frame_preds": per_frame_preds,
        "per_frame_kf":    per_frame_kf,
        "gt_labels":       gt_labels,
        "accuracy":        accuracy,
        "annotated_rgb":   annotated_rgb,
    })


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass
