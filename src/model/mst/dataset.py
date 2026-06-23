"""dataset.py — 动作识别数据集加载器 v2

支持三路视觉流（全帧 + 球员1 + 球员2），可通过 config 开关控制
pose_tensor 维度：125 = 17×3(绝对坐标+conf) + 17×2(相对中心偏移) + 2(人物中心，相对球场中心)
+ 2(速度，球场相对坐标差分) + 2(加速度)
+ 6(球: 位置2+速度2+加速度2，暂时全零)
+ 28(球场14点×2，conf<0.3时置零)
"""
import os
import json
import ctypes
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

POSE_DIM = 125  # 91 + 6(球) + 28(球场14点×2)

# 统一高度 320，三路横向拼接后宽度 = 320+320+320 = 960
# 归一化在 GPU 端做，CPU 端保持 uint8 节省 PCIe 带宽
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _get_short_path(path):
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # 非 Windows 直接用原路径
        return path
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path


def _resize_uint8(img_bgr, h, w):
    """resize BGR uint8 图像，返回 RGB uint8 numpy [H, W, 3]"""
    img = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── 图像数据增强（训练时用） ────────────────────────────────────────────

def _apply_augmentations(rgb):
    """对单张 RGB uint8 [H,W,3] 依次应用增强。"""
    # 1) 颜色抖动（亮度/对比度/饱和度/色相）
    if np.random.rand() < 0.6:
        brightness = np.random.uniform(0.7, 1.3)
        contrast   = np.random.uniform(0.7, 1.3)
        saturation = np.random.uniform(0.7, 1.3)
        hue        = np.random.uniform(-0.04, 0.04)
        img = rgb.astype(np.float32)
        img = img * contrast + 128 * (1 - contrast)
        img = np.clip(img * brightness, 0, 255)
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= saturation
        hsv[:, :, 0] += hue * 180
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 2) 高斯噪声
    if np.random.rand() < 0.4:
        noise = np.random.randn(*rgb.shape).astype(np.float32) * 8
        rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 3) 高斯模糊
    if np.random.rand() < 0.3:
        k = np.random.choice([3, 5])
        rgb = cv2.GaussianBlur(rgb, (k, k), 0)

    # 4) 随机擦除（Cutout）
    if np.random.rand() < 0.3:
        h, w = rgb.shape[:2]
        area = h * w
        for _ in range(np.random.randint(1, 3)):
            erase_ratio = np.random.uniform(0.02, 0.12)
            ew = max(8, int(np.sqrt(area * erase_ratio)))
            eh = max(8, int(np.sqrt(area * erase_ratio)))
            x = np.random.randint(0, w - ew) if w > ew else 0
            y = np.random.randint(0, h - eh) if h > eh else 0
            color = np.random.randint(0, 256, size=(3,)).tolist()
            rgb[y:y+eh, x:x+ew] = color

    # 5) 半透明彩色覆盖（用户提出的）
    if np.random.rand() < 0.25:
        h, w = rgb.shape[:2]
        overlay = np.full((h, w, 3), np.random.randint(0, 256, size=(3,)), dtype=np.uint8)
        alpha = np.random.uniform(0.05, 0.2)
        rgb = np.clip(rgb.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha,
                      0, 255).astype(np.uint8)

    return rgb


def _parse_player(frame_data, player_key):
    """从单帧数据中提取指定球员的 bbox 中心和 17 个关键点，失败返回 None。"""
    p = frame_data.get(player_key) if isinstance(frame_data, dict) else None
    if p is None:
        return None
    kps = p.get("keypoints", [])
    bbox = p.get("bbox")
    if len(kps) != 17 or bbox is None:
        return None
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return {"cx": cx, "cy": cy, "kps": kps}


def _build_pose_vec(player, cx_prev, cy_prev, vx_prev, vy_prev, court_kps=None, W=1920.0, H=1080.0):
    """
    构建单帧 125 维物理特征向量。
    player: _parse_player 的返回值（dict），None 时返回全零向量。
    court_kps: 14 个球场关键点 [[x,y,conf],...] 或 None。
    cx_prev/cy_prev: 上一帧的球场相对坐标（已归一化），None 时为首帧。
    返回 (vec_125, rel_cx, rel_cy, vx, vy) 供下一帧使用（坐标均为球场相对归一化值）。
    """
    # 球场中心（conf≥0.3 的点均值，回退到帧中心）
    court_cx, court_cy = W / 2, H / 2
    if court_kps:
        valid = [kp for kp in court_kps if len(kp) >= 3 and kp[2] >= 0.3]
        if valid:
            court_cx = float(np.mean([kp[0] for kp in valid]))
            court_cy = float(np.mean([kp[1] for kp in valid]))

    if player is None:
        return np.zeros(POSE_DIM, dtype=np.float32), cx_prev, cy_prev, vx_prev, vy_prev

    cx, cy = player["cx"], player["cy"]
    kps = player["kps"]

    # 17×3：绝对坐标 + conf
    abs_part = []
    for kp in kps:
        abs_part.extend([kp[0] / W, kp[1] / H, kp[2]])

    # 17×2：相对人物中心偏移
    rel_part = []
    for kp in kps:
        rel_part.extend([(kp[0] - cx) / W, (kp[1] - cy) / H])

    # 人物中心（相对球场中心的归一化坐标）
    rel_cx = (cx - court_cx) / W
    rel_cy = (cy - court_cy) / H
    center_part = [rel_cx, rel_cy]

    # 速度（球场相对坐标的帧间差分）
    vx = rel_cx - cx_prev if cx_prev is not None else 0.0
    vy = rel_cy - cy_prev if cy_prev is not None else 0.0

    # 加速度
    ax = vx - vx_prev
    ay = vy - vy_prev

    # 球场 14 点坐标（conf<0.3 时置零，共 28 维）
    court_part = []
    for i in range(14):
        if court_kps and i < len(court_kps):
            kp = court_kps[i]
            if len(kp) >= 3 and kp[2] >= 0.3:
                court_part.extend([kp[0] / W, kp[1] / H])
            else:
                court_part.extend([0.0, 0.0])
        else:
            court_part.extend([0.0, 0.0])

    vec = np.array(abs_part + rel_part + center_part + [vx, vy, ax, ay]
                   + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 球: 位置xy + 速度xy + 加速度xy
                   + court_part,
                   dtype=np.float32)
    return vec, rel_cx, rel_cy, vx, vy


def _read_crop(path):
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((320, 320, 3), dtype=np.uint8)
    return _resize_uint8(img, 320, 320)


class TennisActionDataset(Dataset):
    def __init__(self, cfg, clip_dirs=None, augment=False):
        self.data_root = cfg["data_root"]
        self.seq_len = cfg["seq_len"]
        self.min_seq_len = cfg.get("min_seq_len", max(30, cfg["seq_len"] // 2))
        self.use_visual = cfg.get("use_visual", True)
        self.use_player_crops = cfg.get("use_player_crops", True)
        self.augment = augment  # 训练集开启，测试集关闭

        if clip_dirs is None:
            clip_dirs = [
                os.path.join(self.data_root, d)
                for d in os.listdir(self.data_root)
                if os.path.isdir(os.path.join(self.data_root, d))
            ]
        self.clip_dirs = clip_dirs

        self.meta_cache = {}
        self.chunks = []

        print(f"[dataset] 预加载 JSON 并切片（seq_len={self.seq_len}）...")
        for clip_dir in self.clip_dirs:
            pose_path = os.path.join(clip_dir, "pose_data.json")
            anno_path = os.path.join(clip_dir, "annotations.json")
            if not os.path.exists(pose_path) or not os.path.exists(anno_path):
                continue

            with open(pose_path, "r", encoding="utf-8") as f:
                pose_json = json.load(f)
            with open(anno_path, "r", encoding="utf-8") as f:
                anno_json = json.load(f)

            if isinstance(pose_json, dict):
                frame_keys = [int(k) for k in pose_json if k.isdigit()]
                total_frames = max(frame_keys) + 1 if frame_keys else 0
            else:
                total_frames = len(pose_json)

            if total_frames == 0:
                continue

            self.meta_cache[clip_dir] = {
                "pose": pose_json,
                "anno": anno_json,
                "total_frames": total_frames,
            }

        self._build_chunks()
        print(f"[dataset] 共生成 {len(self.chunks)} 个切片。")

    def _build_chunks(self):
        """固定切片，用于测试集或初始化。"""
        self.chunks = []
        for clip_dir, meta in self.meta_cache.items():
            total_frames = meta["total_frames"]
            for start in range(0, total_frames, self.seq_len):
                self.chunks.append({
                    "clip_dir": clip_dir,
                    "start_frame": start,
                    "end_frame": min(start + self.seq_len, total_frames),
                })

    def reshuffle(self):
        """每个 epoch 开始前调用，随机重新划分训练切片。"""
        if not self.augment:
            return
        self.chunks = []
        for clip_dir, meta in self.meta_cache.items():
            total_frames = meta["total_frames"]
            if total_frames < self.min_seq_len:
                continue
            start = 0
            while start < total_frames:
                remaining = total_frames - start
                if remaining < self.min_seq_len:
                    break
                length = np.random.randint(self.min_seq_len, min(self.seq_len, remaining) + 1)
                self.chunks.append({
                    "clip_dir": clip_dir,
                    "start_frame": start,
                    "end_frame": start + length,
                })
                start += length

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        clip_dir = chunk["clip_dir"]
        start = chunk["start_frame"]
        end = chunk["end_frame"]
        actual_len = end - start

        cache = self.meta_cache[clip_dir]
        pose_json = cache["pose"]
        anno_json = cache["anno"]

        pose_tensor = torch.zeros(self.seq_len, POSE_DIM)
        labels = torch.full((self.seq_len,), -100, dtype=torch.long)
        keyframe_labels = torch.zeros(self.seq_len, dtype=torch.long)
        # uint8 拼接 tensor：[T, 3, 320, 960]（全帧pad到320×320 + p1 320×320 + p2 320×320）
        # 归一化在 GPU 端做，CPU 端保持 uint8 节省 PCIe 带宽
        if self.use_visual:
            packed_frames = torch.zeros(self.seq_len, 3, 320, 960, dtype=torch.uint8)
        else:
            packed_frames = torch.zeros(1, dtype=torch.uint8)

        frames_dir = os.path.join(clip_dir, "frames")
        use_frames_dir = self.use_visual and os.path.isdir(frames_dir)

        cap = None
        if self.use_visual and not use_frames_dir:
            short_path = _get_short_path(os.path.join(clip_dir, "raw_clip.mp4"))
            cap = cv2.VideoCapture(short_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 30.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        else:
            fps = 30.0

        # 预计算关键帧集合（±2 帧容差）
        key_frames = set()
        if isinstance(anno_json, list):
            for seg in anno_json:
                for t_sec in (seg.get("start_time", -1), seg.get("end_time", -1)):
                    if t_sec < 0:
                        continue
                    fi = round(t_sec * fps)
                    for delta in (-2, -1, 0, 1, 2):
                        key_frames.add(fi + delta)

        p1_dir = os.path.join(clip_dir, "player1")
        p2_dir = os.path.join(clip_dir, "player2")
        has_crops = (self.use_visual and self.use_player_crops
                     and os.path.isdir(p1_dir) and os.path.isdir(p2_dir))

        cx_prev = cy_prev = vx_prev = vy_prev = None

        for t in range(actual_len):
            global_idx = start + t
            current_time = global_idx / fps

            # 动作标签
            action_id = 0
            if isinstance(anno_json, list):
                for seg in anno_json:
                    if seg.get("start_time", 0) <= current_time <= seg.get("end_time", 0):
                        action_id = seg.get("action_id", 0)
                        break
            labels[t] = action_id

            # 关键帧标签
            keyframe_labels[t] = 1 if global_idx in key_frames else 0

            # 姿态（125维）
            frame_data = (pose_json.get(str(global_idx))
                          if isinstance(pose_json, dict)
                          else (pose_json[global_idx] if global_idx < len(pose_json) else None))
            player = _parse_player(frame_data, "near_player") if frame_data else None
            court_kps = frame_data.get("court") if isinstance(frame_data, dict) else None
            vec, cx_prev, cy_prev, vx_prev, vy_prev = _build_pose_vec(
                player, cx_prev, cy_prev,
                vx_prev if vx_prev is not None else 0.0,
                vy_prev if vy_prev is not None else 0.0,
                court_kps=court_kps,
            )
            pose_tensor[t] = torch.from_numpy(vec)

            # 全帧视觉（pad 到 320×320，写入拼接 tensor 的 [:, :, 0:320]）
            if use_frames_dir:
                fp = os.path.join(frames_dir, f"{global_idx:06d}.jpg")
                if os.path.exists(fp):
                    raw = np.fromfile(fp, dtype=np.uint8)
                    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    if img is not None:
                        rgb = _resize_uint8(img, 320, 320)
                        packed_frames[t, :, :, :320] = torch.from_numpy(rgb.transpose(2, 0, 1))
            elif cap is not None:
                ret, frame = cap.read()
                if ret:
                    rgb = _resize_uint8(frame, 320, 320)
                    packed_frames[t, :, :, :320] = torch.from_numpy(rgb.transpose(2, 0, 1))

            # 裁剪图（p1 写入 [:, :, 320:640]，p2 写入 [:, :, 640:960]）
            if has_crops:
                name = f"{global_idx:06d}.jpg"
                p1_path = os.path.join(p1_dir, name)
                p2_path = os.path.join(p2_dir, name)
                if os.path.exists(p1_path):
                    rgb1 = _read_crop(p1_path)
                    packed_frames[t, :, :, 320:640] = torch.from_numpy(rgb1.transpose(2, 0, 1))
                if os.path.exists(p2_path):
                    rgb2 = _read_crop(p2_path)
                    packed_frames[t, :, :, 640:960] = torch.from_numpy(rgb2.transpose(2, 0, 1))

        if cap is not None:
            cap.release()

        return pose_tensor, packed_frames, labels, keyframe_labels
