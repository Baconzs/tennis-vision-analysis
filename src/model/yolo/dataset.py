"""
YOLO 单帧动作分类 — 数据集
从 rallies_train/ 读取预提取帧（frames/*.jpg）+ annotations.json，返回单帧 (image, label)
"""

import os
import json
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ACTION_NAMES = ["idle", "forehand", "backhand", "serve", "move"]
NUM_CLASSES = 5
IMG_SIZE = 224  # YOLO 分类输入尺寸


def _read_frame(path):
    """读取图像（支持中文路径），返回 RGB uint8 [H, W, 3]"""
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def collect_samples(data_root):
    """
    遍历 rallies_train/，收集所有帧样本。
    返回 [(image_path, action_id), ...]
    """
    samples = []
    rallies = sorted([
        os.path.join(data_root, d) for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    for rally_dir in rallies:
        anno_path = os.path.join(rally_dir, "annotations.json")
        frames_dir = os.path.join(rally_dir, "frames")
        if not os.path.exists(anno_path):
            continue

        # 使用预提取帧
        if not os.path.isdir(frames_dir):
            continue

        with open(anno_path, "r", encoding="utf-8") as f:
            anno = json.load(f)

        # 将 annotations 转换为帧级别标签（以 30fps 计）
        fps = 30.0
        frame_files = sorted([
            f for f in os.listdir(frames_dir) if f.endswith(".jpg")
        ])
        if not frame_files:
            continue

        # 预计算每帧标签
        max_idx = int(frame_files[-1].replace(".jpg", "")) + 1
        frame_labels = np.zeros(max_idx, dtype=int)
        for seg in anno:
            start_frame = round(seg["start_time"] * fps)
            end_frame = round(seg["end_time"] * fps)
            action_id = seg["action_id"]
            for fi in range(max(0, start_frame), min(end_frame + 1, max_idx)):
                frame_labels[fi] = action_id

        for fname in frame_files:
            frame_idx = int(fname.replace(".jpg", ""))
            if frame_idx < len(frame_labels):
                img_path = os.path.join(frames_dir, fname)
                samples.append((img_path, int(frame_labels[frame_idx])))

    return samples


class TennisFrameDataset(Dataset):
    """单帧数据集，每样本 = 一张图 + 一个动作标签"""

    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = _read_frame(img_path)

        # 简单数据增强（训练集）
        if self.augment:
            # 随机水平翻转
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
            # 随机亮度/对比度调整
            if np.random.rand() > 0.5:
                alpha = 1.0 + np.random.uniform(-0.2, 0.2)
                beta = np.random.randint(-30, 30)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # HWC → CHW, uint8
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().div_(255.0)
        # ImageNet 归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor, torch.tensor(label, dtype=torch.long)


def split_dataset(data_root, train_ratio=0.8, seed=42):
    """按 rally 划分训练/测试集（与 mst 训练一致）"""
    random.seed(seed)
    import cv2 as _cv2

    rallies = []
    for d in sorted(os.listdir(data_root)):
        clip_path = os.path.join(data_root, d)
        if not os.path.isdir(clip_path):
            continue
        video = os.path.join(clip_path, "raw_clip.mp4")
        anno = os.path.join(clip_path, "annotations.json")
        if not os.path.exists(video) or not os.path.exists(anno):
            continue
        cap = _cv2.VideoCapture(video)
        frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frames > 0:
            rallies.append({"path": clip_path, "frames": frames})

    random.shuffle(rallies)
    train_dirs, test_dirs = [], []
    target = sum(r["frames"] for r in rallies) * train_ratio
    current = 0
    for r in rallies:
        if current < target:
            train_dirs.append(r["path"])
            current += r["frames"]
        else:
            test_dirs.append(r["path"])
    return train_dirs, test_dirs


def create_datasets(data_root, train_ratio=0.8):
    """创建训练和测试数据集"""
    train_dirs, test_dirs = split_dataset(data_root, train_ratio)

    train_samples = []
    for d in train_dirs:
        frames_dir = os.path.join(d, "frames")
        anno_path = os.path.join(d, "annotations.json")
        if not os.path.isdir(frames_dir) or not os.path.exists(anno_path):
            continue
        fps = 30.0
        with open(anno_path, "r", encoding="utf-8") as f:
            anno = json.load(f)
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        if not frame_files:
            continue
        max_idx = int(frame_files[-1].replace(".jpg", "")) + 1
        frame_labels = np.zeros(max_idx, dtype=int)
        for seg in anno:
            start_frame = round(seg["start_time"] * fps)
            end_frame = round(seg["end_time"] * fps)
            for fi in range(max(0, start_frame), min(end_frame + 1, max_idx)):
                frame_labels[fi] = seg["action_id"]
        for fname in frame_files:
            frame_idx = int(fname.replace(".jpg", ""))
            if frame_idx < len(frame_labels):
                train_samples.append((os.path.join(frames_dir, fname),
                                      int(frame_labels[frame_idx])))

    test_samples = []
    for d in test_dirs:
        frames_dir = os.path.join(d, "frames")
        anno_path = os.path.join(d, "annotations.json")
        if not os.path.isdir(frames_dir) or not os.path.exists(anno_path):
            continue
        fps = 30.0
        with open(anno_path, "r", encoding="utf-8") as f:
            anno = json.load(f)
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        if not frame_files:
            continue
        max_idx = int(frame_files[-1].replace(".jpg", "")) + 1
        frame_labels = np.zeros(max_idx, dtype=int)
        for seg in anno:
            start_frame = round(seg["start_time"] * fps)
            end_frame = round(seg["end_time"] * fps)
            for fi in range(max(0, start_frame), min(end_frame + 1, max_idx)):
                frame_labels[fi] = seg["action_id"]
        for fname in frame_files:
            frame_idx = int(fname.replace(".jpg", ""))
            if frame_idx < len(frame_labels):
                test_samples.append((os.path.join(frames_dir, fname),
                                     int(frame_labels[frame_idx])))

    print(f"[yolo dataset] Train: {len(train_samples)} frames, Test: {len(test_samples)} frames")

    train_ds = TennisFrameDataset(train_samples, augment=True)
    test_ds = TennisFrameDataset(test_samples, augment=False)
    return train_ds, test_ds
