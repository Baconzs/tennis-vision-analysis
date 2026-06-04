"""test_matrix.py — 混淆矩阵评估脚本"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TennisActionDataset
from model_main import MSTFormer
from config import load_config

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

_MST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # mst/
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_MST_DIR)))  # 项目根


def split_dataset(data_root, train_ratio=0.8, seed=42):
    random.seed(seed)
    clips = []
    total_frames = 0
    for d in os.listdir(data_root):
        clip_path = os.path.join(data_root, d)
        if not os.path.isdir(clip_path):
            continue
        video = os.path.join(clip_path, "raw_clip.mp4")
        if not os.path.exists(video):
            continue
        cap = cv2.VideoCapture(video)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frames > 0:
            clips.append({"path": clip_path, "frames": frames})
            total_frames += frames

    random.shuffle(clips)
    train_dirs, test_dirs = [], []
    target = total_frames * train_ratio
    current = 0
    for c in clips:
        if current < target:
            train_dirs.append(c["path"])
            current += c["frames"]
        else:
            test_dirs.append(c["path"])
    return train_dirs, test_dirs


def evaluate_and_plot(yaml_path=None):
    cfg = load_config(yaml_path)
    device = cfg["device"]
    classes = ["待机", "正手", "反手", "发球", "移动"]

    weights_path = os.path.join(_PROJECT_DIR, "models", "action", "mst_former_final.pth")

    train_dirs, test_dirs = split_dataset(cfg["data_root"])
    train_ds = TennisActionDataset(cfg, clip_dirs=train_dirs)
    test_ds = TennisActionDataset(cfg, clip_dirs=test_dirs)

    loader_kwargs = dict(batch_size=cfg["batch_size"], shuffle=False,
                         num_workers=cfg["num_workers"])
    train_loader = DataLoader(train_ds, **loader_kwargs)
    test_loader = DataLoader(test_ds, **loader_kwargs)

    model = MSTFormer(cfg).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"成功加载模型权重: {weights_path}")
    else:
        print(f"找不到权重文件 {weights_path}")
        return
    model.eval()

    def get_predictions(loader, desc):
        all_preds, all_labels = [], []
        with torch.no_grad():
            for pose, packed, labels in tqdm(loader, desc=desc):
                pose   = pose.to(device, non_blocking=True)
                packed = packed.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(pose, packed)
                preds = logits.argmax(-1)
                mask = labels != -100
                all_preds.extend(preds[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
        return all_labels, all_preds

    print("\n正在评估训练集...")
    train_true, train_pred = get_predictions(train_loader, "Train Eval")
    print("\n正在评估测试集...")
    test_true, test_pred = get_predictions(test_loader, "Test Eval")

    if train_true:
        print(f"\n训练集准确率: {accuracy_score(train_true, train_pred)*100:.2f}%")
    if test_true:
        print(f"测试集准确率: {accuracy_score(test_true, test_pred)*100:.2f}%")

    def plot_cm(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4], normalize="true")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
        plt.title(title, fontsize=16)
        plt.ylabel("真实动作", fontsize=14)
        plt.xlabel("模型预测", fontsize=14)
        plt.tight_layout()
        plt.show()

    plot_cm(train_true, train_pred, "训练集混淆矩阵")
    plot_cm(test_true, test_pred, "测试集混淆矩阵")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    evaluate_and_plot(args.config)
