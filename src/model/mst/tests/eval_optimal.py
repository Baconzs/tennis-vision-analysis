"""评估 MSTFormer 模型并生成混淆矩阵"""
import sys, os, random, argparse
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import TennisActionDataset
from model_main import MSTFormer
from config import load_config

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

_MST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_MST_DIR)))

CLASSES = ["待机", "正手", "反手", "发球", "移动"]

def split_dataset(data_root, test_root=None, train_ratio=0.8, seed=42):
    random.seed(seed)
    clips = []
    total_frames = 0
    for d in os.listdir(data_root):
        clip_path = os.path.join(data_root, d)
        if not os.path.isdir(clip_path):
            continue
        video = os.path.join(clip_path, "raw_clip.mp4")
        anno = os.path.join(clip_path, "annotations.json")
        if not os.path.exists(video) or not os.path.exists(anno):
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

    if test_root is not None:
        test_dirs = [os.path.join(test_root, os.path.relpath(d, data_root))
                     for d in test_dirs]
    return train_dirs, test_dirs


def compute_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    return cm, cm_norm


def plot_confusion_matrix(cm_norm, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", vmin=0, vmax=1,
                xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={"size": 12})
    plt.title(title, fontsize=16)
    plt.ylabel("真实动作", fontsize=14)
    plt.xlabel("模型预测", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  保存混淆矩阵: {save_path}")


def print_classification_report(y_true, y_pred, name):
    print(f"\n{'='*60}")
    print(f"  {name} 分类报告")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, labels=[0,1,2,3,4],
                                target_names=CLASSES, digits=4))

    # 按类输出详细统计
    cm, cm_norm = compute_cm(y_true, y_pred)
    print(f"\n  各类别详细统计:")
    print(f"  {'类别':<8} {'GT数量':<8} {'预测数量':<10} {'正确数':<8} {'正确率':<8} {'召回率':<8}")
    print(f"  {'-'*52}")
    for i, name_i in enumerate(CLASSES):
        gt_count = (np.array(y_true) == i).sum()
        pred_count = (np.array(y_pred) == i).sum()
        correct = cm[i, i]
        precision = correct / pred_count if pred_count > 0 else 0
        recall = correct / gt_count if gt_count > 0 else 0
        print(f"  {name_i:<8} {gt_count:<8} {pred_count:<10} {correct:<8} {precision*100:<7.2f}% {recall*100:<7.2f}%")

    # 错误分析：哪些类最容易被混淆
    print(f"\n  主要混淆方向 (占GT比例):")
    for i in range(5):
        row = cm[i]
        total = row.sum()
        if total == 0:
            continue
        # 找出误分类最多的目标类
        misclass = [(j, row[j]) for j in range(5) if j != i and row[j] > 0]
        misclass.sort(key=lambda x: -x[1])
        if misclass and misclass[0][1] / total >= 0.05:
            top_mis = ", ".join(f"{CLASSES[j]}({c}/{total}={c/total*100:.1f}%)" for j, c in misclass[:3])
            print(f"    {CLASSES[i]:<6}: → {top_mis}")


def print_top_k_accuracy(y_true, y_pred, k=2):
    """需要用 softmax logits，这里用混淆矩阵近似报错"""
    pass


def main(yaml_path, weights_path):
    # 从路径推导模型名和输出目录
    model_dir = os.path.dirname(weights_path)
    model_name = os.path.basename(os.path.dirname(model_dir)) + "/" + os.path.basename(model_dir)
    out_dir = os.path.join(model_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    print(f"配置: {yaml_path}")
    print(f"权重: {weights_path}")
    print(f"输出: {out_dir}")

    cfg = load_config(yaml_path)
    device = cfg["device"]

    # 数据集分割（与训练时一致，seed=42）
    if cfg.get("test_data_root"):
        train_dirs, test_dirs = split_dataset(cfg["data_root"], cfg["test_data_root"])
    else:
        train_dirs, test_dirs = split_dataset(cfg["data_root"])

    print(f"\n数据集: {cfg['data_root']}")
    print(f"  训练集 clips: {len(train_dirs)}")
    print(f"  测试集 clips: {len(test_dirs)}")

    train_ds = TennisActionDataset(cfg, clip_dirs=train_dirs)
    test_ds  = TennisActionDataset(cfg, clip_dirs=test_dirs)

    loader_kwargs = dict(batch_size=cfg["batch_size"], shuffle=False,
                         num_workers=cfg["num_workers"])
    train_loader = DataLoader(train_ds, **loader_kwargs)
    test_loader  = DataLoader(test_ds, **loader_kwargs)

    # 加载模型
    model = MSTFormer(cfg).to(device)
    state = torch.load(weights_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print("\n模型权重加载成功")
    model.eval()

    def get_predictions(loader, desc):
        all_preds, all_labels = [], []
        with torch.no_grad():
            for pose, packed, labels, _kf in tqdm(loader, desc=desc):
                pose   = pose.to(device, non_blocking=True)
                packed = packed.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    output = model(pose, packed)
                logits = output[0]  # (action_logits, keyframe_logits)
                preds = logits.argmax(-1)
                mask = labels != -100
                all_preds.extend(preds[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
        return np.array(all_labels), np.array(all_preds)

    print("\n评估训练集...")
    train_true, train_pred = get_predictions(train_loader, "Train")
    train_acc = accuracy_score(train_true, train_pred)
    print(f"  训练集准确率: {train_acc*100:.2f}%")

    print("\n评估测试集...")
    test_true, test_pred = get_predictions(test_loader, "Test")
    test_acc = accuracy_score(test_true, test_pred)
    print(f"  测试集准确率: {test_acc*100:.2f}%")

    # 混淆矩阵
    _, train_cm_norm = compute_cm(train_true, train_pred)
    _, test_cm_norm  = compute_cm(test_true, test_pred)

    plot_confusion_matrix(train_cm_norm, f"训练集混淆矩阵 (Acc={train_acc*100:.2f}%)",
                          os.path.join(out_dir, "confusion_matrix_train.png"))
    plot_confusion_matrix(test_cm_norm, f"测试集混淆矩阵 (Acc={test_acc*100:.2f}%)",
                          os.path.join(out_dir, "confusion_matrix_test.png"))

    # 训练集原始计数混淆矩阵
    cm_train_raw, _ = compute_cm(train_true, train_pred)
    plot_confusion_matrix(cm_train_raw.astype(float),
                          f"训练集混淆矩阵 (计数)",
                          os.path.join(out_dir, "confusion_matrix_train_counts.png"))

    cm_test_raw, _ = compute_cm(test_true, test_pred)
    plot_confusion_matrix(cm_test_raw.astype(float),
                          f"测试集混淆矩阵 (计数)",
                          os.path.join(out_dir, "confusion_matrix_test_counts.png"))

    # 分类报告
    print_classification_report(train_true, train_pred, "训练集")
    print_classification_report(test_true, test_pred, "测试集")

    # 保存数值混淆矩阵
    np.savetxt(os.path.join(out_dir, "cm_train.csv"), cm_train_raw, delimiter=",", fmt="%d")
    np.savetxt(os.path.join(out_dir, "cm_test.csv"), cm_test_raw, delimiter=",", fmt="%d")

    print(f"\n评估完成，结果已保存至: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="config path (override)")
    parser.add_argument("--weights", default=None, help="weights path (override)")
    args = parser.parse_args()

    if args.config and args.weights:
        # 用外部指定的配置和权重
        yaml_path = args.config
        weights_path = args.weights
    else:
        yaml_path = os.path.join(_PROJECT_DIR, "configs", "optimal.yaml")
        weights_path = os.path.join(_PROJECT_DIR, "models", "action",
                                    "optimal", "20260429_134556", "best.pth")
    main(yaml_path, weights_path)
