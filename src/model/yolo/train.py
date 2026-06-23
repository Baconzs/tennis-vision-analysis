"""
YOLO 单帧动作分类 — 训练入口

用法:
  cd 项目标注与测试
  .venv/Scripts/python src/model/yolo/train.py

产出: models/action/yolo_single_frame/<timestamp>/
  ├── best.pth
  ├── final.pth
  ├── train_log.csv
  └── config.txt
"""

import os, sys, csv, threading, json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np

# 添加搜索路径使 ultralytics 可用
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

from dataset import create_datasets, split_dataset, collect_samples
from model import YoloFrameClassifier

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_ROOT = "data/rallies_train"
WEIGHTS_PATH = "models/yolo/yolo11n.pt"
NUM_CLASSES = 5
TRAIN_RATIO = 0.8
BATCH_SIZE = 64
VIRTUAL_BATCH_SIZE = 128  # accumulation_steps = 2
TOTAL_EPOCHS = 60
LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 3
UNFREEZE_BACKBONE = True
IMG_SIZE = 224
NUM_WORKERS = 2

# 类别权重（与 MST 训练一致）
CLASS_WEIGHTS = [1.0, 4.0, 5.0, 4.0, 1.5]

# ── 输出 ──────────────────────────────────────────────────────────────────────
_MST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_MST_DIR)))
RUN_DIR = os.path.join(_PROJECT_DIR, "models", "action", "yolo_single_frame",
                        datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(RUN_DIR, exist_ok=True)
BEST_PATH = os.path.join(RUN_DIR, "best.pth")
FINAL_PATH = os.path.join(RUN_DIR, "final.pth")
LOG_PATH = os.path.join(RUN_DIR, "train_log.csv")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据
    train_ds, test_ds = create_datasets(DATA_ROOT, TRAIN_RATIO)

    accum = VIRTUAL_BATCH_SIZE // BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             persistent_workers=NUM_WORKERS > 0)

    # 模型
    model = YoloFrameClassifier(WEIGHTS_PATH, NUM_CLASSES, unfreeze_backbone=UNFREEZE_BACKBONE,
                                 img_size=IMG_SIZE).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")

    # 损失 + 优化器
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS),
        CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS,
                          eta_min=LEARNING_RATE * 0.01),
    ], milestones=[WARMUP_EPOCHS])

    scaler = torch.amp.GradScaler("cuda")

    # CSV 日志初始化
    log_fields = ["epoch", "lr", "train_loss", "train_acc", "test_acc",
                   "precision", "recall", "f1"]
    # 每类指标
    for n in ["idle", "fh", "bh", "serve", "move"]:
        log_fields.extend([f"pred_{n}", f"gt_{n}"])
    log_fields.append("best_metric")

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_acc = 0.0
    best_state = None
    save_thread = None

    print(f"\nStarting training: {TOTAL_EPOCHS} epochs, batch={BATCH_SIZE}, accum={accum}")
    print("=" * 70)

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        total_loss = correct = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{TOTAL_EPOCHS}]")
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels) / accum

            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                preds = logits.argmax(-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            total_loss += loss.item() * accum
            pbar.set_postfix(loss=f"{loss.item()*accum:.4f}")

        scheduler.step()
        train_acc = correct / total * 100
        avg_loss = total_loss / len(train_loader)

        # 测试
        model.eval()
        t_correct = t_total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                preds = logits.argmax(-1)
                t_correct += (preds == labels).sum().item()
                t_total += labels.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        test_acc = t_correct / t_total * 100
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # 每类 P/R/F1
        pred_counts = torch.bincount(all_preds, minlength=NUM_CLASSES).tolist()
        gt_counts = torch.bincount(all_labels, minlength=NUM_CLASSES).tolist()

        # 宏平均
        per_class_f1 = []
        for c in range(NUM_CLASSES):
            tp = ((all_preds == c) & (all_labels == c)).sum().item()
            fp = ((all_preds == c) & (all_labels != c)).sum().item()
            fn = ((all_preds != c) & (all_labels == c)).sum().item()
            prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class_f1.append(f1)

        macro_prec = sum(per_class_f1) / NUM_CLASSES  # 近似

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n  LR={lr_now:.2e} | Train Loss={avg_loss:.4f} | "
              f"Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}%")
        print(f"  Pred: {pred_counts}")
        print(f"  GT:   {gt_counts}")

        # 保存最优
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_thread is not None:
                save_thread.join()
            save_thread = threading.Thread(
                target=lambda sd, p: torch.save(sd, p),
                args=(best_state, BEST_PATH), daemon=True)
            save_thread.start()
            print(f"  New best: {test_acc:.2f}%")

        # CSV 日志
        row = {"epoch": epoch + 1, "lr": f"{lr_now:.2e}",
               "train_loss": f"{avg_loss:.4f}",
               "train_acc": f"{train_acc:.2f}", "test_acc": f"{test_acc:.2f}",
               "precision": f"{macro_prec:.2f}", "recall": f"{macro_prec:.2f}",
               "f1": f"{macro_prec:.2f}",
               "pred_idle": pred_counts[0], "pred_fh": pred_counts[1],
               "pred_bh": pred_counts[2], "pred_serve": pred_counts[3],
               "pred_move": pred_counts[4],
               "gt_idle": gt_counts[0], "gt_fh": gt_counts[1],
               "gt_bh": gt_counts[2], "gt_serve": gt_counts[3],
               "gt_move": gt_counts[4],
               "best_metric": f"{best_acc:.2f}"}
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)
        print("-" * 70)

    # 最终权重
    torch.save(model.state_dict(), FINAL_PATH)
    if save_thread is not None:
        save_thread.join()

    # 保存配置
    config_info = {
        "data_root": DATA_ROOT, "weights_path": WEIGHTS_PATH,
        "num_classes": NUM_CLASSES, "train_ratio": TRAIN_RATIO,
        "batch_size": BATCH_SIZE, "virtual_batch_size": VIRTUAL_BATCH_SIZE,
        "total_epochs": TOTAL_EPOCHS, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "warmup_epochs": WARMUP_EPOCHS,
        "unfreeze_backbone": UNFREEZE_BACKBONE, "img_size": IMG_SIZE,
    }
    with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
        json.dump(config_info, f, indent=2)

    print(f"\nDone! Best test acc: {best_acc:.2f}%")
    print(f"Best: {BEST_PATH}")
    print(f"Final: {FINAL_PATH}")


if __name__ == "__main__":
    train()
