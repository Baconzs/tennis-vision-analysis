"""seq_len_sweep.py — 序列长度扫描：固定权重，评估不同 seq_len 下的准确率"""
import os
import sys
import csv
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_DEMO_DIR)
_MST_DIR = os.path.join(_SRC_DIR, "model", "mst")
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _MST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_main import MSTFormer  # noqa
from dataset import TennisActionDataset  # noqa
from config import load_config  # noqa
from train import split_dataset  # noqa


def evaluate(model, loader, device, keyframe_only):
    model.eval()
    correct = total = 0
    kf_tp = kf_fp = kf_fn = 0
    per_class = {}

    with torch.no_grad():
        for pose, packed, labels, kf_labels in loader:
            pose = pose.to(device, non_blocking=True)
            packed = packed.to(device, non_blocking=True)
            labels = labels.to(device)
            kf_labels = kf_labels.to(device)

            ctx = torch.amp.autocast("cuda") if device.type == "cuda" else _nullctx()
            with ctx:
                if keyframe_only:
                    kf_logits = model(pose, packed)
                    action_preds = torch.zeros(kf_logits.shape[:2], dtype=torch.long, device=device)
                else:
                    action_logits, kf_logits = model(pose, packed)
                    action_preds = action_logits.argmax(-1)

            kf_preds = kf_logits.argmax(-1)

            mask = labels != -100
            if mask.any():
                c = ((action_preds == labels) & mask).sum().item()
                t = mask.sum().item()
                correct += c
                total += t
                for pred, gt in zip(action_preds[mask].cpu().tolist(), labels[mask].cpu().tolist()):
                    per_class.setdefault(gt, {"correct": 0, "total": 0})
                    per_class[gt]["total"] += 1
                    if pred == gt:
                        per_class[gt]["correct"] += 1

            kf_tp += ((kf_preds == 1) & (kf_labels == 1)).sum().item()
            kf_fp += ((kf_preds == 1) & (kf_labels == 0)).sum().item()
            kf_fn += ((kf_preds == 0) & (kf_labels == 1)).sum().item()

    acc = correct / total if total > 0 else 0.0
    prec = kf_tp / (kf_tp + kf_fp) if (kf_tp + kf_fp) > 0 else 0.0
    rec = kf_tp / (kf_tp + kf_fn) if (kf_tp + kf_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1, per_class


def main():
    parser = argparse.ArgumentParser(description="序列长度扫描")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[30, 60, 90, 120, 150, 180])
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg["device"]

    _, test_dirs = split_dataset(cfg["data_root"], cfg.get("train_ratio", 0.8))
    if not test_dirs:
        print("测试集为空，请检查 data_root")
        return

    model = MSTFormer(cfg).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)

    ACTION_NAMES = ["待机", "正手", "反手", "发球", "移动"]
    keyframe_only = cfg.get("keyframe_only", False)

    rows = []
    print(f"\n{'seq_len':>8}  {'Acc':>7}  {'KF-P':>7}  {'KF-R':>7}  {'KF-F1':>7}")
    print("-" * 50)

    for seq_len in args.seq_lens:
        cfg["seq_len"] = seq_len
        cfg["min_seq_len"] = max(10, seq_len // 2)
        ds = TennisActionDataset(cfg, clip_dirs=test_dirs, augment=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        acc, prec, rec, f1, per_class = evaluate(model, loader, device, keyframe_only)

        row = {"seq_len": seq_len, "accuracy": acc, "kf_precision": prec,
               "kf_recall": rec, "kf_f1": f1}
        for cid, name in enumerate(ACTION_NAMES):
            info = per_class.get(cid, {"correct": 0, "total": 0})
            row[f"acc_{name}"] = info["correct"] / info["total"] if info["total"] > 0 else 0.0
        rows.append(row)

        print(f"{seq_len:>8}  {acc*100:>6.2f}%  {prec*100:>6.2f}%  {rec*100:>6.2f}%  {f1*100:>6.2f}%")

    # 保存 CSV
    os.makedirs(os.path.join(_PROJECT_DIR, "runs"), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_stem = os.path.splitext(os.path.basename(args.config))[0]
    csv_path = os.path.join(_PROJECT_DIR, "runs", f"seq_len_sweep_{cfg_stem}_{ts}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n结果已保存：{csv_path}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
            import matplotlib.pyplot as plt
            seq_lens = [r["seq_len"] for r in rows]
            accs = [r["accuracy"] * 100 for r in rows]
            f1s = [r["kf_f1"] * 100 for r in rows]
            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(seq_lens, accs, "o-", color="#4CAF50", label="动作准确率 (%)")
            ax1.set_xlabel("序列长度（帧）")
            ax1.set_ylabel("准确率 (%)", color="#4CAF50")
            ax2 = ax1.twinx()
            ax2.plot(seq_lens, f1s, "s--", color="#FF9800", label="关键帧 F1 (%)")
            ax2.set_ylabel("关键帧 F1 (%)", color="#FF9800")
            fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
            plt.title("序列长度 vs 准确率")
            plt.tight_layout()
            plot_path = csv_path.replace(".csv", ".png")
            plt.savefig(plot_path, dpi=150)
            print(f"折线图已保存：{plot_path}")
        except ImportError:
            print("matplotlib 未安装，跳过绘图")


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


if __name__ == "__main__":
    main()
