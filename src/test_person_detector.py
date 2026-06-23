"""
test_person_detector.py — 球员检测模型全量测试
功能：在所有训练和测试图片上运行推理，生成准确率报告和可视化结果
"""
import os
import json
import ctypes
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# ── 路径配置 ──────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent  # 项目标注与测试/
MODEL_PATH = PROJECT_DIR / "runs" / "person_training" / "hard_neg_finetune_v12" / "weights" / "best.pt"
DATA_DIR = PROJECT_DIR / "data" / "person_sorter" / "images"
LABELS_DIR = PROJECT_DIR / "data" / "person_sorter" / "labels"
RESULTS_DIR = PROJECT_DIR / "results" / "person_test"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_short_path(path_str):
    """Windows 中文路径转短路径"""
    try:
        buf = ctypes.create_unicode_buffer(260)
        if not hasattr(ctypes, "windll"):  # 非 Windows 直接用原路径
            return path_str
        ctypes.windll.kernel32.GetShortPathNameW(path_str, buf, 260)
        return buf.value
    except:
        return path_str

# 类别映射
CLASS_NAMES = {0: "player_near", 1: "player_far"}


def load_labels(img_path, split):
    """从 .txt 标签文件读取 GT（YOLO 格式）"""
    label_path = LABELS_DIR / split / img_path.name.replace(".jpg", ".txt")
    if not label_path.exists():
        return []

    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                labels.append({
                    "class": cls_id,
                    "bbox": (x_center, y_center, w, h)
                })
    return labels


def iou(box1, box2):
    """计算两个 YOLO 格式 bbox 的 IoU"""
    def yolo_to_xyxy(x_c, y_c, w, h):
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return x1, y1, x2, y2

    x1_1, y1_1, x2_1, y2_1 = yolo_to_xyxy(*box1)
    x1_2, y1_2, x2_2, y2_2 = yolo_to_xyxy(*box2)

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def test_dataset(split="all"):
    """测试数据集"""
    if split == "all":
        img_dirs = [DATA_DIR / "train", DATA_DIR / "val"]
    elif split == "train":
        img_dirs = [DATA_DIR / "train"]
    else:
        img_dirs = [DATA_DIR / "val"]

    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model_short = get_short_path(str(MODEL_PATH))
    model = YOLO(model_short)

    # 收集所有图片
    all_imgs = []
    for img_dir in img_dirs:
        if img_dir.exists():
            for f in os.listdir(str(img_dir)):
                if f.endswith(".jpg"):
                    all_imgs.append(img_dir / f)

    print(f"测试集大小: {len(all_imgs)} 张图片")

    # 统计指标
    stats = {
        "total": len(all_imgs),
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "class_stats": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        "per_image": []
    }

    # 逐图推理
    for idx, img_path in enumerate(all_imgs):
        if (idx + 1) % 100 == 0:
            print(f"  进度: {idx + 1}/{len(all_imgs)}")

        # 确定 split
        split = "train" if "train" in str(img_path) else "val"

        # 读取 GT
        gt_labels = load_labels(img_path, split)

        # 推理
        img_short = get_short_path(str(img_path))
        results = model.predict(source=img_short, conf=0.5, verbose=False)
        pred_boxes = []
        if results and len(results) > 0:
            for det in results[0].boxes:
                x_c = (det.xyxy[0][0] + det.xyxy[0][2]) / 2 / results[0].orig_shape[1]
                y_c = (det.xyxy[0][1] + det.xyxy[0][3]) / 2 / results[0].orig_shape[0]
                w = (det.xyxy[0][2] - det.xyxy[0][0]) / results[0].orig_shape[1]
                h = (det.xyxy[0][3] - det.xyxy[0][1]) / results[0].orig_shape[0]
                pred_boxes.append({
                    "class": int(det.cls[0]),
                    "conf": float(det.conf[0]),
                    "bbox": (x_c, y_c, w, h)
                })

        # 匹配 GT 和预测（贪心匹配）
        matched_gt = set()
        matched_pred = set()

        for pred_idx, pred in enumerate(pred_boxes):
            best_iou = 0.5
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_labels):
                if gt_idx in matched_gt:
                    continue
                if pred["class"] != gt["class"]:
                    continue
                box_iou = iou(pred["bbox"], gt["bbox"])
                if box_iou > best_iou:
                    best_iou = box_iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                stats["tp"] += 1
                stats["class_stats"][pred["class"]]["tp"] += 1
            else:
                stats["fp"] += 1
                stats["class_stats"][pred["class"]]["fp"] += 1

        # FN
        for gt_idx in range(len(gt_labels)):
            if gt_idx not in matched_gt:
                stats["fn"] += 1
                stats["class_stats"][gt_labels[gt_idx]["class"]]["fn"] += 1

        # 记录单张图片结果
        stats["per_image"].append({
            "image": str(img_path.relative_to(PROJECT_DIR)),
            "gt_count": len(gt_labels),
            "pred_count": len(pred_boxes),
            "tp": len(matched_pred),
            "fp": len(pred_boxes) - len(matched_pred),
            "fn": len(gt_labels) - len(matched_gt)
        })

    # 计算指标
    precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
    recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 输出报告
    report = {
        "split": split,
        "total_images": stats["total"],
        "total_tp": stats["tp"],
        "total_fp": stats["fp"],
        "total_fn": stats["fn"],
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "class_metrics": {}
    }

    for cls_id, cls_name in CLASS_NAMES.items():
        cls_stat = stats["class_stats"][cls_id]
        cls_p = cls_stat["tp"] / (cls_stat["tp"] + cls_stat["fp"]) if (cls_stat["tp"] + cls_stat["fp"]) > 0 else 0
        cls_r = cls_stat["tp"] / (cls_stat["tp"] + cls_stat["fn"]) if (cls_stat["tp"] + cls_stat["fn"]) > 0 else 0
        cls_f1 = 2 * cls_p * cls_r / (cls_p + cls_r) if (cls_p + cls_r) > 0 else 0

        report["class_metrics"][cls_name] = {
            "tp": cls_stat["tp"],
            "fp": cls_stat["fp"],
            "fn": cls_stat["fn"],
            "precision": round(cls_p, 4),
            "recall": round(cls_r, 4),
            "f1": round(cls_f1, 4)
        }

    return report, stats["per_image"]


if __name__ == "__main__":
    print("开始测试 person 检测模型\n")

    # 测试全量数据
    report, per_image = test_dataset("all")

    # 保存报告
    report_path = RESULTS_DIR / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 保存逐图结果
    per_image_path = RESULTS_DIR / "per_image_results.json"
    with open(per_image_path, "w", encoding="utf-8") as f:
        json.dump(per_image, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print("\n" + "="*60)
    print("测试结果摘要")
    print("="*60)
    print(f"总图片数: {report['total_images']}")
    print(f"TP: {report['total_tp']}, FP: {report['total_fp']}, FN: {report['total_fn']}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall: {report['recall']:.4f}")
    print(f"F1-Score: {report['f1']:.4f}")
    print("\n按类别:")
    for cls_name, metrics in report["class_metrics"].items():
        print(f"  {cls_name}:")
        print(f"    P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    print(f"\n报告已保存到: {report_path}")
    print(f"逐图结果已保存到: {per_image_path}")
