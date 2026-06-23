"""
merge_hard_negatives.py — 难例数据合并脚本
功能：将 hard_negatives 中已标注的样本按 8:2 比例合并到 train/val 集
"""
import os
import shutil
from pathlib import Path
import random

# 路径配置
CURRENT_DIR = Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent.parent
PERSON_SORTER_DIR = PROJECT_DIR / "data" / "person_sorter"

HARD_NEG_IMAGES = PERSON_SORTER_DIR / "hard_negatives" / "images"
HARD_NEG_LABELS = PERSON_SORTER_DIR / "hard_negatives" / "labels"

TRAIN_IMAGES = PERSON_SORTER_DIR / "images" / "train"
TRAIN_LABELS = PERSON_SORTER_DIR / "labels" / "train"
VAL_IMAGES = PERSON_SORTER_DIR / "images" / "val"
VAL_LABELS = PERSON_SORTER_DIR / "labels" / "val"


def main():
    print("扫描 hard_negatives 目录...")

    # 获取所有已标注的图片（有对应label文件的）
    all_images = []
    for img_name in os.listdir(HARD_NEG_IMAGES):
        if not img_name.lower().endswith(".jpg"):
            continue
        label_name = img_name.rsplit(".", 1)[0] + ".txt"
        label_path = HARD_NEG_LABELS / label_name
        if label_path.exists():
            all_images.append(img_name)

    print(f"找到 {len(all_images)} 个已标注样本")

    if len(all_images) == 0:
        print("没有找到已标注的样本，请先运行 hard_negative_reviewer.py")
        return

    # 8:2 分割
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_list = all_images[:split_idx]
    val_list = all_images[split_idx:]

    print(f"分割比例: train={len(train_list)}, val={len(val_list)}")

    # 复制到 train
    print("\n复制到 train 集...")
    for img_name in train_list:
        label_name = img_name.rsplit(".", 1)[0] + ".txt"
        shutil.copy2(HARD_NEG_IMAGES / img_name, TRAIN_IMAGES / img_name)
        shutil.copy2(HARD_NEG_LABELS / label_name, TRAIN_LABELS / label_name)

    # 复制到 val
    print("复制到 val 集...")
    for img_name in val_list:
        label_name = img_name.rsplit(".", 1)[0] + ".txt"
        shutil.copy2(HARD_NEG_IMAGES / img_name, VAL_IMAGES / img_name)
        shutil.copy2(HARD_NEG_LABELS / label_name, VAL_LABELS / label_name)

    # 统计最终数量
    final_train = len(os.listdir(TRAIN_IMAGES))
    final_val = len(os.listdir(VAL_IMAGES))

    print(f"\n合并完成！")
    print(f"   Train: {final_train} 张")
    print(f"   Val: {final_val} 张")
    print(f"\n下一步: 运行 python src/training/train_person_detector.py")


if __name__ == "__main__":
    main()
