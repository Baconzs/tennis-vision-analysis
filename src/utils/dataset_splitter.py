"""dataset_splitter.py — 数据集 train/val 划分工具

功能：将 data/person_sorter/ 中的图片按比例随机划分为训练集和验证集
"""
import os
import random
import shutil
from pathlib import Path


def split_dataset(data_dir, train_ratio=0.8):
    base_path = Path(data_dir)

    # 原始数据路径
    src_images = base_path / "image"
    src_labels = base_path / "labels"

    # YOLO 标准目标路径
    # 注意：YOLO 默认习惯文件夹叫 'images' 而不是 'image' (带 s)
    train_images_dir = base_path / "images" / "train"
    val_images_dir = base_path / "images" / "val"
    train_labels_dir = base_path / "labels" / "train"
    val_labels_dir = base_path / "labels" / "val"

    # 创建目标文件夹
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("开始扫描已标注的数据...")

    # 获取所有图片文件
    valid_exts = {'.jpg', '.jpeg', '.png'}
    all_images = [f for f in src_images.iterdir() if f.suffix.lower() in valid_exts]

    # 筛选出有对应 txt 标签的图片（过滤掉没标的废片）
    valid_data_pairs = []
    for img_path in all_images:
        label_name = img_path.stem + ".txt"
        label_path = src_labels / label_name

        if label_path.exists():
            valid_data_pairs.append((img_path, label_path))

    total_valid = len(valid_data_pairs)
    if total_valid == 0:
        print("没有找到配对的图片和标签，请检查路径。")
        return

    print(f"找到 {total_valid} 张有效标注的图片。")

    # 随机打乱数据
    random.seed(42)  # 固定随机种子，保证每次划分结果一致
    random.shuffle(valid_data_pairs)

    # 计算划分索引
    split_index = int(total_valid * train_ratio)
    train_data = valid_data_pairs[:split_index]
    val_data = valid_data_pairs[split_index:]

    print(f"开始划分数据 (训练集: {len(train_data)} 张, 验证集: {len(val_data)} 张)...")

    # 辅助复制函数
    def copy_data(data_list, target_img_dir, target_label_dir):
        for img_src, label_src in data_list:
            shutil.copy2(img_src, target_img_dir / img_src.name)
            shutil.copy2(label_src, target_label_dir / label_src.name)

    # 执行复制
    copy_data(train_data, train_images_dir, train_labels_dir)
    copy_data(val_data, val_images_dir, val_labels_dir)

    print("-" * 30)
    print("数据集划分完成！目前的目录结构已满足 YOLO 要求。")


if __name__ == "__main__":
    # 指向你的 data 根目录
    DATA_DIR = "data/person_sorter"
    split_dataset(DATA_DIR, train_ratio=0.8)