"""prepare_weighted_dataset.py — 加权数据集合并工具

功能：将新标注数据与旧数据集合并，按比例划分 train/val，用于球场模型微调
"""
import os
import glob
import random
import shutil


def merge_and_split(source_img_dir, source_lbl_dir, target_base_dir, total_samples=1300, split_ratio=0.8):
    # 确保目标文件夹（旧数据集目录）存在
    for split in ['train', 'val']:
        os.makedirs(os.path.join(target_base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_base_dir, split, 'labels'), exist_ok=True)

    print("正在扫描你新标注的 300 张数据...")
    img_files = sorted(glob.glob(os.path.join(source_img_dir, "*.jpg")))

    valid_pairs = []
    for img_path in img_files:
        lbl_path = os.path.join(source_lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        if os.path.exists(lbl_path):
            valid_pairs.append((img_path, lbl_path))

    if len(valid_pairs) < total_samples:
        print(f"提示: 找到的有效文件只有 {len(valid_pairs)} 张。将全部合并。")
        total_samples = len(valid_pairs)

    # 截取你要的前 300 张
    selected_pairs = valid_pairs[:total_samples]

    # 随机打乱，确保训练集和验证集的样本分布均匀
    random.seed(42)
    random.shuffle(selected_pairs)

    # 8:2 拆分
    train_count = int(total_samples * split_ratio)
    train_pairs = selected_pairs[:train_count]
    val_pairs = selected_pairs[train_count:]

    print(f"准备汇入原有数据集: 新增训练集 {len(train_pairs)} 张 | 新增验证集 {len(val_pairs)} 张")

    # 执行复制合并操作
    def append_to_dataset(pairs, split_name):
        added_count = 0
        for img_src, lbl_src in pairs:
            base_name = os.path.basename(img_src)
            img_dst = os.path.join(target_base_dir, split_name, 'images', base_name)
            lbl_dst = os.path.join(target_base_dir, split_name, 'labels', os.path.basename(lbl_src))

            # 如果碰巧有同名文件，为了防止覆盖老数据，自动加个后缀
            if os.path.exists(img_dst):
                name, ext = os.path.splitext(base_name)
                new_base = f"{name}_v2{ext}"
                img_dst = os.path.join(target_base_dir, split_name, 'images', new_base)
                lbl_dst = os.path.join(target_base_dir, split_name, 'labels', f"{name}_v2.txt")

            shutil.copy(img_src, img_dst)
            shutil.copy(lbl_src, lbl_dst)
            added_count += 1

        # 统计目前该文件夹下的总图片数
        total_now = len(glob.glob(os.path.join(target_base_dir, split_name, 'images', '*.jpg')))
        print(f"  -> 成功将 {added_count} 张图片混入 {split_name} 文件夹！(当前 {split_name} 总数据量: {total_now} 张)")

    append_to_dataset(train_pairs, 'train')
    append_to_dataset(val_pairs, 'val')

    print(f"\n数据集扩充完毕！你的主数据集已变得更加强大。")


if __name__ == "__main__":
    import os as _os
    _PROJECT_DIR = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    SOURCE_IMAGES = _os.path.join(_PROJECT_DIR, "_archive", "Second_Train_Dataset", "images")
    SOURCE_LABELS = _os.path.join(_PROJECT_DIR, "_archive", "Second_Train_Dataset", "labels")
    TARGET_DATASET = _os.path.join(_PROJECT_DIR, "data", "court_finetune")
    merge_and_split(SOURCE_IMAGES, SOURCE_LABELS, TARGET_DATASET, total_samples=1300)