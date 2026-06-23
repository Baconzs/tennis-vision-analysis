"""
将 data/rallies_annotated/ 中的标注数据复制到 data/rallies_train/，
只复制训练所需的 3 个文件，跳过 annotated_clip.mp4。
支持断点续跑。
"""
import os
import shutil

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(_PROJECT_DIR, "data", "rallies_annotated")
DST_DIR = os.path.join(_PROJECT_DIR, "data", "rallies_train")

REQUIRED_FILES = ["raw_clip.mp4", "pose_data.json", "annotations.json"]


def is_complete(dst_rally_dir):
    return all(os.path.exists(os.path.join(dst_rally_dir, f)) for f in REQUIRED_FILES)


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    rally_dirs = sorted(
        d for d in os.listdir(SRC_DIR)
        if os.path.isdir(os.path.join(SRC_DIR, d))
    )

    copied = skipped = missing = 0

    for rally in rally_dirs:
        src_rally = os.path.join(SRC_DIR, rally)
        dst_rally = os.path.join(DST_DIR, rally)

        if is_complete(dst_rally):
            skipped += 1
            continue

        # 检查源目录是否有所有必要文件
        if not all(os.path.exists(os.path.join(src_rally, f)) for f in REQUIRED_FILES):
            print(f"  [SKIP] {rally} — 源目录缺少必要文件")
            missing += 1
            continue

        os.makedirs(dst_rally, exist_ok=True)
        for fname in REQUIRED_FILES:
            shutil.copy2(os.path.join(src_rally, fname), os.path.join(dst_rally, fname))

        print(f"  [COPY] {rally}")
        copied += 1

    print(f"\n完成：复制 {copied} 个，跳过 {skipped} 个（已存在），缺失 {missing} 个")
    print(f"目标目录：{DST_DIR}")


if __name__ == "__main__":
    main()
