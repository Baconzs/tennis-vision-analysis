"""
train_person_detector.py — 球员检测模型训练入口
功能：基于 data/person_sorter/ 微调 YOLO 球员检测模型
权重输出：runs/person_training/<run_name>/weights/best.pt

启动方式（在项目根目录 项目标注与测试/ 下执行）：
    python src/training/train_person_detector.py
"""
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# ── 路径配置 ──────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent.parent          # 项目标注与测试/
DATASET_YAML = PROJECT_DIR / "configs" / "person_sorter_dataset.yaml"
PRETRAIN_WEIGHTS = PROJECT_DIR / "models" / "person" / "best.pt"
RUNS_DIR = PROJECT_DIR / "runs" / "person_training"
RUN_NAME = "hard_neg_finetune_v1"


def prepare_dataset_yaml():
    """生成绝对路径版 dataset.yaml，避免 YOLO 路径解析问题"""
    data_root = (PROJECT_DIR / "data" / "person_sorter").as_posix()
    content = f"""# person_sorter_dataset.yaml — 自动生成，勿手动修改路径
path: {data_root}
train: images/train
val: images/val

names:
  0: player_near
  1: player_far
"""
    DATASET_YAML.write_text(content, encoding="utf-8")
    print(f"dataset.yaml 已更新: {DATASET_YAML}")
    return str(DATASET_YAML)


def check_data():
    """训练前检查数据集是否就绪"""
    train_imgs = PROJECT_DIR / "data" / "person_sorter" / "images" / "train"
    val_imgs = PROJECT_DIR / "data" / "person_sorter" / "images" / "val"
    n_train = len([f for f in os.listdir(train_imgs) if f.endswith(".jpg")])
    n_val = len([f for f in os.listdir(val_imgs) if f.endswith(".jpg")])
    print(f"数据集: train={n_train}, val={n_val}")
    if n_train < 10:
        raise RuntimeError("训练集样本不足，请先运行 merge_hard_negatives.py")
    return n_train, n_val


def train():
    yaml_path = prepare_dataset_yaml()
    n_train, n_val = check_data()

    print(f"\n开始训练 — 权重将保存到: {RUNS_DIR / RUN_NAME}")
    print(f"   基础权重: {PRETRAIN_WEIGHTS}")

    model = YOLO(str(PRETRAIN_WEIGHTS))

    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,

        # ── 性能优化 ──────────────────────────────────────────
        cache=True,
        amp=True,
        workers=4,

        # ── 输出路径控制（关键：防止权重散落） ──────────────────
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=False,          # 同名run会报错，避免覆盖

        # ── 训练策略 ──────────────────────────────────────────
        patience=20,
        save_period=10,
        optimizer="AdamW",
        lr0=1e-4,                # 微调用小学习率
        warmup_epochs=3,
    )

    # 训练完成后，将最优权重复制到 models/person/
    best_src = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    best_dst = PROJECT_DIR / "models" / "person" / f"best_{RUN_NAME}.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\n最优权重已复制到: {best_dst}")
        print(f"   如需替换生产权重，手动执行:")
        print(f"   copy {best_dst} {PROJECT_DIR / 'models' / 'person' / 'best.pt'}")
    else:
        print(f" 未找到 best.pt，请检查训练是否正常完成")


if __name__ == "__main__":
    train()
