"""yolo-train.py — YOLO 人员分类模型训练脚本

功能：基于 data/person_sorter/ 数据集，微调 YOLO 模型用于球员分类
"""
from ultralytics import YOLO


def main():
    # 加载预训练模型
    model = YOLO("yolo26x.pt")

    # 启动极致优化训练
    results = model.train(
        data="data/dataset.yaml",
        epochs=150,
        imgsz=640,  # 针对小目标，切勿随意缩小分辨率
        batch=4,  # 根据你的显存大小调整 (例如 3090/4090 可以尝试 8 或 16)
        device=0,

        # --- 提速与内存/显存优化核心参数 ---
        cache=True,  # 【核心】将图片预载入内存，彻底消灭硬盘 I/O 等待
        amp=True,  # 【核心】开启 FP16 混合精度，显存占用减半，速度翻倍
        workers=8,  # 【核心】Dataloader 线程数，Windows 建议 4 或 8
        # accumulate=4,     # 【备选】如果显存依然爆炸，解开此注释，配合 batch=2 使用

        optimizer="MuSGD",  # YOLO26 新特性优化器
        project="tennis_tracking",
        name="yolo26x_optimized_run",

        # 其他实用参数
        patience=30,  # 引入早停机制：如果 30 个 epoch 精度无提升则提前结束训练
        save_period=10  # 每 10 个 epoch 备份一次权重，防止电脑意外死机
    )


if __name__ == "__main__":
    main()