"""test_dataset.py — 数据集加载验证脚本"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dataset import TennisActionDataset
from config import load_config


def test_dataset_pipeline(yaml_path=None):
    cfg = load_config(yaml_path)
    seq_len = cfg["seq_len"]

    print("=" * 50)
    print(f"数据根目录: {cfg['data_root']}")

    try:
        print("⏳ 初始化 TennisActionDataset...")
        dataset = TennisActionDataset(cfg)
        print(f"初始化成功，共 {len(dataset)} 个切片。")

        if len(dataset) == 0:
            print("没有找到任何数据，请检查 data_root 路径。")
            return

        print(f"\n⏳ 读取第一个样本 (idx=0)...")
        pose, packed, labels = dataset[0]

        print("\n样本加载成功，张量形状：")
        print("-" * 50)
        print(f"  pose:   {tuple(pose.shape)}   期望: ({seq_len}, 97)")
        print(f"  packed: {tuple(packed.shape)}  期望: ({seq_len}, 3, 320, 960)")
        print(f"  labels: {tuple(labels.shape)}  期望: ({seq_len},)")
        print("-" * 50)
        print(f"标签（前30帧）: {labels[:30].tolist()}")
        print("=" * 50)
        print("测试通过！")

    except Exception:
        import traceback
        print("\nDataset 抛出异常：")
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    test_dataset_pipeline(args.config)
