# model/mst/ — MSTFormer 动作识别模型（项目核心）

MSTFormer（Multi-Stream Transformer）是本项目的核心模型。它把**球员姿态序列**、**球场几何位置**和**多路视觉裁剪图**融合进一个 Transformer，**双头输出**：

- 动作分类：待机 / 正手 / 反手 / 发球 / 移动（5 类）
- 关键帧检测：每帧是否为动作切换的关键帧（二分类）

## 输入构成

| 输入 | 维度 | 来源 |
| --- | --- | --- |
| 姿态物理特征 `pose` | `[B, T, 125]` | `pose_data.json`，由 `dataset.py` 的 `_build_pose_vec` 构建 |
| 打包视觉帧 `packed_frames` | `[B, T, 3, 320, 960]` uint8 | 全帧 + player1 + player2 三路横向拼接 |

> 125 维 = 17×3(关键点绝对坐标+置信度) + 17×2(相对人物中心) + 2(人物中心相对球场) + 2(速度) + 2(加速度) + 6(球，预留) + 28(球场14点×2)。视觉帧归一化在 GPU 端做，CPU 端保持 uint8 省带宽。

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `model_main.py` | **模型定义** `MSTFormer`。三路视觉流 → 可选合并（`merge_visual_tokens`）→ 拼接姿态 token → Transformer → 双头输出。各开关：`use_pose` / `use_player_crops` / `use_visual` / `merge_visual_tokens` |
| `dataset.py` | 数据集 `TennisActionDataset`。读 `pose_data.json` + `annotations.json`，滑动窗口切片，构建 125 维姿态向量与三路视觉帧；含图像增强 |
| `train.py` | **训练入口**。联合训练动作分类 + 关键帧检测，AMP + 梯度累积，按视频划分 train/val，输出到 `models/action/<config>/<时间戳>/` |
| `config.py` | YAML 配置解析器，把相对路径转绝对、补设备与梯度累积步数 |
| `augment.py` | 异步图像增强缓冲区，把增强从 DataLoader worker 移到独立线程池 |
| `extract_frames.py` | 预提取回合视频的全帧到 `frames/`（加速训练读图） |
| `extract_crops.py` | 预提取 player1/player2 裁剪图到 `player1/`、`player2/` |
| `run_ablation.py` | 批量跑 `configs/ablation`、`components`、`hyperparams` 下的实验 |
| `modules/` | 模型子模块（见下） |
| `tests/` | `eval_optimal.py`（评估+混淆矩阵）、`test_matrix.py`、`test_dataset.py` |

### modules/ 子模块

| 文件 | 作用 |
| --- | --- |
| `backbone_factory.py` | 视觉骨干工厂，按 `visual_backbone` 配置构建下面四种之一 |
| `yolo_extractor.py` | YOLO11 backbone 截取 P3/P4/P5，跨尺度注意力融合（主力） |
| `resnet_extractor.py` | ResNet18 骨干（对比） |
| `vit_extractor.py` | 轻量 ViT patch embedding（对比） |
| `raw_extractor.py` | 原始像素投影（对比） |
| `token_resampler.py` | Perceiver 风格 cross-attention，把任意数量 token 压到固定个数 |
| `pos_encoding.py` | 正弦位置编码（默认关闭） |
| `action_head.py` | 动作分类头 + 关键帧检测头 |

## 怎么训练

```bash
# 0) 准备：每个回合目录需有 pose_data.json + annotations.json
#    可选预提取视觉数据（否则训练时从 raw_clip.mp4 实时解码，较慢）：
python src/model/mst/extract_frames.py
python src/model/mst/extract_crops.py

# 1) 冒烟测试：1 条数据跑 1 轮，验证流水线通畅
python src/model/mst/train.py --config configs/main.yaml --smoke

# 2) 正式训练（main.yaml 为当前最优基准）
python src/model/mst/train.py --config configs/main.yaml

# 3) 评估 + 混淆矩阵
python src/model/mst/tests/eval_optimal.py --config configs/main.yaml --weights <best.pth>

# 4) 批量消融/超参/组件实验
python src/model/mst/run_ablation.py
```

配置说明见 [`configs/CONFIG_REFERENCE.md`](../../../configs/CONFIG_REFERENCE.md)。
