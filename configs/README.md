# configs/ — 训练配置

YAML 配置分两组：YOLO 类（球场/人员）和 MSTFormer 动作识别类。完整逐项说明见 [`CONFIG_REFERENCE.md`](./CONFIG_REFERENCE.md)。

## YOLO 类（供 ultralytics 训练）

| 文件 | 用途 |
| --- | --- |
| `court_14pts_weighted.yaml` | 球场 14 关键点数据集配置（当前主力） |
| `court_keypoints*.yaml` | 球场关键点历史版本配置 |
| `person_sorter_dataset.yaml` | 人员分类数据集（近端/远端 2 类） |

## MSTFormer 类

统一基准：`embed_dim=128`、`depth=8`、`use_pos_encoding=false`。

| 路径 | 用途 |
| --- | --- |
| `main.yaml` | **当前最优基准**，直接用于正式训练 |
| `main_shared.yaml` | 共享 YOLO 骨干的变体 |
| `hyperparams/` | 超参调优（embed_dim、depth、visual_tokens 等） |
| `ablation/` | 消融实验（去姿态/去裁剪图/纯姿态/仅全帧） |
| `components/` | 组件对比（Focal vs CE、是否合并 token、不同骨干等） |
| `single_frame/` | 单帧分类基线配置 |

## 用法

```bash
python src/model/mst/train.py --config configs/main.yaml
python src/model/mst/run_ablation.py        # 批量跑 ablation/components/hyperparams
```
