# src/ — 源代码总览

本目录是网球比赛视觉分析系统的全部源代码。按功能分为「顶层入口 + 5 个子包」。

> 所有脚本默认从**仓库根目录**运行（路径相对根目录），例如 `python src/main.py`。

## 顶层文件（批量处理流水线）

| 文件 | 作用 | 怎么用 |
| --- | --- | --- |
| `main.py` | 批量视频处理主入口：遍历 `videos/`，CPU 线程巡场切回合 + GPU 线程跑姿态追踪，输出每个回合的 `raw_clip.mp4` / `annotated_clip.mp4` / `pose_data.json`，支持断点续跑 | `python src/main.py` |
| `config_legacy.py` | `main.py` 的配置（视频/输出/模型路径、置信度、EMA 参数等），改路径和阈值在这里 | 被 `main.py`、`pose_tracker.py` import |
| `court_detector.py` | 巡场阶段的球场 ROI 检测器：用霍夫直线快速判断画面里有没有球场、框出远/近端区域（轻量，不是关键点模型） | 被 `main.py` 调用 |
| `pose_tracker.py` | 姿态追踪器：在 ROI 内跑 YOLO-pose，多维打分选出真正的球员，含 EMA 平滑与丢帧补偿 | 被 `main.py` 调用 |
| `train_court_pipeline.py` | 球场 **14 关键点**检测模型训练入口（YOLO-pose 微调），并导出 Bad Cases 供迭代 | `python src/train_court_pipeline.py` |
| `test_person_detector.py` | 人员检测/分类模型的快速测试脚本 | `python src/test_person_detector.py` |

## 子包

| 目录 | 内容 | 详见 |
| --- | --- | --- |
| `pipeline/` | 离线精追踪（球场单应矩阵 + 轨迹）、球场标注采样/精修工具、数据集合并 | [`pipeline/README.md`](./pipeline/README.md) |
| `model/mst/` | **MSTFormer** 动作识别模型（核心）：模型定义、数据集、训练、消融、评估 | [`model/mst/README.md`](./model/mst/README.md) |
| `model/yolo/` | 单帧 YOLO 动作分类模型（对比基线） | [`model/yolo/README.md`](./model/yolo/README.md) |
| `demo/` | PyQt5 桌面 Demo：视频播放 + 时间轴 + 实时推理可视化 | [`demo/README.md`](./demo/README.md) |
| `utils/` | 标注工具、数据处理、论文图表与评估脚本 | [`utils/README.md`](./utils/README.md) |
| `training/` | 人员检测模型训练与难例挖掘 | [`training/README.md`](./training/README.md) |

## 两条主线一图看懂

```
A. 数据生产线（从视频到训练样本）
   videos/ ──main.py──▶ data/rallies_new/（回合切片 + pose_data.json）
                          │
                          ├─ utils/action_annotator.py ─▶ 标动作 annotations.json
                          └─ model/mst/extract_crops.py ─▶ 球员裁剪图 player1/ player2/

B. 模型线
   球场关键点:  train_court_pipeline.py ─▶ models/court/best.pt
   人员分类:    training/train_person_detector.py ─▶ models/person/best.pt
   动作识别:    model/mst/train.py ─▶ models/action/<config>/<时间戳>/best.pth
```

## 约定

- 代码注释统一中文；模块顶部统一用 `"""docstring"""` 说明文件用途。
- 处理中文路径：Windows 下用短路径规避 OpenCV 编码问题，非 Windows 自动跳过（见各文件 `_get_short_path`）。
- 大文件（`videos/`、`data/`、`models/`、`runs/`）不在仓库内，需自行准备，路径约定见 `config_legacy.py` 与 `configs/`。
