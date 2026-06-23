# utils/ — 标注、数据处理与评估工具

本目录是各类一站式工具脚本，按用途分五类。大多数从仓库根目录运行。

## 1. 标注工具

| 文件 | 作用 | 运行 |
| --- | --- | --- |
| `action_annotator.py` | **动作时序标注**（Flask Web 应用）。从 `data/rallies_new/` 轮换抽片段，浏览器里标 5 类动作时间段，存 `annotations.json`；支持删片段、进度持久化 | `python src/utils/action_annotator.py` 后访问 http://localhost:5000 |
| `label_tool.py` | 球员 **bbox 标注**（OpenCV GUI）：拖框标近/远端球员，存 YOLO txt | `python src/utils/label_tool.py` |

## 2. 数据生产与处理

| 文件 | 作用 |
| --- | --- |
| `data-batch-extractor.py` | 批量回合数据提取流水线：遍历 `rallies_new/` 跑球场+姿态，输出 `tracking_data.json`，断点续跑 |
| `data-creater.py` | 人员分类采样：从回合随机抽帧到 `data/person_sorter/image/` |
| `dataset_splitter.py` | 把 `person_sorter/` 图片按比例划分 train/val |
| `prepare_train_dataset.py` | 从 `rallies_annotated/` 复制训练所需文件到 `rallies_train/`，断点续跑 |
| `merge_annotating_data.py` | 把 `rallies_annotating/` 新标注合并进 `rallies_annotated/`（转换 tracking→pose 格式） |
| `add_court_to_pose.py` | 给旧数据逐帧补球场 14 点，写入 `pose_data.json` 的 `court` 字段 |
| `rerun_pose_detection.py` | 在球员裁剪图上低阈值重跑 pose，坐标映射回原图并过滤 |
| `trim_waiting_segments.py` | 修剪过长的「待机」片段，缓解类别不均衡 |

## 3. 推理可视化与测试

| 文件 | 作用 |
| --- | --- |
| `inference_viewer.py` | 人员分类模型推理可视化 |
| `test_person_on_video.py` / `visualize_person_test.py` | 在视频上测试人员检测并可视化 |
| `visualize_data_quality.py` | 数据质量可视化（标注/姿态是否正常） |
| `side_by_side_viewer.py` | 多个结果并排对比查看 |

## 4. 评估与报告

| 文件 | 作用 |
| --- | --- |
| `batch_eval_all.py` | 批量评估所有训练好的模型 |
| `generate_model_report.py` | 汇总各模型指标生成报告 |
| `analyze_class_distribution.py` | 统计动作类别分布 |

## 5. 难例挖掘

| 文件 | 作用 |
| --- | --- |
| `hard_negative_extractor.py` | 从误检中挖掘难负样本 |
| `hard_negative_reviewer.py` | 人工复核难例 |

## 6. 论文图表（仅用于撰写论文，与系统运行无关）

`generate_thesis_figures.py`、`generate_ch3_figures.py`、`generate_confusion_figures.py`、`generate_confusion_matrices.py`、`create_thesis_figure_N.py`、`extract_forehand_frame.py`、`unify_citations.py` —— 生成论文用的训练曲线、混淆矩阵、配图等，输出到 `docs/figures/`。

> 提示：本类脚本依赖本地数据集才能复现，开源仓库未附带数据，结果图见 `docs/figures/`。
