# training/ — 人员检测/分类模型训练

训练「近端球员 / 远端球员」识别模型，并配套难例挖掘（Hard Negative Mining）。

| 文件 | 作用 | 运行 |
| --- | --- | --- |
| `train_person_detector.py` | 基于 `data/person_sorter/` 微调 YOLO，区分 `player_near` / `player_far` | `python src/training/train_person_detector.py` |
| `merge_hard_negatives.py` | 把挖到的难例（误检的球童/观众等）合并进训练集，提升判别力 | `python src/training/merge_hard_negatives.py` |
| `yolo-train-legacy.py` | 旧版训练脚本，保留备查 | — |

## 配套数据工具（在 `../utils/`）

```
data-creater.py      采样图片到 data/person_sorter/image/
label_tool.py        标注 bounding box（近/远端）
dataset_splitter.py  划分 train/val
─────────────────────────────────────────
train_person_detector.py   训练
hard_negative_extractor.py / hard_negative_reviewer.py   难例挖掘与复核（../utils/）
```

数据集配置：`configs/person_sorter_dataset.yaml`。
