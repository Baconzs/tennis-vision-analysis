"""config.py — YAML 配置解析器"""
import os
import torch
import yaml


def load_config(yaml_path=None):
    _mst_dir = os.path.dirname(os.path.abspath(__file__))          # .../src/model/mst
    _project_dir = os.path.dirname(os.path.dirname(os.path.dirname(_mst_dir)))  # .../项目标注与测试

    if yaml_path is None:
        yaml_path = os.path.join(_project_dir, "configs", "main.yaml")
    elif not os.path.isabs(yaml_path):
        yaml_path = os.path.join(_project_dir, yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["accumulation_steps"] = cfg["virtual_batch_size"] // cfg["batch_size"]

    for key in ("data_root", "crops_root", "backbone_weights"):
        if key in cfg and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(_project_dir, cfg[key])

    return cfg
