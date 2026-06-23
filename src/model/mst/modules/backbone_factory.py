"""backbone_factory.py — 视觉骨干工厂（按配置构建 YOLO11/ResNet/ViT/Raw）"""
import torch.nn as nn
from .yolo_extractor import Yolo11BackboneExtractor
from .resnet_extractor import ResNet18BackboneExtractor
from .raw_extractor import RawProjectionExtractor
from .vit_extractor import ViTPatchExtractor
from .token_resampler import TokenResampler


class ViTPatchExtractorWithResampler(nn.Module):
    """ViTPatchExtractor + TokenResampler：保留细粒度 patch 特征，压缩到固定 num_out 个 token。"""

    def __init__(self, patch_grid, embed_dim, num_out=16):
        super().__init__()
        self.extractor = ViTPatchExtractor(patch_grid, embed_dim)
        self.resampler = TokenResampler(embed_dim, num_out=num_out)

    def forward(self, x):
        tokens = self.extractor(x)    # (B, patch_grid², embed_dim)
        return self.resampler(tokens)  # (B, num_out, embed_dim)


def build_visual_extractor(cfg, shared_backbone=None, use_global_weights=False):
    backbone = cfg.get("visual_backbone", "yolo11")
    k = cfg["tokens_per_scale"]
    d = cfg["embed_dim"]
    vt = cfg.get("visual_tokens", 16)
    if backbone == "yolo11":
        weights = (cfg.get("global_backbone_weights") or cfg["backbone_weights"]) \
                  if use_global_weights else cfg["backbone_weights"]
        return Yolo11BackboneExtractor(
            weights,
            cfg["multi_scale_levels"],
            k, d, visual_tokens=vt,
            shared_backbone=shared_backbone,
            unfreeze_backbone=cfg.get("unfreeze_backbone", False),
        )
    elif backbone == "resnet18":
        return ResNet18BackboneExtractor(k, d, visual_tokens=vt)
    elif backbone == "raw":
        return RawProjectionExtractor(k, d, visual_tokens=vt)
    elif backbone == "vit":
        return ViTPatchExtractorWithResampler(
            patch_grid=cfg.get("vit_patch_grid", 4),
            embed_dim=d,
            num_out=vt,
        )
    else:
        raise ValueError(f"未知 visual_backbone: {backbone}")
