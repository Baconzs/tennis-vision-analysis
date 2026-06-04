"""yolo_extractor.py — YOLO backbone P3/P4/P5 多尺度特征提取器"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ultralytics import YOLO
from .token_resampler import TokenResampler

# yolo11x-pose: P3=layer16(384ch), P4=layer19(768ch), P5=layer22(768ch)
_YOLO_CONFIGS = {
    "yolo11x": {
        "channels": {3: 384, 4: 768, 5: 768},
        "hook_layers": {3: 16, 4: 19, 5: 22},
    },
    "yolo11n": {
        "channels": {3: 64, 4: 128, 5: 256},
        "hook_layers": {3: 16, 4: 19, 5: 22},
    },
    "yolo11n_det": {
        "channels": {3: 64, 4: 128, 5: 256},
        "hook_layers": {3: 16, 4: 19, 5: 22},
    },
    "yolov8n": {
        "channels": {3: 64, 4: 128, 5: 256},
        "hook_layers": {3: 15, 4: 18, 5: 21},
    },
}

# 默认兼容旧配置
_YOLO11X_CHANNELS = _YOLO_CONFIGS["yolo11x"]["channels"]
_YOLO11X_HOOK_LAYERS = _YOLO_CONFIGS["yolo11x"]["hook_layers"]


def _detect_yolo_variant(weights_path: str) -> str:
    p = weights_path.lower()
    if ("yolo11n" in p or "11n" in p) and "pose" not in p:
        return "yolo11n_det"
    if "yolo11n" in p or "11n" in p:
        return "yolo11n"
    if "yolov8n" in p or "v8n" in p:
        return "yolov8n"
    return "yolo11x"


def _run_layers(layers, save_set, x, y):
    """按 YOLO _predict_once 逻辑逐层运行，更新 y 列表并返回最后输出。"""
    for m in layers:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in save_set else None)
    return x


class Yolo11BackboneExtractor(nn.Module):
    """从 YOLO backbone 截取 P3/P4/P5 特征，通过跨尺度注意力融合后，
    经 TokenResampler 输出固定 visual_tokens 个 token。

    支持 yolo11x-pose 和 yolov8n-pose，通过 weights_path 自动识别。
    shared_backbone: 若传入已有实例则共享权重（节省显存），否则自行加载。
    """

    def __init__(self, weights_path, scale_levels, tokens_per_scale, embed_dim,
                 visual_tokens=16, shared_backbone=None, unfreeze_backbone=False):
        super().__init__()
        self.scale_levels = scale_levels
        self.tokens_per_scale = tokens_per_scale
        self.embed_dim = embed_dim
        self.unfreeze_backbone = unfreeze_backbone

        variant = _detect_yolo_variant(weights_path)
        cfg = _YOLO_CONFIGS[variant]
        self._channels = cfg["channels"]
        self._hook_layers = cfg["hook_layers"]

        if shared_backbone is not None:
            object.__setattr__(self, 'backbone', shared_backbone)
            self._owns_backbone = False
        else:
            yolo = YOLO(weights_path)
            self.backbone = yolo.model
            if unfreeze_backbone:
                self.backbone.train()
                for param in self.backbone.parameters():
                    param.requires_grad = True
            else:
                self.backbone.eval()
                for param in self.backbone.parameters():
                    param.requires_grad = False
            self._owns_backbone = True

        # 按 P3/P4/P5 层索引把 backbone 切成三段，用于分段 checkpoint
        lvls = sorted(scale_levels)
        all_layers = list(self.backbone.model)
        p3_idx = self._hook_layers[lvls[0]]
        p4_idx = self._hook_layers[lvls[1]]
        p5_idx = self._hook_layers[lvls[2]]
        self._seg_layers = [
            all_layers[: p3_idx + 1],
            all_layers[p3_idx + 1 : p4_idx + 1],
            all_layers[p4_idx + 1 : p5_idx + 1],
        ]
        self._save_set = self.backbone.save  # YOLO 内部需要缓存的层索引集合
        self._p_idxs = (p3_idx, p4_idx, p5_idx)
        self._lvls = lvls

        self.proj = nn.ModuleDict({
            str(lvl): nn.Linear(self._channels[lvl], embed_dim)
            for lvl in scale_levels
        })

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.resampler = TokenResampler(embed_dim, num_out=visual_tokens)

    def _forward_backbone(self, x):
        """分三段运行 backbone，解冻时每段用 checkpoint 节省显存。"""
        y = []  # 与 YOLO _predict_once 的 y 列表对应

        def seg0(x):
            return _run_layers(self._seg_layers[0], self._save_set, x, y)

        def seg1(x):
            return _run_layers(self._seg_layers[1], self._save_set, x, y)

        def seg2(x):
            return _run_layers(self._seg_layers[2], self._save_set, x, y)

        if self.unfreeze_backbone:
            f0 = checkpoint(seg0, x, use_reentrant=False)
            f1 = checkpoint(seg1, f0, use_reentrant=False)
            f2 = checkpoint(seg2, f1, use_reentrant=False)
        else:
            f0 = seg0(x)
            f1 = seg1(f0)
            f2 = seg2(f1)

        p3_idx, p4_idx, p5_idx = self._p_idxs
        return {
            self._lvls[0]: y[p3_idx],
            self._lvls[1]: y[p4_idx],
            self._lvls[2]: y[p5_idx],
        }

    def forward(self, x):
        # 只在训练模式下开梯度，eval 时始终 no_grad
        ctx = torch.enable_grad() if (self.unfreeze_backbone and self.training) else torch.no_grad()
        with ctx:
            feats = self._forward_backbone(x)

        scale_tokens = []
        for lvl in self.scale_levels:
            feat = feats[lvl]
            pool_size = int(self.tokens_per_scale ** 0.5)
            feat = F.adaptive_avg_pool2d(feat, pool_size)
            feat = feat.flatten(2).transpose(1, 2)
            feat = self.proj[str(lvl)](feat)
            scale_tokens.append(feat)

        query = scale_tokens[-1]
        kv = torch.cat(scale_tokens[:-1], dim=1)
        fused, _ = self.cross_attn(query, kv, kv)
        fused = self.cross_norm(fused + query)          # (B, tokens_per_scale, D)

        all_tokens = torch.cat(scale_tokens[:-1] + [fused], dim=1)
        return self.resampler(all_tokens)               # (B, visual_tokens, D)
