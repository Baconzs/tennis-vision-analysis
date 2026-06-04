"""resnet_extractor.py — ResNet18 视觉特征提取器"""
import torch.nn as nn
import torchvision.models as tv_models
from .token_resampler import TokenResampler


class ResNet18BackboneExtractor(nn.Module):
    """ResNet18 骨干，经 TokenResampler 输出固定 visual_tokens 个 token。"""

    def __init__(self, tokens_per_scale, embed_dim, visual_tokens=16):
        super().__init__()
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        pool_size = int(tokens_per_scale ** 0.5)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(1),
            nn.Linear(512 * pool_size * pool_size, tokens_per_scale * embed_dim)
        )
        self.tokens_per_scale = tokens_per_scale
        self.embed_dim = embed_dim
        self.resampler = TokenResampler(embed_dim, num_out=visual_tokens)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.proj(feat).view(x.size(0), self.tokens_per_scale, self.embed_dim)
        return self.resampler(out)
