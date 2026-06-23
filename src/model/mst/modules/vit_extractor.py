"""vit_extractor.py — 轻量 ViT patch embedding 特征提取器"""
import torch
import torch.nn as nn


class ViTPatchExtractor(nn.Module):
    """
    轻量 patch embedding：将输入图像切成 patch_grid×patch_grid 个 16×16 patch，
    每个 patch 展平后过两层 FFN 投影到 embed_dim，直接输出 token。
    无内部 Transformer，特征提取交给主网络 MSTFormer。

    输入: (B, 3, H, W)
    输出: (B, num_patches, embed_dim)   num_patches = patch_grid^2
    """

    def __init__(self, patch_grid, embed_dim, vit_depth=2, num_heads=4):
        # vit_depth / num_heads 保留参数签名兼容性，不再使用
        super().__init__()
        self.patch_grid = patch_grid
        patch_px = 16
        in_dim = 3 * patch_px * patch_px  # 768

        self.pool = nn.AdaptiveAvgPool2d((patch_grid * patch_px, patch_grid * patch_px))
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        g = self.patch_grid
        patch_px = 16

        x = self.pool(x)                                          # (B, 3, g*16, g*16)
        x = x.view(B, 3, g, patch_px, g, patch_px)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()            # (B, g, g, 3, 16, 16)
        x = x.view(B, g * g, 3 * patch_px * patch_px)            # (B, num_patches, 768)
        return self.ffn(x)                                        # (B, num_patches, embed_dim)
