"""token_resampler.py — Token 重采样模块（Perceiver 风格 cross-attention）"""
import torch
import torch.nn as nn


class TokenResampler(nn.Module):
    """
    将任意数量的输入 token 重采样到固定 num_out 个 token。
    用 num_out 个可学习 query 对输入做 cross-attention（Perceiver 风格）。

    输入: (B, N_in, D)
    输出: (B, num_out, D)
    """

    def __init__(self, embed_dim, num_out=16, num_heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, num_out, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads,
                                                batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        q = self.queries.expand(x.shape[0], -1, -1)
        out, _ = self.cross_attn(q, x, x)
        return self.norm(out + q)
