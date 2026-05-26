"""
Lightweight early cross-modal interaction modules.

These sit between encoder outputs and AGF fusion, providing the "small step
forward" from pure prediction-level correction toward representation-level
interaction, without modifying either backbone.

Modules:
  GFFMBlock / GFFM          — per-scale bidirectional cross-attention with
                               learnable zero-init gates and adaptive pooling
  MidLevelCorrectionBlock   — feature-level disagreement-gated correction
                               (stride 16/32 only)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation.models.fusion.mmsa_fusion import ConvBNReLU


class GFFMBlock(nn.Module):
    """Single-scale bidirectional cross-attention with learnable gate.

    Uses adaptive spatial pooling to cap token count before cross-attention,
    then upsamples the delta back to the original resolution.
    Gamma is initialised to zero so training starts from identity.
    """

    def __init__(
        self,
        rgb_ch: int,
        aux_ch: int,
        num_heads: int = 4,
        max_tokens: int = 1024,
        init_gamma: float = 0.0,
    ):
        super().__init__()
        mid_ch = min(rgb_ch, aux_ch)
        self.max_tokens = max_tokens

        self.align_rgb = ConvBNReLU(rgb_ch, mid_ch, 1)
        self.align_aux = ConvBNReLU(aux_ch, mid_ch, 1)

        self.norm_rgb = nn.LayerNorm(mid_ch)
        self.norm_aux = nn.LayerNorm(mid_ch)
        self.rgb_cross = nn.MultiheadAttention(mid_ch, num_heads, batch_first=True)
        self.aux_cross = nn.MultiheadAttention(mid_ch, num_heads, batch_first=True)

        self.proj_rgb_out = nn.Conv2d(mid_ch, rgb_ch, 1, bias=False)
        self.proj_aux_out = nn.Conv2d(mid_ch, aux_ch, 1, bias=False)

        self.gamma_rgb = nn.Parameter(torch.tensor(init_gamma))
        self.gamma_aux = nn.Parameter(torch.tensor(init_gamma))

    def _pool_tokens(self, x):
        b, c, h, w = x.shape
        tokens = h * w
        if tokens <= self.max_tokens:
            return x, 1
        stride = int(math.ceil(math.sqrt(tokens / float(self.max_tokens))))
        return F.avg_pool2d(x, kernel_size=stride, stride=stride, ceil_mode=True), stride

    def forward(self, rgb_feat: torch.Tensor, aux_feat: torch.Tensor):
        b, _, h, w = rgb_feat.shape

        rgb = self.align_rgb(rgb_feat)
        aux = self.align_aux(aux_feat)

        # Adaptive pool to bound memory
        rgb_ds, stride_r = self._pool_tokens(rgb)
        aux_ds, stride_a = self._pool_tokens(aux)
        stride = max(stride_r, stride_a)

        if stride > 1:
            rgb_ds = F.avg_pool2d(rgb, kernel_size=stride, stride=stride, ceil_mode=True)
            aux_ds = F.avg_pool2d(aux, kernel_size=stride, stride=stride, ceil_mode=True)

        h_ds, w_ds = rgb_ds.shape[2], rgb_ds.shape[3]

        rgb_tok = rgb_ds.flatten(2).transpose(1, 2)
        aux_tok = aux_ds.flatten(2).transpose(1, 2)

        rgb_delta, _ = self.rgb_cross(
            self.norm_rgb(rgb_tok), self.norm_aux(aux_tok), self.norm_aux(aux_tok)
        )
        aux_delta, _ = self.aux_cross(
            self.norm_aux(aux_tok), self.norm_rgb(rgb_tok), self.norm_rgb(rgb_tok)
        )

        rgb_delta = rgb_delta.transpose(1, 2).reshape(b, -1, h_ds, w_ds)
        aux_delta = aux_delta.transpose(1, 2).reshape(b, -1, h_ds, w_ds)

        if stride > 1:
            rgb_delta = F.interpolate(
                rgb_delta, size=(h, w), mode="bilinear", align_corners=True
            )
            aux_delta = F.interpolate(
                aux_delta, size=(h, w), mode="bilinear", align_corners=True
            )

        rgb_out = rgb_feat + self.gamma_rgb * self.proj_rgb_out(rgb_delta)
        aux_out = aux_feat + self.gamma_aux * self.proj_aux_out(aux_delta)
        return rgb_out, aux_out


class GFFM(nn.Module):
    """Multi-scale GFFM — one block per encoder level, with tighter token caps
    on higher-resolution features."""

    def __init__(
        self,
        rgb_channels=(144, 288, 576, 1152),
        aux_channels=(96, 192, 384, 768),
        num_heads: int = 4,
        max_tokens_per_level=(1024, 1024, 1024, 1024),
        init_gamma: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                GFFMBlock(
                    r_ch, a_ch,
                    num_heads=num_heads,
                    max_tokens=mt,
                    init_gamma=init_gamma,
                )
                for r_ch, a_ch, mt in zip(rgb_channels, aux_channels, max_tokens_per_level)
            ]
        )

    def forward(self, rgb_features, aux_features):
        out_rgb, out_aux = [], []
        for block, rf, af in zip(self.blocks, rgb_features, aux_features):
            r, a = block(rf, af)
            out_rgb.append(r)
            out_aux.append(a)
        return out_rgb, out_aux


class MidLevelCorrectionBlock(nn.Module):
    """Feature-level disagreement-gated correction at a single scale.

    Computes cosine similarity between aligned RGB and Thermal features, then
    uses the disagreement signal (1 - sim) as a spatial gate to correct the
    weaker modality with the stronger one.  Only Thermal is corrected; RGB
    passes through unchanged.

    Designed for stride-16 and stride-32 features where semantics are strong
    enough for meaningful cosine similarity.
    """

    def __init__(self, rgb_ch: int, aux_ch: int, init_gamma: float = 0.0):
        super().__init__()
        self.align_aux = ConvBNReLU(aux_ch, rgb_ch, 1)
        self.correction = nn.Sequential(
            nn.Conv2d(rgb_ch * 2, rgb_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_ch, aux_ch, 1),
        )
        self.gamma = nn.Parameter(torch.tensor(init_gamma))

    def forward(self, rgb_feat: torch.Tensor, aux_feat: torch.Tensor):
        aux_aligned = self.align_aux(aux_feat)
        rgb_n = F.normalize(rgb_feat, dim=1)
        aux_n = F.normalize(aux_aligned, dim=1)
        sim = (rgb_n * aux_n).sum(dim=1, keepdim=True)
        gate = 1.0 - sim

        correction = self.correction(torch.cat([rgb_feat, aux_aligned], dim=1))
        aux_out = aux_feat + self.gamma * gate * correction
        return rgb_feat, aux_out


class MidLevelCorrection(nn.Module):
    """Mid-level correction applied to the deepest two scales (stride 16/32)."""

    def __init__(
        self,
        rgb_channels=(144, 288, 576, 1152),
        aux_channels=(96, 192, 384, 768),
        init_gamma: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for idx, (r_ch, a_ch) in enumerate(zip(rgb_channels, aux_channels)):
            if idx >= 2:   # stride 16 and 32
                self.blocks.append(
                    MidLevelCorrectionBlock(r_ch, a_ch, init_gamma=init_gamma)
                )
            else:
                self.blocks.append(nn.Identity())

    def forward(self, rgb_features, aux_features):
        out_rgb, out_aux = [], []
        for idx, (rf, af) in enumerate(zip(rgb_features, aux_features)):
            if idx >= 2:
                r, a = self.blocks[idx](rf, af)
                out_rgb.append(r)
                out_aux.append(a)
            else:
                out_rgb.append(rf)
                out_aux.append(af)
        return out_rgb, out_aux
