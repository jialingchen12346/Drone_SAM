"""
CACAF: Condition-Aware Cross-modal Adaptive Fusion.

For each scale level, CACAF performs:
  1. Channel alignment (1x1 conv to a common dim)
  2. Modal Quality Assessment (MQA): GAP -> MLP -> softmax weights
  3. Cross-Modal Enhancement: bidirectional cross-attention (skipped at f1 due to memory)
  4. Adaptive Fusion: w_rgb * rgb_enhanced + w_aux * aux_enhanced
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalQualityAssessment(nn.Module):
    """Lightweight modal quality assessment via GAP + MLP."""

    def __init__(self, in_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, 1),
        )

    def forward(self, x):
        """x: [B, C, H, W] -> quality_score: [B, 1]"""
        return self.mlp(self.gap(x).flatten(1))


class CrossModalEnhancement(nn.Module):
    """Bidirectional cross-attention between two modalities.

    Uses nn.MultiheadAttention with spatial tokens (HW).
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn_rgb2aux = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_aux2rgb = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_rgb = nn.LayerNorm(dim)
        self.norm_aux = nn.LayerNorm(dim)

    def forward(self, rgb_feat, aux_feat):
        """
        Args:
            rgb_feat, aux_feat: [B, C, H, W]
        Returns:
            rgb_enhanced, aux_enhanced: [B, C, H, W]
        """
        B, C, H, W = rgb_feat.shape
        rgb_seq = rgb_feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        aux_seq = aux_feat.flatten(2).permute(0, 2, 1)

        # Pre-norm: LayerNorm BEFORE attention (stable in FP16, avoids overflow)
        rgb_normed = self.norm_rgb(rgb_seq)
        aux_normed = self.norm_aux(aux_seq)

        # Bidirectional cross-attention with residual
        rgb_enhanced = rgb_seq + self.attn_aux2rgb(rgb_normed, aux_normed, aux_normed)[0]
        aux_enhanced = aux_seq + self.attn_rgb2aux(aux_normed, rgb_normed, rgb_normed)[0]

        rgb_enhanced = rgb_enhanced.permute(0, 2, 1).view(B, C, H, W)
        aux_enhanced = aux_enhanced.permute(0, 2, 1).view(B, C, H, W)
        return rgb_enhanced, aux_enhanced


class CACAfBlock(nn.Module):
    """Single-scale CACAF block.

    Args:
        rgb_channels: input channels from RGB encoder at this scale
        aux_channels: input channels from aux encoder at this scale
        out_channels: unified output channels after alignment
        num_heads: number of attention heads for cross-modal enhancement
        use_cross_attn: if False, skip cross-attention (for high-res f1 level)
    """

    def __init__(self, rgb_channels, aux_channels, out_channels,
                 num_heads=4, use_cross_attn=True):
        super().__init__()
        # Channel alignment
        self.align_rgb = (nn.Conv2d(rgb_channels, out_channels, 1)
                          if rgb_channels != out_channels else nn.Identity())
        self.align_aux = (nn.Conv2d(aux_channels, out_channels, 1)
                          if aux_channels != out_channels else nn.Identity())

        # Quality assessment
        self.mqa_rgb = ModalQualityAssessment(out_channels)
        self.mqa_aux = ModalQualityAssessment(out_channels)

        # Cross-modal enhancement (disabled for high-res levels)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_enhance = CrossModalEnhancement(out_channels, num_heads)

    def forward(self, rgb_feat, aux_feat):
        """
        Returns:
            fused: [B, out_channels, H, W]
            weights: [B, 2] detached (w_rgb, w_aux) for visualization
        """
        rgb_feat = self.align_rgb(rgb_feat)
        aux_feat = self.align_aux(aux_feat)

        # MQA -> adaptive weights
        q_rgb = self.mqa_rgb(rgb_feat)  # [B, 1]
        q_aux = self.mqa_aux(aux_feat)  # [B, 1]
        weights = F.softmax(torch.cat([q_rgb, q_aux], dim=1), dim=1)  # [B, 2]
        w_rgb = weights[:, 0:1, None, None]  # [B, 1, 1, 1]
        w_aux = weights[:, 1:2, None, None]

        # Cross-modal enhancement (or identity for f1)
        if self.use_cross_attn:
            rgb_enhanced, aux_enhanced = self.cross_enhance(rgb_feat, aux_feat)
        else:
            rgb_enhanced, aux_enhanced = rgb_feat, aux_feat

        # Adaptive fusion
        fused = w_rgb * rgb_enhanced + w_aux * aux_enhanced
        return fused, weights.detach()


class CACAF(nn.Module):
    """Multi-scale Condition-Aware Cross-modal Adaptive Fusion.

    Fuses 4-level features from RGB encoder (SAM2 Hiera) and auxiliary encoder
    (ConvNeXt-Tiny). Cross-attention is disabled at the f1 level (128x128) to
    avoid excessive memory usage.

    Args:
        rgb_channels: list of 4 channel counts from SAM2 Hiera
        aux_channels: list of 4 channel counts from aux encoder
        out_channels: list of 4 output channel counts (usually = rgb_channels)
        num_heads: attention heads per level
    """

    def __init__(
        self,
        rgb_channels=(144, 288, 576, 1152),
        aux_channels=(96, 192, 384, 768),
        out_channels=(144, 288, 576, 1152),
        num_heads=4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(4):
            use_cross_attn = (i > 0)  # f1 (i=0) skips cross-attention
            self.blocks.append(
                CACAfBlock(
                    rgb_channels[i], aux_channels[i], out_channels[i],
                    num_heads=num_heads, use_cross_attn=use_cross_attn,
                )
            )

    def forward(self, rgb_features, aux_features):
        """
        Args:
            rgb_features: list of 4 tensors from SAM2 Hiera
            aux_features: list of 4 tensors from ConvNeXt-Tiny
        Returns:
            fused_features: list of 4 fused tensors
            all_weights: list of 4 [B, 2] tensors (for visualization)
        """
        fused_features = []
        all_weights = []
        for i, block in enumerate(self.blocks):
            fused, w = block(rgb_features[i], aux_features[i])
            fused_features.append(fused)
            all_weights.append(w)
        return fused_features, all_weights
