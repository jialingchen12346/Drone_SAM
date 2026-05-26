"""
RRF: Region Reliability Fusion.

For each feature level:
  1. Channel alignment for RGB / auxiliary features
  2. Lightweight local interaction on concatenated features
  3. Spatial reliability estimation (2-channel logits)
  4. Position-wise softmax fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """1x1 or 3x3 conv block with BN + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LocalInteractionBlock(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 residual interaction."""

    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x + residual


class ReliabilityEstimator(nn.Module):
    """Predict spatial reliability logits for 2 modalities."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class RRFBlock(nn.Module):
    """Single-scale Region Reliability Fusion block."""

    def __init__(self, rgb_channels, aux_channels, out_channels):
        super().__init__()
        self.align_rgb = ConvBNReLU(rgb_channels, out_channels, kernel_size=1)
        self.align_aux = ConvBNReLU(aux_channels, out_channels, kernel_size=1)

        joint_channels = out_channels * 2
        self.local_interaction = LocalInteractionBlock(joint_channels)
        hidden_channels = max(out_channels // 2, 16)
        self.reliability = ReliabilityEstimator(joint_channels, hidden_channels)
        self.detail_boost = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip_scale = nn.Parameter(torch.tensor(0.2))
        self.detail_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, rgb_feat, aux_feat):
        if aux_feat.shape[2:] != rgb_feat.shape[2:]:
            aux_feat = F.interpolate(
                aux_feat, size=rgb_feat.shape[2:], mode="bilinear", align_corners=True
            )
        rgb_feat = self.align_rgb(rgb_feat)
        aux_feat = self.align_aux(aux_feat)

        joint_feat = torch.cat([rgb_feat, aux_feat], dim=1)
        joint_feat = self.local_interaction(joint_feat)
        rgb_enhanced, aux_enhanced = torch.chunk(joint_feat, 2, dim=1)

        reliability_logits = self.reliability(joint_feat)  # [B, 2, H, W]
        reliability_map = F.softmax(reliability_logits, dim=1)

        fused = (
            reliability_map[:, 0:1] * rgb_enhanced
            + reliability_map[:, 1:2] * aux_enhanced
        )
        base_skip = 0.5 * (rgb_feat + aux_feat)
        detail_residual = self.detail_boost(torch.abs(rgb_enhanced - aux_enhanced))
        fused = fused + self.skip_scale * base_skip + self.detail_scale * detail_residual

        pooled_weights = reliability_map.mean(dim=(2, 3))  # [B, 2]
        return fused, pooled_weights.detach(), reliability_map


class RRF(nn.Module):
    """Multi-scale Region Reliability Fusion."""

    def __init__(
        self,
        rgb_channels=(144, 288, 576, 1152),
        aux_channels=(96, 192, 384, 768),
        out_channels=(144, 288, 576, 1152),
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            RRFBlock(rgb_ch, aux_ch, out_ch)
            for rgb_ch, aux_ch, out_ch in zip(rgb_channels, aux_channels, out_channels)
        ])

    def forward(self, rgb_features, aux_features):
        fused_features = []
        pooled_weights = []
        reliability_maps = []
        for idx, block in enumerate(self.blocks):
            fused, pooled, rel_map = block(rgb_features[idx], aux_features[idx])
            fused_features.append(fused)
            pooled_weights.append(pooled)
            reliability_maps.append(rel_map)
        return fused_features, pooled_weights, reliability_maps
