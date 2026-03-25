"""
HGSOAD: Hierarchical-Guided Small Object Aware Decoder.

Components:
  1. SAGU (Scale-Aware Gating Unit): channel attention gate per level
  2. Hierarchical upsampling path: U-Net style with skip connections
  3. Auxiliary supervision heads at d2 and d3 (training only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAwareGatingUnit(nn.Module):
    """Channel attention gate combining avg-pool and max-pool cues."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate = self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        return x * gate


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """Upsample 2x -> concat skip -> DoubleConv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes don't match exactly
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class HGSOAD(nn.Module):
    """Hierarchical-Guided Small Object Aware Decoder.

    Args:
        in_channels_list: 4-element list [f1_ch, f2_ch, f3_ch, f4_ch]
        num_classes: number of segmentation classes
        decode_channels: internal decoder channel width (default 256)
        use_sagu: if False, skip SAGU channel attention (ablation A1/A2)
    """

    def __init__(self, in_channels_list, num_classes, decode_channels=256,
                 use_sagu=True):
        super().__init__()
        self.use_sagu = use_sagu
        c1, c2, c3, c4 = in_channels_list

        # 1x1 channel reduction to decode_channels
        self.reduce4 = nn.Conv2d(c4, decode_channels, 1)
        self.reduce3 = nn.Conv2d(c3, decode_channels, 1)
        self.reduce2 = nn.Conv2d(c2, decode_channels, 1)
        self.reduce1 = nn.Conv2d(c1, decode_channels, 1)

        # Scale-Aware Gating Units (only instantiated when use_sagu=True)
        if use_sagu:
            self.sagu4 = ScaleAwareGatingUnit(decode_channels)
            self.sagu3 = ScaleAwareGatingUnit(decode_channels)
            self.sagu2 = ScaleAwareGatingUnit(decode_channels)
            self.sagu1 = ScaleAwareGatingUnit(decode_channels)

        # U-Net upsampling path
        self.up3 = UpBlock(decode_channels, decode_channels, decode_channels)
        self.up2 = UpBlock(decode_channels, decode_channels, decode_channels)
        self.up1 = UpBlock(decode_channels, decode_channels, decode_channels)

        # Main classification head
        self.cls_head = nn.Conv2d(decode_channels, num_classes, 1)

        # Auxiliary supervision heads (active during training only)
        self.aux_head2 = nn.Conv2d(decode_channels, num_classes, 1)
        self.aux_head3 = nn.Conv2d(decode_channels, num_classes, 1)

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, f4] from CACAF fusion
        Returns:
            training:  (main_out, [aux_out2, aux_out3])
            inference: main_out
        """
        f1, f2, f3, f4 = features

        # Channel reduction + gating
        f4 = self.reduce4(f4)
        f3 = self.reduce3(f3)
        f2 = self.reduce2(f2)
        f1 = self.reduce1(f1)
        if self.use_sagu:
            f4 = self.sagu4(f4)
            f3 = self.sagu3(f3)
            f2 = self.sagu2(f2)
            f1 = self.sagu1(f1)

        # Hierarchical upsampling
        d3 = self.up3(f4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)

        main_out = self.cls_head(d1)

        if self.training:
            aux_out2 = self.aux_head2(d2)
            aux_out3 = self.aux_head3(d3)
            return main_out, [aux_out2, aux_out3]
        return main_out
