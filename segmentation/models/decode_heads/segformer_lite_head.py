"""
SegFormer-style lightweight decoder head for engineering baseline.

Input: 4-level features [f1, f2, f3, f4]
Output:
  - Main logits at stride-4
  - Auxiliary logits at stride-8 and stride-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Conv + BN + ReLU helper block."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SegFormerLiteHead(nn.Module):
    """Simple SegFormer-like multi-scale aggregation head."""

    def __init__(self, in_channels_list, num_classes, decode_channels=256, dropout=0.1):
        super().__init__()
        c1, c2, c3, c4 = in_channels_list

        self.proj1 = nn.Conv2d(c1, decode_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(c2, decode_channels, kernel_size=1)
        self.proj3 = nn.Conv2d(c3, decode_channels, kernel_size=1)
        self.proj4 = nn.Conv2d(c4, decode_channels, kernel_size=1)

        self.fuse = nn.Sequential(
            ConvBNReLU(decode_channels * 4, decode_channels, kernel_size=3),
            ConvBNReLU(decode_channels, decode_channels, kernel_size=3),
        )

        self.dropout = nn.Dropout2d(dropout)
        self.cls_head = nn.Conv2d(decode_channels, num_classes, kernel_size=1)

        self.aux_head2 = nn.Conv2d(decode_channels, num_classes, kernel_size=1)
        self.aux_head3 = nn.Conv2d(decode_channels, num_classes, kernel_size=1)

    def forward(self, features):
        """Return (main_logits, [aux2_logits, aux3_logits])."""
        f1, f2, f3, f4 = features

        p1 = self.proj1(f1)
        p2 = self.proj2(f2)
        p3 = self.proj3(f3)
        p4 = self.proj4(f4)

        target_size = p1.shape[2:]
        p2_up = F.interpolate(p2, size=target_size, mode="bilinear", align_corners=True)
        p3_up = F.interpolate(p3, size=target_size, mode="bilinear", align_corners=True)
        p4_up = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=True)

        fused = self.fuse(torch.cat([p1, p2_up, p3_up, p4_up], dim=1))
        fused = self.dropout(fused)
        main_logits = self.cls_head(fused)

        aux2_logits = self.aux_head2(p2)
        aux3_logits = self.aux_head3(p3)
        return main_logits, [aux2_logits, aux3_logits]

