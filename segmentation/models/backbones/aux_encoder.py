"""
Auxiliary modality encoder based on ConvNeXt-Tiny.

Uses timm to load a pretrained ConvNeXt-Tiny and extracts 4-level features:
  f1: [B,  96, H/4,  W/4]   stride=4
  f2: [B, 192, H/8,  W/8]   stride=8
  f3: [B, 384, H/16, W/16]  stride=16
  f4: [B, 768, H/32, W/32]  stride=32
"""

import torch.nn as nn
import timm


class ConvNeXtTinyAux(nn.Module):
    """ConvNeXt-Tiny as an auxiliary encoder for thermal / LiDAR modality.

    Args:
        pretrained: whether to load ImageNet-22K pretrained weights
        in_chans: number of input channels (3 for thermal pseudo-RGB or LiDAR)
    """

    OUT_CHANNELS = [96, 192, 384, 768]

    def __init__(self, pretrained=True, in_chans=3):
        super().__init__()
        self.model = timm.create_model(
            'convnext_tiny.fb_in22k',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=in_chans,
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] auxiliary modality input
        Returns:
            list of 4 feature maps at strides [4, 8, 16, 32]
        """
        return self.model(x)

    @property
    def out_channels(self):
        return self.OUT_CHANNELS
