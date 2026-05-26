"""
SAM3 ViTDet vision backbone adapter for RGB encoder.

This adapter only loads and uses SAM3 visual backbone (ViT + neck), without
the text encoder / detector heads, so it can be dropped into existing
RGB-T segmentation code paths that expect 4-level feature lists.
"""

from __future__ import annotations

import os
import os.path as osp
import sys
from typing import List

import torch
import torch.nn as nn

# Ensure repository root is importable.
sys.path.insert(0, osp.join(osp.dirname(__file__), "../../.."))


class SpatialBottleneckAdapter(nn.Module):
    """Lightweight spatial bottleneck adapter with SAM2-compatible name pattern."""

    def __init__(self, channels: int, bottleneck_dim: int = 32):
        super().__init__()
        hidden = max(8, int(bottleneck_dim))
        # Keep "prompt_learn" in module path to reuse existing optimizer grouping.
        self.prompt_learn = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.prompt_learn(x)


class SAM3ViTDetAdapter(nn.Module):
    """
    SAM3 visual backbone adapter.

    Output:
      list[Tensor]: 4-level feature pyramid, all 256 channels.
    """

    OUT_CHANNELS = [256, 256, 256, 256]

    def __init__(
        self,
        checkpoint: str,
        bottleneck_dim: int = 32,
        freeze_backbone: bool = True,
        compile_backbone: bool = False,
    ):
        super().__init__()

        from sam3.model_builder import _create_vision_backbone

        compile_mode = "default" if compile_backbone else None
        self.visual_backbone = _create_vision_backbone(
            compile_mode=compile_mode,
            enable_inst_interactivity=False,
        )
        self._load_visual_checkpoint(checkpoint)

        if freeze_backbone:
            for param in self.visual_backbone.parameters():
                param.requires_grad = False

        self.adapters = nn.ModuleList(
            [SpatialBottleneckAdapter(ch, bottleneck_dim=bottleneck_dim) for ch in self.OUT_CHANNELS]
        )

    def _load_visual_checkpoint(self, checkpoint: str) -> None:
        if not osp.isfile(checkpoint):
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")

        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, dict):
            raise RuntimeError(f"Unexpected SAM3 checkpoint format: {type(state)}")

        prefix = "detector.backbone.vision_backbone."
        visual_state = {
            k[len(prefix) :]: v
            for k, v in state.items()
            if isinstance(k, str) and k.startswith(prefix)
        }
        if not visual_state:
            raise RuntimeError(
                "No visual-backbone weights found in SAM3 checkpoint with "
                f"prefix '{prefix}'."
            )

        self.visual_backbone.load_state_dict(visual_state, strict=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.shape[-2:] != (1008, 1008):
            x = torch.nn.functional.interpolate(
                x, size=(1008, 1008), mode="bilinear", align_corners=False
            )
        # SAM3 ViTDet fused kernels expect frozen-backbone inference semantics.
        # We keep backbone strictly no-grad and only train lightweight adapters.
        with torch.no_grad():
            sam3_out, _, _, _ = self.visual_backbone.forward(x)
        return [adapter(feat) for adapter, feat in zip(self.adapters, sam3_out)]

    @property
    def out_channels(self):
        return self.OUT_CHANNELS
