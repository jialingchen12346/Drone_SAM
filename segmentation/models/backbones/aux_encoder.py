"""
Auxiliary modality encoder based on ConvNeXt.

Uses timm to load a pretrained ConvNeXt and extracts 4-level features:
  f1: [B,  96, H/4,  W/4]   stride=4
  f2: [B, 192, H/8,  W/8]   stride=8
  f3: [B, 384, H/16, W/16]  stride=16
  f4: [B, 768, H/32, W/32]  stride=32
"""

import re

import torch
import torch.nn as nn
import timm


_CONVNEXT_MODEL_NAMES = {
    "tiny": "convnext_tiny.fb_in22k",
    "small": "convnext_small.fb_in22k",
}


class ConvNeXtAux(nn.Module):
    """ConvNeXt auxiliary encoder for thermal / LiDAR modality.

    Args:
        pretrained: whether to load ImageNet-22K pretrained weights from timm
        in_chans: number of input channels (3 for thermal pseudo-RGB or LiDAR)
        size: ConvNeXt variant. Currently supports "tiny" and "small".
        pretrained_path: optional local checkpoint path. Supports timm checkpoints and
            OpenMMLab ConvNeXt backbone checkpoints.
    """

    def __init__(self, pretrained=True, in_chans=3, size="tiny", pretrained_path=None):
        super().__init__()
        if size not in _CONVNEXT_MODEL_NAMES:
            raise ValueError(
                f"Unknown ConvNeXt aux encoder size: {size}. "
                f"Available: {sorted(_CONVNEXT_MODEL_NAMES)}"
            )
        self.size = size
        self.model_name = _CONVNEXT_MODEL_NAMES[size]
        self.model = timm.create_model(
            self.model_name,
            pretrained=pretrained and not pretrained_path,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=in_chans,
        )
        if pretrained_path:
            self._load_pretrained_path(pretrained_path)
        self._out_channels = list(self.model.feature_info.channels())

    def _load_pretrained_path(self, pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
        model_state = self.model.state_dict()
        converted = {}
        skipped = []

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("backbone."):
                key = key[len("backbone.") :]

            new_key = self._convert_openmmlab_key(key)
            if new_key is None:
                new_key = key

            if new_key in model_state and tuple(model_state[new_key].shape) == tuple(value.shape):
                converted[new_key] = value
            else:
                skipped.append(key)

        missing, unexpected = self.model.load_state_dict(converted, strict=False)
        print(
            f"[ConvNeXtAux] loaded local checkpoint: {pretrained_path} "
            f"matched={len(converted)} missing={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)}"
        )

    @staticmethod
    def _convert_openmmlab_key(key):
        if key.startswith("downsample_layers.0.0."):
            return key.replace("downsample_layers.0.0.", "stem_0.", 1)
        if key.startswith("downsample_layers.0.1."):
            return key.replace("downsample_layers.0.1.", "stem_1.", 1)

        match = re.match(r"downsample_layers\.([1-3])\.([01])\.(.+)", key)
        if match:
            stage_idx, layer_idx, rest = match.groups()
            return f"stages_{stage_idx}.downsample.{layer_idx}.{rest}"

        match = re.match(r"stages\.([0-3])\.(\d+)\.(.+)", key)
        if match:
            stage_idx, block_idx, rest = match.groups()
            rest = rest.replace("depthwise_conv.", "conv_dw.")
            rest = rest.replace("pointwise_conv1.", "mlp.fc1.")
            rest = rest.replace("pointwise_conv2.", "mlp.fc2.")
            return f"stages_{stage_idx}.blocks.{block_idx}.{rest}"

        return None

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
        return self._out_channels


class ConvNeXtTinyAux(ConvNeXtAux):
    """Backward-compatible alias for the historical ConvNeXt-Tiny encoder."""

    def __init__(self, pretrained=True, in_chans=3, pretrained_path=None):
        super().__init__(
            pretrained=pretrained,
            in_chans=in_chans,
            size="tiny",
            pretrained_path=pretrained_path,
        )
