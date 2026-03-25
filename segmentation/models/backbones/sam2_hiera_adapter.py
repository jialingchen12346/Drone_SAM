"""
SAM2 Hiera-L backbone with Bottleneck Adapter (PEFT).

特征输出 (512×512 输入):
  f1: [B, 144, 128, 128]  stride=4
  f2: [B, 288,  64,  64]  stride=8
  f3: [B, 576,  32,  32]  stride=16
  f4: [B,1152,  16,  16]  stride=32
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

import sam2  # noqa: F401 — triggers hydra initialize_config_module
from sam2.build_sam import build_sam2


class BottleneckAdapter(nn.Module):
    """在每个 Hiera block 前插入轻量 bottleneck，用于参数高效微调。

    参考 SAM2-UNeXT 的 Adapter 设计。
    可训练参数: 2 × dim × bottleneck_dim (约 2M for dim=144~1152, d=32)
    """

    def __init__(self, block, bottleneck_dim=32):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, dim),
            nn.GELU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.prompt_learn.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        prompt = self.prompt_learn(x)
        return self.block(x + prompt)


class SAM2HieraAdapter(nn.Module):
    """SAM2 Hiera-L backbone，冻结主干权重，仅 Adapter 参与训练。

    Args:
        checkpoint (str): SAM2.1 检查点路径
        config (str): Hydra config 名称，默认 configs/sam2.1/sam2.1_hiera_l.yaml
        bottleneck_dim (int): Adapter bottleneck 维度，默认 32
        freeze_backbone (bool): 是否冻结 Hiera 原始权重，默认 True
    """

    # SAM2 Hiera-L 各层输出通道数
    OUT_CHANNELS = [144, 288, 576, 1152]

    def __init__(
        self,
        checkpoint,
        config='configs/sam2.1/sam2.1_hiera_l.yaml',
        bottleneck_dim=32,
        freeze_backbone=True,
    ):
        super().__init__()

        # 加载 SAM2，只保留 image_encoder.trunk
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam2(config, checkpoint, device=device)
        self.trunk = model.image_encoder.trunk

        # 冻结原始 Hiera 权重
        if freeze_backbone:
            for param in self.trunk.parameters():
                param.requires_grad = False

        # 将每个 block 替换为 Adapter 包装版本
        adapted_blocks = nn.Sequential(*[
            BottleneckAdapter(blk, bottleneck_dim)
            for blk in self.trunk.blocks
        ])
        self.trunk.blocks = adapted_blocks

    def forward(self, x):
        """
        Args:
            x: RGB 图像 [B, 3, H, W]
        Returns:
            list[Tensor]: [f1, f2, f3, f4]，4 级多尺度特征
        """
        features = self.trunk(x)
        # trunk 输出已经是 4 级特征列表，直接返回
        return features

    @property
    def out_channels(self):
        return self.OUT_CHANNELS
