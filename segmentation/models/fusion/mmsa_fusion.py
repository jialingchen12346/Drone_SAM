"""
MMSA-style fusion for engineering baseline.

Design goals:
  1) Per-level channel alignment for RGB / auxiliary features
  2) Sample-wise modality quality weighting (dynamic fusion)
  3) Lightweight bidirectional cross-modal interaction

Notes:
  - To keep memory predictable, cross-attention uses adaptive token downsampling.
  - High-resolution level f1 can skip interaction, matching practical trade-offs.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Conv + BN + ReLU helper block."""

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


class CrossModalInteractionLite(nn.Module):
    """Bidirectional cross-modal token interaction with adaptive downsampling."""

    def __init__(self, channels, num_heads=4, max_tokens=1024):
        super().__init__()
        self.max_tokens = max_tokens

        self.norm_rgb = nn.LayerNorm(channels)
        self.norm_aux = nn.LayerNorm(channels)
        self.attn_rgb = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.attn_aux = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.ffn_rgb = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.ffn_aux = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def _adaptive_pool(self, x, target_tokens):
        """Reduce spatial token count by avg pooling when needed."""
        b, c, h, w = x.shape
        tokens = h * w
        if tokens <= target_tokens:
            return x, 1
        stride = int(math.ceil(math.sqrt(tokens / float(target_tokens))))
        pooled = F.avg_pool2d(x, kernel_size=stride, stride=stride, ceil_mode=True)
        return pooled, stride

    def forward(self, rgb_feat, aux_feat):
        b, c, h, w = rgb_feat.shape

        # Use identical stride for both streams to keep token grids aligned.
        _, stride_rgb = self._adaptive_pool(rgb_feat, self.max_tokens)
        _, stride_aux = self._adaptive_pool(aux_feat, self.max_tokens)
        stride = max(stride_rgb, stride_aux)

        if stride > 1:
            rgb_ds = F.avg_pool2d(rgb_feat, kernel_size=stride, stride=stride, ceil_mode=True)
            aux_ds = F.avg_pool2d(aux_feat, kernel_size=stride, stride=stride, ceil_mode=True)
        else:
            rgb_ds = rgb_feat
            aux_ds = aux_feat

        h_ds, w_ds = rgb_ds.shape[2], rgb_ds.shape[3]

        rgb_tokens = rgb_ds.flatten(2).transpose(1, 2)  # [B, N, C]
        aux_tokens = aux_ds.flatten(2).transpose(1, 2)  # [B, N, C]

        rgb_norm = self.norm_rgb(rgb_tokens)
        aux_norm = self.norm_aux(aux_tokens)

        rgb_delta, _ = self.attn_rgb(
            query=rgb_norm, key=aux_norm, value=aux_norm, need_weights=False
        )
        aux_delta, _ = self.attn_aux(
            query=aux_norm, key=rgb_norm, value=rgb_norm, need_weights=False
        )

        rgb_tokens = rgb_tokens + rgb_delta + self.ffn_rgb(rgb_tokens)
        aux_tokens = aux_tokens + aux_delta + self.ffn_aux(aux_tokens)

        rgb_delta_map = rgb_tokens.transpose(1, 2).reshape(b, c, h_ds, w_ds)
        aux_delta_map = aux_tokens.transpose(1, 2).reshape(b, c, h_ds, w_ds)

        if stride > 1:
            rgb_delta_map = F.interpolate(
                rgb_delta_map, size=(h, w), mode="bilinear", align_corners=True
            )
            aux_delta_map = F.interpolate(
                aux_delta_map, size=(h, w), mode="bilinear", align_corners=True
            )

        # Residual enhancement at original resolution.
        rgb_enhanced = rgb_feat + rgb_delta_map
        aux_enhanced = aux_feat + aux_delta_map
        return rgb_enhanced, aux_enhanced


class ModalityQualityWeight(nn.Module):
    """Predict per-sample RGB/Aux quality weights from global descriptors."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 32)
        self.rgb_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.aux_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, rgb_feat, aux_feat):
        rgb_vec = rgb_feat.mean(dim=(2, 3))
        aux_vec = aux_feat.mean(dim=(2, 3))
        rgb_score = self.rgb_mlp(rgb_vec)
        aux_score = self.aux_mlp(aux_vec)
        weights = torch.softmax(torch.cat([rgb_score, aux_score], dim=1), dim=1)  # [B, 2]
        return weights


class AgreementGuidedGate(nn.Module):
    """Predict a spatial RGB/Thermal fusion gate from features + agreement."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, rgb_feat, aux_feat, agreement_map):
        gate_in = torch.cat([rgb_feat, aux_feat, agreement_map], dim=1)
        return torch.sigmoid(self.net(gate_in))


class MMSAFusionBlock(nn.Module):
    """Single-scale fusion block."""

    def __init__(
        self,
        rgb_channels,
        aux_channels,
        out_channels,
        num_heads=4,
        max_tokens=1024,
        enable_interaction=True,
        use_agreement_gate=False,
    ):
        super().__init__()
        self.enable_interaction = enable_interaction
        self.use_agreement_gate = use_agreement_gate

        self.align_rgb = ConvBNReLU(rgb_channels, out_channels, kernel_size=1)
        self.align_aux = ConvBNReLU(aux_channels, out_channels, kernel_size=1)
        self.pre_refine = ConvBNReLU(out_channels * 2, out_channels * 2, kernel_size=3)

        if enable_interaction:
            self.interaction = CrossModalInteractionLite(
                channels=out_channels, num_heads=num_heads, max_tokens=max_tokens
            )
        else:
            self.interaction = None

        self.quality = None if use_agreement_gate else ModalityQualityWeight(out_channels)
        self.agreement_gate = AgreementGuidedGate(out_channels) if use_agreement_gate else None
        self.post_refine = nn.Sequential(
            ConvBNReLU(out_channels, out_channels, kernel_size=3),
            ConvBNReLU(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, rgb_feat, aux_feat, agreement_map=None):
        rgb = self.align_rgb(rgb_feat)
        aux = self.align_aux(aux_feat)

        joint = self.pre_refine(torch.cat([rgb, aux], dim=1))
        rgb, aux = torch.chunk(joint, 2, dim=1)

        if self.interaction is not None:
            rgb, aux = self.interaction(rgb, aux)

        if self.agreement_gate is not None and agreement_map is not None:
            agreement_resized = F.interpolate(
                agreement_map, size=rgb.shape[2:], mode="bilinear", align_corners=True
            )
            rgb_w = self.agreement_gate(rgb, aux, agreement_resized)
            aux_w = 1.0 - rgb_w
            pooled_rgb = rgb_w.mean(dim=(2, 3))
            pooled_aux = aux_w.mean(dim=(2, 3))
            weights = torch.cat([pooled_rgb, pooled_aux], dim=1)
        else:
            if self.quality is None:
                raise RuntimeError("quality branch is unavailable when use_agreement_gate=True")
            weights = self.quality(rgb, aux)  # [B, 2]
            rgb_w = weights[:, 0:1].unsqueeze(-1).unsqueeze(-1)
            aux_w = weights[:, 1:2].unsqueeze(-1).unsqueeze(-1)

        fused = rgb_w * rgb + aux_w * aux
        fused = self.post_refine(fused)
        return fused, weights.detach()


class NaiveFusionBlock(nn.Module):
    """Channel-align both modalities and fuse them with a fixed 0.5/0.5 average."""

    def __init__(self, rgb_channels, aux_channels, out_channels):
        super().__init__()
        self.align_rgb = ConvBNReLU(rgb_channels, out_channels, kernel_size=1)
        self.align_aux = ConvBNReLU(aux_channels, out_channels, kernel_size=1)

    def forward(self, rgb_feat, aux_feat, agreement_map=None):
        del agreement_map
        rgb = self.align_rgb(rgb_feat)
        aux = self.align_aux(aux_feat)
        fused = 0.5 * (rgb + aux)
        weights = rgb.new_full((rgb.shape[0], 2), 0.5)
        return fused, weights


class MMSAFusion(nn.Module):
    """Multi-scale MMSA-style fusion."""

    def __init__(
        self,
        rgb_channels=(144, 288, 576, 1152),
        aux_channels=(96, 192, 384, 768),
        out_channels=(144, 288, 576, 1152),
        num_heads=4,
        max_tokens_per_level=(1024, 1024, 1024, 1024),
        use_agreement_gate=False,
        fusion_mode="mmsa",
    ):
        super().__init__()
        if fusion_mode not in {"mmsa", "naive"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.blocks = nn.ModuleList()
        for idx, (rgb_ch, aux_ch, out_ch, max_tokens) in enumerate(
            zip(rgb_channels, aux_channels, out_channels, max_tokens_per_level)
        ):
            if fusion_mode == "naive":
                self.blocks.append(
                    NaiveFusionBlock(
                        rgb_channels=rgb_ch,
                        aux_channels=aux_ch,
                        out_channels=out_ch,
                    )
                )
            else:
                # Skip interaction on f1 (highest resolution) to control cost.
                enable_interaction = idx > 0
                self.blocks.append(
                    MMSAFusionBlock(
                        rgb_channels=rgb_ch,
                        aux_channels=aux_ch,
                        out_channels=out_ch,
                        num_heads=num_heads,
                        max_tokens=max_tokens,
                        enable_interaction=enable_interaction,
                        use_agreement_gate=use_agreement_gate,
                    )
                )

    def forward(self, rgb_features, aux_features, agreement_map=None):
        fused_features = []
        pooled_weights = []
        for idx, block in enumerate(self.blocks):
            fused, weights = block(rgb_features[idx], aux_features[idx], agreement_map=agreement_map)
            fused_features.append(fused)
            pooled_weights.append(weights)
        return fused_features, pooled_weights
