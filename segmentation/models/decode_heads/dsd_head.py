"""
DSDHead: Detail Semantic Decoupling decoder.

Design:
  - Semantic branch: f4 -> f3 -> f2 (context aggregation)
  - Detail branch: f1 + upsampled f2 (detail refinement)
  - Edge prior branch at stride-4
  - Small-object enhancement at stride-4
  - Guided merge with residual semantic gate (avoid over-suppressing detail)
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


class DetailRefineBlock(nn.Module):
    """Lightweight detail refinement block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=3),
            ConvBNReLU(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, x):
        return self.block(x)


class SemanticUpBlock(nn.Module):
    """Upsample semantic feature and fuse with skip."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNReLU(in_channels + skip_channels, out_channels, kernel_size=3),
            ConvBNReLU(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return self.fuse(torch.cat([x, skip], dim=1))


class GuidedMergeBlock(nn.Module):
    """Semantic-guided residual detail modulation and merge."""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            ConvBNReLU(channels * 4, channels, kernel_size=3),
            ConvBNReLU(channels, channels, kernel_size=3),
        )

    def forward(self, detail_feat, semantic_feat, edge_feat, small_obj_feat):
        gate = self.gate(torch.cat([detail_feat, semantic_feat], dim=1))
        detail_feat = detail_feat * (1.0 + gate)
        merged = torch.cat([detail_feat, semantic_feat, edge_feat, small_obj_feat], dim=1)
        return self.fuse(merged)


class EdgePriorBlock(nn.Module):
    """Extract high-resolution edge prior from f1 feature."""

    def __init__(self, channels):
        super().__init__()
        self.edge_mask = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        mask = torch.sigmoid(self.edge_mask(x))
        return x * (1.0 + mask)


class SmallObjectEnhancer(nn.Module):
    """Fuse f2-up and semantic s4 feature to boost small-object cues."""

    def __init__(self, channels):
        super().__init__()
        self.enhance = nn.Sequential(
            ConvBNReLU(channels * 2, channels, kernel_size=1),
            ConvBNReLU(channels, channels, kernel_size=3),
            ConvBNReLU(channels, channels, kernel_size=3),
        )

    def forward(self, f2_up, semantic_s4):
        return self.enhance(torch.cat([f2_up, semantic_s4], dim=1))


class ThinStructureCompensator(nn.Module):
    """Compensate thin-structure details with multi-dilation depthwise refinement."""

    def __init__(self, channels):
        super().__init__()
        self.reduce = ConvBNReLU(channels * 3, channels, kernel_size=1)
        self.dw_d2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dw_d3 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            ConvBNReLU(channels * 2, channels, kernel_size=1),
            ConvBNReLU(channels, channels, kernel_size=3),
        )

    def forward(self, detail_feat, edge_feat, semantic_feat):
        base = self.reduce(torch.cat([detail_feat, edge_feat, semantic_feat], dim=1))
        d2 = self.dw_d2(base)
        d3 = self.dw_d3(base)
        return self.fuse(torch.cat([d2, d3], dim=1))


class DSDHead(nn.Module):
    """Detail Semantic Decoupling decoder head."""

    def __init__(
        self,
        in_channels_list,
        num_classes,
        decode_channels=256,
        use_thin_structure_refiner=False,
        thin_refiner_scale=0.15,
        use_boundary_refiner=False,
        boundary_refiner_scale=0.5,
        use_rare_class_residual=False,
        rare_class_indices=None,
        rare_class_scale=1.0,
    ):
        super().__init__()
        c1, c2, c3, c4 = in_channels_list
        self.use_thin_structure_refiner = use_thin_structure_refiner
        self.use_boundary_refiner = use_boundary_refiner
        self.use_rare_class_residual = use_rare_class_residual

        self.reduce1 = nn.Conv2d(c1, decode_channels, kernel_size=1)
        self.reduce2 = nn.Conv2d(c2, decode_channels, kernel_size=1)
        self.reduce3 = nn.Conv2d(c3, decode_channels, kernel_size=1)
        self.reduce4 = nn.Conv2d(c4, decode_channels, kernel_size=1)

        # Semantic branch: f4 -> f3 -> f2
        self.semantic_up3 = SemanticUpBlock(decode_channels, decode_channels, decode_channels)
        self.semantic_up2 = SemanticUpBlock(decode_channels, decode_channels, decode_channels)

        # Detail branch: f1 + upsampled f2
        self.detail_refine = DetailRefineBlock(decode_channels * 2, decode_channels)
        self.edge_prior = EdgePriorBlock(decode_channels)
        self.small_obj_enhancer = SmallObjectEnhancer(decode_channels)
        if self.use_thin_structure_refiner:
            self.thin_refiner = ThinStructureCompensator(decode_channels)
            self.thin_refiner_scale = nn.Parameter(torch.tensor(thin_refiner_scale))
        else:
            self.thin_refiner = None
            self.register_parameter("thin_refiner_scale", None)

        if self.use_boundary_refiner:
            self.boundary_head = nn.Conv2d(decode_channels, 1, kernel_size=1)
            self.boundary_refine = nn.Sequential(
                ConvBNReLU(decode_channels * 2, decode_channels, kernel_size=3),
                nn.Conv2d(decode_channels, num_classes, kernel_size=1),
            )
            self.boundary_refiner_scale = nn.Parameter(torch.tensor(boundary_refiner_scale))
        else:
            self.boundary_head = None
            self.boundary_refine = None
            self.register_parameter("boundary_refiner_scale", None)

        if self.use_rare_class_residual:
            if rare_class_indices is None:
                rare_class_indices = [1, 3, 4, 13]
            self.rare_class_indices = [int(x) for x in rare_class_indices]
            self.register_buffer(
                "rare_class_indices_tensor",
                torch.tensor(self.rare_class_indices, dtype=torch.long),
                persistent=False,
            )
            self.rare_class_head = nn.Conv2d(
                decode_channels,
                len(self.rare_class_indices),
                kernel_size=1,
            )
            self.rare_class_scale = nn.Parameter(torch.tensor(rare_class_scale))
        else:
            self.rare_class_indices = []
            self.register_buffer(
                "rare_class_indices_tensor",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            self.rare_class_head = None
            self.register_parameter("rare_class_scale", None)

        # Guided merge at stride-4
        self.guided_merge = GuidedMergeBlock(decode_channels)

        self.cls_head = nn.Conv2d(decode_channels, num_classes, kernel_size=1)
        self.aux_head2 = nn.Conv2d(decode_channels, num_classes, kernel_size=1)  # stride-8
        self.aux_head3 = nn.Conv2d(decode_channels, num_classes, kernel_size=1)  # stride-16
        self.aux_head_detail = nn.Conv2d(
            decode_channels, num_classes, kernel_size=1
        )  # stride-4

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, f4]
        Returns:
            training:  (main_out, [aux_out2, aux_out3, aux_out_detail])
            inference: main_out
        """
        f1, f2, f3, f4 = features
        f1 = self.reduce1(f1)
        f2 = self.reduce2(f2)
        f3 = self.reduce3(f3)
        f4 = self.reduce4(f4)

        semantic_s16 = self.semantic_up3(f4, f3)
        semantic_s8 = self.semantic_up2(semantic_s16, f2)

        f2_up = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        detail = self.detail_refine(torch.cat([f1, f2_up], dim=1))
        edge_feat = self.edge_prior(f1)

        semantic_s4 = F.interpolate(
            semantic_s8, size=f1.shape[2:], mode='bilinear', align_corners=True
        )
        small_obj_feat = self.small_obj_enhancer(f2_up, semantic_s4)
        merged = self.guided_merge(detail, semantic_s4, edge_feat, small_obj_feat)
        if self.thin_refiner is not None:
            thin_feat = self.thin_refiner(detail, edge_feat, semantic_s4)
            merged = merged + self.thin_refiner_scale * thin_feat

        extra = {}
        if self.boundary_head is not None:
            boundary_logit = self.boundary_head(merged)
            boundary_prob = torch.sigmoid(boundary_logit)
            refine_logits = self.boundary_refine(torch.cat([merged, edge_feat], dim=1))
            merged_logits_refine = self.boundary_refiner_scale * boundary_prob * refine_logits
            extra["boundary_logit"] = boundary_logit
        else:
            merged_logits_refine = 0.0

        main_out = self.cls_head(merged)
        if self.boundary_head is not None:
            main_out = main_out + merged_logits_refine

        if self.rare_class_head is not None:
            rare_logits = self.rare_class_head(merged)
            rare_full = torch.zeros_like(main_out)
            rare_full.index_copy_(1, self.rare_class_indices_tensor, rare_logits)
            main_out = main_out + self.rare_class_scale * rare_full
            extra["rare_logits"] = rare_logits

        if self.training:
            aux_out2 = self.aux_head2(semantic_s8)
            aux_out3 = self.aux_head3(semantic_s16)
            aux_out_detail = self.aux_head_detail(small_obj_feat)
            return main_out, [aux_out2, aux_out3, aux_out_detail], extra
        return main_out
