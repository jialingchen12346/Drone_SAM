"""
CACafSegmentor: full segmentation model assembling all components.

Architecture:
  SAM2 Hiera-L (frozen + Adapter)  ──┐
                                      ├──> CACAF (x4 levels) ──> HGSOAD ──> prediction
  ConvNeXt-Tiny (trainable)        ──┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation.models.backbones.sam2_hiera_adapter import SAM2HieraAdapter
from segmentation.models.backbones.aux_encoder import ConvNeXtTinyAux
from segmentation.models.fusion.cacaf import CACAF
from segmentation.models.decode_heads.hgsoad_head import HGSOAD


class SimpleFusion(nn.Module):
    """Ablation baseline fusion: channel-align both modalities → concat → 1×1 reduce.

    Produces the same output shape as CACAF but with no cross-attention or
    quality-aware weighting. Returns dummy [0.5, 0.5] fusion weights for API
    compatibility with CACAF.
    """

    def __init__(self, rgb_channels, aux_channels, out_channels):
        super().__init__()
        self.aligns = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(rgb_ch + aux_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for rgb_ch, aux_ch, out_ch in zip(rgb_channels, aux_channels, out_channels)
        ])

    def forward(self, rgb_features, aux_features):
        fused = [
            self.aligns[i](torch.cat([rgb_features[i], aux_features[i]], dim=1))
            for i in range(4)
        ]
        # Dummy weights [B, 2] = [0.5, 0.5] for API compatibility
        B = rgb_features[0].shape[0]
        dummy = torch.full((B, 2), 0.5, device=rgb_features[0].device)
        return fused, [dummy] * 4


def dice_loss(pred, target, num_classes, ignore_index=255, smooth=1.0):
    """Per-class Dice loss.

    Args:
        pred: [B, C, H, W] logits
        target: [B, H, W] integer labels
        num_classes: number of classes
        ignore_index: label to ignore
        smooth: smoothing factor to avoid division by zero
    Returns:
        scalar mean Dice loss (1 - Dice)
    """
    mask = target != ignore_index
    target_clean = target.clone()
    target_clean[~mask] = 0

    pred_soft = F.softmax(pred, dim=1)  # [B, C, H, W]
    one_hot = F.one_hot(target_clean, num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
    mask_expanded = mask.unsqueeze(1).float()  # [B, 1, H, W]

    pred_soft = pred_soft * mask_expanded
    one_hot = one_hot * mask_expanded

    dims = (0, 2, 3)  # sum over batch, H, W
    intersection = (pred_soft * one_hot).sum(dim=dims)
    cardinality = pred_soft.sum(dim=dims) + one_hot.sum(dim=dims)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return (1.0 - dice).mean()


def ohem_cross_entropy_loss(
    logits,
    target,
    ignore_index=255,
    thresh=0.7,
    min_kept=100000,
    class_weight=None,
):
    """Online Hard Example Mining cross-entropy."""
    # Per-pixel CE
    pixel_losses = F.cross_entropy(
        logits,
        target,
        ignore_index=ignore_index,
        reduction="none",
        weight=class_weight,
    ).contiguous().view(-1)

    # Valid mask
    target_flat = target.contiguous().view(-1)
    valid_mask = target_flat != ignore_index
    if valid_mask.sum() == 0:
        return logits.new_tensor(0.0)

    # Confidence of the target class at each pixel
    with torch.no_grad():
        prob = F.softmax(logits, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == ignore_index] = 0
        prob = prob.gather(1, tmp_target.unsqueeze(1)).contiguous().view(-1)
        valid_prob, sort_idx = prob[valid_mask].sort()

    valid_losses = pixel_losses[valid_mask][sort_idx]
    min_kept = max(1, min(min_kept, valid_prob.numel()))
    threshold = torch.maximum(valid_prob[min_kept - 1], logits.new_tensor(thresh))
    hard_losses = valid_losses[valid_prob < threshold]

    # Fallback when no pixel is below threshold
    if hard_losses.numel() == 0:
        hard_losses = valid_losses[:min_kept]

    return hard_losses.mean()


class CACafSegmentor(nn.Module):
    """End-to-end multimodal segmentation model.

    Args:
        num_classes: number of segmentation classes
        sam2_checkpoint: path to SAM2 Hiera-L checkpoint
        sam2_config: hydra config name for SAM2
        bottleneck_dim: adapter bottleneck dimension
        decode_channels: HGSOAD internal channel width
        num_heads: attention heads in CACAF cross-modal enhancement
        aux_loss_weight: weight for auxiliary supervision losses
        pretrained_aux: whether to load ImageNet pretrained ConvNeXt
        use_dice: whether to add Dice loss alongside CE
        dice_weight: weight for Dice loss component
        use_ohem: whether to replace CE with OHEM CE
        ohem_thresh: OHEM confidence threshold
        ohem_min_kept: minimum number of hard pixels kept in OHEM
    """

    def __init__(
        self,
        num_classes,
        sam2_checkpoint,
        sam2_config='configs/sam2.1/sam2.1_hiera_l.yaml',
        bottleneck_dim=32,
        decode_channels=256,
        num_heads=4,
        aux_loss_weight=0.4,
        pretrained_aux=True,
        use_dice=False,
        dice_weight=1.0,
        use_ohem=False,
        ohem_thresh=0.7,
        ohem_min_kept=100000,
        use_cacaf=True,
        use_sagu=True,
        ce_class_weight=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.aux_loss_weight = aux_loss_weight
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        self.use_ohem = use_ohem
        self.ohem_thresh = ohem_thresh
        self.ohem_min_kept = ohem_min_kept
        if ce_class_weight is None:
            self.register_buffer("ce_class_weight", None, persistent=False)
        else:
            self.register_buffer(
                "ce_class_weight",
                torch.tensor(ce_class_weight, dtype=torch.float32),
                persistent=False,
            )

        # Encoders
        self.rgb_encoder = SAM2HieraAdapter(
            checkpoint=sam2_checkpoint,
            config=sam2_config,
            bottleneck_dim=bottleneck_dim,
            freeze_backbone=True,
        )
        self.aux_encoder = ConvNeXtTinyAux(pretrained=pretrained_aux)

        # Fusion: full CACAF or simple concat (ablation)
        rgb_ch = tuple(self.rgb_encoder.out_channels)
        aux_ch = tuple(self.aux_encoder.out_channels)
        if use_cacaf:
            self.fusion = CACAF(
                rgb_channels=rgb_ch,
                aux_channels=aux_ch,
                out_channels=rgb_ch,
                num_heads=num_heads,
            )
        else:
            self.fusion = SimpleFusion(
                rgb_channels=rgb_ch,
                aux_channels=aux_ch,
                out_channels=rgb_ch,
            )

        # Decoder: HGSOAD with or without SAGU (ablation)
        self.decoder = HGSOAD(
            in_channels_list=list(self.rgb_encoder.out_channels),
            num_classes=num_classes,
            decode_channels=decode_channels,
            use_sagu=use_sagu,
        )

    def _compute_loss(self, logits, gt):
        """Compute CE (+ optional Dice) loss for a single output."""
        logits = logits.contiguous()
        gt = gt.contiguous()
        if self.use_ohem:
            ce = ohem_cross_entropy_loss(
                logits,
                gt,
                ignore_index=255,
                thresh=self.ohem_thresh,
                min_kept=self.ohem_min_kept,
                class_weight=self.ce_class_weight,
            )
        else:
            ce = F.cross_entropy(
                logits,
                gt,
                ignore_index=255,
                weight=self.ce_class_weight,
            )
        if self.use_dice:
            dl = dice_loss(logits, gt, self.num_classes, ignore_index=255)
            return ce + self.dice_weight * dl
        return ce

    def forward(self, rgb, aux, gt=None):
        """
        Args:
            rgb: [B, 3, H, W] RGB image
            aux: [B, 3, H, W] auxiliary modality (thermal / LiDAR)
            gt:  [B, H, W] ground truth labels (optional, for loss computation)
        Returns:
            dict with keys:
              'pred': [B, num_classes, H, W] main prediction (upsampled to input size)
              'loss': scalar total loss (only when gt is provided and training)
              'fusion_weights': list of 4 [B, 2] tensors (for visualization)
        """
        H, W = rgb.shape[2], rgb.shape[3]

        # Encode
        rgb_features = self.rgb_encoder(rgb)
        aux_features = self.aux_encoder(aux)

        # Fuse
        fused_features, fusion_weights = self.fusion(rgb_features, aux_features)

        # Decode
        output = {}
        if self.training and gt is not None:
            main_out, aux_outs = self.decoder(fused_features)
            # Upsample all outputs to input resolution
            main_out = F.interpolate(main_out, size=(H, W), mode='bilinear', align_corners=True)
            loss = self._compute_loss(main_out, gt)
            for aux_out in aux_outs:
                aux_up = F.interpolate(aux_out, size=(H, W), mode='bilinear', align_corners=True)
                loss = loss + self.aux_loss_weight * self._compute_loss(aux_up, gt)
            output['pred'] = main_out
            output['loss'] = loss
        else:
            main_out = self.decoder(fused_features)
            main_out = F.interpolate(main_out, size=(H, W), mode='bilinear', align_corners=True)
            output['pred'] = main_out

        output['fusion_weights'] = fusion_weights
        return output
