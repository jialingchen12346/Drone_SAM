"""
MMSA baseline segmentor (engineering-oriented).

Architecture:
  SAM2 Hiera-L (frozen + adapter) ──┐
                                     ├─> MMSAFusion (x4) ─> SegFormerLiteHead ─> prediction
  ConvNeXt-Tiny (trainable)       ──┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation.models.backbones.aux_encoder import ConvNeXtTinyAux
from segmentation.models.backbones.sam2_hiera_adapter import SAM2HieraAdapter
from segmentation.models.decode_heads.segformer_lite_head import SegFormerLiteHead
from segmentation.models.fusion.mmsa_fusion import MMSAFusion


def dice_loss(pred, target, num_classes, ignore_index=255, smooth=1.0):
    """Per-class Dice loss."""
    mask = target != ignore_index
    target_clean = target.clone()
    target_clean[~mask] = 0

    pred_soft = F.softmax(pred, dim=1)
    one_hot = F.one_hot(target_clean, num_classes).permute(0, 3, 1, 2).float()
    mask_expanded = mask.unsqueeze(1).float()

    pred_soft = pred_soft * mask_expanded
    one_hot = one_hot * mask_expanded

    dims = (0, 2, 3)
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
    pixel_losses = F.cross_entropy(
        logits,
        target,
        ignore_index=ignore_index,
        reduction="none",
        weight=class_weight,
    ).contiguous().view(-1)

    target_flat = target.contiguous().view(-1)
    valid_mask = target_flat != ignore_index
    if valid_mask.sum() == 0:
        return logits.new_tensor(0.0)

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
    if hard_losses.numel() == 0:
        hard_losses = valid_losses[:min_kept]
    return hard_losses.mean()


class MMSABaselineSegmentor(nn.Module):
    """End-to-end multimodal segmentor with MMSA-style fusion."""

    def __init__(
        self,
        num_classes,
        sam2_checkpoint,
        sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        bottleneck_dim=32,
        decode_channels=256,
        aux_loss_weight=0.4,
        pretrained_aux=True,
        use_dice=False,
        dice_weight=1.0,
        use_ohem=False,
        ohem_thresh=0.7,
        ohem_min_kept=100000,
        fusion_num_heads=4,
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

        self.rgb_encoder = SAM2HieraAdapter(
            checkpoint=sam2_checkpoint,
            config=sam2_config,
            bottleneck_dim=bottleneck_dim,
            freeze_backbone=True,
        )
        self.aux_encoder = ConvNeXtTinyAux(pretrained=pretrained_aux)

        rgb_ch = tuple(self.rgb_encoder.out_channels)
        aux_ch = tuple(self.aux_encoder.out_channels)
        self.fusion = MMSAFusion(
            rgb_channels=rgb_ch,
            aux_channels=aux_ch,
            out_channels=rgb_ch,
            num_heads=fusion_num_heads,
        )

        self.decoder = SegFormerLiteHead(
            in_channels_list=list(self.rgb_encoder.out_channels),
            num_classes=num_classes,
            decode_channels=decode_channels,
        )

    def _compute_loss(self, logits, gt):
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
            ce = F.cross_entropy(logits, gt, ignore_index=255, weight=self.ce_class_weight)

        if self.use_dice:
            dl = dice_loss(logits, gt, self.num_classes, ignore_index=255)
            return ce + self.dice_weight * dl
        return ce

    def forward(self, rgb, aux, gt=None):
        h, w = rgb.shape[2], rgb.shape[3]

        rgb_features = self.rgb_encoder(rgb)
        aux_features = self.aux_encoder(aux)
        fused_features, fusion_weights = self.fusion(rgb_features, aux_features)

        output = {}
        if self.training and gt is not None:
            main_out, aux_outs = self.decoder(fused_features)
            main_out = F.interpolate(main_out, size=(h, w), mode="bilinear", align_corners=True)
            loss = self._compute_loss(main_out, gt)
            for aux_out in aux_outs:
                aux_up = F.interpolate(aux_out, size=(h, w), mode="bilinear", align_corners=True)
                loss = loss + self.aux_loss_weight * self._compute_loss(aux_up, gt)
            output["pred"] = main_out
            output["loss"] = loss
        else:
            main_out, _ = self.decoder(fused_features)
            main_out = F.interpolate(main_out, size=(h, w), mode="bilinear", align_corners=True)
            output["pred"] = main_out

        output["fusion_weights"] = fusion_weights
        return output

