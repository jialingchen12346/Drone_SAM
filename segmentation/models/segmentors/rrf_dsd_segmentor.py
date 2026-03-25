"""
RRFDSDSegmentor: SAM2 + ConvNeXt + RRF + DSDHead.

Architecture:
  SAM2 Hiera-L (frozen + Adapter)  ──┐
                                      ├──> RRF (x4 levels) ──> DSDHead ──> prediction
  ConvNeXt-Tiny (trainable)        ──┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation.models.backbones.sam2_hiera_adapter import SAM2HieraAdapter
from segmentation.models.backbones.aux_encoder import ConvNeXtTinyAux
from segmentation.models.fusion.rrf import RRF
from segmentation.models.decode_heads.dsd_head import DSDHead


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


def build_boundary_target(target, ignore_index=255):
    """Build 1-pixel boundary supervision map from segmentation labels."""
    valid = target != ignore_index
    boundary = torch.zeros_like(target, dtype=torch.float32)

    diff_h = (
        (target[:, :, 1:] != target[:, :, :-1])
        & valid[:, :, 1:]
        & valid[:, :, :-1]
    )
    diff_v = (
        (target[:, 1:, :] != target[:, :-1, :])
        & valid[:, 1:, :]
        & valid[:, :-1, :]
    )

    boundary[:, :, 1:] = torch.maximum(boundary[:, :, 1:], diff_h.float())
    boundary[:, :, :-1] = torch.maximum(boundary[:, :, :-1], diff_h.float())
    boundary[:, 1:, :] = torch.maximum(boundary[:, 1:, :], diff_v.float())
    boundary[:, :-1, :] = torch.maximum(boundary[:, :-1, :], diff_v.float())
    return boundary, valid.float()


def boundary_bce_loss(logits, target, ignore_index=255):
    """BCE loss for boundary prediction with ignore-index masking."""
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"boundary logits must be [B,1,H,W], got {tuple(logits.shape)}")

    if logits.shape[2:] != target.shape[1:]:
        logits = F.interpolate(
            logits, size=target.shape[1:], mode='bilinear', align_corners=True
        )

    logits = logits[:, 0]
    boundary, valid = build_boundary_target(target, ignore_index=ignore_index)
    loss = F.binary_cross_entropy_with_logits(logits, boundary, reduction="none")

    denom = valid.sum().clamp_min(1.0)
    return (loss * valid).sum() / denom


class RRFDSDSegmentor(nn.Module):
    """End-to-end multimodal segmentor with RRF fusion + DSD decoder."""

    def __init__(
        self,
        num_classes,
        sam2_checkpoint,
        sam2_config='configs/sam2.1/sam2.1_hiera_l.yaml',
        bottleneck_dim=32,
        decode_channels=256,
        aux_loss_weight=0.4,
        detail_aux_loss_weight=0.2,
        pretrained_aux=True,
        use_dice=False,
        dice_weight=1.0,
        use_ohem=False,
        ohem_thresh=0.7,
        ohem_min_kept=100000,
        use_thin_structure_refiner=False,
        thin_refiner_scale=0.15,
        use_boundary_refiner=False,
        boundary_refiner_scale=0.5,
        boundary_aux_loss_weight=0.0,
        use_rare_class_residual=False,
        rare_class_indices=None,
        rare_class_scale=1.0,
        ce_class_weight=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.aux_loss_weight = aux_loss_weight
        self.detail_aux_loss_weight = detail_aux_loss_weight
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        self.use_ohem = use_ohem
        self.ohem_thresh = ohem_thresh
        self.ohem_min_kept = ohem_min_kept
        self.boundary_aux_loss_weight = boundary_aux_loss_weight
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
        self.fusion = RRF(
            rgb_channels=rgb_ch,
            aux_channels=aux_ch,
            out_channels=rgb_ch,
        )

        self.decoder = DSDHead(
            in_channels_list=list(self.rgb_encoder.out_channels),
            num_classes=num_classes,
            decode_channels=decode_channels,
            use_thin_structure_refiner=use_thin_structure_refiner,
            thin_refiner_scale=thin_refiner_scale,
            use_boundary_refiner=use_boundary_refiner,
            boundary_refiner_scale=boundary_refiner_scale,
            use_rare_class_residual=use_rare_class_residual,
            rare_class_indices=rare_class_indices,
            rare_class_scale=rare_class_scale,
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
            rgb: [B, 3, H, W]
            aux: [B, 3, H, W]
            gt:  [B, H, W] optional
        Returns:
            dict with keys:
              - pred: [B, num_classes, H, W]
              - loss: scalar (training with gt only)
              - fusion_weights: list of 4 [B, 2]
              - reliability_maps: list of 4 [B, 2, h, w]
        """
        h, w = rgb.shape[2], rgb.shape[3]

        rgb_features = self.rgb_encoder(rgb)
        aux_features = self.aux_encoder(aux)

        fused_features, pooled_weights, reliability_maps = self.fusion(
            rgb_features, aux_features
        )

        output = {}
        if self.training and gt is not None:
            decoder_out = self.decoder(fused_features)
            decoder_extra = {}
            if isinstance(decoder_out, tuple):
                if len(decoder_out) == 3:
                    main_out, aux_outs, decoder_extra = decoder_out
                elif len(decoder_out) == 2:
                    main_out, aux_outs = decoder_out
                else:
                    raise ValueError(f"Unexpected decoder outputs: len={len(decoder_out)}")
            else:
                main_out, aux_outs = decoder_out, []

            main_out = F.interpolate(main_out, size=(h, w), mode='bilinear', align_corners=True)
            loss = self._compute_loss(main_out, gt)
            for aux_idx, aux_out in enumerate(aux_outs):
                aux_up = F.interpolate(aux_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_weight = self.aux_loss_weight if aux_idx < 2 else self.detail_aux_loss_weight
                loss = loss + aux_weight * self._compute_loss(aux_up, gt)

            if (
                self.boundary_aux_loss_weight > 0.0
                and isinstance(decoder_extra, dict)
                and "boundary_logit" in decoder_extra
            ):
                b_loss = boundary_bce_loss(decoder_extra["boundary_logit"], gt, ignore_index=255)
                loss = loss + self.boundary_aux_loss_weight * b_loss
                output["loss_boundary"] = b_loss.detach()

            output["pred"] = main_out
            output["loss"] = loss
        else:
            main_out = self.decoder(fused_features)
            main_out = F.interpolate(main_out, size=(h, w), mode='bilinear', align_corners=True)
            output["pred"] = main_out

        output["fusion_weights"] = pooled_weights
        output["reliability_maps"] = reliability_maps
        return output
