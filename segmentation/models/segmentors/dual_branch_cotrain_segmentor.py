"""
Dual-branch co-training segmentor without feature fusion.

Architecture:
  RGB      -> SAM2 Hiera-L      -> SegFormerLiteHead -> pred_rgb
  Thermal  -> ConvNeXt-Tiny     -> SegFormerLiteHead -> pred_thm
                                           |
                                   late logit ensemble -> pred
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation.models.backbones.aux_encoder import ConvNeXtTinyAux
from segmentation.models.backbones.sam2_hiera_adapter import SAM2HieraAdapter
from segmentation.models.decode_heads.segformer_lite_head import SegFormerLiteHead
from segmentation.models.segmentors.mmsa_baseline_segmentor import (
    dice_loss,
    ohem_cross_entropy_loss,
    weighted_cross_entropy_loss,
    weighted_dice_loss,
)


class DualBranchCoTrainSegmentor(nn.Module):
    """No-fusion RGB/Thermal segmentor with late decision aggregation."""

    def __init__(
        self,
        num_classes,
        sam2_checkpoint,
        sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        bottleneck_dim=32,
        decode_channels=256,
        aux_loss_weight=0.4,
        branch_loss_weight=0.2,
        pretrained_aux=True,
        use_dice=False,
        dice_weight=1.0,
        use_ohem=False,
        ohem_thresh=0.7,
        ohem_min_kept=100000,
        ce_class_weight=None,
        ensemble_mode="avg",
        agreement_mode="prob",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.aux_loss_weight = aux_loss_weight
        self.branch_loss_weight = branch_loss_weight
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        self.use_ohem = use_ohem
        self.ohem_thresh = ohem_thresh
        self.ohem_min_kept = ohem_min_kept
        self.ensemble_mode = ensemble_mode
        self.agreement_mode = agreement_mode

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

        self.rgb_head = SegFormerLiteHead(
            in_channels_list=list(self.rgb_encoder.out_channels),
            num_classes=num_classes,
            decode_channels=decode_channels,
        )
        self.thm_head = SegFormerLiteHead(
            in_channels_list=list(self.aux_encoder.out_channels),
            num_classes=num_classes,
            decode_channels=decode_channels,
        )

    def _compute_loss(self, logits, gt, pixel_weight=None):
        logits = logits.contiguous()
        gt = gt.contiguous()
        if pixel_weight is None and self.use_ohem:
            ce = ohem_cross_entropy_loss(
                logits,
                gt,
                ignore_index=255,
                thresh=self.ohem_thresh,
                min_kept=self.ohem_min_kept,
                class_weight=self.ce_class_weight,
            )
        elif pixel_weight is not None:
            ce = weighted_cross_entropy_loss(
                logits,
                gt,
                pixel_weight=pixel_weight,
                ignore_index=255,
                class_weight=self.ce_class_weight,
            )
        else:
            ce = F.cross_entropy(logits, gt, ignore_index=255, weight=self.ce_class_weight)

        if self.use_dice:
            if pixel_weight is None:
                dl = dice_loss(logits, gt, self.num_classes, ignore_index=255)
            else:
                dl = weighted_dice_loss(
                    logits,
                    gt,
                    pixel_weight=pixel_weight,
                    num_classes=self.num_classes,
                    ignore_index=255,
                )
            return ce + self.dice_weight * dl
        return ce

    def _compute_agreement_map(self, pred_rgb, pred_thm):
        if self.agreement_mode == "argmax":
            return (pred_rgb.argmax(dim=1) == pred_thm.argmax(dim=1)).float().unsqueeze(1)
        if self.agreement_mode == "prob":
            prob_rgb = F.softmax(pred_rgb, dim=1)
            prob_thm = F.softmax(pred_thm, dim=1)
            return (prob_rgb * prob_thm).sum(dim=1, keepdim=True)
        raise ValueError(f"Unsupported agreement_mode: {self.agreement_mode}")

    def _ensemble_logits(self, pred_rgb, pred_thm):
        if self.ensemble_mode == "avg":
            weight_rgb = torch.full_like(pred_rgb[:, :1], 0.5)
            weight_thm = torch.full_like(pred_thm[:, :1], 0.5)
        elif self.ensemble_mode == "confidence":
            conf_rgb = torch.softmax(pred_rgb, dim=1).amax(dim=1, keepdim=True)
            conf_thm = torch.softmax(pred_thm, dim=1).amax(dim=1, keepdim=True)
            denom = (conf_rgb + conf_thm).clamp_min(1e-6)
            weight_rgb = conf_rgb / denom
            weight_thm = conf_thm / denom
        else:
            raise ValueError(f"Unsupported ensemble_mode: {self.ensemble_mode}")

        pred = weight_rgb * pred_rgb + weight_thm * pred_thm
        return pred, weight_rgb, weight_thm

    def _forward_branch(self, head, features, size):
        main_out, aux_outs = head(features)
        main_out = F.interpolate(main_out, size=size, mode="bilinear", align_corners=True)
        aux_up = [
            F.interpolate(aux_out, size=size, mode="bilinear", align_corners=True)
            for aux_out in aux_outs
        ]
        return main_out, aux_up

    def forward(self, rgb, aux, gt=None, pixel_weight=None):
        h, w = rgb.shape[2], rgb.shape[3]

        rgb_features = self.rgb_encoder(rgb)
        aux_features = self.aux_encoder(aux)

        pred_rgb, aux_rgb = self._forward_branch(self.rgb_head, rgb_features, size=(h, w))
        pred_thm, aux_thm = self._forward_branch(self.thm_head, aux_features, size=(h, w))
        pred, weight_rgb, weight_thm = self._ensemble_logits(pred_rgb, pred_thm)
        agreement_map = self._compute_agreement_map(pred_rgb, pred_thm)

        output = {
            "pred": pred,
            "pred_rgb": pred_rgb,
            "pred_thm": pred_thm,
            "agreement_map": agreement_map,
            "ensemble_weights": torch.cat([weight_rgb, weight_thm], dim=1).detach(),
        }

        if self.training and gt is not None:
            loss = self._compute_loss(pred, gt, pixel_weight=pixel_weight)

            for aux_out in aux_rgb:
                loss = loss + self.aux_loss_weight * self._compute_loss(
                    aux_out, gt, pixel_weight=pixel_weight
                )
            for aux_out in aux_thm:
                loss = loss + self.aux_loss_weight * self._compute_loss(
                    aux_out, gt, pixel_weight=pixel_weight
                )

            loss = loss + self.branch_loss_weight * self._compute_loss(
                pred_rgb, gt, pixel_weight=pixel_weight
            )
            loss = loss + self.branch_loss_weight * self._compute_loss(
                pred_thm, gt, pixel_weight=pixel_weight
            )
            output["loss"] = loss

        return output
