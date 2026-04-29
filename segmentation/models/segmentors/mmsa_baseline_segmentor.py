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

from segmentation.models.backbones.aux_encoder import ConvNeXtAux
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


def weighted_cross_entropy_loss(logits, target, pixel_weight, ignore_index=255, class_weight=None):
    ce_map = F.cross_entropy(
        logits,
        target,
        ignore_index=ignore_index,
        reduction="none",
        weight=class_weight,
    )
    valid_mask = (target != ignore_index).float()
    weight = pixel_weight.float() * valid_mask
    denom = weight.sum().clamp_min(1.0)
    return (ce_map * weight).sum() / denom


def weighted_dice_loss(
    pred,
    target,
    pixel_weight,
    num_classes,
    ignore_index=255,
    smooth=1.0,
):
    mask = target != ignore_index
    target_clean = target.clone()
    target_clean[~mask] = 0

    pred_soft = F.softmax(pred, dim=1)
    one_hot = F.one_hot(target_clean, num_classes).permute(0, 3, 1, 2).float()
    weighted_mask = mask.unsqueeze(1).float() * pixel_weight.unsqueeze(1).float()

    pred_soft = pred_soft * weighted_mask
    one_hot = one_hot * weighted_mask

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


class ThermalPriorInjectionBlock(nn.Module):
    """Inject aligned thermal features into RGB features before fusion."""

    def __init__(self, rgb_channels, aux_channels, init_alpha=0.1):
        super().__init__()
        self.align_aux = nn.Sequential(
            nn.Conv2d(aux_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
        )
        hidden_channels = max(rgb_channels // 4, 32)
        self.gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, rgb_feat, aux_feat):
        aux_aligned = self.align_aux(aux_feat)
        if aux_aligned.shape[2:] != rgb_feat.shape[2:]:
            aux_aligned = F.interpolate(
                aux_aligned,
                size=rgb_feat.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
        gate = torch.sigmoid(self.gate(torch.cat([rgb_feat, aux_aligned], dim=1)))
        return rgb_feat + self.alpha * gate * aux_aligned, gate


class ThermalPriorInjection(nn.Module):
    """Lightweight MM-SAM-Adapter-style thermal prior injection."""

    def __init__(self, rgb_channels, aux_channels, init_alpha=0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ThermalPriorInjectionBlock(r_ch, a_ch, init_alpha=init_alpha)
                for r_ch, a_ch in zip(rgb_channels, aux_channels)
            ]
        )

    def forward(self, rgb_features, aux_features):
        injected_features = []
        gates = []
        for block, rgb_feat, aux_feat in zip(self.blocks, rgb_features, aux_features):
            injected, gate = block(rgb_feat, aux_feat)
            injected_features.append(injected)
            gates.append(gate)
        return injected_features, gates


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
        aux_encoder_size="tiny",
        aux_pretrained_path=None,
        unfreeze_rgb_last_n_blocks=0,
        use_dice=False,
        dice_weight=1.0,
        use_ohem=False,
        ohem_thresh=0.7,
        ohem_min_kept=100000,
        fusion_num_heads=4,
        mmsa_fusion_mode="mmsa",
        ce_class_weight=None,
        enable_modality_heads=False,
        modality_head_weight=0.2,
        fusion_use_agreement_map=False,
        fusion_agreement_mode="prob",
        enable_disagreement_refine=False,
        disagreement_refine_weight=0.5,
        disagreement_refine_mode="argmax",
        disagreement_refine_use_gate=True,
        disagreement_refine_channels=64,
        thermal_prior_injection=False,
        thermal_prior_init=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.aux_loss_weight = aux_loss_weight
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        self.use_ohem = use_ohem
        self.ohem_thresh = ohem_thresh
        self.ohem_min_kept = ohem_min_kept
        self.enable_modality_heads = enable_modality_heads
        self.mmsa_fusion_mode = mmsa_fusion_mode
        self.modality_head_weight = modality_head_weight
        self.fusion_use_agreement_map = fusion_use_agreement_map
        self.fusion_agreement_mode = fusion_agreement_mode
        self.enable_disagreement_refine = enable_disagreement_refine
        self.disagreement_refine_weight = disagreement_refine_weight
        self.disagreement_refine_mode = disagreement_refine_mode
        self.disagreement_refine_use_gate = disagreement_refine_use_gate
        self.thermal_prior_injection = thermal_prior_injection

        if self.fusion_use_agreement_map and not self.enable_modality_heads:
            raise ValueError("fusion_use_agreement_map requires enable_modality_heads=True")
        if self.enable_disagreement_refine and not self.enable_modality_heads:
            raise ValueError("enable_disagreement_refine requires enable_modality_heads=True")

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
            unfreeze_last_n_blocks=unfreeze_rgb_last_n_blocks,
        )
        self.aux_encoder = ConvNeXtAux(
            pretrained=pretrained_aux,
            size=aux_encoder_size,
            pretrained_path=aux_pretrained_path,
        )

        rgb_ch = tuple(self.rgb_encoder.out_channels)
        aux_ch = tuple(self.aux_encoder.out_channels)
        if self.thermal_prior_injection:
            self.thermal_prior = ThermalPriorInjection(
                rgb_channels=rgb_ch,
                aux_channels=aux_ch,
                init_alpha=thermal_prior_init,
            )
        self.fusion = MMSAFusion(
            rgb_channels=rgb_ch,
            aux_channels=aux_ch,
            out_channels=rgb_ch,
            num_heads=fusion_num_heads,
            use_agreement_gate=fusion_use_agreement_map,
            fusion_mode=mmsa_fusion_mode,
        )

        self.decoder = SegFormerLiteHead(
            in_channels_list=list(self.rgb_encoder.out_channels),
            num_classes=num_classes,
            decode_channels=decode_channels,
        )
        if self.enable_modality_heads:
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
            # The low-resolution auxiliary logits of modality heads are not used anywhere
            # in the current label-efficient training loop, so keep them out of DDP.
            for head in (self.rgb_head, self.thm_head):
                for aux_name in ("aux_head2", "aux_head3"):
                    aux_mod = getattr(head, aux_name, None)
                    if aux_mod is not None:
                        aux_mod.requires_grad_(False)
        if self.enable_disagreement_refine:
            self.disagreement_refine_head = nn.Sequential(
                nn.Conv2d(num_classes * 3 + 1, disagreement_refine_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(disagreement_refine_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(disagreement_refine_channels, disagreement_refine_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(disagreement_refine_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(disagreement_refine_channels, num_classes, kernel_size=1),
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

    def _compute_agreement_map(self, pred_rgb, pred_thm, mode):
        if mode == "argmax":
            return (pred_rgb.argmax(dim=1) == pred_thm.argmax(dim=1)).float().unsqueeze(1)
        if mode == "prob":
            prob_rgb = F.softmax(pred_rgb, dim=1)
            prob_thm = F.softmax(pred_thm, dim=1)
            return (prob_rgb * prob_thm).sum(dim=1, keepdim=True)
        raise ValueError(f"Unsupported fusion_agreement_mode: {mode}")

    def _compute_disagreement_map(self, pred_rgb, pred_thm, mode):
        if mode == "argmax":
            return (pred_rgb.argmax(dim=1) != pred_thm.argmax(dim=1)).float().unsqueeze(1)
        if mode == "prob":
            prob_rgb = F.softmax(pred_rgb, dim=1)
            prob_thm = F.softmax(pred_thm, dim=1)
            overlap = (prob_rgb * prob_thm).sum(dim=1, keepdim=True)
            return 1.0 - overlap
        raise ValueError(f"Unsupported disagreement_refine_mode: {mode}")

    def forward(self, rgb, aux, gt=None, pixel_weight=None):
        h, w = rgb.shape[2], rgb.shape[3]

        rgb_features_raw = self.rgb_encoder(rgb)
        aux_features = self.aux_encoder(aux)

        output = {}
        rgb_features = rgb_features_raw
        if self.thermal_prior_injection:
            rgb_features, thermal_prior_gates = self.thermal_prior(rgb_features_raw, aux_features)
            output["thermal_prior_gates"] = thermal_prior_gates
            output["feat_rgb_raw_enc"] = rgb_features_raw[0]

        pred_rgb = None
        pred_thm = None
        agreement_map = None
        disagreement_map = None
        if self.enable_modality_heads:
            pred_rgb, _ = self.rgb_head(rgb_features)
            pred_thm, _ = self.thm_head(aux_features)
            pred_rgb = F.interpolate(pred_rgb, size=(h, w), mode="bilinear", align_corners=True)
            pred_thm = F.interpolate(pred_thm, size=(h, w), mode="bilinear", align_corners=True)
            output["pred_rgb"] = pred_rgb
            output["pred_thm"] = pred_thm
            if self.fusion_use_agreement_map:
                agreement_map = self._compute_agreement_map(
                    pred_rgb, pred_thm, self.fusion_agreement_mode
                )
                output["agreement_map"] = agreement_map
            if self.enable_disagreement_refine:
                disagreement_map = self._compute_disagreement_map(
                    pred_rgb, pred_thm, self.disagreement_refine_mode
                )
                output["disagreement_map"] = disagreement_map

        fused_features, fusion_weights = self.fusion(
            rgb_features, aux_features, agreement_map=agreement_map
        )

        # Expose stride-4 encoder features for encoder-level gating (e.g. point diffusion).
        output["feat_rgb_enc"] = rgb_features[0]   # [B, 144, H/4, W/4]
        output["feat_thm_enc"] = aux_features[0]   # [B, 96,  H/4, W/4]

        if self.training and gt is not None:
            main_out, aux_outs = self.decoder(fused_features)
            main_out = F.interpolate(main_out, size=(h, w), mode="bilinear", align_corners=True)
            loss = self._compute_loss(main_out, gt, pixel_weight=pixel_weight)
            for aux_out in aux_outs:
                aux_up = F.interpolate(aux_out, size=(h, w), mode="bilinear", align_corners=True)
                loss = loss + self.aux_loss_weight * self._compute_loss(
                    aux_up, gt, pixel_weight=pixel_weight
                )
            if self.enable_modality_heads:
                loss = loss + self.modality_head_weight * self._compute_loss(
                    pred_rgb, gt, pixel_weight=pixel_weight
                )
                loss = loss + self.modality_head_weight * self._compute_loss(
                    pred_thm, gt, pixel_weight=pixel_weight
                )
            refined_out = main_out
            if self.enable_disagreement_refine:
                refine_in = torch.cat([main_out, pred_rgb, pred_thm, disagreement_map], dim=1)
                delta = self.disagreement_refine_head(refine_in)
                if self.disagreement_refine_use_gate:
                    refined_out = main_out + disagreement_map * delta
                else:
                    refined_out = main_out + delta
                loss = loss + self.disagreement_refine_weight * self._compute_loss(
                    refined_out, gt, pixel_weight=pixel_weight
                )
                output["pred_main"] = main_out
                output["pred_refined"] = refined_out
            output["pred"] = refined_out
            output["loss"] = loss
        else:
            main_out, _ = self.decoder(fused_features)
            main_out = F.interpolate(main_out, size=(h, w), mode="bilinear", align_corners=True)
            refined_out = main_out
            if self.enable_disagreement_refine:
                refine_in = torch.cat([main_out, pred_rgb, pred_thm, disagreement_map], dim=1)
                delta = self.disagreement_refine_head(refine_in)
                if self.disagreement_refine_use_gate:
                    refined_out = main_out + disagreement_map * delta
                else:
                    refined_out = main_out + delta
                output["pred_main"] = main_out
                output["pred_refined"] = refined_out
            output["pred"] = refined_out

        output["fusion_weights"] = fusion_weights
        return output
