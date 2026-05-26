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
from segmentation.models.backbones.sam3_vitdet_adapter import SAM3ViTDetAdapter
from segmentation.models.decode_heads.segformer_lite_head import SegFormerLiteHead
from segmentation.models.fusion.early_interaction import GFFM, MidLevelCorrection
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
        rgb_backbone_type="sam2",
        sam3_checkpoint=None,
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
        enable_prompt_guided_dref=False,
        ppal_mode="ppal",
        num_refine_rounds=1,
        refine_round_weights=None,
        infer_refine_step_scales=None,
        infer_refine_use_conf_gate=False,
        infer_refine_conf_thresh=0.6,
        infer_refine_early_stop=False,
        infer_refine_min_update_ratio=0.005,
        thermal_prior_injection=False,
        thermal_prior_init=0.1,
        enable_gffm=False,
        enable_mid_correction=False,
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
        self.enable_prompt_guided_dref = bool(enable_prompt_guided_dref)
        self.enable_ppal = self.enable_prompt_guided_dref
        self.ppal_mode = str(ppal_mode)
        self.num_refine_rounds = num_refine_rounds
        if refine_round_weights is None:
            self.refine_round_weights = [1.0] * num_refine_rounds
        else:
            self.refine_round_weights = list(refine_round_weights)
        if infer_refine_step_scales is None:
            self.infer_refine_step_scales = [1.0]
        else:
            self.infer_refine_step_scales = [float(x) for x in infer_refine_step_scales]
        self.infer_refine_use_conf_gate = bool(infer_refine_use_conf_gate)
        self.infer_refine_conf_thresh = float(infer_refine_conf_thresh)
        self.infer_refine_early_stop = bool(infer_refine_early_stop)
        self.infer_refine_min_update_ratio = float(infer_refine_min_update_ratio)
        self.thermal_prior_injection = thermal_prior_injection

        if self.fusion_use_agreement_map and not self.enable_modality_heads:
            raise ValueError("fusion_use_agreement_map requires enable_modality_heads=True")
        if self.enable_disagreement_refine and not self.enable_modality_heads:
            raise ValueError("enable_disagreement_refine requires enable_modality_heads=True")
        if self.enable_prompt_guided_dref and not self.enable_disagreement_refine:
            raise ValueError("enable_prompt_guided_dref requires enable_disagreement_refine=True")
        if self.enable_ppal and self.ppal_mode not in ("ppal", "concat"):
            raise ValueError("ppal_mode must be one of {'ppal', 'concat'}")

        if ce_class_weight is None:
            self.register_buffer("ce_class_weight", None, persistent=False)
        else:
            self.register_buffer(
                "ce_class_weight",
                torch.tensor(ce_class_weight, dtype=torch.float32),
                persistent=False,
            )

        if rgb_backbone_type == "sam3":
            if not sam3_checkpoint:
                raise ValueError("sam3_checkpoint is required when rgb_backbone_type='sam3'")
            self.rgb_encoder = SAM3ViTDetAdapter(
                checkpoint=sam3_checkpoint,
                bottleneck_dim=bottleneck_dim,
                freeze_backbone=True,
            )
        else:
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
        self.enable_gffm = enable_gffm
        if self.enable_gffm:
            self.gffm = GFFM(
                rgb_channels=rgb_ch,
                aux_channels=aux_ch,
                num_heads=fusion_num_heads,
                init_gamma=0.0,
            )
        self.enable_mid_correction = enable_mid_correction
        if self.enable_mid_correction:
            self.mid_correction = MidLevelCorrection(
                rgb_channels=rgb_ch,
                aux_channels=aux_ch,
                init_gamma=0.0,
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
            if self.enable_ppal:
                # PPAL-v1: prompt injection + prompt gate + residual delta prediction.
                if self.ppal_mode == "ppal":
                    self.prompt_inject_conv = nn.Sequential(
                        nn.Conv2d(num_classes + 2, disagreement_refine_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(disagreement_refine_channels),
                        nn.ReLU(inplace=True),
                    )
                    self.prompt_gate_conv = nn.Sequential(
                        nn.Conv2d(num_classes + 2, disagreement_refine_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(disagreement_refine_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(disagreement_refine_channels, 1, kernel_size=1),
                        nn.Sigmoid(),
                    )
                    delta_in_channels = num_classes * 3 + 1 + disagreement_refine_channels
                else:
                    delta_in_channels = num_classes * 3 + 1 + num_classes + 1

                self.delta_head = nn.Sequential(
                    nn.Conv2d(delta_in_channels, disagreement_refine_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(disagreement_refine_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(disagreement_refine_channels, disagreement_refine_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(disagreement_refine_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(disagreement_refine_channels, num_classes, kernel_size=1),
                )
            else:
                refine_in_channels = num_classes * 3 + 1
                self.disagreement_refine_head = nn.Sequential(
                    nn.Conv2d(refine_in_channels, disagreement_refine_channels, kernel_size=3, padding=1),
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

    def _compute_refined_disagreement_map(self, refined_pred, pred_rgb, pred_thm, mode):
        """Disagreement between refined output and modality heads (union)."""
        if mode == "argmax":
            d_rgb = (refined_pred.argmax(dim=1) != pred_rgb.argmax(dim=1))
            d_thm = (refined_pred.argmax(dim=1) != pred_thm.argmax(dim=1))
            return (d_rgb | d_thm).float().unsqueeze(1)
        if mode == "prob":
            prob_ref = F.softmax(refined_pred, dim=1)
            prob_rgb = F.softmax(pred_rgb, dim=1)
            prob_thm = F.softmax(pred_thm, dim=1)
            overlap_rgb = (prob_ref * prob_rgb).sum(dim=1, keepdim=True)
            overlap_thm = (prob_ref * prob_thm).sum(dim=1, keepdim=True)
            return 1.0 - torch.max(overlap_rgb, overlap_thm)
        raise ValueError(f"Unsupported disagreement_refine_mode: {mode}")

    def _prepare_prompt_conf_maps(self, refined_out, prompt_map=None, teacher_conf_map=None):
        b, _, h, w = refined_out.shape
        if prompt_map is None:
            prompt_map = refined_out.new_zeros((b, self.num_classes, h, w))
        elif prompt_map.shape[-2:] != (h, w):
            prompt_map = F.interpolate(prompt_map, size=(h, w), mode="bilinear", align_corners=False)
        prompt_map = prompt_map.clamp_min(0.0)

        if teacher_conf_map is None:
            conf_map = F.softmax(refined_out.detach(), dim=1).max(dim=1, keepdim=True)[0]
        else:
            conf_map = teacher_conf_map
            if conf_map.shape[-2:] != (h, w):
                conf_map = F.interpolate(conf_map, size=(h, w), mode="bilinear", align_corners=False)
            conf_map = conf_map.clamp(0.0, 1.0)
        return prompt_map, conf_map

    def _compute_ppal_delta_gate(
        self,
        refined_out: torch.Tensor,
        pred_rgb: torch.Tensor,
        pred_thm: torch.Tensor,
        cur_disagreement_map: torch.Tensor,
        prompt_map: torch.Tensor,
        conf_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_in = torch.cat([prompt_map, conf_map, cur_disagreement_map], dim=1)
        if self.ppal_mode == "ppal":
            prompt_feat = self.prompt_inject_conv(prompt_in)
            refine_in = torch.cat(
                [refined_out, pred_rgb, pred_thm, cur_disagreement_map, prompt_feat], dim=1
            )
            delta = self.delta_head(refine_in)
            prompt_gate = self.prompt_gate_conv(prompt_in)
            gate_map = cur_disagreement_map * prompt_gate
        else:
            refine_in = torch.cat(
                [refined_out, pred_rgb, pred_thm, cur_disagreement_map, prompt_map, conf_map], dim=1
            )
            delta = self.delta_head(refine_in)
            gate_map = cur_disagreement_map
        return delta, gate_map

    def forward(self, rgb, aux, gt=None, pixel_weight=None, prompt_map=None, teacher_conf_map=None):
        h, w = rgb.shape[2], rgb.shape[3]

        rgb_features_raw = self.rgb_encoder(rgb)
        aux_features = self.aux_encoder(aux)

        output = {}
        rgb_features = rgb_features_raw
        if self.thermal_prior_injection:
            rgb_features, thermal_prior_gates = self.thermal_prior(rgb_features_raw, aux_features)
            output["thermal_prior_gates"] = thermal_prior_gates
            output["feat_rgb_raw_enc"] = rgb_features_raw[0]

        if self.enable_gffm:
            rgb_features, aux_features = self.gffm(rgb_features, aux_features)
        if self.enable_mid_correction:
            rgb_features, aux_features = self.mid_correction(rgb_features, aux_features)

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
                cur_disagreement_map = disagreement_map
                for r in range(self.num_refine_rounds):
                    round_weight = self.refine_round_weights[r]
                    if self.enable_ppal:
                        prompt_map_r, conf_map_r = self._prepare_prompt_conf_maps(
                            refined_out, prompt_map=prompt_map, teacher_conf_map=teacher_conf_map
                        )
                        delta, gate_map = self._compute_ppal_delta_gate(
                            refined_out, pred_rgb, pred_thm, cur_disagreement_map, prompt_map_r, conf_map_r
                        )
                    else:
                        refine_in = torch.cat(
                            [refined_out, pred_rgb, pred_thm, cur_disagreement_map], dim=1
                        )
                        gate_map = cur_disagreement_map
                        delta = self.disagreement_refine_head(refine_in)
                    if self.disagreement_refine_use_gate:
                        refined_out = refined_out + gate_map * delta
                    else:
                        refined_out = refined_out + delta
                    loss = loss + round_weight * self.disagreement_refine_weight * self._compute_loss(
                        refined_out, gt, pixel_weight=pixel_weight
                    )
                    if r < self.num_refine_rounds - 1:
                        cur_disagreement_map = self._compute_refined_disagreement_map(
                            refined_out, pred_rgb, pred_thm, self.disagreement_refine_mode
                        )
                output["pred_main"] = main_out
                output["pred_prompt"] = refined_out
                output["pred_refined"] = refined_out
            else:
                output["pred_main"] = main_out
                output["pred_prompt"] = main_out
            output["pred"] = refined_out
            output["loss"] = loss
        else:
            main_out, _ = self.decoder(fused_features)
            main_out = F.interpolate(main_out, size=(h, w), mode="bilinear", align_corners=True)
            refined_out = main_out
            if self.enable_disagreement_refine:
                cur_disagreement_map = disagreement_map
                for r in range(self.num_refine_rounds):
                    step_scale = self.infer_refine_step_scales[
                        min(r, len(self.infer_refine_step_scales) - 1)
                    ]
                    if self.enable_ppal:
                        prompt_map_r, conf_map_r = self._prepare_prompt_conf_maps(
                            refined_out, prompt_map=prompt_map, teacher_conf_map=teacher_conf_map
                        )
                        delta, gate_map = self._compute_ppal_delta_gate(
                            refined_out, pred_rgb, pred_thm, cur_disagreement_map, prompt_map_r, conf_map_r
                        )
                    else:
                        refine_in = torch.cat(
                            [refined_out, pred_rgb, pred_thm, cur_disagreement_map], dim=1
                        )
                        gate_map = cur_disagreement_map
                        delta = self.disagreement_refine_head(refine_in)
                    delta = step_scale * delta
                    update_gate = torch.ones_like(cur_disagreement_map)
                    if self.disagreement_refine_use_gate:
                        update_gate = update_gate * gate_map
                    if self.infer_refine_use_conf_gate:
                        conf_map = F.softmax(refined_out, dim=1).max(dim=1, keepdim=True)[0]
                        conf_gate = (conf_map < self.infer_refine_conf_thresh).float()
                        update_gate = update_gate * conf_gate
                    if self.disagreement_refine_use_gate:
                        refined_out = refined_out + update_gate * delta
                    else:
                        refined_out = refined_out + (update_gate * delta if self.infer_refine_use_conf_gate else delta)
                    if self.infer_refine_early_stop:
                        update_ratio = (update_gate > 0.5).float().mean()
                        if update_ratio.item() < self.infer_refine_min_update_ratio:
                            break
                    if r < self.num_refine_rounds - 1:
                        cur_disagreement_map = self._compute_refined_disagreement_map(
                            refined_out, pred_rgb, pred_thm, self.disagreement_refine_mode
                        )
                output["pred_main"] = main_out
                output["pred_prompt"] = refined_out
                output["pred_refined"] = refined_out
            else:
                output["pred_main"] = main_out
                output["pred_prompt"] = main_out
            output["pred"] = refined_out

        output["fusion_weights"] = fusion_weights
        return output
