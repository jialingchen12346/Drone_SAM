#!/usr/bin/env python3
"""
Test-set evaluation for train_label_efficient checkpoints.

Usage:
  python segmentation/eval_label_eff.py \
    --checkpoint /path/to/best.pth \
    --data-root /root/autodl-tmp/datasets/FMB \
    --split test \
    --absent-score 0.0
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segmentation.datasets.fmb_dataset import CLASSES, FMBDataset, IGNORE_INDEX, NUM_CLASSES
from segmentation.train_label_efficient import (
    SegMetric,
    PointLabelStore,
    apply_single_modality_inputs,
    build_point_prompt_map,
    build_model,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--checkpoint-state",
        default="model",
        choices=["model", "teacher"],
        help="State dict inside train_label_efficient checkpoint to evaluate.",
    )
    p.add_argument("--data-root", default="/root/autodl-tmp/datasets/FMB")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--point-index", default=None, help="Point prompt index json (r10/all recommended for PPAL eval).")
    p.add_argument(
        "--model-variant",
        default="mmsa_baseline",
        choices=["mmsa_baseline", "rrf_dsd", "cacaf", "dual_branch_ct"],
    )
    p.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--rgb-backbone", default="sam2", choices=["sam2", "sam3"])
    p.add_argument("--sam3-ckpt", default="/home/jl/sam3/sam3.1_multiplex.pt")
    p.add_argument("--absent-score", type=float, default=0.0)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--aux-encoder-size", default="tiny", choices=["tiny", "small"])
    p.add_argument("--aux-pretrained-path", default=None,
                   help="Optional local ConvNeXt aux checkpoint path; must match training if aux size differs")
    p.add_argument(
        "--single-modality",
        default="none",
        choices=["none", "rgb", "thermal"],
        help="Input ablation: rgb keeps RGB and zeros thermal; thermal keeps thermal and zeros RGB.",
    )
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-cacaf", action="store_true", default=False)
    p.add_argument("--no-sagu", action="store_true", default=False)
    p.add_argument("--out", default=None, help="Output txt path (default: next to checkpoint)")
    p.add_argument("--use-dice", action="store_true", default=False)
    p.add_argument("--use-ohem", action="store_true", default=False)
    p.add_argument("--ohem-thresh", type=float, default=0.7)
    p.add_argument("--ohem-min-kept", type=int, default=100000)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--detail-aux-weight", type=float, default=0.2)
    p.add_argument("--eval-resize-mode", default="letterbox", choices=["stretch", "letterbox"])
    p.add_argument("--enable-modality-heads", action="store_true", default=False)
    p.add_argument("--modality-head-weight", type=float, default=0.2)
    p.add_argument("--fusion-use-agreement-map", action="store_true", default=False)
    p.add_argument("--fusion-agreement-mode", default="prob", choices=["argmax", "prob"])
    p.add_argument("--mmsa-fusion-mode", default="mmsa", choices=["mmsa", "naive"])
    p.add_argument("--enable-disagreement-refine", action="store_true", default=False)
    p.add_argument("--disagreement-refine-weight", type=float, default=0.5)
    p.add_argument("--disagreement-refine-mode", default="argmax", choices=["argmax", "prob"])
    p.add_argument("--no-disagreement-refine-gate", action="store_true", default=False)
    p.add_argument("--prompt-dref", action="store_true", default=False)
    p.add_argument("--prompt-dref-map-type", default="gaussian", choices=["gaussian", "binary"])
    p.add_argument("--prompt-dref-sigma", type=float, default=5.0)
    p.add_argument("--prompt-dref-strength", type=float, default=1.0)
    p.add_argument("--prompt-dref-apply-teacher", action="store_true", default=False)
    p.add_argument("--ppal-mode", default="ppal", choices=["ppal", "concat"])
    p.add_argument("--num-refine-rounds", type=int, default=1,
                   help="Number of iterative refinement rounds (must match training).")
    p.add_argument(
        "--infer-refine-step-scales",
        default="1.0,0.5,0.25,0.125",
        help="Comma-separated per-round step scales for inference refinement.",
    )
    p.add_argument("--infer-refine-use-conf-gate", action="store_true", default=False,
                   help="Apply low-confidence gate during inference refinement.")
    p.add_argument("--infer-refine-conf-thresh", type=float, default=0.6,
                   help="Confidence threshold for inference refinement gate.")
    p.add_argument("--infer-refine-early-stop", action="store_true", default=False,
                   help="Early-stop inference refinement when update area becomes tiny.")
    p.add_argument("--infer-refine-min-update-ratio", type=float, default=0.005,
                   help="Minimum gated update ratio for continuing inference refinement.")
    p.add_argument("--enable-reliability-guided-refine", action="store_true", default=False)
    p.add_argument("--thermal-prior-injection", action="store_true", default=False)
    p.add_argument("--thermal-prior-init", type=float, default=0.1)
    p.add_argument("--enable-gffm", action="store_true", default=False,
                   help="Enable GFFM: per-scale bidirectional cross-attention before AGF fusion.")
    p.add_argument("--enable-mid-correction", action="store_true", default=False,
                   help="Enable mid-level feature disagreement-gated correction (stride 16/32).")
    p.add_argument("--dual-ensemble-mode", default="avg", choices=["avg", "confidence"])
    p.add_argument("--pseudo-use-agreement", action="store_true", default=False)
    p.add_argument("--pseudo-agreement-mode", default="argmax", choices=["argmax", "prob"])
    p.add_argument(
        "--pseudo-agreement-policy",
        default="hard_filter",
        choices=["hard_filter", "soft_weight"],
    )
    p.add_argument("--pseudo-agreement-floor", type=float, default=0.5)
    return p.parse_args()


@torch.no_grad()
def run_eval(
    model,
    loader,
    device,
    amp_dtype,
    absent_score,
    single_modality,
    point_store=None,
    use_prompt=False,
    prompt_map_type="gaussian",
    prompt_sigma=5.0,
    prompt_strength=1.0,
    output_key="pred",
):
    model.eval()
    metric = SegMetric(NUM_CLASSES, IGNORE_INDEX)
    for i, (rgb, thm, gt) in enumerate(loader):
        rgb = rgb.to(device, non_blocking=True)
        thm = thm.to(device, non_blocking=True)
        rgb, thm = apply_single_modality_inputs(rgb, thm, single_modality)
        kwargs = {}
        if use_prompt and point_store is not None:
            sample = loader.dataset.samples[i]
            sample_id = f"{osp.basename(osp.dirname(sample['rgb']))}/{osp.basename(sample['rgb'])}"
            prompt_map = build_point_prompt_map(
                sample_ids=[sample_id],
                point_store=point_store,
                height=rgb.shape[-2],
                width=rgb.shape[-1],
                num_classes=NUM_CLASSES,
                device=device,
                map_type=prompt_map_type,
                sigma=prompt_sigma,
                strength=prompt_strength,
            )
            kwargs["prompt_map"] = prompt_map
        with autocast("cuda", dtype=amp_dtype):
            out = model(rgb, thm, gt=None, **kwargs)
        logits = out.get(output_key, out["pred"])
        pred = logits.argmax(dim=1).cpu().numpy()
        for p, g in zip(pred, gt.numpy()):
            metric.update(p, g)
    return metric.compute(absent_score=absent_score)


def main():
    args = parse_args()
    args.infer_refine_step_scales = [
        float(x) for x in str(args.infer_refine_step_scales).split(",") if str(x).strip()
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    model = build_model(args, device)
    point_store = PointLabelStore(args.point_index) if args.point_index else None

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.checkpoint_state == "teacher":
        if not isinstance(ckpt, dict) or "teacher" not in ckpt:
            raise KeyError(f"Checkpoint has no teacher state: {args.checkpoint}")
        state = ckpt["teacher"]
    else:
        state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    print(f"[eval] loaded {args.checkpoint} state={args.checkpoint_state}")

    ds = FMBDataset(
        args.data_root,
        split=args.split,
        crop_size=args.crop_size,
        augment=False,
        eval_resize_mode=args.eval_resize_mode,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    results_with_prompt = run_eval(
        model,
        loader,
        device,
        amp_dtype,
        args.absent_score,
        args.single_modality,
        point_store=point_store,
        use_prompt=bool(args.prompt_dref and point_store is not None),
        prompt_map_type=args.prompt_dref_map_type,
        prompt_sigma=args.prompt_dref_sigma,
        prompt_strength=args.prompt_dref_strength,
        output_key="pred_prompt",
    )
    results_without_prompt = run_eval(
        model,
        loader,
        device,
        amp_dtype,
        args.absent_score,
        args.single_modality,
        point_store=None,
        use_prompt=False,
        prompt_map_type=args.prompt_dref_map_type,
        prompt_sigma=args.prompt_dref_sigma,
        prompt_strength=args.prompt_dref_strength,
        output_key="pred_main",
    )
    iou_with = results_with_prompt["iou_per_class"]
    iou_without = results_without_prompt["iou_per_class"]
    valid_cls = [c for c, v in zip(CLASSES, iou_with) if not (v != v)]
    absent_cls = [c for c, v in zip(CLASSES, iou_with) if (v != v)]

    lines = [
        f"model_variant: {args.model_variant}",
        f"checkpoint: {args.checkpoint}",
        f"checkpoint_state: {args.checkpoint_state}",
        f"split: {args.split}",
        f"with_prompt.strict_mIoU: {results_with_prompt['mIoU']:.2f}",
        f"with_prompt.strict_mAcc: {results_with_prompt['mAcc']:.2f}",
        f"with_prompt.aAcc: {results_with_prompt['aAcc']:.2f}",
        f"without_prompt.strict_mIoU: {results_without_prompt['mIoU']:.2f}",
        f"without_prompt.strict_mAcc: {results_without_prompt['mAcc']:.2f}",
        f"without_prompt.aAcc: {results_without_prompt['aAcc']:.2f}",
        f"delta_prompt_minus_base: {results_with_prompt['mIoU'] - results_without_prompt['mIoU']:.2f}",
        f"valid_classes: {len(valid_cls)}/{NUM_CLASSES}",
        f"absent_classes: {absent_cls}",
        "",
        "Per-class IoU with_prompt (strict absent=0):",
    ]
    for c, v in zip(CLASSES, iou_with):
        lines.append(f"  {c}: {v:.1f}" if v == v else f"  {c}: nan")
    lines.append("")
    lines.append("Per-class IoU without_prompt (strict absent=0):")
    for c, v in zip(CLASSES, iou_without):
        lines.append(f"  {c}: {v:.1f}" if v == v else f"  {c}: nan")

    report = "\n".join(lines)
    print(report)

    out_path = args.out or osp.join(osp.dirname(args.checkpoint), "test_results.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"[eval] saved -> {out_path}")


if __name__ == "__main__":
    main()
