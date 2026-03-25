#!/usr/bin/env python3
"""Quick DDP preflight for backend/model compatibility.

Run with torchrun, e.g.:
  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
    segmentation/tools/ddp_model_preflight.py --backend nccl --model-variant rrf_dsd \
    --sam2-ckpt checkpoints/sam2_hiera_large.pt --sam2-cfg configs/sam2.1/sam2.1_hiera_l.yaml
"""

import argparse
import inspect
import os
import os.path as osp
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segmentation.datasets.fmb_dataset import NUM_CLASSES
from segmentation.models.segmentors.cacaf_segmentor import CACafSegmentor
from segmentation.models.segmentors.rrf_dsd_segmentor import RRFDSDSegmentor


def parse_args():
    p = argparse.ArgumentParser("DDP backend preflight")
    p.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--task", default="full", choices=["full", "allreduce", "tiny_ddp"])
    p.add_argument("--model-variant", default="rrf_dsd", choices=["cacaf", "rrf_dsd"])
    p.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt", default="checkpoints/sam2_hiera_large.pt")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ddp-find-unused-params", action="store_true", default=False)
    p.add_argument("--ddp-bucket-cap-mb", type=int, default=25)
    p.add_argument("--ddp-broadcast-buffers", action="store_true", default=False)
    p.add_argument("--ddp-init-sync", action="store_true", default=True)
    p.add_argument("--no-ddp-init-sync", dest="ddp_init_sync", action="store_false")
    p.add_argument("--ddp-gradient-as-bucket-view", action="store_true", default=False)
    p.add_argument("--disable-cudnn", action="store_true", default=False)
    p.add_argument("--freeze-encoders", action="store_true", default=False)
    return p.parse_args()


def build_model(args, device):
    if args.model_variant == "cacaf":
        model = CACafSegmentor(
            sam2_checkpoint=args.sam2_ckpt,
            sam2_config=args.sam2_cfg,
            num_classes=NUM_CLASSES,
            use_dice=False,
            use_ohem=False,
            use_cacaf=True,
            use_sagu=True,
        )
    else:
        model = RRFDSDSegmentor(
            sam2_checkpoint=args.sam2_ckpt,
            sam2_config=args.sam2_cfg,
            num_classes=NUM_CLASSES,
            use_dice=False,
            use_ohem=False,
        )
    return model.to(device)


def main():
    args = parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError("Please run this script with torchrun and at least 2 processes.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    init_pg_kwargs = dict(backend=args.backend, init_method="env://")
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        init_pg_kwargs["device_id"] = torch.device("cuda", local_rank)
    dist.init_process_group(**init_pg_kwargs)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[rank{rank}] stage=init backend={args.backend} task={args.task} world={world_size}", flush=True)

    if args.task == "allreduce":
        x = torch.ones(1024, device=device) * (rank + 1)
        for i in range(20):
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            if i in (0, 19) and rank == 0:
                print(f"[rank0] allreduce iter={i} mean={float(x.mean()):.4f}", flush=True)
            x = x * 0.5
        print(f"[rank{rank}] stage=allreduce_done", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    if args.task == "tiny_ddp":
        torch.manual_seed(args.seed)
        tiny = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 14, kernel_size=1),
        ).to(device)
        ddp_tiny = nn.parallel.DistributedDataParallel(
            tiny,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        print(f"[rank{rank}] stage=tiny_ddp_wrapped", flush=True)
        inp = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        out = ddp_tiny(inp)
        gt = torch.randint(low=0, high=14, size=(args.batch_size, args.image_size, args.image_size), device=device)
        loss = torch.nn.functional.cross_entropy(out, gt)
        loss.backward()
        dist.barrier()
        if rank == 0:
            print(f"[rank0] tiny_ddp_loss={float(loss):.6f}", flush=True)
        dist.destroy_process_group()
        return

    torch.manual_seed(args.seed)
    model = build_model(args, device)
    if args.freeze_encoders:
        for name, p in model.named_parameters():
            if ("rgb_encoder" in name) or ("aux_encoder" in name):
                p.requires_grad_(False)
        if rank == 0:
            print("[rank0] freeze_encoders=True", flush=True)
    print(f"[rank{rank}] stage=model_built", flush=True)

    ddp_kwargs = dict(
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=args.ddp_find_unused_params,
        bucket_cap_mb=args.ddp_bucket_cap_mb,
        broadcast_buffers=args.ddp_broadcast_buffers,
        gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
    )
    if "init_sync" in inspect.signature(nn.parallel.DistributedDataParallel).parameters:
        ddp_kwargs["init_sync"] = args.ddp_init_sync

    ddp_model = nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    print(f"[rank{rank}] stage=ddp_wrapped", flush=True)
    dist.barrier()

    b = args.batch_size
    h = w = args.image_size
    rgb = torch.randn(b, 3, h, w, device=device)
    aux = torch.randn(b, 3, h, w, device=device)
    gt = torch.randint(low=0, high=NUM_CLASSES, size=(b, h, w), device=device)

    ddp_model.train()
    out = ddp_model(rgb, aux, gt)
    print(f"[rank{rank}] stage=forward_done", flush=True)
    loss = out["loss"]
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss in preflight: {float(loss)}")
    loss.backward()
    print(f"[rank{rank}] stage=backward_done", flush=True)

    loss_det = loss.detach().clone()
    dist.all_reduce(loss_det, op=dist.ReduceOp.SUM)
    loss_det /= world_size

    if rank == 0:
        print(
            "[preflight:ok] "
            f"backend={args.backend} variant={args.model_variant} "
            f"loss={float(loss_det):.6f} world_size={world_size}"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
