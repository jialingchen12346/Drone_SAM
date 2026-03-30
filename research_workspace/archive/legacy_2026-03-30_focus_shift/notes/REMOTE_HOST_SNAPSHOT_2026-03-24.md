# 远程主机结构快照（2026-03-24）

## 连接信息

- Host: `connect.bjb1.seetacloud.com`
- Port: `37143`
- User: `root`
- 登录后工作目录: `/root`

## 机器与驱动

- Hostname: `autodl-container-fd404f8665-3dbaface`
- GPU:
  - `NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.76.05`
  - `NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.76.05`
- 时间快照: `2026-03-24 10:16:38 CST`

## 项目主路径

- 项目根目录: `/root/Drone-SAM-Adapter`
- 关键一级目录:
  - `checkpoints/`
  - `configs/`
  - `data/`
  - `docs/`
  - `research_workspace/`
  - `sam2/`
  - `scripts/`
  - `segmentation/`
  - `work_dirs/`

## 研究与计划目录

- 研究笔记: `/root/Drone-SAM-Adapter/research_workspace/notes`
- 当前计划: `/root/Drone-SAM-Adapter/research_workspace/plans/current_plan.yaml`

## 数据目录

- 数据盘根目录: `/root/autodl-tmp/datasets`
- FMB: `/root/autodl-tmp/datasets/FMB/{train,val,test}`
- 额外数据: `/root/autodl-tmp/datasets/drive-download-20260316T091026Z-3-001`
  - 子目录: `anno_json/`, `images/`, `labels/`, `visual/`

## 模型与训练产物

- SAM2 权重: `/root/Drone-SAM-Adapter/checkpoints/sam2_hiera_large.pt`（约 857MB）
- 当前主实验工作目录: `/root/Drone-SAM-Adapter/work_dirs/rrf_dsd_v2_fmb`
  - `best.pth`
  - `epoch_10/20/30/40/50/60/70/80.pth`
  - `val_results.txt`
  - `test_results.txt`
- 训练日志:
  - `/tmp/train_rrf_dsd_v2.log`

## 脚本状态

- 评测脚本: `/root/Drone-SAM-Adapter/segmentation/test_rrf_dsd.py`
  - 已支持 `--split {val,test}`
  - 结果按 split 分开保存（`val_results.txt` / `test_results.txt`）
- 训练脚本: `/root/Drone-SAM-Adapter/segmentation/train_cacaf.py`
  - 已接入 DDP（含 rank0 保存/日志、val barrier 修复）
  - 支持 `--dist-backend {auto,nccl,gloo}` 与 `--force-nccl-on-5090`
- 启动脚本: `/root/Drone-SAM-Adapter/scripts/run_ddp_train.sh`
  - 支持 `NCCL_PREFLIGHT=1` 与 `NCCL_LIB_DIR`（用于指定 NCCL 动态库路径）

## 环境状态（NCCL 修复）

- 旧环境（问题环境）:
  - Python: `/root/miniconda3/bin/python`
  - `torch 2.7.0+cu128`, `NCCL 2.26.2`
  - 现象：full model DDP backward 阶段 `CUDA illegal memory access`
- 新环境（已验证）:
  - Python: `/root/miniconda3/envs/torch211/bin/python`
  - `torch 2.11.0+cu128`, `NCCL 2.28.9`
  - 依赖补齐：`timm`, `hydra-core`, `omegaconf`, `Pillow`, `iopath`
  - 关键变量：`NCCL_LIB_DIR=/root/miniconda3/envs/torch211/lib/python3.12/site-packages/nvidia/nccl/lib`
  - 验证结果：`ddp_model_preflight.py --task full`（含 forward/backward）通过

## 运维注意事项

- `/root/Drone-SAM-Adapter` 下文件所有者多为 `uid=1000`，当前 `root` 可读写。
- 建议继续采用“本地为真源 + 增量 rsync + 哈希校验”的同步策略，避免远程手改漂移。
- 若再次更换实例或端口，优先更新本文件并在 `SESSION_HANDOFF.md` 追加链接。
