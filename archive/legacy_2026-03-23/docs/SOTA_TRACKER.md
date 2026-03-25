# SOTA Tracker

> 用途：为论文中的 FMB / MFNet 对比表建立可追溯证据池
> 原则：只记录“已核对来源”的方法、数值和链接；未核到原始来源的不写进正式表格
> 更新：2026-03-18（MFNet 口径修正为 9 类正式协议）

---

## 1. 口径约定

- 论文正文只使用 **最终模型** 与 **正式消融**，不出现 `v4 / v5 / v7` 等研发过程标记。
- 对比表优先使用 **原论文 / 官方项目页 / 官方代码仓库** 中明确给出的结果。
- 对于官方仓库列出多个权重时，先记录全部数值，再单独标记“候选最好值”。
- 对于只给出 `RGB-easy / RGB-hard` 的方法，不直接混入整体 `Test mIoU` 主表。

---

## 2. FMB 已核对条目

| 方法 | 模态 | Backbone | mIoU | 备注 | 来源状态 |
|------|------|----------|------|------|---------|
| MM SAM-Adapter | RGB-T | SAM ViT-L + ConvNeXt-S | 66.10 | 官方 README 明确给出 FMB test 结果 | 已核对 |
| StitchFusion | RGB-T | 未在 README 明确写 backbone | 64.85 | 官方仓库 Model Zoo 给出 FMB 结果 | 已核对 |
| Ours (CACAF-HGSOAD) | RGB-T | SAM2 Hiera-L + ConvNeXt-Tiny | 68.62 | 本地 checkpoint + `eval_fullres.py` 于 2026-03-18 复核 | 已核对 |

### 2.1 FMB split 结果的辅助证据

以下数值来自 MM SAM-Adapter 论文正文对 Table 5 的文字分析，可作为辅助叙述，但**不能替代整体 Test mIoU 主表**：

| 方法 | Split | mIoU | 说明 | 来源状态 |
|------|-------|------|------|---------|
| MM SAM-Adapter | RGB-hard | 62.59 | 论文正文描述值 | 已核对 |
| RoadFormer+ | RGB-hard | 61.79 | 论文正文描述值 | 已核对 |
| MM SAM-Adapter | RGB-easy | 68.45 | 论文正文描述值 | 已核对 |
| GeminiFusion | RGB-easy | 68.05 | 论文正文描述值 | 已核对 |

### 2.2 FMB 待核对候选方法

- GeminiFusion
- RoadFormer+
- CAFuser
- 其他在 FMB 上公开报告整体 `Test mIoU` 的 RGB-T 方法

---

## 3. MFNet 已核对条目

| 方法 | 模态 | Backbone | mIoU | 备注 | 来源状态 |
|------|------|----------|------|------|---------|
| CMNeXt | RGB-T | MiT-B4 | 59.9 | CVPR 2023 论文 Table 1(b) | 已核对 |
| CRM | RGB-T | Swin-B | 61.4 | 官方仓库 README | 已核对 |
| StitchFusion | RGB-T | 未在 README 明确写 backbone | 57.91 / 57.80 / 58.13 | 官方仓库列出 3 个权重结果，候选最好值为 58.13 | 已核对 |
| MFNet (2017) | RGB-T | - | 45.1 | 用户提供，按 9 类协议整理 | 用户提供 |
| RTFNet (2019) | RGB-T | - | 53.2 | 用户提供，按 9 类协议整理 | 用户提供 |
| EGFNet (2022) | RGB-T | - | 54.8 | 用户提供，按 9 类协议整理 | 用户提供 |
| GMNet (2021) | RGB-T | - | 57.3 | 与 CMNeXt 论文 Table 1(b) 中数值一致 | 已核对 / 用户提供 |
| Ours (CACAF-HGSOAD) | RGB-T | SAM2 Hiera-L + ConvNeXt-Tiny | 54.73 | MFNet v2 Test，严格 9 类协议 | 用户提供 |

### 3.1 MFNet 待补候选方法

- FEANet
- FuseSeg
- MFTNet
- DooDLeNet

### 3.2 MFNet 正式口径说明

- 论文只采用 **MFNet v2 Test** 的 **9 类协议** 结果。
- 早期一次 `best.pth` 结果因按 **8 类口径** 统计而偏高，**不得写入论文**。
- 正式结果：
  - `mIoU = 54.73`
  - `mAcc = 68.14`
  - `aAcc = 97.90`

### 3.3 MFNet per-class IoU（9 类协议）

| 类别 | IoU |
|------|-----|
| unlabeled | 98.0 |
| car | 90.2 |
| person | 74.7 |
| bike | 62.8 |
| curve | 47.3 |
| car_stop | 13.1 |
| guardrail | 2.2 |
| color_cone | 54.7 |
| bump | 49.6 |

---

## 4. 本地正式结果

### 4.1 FMB 主结果与正式消融

以下结果已于 2026-03-18 使用本地脚本复核：

| 配置 | mIoU | mAcc | aAcc | 备注 |
|------|------|------|------|------|
| Full | 68.62 | 75.75 | 93.27 | `work_dirs/v5_ablation_no_ohem/epoch_60.pth` |
| w.o. CACAF | 62.86 | 69.52 | 93.32 | `work_dirs/abl_clean_no_cacaf/epoch_60.pth` |
| w.o. SAGU | 65.57 | 71.53 | 93.45 | `work_dirs/abl_clean_no_sagu/epoch_60.pth` |
| w.o. CACAF and SAGU | 64.78 | 70.69 | 93.31 | `work_dirs/abl_clean_no_cacaf_no_sagu/epoch_60.pth` |

### 4.2 论文中建议采用的命名

- `Full`
- `w.o. CACAF`
- `w.o. SAGU`
- `w.o. CACAF and SAGU`

不要写：

- `v5_ablation_no_ohem`
- `abl_clean_no_cacaf`
- 任何中间实验代号

---

## 5. 已核对来源

### 5.1 官方论文 / 项目页

1. MM SAM-Adapter  
   https://arxiv.org/abs/2509.10408  
   https://ieeexplore.ieee.org/document/11162503  
   https://github.com/mug-ml/Multimodal-SAM-Adapter
2. CMNeXt / DeLiVER  
   https://arxiv.org/abs/2303.01480  
   https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Delivering_Arbitrary-Modal_Semantic_Segmentation_CVPR_2023_paper.pdf
3. CRM  
   https://arxiv.org/abs/2303.17324  
   https://github.com/UkcheolShin/CRM_RGBTSeg
4. StitchFusion  
   https://github.com/LiBingyu01/StitchFusion
5. RTFNet  
   https://labsun.org/pub/RAL2019_rtfnet.pdf  
   https://github.com/yuxiangsun/RTFNet
6. MFNet  
   https://cir.nii.ac.jp/crid/1360013267532856704

### 5.2 本地证据

- 主文稿：[论文草稿.md](/home/jl/Drone-SAM-Adapter/docs/论文草稿.md)
- 项目快照：[SESSION_SNAPSHOT.md](/home/jl/Drone-SAM-Adapter/docs/SESSION_SNAPSHOT.md)
- FMB 评测脚本：[eval_fullres.py](/home/jl/Drone-SAM-Adapter/segmentation/eval_fullres.py)
- 主模型 checkpoint：`/home/jl/Drone-SAM-Adapter/work_dirs/v5_ablation_no_ohem/epoch_60.pth`
- 干净消融 checkpoints：
  - `/home/jl/Drone-SAM-Adapter/work_dirs/abl_clean_no_cacaf/epoch_60.pth`
  - `/home/jl/Drone-SAM-Adapter/work_dirs/abl_clean_no_sagu/epoch_60.pth`
  - `/home/jl/Drone-SAM-Adapter/work_dirs/abl_clean_no_cacaf_no_sagu/epoch_60.pth`

---

## 6. 下一步

1. 从官方来源继续补齐 FMB 整体 `Test mIoU` 方法行，优先 GeminiFusion / RoadFormer+ / CAFuser。
2. 为 MFNet `54.73 mIoU` 补齐可重复的评测命令、日志或脚本注释。
3. 将本文件中“已核对”条目转写为论文中的正式 Table 1 / Table 4。
