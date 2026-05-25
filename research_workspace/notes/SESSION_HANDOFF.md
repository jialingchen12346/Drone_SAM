# 会话交接（当前版）

更新时间：2026-05-25

历史版本已归档：

- `research_workspace/archive/2026-04-16_label_eff_refresh/notes/SESSION_HANDOFF.pre_refresh.md`

## 当前目标

- 主线：标注高效 RGB-T 语义分割。
- 核心协议：`10% dense labels + unlabeled RGB-T + class-presence point prompts`。
- 核心方法：`SAM2 Hiera Adapter + ConvNeXt thermal branch + modality heads + confidence-gated/consensus-weighted pseudo labels + disagreement refinement`。
- 主指标：FMB test strict mIoU（`absent-score=0.0`）。

## 2026-05-06 紧急更新（PPAL 快筛 + 评估口径）

### A. PPAL e20 快筛结果（r10, seed42）

| 实验 | with prompt | without prompt | delta |
|---|---:|---:|---:|
| E2 concat | 57.57 | 55.72 | +1.85 |
| E3 ppal | 55.96 | 54.34 | +1.62 |

补充：

- E1 point-loss-only（best@epoch24）test strict：58.69（无 prompt-aware 前向）。
- 当前同预算结论：`E2 > E3`。

### B. 关键口径修正

问题根因：

- 之前 with-prompt 评估曾使用 `fmb_points_seed42_r10_all`（train-only）。
- 导致 val/test 提示命中近似 0，with-prompt 增益被低估。

修复动作（已完成）：

- 生成并同步：
  - `research_workspace/artifacts/weak_labels/fmb_points_seed42_val_all/index.json`
  - `research_workspace/artifacts/weak_labels/fmb_points_seed42_test_all/index.json`
- 命中率核验：
  - val: `160/160`
  - test: `280/280`

后续规则：

- val/test with-prompt 评估必须用对应 split 的 point index。
- 日志中必须附带 `point_index` 路径与 split 命中率说明。

## 2026-04-23 新增：全监督对齐快照

同口径对齐结论：

| 项目 | strict test mIoU | 位置 |
|---|---:|---|
| MM-SAM official checkpoint | 66.10 | 本地 4070TiS / 远端 4090 (`ssh -p 55633 root@connect.westc.seetacloud.com`) |
| Ours best (`unfreeze4+tpi`, epoch_30) | 64.35（远端）/ 64.31（本地重测） | `ssh -p 36176 root@connect.bjb2.seetacloud.com` |
| Ours (`unfreeze8+tpi`, e50 best) | 63.07 | `ssh -p 46181 root@connect.bjb2.seetacloud.com` |
| Ours (`unfreeze4` no tpi) | 61.64 | `ssh -p 46181 root@connect.bjb2.seetacloud.com` |

关键结论：

- `unfreeze4+tpi` 是当前全监督最优线，`epoch_30` 优于 `best.pth`。
- `unfreeze8+tpi` 与 `unfreeze4` no-tpi 都低于 `unfreeze4+tpi`。
- 我们与 MM-SAM 同口径当前差距约 `1.79`。

关键 work_dir：

```text
/root/autodl-tmp/work_dirs/r100_sup_mmsa_agf_dref_prob_crop800_ohem_stretch_unfreeze4_tpi_e30_seed42
/root/autodl-tmp/work_dirs/r100_sup_mmsa_agf_dref_prob_crop800_ohem_stretch_unfreeze8_tpi_e50_seed42
/root/autodl-tmp/work_dirs/r100_sup_mmsa_agf_dref_prob_crop800_ohem_stretch_unfreeze4_e30_seed42
```

## 2026-05-02 口径修正：`seed2026 = 62.69` 属于全监督，不属于 label-efficient

远端核对结论：

- `357机` / `ssh -p 35877 root@connect.bjb1.seetacloud.com`
- 启动脚本：`/root/run_rrf_relrefine_v2_ema_valteacher_seed2026_35877.sh`
- 训练入口：`segmentation/train_cacaf.py`
- 日志：`Training on 1060 images, validating on 160 images`

这说明：

- `seed2026 = 62.69` 是全监督 `RRF-DSD + reliability-guided refine + EMA teacher validation/selection` 实验。
- 它不属于 `r10 label-efficient` 主线，之前文档把它记成 label-efficient 主结果是错误归档。

机器映射：

- `357机` = `ssh -p 35877 root@connect.bjb1.seetacloud.com`
- `481机` = `ssh -p 21824 root@connect.bjb2.seetacloud.com`

已完成的全监督方向1探索：

| 实验 | 机器/远端 | best 选择 | work_dir | best teacher test strict |
|---|---|---|---|---:|
| v2 ema valteacher seed42 | `357机` / `35877` | overall `mIoU` | `/root/autodl-tmp/work_dirs/rrf_dsd_relrefine_v2_ema_valteacher_seed42` | `62.17` |
| v2 ema critsel seed42 | `481机` / `21824` | `critical_mean` | `/root/autodl-tmp/work_dirs/rrf_dsd_relrefine_v2_ema_critsel_seed42` | `61.95` |
| v2 ema valteacher seed3407 | `357机` / `35877` | overall `mIoU` | `/root/autodl-tmp/work_dirs/rrf_dsd_relrefine_v2_ema_valteacher_seed3407` | `60.11` |
| v2 ema valteacher seed2026 | `357机` / `35877` | overall `mIoU` | `/root/autodl-tmp/work_dirs/rrf_dsd_relrefine_v2_ema_valteacher_seed2026` | **62.69** |
| v2 ema valteacher seed1234 | `481机` / `21824` | overall `mIoU` | `/root/autodl-tmp/work_dirs/rrf_dsd_relrefine_v2_ema_valteacher_seed1234` | `61.92` |

全监督方向1分布：

- `seed42 / teacher / overall miou`: `62.17`
- `seed3407 / teacher / overall miou`: `60.11`
- `seed2026 / teacher / overall miou`: **`62.69`**
- `seed1234 / teacher / overall miou`: `61.92`
- `seed42 / teacher / critical_mean`: `61.95`

四个 `overall miou + teacher` seed 的均值：

```text
(62.17 + 60.11 + 62.69 + 61.92) / 4 = 61.72
```

当前正确口径：

- label-efficient 主线结果仍是 `r10 dref prob / w=0.5 / bs=2+2 / seed42 = 60.59`。
- `seed2026 = 62.69` 只能写成全监督方向1探索结果，不能写成 label-efficient 主结果。
- 单模态对照 `58.09 / 51.72` 的同骨架公平比较对象仍是 `MMSA multi-modal seed42 = 60.59`，不是 `62.69`。

## 当前最佳配置

```text
r10 / e30 / seed42 / bs=2+2
enable modality heads
pseudo agreement: argmax + soft_weight + floor 0.5
fusion agreement map: argmax
disagreement refinement: prob / weight 0.5 / gate on
```

结果：

| 配置 | val strict | test strict | 备注 |
|---|---:|---:|---|
| dref prob w=0.5 bs=2+2 seed42 | 66.70 | 60.59 | 当前最佳 |
| dref prob w=0.5 bs=2+2 seed3407 | 65.79 | 59.88 | 复现种子 |
| thermal-prior-injection seed42 | 65.25 | 60.53 | 打平但无增益 |
| dref prob w=0.5 no-gate seed42 | 65.58 | 58.54 | 证明 gate 必要 |
| SAM2 point proxy seed42 | 67.70 | 58.48 | val 高但 test 下降 |
| SAM2 point proxy w=0.1 | 66.98 | 59.70 | 降权有恢复但未超主线 |
| SAM2 point proxy w=0.1 + agreement gate | 66.31 | 59.08 | gate 无收益 |

## 2026-05-02 新增：单模态对照（同协议，label-efficient）

已完成：

| 实验 | 机器/远端 | work_dir | best test strict |
|---|---|---|---:|
| RGB-only | `481机` / `21824` | `/root/autodl-tmp/work_dirs/label_eff_r10_rgbonly_e30_seed42_softw_agf_dref_prob` | `58.09` |
| Thermal-only | `357机` / `35877` | `/root/autodl-tmp/work_dirs/label_eff_r10_thermalonly_e30_seed42_softw_agf_dref_prob` | `51.72` |

结论分两层写：

- 同骨架公平对比：
  - `MMSA multi-modal seed42 = 60.59`
  - `RGB-only = 58.09`
  - `Thermal-only = 51.72`
- 对当前最好主结果的实际差距：
  - 当前 label-efficient 主线 `MMSA multi-modal seed42 = 60.59`
  - 相比 `RGB-only` 高 `2.50`
  - 相比 `Thermal-only` 高 `8.87`

补充观察：

- `RGB-only` 训练后期出现大量 `non-finite loss`，但 `best.pth` 可正常测试到 `58.09`。
- 单模态结果说明：RGB 明显强于 Thermal；多模态融合收益是实质性的，不是任一单模态本身已足够强。

## 当前运行实验

### E1. no-gate 消融

远端：

```bash
ssh -p 38755 root@connect.bjb2.seetacloud.com
```

任务：

```text
dref_prob_nogate_r10_e30_seed42
```

目的：验证 `disagreement_map * delta` 的空间门控是否为核心贡献。

状态（2026-04-16 回收完成）：

- 训练完成，best val strict mIoU `65.58`。
- 已补 test，test strict mIoU `58.54`。
- 结论：低于 gate-on 主线 `60.59`，说明 `disagreement_map * delta` 空间门控是核心机制。

日志：

```bash
/root/autodl-tmp/logs/dref_prob_nogate_r10_e30_seed42.log
```

work_dir：

```bash
/root/autodl-tmp/work_dirs/dref_prob_nogate_r10_e30_seed42
```

### E2. SAM2-Guided Label Propagation 探索

远端：

```bash
ssh -p 21824 root@connect.bjb2.seetacloud.com
```

任务：

```text
sam2_proxy_r10_e30_seed42
```

流程：

1. 生成 SAM2 point proxy masks：
   - `/root/autodl-tmp/sam2_point_proxy_r10_box96_a015/index.json`
   - `box_radius=96`
   - `max_mask_area_ratio=0.15`
2. 自动启动 r10/e30 训练：
   - `--point-proxy-index .../index.json`
   - `--point-proxy-weight 1.0`
   - `--point-proxy-intersect-pseudo`

状态（2026-04-16 回收完成）：

- proxy mask 全量生成完成：954 samples，7636 kept masks。
- 训练完成，best val strict mIoU `67.70`。
- 已补 test，test strict mIoU `58.48`。
- 结论：val 明显升高但 test 下降，暂不并入主线。

### E3. SAM2 Proxy 减法重试

新增代码：

```text
--point-proxy-agreement-gate
--point-proxy-agreement-mode {argmax,prob}
--point-proxy-agreement-thresh
```

已完成两组：

| 实验 | 远端 | 日志 | work_dir |
|---|---|---|---|
| proxy w=0.1 | `ssh -p 21824 root@connect.bjb2.seetacloud.com` | `/root/autodl-tmp/logs/sam2_proxy_w01_r10_e30_seed42.log` | `/root/autodl-tmp/work_dirs/sam2_proxy_w01_r10_e30_seed42` |
| proxy w=0.1 + agreement gate | `ssh -p 38755 root@connect.bjb2.seetacloud.com` | `/root/autodl-tmp/logs/sam2_proxy_w01_agree_r10_e30_seed42.log` | `/root/autodl-tmp/work_dirs/sam2_proxy_w01_agree_r10_e30_seed42` |

结果：

- `proxy w=0.1`：val strict `66.98`，test strict `59.70`。
- `proxy w=0.1 + agreement gate`：val strict `66.31`，test strict `59.08`。
- 结论：降权可以缓解原始 proxy 的 test 崩塌，但仍低于主线；agreement gate 没有收益，SAM2 proxy 暂不并入主线。

## 已知实验结论

1. `modality heads` 是早期最大确定收益来源。
2. `soft agreement weighting` 优于硬过滤。
3. `Agreement-Guided Fusion` 有效，但不是单独主贡献。
4. `Disagreement Refinement prob w=0.5` 是当前最强结构组件。
5. `prob mode` 对 batch size 敏感，`bs=1+1` 崩，`bs=2+2` 稳定。
6. point diffusion、class-balanced sampler、MFNet external unlabeled、dual-branch no-fusion 均不作为主线。
7. 训练代码本来已有 EMA teacher，伪标签来自 teacher；新补的是 teacher 验证/评测入口。

### E4. r20 EMA Teacher 选点验证

已新增代码：

```text
train_label_efficient.py: --val-model {student,teacher}
eval_label_eff.py: --checkpoint-state {model,teacher}
```

新增软一致性蒸馏代码（默认关闭）：

```text
--soft-consistency-weight
--soft-consistency-loss {kl,mse}
--soft-consistency-temperature
--soft-consistency-reliability {none,agreement}
--soft-consistency-agreement-mode {argmax,prob}
--soft-consistency-agreement-floor
```

推荐首跑配置：

```text
--val-model teacher
--soft-consistency-weight 0.2
--soft-consistency-loss kl
--soft-consistency-temperature 2.0
--soft-consistency-reliability agreement
--soft-consistency-agreement-mode prob
--soft-consistency-agreement-floor 0.5
```

已有 r20 best checkpoint 诊断：

| checkpoint state | test strict |
|---|---:|
| student/model | 59.02 |
| teacher | 58.58 |

结论：直接评测现有 best checkpoint 的 teacher 不能修复 r20 倒挂。

已完成：

| 实验 | 远端 | 日志 | work_dir |
|---|---|---|---|
| r20 dref prob `--val-model teacher` | `ssh -p 38755 root@connect.bjb2.seetacloud.com` | `/root/autodl-tmp/logs/r20_dref_valteacher_e30_seed42.log` | `/root/autodl-tmp/work_dirs/label_eff_r20_e30_sp_seed42_softw_agf_dref_prob_valteacher` |

结果：

```text
best val strict mIoU = 71.14
test_results_student.txt: test strict mIoU = 60.98
test_results_teacher.txt: test strict mIoU = 61.48
```

结论：用 EMA teacher 做验证和 best checkpoint 选择有效。直接评测 student-selected checkpoint 的 teacher 是 `58.58`，但 teacher-selected checkpoint 的 teacher test 达到 `61.48`，说明关键不是“打捞 teacher”，而是用 teacher 稳健性作为 checkpoint selection（检查点选择）准则。

### E4b. EMA Soft Consistency

已完成：

| 实验 | 远端 | 日志 | work_dir |
|---|---|---|---|
| r20 dref prob `--val-model teacher` + soft consistency w=0.2 | `ssh -p 38755 root@connect.bjb2.seetacloud.com` | `/root/autodl-tmp/logs/r20_dref_valteacher_scons_w02_e30_seed42.log` | `/root/autodl-tmp/work_dirs/label_eff_r20_e30_sp_seed42_softw_agf_dref_prob_valteacher_scons_w02` |

关键参数：

```text
--soft-consistency-weight 0.2
--soft-consistency-loss kl
--soft-consistency-temperature 2.0
--soft-consistency-reliability agreement
--soft-consistency-agreement-mode prob
--soft-consistency-agreement-floor 0.5
```

目的：验证 EMA teacher 是否能从 checkpoint selection 进一步扩展为概率分布教师，并用跨模态 agreement 对软一致性监督做像素级可靠性加权。

结果：

```text
best val strict mIoU = 67.04
student/model test strict mIoU = 56.99
teacher test strict mIoU = 57.09
```

结论：明显低于 r20 teacher-selection baseline `61.48`。当前软一致性会过度平滑或强化错误伪标签，不并入主线。

### E4c. DGR 几何约束

已新增训练参数：

```text
--dgr-loss-weight
--dgr-disagreement-mode {argmax,prob}
--dgr-feature-key {feat_rgb_enc,feat_rgb_raw_enc}
```

已完成：

| 实验 | 远端 | 日志 | work_dir |
|---|---|---|---|
| r20 dref prob `--val-model teacher` + DGR w=0.05 | `ssh -p 21824 root@connect.bjb2.seetacloud.com` | `/root/autodl-tmp/logs/r20_dref_valteacher_dgr005_e30_seed42.log` | `/root/autodl-tmp/work_dirs/label_eff_r20_e30_sp_seed42_softw_agf_dref_prob_valteacher_dgr005` |

关键参数：

```text
--dgr-loss-weight 0.05
--dgr-disagreement-mode prob
--dgr-feature-key feat_rgb_enc
```

目的：只在 RGB/Thermal 分歧区域，用 SAM2 低层特征边缘约束 refined prediction（细化预测）的边界梯度。该组不启用 soft consistency，用于单独判断几何先验是否有效。

结果：

```text
best val strict mIoU = 65.78
student/model test strict mIoU = 60.77
teacher test strict mIoU = 60.81
```

类别观察：

```text
Bus = 63.6
Truck = 16.8
Traffic Light = 40.4
Pole = 46.9
```

结论：低于 r20 teacher-selection baseline `61.48`。DGR 对 Bus 有明显正向信号，但 Truck 大幅下降，整体不稳定，暂不并入主线。

### E5. Naive Fusion Baseline

新增代码：

```text
MMSABaselineSegmentor: --mmsa-fusion-mode {mmsa,naive}
```

`naive` 定义：

```text
RGB feature 1x1 align
Thermal feature 1x1 align
fused = 0.5 * (rgb + thermal)
```

不使用 cross-attention（交叉注意力）、quality weighting（质量加权）或 AGF（一致性引导融合）。

已完成：

| 实验 | 远端 | 日志 | work_dir |
|---|---|---|---|
| r10 naive fusion + soft agreement + dref prob | `ssh -p 21824 root@connect.bjb2.seetacloud.com` | `/root/autodl-tmp/logs/naive_fusion_r10_e30_seed42.log` | `/root/autodl-tmp/work_dirs/naive_fusion_r10_e30_sp_seed42_softw_dref_prob` |

结果：

```text
val strict mIoU = 64.10
test strict mIoU = 55.79
```

结论：明显低于 r10 主线 `60.59`，证明当前收益不是任意 RGB-T 融合都能得到，而是依赖当前融合/跨模态可靠性/分歧纠错组合。

## 代码状态

关键文件：

| 文件 | 作用 |
|---|---|
| `segmentation/train_label_efficient.py` | 半监督 + 点监督训练入口 |
| `segmentation/eval_label_eff.py` | checkpoint test/val 评测 |
| `segmentation/models/segmentors/mmsa_baseline_segmentor.py` | 当前主模型 |
| `segmentation/models/fusion/mmsa_fusion.py` | Agreement-Guided Fusion |
| `scripts/generate_point_labels.py` | point prompt 生成 |
| `scripts/prepare_label_splits.py` | label ratio 划分 |

## 点标注协议事实

- 文件：`research_workspace/artifacts/weak_labels/fmb_points_seed42_r10_u/index.json`
- 生成逻辑：每张无标注图中，每个出现类别随机采 1 个 mask 内点。
- 过滤：类别像素数小于 8 跳过。
- 统计：954 张图，7671 点，平均 8.04 点/图。
- loss：严格单点 CE，无窗口平均；当前主线不启用 diffusion。

## 下一步

1. 回收 E6 (GFFM+MidCorr)：判断早期轻量交互是否提升 r10。
2. 如果两个 seed 均正收益，做 GFFM-only / MidCorr-only 消融。
3. 补 `MeanTeacher/UniMatch-style RGB-T baseline`。
4. 写 failure visualization 脚本。

## 2026-05-02 新增：GFFM + MidLevelCorrection 早期交互实验

新增代码文件：

| 文件 | 作用 |
|---|---|
| `segmentation/models/fusion/early_interaction.py` | GFFM + MidLevelCorrection 模块 |

新增 CLI 参数（`train_label_efficient.py` + `eval_label_eff.py`）：

```text
--enable-gffm           # 开启 GFFM 多尺度双向交叉注意力
--enable-mid-correction  # 开启 Mid-Level feature disagreement-gated correction
```

### E6. GFFM + MidLevelCorrection

正在进行：

| 实验 | 机器/远端 | seed | work_dir |
|---|---|---|---|
| r10 gffm+midcorr | `357机` / `35877` | 42 | `/root/autodl-tmp/work_dirs/label_eff_r10_gffm_midcorr_e30_seed42_softw_agf_dref_prob` |
| r10 gffm+midcorr | `481机` / `21824` | 3407 | `/root/autodl-tmp/work_dirs/label_eff_r10_gffm_midcorr_e30_seed3407_softw_agf_dref_prob` |

日志：

```text
/root/autodl-tmp/logs/label_eff_r10_gffm_midcorr_e30_seed42_softw_agf_dref_prob.log
/root/autodl-tmp/logs/label_eff_r10_gffm_midcorr_e30_seed3407_softw_agf_dref_prob.log
```

启动脚本：

```text
/root/run_gffm_midcorr_seed42.sh
/root/run_gffm_midcorr_seed3407.sh
```

关键参数（与主线一致，新增 `--enable-gffm --enable-mid-correction`）：

```bash
--label-ratio 10
--batch-size-l 2 --batch-size-u 2
--epochs 30
--val-model teacher
--enable-modality-heads --modality-head-weight 0.2
--fusion-use-agreement-map --fusion-agreement-mode argmax
--enable-disagreement-refine --disagreement-refine-mode prob --disagreement-refine-weight 0.5
--pseudo-use-agreement --pseudo-agreement-policy soft_weight --pseudo-agreement-floor 0.5
--enable-gffm --enable-mid-correction
--bf16
```

GFFM 设计要点：
- 每尺度双向交叉注意力，mid_ch = min(rgb_ch, aux_ch)
- Adaptive pooling 控制 token 数（max 1024/f1, 1024/f2, 1024/f3, 1024/f4）
- Gamma 可学习参数初始化为 0（训练从恒等开始）
- 逐尺度独立参数，不共享

MidLevelCorrection 设计要点：
- 仅在 stride 16/32 上做（f3, f4）
- Cosine similarity 作为 feature-level disagreement gate
- 只校正 Thermal 分支，RGB 不变
- Gamma 初始化为 0

对照 baseline：
- seed42: 60.59 (dref prob)
- seed3407: 59.88 (dref prob)

## 快速命令

```bash
# 查看 no-gate 训练
ssh -p 38755 root@connect.bjb2.seetacloud.com
tail -n 80 /root/autodl-tmp/logs/dref_prob_nogate_r10_e30_seed42.log

# 训练结束后评测 test
cd /root/Drone-SAM-Adapter
python segmentation/eval_label_eff.py \
  --checkpoint /root/autodl-tmp/work_dirs/dref_prob_nogate_r10_e30_seed42/best.pth \
  --data-root /root/autodl-tmp/datasets/FMB \
  --split test \
  --model-variant mmsa_baseline \
  --sam2-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-ckpt checkpoints/sam2.1_hiera_large.pt \
  --enable-modality-heads --modality-head-weight 0.2 \
  --fusion-use-agreement-map --fusion-agreement-mode argmax \
  --enable-disagreement-refine --disagreement-refine-mode prob --disagreement-refine-weight 0.5 \
  --no-disagreement-refine-gate \
  --bf16 \
  --out /root/autodl-tmp/work_dirs/dref_prob_nogate_r10_e30_seed42/test_results.txt
```
