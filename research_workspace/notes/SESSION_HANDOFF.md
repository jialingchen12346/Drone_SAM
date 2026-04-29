# 会话交接（当前版）

更新时间：2026-04-23

历史版本已归档：

- `research_workspace/archive/2026-04-16_label_eff_refresh/notes/SESSION_HANDOFF.pre_refresh.md`

## 当前目标

- 主线：标注高效 RGB-T 语义分割。
- 核心协议：`10% dense labels + unlabeled RGB-T + class-presence point prompts`。
- 核心方法：`SAM2 Hiera Adapter + ConvNeXt thermal branch + modality heads + confidence-gated/consensus-weighted pseudo labels + disagreement refinement`。
- 主指标：FMB test strict mIoU（`absent-score=0.0`）。

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

1. 回收 E4：判断 teacher-based checkpoint selection 是否改善 r20。
2. 回收 E5：判断 naive fusion baseline 与主线 `60.59` 的差距。
3. 启动 `RGB-only / Thermal-only + same semi+point`。
4. 补 `MeanTeacher/UniMatch-style RGB-T baseline`。
5. 写 failure visualization 脚本。

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
