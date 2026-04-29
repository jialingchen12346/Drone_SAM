# 标注高效 RGB-T 方法状态（2026-04-16）

> 补充更新：2026-04-23（全监督对齐快照）

```text
MM-SAM official strict test: 66.10
Ours best strict test (unfreeze4+tpi, epoch_30): 64.35(remote) / 64.31(local)
Gap: 1.79
```

对应实例：

- `ssh -p 36176 root@connect.bjb2.seetacloud.com`：`unfreeze4+tpi e30`（最佳）
- `ssh -p 46181 root@connect.bjb2.seetacloud.com`：`unfreeze8+tpi e50`、`unfreeze4 no-tpi e30`
- `ssh -p 55633 root@connect.westc.seetacloud.com`：MM-SAM official 4090 复现

## 1. 当前问题定义

目标不是继续卷 FMB 全监督分数，而是在低标注 RGB-T 场景中证明：

```text
少量 dense mask + 大量 unlabeled RGB-T + 低成本 point prompt
可以通过跨模态可靠性估计和分歧纠错形成有效监督。
```

当前最稳题目：

```text
Cross-Modal Reliability and Disagreement Refinement for Label-Efficient RGB-T Semantic Segmentation
```

中文：

```text
面向标注高效 RGB-T 语义分割的跨模态可靠性估计与分歧细化
```

## 2. 模型骨架

```text
RGB → SAM2 Hiera-L frozen backbone + Bottleneck Adapter → rgb_feats
                                                             ↓
                                                       MMSA Fusion → main pred
                                                             ↑
Thermal → ConvNeXt-Tiny ImageNet-22K pretrained encoder → thm_feats

rgb_feats → RGB-only head     → pred_rgb
thm_feats → Thermal-only head → pred_thm

pred_main + pred_rgb + pred_thm + disagreement_map
        → Disagreement Refinement Branch
        → pred_refined
```

预训练状态：

- RGB encoder：SAM2.1 Hiera-L checkpoint；原始 Hiera 权重冻结，只训练 Bottleneck Adapter。
- Thermal encoder：ConvNeXt-Tiny `convnext_tiny.fb_in22k`，ImageNet-22K 预训练。

## 3. 点提示协议

点标注不是每图固定 N 点，而是：

```text
对每张无标注图：
  对每个出现类别，从该类 mask 内随机采 1 个点
  若该类像素数 < 8，跳过
```

r10 当前统计：

| 项 | 数值 |
|---|---:|
| 无标注图像 | 954 |
| 总点数 | 7671 |
| 平均点数/图 | 8.04 |
| 中位数 | 8 |
| 最少/最多 | 4 / 12 |

当前 `L_point` 是严格单像素交叉熵：

```text
point_logits = pred[:, :, y, x]
L_point = CE(point_logits, class_id)
```

默认权重：

```text
lambda_unsup = 1.0
lambda_point = 0.2
```

主线不启用 point diffusion。

## 4. 伪标签生成

当前策略是：

```text
confidence-gated, consensus-weighted pseudo-labeling
```

即：

```text
teacher confidence >= 0.80 → 允许进入伪标签
RGB/Thermal agreement → 调整像素权重
```

当前主线配置：

```text
pseudo_conf_thresh = 0.80
pseudo_agreement_mode = argmax
pseudo_agreement_policy = soft_weight
pseudo_agreement_floor = 0.5
```

因此它不是纯共识驱动；一致性只调权，不单独决定准入。

Teacher 机制：

```text
student: 参与反向传播
teacher: student 的 EMA（指数移动平均）副本
teacher 输出用于生成伪标签
```

当前补充实现：

- `train_label_efficient.py` 增加 `--val-model {student,teacher}`，允许用 EMA teacher 做验证和 best checkpoint 选择。
- `eval_label_eff.py` 增加 `--checkpoint-state {model,teacher}`，允许同一 checkpoint 分别测试 student 和 teacher。

已知诊断：

```text
r20 原 best checkpoint:
student test strict mIoU = 59.02
teacher test strict mIoU = 58.58
```

说明直接评测现有 checkpoint 的 teacher 不能解决 r20 倒挂；关键在于用 teacher 作为验证与 best checkpoint 选择对象。

teacher-based checkpoint selection 已完成：

```text
r20/e30/dref_prob/seed42/--val-model teacher
remote: ssh -p 38755 root@connect.bjb2.seetacloud.com
log: /root/autodl-tmp/logs/r20_dref_valteacher_e30_seed42.log
work_dir: /root/autodl-tmp/work_dirs/label_eff_r20_e30_sp_seed42_softw_agf_dref_prob_valteacher
```

结果：

```text
best val strict mIoU = 71.14
student/model test strict mIoU = 60.98
teacher test strict mIoU = 61.48
```

结论：`--val-model teacher` 能明显缓解 r20 倒挂。相比原 r20 student-val checkpoint 的 `59.02`，teacher-val + teacher state 提升到 `61.48`，说明 EMA teacher 的价值主要体现在 checkpoint selection（检查点选择）和时域平滑，而不是直接评测 student-selected checkpoint 的 teacher。

新增软一致性蒸馏（默认关闭）：

```text
--soft-consistency-weight
--soft-consistency-loss {kl,mse}
--soft-consistency-temperature
--soft-consistency-reliability {none,agreement}
--soft-consistency-agreement-mode {argmax,prob}
--soft-consistency-agreement-floor
```

形式：

```text
L_soft = KL(softmax(student / T), softmax(teacher / T))
```

可选用 teacher 的 RGB/Thermal agreement 作为像素级 reliability weight：

```text
w = floor + (1 - floor) * agreement(pred_rgb_teacher, pred_thm_teacher)
```

该设计目标是把 EMA teacher 从“伪标签生成器”升级为“概率分布教师”，并把跨模态可靠性叙事扩展到软监督。

新增 DGR（Disagreement-driven Geometric Refinement，分歧驱动几何细化）训练损失（默认关闭）：

```text
--dgr-loss-weight
--dgr-disagreement-mode {argmax,prob}
--dgr-feature-key {feat_rgb_enc,feat_rgb_raw_enc}
```

形式：

```text
disagreement = 1 - agreement(pred_rgb_teacher, pred_thm_teacher)
edge_sam = Sobel(normalize(feat_rgb_enc_teacher))
edge_pred = Sobel(softmax(pred_student_refined))
L_dgr = SmoothL1(edge_pred, edge_sam) * disagreement * edge_sam
```

该设计不让 SAM2 产生类别监督，只使用其低层特征边缘作为几何先验，并且只在 RGB/Thermal 分歧区域生效。

## 5. Disagreement Refinement

当前最强结构组件是预测层分歧细化：

```text
disagreement_map = 1 - Σ softmax(pred_rgb) * softmax(pred_thm)
refine_in = concat(pred_main, pred_rgb, pred_thm, disagreement_map)
delta = refine_head(refine_in)
pred_refined = pred_main + disagreement_map * delta
```

`refine_head`：

```text
3x3 Conv + BN + ReLU
3x3 Conv + BN + ReLU
1x1 Conv
```

它是 `prediction-level error correction`，不是 `feature-level re-alignment`。

## 6. 当前最佳结果

主口径：FMB test strict mIoU，`absent-score=0.0`。

| 配置 | val strict | test strict |
|---|---:|---:|
| dref prob w=0.5 bs=2+2 seed42 | 66.70 | 60.59 |
| dref prob w=0.5 bs=2+2 seed3407 | 65.79 | 59.88 |
| r20 dref prob `--val-model teacher` teacher state | 71.14 | 61.48 |
| r20 + EMA soft consistency w=0.2 teacher state | 67.04 | 57.09 |
| r20 + DGR w=0.05 teacher state | 65.78 | 60.81 |
| dref prob w=0.5 no-gate seed42 | 65.58 | 58.54 |
| naive fusion + same semi+point + dref prob | 64.10 | 55.79 |
| SAM2 point proxy seed42 | 67.70 | 58.48 |
| SAM2 point proxy w=0.1 | 66.98 | 59.70 |
| SAM2 point proxy w=0.1 + agreement gate | 66.31 | 59.08 |
| thermal-prior-injection seed42 | 65.25 | 60.53 |

当前可报告均值：

```text
seed42/3407 mean test strict mIoU ≈ 60.24
```

## 7. 已知负结果

| 方向 | 结论 |
|---|---|
| Dual-Branch Co-Training | 去融合后性能不足 |
| Point Diffusion | val 可涨但 test 不稳 |
| Consensus/Feature-gated Point Diffusion | 未稳定提升 |
| SAM2 point proxy region supervision | val 明显升高但 test 下降，放大 val-test gap |
| SAM2 point proxy w=0.1 | 从 58.48 恢复到 59.70，但仍低于主线 60.59 |
| SAM2 point proxy agreement gate | test 59.08，低于单纯 w=0.1；`proxy ∩ pseudo` 后 agreement gate 过滤强度不足 |
| Disagreement Refinement no-gate | test 58.54，证明空间 gate 是必要机制 |
| Naive Fusion baseline | test 55.79，低于主线 60.59，证明不是任意 RGB-T 融合都能得到收益 |
| EMA soft consistency w=0.2 | r20 test 57.09，显著低于 61.48，说明当前软蒸馏过度平滑或强化错误伪标签 |
| DGR w=0.05 | r20 test 60.81，低于 61.48；Bus 上升但 Truck 大幅下降，几何先验未形成稳定总收益 |
| Class-balanced Sampler | r20 test 下降 |
| MFNet external unlabeled | 域差异导致下降 |
| Thermal Prior Injection | 打平主线，无明确收益 |

## 8. 当前缺口

必须补的审稿证据：

1. `Naive Fusion + same training`。
2. `RGB-only / Thermal-only + same semi+point protocol`。
3. `MeanTeacher/UniMatch-style RGB-T baseline`。
4. Failure visualization。
5. Point protocol cost curve。

## 9. 新探索：SAM2-Guided Label Propagation

动机：

```text
当前 SAM2 只作为 frozen backbone 使用，point prompt 只计算单点 CE。
这没有利用 SAM2 原生 promptable segmentation（可提示分割）的点到掩码能力。
```

探索方案：

```text
每类 1 个 point prompt
  → 冻结原始 SAM2 生成 Local Proxy Mask
  → 与 teacher pseudo label 取类别一致交集
  → 作为区域级弱监督补充 L_point
```

当前实现状态：

- 新脚本：`scripts/generate_sam2_point_proxies.py`
- 训练入口新增：
  - `--point-proxy-index`
  - `--point-proxy-weight`
  - `--point-proxy-intersect-pseudo`
  - `--point-proxy-min-pixels`
- 当前远端实验：
  - `ssh -p 21824 root@connect.bjb2.seetacloud.com`
  - 先生成 `/root/autodl-tmp/sam2_point_proxy_r10_box96_a015/index.json`
  - 再训练 `/root/autodl-tmp/work_dirs/sam2_proxy_r10_e30_seed42`
  - 日志 `/root/autodl-tmp/logs/sam2_proxy_r10_e30_seed42.log`

实验结果：

```text
val strict mIoU = 67.70
test strict mIoU = 58.48
```

结论：暂不并入主线。该方案显著提高 val，但 test 低于主线 `60.59`，说明 SAM2 class-agnostic proxy mask
在 FMB 上可能引入语义粒度错配，并放大 checkpoint selection bias / val-test gap。

### 9.1 Proxy 减法重试

针对上述失败，新增两类去噪控制：

```text
--point-proxy-weight 0.1 / 0.2
--point-proxy-agreement-gate
--point-proxy-agreement-mode argmax
```

目的：

- 把 proxy 从强监督降级为 weak regional guidance（弱区域引导）。
- 只在 RGB/Thermal teacher heads 达成一致的区域使用 proxy loss。

当前运行：

| 实验 | 远端 | work_dir | 目的 |
|---|---|---|---|
| proxy w=0.1 | `ssh -p 21824 root@connect.bjb2.seetacloud.com` | `/root/autodl-tmp/work_dirs/sam2_proxy_w01_r10_e30_seed42` | 验证降权能否修复 test |
| proxy w=0.1 + agreement gate | `ssh -p 38755 root@connect.bjb2.seetacloud.com` | `/root/autodl-tmp/work_dirs/sam2_proxy_w01_agree_r10_e30_seed42` | 验证多模态去噪是否有效 |

结果：

```text
proxy w=0.1: val 66.98 / test 59.70
proxy w=0.1 + agreement gate: val 66.31 / test 59.08
```

结论：

- 降权能缓解原始 proxy 的 test 崩塌，但不能超过主线 `60.59`。
- agreement gate 未带来收益，原因是 `proxy ∩ pseudo` 之后剩余区域多数已经满足 RGB/Thermal argmax 一致，额外过滤强度有限。
- SAM2 proxy 暂不作为主线贡献；如继续探索，应优先尝试更强的空间减法，例如中心性衰减或更小 box，而不是继续提高 proxy 权重。

## 10. 写作红线

- 不说“解决跨模态配准”：当前没有 spatial transformer 或 offset prediction。
- 不说“特征重对齐”：当前 refinement 只看预测 logits。
- 不说“纯共识驱动”：当前仍由 confidence threshold 控制伪标签准入。
- 不说“极端单点监督”：当前是每出现类别 1 点，平均 8.04 点/图。
