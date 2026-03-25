# 模型结构重研（2026-03-24）

## 1. 触发原因

当前最优实验 `gen_ohem_only_cmr060_aux010_bs8_r1` 已把 `test mIoU` 提到 `66.36`，但类别级表现仍存在结构性短板：

- 回退类别：`Pole (-2.4)`、`Traffic Sign (-2.0)`、`Traffic Light (-1.9)`、`Sidewalk (-1.5)`
- 提升主要来自大目标与车辆相关类别（`Bus +19.5`）

这说明现有 `RRF + DSDHead` 在“中大目标语义融合”上有效，但对“细长结构 + 小亮目标 + 边界带状区域”仍有表达瓶颈。

## 2. 根因假设

1. **细结构特征在 stride-4 融合后被语义分支压制**
- `DSDHead` 虽有 detail 分支，但最终融合仍以通道门控为主，细长目标在语义一致性约束下容易被平滑。

2. **边界先验是局部增强，缺少跨尺度一致性约束**
- 现有 `edge_prior` 偏局部纹理强化，无法稳定对齐 `f1/f2/semantic_s4` 的几何一致性。

3. **小亮目标（Traffic Light/Sign）缺少高频-上下文协同分支**
- 只靠 `small_obj_enhancer`（f2_up + semantic_s4）不足以抵御背景噪声与亮点混淆。

## 3. 新结构方向（V3 原型）

### 方向 A：Thin-Structure Compensation Branch (TSCB)

目标：专门补偿 `Pole/Sidewalk edge/Traffic Light` 的细结构细节。

设计：
- 输入：`detail_feat`, `edge_feat`, `semantic_s4`
- 结构：`1x1降维 -> depthwise dilated conv(k=3, d=2/3) -> 1x1恢复`
- 融合：`merged = merged + alpha * thin_feat`，`alpha` 可学习

预期收益：
- 提升细结构连通性，减少柱状目标断裂和边界吞噬。

### 方向 B：Class-Aware Boundary Reweighting (CBR)

目标：抑制“背景边界过强导致的小目标误分”。

设计：
- 在 loss 端引入轻量类别重加权（只对 `TrafficLight/Sign/Pole/Sidewalk`）
- 与 OHEM 联合，避免仅优化易类像素

预期收益：
- 提升小目标召回而不显著拉低整体稳定性。

### 方向 C：Dual-Resolution Logit Refinement (DRLR)

目标：在输出端保留高频细节，不依赖主干大改。

设计：
- 主 logit（stride-4） + 细节补偿 logit（thin branch）
- 最终 `logit = main_logit + beta * thin_logit`

预期收益：
- 用最小代价回补边界细节类指标。

## 4. 执行优先级

优先级 1（本周落地）：方向 A（TSCB）
- 改动小、工程风险低、直接对齐当前痛点类别。

优先级 2（A 稳定后）：方向 C（DRLR）
- 在不重写 backbone/fusion 的前提下继续挖掘细节上限。

优先级 3（如 A/C 收益不足）：方向 B（CBR）
- 作为训练策略兜底，避免结构收益被损失函数稀释。

## 5. 验证计划

固定条件：
- 基线：`OHEM-only + batch=8/12`
- 数据口径：FMB `split=test` + absent-class 显式标注

判据：
1. 主指标：`test mIoU` 是否超过 `66.36`
2. 关键类：`Pole/Traffic Light/Traffic Sign/Sidewalk` 平均 IoU 提升 >= `+1.0`
3. 稳定性：`val-test gap` 绝对值 <= `2.5`
4. 工程成本：训练吞吐下降不超过 `15%`

## 6. 风险

- 细节分支过强可能导致噪声放大，出现边缘伪阳性。
- batch 从 8 提到 12 后，若显存/通信压力过大，可能引入不稳定梯度。

## 7. 当前动作

- 已启动 `batch=12` 的 OHEM-only 轮次验证吞吐与精度上限。
- 下一步将实现 `TSCB` 可开关原型并纳入自动循环配置做 A/B 对照。
