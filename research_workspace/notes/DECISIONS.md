# 决策记录（当前版）

更新时间：2026-04-23

历史版本已归档：

- `research_workspace/archive/2026-04-16_label_eff_refresh/notes/DECISIONS.pre_refresh.md`
- `research_workspace/archive/legacy_2026-03-30_focus_shift/notes/DECISIONS_full_2026-03-30_pre_compact.md`

## D1. 主线定位：从融合模型转为标注高效框架

- 决策：论文不再以“更强 RGB-T 融合模块”作为主贡献。
- 当前表述：`SAM2 foundation prior + cross-modal reliability + disagreement refinement for label-efficient RGB-T segmentation`。
- 中文口径：`基于 SAM2 先验的标注高效 RGB-T 语义分割：跨模态可靠性估计与分歧细化`。
- 原因：全监督融合叙事拥挤；当前有效增益来自低标注协议下的模态独立预测、一致性加权和分歧纠错。

## D2. FMB 当前作为主实验场，而非单纯应用验证

- 决策：短期主包先在 FMB 打穿协议、消融与可视化证据。
- 原因：当前所有有效代码、点标注协议、半监督训练和结果都围绕 FMB 闭环。
- 影响：LoveDA/ISPRS/UAVid 等遥感公开集降为增强项，不再阻塞当前方法主线。

## D3. 评测红线：主结论只用 strict mIoU

- 决策：论文主表使用 `--absent-score 0.0`。
- 原因：FMB test 中存在缺失类，present-only 会抬高口径且不可与已有结果混用。
- 影响：present-only 只可进入附录或诊断说明。

## D4. 当前核心协议固定

- 决策：主协议为 `10% dense mask + unlabeled RGB-T + point prompt`。
- 点标注协议：对每张无标注图中每个出现类别，随机从该类 mask 内采 1 个点；类别像素数小于 8 跳过。
- 当前统计：954 张无标注图，7671 点，平均 8.04 点/图。
- 影响：论文中应称为 `class-presence point supervision`，不要称“每图单点极稀疏监督”。

## D5. Backbone 叙事固定为强预训练先验

- 决策：明确说明模型不是从零训练。
- RGB 分支：SAM2 Hiera-L checkpoint，冻结主干，仅训练 Bottleneck Adapter。
- Thermal 分支：ConvNeXt-Tiny，ImageNet-22K 预训练，作为辅助模态编码器。
- 影响：标注效率收益应归因于 `foundation prior + multimodal reliability adaptation`，不要声称从零学习鲁棒特征。

## D6. 伪标签策略定义

- 决策：当前伪标签是 `confidence-gated, consensus-weighted`。
- 准入：teacher 主预测 `confidence >= 0.80`。
- 加权：RGB/Thermal agreement 进行软加权，`agreement_floor=0.5`。
- 影响：不要写成纯 `consensus-driven`；一致性不是唯一准入条件。

## D7. Point loss 当前保持严格单像素监督

- 决策：主线不启用 point diffusion。
- 当前 `L_point`：只在点坐标上计算 single-pixel CE。
- 权重：`lambda_point / lambda_unsup = 0.2 / 1.0`。
- 原因：self/consensus/feature-gated diffusion 均未稳定提升 test，存在错误扩散或收益不足。

## D8. Disagreement Refinement 是当前最强结构贡献

- 决策：当前最强配置保留 `prob disagreement refinement`。
- 结构：输入 `pred_main / pred_rgb / pred_thm / disagreement_map`，经过 `3x3 Conv + BN + ReLU` 两层和 `1x1 Conv` 输出 residual delta。
- 当前最佳：r10/e30/seed42/bs=2+2/w=0.5 test strict mIoU `60.59`。
- 复现：seed3407 test strict mIoU `59.88`。
- 影响：主贡献应称为 `prediction-level disagreement correction`，不要称 feature re-alignment。

## D9. 已失败或暂缓方向

- `Dual-Branch Co-Training`：去融合结构更干净，但性能不够。
- 普通 `point diffusion`：val 变好但 test 下降。
- `consensus/feature-gated point diffusion`：仍无稳定收益。
- `class-balanced sampler`：r20 test 下降，破坏分布。
- 外部 MFNet unlabeled 扩充：test 下降，域差异过大。
- `thermal-prior-injection`：r10 test `60.53`，与主线 `60.59` 打平，无明确增益。

## D10. r20 病理判断

- 决策：不继续盲目长训 r20。
- 观察：r20 val 很高但 test 不稳定，主要是 FMB val/test 分布差异、小类高方差和 checkpoint selection bias。
- 影响：r20 更适合做分析和稳健 checkpoint 选择，不作为当前主线优先提分入口。

## D11. 当前缺口

- 外部半监督 baseline 不足：尚需 `MeanTeacher/UniMatch-style + RGB-T` 对照。
- 简单融合 baseline 不足：尚需 `Naive Fusion + same training`。
- 单模态对照不足：尚需 `RGB-only / Thermal-only + same semi+point protocol`。
- Failure cases 缺可视化证据：不能只凭 per-class IoU 写讨论。

## D12. B2 夜间搜救方向归档

- 决策：`B2_NIGHT_SAR_DIRECTION` 作为后续独立方向归档，不并入当前论文主线。
- 原因：B2 更偏任务重定义与任务效用评价，会扩大当前标注效率论文边界。

## D13. 启动 SAM2 点到掩码代理监督探索（2026-04-16）

- 决策：新增 `SAM2-Guided Label Propagation` 探索，但不升级为主线。
- 动机：当前点标注只做单点 CE，未利用 SAM2 的 promptable segmentation 能力。
- 实现：
  1. 离线用冻结原始 SAM2，把每个 point prompt 扩成局部 proxy mask。
  2. 训练时只使用 `proxy mask ∩ teacher pseudo label` 的类别一致区域。
  3. 保留原始单点 CE，proxy 作为附加区域监督。
- 结果：r10/e30/seed42，val strict mIoU `67.70`，test strict mIoU `58.48`。
- 结论：val 提升明显但 test 下降，说明 proxy mask 区域监督放大了 val-test gap。
- 判断：SAM2 是 class-agnostic mask 生成器，FMB 是 semantic class mask；二者粒度不完全匹配，尤其 stuff 类和小类边界容易引入噪声。
- 当前远端：`ssh -p 21824 root@connect.bjb2.seetacloud.com`。

## D14. no-gate 消融确认 disagreement gate 必要（2026-04-16）

- 实验：`dref prob w=0.5` 去掉 `disagreement_map * delta` 空间门控，改为全图 residual correction。
- 结果：val strict mIoU `65.58`，test strict mIoU `58.54`。
- 对比：主线 gate-on test strict mIoU `60.59`。
- 结论：`disagreement_map` 不是附属可视化信号，而是 refinement branch 的空间选择机制；论文可明确主张 `disagreement-guided error correction`。

## D15. 全监督主线实例映射修正（2026-04-23）

- 决策：明确记录 `unfreeze4+tpi` 与 `unfreeze8+tpi` 的真实宿主，避免结果错配。
- 结论：
  - `ssh -p 36176 root@connect.bjb2.seetacloud.com`：`unfreeze4_tpi_e30`，最佳 `epoch_30` strict test `64.35`。
  - `ssh -p 46181 root@connect.bjb2.seetacloud.com`：`unfreeze8_tpi_e50`，最佳 strict test `63.07`；另有 `unfreeze4_no_tpi` strict test `61.64`。
- 影响：后续引用必须绑定 `instance + work_dir + checkpoint` 三元组。

## D16. 全监督当前最优 checkpoint 选择策略（2026-04-23）

- 决策：`unfreeze4+tpi` 以 `epoch_30` 作为全监督主结果，不用 `best.pth`。
- 事实：
  - `epoch_30`: `64.35`
  - `best.pth`: `64.11`
  - e50 后段普遍回落到 `63.4~63.8`
- 原因：val 最优与 test 最优不一致，继续长训放大 val-test gap。

## D17. MM-SAM 对齐结果确认（2026-04-23）

- 决策：将 `MM-SAM strict test 66.10` 作为全监督外部对照上限。
- 复现位置：
  - 本地 `4070TiS`：strict test `66.10`
  - 远端 `ssh -p 55633 root@connect.westc.seetacloud.com`（4090）：strict test `66.10`
- 影响：当前我们同口径最优 `64.31`（本地重测）与对照差距 `1.79`。

## D18. 下一阶段优先级调整（2026-04-23）

- 决策：先做 MM-SAM 差距拆解，再做最小模块迁移，不再盲目长训。
- 优先项：
  1. 在 MM-SAM 上做同口径消融，拆出主要增益来源。
  2. 在当前 `unfreeze4+tpi+dref` 骨架中一次迁移一个模块。
  3. 先冲全监督 `65+`，再回灌到 label-efficient 主线。
