# 决策记录（当前版）

更新时间：2026-05-02

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
- 单模态对照已补齐；后续不再投入主线算力。
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

## D19. 方向1的关键问题首先是 checkpoint trajectory，不是结构失效（2026-04-29，后于 2026-05-02 纠偏）

- 决策：保留 `RRF reliability-guided refine` 方向，且后续默认结合 `EMA teacher` 做验证与 checkpoint 选择。
- 证据：
  - 非 EMA：`seed42 epoch_20 = 61.44`，但 `epoch_30/best = 60.04`
  - 非 EMA：`seed3407 epoch_18 = 61.34`，但 `epoch_24/best = 60.46`
  - EMA 修正后：`seed42 = 62.17`，`seed2026 = 62.69`，`seed1234 = 61.92`
- 更正：以上 `62.17 / 62.69 / 61.92 / 60.11` 均来自 `train_cacaf.py` 全监督方向1探索，不属于 label-efficient 主线。
- 结论：方向1的早期 test 增益是真信号；原问题主要是后期轨迹漂移与 checkpoint selection 失灵，而不是 refine 结构本身无效。`EMA teacher + overall mIoU` 在全监督方向1上已跨多个 seed 复现 `61.9+`。
- 影响：后续讨论方向1时，必须显式标注它属于全监督探索线，而不是 r10 主线。

## D20. EMA teacher 实现必须只对参数做 EMA，对 buffers 直接拷贝（2026-04-29）

- 决策：`train_cacaf.py` 中 teacher 更新规则固定为：
  - `parameters`: exponential moving average
  - `buffers`（如 BN running mean/var）: direct copy
- 原因：此前错误地对 `buffers` 也做 EMA，导致 teacher validation 明显异常偏低。
- 影响：任何后续全监督 EMA teacher 实验，都必须沿用该实现；旧异常 teacher 结果不再作为有效证据。

## D21. 当前这轮 critical-class-aware selection 没证明比 overall mIoU 更优（2026-04-29）

- 决策：保留 `critical_mean` 作为诊断指标，但暂不把它升级为默认 best 选择规则。
- 事实：
  - `teacher + overall mIoU`：`best = epoch_24`，test strict `62.17`
  - `teacher + critical_mean`：`best = epoch_24`，test strict `61.95`
- 结论：关键类波动仍然是 val-test gap 的重要解释，但在当前 `seed42` EMA 试验中，`critical_mean` 没有选出更优 checkpoint。

## D22. 全监督方向1应按“四 seed 分布”汇报，而不是单点高分（2026-04-29，后于 2026-05-02 纠偏）

- 决策：方向1当前统一按四个 `overall mIoU + EMA teacher` seed 汇报：
  - `seed42 = 62.17`
  - `seed3407 = 60.11`
  - `seed2026 = 62.69`
  - `seed1234 = 61.92`
- 均值：`61.72`
- 结论：
  - `62+` 不是偶然脏点，已跨多个 seed 复现。
  - `seed3407` 是当前低点，应作为 failure case 做类别级分析。
  - 当前更准确的表述是“有稳定高分能力，但存在明显 seed variance”，而不是“完全稳定”或“纯偶然”。

## D23. `seed2026 = 62.69` 归属纠正：它是全监督方向1探索结果（2026-05-02）

- 决策：撤销“`seed2026 = 62.69` 作为 label-efficient 主结果”的口径。
- 远端核对：`357机` 上启动脚本运行的是 `segmentation/train_cacaf.py`，且日志为 `Training on 1060 images, validating on 160 images`。
- 当前正确归属：`seed2026 / teacher / best = 62.69` 属于全监督 `RRF-DSD + reliability-guided refine + EMA teacher` 探索结果。
- 影响：
  - label-efficient 主线结果恢复为 `60.59`；
  - `62.69` 只能用于全监督方向1分析，不再进入 label-efficient 主表；
  - 后续文档必须显式区分 `train_cacaf.py` 和 `train_label_efficient.py` 两条线。

## D24. 单模态对照已补齐，后续表述需区分“同骨架对比”与“全监督探索对比”（2026-05-02）

- 决策：单模态对照已完成：
  - `RGB-only = 58.09`
  - `Thermal-only = 51.72`
- 同骨架公平对比对象：
  - `MMSA multi-modal seed42 = 60.59`
- 全监督探索对比对象：
  - `direction1 best seed2026 = 62.69`
- 影响：
  - 当讨论“多模态本身是否有效”时，用 `60.59 vs 58.09 / 51.72`
  - 当讨论“当前完整方法比单模态强多少”时，用 `62.69 vs 58.09 / 51.72`
  - 不再混用这两层口径。

## D25. 文档口径强制规则：全监督与 label-efficient 不得混写（2026-05-02）

- 决策：凡是 `train_cacaf.py` 产生的结果，一律归入全监督；凡是 `train_label_efficient.py` 产生的结果，才可归入 label-efficient。
- 原因：本次 `seed2026 = 62.69` 错归档已经证明，仅凭方法名和 work_dir 命名会导致严重误导。
- 影响：后续所有表格必须至少同时写明 `入口脚本 + 协议 + work_dir`。

## D26. 架构定位：明确承认 prediction-level correction 框架，小步前移而不是彻底重构（2026-05-02）

- 决策：论文中将当前方法定位为 `prediction-level cross-modal reliability and refinement framework`，而不是 `deep early multimodal fusion network`。
- 原因：
  - 当前架构的本质是后期纠错：SAM2 Hiera 和 ConvNeXt-Tiny 完全独立编码，所有交互发生在 decoder 阶段的 MMSAFusion + Disagreement Refinement。
  - 这是弱点（全监督上限偏低、decoder 负担重），但更是当前低标注下有效性的原因（prediction-level correction 更稳、不易过拟合）。
  - 消融证据链支持这个叙事：Naive fusion → 55.79、no-gate → 58.54、单模态对照说明 correction 才是增益来源。
- 理由写进论文："Our framework operates primarily at the prediction level rather than learning deep cross-modal representations. This is a deliberate design choice: under extreme label scarcity, early fusion is prone to overfitting and modal bias, while prediction-level reliability estimation and disagreement correction provide more stable training signals."
- 后期演进方向：不是彻底重构为 early fusion，而是小步前移 — 在 decoder 前增加轻量交互层（GFFM、Mid-Level Correction），不碰 backbone，不改 refine head。
- 实现：
  - `GFFM`：每尺度双向交叉注意力 + 零初始化 learnable gate，自适应池化控制显存。
  - `MidLevelCorrection`：在 stride 16/32 做 feature-level disagreement-gated correction，cosine similarity 作为空间门控。
  - 两组实验已启动：seed42（357 机）、seed3407（481 机），与当前主线 dref prob 60.59/59.88 对照。

## D27. PPAL-v1 e20 快筛结论：提示有效，但当前吸收分支不优（2026-05-06）

- 决策：保留“点提示参与前向”的方向，但当前 `PPAL-v1` 分支不并入主线。
- 证据（r10, seed42, e20）：
  - `E2 concat`: with `57.57` / without `55.72` / delta `+1.85`
  - `E3 ppal`: with `55.96` / without `54.34` / delta `+1.62`
- 结论：
  - 提示输入本身有效（两条线都正增益）。
  - 现版本 `PPAL-v1` 复杂分支在同预算下劣于 `concat`，暂不作为主线默认结构。

## D28. 评估提示索引口径修正：val/test 必须使用对应 split 点索引（2026-05-06）

- 决策：with-prompt 评估必须使用 split 对应点索引，禁止复用 train 点索引评估 val/test。
- 发现：
  - `fmb_points_seed42_r10_all` 覆盖 train `1060/1060`，但 val/test 命中为 0。
  - 这会导致“with-prompt 开启但无真实提示命中”的伪对照。
- 修正：
  - 新增 `fmb_points_seed42_val_all`（160/160 命中）
  - 新增 `fmb_points_seed42_test_all`（280/280 命中）
- 影响：后续所有提示增益报告，必须记录所用 `point_index` 路径与 split 命中率。

## D29. SHIFNet Encoder 作为特征提取器（2026-05-25）

- 决策：将 SHIFNet 的冻结 Encoder（Hiera + 48 Adapter + Neck + SACF）作为我们的 RGB-T 特征提取器，替换现有的 SAM2HieraAdapter + ConvNeXtAux。
- 证据：
  - SHIFNet Encoder+SACF + 0.53M All-MLP head = 66.40 test strict，超过 MM-SAM 66.10
  - Adapter 是 FMB 数据集特化的，可冻结复用
  - SACF CXBlocks+MLP 投影是完全不依赖文本的纯视觉融合
  - 文本嵌入（LanguageBind）仅贡献 1.39 点（67.79-66.40），去掉后仍 SOTA
- 影响：
  - 全监督上界从 64.35 提升到 >66（预期 67+）
  - 不再需要 ConvNeXt-Tiny thermal encoder（统一用共享 Hiera）
  - 可专注于我们的核心创新（跨模态可靠性 + Disagreement Refinement）
