# SHIFNet 对标分析与叙事草案（2026-03-24）

## 1. 为什么 SHIFNet 分数领先（FMB 67.8）

基于论文 `2503.02581` 文本证据：

- 路线层面：采用 **SAM2 驱动** 的跨模态框架，天然带有 foundation model 预训练先验。
- 融合层面：提出 `SACF`（Semantic-Aware Cross-modal Fusion），并引入 **text guide** 做动态模态权重重标定，缓解 RGB 偏置。
- 解码层面：提出 `HPD`（Heterogeneous Prompting Decoder），通过语义增强与类别嵌入提高全局语义一致性。
- 消融证据（FMB，Table VI）：`Add 65.5 -> Concat 66.6 -> SACF w/o text 67.1 -> SACF 67.8`，文本引导在该框架下有显著增益。

## 2. 与我们工作的可比性

### 2.1 可比部分

- 同一任务：RGB-T 语义分割。
- 同一公开数据集：FMB。
- 同一主指标：mIoU（test）。

### 2.2 不完全可比部分（论文中必须主动声明）

- 预训练范式不同：SHIFNet 使用 SAM2 + 语言引导；我们当前主线是 RRF-DSD + 工程化泛化优化。
- 训练配方/算力预算可能不同（epoch、输入尺度、增强、显存与并行策略），会影响绝对分数。
- 对比表基线代际不同，SHIFNet 主要对比到 MMSFormer/U3M 一代方法，不等于与我们当前工程口径一一对齐。

结论：**可以对比，但必须写成“同任务结果对照 + 不同范式下的公平性声明”**，不要直接宣称“同条件全面超越/落后”。

## 3. 我们后续论文故事怎么讲（建议主线）

### 3.1 叙事定位

- 不走“最大模型堆参数”路线，而是走 **可复现、可部署、训练成本可控** 的泛化增强路线。
- 核心卖点：在有限算力与有限数据下，把 test mIoU 做到高位，并把 `val-test gap` 控到稳定区间。

### 3.2 建议标题方向（内部）

- 方向 A：`Generalization-First RGB-T Segmentation under Limited Compute`
- 方向 B：`Efficient Cross-Modal Segmentation with Thin-Structure Compensation`

### 3.3 贡献点建议（对应你当前资产）

1. 一个以泛化为目标的训练-评测闭环（含 split 显式评测、leaderboard、自动清理与监控）。
2. OHEM-only 主线在 FMB 上形成强基线（当前最佳 `test mIoU=66.36`）。
3. 针对薄结构/小目标类别回退，提出结构补偿分支（TSCB）并做类别级误差分析。
4. 给出“精度-吞吐-稳定性”三维报告（不仅报最高分，也报告 gap 与代价）。

## 4. 对 SHIFNet 的论文写法建议（避免被审稿人卡）

- 写法 1：`Compared with foundation-model-heavy methods (e.g., SHIFNet), we focus on a compute-efficient setting...`
- 写法 2：明确报告训练成本、参数量、吞吐（epoch 时间/GPU 小时），形成“性价比”证据。
- 写法 3：主张“补齐关键安全类（Pole/Traffic Light/Traffic Sign/Sidewalk）”而不只追总 mIoU。

## 5. 下一步实验（可直接执行）

1. 公平性补充：固定我们当前口径（split/test，absent class 显式）并补一组“统一 epoch/输入尺度”对照。
2. 结构主线：`OHEM-only + bs8` 作为主基线，叠加 TSCB 做单变量消融。
3. 结果汇报：同时输出总 mIoU、关键类均值、`|val-test gap|`、训练时间与显存占用。
4. 投稿目标：短期冲 `66.8+`，中期冲 `67.5+`。
