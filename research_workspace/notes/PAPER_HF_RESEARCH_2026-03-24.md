# 论文与 Hugging Face 开源模型调研（2026-03-24）

## 1) 调研结论（面向当前项目）

- 近两年 RGB-T/RGB-X 分割趋势从“单纯双分支融合”转向：
  1. **融合-解耦协同**（兼顾模态互补与缺失鲁棒）
  2. **Foundation Model 迁移**（SAM2 + 语言/提示）
  3. **轻量实时化**（统一编码、减少冗余分支）
- 结合我们当前痛点（`Pole/Traffic Light/Traffic Sign/Sidewalk`），结构优化应优先加强：
  - 细结构补偿
  - 边界一致性
  - 小目标语义对齐

## 2) 关键论文（与本项目最相关）

1. CMX (2022): 跨模态 Transformer 融合范式，证明 RGB-X 融合有效。
2. Sigma (2024): Mamba 用于多模态分割，强调高效长程建模。
3. Open-RGBT (2024): 开放词汇 RGB-T 分割方向，提示语言语义可提升泛化。
4. SHIFNet / SAM2+语言引导 (2025): SAM2 在 RGB-T 的可迁移性与文本引导融合价值。
5. RTFDNet (2026): 强调“融合与模态鲁棒解耦联合建模”，面向传感器退化场景。

## 3) Hugging Face 可复用开源模型资产

- `facebook/sam2.1-hiera-large`：当前最直接的分割基础模型迁移候选。
- `facebook/mask2former-swin-large-cityscapes-semantic`：道路场景语义分割强基线。
- `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`：轻量高效 backbone 对照基线。

## 4) 优化模型设计（RRF-DSD V3）

### 4.1 结构主线

- 保留当前 `RRF + DSD` 主干（已验证性能与吞吐）
- 新增 **TSCB: Thin-Structure Compensation Branch**（已代码落地，可开关）
  - 输入：`detail_feat + edge_feat + semantic_s4`
  - 核心：多膨胀 depthwise 细结构补偿
  - 融合：`merged + alpha * thin_feat`

### 4.2 训练主线

- 主基线：`OHEM-only`
- 吞吐验证：`batch=12`
- 结构对照：
  1. `bs12 + TSCB(0.15)`
  2. `bs12 + TSCB(0.12) + detail_aux(0.08) + no_photo`

### 4.3 评估重点

- 主指标：`test mIoU`
- 关键类均值：`Pole / Traffic Light / Traffic Sign / Sidewalk`
- 稳定性：`|val-test gap| <= 2.5`

## 5) 当前落地状态

- 已运行：`bs12` 的 OHEM-only 轮次（进行中）
- 已落地：TSCB 可开关实现 + 自动循环支持 recipe 级 eval flags
- 已配置：后续两条 TSCB 结构实验 recipe（待当前轮次完成后自动可跑）
