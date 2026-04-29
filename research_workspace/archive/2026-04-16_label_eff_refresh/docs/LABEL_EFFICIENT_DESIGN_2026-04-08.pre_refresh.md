# 标签效率导向架构设计分析（2026-04-08）

本文档记录本次会话的核心分析与设计思考，供后续会话参考。

---

## 1. 当前模型结构

```
RGB  → SAM2-Hiera (冻结+BottleneckAdapter, ~2M 可训参数) → 4级特征 [144/288/576/1152 ch]
                                                                    ↓
                                                           Fusion (4级, 三选一)
                                                                    ↑
Thm  → ConvNeXt-Tiny (全量训练)                          → 4级特征 [96/192/384/768 ch]
                                                                    ↓
                                                           Decoder → [B, 14, H, W]
```

**三种融合模块**：
- `MMSAFusion`：轻量交互 + quality-weighted blend（adaptive token downsampling）
- `CACAF`：bidirectional cross-attention（跳过 f1 层）+ quality-aware weighting
- `RRF`：pixel-wise reliability map [B,2,H,W] + detail residual boost

**三种解码器**：
- `SegFormerLiteHead`：all-scale concat → projection
- `HGSOAD`：U-Net 式上采样 + 可选 SAGU channel attention
- `DSDHead`：Semantic/Detail/Edge 三分支 + GuidedMerge

---

## 2. 核心问题：模型与训练策略完全解耦

当前实际结构：

```
[全监督多模态融合模型]   ←── 完全不知道自己在用多少标注
        ↑
[label-efficient 训练策略]  ←── EMA伪标签、point loss、阈值过滤
```

所有标注效率增益（10% 预算下 Point +3.25，Semi +0.51）**全部来自训练策略**，模型结构本身对"只有 10% 标注"无感知。

审稿风险：融合贡献和 label-efficiency 贡献是独立的，审稿人会问为什么放在一篇论文里。

---

## 3. 叙事重定向（本次会话核心决策）

**旧叙事**：更好的 RGB-T 融合模块 → 全监督 SOTA

**新叙事**：多模态提供了一种在标注稀缺下**免费的可靠性信号**——两个模态的预测一致性比单模态置信度更能筛选伪标签。

```
旧角色：融合模块 → 更好特征 → 更高 mIoU（全监督）
新角色：双模态 → 跨模态一致性 → 更可靠伪标签 → 少标注也能训好
```

融合不消失，但从"贡献"变为"使能条件"。这在半监督分割文献里几乎没有从多模态角度讲过。

---

## 4. 设计方向分析

### 4.1 三个候选架构设计

| 设计 | 解决的问题 | 证据支撑 | 实现成本 |
|------|-----------|---------|---------|
| **双路预测头**（跨模态一致性） | semi 增益只有 +0.51，高置信错误预测 | 有（+0.51 明显偏小） | 低 |
| **原型记忆** | 精标注知识传播不足 | 无，猜测 | 中 |
| **Point 扩散**（亲和力传播） | point 信号太稀疏 | 无，point 已是主增益 | 高 |

### 4.2 收敛结论：只做一个

**当前只有一个问题有实验证据支撑**：semi 增益为什么只有 +0.51。

最直接假说：teacher 在高置信区域仍有错误预测，单模态 confidence+entropy 无法发现，但两个模态预测一致性可以。

**选择：双路预测头 + 跨模态一致性伪标签过滤**。其余两个设计没有当前数据显示其对应问题是瓶颈，推后。

### 4.3 GPT 建议（参考）

GPT 同样收敛到三个候选方案：
1. `Cross-Modal Agreement-Guided Semi-supervision`（最贴主线）
2. `Reliability-Aware Dense Fusion`（注：RRF 已有 pixel-level reliability map，升级到 class-aware 才算创新）
3. `Point-Supervised Fusion Gate`（新颖但实现成本高，point 信号传播到 gate 需要额外扩散机制）

---

## 5. 核心假说与验证路径

**H1**：双模态一致性能降低伪标签噪声
- 验证方式：`build_pseudo_targets()` 加入 agreement mask，对比 semi 增益是否从 +0.51 → +1.0+

**验证前置诊断（diag_pseudo_quality.py）**：
- 用 ablation forward（thermal 置零 / RGB 置零）获得两路独立预测
- 统计：agree 区域 vs disagree 区域在 GT 上的准确率差异
- **决策门槛**：
  - agree 区域准确率显著高于 disagree（差距 > 0.05）→ 假说成立，实现双路预测头
  - 差距不显著 → 问题在别处，不做双路预测头

---

## 6. 实验计划（2026-04-08 启动）

### 第一步：预算曲线趋势验证（30-epoch 快筛）

**目的**：确认 10% 预算的四路线排序在 5%/20% 下是否稳定。

| GPU | 预算 | 实验 | 配置 |
|-----|------|------|------|
| 0 | 5% | Sup-only / Semi-only / Point-only / Semi+Point(×2 seed) | 30 epoch，bs=4，warmup=250，bf16 |
| 1 | 20% | 同上 | 同上 |

work_dir 命名规则：`label_eff_r{ratio}_e30_{mode}_{seed}`

**决策门槛**：
- 趋势一致（Point > Sup，Semi 正增益）→ A 主线成立，进入完整训练
- 趋势不一致 → 先理解原因，暂停架构实验

### 第二步：伪标签质量诊断（同步运行）

脚本：`segmentation/diag_pseudo_quality.py`

输入：`label_eff_r10_bs4_w250_seed42/best.pth`（已训练的 10% Semi+Point teacher）

输出：
- 整体伪标签准确率（vs GT）
- confidence 分桶准确率（0.80-0.85 / 0.85-0.90 / 0.90-0.95 / 0.95-1.00）
- agree/disagree 区域准确率差异

结果保存：`/root/autodl-tmp/work_dirs/label_eff_r10_bs4_w250_seed42/diag_pseudo_quality.txt`

---

## 7. 远端资源（2026-04-08 当前实例）

- 机器：`root@connect.bjb2.seetacloud.com -p 31616`（两张 RTX 5090，各 32GB）
- 代码：`/root/Drone-SAM-Adapter`
- 数据：`/root/autodl-tmp/datasets/FMB`
- 弱标注：r5/r10/r20 均已生成（`research_workspace/artifacts/weak_labels/`）
- 新增脚本：
  - `segmentation/eval_label_eff.py`：label-efficient checkpoint 的测试集评测
  - `segmentation/diag_pseudo_quality.py`：伪标签质量诊断
  - `run_r5_gpu0.sh` / `run_r20_gpu1.sh`：30-epoch 扫描启动脚本

---

## 8. 当前运行状态（2026-04-08 ~20:00）

| GPU | 进程 | 状态 |
|-----|------|------|
| 0 | diag_pseudo_quality | 运行中（~30min），完成后自动接 r5 e30 sweep |
| 1 | r20 e30 sup_only_seed42 | 运行中（~34min/实验，共 5 个） |

预计完成：GPU1 约 3h 后，GPU0 约 4.5h 后（含 diag）。

---

## 9. 结果解读框架

拿到 30-epoch 结果后，检查以下四个问题：

1. **排序是否保持**：`Sup < Semi < Point < Semi+Point`（对 5% 和 20%）
2. **semi 增益是否稳定**：在 5%/20% 下是否仍然 < Point-Sup 的 15%
3. **diag agree gap**：agree 准确率 - disagree 准确率，是否 > 0.05
4. **绝对值**：30-epoch 数字比 120-epoch 低，但相对关系是判断依据

基于以上结果决定：是否启动完整 120-epoch 训练 + 是否实现跨模态一致性过滤。
