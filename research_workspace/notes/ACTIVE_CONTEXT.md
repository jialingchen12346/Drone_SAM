# 当前必读索引（Active Context）

更新时间：2026-05-25

## 先读这 4 个文件

1. `research_workspace/plans/current_plan.yaml`
2. `research_workspace/notes/DECISIONS.md`
3. `research_workspace/notes/SESSION_HANDOFF.md`
4. `docs/LABEL_EFFICIENT_METHOD_STATUS_2026-04-16.md`

## 当前主线一句话

- 以 `10% dense mask + 无标注 RGB-T + 每存在类别 1 个 point prompt` 为核心协议，研究 `SAM2 先验 + 跨模态可靠性估计 + 分歧区域纠错` 的标注高效 RGB-T 语义分割。

## 当前论文叙事

- 不再讲“更强 RGB-T 融合模块刷全监督分数”。
- 融合是基础设施；真正主张是：
  - `RGB-only` 与 `Thermal-only` 独立预测提供跨模态可靠性信号。
  - 伪标签由 teacher confidence 准入，并由 cross-modal agreement 加权。
  - disagreement map 触发 prediction-level residual correction。
  - 点标注作为低成本、高可信锚点，而不是 dense mask 的替代品。

## 当前最佳事实

- 主协议：FMB，strict mIoU（`absent-score=0.0`），r10。
- 当前 label-efficient 最佳：`dref prob / w=0.5 / bs=2+2 / seed42`，test strict mIoU `60.59`。
- 复现种子：seed3407 test strict mIoU `59.88`，均值约 `60.24`。
- 单模态对照已补齐：`RGB-only = 58.09`，`Thermal-only = 51.72`。
- 低成本点协议：unlabeled pool 954 张，7671 点，平均 `8.04` 点/图；每个出现类别随机采 1 点。

## 全监督对齐快照（新增）

- MM-SAM official checkpoint：strict test `66.10`（本地 4070TiS 与远端 4090 均复现）。
- 我们当前最优全监督：`unfreeze4+tpi / epoch_30`，strict test `64.35`（远端）/ `64.31`（本地重测）。
- `RRF + reliability-guided refine + EMA teacher` 的 `seed2026 = 62.69` 已核对为 `train_cacaf.py` 全监督实验，不属于 label-efficient 主线。
- 同口径差距：`1.79`。
- 实例映射：
  - `36176`：`unfreeze4+tpi e30`（最佳）
  - `46181`：`unfreeze8+tpi e50`、`unfreeze4 no-tpi e30`
  - `35877`：`rrf_dsd_relrefine_v2_ema_valteacher_seed2026`（全监督方向1探索）

## 当前最重要的待办

1. **方向A: 将 SHIFNet Encoder+SACF 作为冻结特征提取器，接入我们的 MMSAFusion + DRef pipeline**
2. 补外部半监督 baseline：`MeanTeacher/UniMatch-style + RGB-T` 同协议对照。
3. 做 failure visualization：输出 RGB、Thermal、GT、pred_main、pred_rgb、pred_thm、disagreement_map、pred_refined、error map。
4. 做点协议成本曲线：每类 1 点 / 2 点 / 每图固定点数。
5. 回收 `seed3407` 低点 failure case 的类别级诊断。

## 2026-05-25 新增：SHIFNet 深度分析与 Encoder 复用

### SHIFNet 官方权重实测
- 官方 fmb.pth: FMB strict test **67.79**（已确认是 strict，不是 nanmean）
- 比 MM-SAM 66.10 高 1.69，是目前 FMB 已知最高分

### SHIFNet 架构分析
- Hiera backbone (224M): 冻结，与 SAM2.1 逐字节一致
- Space_Adapter + MLP_Adapter ×48 blocks (17.86M): 训练，**FMB 数据集特化**
- Neck (0.55M): 冻结，跨数据集一致
- SACF 融合 (10.88M): 训练，文本引导 + CXBlocks
- HPD 解码器 (3.54M): 训练，依赖 LanguageBind 文本嵌入
- 总可训练: ~32.8M

### 推理时消融结果（官方权重，无训练）
| 实验 | FMB test strict |
|------|----------------:|
| Baseline | 67.79 |
| No Space_Adapter | 59.30 |
| No MLP_Adapter | 38.97 |
| No Adapter (all) | 27.47 |
| No Text (random embed) | 1.10 |

### All-MLP Head 实验（Encoder+SACF 冻结，0.53M MLP head 训练）
- **Best (epoch 9): 66.40**，超越论文 All-MLP 66.1 和 MM-SAM 66.10
- 零文本依赖，纯视觉 MLP head
- 证明 SHIFNet 的价值在 encoder+SACF 特征质量，不在 HPD 文本解码器

### 后续方向
- SHIFNet Encoder (Hiera+Adapter+Neck+SACF) 整体作为冻结特征提取器
- 接入我们的 MMSAFusion + SegFormerLiteHead + Disagreement Refinement
- 预期待再提升 1-2 点（67+），且完全无文本依赖

## 归档入口

- 本轮文档刷新归档：
  - `research_workspace/archive/2026-04-16_label_eff_refresh/`
- 旧焦点切换归档：
  - `research_workspace/archive/legacy_2026-03-30_focus_shift/`
- 3 月旧代码/文稿归档：
  - `archive/legacy_2026-03-23/`
