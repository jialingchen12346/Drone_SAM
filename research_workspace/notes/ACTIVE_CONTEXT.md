# 当前必读索引（Active Context）

更新时间：2026-04-23

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
- 当前最佳：`dref prob / w=0.5 / bs=2+2 / seed42`，test strict mIoU `60.59`。
- 复现种子：seed3407 test strict mIoU `59.88`，均值约 `60.24`。
- 低成本点协议：unlabeled pool 954 张，7671 点，平均 `8.04` 点/图；每个出现类别随机采 1 点。

## 全监督对齐快照（新增）

- MM-SAM official checkpoint：strict test `66.10`（本地 4070TiS 与远端 4090 均复现）。
- 我们当前最优全监督：`unfreeze4+tpi / epoch_30`，strict test `64.35`（远端）/ `64.31`（本地重测）。
- 同口径差距：`1.79`。
- 实例映射：
  - `36176`：`unfreeze4+tpi e30`（最佳）
  - `46181`：`unfreeze8+tpi e50`、`unfreeze4 no-tpi e30`

## 当前最重要的待办

1. 回收 `no-disagreement-refine-gate` 消融结果。
2. 补 `Naive Fusion + same training` 对照。
3. 补 `RGB-only / Thermal-only + same semi+point protocol` 对照。
4. 做 failure visualization：输出 RGB、Thermal、GT、pred_main、pred_rgb、pred_thm、disagreement_map、pred_refined、error map。
5. 点协议成本曲线：每类 1 点 / 2 点 / 每图固定点数。

## 归档入口

- 本轮文档刷新归档：
  - `research_workspace/archive/2026-04-16_label_eff_refresh/`
- 旧焦点切换归档：
  - `research_workspace/archive/legacy_2026-03-30_focus_shift/`
- 3 月旧代码/文稿归档：
  - `archive/legacy_2026-03-23/`
