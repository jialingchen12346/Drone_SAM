# 会话交接（当前版）

历史全量交接记录已归档：
`research_workspace/archive/legacy_2026-03-30_focus_shift/notes/SESSION_HANDOFF_full_2026-03-30_pre_compact.md`

## 当前目标

- 主线：标签效率导向语义分割（半监督 + 弱监督）。
- 策略：主实验迁移到遥感公开集（LoveDA 优先），FMB 作为应用有效性验证。
- 投稿：BMVC 主投，ACCV 备投增强。

## 当前状态

- 文档主线已重构：
  - `research_workspace/plans/current_plan.yaml`
  - `research_workspace/plans/LABEL_EFFICIENT_RS_PLAN_2026-03-30.md`
  - `research_workspace/plans/PUBLICATION_STRATEGY_2026-03-30.md`
  - `docs/RESEARCH_WORKFLOW.md`
  - `research_workspace/README.md`
- 口径红线有效：主表 strict（`--absent-score 0.0`）。
- 旧阶段文档已归档到：
  - `research_workspace/archive/legacy_2026-03-30_focus_shift/`

## 当前必做任务（按优先级）

1. `T31`：实现 label ratio 划分脚本（`1/2/5/10/20/50%`，固定随机种子）。
2. `T32`：实现 weak label 生成脚本（point 优先，后续再加 scribble）。
3. `T33`：落地半监督最小闭环（teacher-student + 伪标签刷新 + 一致性损失）。
4. `T34`：加入可靠性筛选（模态质量/不确定性权重）。

## 本轮验收门槛

- 协议稳定性：同配置复跑波动不超过 `±0.3 mIoU`。
- 半监督收益：10% 标注预算下相对监督基线 `+1.0 mIoU` 或关键类均值 `+1.5`。
- 止损规则：连续 3 组候选无正增益即回滚复杂模块。

## 风险与应对

- 风险：伪标签噪声累积。
  - 应对：动态阈值 + 可靠性过滤 + EMA 慢更新。
- 风险：弱监督增益不足。
  - 应对：先固定 point，再评估是否扩展 scribble。
- 风险：跨数据集投入过大。
  - 应对：先落地 1 个遥感公开集可复现结果。

## 交接后立即执行命令（建议）

```bash
# 1) 阅读当前计划
sed -n '1,260p' research_workspace/plans/current_plan.yaml

# 2) 建索引并检索当前主线关键词
python scripts/index_workspace.py build
python scripts/index_workspace.py search "label ratio 半监督"
python scripts/index_workspace.py search "weak label point"

# 3) 开始 T31/T32 实现（先建脚本骨架）
# scripts/prepare_label_splits.py
# scripts/generate_point_labels.py
```
