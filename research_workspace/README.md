# 研究工作区说明

`research_workspace/` 是项目的长期研究层，用来保存研究路线、证据链、实验资产与投稿准备材料，避免会话切换造成上下文丢失。

## 目录结构

- `notes/`
  持久化笔记、会话交接、决策记录、调研备忘。
- `plans/`
  当前研究目标、里程碑、实验矩阵与执行计划。
- `artifacts/`
  检索产物、自动循环记录、对标表、图表草稿。
- `experiments/summary/`
  实验摘要与导出的汇总表。

## 2026-03-30 起的新主线

从“FMB 全监督卷分”切换到“标签效率导向”的遥感/无人机语义分割：

1. 主问题：半监督 + 弱监督（点标注/稀疏标注）下的多模态分割鲁棒性。
2. 主卖点：更少标注成本下保持稳定性能，而非堆模块追单点 mIoU。
3. 主实验场：FMB（短期主实验）+ 遥感数据集补充验证（中期）。
4. 主结论口径：论文主表继续使用 strict（`--absent-score 0.0`）。

## 当前原则

1. 文档统一使用中文，便于直接阅读和维护。
2. 旧研究材料不删除，统一归档在 `archive/legacy_2026-03-23/`。
3. 新研究优先做“问题重定义 + 评测协议重建 + 标签效率证据”，再做结构复杂化。
4. 每次关键判断必须写入 `notes/DECISIONS.md`。

## 建议工作流

1. 新会话开始先读取：
   - `research_workspace/notes/ACTIVE_CONTEXT.md`
   - `research_workspace/plans/current_plan.yaml`
   - `research_workspace/notes/DECISIONS.md`
   - `research_workspace/notes/SESSION_HANDOFF.md`
2. 研究推进中：
   - 长期判断写入 `notes/DECISIONS.md`
   - 阶段总结写入 `notes/`
   - 必要时重建本地索引
3. 实验结束后：
   - 运行 `python scripts/summarize_experiments.py`
   - 把结论回填到 `plans/` 与 `notes/`
4. 自动闭环运行：
   - 配置：`research_workspace/plans/auto_loop_config.json`
   - dry-run：`python scripts/auto_train_eval_loop.py --dry-run`
   - 执行：`python scripts/auto_train_eval_loop.py --execute --max-cycles 3`

## 旧文档归档

- 为提高信息密度，旧阶段文档已迁移至：
  - `research_workspace/archive/legacy_2026-03-30_focus_shift/`
