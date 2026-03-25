# 研究工作区说明

`research_workspace/` 是当前项目的轻量研究层，用来保存长期任务的计划、笔记、检索产物和实验摘要，避免每次新对话都重新恢复全部上下文。

## 目录结构

- `notes/`
  持久化笔记、会话交接、决策记录、调研备忘。
- `plans/`
  当前研究目标、里程碑和短期任务。
- `artifacts/`
  中间产物，例如本地索引、检索结果、表格草稿。
- `experiments/summary/`
  实验摘要和导出的汇总表。
- `artifacts/auto_loop/`
  自动闭环运行记录（`history.jsonl`、`leaderboard.*`、日志）。

## 当前原则

1. 文档统一使用中文，便于直接阅读和维护。
2. 旧研究材料不直接删除，统一放入 `archive/legacy_2026-03-23/`。
3. 新一轮研究优先保留代码主干，创新集中在融合模块、解码器和训练策略。
4. 每次完成重要判断后，都要写入 `notes/DECISIONS.md`。

## 建议工作流

1. 开始新会话时，先读取：
   - `docs/RESEARCH_WORKFLOW.md`
   - `research_workspace/plans/current_plan.yaml`
   - `research_workspace/notes/SESSION_HANDOFF.md`
   - 当前最相关的调研笔记
2. 过程中：
   - 将长期有效的判断写入 `notes/DECISIONS.md`
   - 将阶段性总结写入 `notes/`
   - 必要时重建本地索引
3. 有新实验结果后：
   - 运行 `python scripts/summarize_experiments.py`
   - 将结论回填到 `plans/` 和 `notes/`
4. 使用自动闭环时：
   - 配置：`research_workspace/plans/auto_loop_config.json`
   - dry-run：`python scripts/auto_train_eval_loop.py --dry-run`
   - 执行：`python scripts/auto_train_eval_loop.py --execute --max-cycles 3`
5. 使用三段式漏斗策略时（推荐新主线）：
   - 配置：`research_workspace/plans/staged_funnel_strategy.json`
   - dry-run：`python scripts/staged_funnel_loop.py --config research_workspace/plans/staged_funnel_strategy.json --dry-run`
   - 执行：`python scripts/staged_funnel_loop.py --config research_workspace/plans/staged_funnel_strategy.json --execute`
