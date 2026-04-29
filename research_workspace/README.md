# 研究工作区说明

`research_workspace/` 保存研究状态、长期决策、实验协议、交接信息和归档材料。

## 当前主线（2026-04-16）

当前唯一主线：

```text
标注高效 RGB-T 语义分割：
SAM2 先验 + 跨模态可靠性估计 + 分歧区域纠错 + 弱点提示
```

不再以“FMB 全监督融合模块刷分”为主目标。

## 当前入口

新会话只需要先读：

1. `notes/ACTIVE_CONTEXT.md`
2. `plans/current_plan.yaml`
3. `notes/DECISIONS.md`
4. `notes/SESSION_HANDOFF.md`
5. `../docs/LABEL_EFFICIENT_METHOD_STATUS_2026-04-16.md`

## 当前目录结构

- `notes/`
  - 当前 active context、决策、交接。
  - 旁支文件只保留索引或归档指针。
- `plans/`
  - 当前计划 `current_plan.yaml`。
  - 自动循环配置 `auto_loop_config.json`。
- `artifacts/`
  - label protocol、weak labels、实验 leaderboard、参考文献。
- `archive/`
  - 旧阶段、旧计划、旁支方向和刷新前文档。

## 文档治理规则

1. 当前入口必须少而准，不把旧想法混进 active docs。
2. 关键判断写入 `notes/DECISIONS.md`。
3. 当前运行状态写入 `notes/SESSION_HANDOFF.md`。
4. 实验协议、最佳配置和负结果写入 `docs/LABEL_EFFICIENT_METHOD_STATUS_2026-04-16.md`。
5. 旧文档不删除，统一归档到 `archive/`。

## 归档入口

- 最新刷新：`archive/2026-04-16_label_eff_refresh/`
- 3 月 30 日主线切换：`archive/legacy_2026-03-30_focus_shift/`
- 3 月旧项目材料：`../archive/legacy_2026-03-23/`
