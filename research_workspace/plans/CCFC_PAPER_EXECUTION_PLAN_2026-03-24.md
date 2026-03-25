# CCF-C 投稿执行计划（2026-03-24）

## 目标定义

- 总目标：完成一篇可投稿 CCF-C 会议的 RGB-T 语义分割论文。
- 技术目标：在 FMB test 上将 `mIoU` 从当前 `65.24` 稳定提升到 `>=67.0`，并将 `val-test gap` 控制在 `<=4.0`。
- 工程目标：建立可复用自动闭环（训练→评测→反馈→下一轮配置），实验可追溯且可复现。

## 阶段里程碑

1. 自动化基础完成（D0-D2）
- 完成 `scripts/auto_train_eval_loop.py` 与配置模板。
- 完成 `scripts/skill_router.py` 技能路由。
- 产出 `history.jsonl + leaderboard.md`。

2. 泛化提升主实验（D2-D10）
- 至少执行 3 轮自动实验。
- 每轮输出 val/test 双 split 指标与 ckpt 选择结果。
- 固化 1 套主配置 + 2 套有效消融配置。

3. 论文实验包补全（D10-D16）
- 补齐对比基线、消融表、类别级误差分析图。
- 明确 absent class 处理口径与评测协议。
- 形成可复现实验说明（环境/命令/日志路径）。

4. 初稿与迭代（D16-D24）
- 完成方法、实验、局限、附录。
- 进行 1 轮内部审阅并修订。

## 自动化闭环规范

1. 训练
- 统一通过 `scripts/run_ddp_train.sh` 启动，默认 `torch211 + NCCL`。
- 每轮实验独立 `work_dir`，禁止复用目录覆盖结果。

2. 评测
- 统一 `segmentation/test_rrf_dsd.py --split {val,test}`。
- 默认比较 `best.pth` 与 `latest epoch`，按 test mIoU 选主 checkpoint。

3. 反馈
- 计算 `val-test gap`，若 `gap >= 5.0` 则优先强泛化 recipe。
- `score = test_mIoU - 0.25 * max(0, gap)` 作为自动排序依据。

4. 追踪
- 历史：`research_workspace/artifacts/auto_loop/history.jsonl`
- 排行：`research_workspace/artifacts/auto_loop/leaderboard.{csv,md}`

## 技能自动匹配策略

- `planning`：`project-planner`, `sprint-planner`, `filesystem-context`
- `training`：`computer-vision-expert`, `hugging-face-vision-trainer`
- `evaluation/analysis`：`data-analyst`, `visualization-expert`, `fact-checker`
- `paper`：`academic-researcher`, `citation-management`, `deep-research`

命令示例：

```bash
python scripts/skill_router.py --stage analysis --val-miou 70.6 --test-miou 65.2 --json
```

## 风险与控制

- 风险：系统盘再次不足导致 checkpoint 保存失败。
- 控制：每轮实验前后检查 `df -h /`；只保留必要 checkpoint。

- 风险：val-test gap 持续大于 5。
- 控制：优先执行泛化导向 recipe（采样、增强、aux 权重）并保留强消融。

- 风险：实验口径漂移。
- 控制：固定 split、resize mode、absent class 处理并写入表格脚注。
