# 12/20/30 三段式漏斗策略（2026-03-24）

## 目标

- 在严格口径（`--absent-score 0.0`）下，提升 FMB `test mIoU`，并减少低价值长训。
- 建立可复用的“快筛 -> 复筛 -> 满训”自动流程，服务投稿窗口。

## 成功标准

- `S1(12轮)` 能在 6 个候选中稳定筛出 Top-3。
- `S2(20轮)` 将候选收敛到 Top-2。
- `S3(30轮)` 形成 Top-1 最终主模型，并产出 strict val/test 结果与可追溯日志。

## 执行脚本

- 配置：`research_workspace/plans/staged_funnel_strategy.json`
- 调度：`scripts/staged_funnel_loop.py`

执行方式：

```bash
cd /root/Drone-SAM-Adapter
/root/miniconda3/envs/torch211/bin/python scripts/staged_funnel_loop.py \
  --config research_workspace/plans/staged_funnel_strategy.json \
  --execute
```

预演：

```bash
cd /root/Drone-SAM-Adapter
/root/miniconda3/envs/torch211/bin/python scripts/staged_funnel_loop.py \
  --config research_workspace/plans/staged_funnel_strategy.json \
  --dry-run
```

## 阶段门控

- S1: `12 epoch`, 候选 6 条，保留 Top-3。
  - 止损线：`best test < 58.0` 直接停止，回到结构重设。
- S2: `20 epoch`, 保留 Top-2。
  - 止损线：`best test < 61.5` 停止继续长训。
- S3: `30 epoch`, 保留 Top-1。
  - 止损线：`best test < 63.0` 标记为“不可投稿主结果”，转备选结构。

## 资源与风控

- 统一使用数据盘：`/root/autodl-tmp/work_dirs/staged_funnel`。
- 每阶段独立 artifacts，便于回滚与复盘。
- 当前磁盘策略：保留关键 best/epoch_30 + 结果文本，历史 checkpoint 自动清理。

## 与当前主线关系

- 当前正在跑的 30 轮主线（`r5_ohem_only_bs8...`）继续作为锚点结果。
- 新漏斗策略用于后续轮次，避免再次出现“10轮分数过低但无法决策”的问题。

## 下一步

1. 待当前 30 轮锚点结束，补跑 strict test，写入同一比较表。
2. 启动 `staged_funnel_loop.py --execute`，跑完整个 S1->S3。
3. 把 S3 Top-1 作为论文主模型，S1/S2 作为消融与选型依据。
