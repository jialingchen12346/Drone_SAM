# 长周期研究工作流

本项目已经完成一次旧研究周期，相关材料已归档到：

- `archive/legacy_2026-03-23/`

当前阶段以“重新设计模型结构并启动新研究”为主，因此工作流也重置为更干净的中文版本。

## 当前保留内容

- `segmentation/`
  训练、评估、模型实现与基础框架。
- `sam2/`
  SAM2 相关代码与配置。
- `docs/`
  当前仅保留数据集说明和本工作流文档。
- `research_workspace/`
  新研究周期的状态管理目录。

## 新会话开始时应读取

1. `docs/RESEARCH_WORKFLOW.md`
2. `research_workspace/plans/current_plan.yaml`
3. `research_workspace/notes/SESSION_HANDOFF.md`
4. 与当前任务最相关的研究笔记

## 工作过程中应维护

- `research_workspace/notes/DECISIONS.md`
  记录应跨会话保留的关键判断。
- `research_workspace/notes/`
  保存调研、方案设计、实验解释等高价值文本。
- `research_workspace/plans/current_plan.yaml`
  维护当前目标、进行中任务和下一个动作。

## 本地检索与汇总工具

### 重建索引

```bash
python scripts/index_workspace.py build
```

### 搜索工作区

```bash
python scripts/index_workspace.py search "区域级可靠性融合"
python scripts/index_workspace.py search "细节语义解耦解码器"
python scripts/index_workspace.py search "模态 masking 鲁棒训练"
```

### 汇总实验结果

```bash
python scripts/summarize_experiments.py
```

输出位置：

- `research_workspace/experiments/summary/experiment_summary.csv`
- `research_workspace/experiments/summary/experiment_summary.md`

## 分阶段实验自动化（推荐）

当前默认策略为 `12/20/30` 三段式漏斗：

- `S1`：12 轮快筛（保留 Top-3）
- `S2`：20 轮复筛（保留 Top-2）
- `S3`：30 轮确认（保留 Top-1，strict 口径）

执行：

```bash
python scripts/staged_funnel_loop.py \
  --config research_workspace/plans/staged_funnel_strategy.json \
  --execute
```

预演：

```bash
python scripts/staged_funnel_loop.py \
  --config research_workspace/plans/staged_funnel_strategy.json \
  --dry-run
```

## 当前研究约束

1. 旧的 `68.62 > 66.10` 主张不再作为严格对比结论。
2. 新研究应首先统一评测口径，再启动重训。
3. 第一轮新模型尽量保留 encoder，不扩大重写范围。
4. 文档默认使用中文。
