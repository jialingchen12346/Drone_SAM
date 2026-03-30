# 长周期研究工作流

本项目已完成一次旧研究周期，旧材料统一归档在：

- `archive/legacy_2026-03-23/`

自 **2026-03-30** 起，研究主线升级为：

- `标签效率导向的遥感/无人机语义分割`
- 核心方向：`半监督 + 弱监督 + 模态鲁棒性`

## 当前保留内容

- `segmentation/`
  训练、评估、模型实现与基础框架。
- `sam2/`
  SAM/SAM2 相关代码与配置。
- `docs/`
  数据集与工作流文档。
- `research_workspace/`
  状态管理、计划、证据链与交接记录。

## 新会话开始时应读取

1. `research_workspace/notes/ACTIVE_CONTEXT.md`
2. `research_workspace/plans/current_plan.yaml`
3. `research_workspace/notes/SESSION_HANDOFF.md`
4. `research_workspace/notes/DECISIONS.md`

## 工作过程中应维护

- `research_workspace/notes/DECISIONS.md`
  记录应跨会话保留的关键判断。
- `research_workspace/notes/`
  保存调研、方案设计、实验解释、失败复盘。
- `research_workspace/plans/current_plan.yaml`
  维护目标、里程碑、任务与近期动作。

## 新主线执行框架

### 阶段 A：协议重建（必须先完成）

1. 统一评测口径（主表 strict，`--absent-score 0.0`）。
2. 固化 label ratio 划分（如 `1/2/5/10/20/50%`）。
3. 固化弱监督生成脚本（point/scribble）。

### 阶段 B：半监督主线

1. 训练范式：`L% 标注 + U% 无标注`。
2. 核心机制：伪标签筛选 + 可靠性加权 + 一致性训练。
3. 输出：label efficiency 曲线、稳定性统计、成本统计。

### 阶段 C：弱监督扩展

1. 监督信号：point/scribble（先点后线）。
2. 融合策略：弱监督损失与半监督伪标签协同。
3. 输出：弱监督增益、标注成本对比、失败模式分析。

### 阶段 D：跨数据集证据

1. 短期主场：FMB。
2. 中期扩展：LoveDA / ISPRS / UAVid（按资源逐步引入）。
3. 输出：跨域泛化与鲁棒性证据，支撑 BMVC/ACCV 稿件。

## 本地检索与汇总工具

### 重建索引

```bash
python scripts/index_workspace.py build
```

### 搜索工作区

```bash
python scripts/index_workspace.py search "半监督 遥感 伪标签"
python scripts/index_workspace.py search "弱监督 点标注 语义分割"
python scripts/index_workspace.py search "模态退化 鲁棒性"
```

### 汇总实验结果

```bash
python scripts/summarize_experiments.py
```

输出位置：

- `research_workspace/experiments/summary/experiment_summary.csv`
- `research_workspace/experiments/summary/experiment_summary.md`

## 自动化执行建议

在新主线仍可复用 `12/20/30` 漏斗思想，但门控指标改为：

1. `strict mIoU`（主指标）
2. `label efficiency`（相同 mIoU 所需标注比例）
3. `robustness drop`（退化条件下跌幅）

执行命令可继续使用：

```bash
python scripts/staged_funnel_loop.py --config research_workspace/plans/staged_funnel_strategy.json --dry-run
python scripts/staged_funnel_loop.py --config research_workspace/plans/staged_funnel_strategy.json --execute
```

## 当前研究约束

1. 不再以“纯 FMB 全监督单点提分”作为主目标。
2. 论文主结论只用 strict 口径；present-only 仅作补充说明。
3. 优先控制方法复杂度，避免再次走向模块堆叠。
4. 文档默认使用中文。

## 旧阶段归档

- 旧阶段文档统一放置于：
  - `research_workspace/archive/legacy_2026-03-30_focus_shift/`
