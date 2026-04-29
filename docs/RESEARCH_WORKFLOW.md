# 长周期研究工作流

更新时间：2026-04-16

## 当前研究入口

新会话先读：

1. `research_workspace/notes/ACTIVE_CONTEXT.md`
2. `research_workspace/plans/current_plan.yaml`
3. `research_workspace/notes/DECISIONS.md`
4. `research_workspace/notes/SESSION_HANDOFF.md`
5. `docs/LABEL_EFFICIENT_METHOD_STATUS_2026-04-16.md`

## 当前研究目标

```text
面向标注高效 RGB-T 语义分割的跨模态可靠性估计与分歧细化
```

当前不再推进：

- FMB 全监督单点刷分。
- 复杂融合模块堆叠。
- B2 夜间搜救方向并线。
- point diffusion 作为主线贡献。

## 实验工作流

每个新实验必须记录：

- 数据集与 split。
- label ratio。
- point prompt 协议。
- seed。
- checkpoint 选择规则。
- strict mIoU。
- 是否启用 modality heads、agreement weighting、AGF、disagreement refinement。

## 当前主口径

- 主指标：FMB test strict mIoU。
- strict 设置：`--absent-score 0.0`。
- 当前主协议：r10。
- 当前最佳：`dref prob / w=0.5 / bs=2+2 / seed42`，test strict mIoU `60.59`。

## 当前优先实验

1. `no-disagreement-refine-gate` 消融。
2. `Naive Fusion + same training`。
3. `RGB-only / Thermal-only + same semi+point`。
4. 半监督 baseline：MeanTeacher / UniMatch-style。
5. Failure visualization。

## 归档规则

- 旧计划、旧 quick-screen JSON、旧 SOTA 快照统一进 `research_workspace/archive/`。
- 当前入口文档不保存长篇历史。
- 历史可追溯，但默认不参与当前决策。
