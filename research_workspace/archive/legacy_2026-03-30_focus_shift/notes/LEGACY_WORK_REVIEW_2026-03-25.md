# 旧工作复盘与迁移建议（2026-03-25）

## 1. 已阅读范围

- `/home/jl/Drone-SAM-Adapter/archive/legacy_2026-03-23/docs/论文初稿.md`
- `/home/jl/Drone-SAM-Adapter/archive/legacy_2026-03-23/docs/论文草稿.md`
- `/home/jl/Drone-SAM-Adapter/archive/legacy_2026-03-23/docs/PLAN_v2.md`
- `/home/jl/Drone-SAM-Adapter/archive/legacy_2026-03-23/docs/ANALYSIS_v2.md`
- `/home/jl/Drone-SAM-Adapter/archive/legacy_2026-03-23/docs/SESSION_SNAPSHOT.md`
- `/home/jl/Drone-SAM-Adapter/archive/legacy_2026-03-23/docs/SOTA_TRACKER.md`
- `/home/jl/Drone-SAM-Adapter/archive/legacy_2026-03-23/docs/EVIDENCE_CHAIN.md`
- `/home/jl/Drone-SAM-Adapter/archive/legacy_2026-03-23/docs/RESOURCE_INDEX.md`

## 2. 旧工作中最有价值的资产

1. 叙事主线清晰：
- 条件感知融合（CACAF）解决“模态质量波动”
- 小目标导向解码（HGSOAD/SAGU）解决“俯视小目标丢失”

2. 工程与实验资产完整：
- 训练/评测脚本、消融开关、可视化链路、表格模板比较齐全
- 有较完整的证据链管理意识（history、ablation、SOTA tracker）

3. 写作素材成熟：
- `论文初稿.md` 的方法与实验章节结构可直接复用到新稿骨架
- 图表占位与“贡献-实验-讨论”映射关系已经搭好

## 3. 与当前主线的冲突点（必须剥离）

1. 指标口径冲突：
- 旧稿大量使用 `68.62` 时代结果（旧口径/旧环境语境）
- 当前主线已切为 strict（`--absent-score 0.0`），两者不能直接混排

2. 环境混杂：
- 旧文档含 4070Ti / 5090 / 不同 PyTorch 版本对照，容易引入“训练结果不可比”争议

3. 数据与投稿主张不一致风险：
- 旧稿中部分“跨数据集泛化”内容带有计划性质，不等同于已完成证据

## 4. 可立即迁移的规范

1. 论文写作规范：
- 保留“问题-方法-消融-可解释性-局限性”结构
- 保留 SOTA 证据链追踪方式（只写已核对来源）

2. 实验组织规范：
- 保留统一配置消融（同 epoch、同 loss、同 resize）思想
- 保留命名规范（论文表中禁用内部 run 名）

3. 可视化规范：
- 继续使用“代表样本定性图 + 融合权重统计图 + per-class 对比”

## 5. 当前阶段建议（基于旧工作但不被其束缚）

1. 把旧稿当“写作模板”，不要当“结果事实来源”：
- 所有数字以当前 strict 重评/新跑结果为准

2. 保留旧方法思想，但改成新一轮可验证假设：
- 单变量、同预算筛选 -> 晋级长训 -> 严格口径汇总

3. 下一版论文故事建议：
- 保留“条件感知 + 小目标恢复”的双问题框架
- 但方法名/模块名可升级，避免与已判退路线（V5）绑定

