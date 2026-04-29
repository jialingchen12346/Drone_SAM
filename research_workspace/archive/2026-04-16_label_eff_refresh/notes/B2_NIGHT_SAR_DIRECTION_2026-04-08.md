# B2 方向备忘：夜间搜救 UAV RGB-T（2026-04-08）

## 方向一句话

- 面向夜间搜救场景，利用无人机俯视 `RGB-T` 感知，将热红外用于“人目标发现”，将可见光用于“环境上下文解释”，并用任务效用而非通用 mIoU 重新定义“什么算好”。

## 为什么先暂存

1. 该方向的问题定义是成立的，但其核心贡献更偏：
   - 任务重定义
   - 任务导向评价体系
   - 人目标优先的非对称多模态建模
2. 这与当前主线 `标签效率（半监督 + 弱监督）` 不同，若现在并线，会导致：
   - 论文主问题发散
   - 实验矩阵显著膨胀
   - 指标和任务形式从 segmentation 向 detection/localization/SAR utility 偏移
3. 现有数据与结果更支持 `A 线`，`B2` 更适合作为后续独立项目。

## 当前保留的核心想法

- 模态角色非对称：
  - `IR`：人目标热显著性
  - `RGB`：地形、障碍、道路、水域、植被、建筑等环境上下文
- 评价重心从通用分割指标转向任务效用：
  - human recall
  - false alarm rate
  - small/remote target sensitivity
  - localization usefulness
  - context usefulness
  - robustness under low light / clutter / occlusion

## 后续重启条件

1. 获得更直接匹配夜间搜救的 RGB-T UAV 数据或可行标注协议。
2. 明确任务形式：`segmentation`、`detection`、`localization` 或 `ranking`。
3. 有单独论文窗口，不与当前 `A 线` 主包共享贡献主叙事。

## 当前关系

- 本方向已暂存。
- 当前主线继续以 `A 线：标签效率 + 公开遥感集迁移` 为唯一主线推进。
