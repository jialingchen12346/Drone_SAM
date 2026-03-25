# 第一版新模型模块规划

日期：2026-03-23

## 目标

在不改动现有 encoder 主体的前提下，为第一版新模型确定清晰的模块命名、文件路径和职责边界，使下一步可以直接进入代码实现。

## 建议命名

模型暂定名：

- `RRF-DSD`

含义：

- `RRF` = Region Reliability Fusion，区域级可靠性融合
- `DSD` = Detail Semantic Decoupling，细节语义解耦解码

## 文件规划

### 1. 融合模块

建议新增文件：

- `segmentation/models/fusion/rrf.py`

职责：

- 对 RGB / thermal 同尺度特征做通道对齐
- 执行轻量局部交互
- 预测空间可靠性图
- 输出融合后的多尺度特征

建议包含的类：

- `LocalInteractionBlock`
- `ReliabilityEstimator`
- `RRFBlock`
- `RRF`

## 2. 解码器

建议新增文件：

- `segmentation/models/decode_heads/dsd_head.py`

职责：

- 构建 detail branch
- 构建 semantic branch
- 在 stride 4 位置做 guided merge
- 输出主预测与可选辅助预测

建议包含的类：

- `DetailRefineBlock`
- `SemanticUpBlock`
- `GuidedMergeBlock`
- `DSDHead`

## 3. 整体组装

建议新增文件：

- `segmentation/models/segmentors/rrf_dsd_segmentor.py`

职责：

- 组装 RGB encoder、thermal encoder、RRF、DSDHead
- 管理 loss、预测输出和训练/推理路径

## 4. 训练入口

建议方式：

- 第一阶段继续复用 `segmentation/train_cacaf.py`
- 先最小代价改成更中性的训练入口，或者后续再重命名

建议：

- 当前不要立刻重写训练脚本
- 先保证新 `segmentor` 能接入现有数据和 loss 流程

## 第一版最小功能边界

### 融合模块最小版

- 每尺度输入两路特征
- 先 `1x1 conv` 对齐到统一维度
- 用 depthwise `3x3` 或轻量卷积做局部交互
- 用 `2-channel` spatial map 做位置级 softmax 融合
- 不上重型全局 cross-attention

### 解码器最小版

- 细节流：以 `f1` 为主，必要时接入 `f2`
- 语义流：`f4 -> f3 -> f2`
- 使用语义引导细节融合
- 保留一个主头，辅助头可以作为第二步再加

## 当前明确不做

- 不在第一版里加 text / language branch
- 不在第一版里改 encoder
- 不在第一版里做复杂频域模块

## 直接下一步

1. 创建 `rrf.py`
2. 创建 `dsd_head.py`
3. 创建 `rrf_dsd_segmentor.py`
4. 用当前训练脚本先把前向跑通
