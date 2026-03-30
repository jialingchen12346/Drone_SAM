# FMB 方向 SOTA 快照（2026-03-25）

## 1) 研究范围与口径

- 任务：RGB-T 语义分割（FMB 数据集）。
- 本文档目标：给出“当前可核验”的 SOTA 参考，不混入未核来源。
- 口径提醒：不同论文对缺失类（如 FMB test 的 `Bicycle`）处理可能不同；与我们 strict(`absent=0`) 对比时必须二次对齐。

## 2) 你指定论文（2503.02581）要点

论文：`Unveiling the Potential of Segment Anything Model 2 for RGB-Thermal Semantic Segmentation with Language Guidance`（SHIFNet）

核心设计（按文中描述）：
1. `SACF`：语言引导的跨模态动态加权融合（缓解 SAM2 的 RGB 偏置）。
2. `HPD`：异构提示解码器（带语义增强与类别嵌入）。
3. 训练参数量：约 `32.27M`（可训练部分）。

文中声明结果：
- FMB `mIoU=67.8`
- PST900 `mIoU=89.8`
- MFNet `mIoU=59.2`

> 证据来源：本地论文文本 `artifacts/refs/papers_fmb/2503.02581.txt`（Table III 与摘要段）。

## 3) 当前可核验对标表（FMB）

| 方法 | FMB mIoU | 证据状态 | 备注 |
|---|---:|---|---|
| SHIFNet (arXiv:2503.02581) | 67.8 | 已核（论文摘要+表格文本） | IROS 2025 接收；已公开 GitHub 仓库 |
| MM SAM-Adapter | 66.10 | 已核（官方 README） | 我们当前已本地复现到该量级 |
| TUNI (arXiv:2509.10005) | 62.4 | 已核（论文文本表格片段） | 主打轻量实时，不是最高精度 |
| SARTM (arXiv:2505.01950) | 61.57 | 已核（论文文本表格片段） | 论文主打“Segment Any + distillation” |

## 4) 对我们最关键的结论

1. 以“公开可核”结果看，当前高位参考仍是 `SHIFNet 67.8`（FMB）。
2. 我们后续论文若要强对标，优先比较：
   - 总 mIoU（统一 strict 口径后）
   - 小目标关键类（Traffic Light / Sign / Pole / Motorcycle）
   - 参数量与训练/推理成本
3. 在未统一缺失类处理之前，不能把任何外部 mIoU 直接当 strict 对照。

## 5) 下一步执行建议（直接可跑）

1. 先做“统一评测口径映射”：
   - 对 SHIFNet / MM-SAM-Adapter 的公开结果建立 `present-only -> strict(absent=0)` 对照说明（可估算+可复现两版）。
2. 启动 `mmsa_baseline` 的单变量快筛（保持我们 strict 流程），先冲稳定 `test`。
3. 形成“对标表 v1”：
   - 一列外部原始口径
   - 一列我们 strict 口径
   - 一列可比性备注（是否同 split / 同类别集合 / 同 resize）。

## 6) 主要参考链接

- SHIFNet 论文：`https://arxiv.org/abs/2503.02581`
- SHIFNet 代码：`https://github.com/iAsakiT3T/SHIFNet`
- MM SAM-Adapter 代码与结果：`https://github.com/mug-ml/Multimodal-SAM-Adapter`
- TUNI 论文：`https://arxiv.org/abs/2509.10005`
- SARTM 论文：`https://arxiv.org/abs/2505.01950`
