# 证据链补全计划（v5 消融重做版）
> 创建时间: 2026-03-05  
> 最近更新: 2026-03-11  
> 目标: 从"模型跑通"到"论文/报告可用的完整证据体系"

---

## 一、证据链总览

论文中需要证明三件事：
1. **整体方法有效**（已完成：v5 Test mIoU=67.87）
2. **每个模块都有贡献**（正在重做：统一到 v5 训练策略）
3. **结果可信、可复现**（持续整理中）

---

## 二、当前状态（截至 2026-03-11）

1. 主实验最佳检查点：`work_dirs/cacaf_fmb_busfix_v1/best.pth`（epoch_40）。
2. 历史消融（v3 体系）已完成，但 epoch 不完全一致（A1/A3=30ep，A2/A4=50ep）。
3. 本轮目标是按 v5 协议重做消融，避免公平性争议并与最终主结果对齐。

---

## 三、模块 A — v5 消融实验（最高优先级）

### A.1 统一协议（所有实验严格一致）

1. 训练脚本：`segmentation/train_cacaf.py`
2. 数据：`/home/jl/dataset/FMB`
3. 训练设置：`epochs=60, batch_size=4, bf16`
4. 增强/预处理：`eval_resize_mode=letterbox, cat_max_ratio=0.75, blur_prob=0.2, photo_distort=True`
5. 评估口径：主表报告 `epoch_60`；`best checkpoint` 作为补充报告。

### A.2 实验矩阵（v5 版）

| 实验ID | 配置描述 | 关键开关 | work_dir | 状态 | 结果(mIoU/mAcc/aAcc) |
|------|------|------|------|------|------|
| V5-Full | 完整模型（CACAF + SAGU + Dice + OHEM） | `--use-dice --use-ohem` | `work_dirs/v5_ablation_full` | [ ] 待执行 | smoke test 已通过（1ep） |
| V5-NoCACAF | 去 CACAF | `--no-cacaf --use-dice --use-ohem` | `work_dirs/v5_ablation_no_cacaf` | [ ] | 待更新 |
| V5-NoSAGU | 去 SAGU | `--no-sagu --use-dice --use-ohem` | `work_dirs/v5_ablation_no_sagu` | [ ] | 待更新 |
| V5-NoDice | 去 Dice | `--use-ohem` | `work_dirs/v5_ablation_no_dice` | [ ] | 待更新 |
| V5-NoOHEM | 去 OHEM | `--use-dice` | `work_dirs/v5_ablation_no_ohem` | [ ] | 待更新 |
| V5-NoCACAF-NoSAGU | 去 CACAF + 去 SAGU | `--no-cacaf --no-sagu --use-dice --use-ohem` | `work_dirs/v5_ablation_no_cacaf_no_sagu` | [ ] | 待更新 |

### A.2.1 六组最终结果（自动更新）

<!-- V5_ABLATION_RESULTS_BEGIN -->
| 实验ID | Checkpoint | mIoU | mAcc | aAcc |
|---|---|---:|---:|---:|
| V5-Full | epoch_60.pth | 66.79 | 72.89 | 93.46 |
| V5-NoCACAF | epoch_60.pth | 64.71 | 70.53 | 93.45 |
| V5-NoSAGU | epoch_60.pth | 64.67 | 70.57 | 93.27 |
| V5-NoDice | epoch_60.pth | 67.78 | 74.69 | 93.58 |
| V5-NoOHEM | epoch_60.pth | 68.62 | 75.75 | 93.27 |
| V5-NoCACAF-NoSAGU | 训练中 | - | - | - |
<!-- V5_ABLATION_RESULTS_END -->

### A.3 执行顺序

1. `V5-Full`（先确认复现稳定）
2. `V5-NoCACAF`
3. `V5-NoSAGU`
4. `V5-NoDice`
5. `V5-NoOHEM`
6. `V5-NoCACAF-NoSAGU`

### A.4 训练命令（直接可跑）

```bash
cd /home/jl/Drone-SAM-Adapter && conda activate sam2-unet

BASE="python segmentation/train_cacaf.py \
  --data-root /home/jl/dataset/FMB \
  --batch-size 4 --epochs 60 --bf16 \
  --eval-resize-mode letterbox \
  --cat-max-ratio 0.75 \
  --blur-prob 0.2 \
  --photo-distort"

# 1) Full
$BASE --work-dir work_dirs/v5_ablation_full --use-dice --use-ohem

# 2) w/o CACAF
$BASE --work-dir work_dirs/v5_ablation_no_cacaf --use-dice --use-ohem --no-cacaf

# 3) w/o SAGU
$BASE --work-dir work_dirs/v5_ablation_no_sagu --use-dice --use-ohem --no-sagu

# 4) w/o Dice
$BASE --work-dir work_dirs/v5_ablation_no_dice --use-ohem

# 5) w/o OHEM
$BASE --work-dir work_dirs/v5_ablation_no_ohem --use-dice

# 6) w/o CACAF + w/o SAGU
$BASE --work-dir work_dirs/v5_ablation_no_cacaf_no_sagu --use-dice --use-ohem --no-cacaf --no-sagu
```

### A.5 结果同步规范

1. 每个实验至少同步：`epoch_60` 的 `mIoU/mAcc/aAcc`。
2. 若另有 `best.pth`，补充记录对应 epoch 与指标。
3. 同步位置：
   - 本文件 A.2 表格（状态 + 指标）
   - `docs/SESSION_SNAPSHOT.md`（新增“v5 消融重做进度”）
4. 论文主表优先使用统一口径（epoch_60），避免 test-driven checkpoint selection 争议。

---

## 四、模块 B — 可视化分析（已完成）

| 内容 | 实现方式 | 状态 |
|------|----------|------|
| B1 | 分割结果对比图（RGB/Thermal/GT/Pred） | `scripts/visualize_predictions.py` | [x] |
| B2 | 失败案例分析（Person/Bicycle） | `scripts/analyze_failure_cases.py` | [x] |
| B3 | CACAF 融合权重可视化 | `scripts/visualize_fusion_weights.py` | [x] |
| B4 | 14 类 IoU 对比图 | `scripts/plot_iou_comparison.py` | [x] |

---

## 五、模块 C — 实验设置文档（进行中）

| 内容 | 状态 |
|------|------|
| C1 | 训练超参数表 | [x] |
| C2 | 增强策略文档 | [x] |
| C3 | 参数量统计表（按模块） | [ ] |
| C4 | 推理速度 / FLOPs（可选） | [ ] |

---

## 六、模块 D — 论文表格（待整合）

| 表格 | 内容 | 状态 |
|------|------|------|
| D1 | 主实验对比表（方法 / mIoU / mAcc / aAcc） | [ ] |
| D2 | 各类别 IoU 详细对比表（14类） | [ ] |
| D3 | v5 消融汇总表 | [ ] |

---

## 七、进度跟踪

| 日期 | 完成内容 |
|------|----------|
| 2026-03-05 | 创建证据链计划文档；B1/B2/B3/B4 可视化链路完成 |
| 2026-03-06 | 历史消融（v3 体系）完成并汇总（用于早期论文草稿） |
| 2026-03-10 | v5 训练策略升级（OHEM + letterbox + 增强项）并完成 Test 评估 |
| 2026-03-11 | 启动 v5 消融重做计划：重建实验矩阵、统一协议、更新执行命令与结果同步规范 |
| 2026-03-11 | V5-Full 已启动训练（`work_dirs/v5_ablation_full`，会话 `session_id=89522`，状态：运行中） |
| 2026-03-11 | 按要求暂停长跑任务，改为先做 6 组 smoke test；当前环境 CUDA 初始化失败（error 304），待 GPU 会话恢复后执行 `scripts/smoke_test_v5_ablations.sh` |
| 2026-03-11 | smoke test 完成：6/6 配置均可正常训练 1 epoch 并产出 checkpoint（`v5_smoke_*` 全 PASS） |
