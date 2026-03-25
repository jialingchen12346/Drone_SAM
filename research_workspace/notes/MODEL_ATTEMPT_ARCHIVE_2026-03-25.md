# 模型尝试结果归档（2026-03-25）

## 1) 评测口径说明

- 旧口径（present-only）：缺失类不计入 mIoU。
- 严格口径（strict, absent=0）：缺失类按 0 计入 mIoU（当前对外口径）。
- FMB `test` 中 `Bicycle` 长期缺失，口径切换会显著改变最终 mIoU。

## 2) 快速重评（本次新增）

| run | checkpoint | 旧口径 mIoU | strict mIoU | 差值 |
|---|---|---:|---:|---:|
| `gen_ohem_only_bs8_cmr056_aux010_r3_20260324_170732` | `epoch_30.pth` | 67.67 | 62.84 | -4.83 |
| `fmb_cb_ohem_cmr055_aux010_alpha03_20260324_203024` | `best.pth` | - | 62.23 | - |

原始结果文件（远程）：
- `/root/autodl-tmp/work_dirs/auto_loop/gen_ohem_only_bs8_cmr056_aux010_r3_20260324_170732/test_results_present_only_backup.txt`
- `/root/autodl-tmp/work_dirs/auto_loop/gen_ohem_only_bs8_cmr056_aux010_r3_20260324_170732/test_results_strict_abs0_epoch30_reval_20260325.txt`
- `/root/autodl-tmp/work_dirs/staged_funnel/s3_confirm_e30/fmb_cb_ohem_cmr055_aux010_alpha03_20260324_203024/test_results_strict_abs0_reval_20260325.txt`

## 3) 已尝试模型结构与效果（摘要）

1. `RRF-DSD` 主结构（有效）
- 改动：RRF 细节残差、DSD 边界先验、小目标增强、残差门控、stride-4 细节辅助头。
- 代表结果：`epoch_80 test=65.24`（早期统一口径对照）。

2. `TSCB` 细结构补偿分支（当前版本未验证出稳定收益）
- 代表结果：`63.69 / 64.60`（旧口径阶段）。
- 结论：未超过 OHEM-only 主线。

3. `class-balanced + OHEM`（训练侧增强，非新骨架）
- `V4-1` strict 代表结果：`62.53`。
- 结论：单独引入 class-balance 收益不足。

4. 三段式漏斗（strict）
- `S1 57.10 -> S2 61.53 -> S3 62.23`，`S3` 触发止损。

## 4) 当前归档结论

- 历史“高分”与 strict 口径存在显著差距（已通过快速重评量化）。
- 当前可复用 strict 锚点约在 `62.2~62.8` 区间。
- 后续比较必须统一 strict 口径，避免再混用 present-only。

## 5) 新一轮结构筛选（进行中）

- 结构版本：`V5 = Boundary-Guided Refiner + Rare-Class Residual`（在 DSDHead 上可开关实现）。
- 运行配置：`12 epoch / batch=8 / OHEM-only / strict(absent=0)`。
- 配置文件：`research_workspace/plans/quick_screen_v5_sam21_e12_2026-03-25.json`。
- 远程运行：
  - 调度进程：`scripts/auto_train_eval_loop.py --max-cycles 1`
  - work_dir 根：`/root/autodl-tmp/work_dirs/quick_screen_v5_sam21_e12/`
  - 训练日志：`research_workspace/artifacts/quick_screen_v5_sam21_e12_20260325/logs/*_train.log`

### 5.1 运行结果（已完成）

- recipe: `qs_v5_sam21_ohem_bs8_e12_bref_rare`
- strict 指标（absent=0）：
  - `val mIoU=56.35`
  - `test mIoU=55.81`
  - `gap=0.54`
- 结论：明显低于当前 strict 锚点（`62.2~62.8`），该 V5 组合在当前参数下不具备晋级价值。

### 5.3 单变量定位（已完成）

统一设置：`12 epoch / bs8 / OHEM-only / strict(absent=0) / SAM2.1 large`

| recipe | val mIoU | test mIoU | 相对 baseline |
|---|---:|---:|---:|
| `qs_v5ctrl_baseline_sam21_ohem_bs8_e12` | 56.80 | 55.61 | 0.00 |
| `qs_v5a_boundary_only_s035_bw002` | 56.80 | 54.48 | -1.13 |
| `qs_v5a_rare_only_rs030` | 56.68 | 54.69 | -0.92 |

结论：在同预算对照下，`boundary-only` 与 `rare-only` 均劣于 baseline，V5 路线本轮判退。

### 5.2 磁盘整理

- 已删除重复 checkpoint：`epoch_12.pth`（与 `best.pth` 指标一致）。
- 保留：`best.pth + val_results.txt + test_results.txt`。
