# 会话交接（当前版）

历史全量交接记录已归档：
`research_workspace/archive/legacy_2026-03-30_focus_shift/notes/SESSION_HANDOFF_full_2026-03-30_pre_compact.md`

## 当前目标

- 主线：标签效率导向语义分割（半监督 + 弱监督）。
- 叙事：多模态提供免费的跨模态一致性信号，服务标注效率学习（不是"更好的融合→全监督 SOTA"）。
- 数据策略：FMB 先打穿方法证据链；后续补遥感公开集（LoveDA 优先）。
- 投稿：BMVC 主投，ACCV 备投增强。

## 最新状态（2026-04-08）

- 口径红线保持：主结论只用 strict（`absent=0`）。
- 训练配置：`batch-size-l/u=4 / num-workers=8 / warmup-iters=250 / bf16`
- 10% 标注预算四路线对照（test strict mIoU，已闭环）：
  1. `Sup-only`（seed42）：`49.70`
  2. `Semi-only`（seed42/3407）：`49.95 / 51.70`，均值 `50.83 ± 0.88`
  3. `Point-only`（seed42）：`52.95`
  4. `Semi+Point`（seed42/3407）：`53.25 / 53.67`，均值 `53.46 ± 0.21`
- 增益分解：`Point-only - Sup-only = +3.25`；`Semi+Point - Point-only = +0.51`

## 当前运行实验（远端）

- 机器：`root@connect.bjb2.seetacloud.com -p 31616`（两张 RTX 5090 各 32GB）
- 代码：`/root/Drone-SAM-Adapter`
- 数据：`/root/autodl-tmp/datasets/FMB`

| GPU | 任务 | 状态 |
|-----|------|------|
| 0 | diag_pseudo_quality（10% teacher，3x forward） | 运行中 → 完成后自动接 r5 e30 sweep |
| 1 | r20 e30 sweep（5 个实验顺序跑） | 运行中 |

预计完成：GPU1 约 3h，GPU0 约 4.5h（含 diag ~30min）。

日志位置：
- `/root/autodl-tmp/logs/r5_e30.log`
- `/root/autodl-tmp/logs/r20_e30.log`
- `/root/autodl-tmp/work_dirs/label_eff_r10_bs4_w250_seed42/diag_pseudo_quality.txt`

## 下一步（优先级）

1. **等待并解读实验结果**（今晚）：
   - 检查 5%/20% 的四路线排序是否与 10% 一致
   - 检查 diag agree/disagree accuracy gap 是否 > 0.05
2. **决策**（基于结果）：
   - 趋势一致 → 启动 120-epoch 完整训练
   - agree gap 显著 → 实现跨模态一致性伪标签过滤（改 `build_pseudo_targets()`）
   - agree gap 不显著 → 另找 semi 增益偏小的原因
3. 进入鲁棒性矩阵（clean/degraded/missing-modality）
4. 启动 LoveDA 等公开遥感集补充实验

## 架构设计背景

详见：`docs/LABEL_EFFICIENT_DESIGN_2026-04-08.md`

核心结论：
- 当前模型与训练策略完全解耦，需要设计"为标注效率服务"的架构组件
- 收敛到单一设计：跨模态一致性伪标签过滤（有证据支撑，实现成本低）
- 原型记忆、Point 扩散暂不做（主张超前于证据）

## 新增代码资产

| 文件 | 功能 |
|------|------|
| `segmentation/eval_label_eff.py` | label-efficient checkpoint 测试集评测 |
| `segmentation/diag_pseudo_quality.py` | 伪标签质量诊断（3x modality ablation forward） |
| `run_r5_gpu0.sh` | 5% 预算 30-epoch 扫描（含 diag） |
| `run_r20_gpu1.sh` | 20% 预算 30-epoch 扫描 |
| `artifacts/weak_labels/fmb_points_seed42_r5_u/` | 5% 弱标注（1007样本，8094点） |
| `artifacts/weak_labels/fmb_points_seed42_r20_u/` | 20% 弱标注（848样本，6820点） |

## 快速命令

```bash
# 登录
ssh -p 31616 root@connect.bjb2.seetacloud.com

# 查看状态
nvidia-smi
tail -20 /root/autodl-tmp/logs/r5_e30.log
tail -20 /root/autodl-tmp/logs/r20_e30.log

# 查看诊断结果
cat /root/autodl-tmp/work_dirs/label_eff_r10_bs4_w250_seed42/diag_pseudo_quality.txt

# 查看各实验测试结果
for d in /root/autodl-tmp/work_dirs/label_eff_r{5,20}_e30_*/; do
  echo "=== $d ==="; grep 'strict_mIoU' $d/test_results.txt 2>/dev/null; done
```
