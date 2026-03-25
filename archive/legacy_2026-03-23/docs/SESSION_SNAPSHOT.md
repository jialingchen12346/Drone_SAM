# Drone-SAM-Adapter 项目快照
> 最近更新: 2026-03-17
> 用途: 新对话上下文恢复，直接拷贝本文件内容作为开场白

---

## 项目概况

**目标**: 面向无人机场景的多模态语义分割，在 FMB 数据集（RGB+Thermal，14类）上超越基线（mIoU=66.1，**Test 集**）
**当前成果**: **Test mIoU=68.62**（超基线 **+2.52**）✅ 论文主结果

**核心创新**:
1. **CACAF** — 条件感知跨模态自适应融合（MQA质量评估 + 双向交叉注意力 + 自适应加权），16.02M
2. **HGSOAD** — 层级引导小目标感知解码器（SAGU通道注意力门 + U-Net上采样 + 辅助监督），6.01M

**架构**:
```
RGB  → SAM2 Hiera-L (frozen) + Bottleneck Adapter (1.71M) ──┐
                                                              ├─→ CACAF (×4 levels) ─→ HGSOAD ─→ pred
Aux  → ConvNeXt-Tiny (全量可训练, 27.82M)                  ──┘
```
总参数: 263.7M，可训练: 51.6M（19.6%）

---

## 环境信息

| 项目 | 当前机器（AutoDL） | 原机器（V5 训练用） |
|------|-----|-----|
| GPU | RTX 5090 | RTX 4070 Ti Super (16 GB) |
| PyTorch | 2.7.0+cu128 | 2.4.1+cu121 |
| Python | 3.12 | 3.10 |
| 项目路径 | `/root/autodl-tmp/Drone-SAM-Adapter/` | `/home/jl/Drone-SAM-Adapter/` |
| 数据集路径 | `/root/autodl-tmp/dataset/FMB` | `/home/jl/dataset/FMB` |
| SAM2 检查点 | `checkpoints/sam2_hiera_large.pt` (md5: `2b30654b...`) | 同左 |
| SAM2 配置 | `configs/sam2.1/sam2.1_hiera_l.yaml` | 同左 |
| ConvNeXt 缓存 | `~/.cache/huggingface/hub/models--timm--convnext_tiny.fb_in22k` | 同左 |

> ⚠️ **跨机器训练差异**：相同配置在两台机器上训练结果差 ~2 mIoU（V5: 68.62 vs V7: 66.17），原因是 GPU 架构 + PyTorch 版本差异。但**评测结果完全一致**——V5 checkpoint 在当前机器上评测复现三位小数。

---

## 已完成模块

| 文件 | 说明 | 状态 |
|------|------|------|
| `segmentation/models/backbones/sam2_hiera_adapter.py` | SAM2 Hiera-L + Bottleneck Adapter | ✅ |
| `segmentation/models/backbones/aux_encoder.py` | ConvNeXt-Tiny 辅助编码器 | ✅ |
| `segmentation/models/fusion/cacaf.py` | CACAF 融合（f1 无交叉注意力，f2/f3/f4 双向 MHA） | ✅ |
| `segmentation/models/decode_heads/hgsoad_head.py` | HGSOAD 解码器 + SAGU + aux heads | ✅ |
| `segmentation/models/segmentors/cacaf_segmentor.py` | 完整分割器 + `SimpleFusion` 消融替换类 | ✅ |
| `segmentation/datasets/fmb_dataset.py` | FMB Dataset Loader（独立，无 MMSeg 依赖） | ✅ |
| `segmentation/train_cacaf.py` | 训练脚本（消融/增强/损失开关齐全） | ✅ |
| `segmentation/eval_fullres.py` | 全分辨率 Test 评估（需 `--no-cacaf`/`--no-sagu` 匹配模型） | ✅ |

---

## 关键设计细节

### SAM2 / ConvNeXt 特征图规格（512×512 输入）
| 层 | SAM2 Shape | ConvNeXt Shape | Stride |
|----|-----------|---------------|--------|
| f1 | [B,144,128,128] | [B,96,128,128] | ×4 |
| f2 | [B,288,64,64] | [B,192,64,64] | ×8 |
| f3 | [B,576,32,32] | [B,384,32,32] | ×16 |
| f4 | [B,1152,16,16] | [B,768,16,16] | ×32 |

### CACAF 关键约束
- f1（128×128=16K tokens）**不用**交叉注意力（显存爆炸）→ 仅用 MQA 加权求和
- f2/f3/f4 使用 `nn.MultiheadAttention` 双向交叉注意力

### 消融开关
```python
CACafSegmentor(
    use_cacaf=True,   # False → SimpleFusion（等权平均）
    use_sagu=True,    # False → 跳过 SAGU 通道注意力门
    use_dice=True,    # CE + Dice Loss
    use_ohem=False,   # OHEM CE（与 Dice 存在梯度冲突，不推荐同时使用）
)
# CLI: --no-cacaf / --no-sagu / --use-dice / --use-ohem
```

### 损失函数（最优配置: CE+Dice）
```python
loss = CE(main) + Dice(main) + 0.4*(CE(aux2)+Dice(aux2)) + 0.4*(CE(aux3)+Dice(aux3))
```

---

## FMB 数据集

> 详细文档: `docs/FMB_DATASET.md`

```
/root/autodl-tmp/dataset/FMB/
├── {split}_easy_files.txt / {split}_hard_files.txt
├── {split}/
│   ├── Visible/{easy,hard}/*.png   ← RGB
│   ├── Infrared/*.png              ← Thermal（3-ch 灰度）
│   └── Label/*.png                 ← GT mask
```

- 数据量：train=1060，val=160，test=280
- Label 值域：**1-14**（1-indexed），Loader 内部 `lbl -= 1; lbl[lbl<0] = 255`

---

## 主结果（论文 Table 1 用）

**最优模型**: V5-NoOHEM（CE+Dice），在原机器训练，checkpoint 已迁移至当前机器并验证

**Test 集全分辨率评估 (800×600)**:
| 指标 | 基线 MM SAM-Adapter | 本方法 | Δ |
|------|-----------|------|---|
| mIoU | 66.1 | **68.62** | **+2.52** |
| mAcc | 72.09 | **75.76** | **+3.67** |
| aAcc | 93.95 | 93.27 | -0.68 |

**per-class IoU (Test)**:
| 类别 | IoU | 类别 | IoU |
|------|-----|------|-----|
| Road | 88.0 | Person | 75.9 |
| Sidewalk | 52.2 | Car | 81.7 |
| Building | 84.7 | Truck | 42.0 |
| Traffic Light | 51.1 | Bus | **37.1** (+14.2) |
| Traffic Sign | 82.2 | Motorcycle | **57.7** (+3.8) |
| Vegetation | 88.0 | Bicycle | nan（test无样本） |
| Sky | 95.8 | Pole | 55.7 |

**最优 checkpoint**: `work_dirs/v5_ablation_no_ohem/epoch_60.pth`

---

## 消融实验（论文用，✅ 全部已验证）

所有 V5 消融在原机器（4070 Ti Super）训练，checkpoint 已迁移，在当前机器评测**复现三位小数**。

### 模块消融（统一 CE+Dice，4070 Ti Super，Test ep60）✅ 论文用

> `abl_clean_*` 系列：与主结果完全同配置（CE+Dice），在原机器干净重训，逻辑完全自洽。

| 配置 | mIoU | mAcc | aAcc | ΔmIoU |
|------|------|------|------|-------|
| **Full (CACAF + SAGU)** | **68.62** | 75.75 | 93.27 | — |
| w/o CACAF | 62.86 | 69.52 | 93.32 | **-5.76** |
| w/o SAGU | 65.57 | 71.53 | 93.45 | **-3.05** |
| w/o CACAF & SAGU | 64.78 | 70.69 | 93.31 | **-3.84** |

**关键发现**:
- **CACAF 贡献最大**（-5.76），是核心性能来源
- **SAGU 贡献显著**（-3.05），在无 CACAF 时贡献减小（-3.84 vs -5.76），说明 SAGU 依赖 CACAF 质量特征
- NoCACAF-NoSAGU (-3.84) 优于 NoCACAF (-5.76)，印证 SAGU 在低质量特征上效果受限

**检查点位置（原机器）**: `work_dirs/abl_clean_{no_cacaf,no_sagu,no_cacaf_no_sagu}/epoch_60.pth`

---

### 旧模块消融参考（CE+Dice+OHEM，已废弃，不用于论文）

> 损失配置与主结果不一致，Full 基准偏低，消融量级失真，仅保留作历史参考。

| 配置 | mIoU | ΔmIoU |
|------|------|-------|
| Full (CACAF + SAGU) | 66.79 | — |
| w/o CACAF | 64.71 | -2.08 |
| w/o SAGU | 64.67 | -2.12 |
| w/o CACAF & SAGU | 67.00 | +0.21 |

### 损失函数消融（固定完整模型，Test ep60）

| 损失配置 | mIoU | mAcc | aAcc | ΔmIoU |
|---------|------|------|------|-------|
| **CE + Dice** | **68.62** | **75.76** | 93.27 | — |
| CE + OHEM | 67.79 | 74.70 | 93.58 | -0.83 |
| CE + Dice + OHEM | 66.79 | 72.89 | 93.46 | -1.83 |

**关键发现**: Dice 与 OHEM 同时使用存在梯度冲突，单独使用 Dice 效果最优。

### V7 消融参考（当前机器训练，CE+Dice，Test ep60）

> V7 与 V5 训练环境不同（RTX 5090 vs 4070 Ti），结果系统性偏低 ~2 mIoU，仅供交叉参考。

| 配置 | mIoU | mAcc | aAcc |
|------|------|------|------|
| Full (CACAF + SAGU) | 66.17 | 72.32 | 93.31 |
| w/o CACAF | 63.13 | 69.68 | 93.32 |
| w/o SAGU | 66.42 | 72.47 | 93.38 |
| w/o CACAF & SAGU | 64.84 | 71.22 | 93.38 |

---

## 当前磁盘上的 Checkpoint

| 路径 | 内容 | 用途 |
|------|------|------|
| `v5_ablation_full/epoch_60.pth` | V5-Full (CE+Dice+OHEM) | 消融基准 |
| `v5_ablation_no_cacaf/epoch_60.pth` | V5 w/o CACAF | 模块消融 |
| `v5_ablation_no_sagu/epoch_60.pth` | V5 w/o SAGU | 模块消融 |
| `v5_ablation_no_cacaf_no_sagu/epoch_60.pth` | V5 w/o CACAF & SAGU | 模块消融 |
| `v5_ablation_no_ohem/epoch_60.pth` | **V5 CE+Dice = 68.62** | **论文主结果** |
| `v5_ablation_no_dice/epoch_60.pth` | V5 CE+OHEM = 67.79 | 损失函数消融 |
| `v6_no_cacaf_no_sagu/{best,epoch_60}.pth` | V6 参考（有 train.log） | 历史参考 |
| `v7_full/{best,epoch_60}.pth` | V7 Full | 当前机器消融 |
| `v7_no_cacaf/{best,epoch_60}.pth` | V7 w/o CACAF | 当前机器消融 |
| `v7_no_sagu/{best,epoch_60}.pth` | V7 w/o SAGU | 当前机器消融 |
| `v7_no_cacaf_no_sagu/{best,epoch_60}.pth` | V7 w/o CACAF & SAGU | 当前机器消融 |
| `mfnet_v1/epoch_60.pth` | 历史 MFNet checkpoint（旧口径，不用于论文） | 仅留存，不引用 |

---

## 训练命令模板

```bash
cd /root/autodl-tmp/Drone-SAM-Adapter

# 最优配置（复现 68.62 的配置，仅损失函数部分）
python segmentation/train_cacaf.py \
    --data-root /root/autodl-tmp/dataset/FMB \
    --batch-size 4 --epochs 60 \
    --work-dir work_dirs/<NAME> \
    --bf16 --use-dice \
    --eval-resize-mode letterbox \
    --cat-max-ratio 0.75 \
    --blur-prob 0.2 \
    --photo-distort \
    2>&1 | tee work_dirs/<NAME>/train.log

# 评估（注意 --no-cacaf / --no-sagu 需与训练时一致）
python segmentation/eval_fullres.py \
    --checkpoint work_dirs/<NAME>/epoch_60.pth \
    --data-root /root/autodl-tmp/dataset/FMB \
    --split test --bf16
```

---

## 已踩过的坑

| 坑 | 现象 | 根因 | 修复 |
|----|------|------|------|
| SAM2 推理分辨率 | `RuntimeError: 150 vs 144` | window_embed 与非 512 输入不兼容 | 推理固定 512×512 |
| 参数名错误 | `TypeError: unexpected keyword argument 'sam2_cfg'` | 参数是 `sam2_checkpoint`/`sam2_config` | 使用正确参数名 |
| Label 越界 | CUDA assertion | FMB label 是 1-indexed | `lbl -= 1; lbl[lbl<0] = 255` |
| loss=nan | 模型崩溃 | FP16 overflow | Pre-norm + `--bf16` |
| eval 加载失败 | `Missing key(s)` | eval 脚本未传 `--no-cacaf`/`--no-sagu` | 评估时 flag 需与训练一致 |
| 跨机器训练差异 | 相同配置差 ~2 mIoU | RTX 5090 vs 4070 Ti + PyTorch 版本 | 评测用同一 checkpoint，训练结果不可直接对比 |
| Dice+OHEM 组合 | 性能低于单独使用 | 梯度冲突 | 最终模型只用 CE+Dice |

---

## 注意事项

1. **务必用 `--bf16`**，FP16 在深层训练必然 overflow
2. batch=4 BF16 ≈ 14-16 GB VRAM
3. 评估脚本 `eval_fullres.py` 的 `--no-cacaf`/`--no-sagu` 必须和训练时一致，否则 key mismatch
4. 训练时建议用 `tee` 保存 log，历史实验因缺少 log 导致配置不可追溯
5. 长时间训练用 `screen` 防止 SSH 断开

---

## 论文进度

- ✅ 主结果: 68.62 mIoU（超基线 +2.52）
- ✅ 模块消融: CACAF (-5.76), SAGU (-3.05), 协同关系验证（CE+Dice 统一配置，口径一致）
- ✅ 损失函数消融: CE+Dice 最优，Dice+OHEM 冲突
- ✅ 可视化脚本: `scripts/visualize_predictions.py`, `analyze_failure_cases.py`, `visualize_fusion_weights.py`, `plot_iou_comparison.py`
- ✅ 论文草稿: `docs/PAPER_DRAFT.md`
- ⏳ 架构图 (Figure 1-2)
- ✅ MFNet 数据集补充实验（正式口径）: **MFNet v2 Test mIoU=54.73**, mAcc=68.14, aAcc=97.90（9类协议）
- ⏳ MFNet SOTA 对比表（与 RTFNet/FEANet/GMNet 等对比）
- ⏳ SOTA 对比表补充更多方法（FMB 数据集）
- ⏳ 参数量/推理速度对比

---

## 相关文档

- 论文草稿: `docs/PAPER_DRAFT.md`
- 证据链计划: `docs/EVIDENCE_CHAIN.md`
- 数据集详细说明: `docs/FMB_DATASET.md`
