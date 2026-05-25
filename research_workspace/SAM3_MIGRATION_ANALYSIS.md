# SAM 2.1 → SAM 3.1 迁移分析

## 1. 当前架构 (SAM 2.1-based)

```
RGB [B,3,H,W] ──> SAM2HieraAdapter (frozen Hiera-L + BottleneckAdapter PEFT)
                      │
                      ├─ f1: [B, 144, H/4, W/4]
                      ├─ f2: [B, 288, H/8, W/8]
                      ├─ f3: [B, 576, H/16, W/16]
                      └─ f4: [B,1152, H/32, W/32]

Thermal [B,3,H,W] ──> ConvNeXtAux (trainable ConvNeXt-Tiny)
                      │
                      ├─ f1: [B,  96, H/4, W/4]
                      ├─ f2: [B, 192, H/8, W/8]
                      ├─ f3: [B, 384, H/16, W/16]
                      └─ f4: [B, 768, H/32, W/32]

Both ──> MMSAFusion (per-level cross-attention + modality weighting)
              │
              └─ 4-level fused features [144, 288, 576, 1152]
                       │
                       └─> SegFormerLiteHead ──> pred [B, C, H, W]
```

**关键数字：**
- RGB backbone: SAM 2.1 Hiera-L, ~224M 参数, frozen
- Thermal backbone: ConvNeXt-Tiny, ~28M 参数, trainable
- PEFT: BottleneckAdapter (dim→32→dim), ~2M 参数
- 总计可训练参数: ~30M
- 显存 (b=2+2, crop=512): ~20GB

## 2. SAM 3/3.1 架构分析

### 2.1 ViTDet Backbone

```python
ViT(
    img_size=1008, patch_size=14,        # stride=14 特征图
    embed_dim=1024, depth=32, num_heads=16,
    global_att_blocks=(7, 15, 23, 31),   # 4个全局注意力层
    return_interm_layers=False,           # 默认只返回最后一层
    window_size=24,
)
```

**默认模式 (return_interm_layers=False):**
- 输出: 单张 `[B, 1024, H/14, W/14]` 特征图
- channel_list = [1024]

**多尺度模式 (return_interm_layers=True):**
- 输出: 4张特征图 (对应 global_att_blocks)
- 每张均为 1024 channels, strides 均为 14 (同一分辨率!)
- channel_list = [1024, 1024, 1024, 1024]

### 2.2 SimpleFPN Neck

```python
Sam3DualViTDetNeck(
    d_model=256,
    scale_factors=[4.0, 2.0, 1.0, 0.5],
)
```

对 ViT 输出的 1024-ch 单张特征图做 FPN:

| Scale | 输出 Stride | 输出 Channel | 操作 |
|-------|-------------|-------------|------|
| 4.0   | ~3.5        | 256 | Deconv×2: 1024→512→256 |
| 2.0   | ~7          | 256 | Deconv×1: 1024→512 |
| 1.0   | ~14         | 256 | 恒等 |
| 0.5   | ~28         | 256 | MaxPool |

4级输出统一 256 channels。与 SAM 2.1 neck 的多尺度输出格式一致。

### 2.3 与 SAM 2.1 的关键差异

| 维度 | SAM 2.1 Hiera | SAM 3 ViTDet |
|------|---------------|--------------|
| 参数量 | ~224M | ~848M (仅 ViT) |
| 特征层级 | 自然多尺度 (4级不同分辨率) | 单分辨率 + FPN 构造多尺度 |
| 各级 channel | [144, 288, 576, 1152] (递增) | [256, 256, 256, 256] (统一) |
| 预训练 | SA-1B (mask) | SA-Co + 检测/跟踪数据 |
| 文本能力 | 无 | 有 (VETextEncoder)，对 FMB 分割无用 |

## 3. 迁移方案

### 方案 A: 只替换 RGB Backbone (推荐)

```
SAM 3 ViT + Neck (frozen + PEFT)  ──> 4×256ch
ConvNeXtAux (trainable)            ──> 4×[96,192,384,768]ch
            │
            └──> 改造 MMSAFusion ──> SegFormerLiteHead
```

**改造点:**

1. **SAM3ViTDetAdapter** (新文件, 替代 SAM2HieraAdapter)
   - 加载 SAM 3 ViT + Neck 权重
   - 冻结 ViT, 可选冻结/解冻 Neck
   - 添加 BottleneckAdapter PEFT (或在 Neck 层间加)
   - 输出: `[B,256,s1,s1], [B,256,s2,s2], [B,256,s3,s3], [B,256,s4,s4]`
   - out_channels = `[256, 256, 256, 256]`

2. **MMSAFusion 改造**
   - 当前 rgb_channels=[144,288,576,1152], aux_channels=[96,192,384,768]
   - 改为 rgb_channels=[256,256,256,256], aux_channels=[96,192,384,768]
   - 1×1 对齐: RGB 256→256 (几乎恒等, 可考虑去掉), Aux 保持 channel 对齐
   - 其余结构 (cross-attention, modality weighting) 不变
   - **关键**: f1 层现在 256ch 而非 144ch, cross-attention 计算量略增

3. **SegFormerLiteHead 改造**
   - decode_channels 保持 256
   - 输入从 [144+288+576+1152]=2160 → [256×4]=1024 channels (concat后)
   - 实际上更简单了

4. **Modality Heads** (rgb_head, thm_head)
   - rgb_head.in_channels: [256,256,256,256] (之前 [144,288,576,1152])
   - thm_head: 不变

**优点:**
- 最小改动, 只替换 RGB encoder
- 保留已验证的 Thermal encoder + Fusion 逻辑
- Neck 输出的 256ch 统一维度简化了后续模块

**缺点:**
- ViT 848M 参数 frozen, 比 Hiera 224M 大 3.8×, 显存压力
- 512×512 crop 时 stride≈14 → 约 37² = 1369 tokens, 还好
- 需额外存储 SAM 3 checkpoint (~3.4GB)

**显存估计:**
- SAM 3 ViT (frozen, bf16, no grad): ~1.7GB
- Neck (trainable): ~0.3GB
- ConvNeXtAux (trainable): ~0.1GB
- MMSAFusion + Decoder: ~0.2GB
- Activations (b=2, crop=512): ~8-12GB
- **总计: ~12-15GB** — RTX 5090 32GB 可容纳

### 方案 B: SAM 3 双模态 (4-ch输入)

```
[C=4 input: R,G,B,T] ──> 改造 patch_embed (3→4 ch) ──> SAM 3 ViT
                                                           │
                              ┌────────────────────────────┘
                              │
                    单次 ViT forward
                    输出 1024ch 特征图
                              │
                    使用 Dual Neck (add_sam2_neck=True)
                    分别解码 RGB 特征和 Thermal 特征
                              │
              ┌───────────────┴───────────────┐
         RGB Neck (256ch×4)            Thermal Neck (256ch×4)
              │                               │
              └──────────> MMSAFusion <────────┘
```

**优点:**
- 单次 backbone forward
- 两个 Neck 轻量, 总训练参数可控
- RGB+T 在早期即融合, ViT 内部 attention 可跨模态

**缺点:**
- patch_embed 第一层 3→4 通道需改造, 新 channel 需随机初始化
- SAM 3 ViT 的预训练权重是 3 通道的, 第 4 通道需从零学
- 无法保证 ViT 内部的跨模态交互是有益的 (GFFM 实验已证明早期交互可能有害)

### 方案 C: 只用 ViT 多尺度特征 (不用 Neck)

SAM 3 ViT 设 `return_interm_layers=True`, 从 global_att_blocks=[7,15,23,31] 直接拿 4 级特征:

```
ViT 4级中间特征: 每级 [B, 1024, H/14, W/14] (同分辨率!)
     │
     ├─ 加 FPN-style 上采样/downsample 构造不同 stride 的多尺度
     │   (类似方案 A 的 Neck, 但输入从最后一层改为 4 层)
     │
     └─> 输出 4 级多尺度特征 → MMSAFusion
```

**优点:**
- 更接近当前 Hiera 的"多级特征"模式
- 各层包含不同粒度的语义信息

**缺点:**
- 4 级全在同一 spatial 分辨率 (stride=14), 要构造多尺度需要更重的 Neck 设计
- 偏离 SAM 3 官方使用方法 (官方用 SimpleFPN 从最后一层构造多尺度)
- 收益不明确

### 方案 D: 完全替换为 SAM 3 DETR 检测头 (最激进)

放弃 MMSAFusion + SegFormerLiteHead, 直接用 SAM 3 的 `UniversalSegmentationHead` (MaskFormer-style).

**不推荐**: 丢弃了已验证有效的 MMSA 融合框架, 且 SAM 3 的 seg head 是为开放词汇设计的, 对 FMB 固定类别不一定更优.

## 4. 推荐路径: 方案 A

### 4.1 实现步骤

| Step | 内容 | 文件 |
|------|------|------|
| 1 | 验证 SAM 3 checkpoint 可用, 本地跑通 inference | 测试脚本 |
| 2 | 实现 `SAM3ViTDetAdapter` | `segmentation/models/backbones/sam3_vitdet_adapter.py` |
| 3 | 添加 PEFT (BottleneckAdapter 在 Neck 层间) | 同上 |
| 4 | 改造 `MMSAFusion` 支持 256ch RGB 输入 | `segmentation/models/fusion/mmsa_fusion.py` |
| 5 | 改造 `SegFormerLiteHead` 适配新 in_channels | `segmentation/models/decode_heads/segformer_lite_head.py` |
| 6 | 新增 CLI args (`--sam3-ckpt`, `--sam3-cfg`) | `segmentation/train_label_efficient.py` |
| 7 | 新训练脚本 + launch | shell scripts |

### 4.2 风险与未知

| 风险 | 等级 | 应对 |
|------|------|------|
| ViT 848M 显存是否够 | 中 | 实测; 可减小 crop_size 或用 gradient checkpointing |
| SAM 3 特征是否比 Hiera 更适合 FMB | 高 | 这是核心假设; 需实验验证 |
| Neck 层是否需要增补 PEFT | 中 | 先不加, 看收敛; 后续可加 BottleneckAdapter |
| 256ch 统一 vs 递增 channel 哪个更好 | 低 | 256ch 已足够, MMSA 用 256 decode_channels |
| SAM 3 checkpoint 权限 | 低 | HF 申请 facebook/sam3 |

### 4.3 关键设计决策

**Q: ViT 是否冻结?**
建议冻结 (frozen backbone + trainable neck + PEFT), 类似当前 SAM2HieraAdapter 模式。848M 全量训练不可行。

**Q: Neck 是否共享?**
Neck 4层 FPN 均可训练 (約 5-10M 参数), 作为 adapter 的一部分。

**Q: PEFT 加在哪?**
可选在 Neck 每一层加 BottleneckAdapter, 或在 ViT 最后几个 global_att_blocks 加。先不加 PEFT, 只训 Neck + MMSAFusion + Decoder, 看效果。

**Q: 是否保留 SAM 2.1 的 Thermal encoder?**
是。ConvNeXtAux 已验证有效, 且与 RGB backbone 独立, 无需改动。

## 5. Checkpoint

- **本地路径**: 需从 HuggingFace 下载 `facebook/sam3`
- **模型文件**: 约 3.4GB
- **下载命令**: `huggingface-cli download facebook/sam3 --local-dir /home/jl/sam3/checkpoints`
- **权限**: 需在 hf.co 申请访问

## 6. 当前状态

- [x] SAM 3 代码已 clone 到 `/home/jl/sam3`
- [ ] SAM 3 checkpoint 下载
- [ ] 本地 inference 验证
- [ ] 架构设计详细 API (接口定义)
- [ ] 实现
