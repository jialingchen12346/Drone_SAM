# Drone-SAM-Adapter v2: 面向恶劣条件的多模态语义分割改进方案

## 项目概述

| 项目 | 描述 |
|------|------|
| **项目名称** | Drone-SAM-Adapter |
| **基线项目** | Multimodal-SAM-Adapter (MM SAM-Adapter) |
| **论文定位** | 应用类期刊，面向恶劣条件（雾/雨/夜间/模糊）的多模态语义分割 |
| **核心卖点** | 条件感知的自适应融合 + 小目标感知解码 |
| **目标硬件** | NVIDIA 4090 (24GB VRAM) |
| **创建日期** | 2026-03-04 |
| **版本** | v2.0（基于专家审核修订） |

---

## 1. v1 方案问题诊断

### 1.1 原方案核心缺陷

| 问题 | 详细描述 | 严重性 |
|------|----------|--------|
| **创新点堆砌** | 同时提出 4 个创新（双编码器/跨模态融合/RFB/轻量解码器），每个都是已有项目直接借鉴，缺乏深度原创性。审稿人会判定为"模块拼装" | 高 |
| **数据集不支撑定位** | 标题定位"无人机搜救"，但 4 个数据集（FLM/DELIVER/MUSES/COD10K）无一是真正的无人机搜救场景 + RGB+热成像配对 | 高 |
| **架构过于复杂** | SAM2 Hiera-L + DINOv2 ViT-L + TwinConvNeXt 三大模型同时加载，4090 显存极度紧张 | 高 |
| **边缘部署卖点空洞** | 深度可分离卷积在分割领域太常见，不构成创新点；且实验不会在边缘设备验证 | 中 |

### 1.2 数据集-场景匹配度分析

| 数据集 | 模态 | 场景 | 与"无人机搜救"匹配度 | 与"恶劣条件多模态分割"匹配度 |
|--------|------|------|--------|--------|
| FLM | RGB + Thermal | 城市街景 | 中（有热成像，但非搜救） | **高**（RGB+热成像融合） |
| DELIVER | RGB + LiDAR/Depth/Event | 自动驾驶（5种天气） | 低（无热成像） | **高**（多天气条件） |
| MUSES | RGB + Event | 自动驾驶（恶劣天气） | 低（无热成像） | 中（有恶劣天气但事件相机不常用） |
| COD10K | RGB only | 伪装目标 | 低（单模态） | 低（无关） |

### 1.3 修订策略

```
原方案: "无人机搜救 + 4个创新点 + 3个大模型"
    ↓ 修订
新方案: "恶劣条件多模态分割 + 2个核心创新 + 精简架构"
```

**核心思路**: 对于应用类期刊，**1-2 个扎实的创新点 > 4 个浅层堆砌**。

---

## 2. 论文定位与创新点设计

### 2.1 论文定位

**建议标题方向**:
- 中文: 面向恶劣条件的条件感知多模态语义分割方法
- 英文: Condition-Aware Adaptive Multimodal Semantic Segmentation under Adverse Conditions

**关键词**: 多模态语义分割、恶劣条件、自适应融合、小目标检测、SAM2

**摘要逻辑**:
```
问题: 恶劣条件（雾/雨/夜间/模糊）导致 RGB 图像质量退化，单模态分割性能下降
     → 引入辅助模态（热成像/LiDAR 等）可以弥补 RGB 的不足
     → 但现有多模态融合方法使用固定融合策略，无法适应不同退化程度
     → 且现有解码器对小目标（人员/车辆）的检测能力不足

方法: 提出 XXX-Net（待定名称），包含:
     1. CACAF: 条件感知跨模态自适应融合模块
     2. HGSOAD: 层级引导小目标感知解码器
     3. 基于 SAM2 Hiera 的参数高效适配策略

实验: 在 FLM (RGB+Thermal) 和 DELIVER (RGB+LiDAR) 两个数据集上验证
     → 整体 mIoU 超过基线 X%
     → 恶劣条件子集上优势更加明显
     → 小目标类别 IoU 显著提升
```

### 2.2 创新点清单

#### 创新 1（核心）: CACAF — Condition-Aware Cross-modal Adaptive Fusion

**问题**: 现有多模态融合方法（如 MM SAM-Adapter 的 MSDeformAttn）使用静态或缓慢学习的融合权重（gamma 初始化为 0 逐渐学习），无法根据当前输入的退化程度动态调整模态贡献。

**洞察**: 在恶劣条件下，RGB 和辅助模态的可靠性是动态变化的：
- 浓雾中: RGB 几乎不可用 → 应大幅依赖热成像
- 强光下: RGB 过曝 → 应增加辅助模态权重
- 正常条件: RGB 信息丰富 → 辅助模态作为补充

**方法设计**:
```
CACAF 模块:

1. 模态质量评估分支 (Modal Quality Assessment, MQA):
   ├── RGB 质量分支: GAP(rgb_feat) → MLP → quality_score_rgb ∈ [0,1]
   ├── Aux 质量分支: GAP(aux_feat) → MLP → quality_score_aux ∈ [0,1]
   └── 输出: 归一化权重 w_rgb, w_aux = softmax([q_rgb, q_aux])

2. 跨模态增强 (Cross-Modal Enhancement):
   ├── RGB→Aux 注意力: Attn(Q=rgb, K=aux, V=aux)  → aux_enhanced
   ├── Aux→RGB 注意力: Attn(Q=aux, K=rgb, V=rgb)  → rgb_enhanced
   └── 注意力机制: 轻量级多头注意力（非变形注意力，避免编译CUDA ops）

3. 自适应融合 (Adaptive Fusion):
   └── output = w_rgb * rgb_enhanced + w_aux * aux_enhanced
```

**与基线的本质区别**:
| 方面 | MM SAM-Adapter | CACAF (Ours) |
|------|----------------|--------------|
| 融合权重 | gamma 初始化~0，训练后固定 | 根据每张输入图像的退化程度动态生成 |
| 模态重要性 | 所有图像相同的融合比例 | 不同图像、不同退化程度有不同的融合比例 |
| 条件感知 | 无 | 显式的模态质量评估分支 |
| 论文故事 | 通用多模态融合 | "在恶劣条件下自适应地利用更可靠的模态" |

**审稿人可能的质疑与回应**:
| 质疑 | 回应 |
|------|------|
| "质量评估分支如何训练？" | 端到端联合训练，通过分割损失反向传播自动学习什么是"高质量"特征 |
| "和注意力权重有什么区别？" | 注意力权重在空间维度上自适应（哪个位置更重要），CACAF 在模态维度上自适应（哪个模态更可靠），两者正交互补 |
| "实验如何证明条件感知？" | 可视化不同天气条件下的 w_rgb/w_aux 变化，展示融合权重与退化程度的相关性 |

---

#### 创新 2（核心）: HGSOAD — Hierarchical-Guided Small Object Aware Decoding

**问题**: 基线的 SegFormer Head 是 all-MLP 结构，对小目标不敏感。SAM2-UNet 的 RFB 模块虽增强了多感受野，但应用在编码器输出端，未在解码过程中逐级利用多尺度信息。

**方法设计**:
```
HGSOAD 解码器:

1. 尺度感知门控单元 (Scale-Aware Gating Unit, SAGU):
   ├── 对每级编码器特征计算门控信号:
   │   gate_i = σ(Conv(GAP(f_i)) + Conv(GMP(f_i)))   (通道注意力)
   ├── 门控后特征: f_i' = gate_i ⊙ f_i
   └── 高分辨率层（f1, f2）的门控增强小目标响应
       低分辨率层（f3, f4）的门控保留全局上下文

2. 层级上采样路径 (Hierarchical Upsampling Path):
   ├── Stage 4: f4' → Up2x → concat(f3') → DoubleConv → d3
   ├── Stage 3: d3  → Up2x → concat(f2') → DoubleConv → d2
   ├── Stage 2: d2  → Up2x → concat(f1') → DoubleConv → d1
   └── Stage 1: d1  → Conv1x1 → num_classes

3. 可选: 辅助监督 (Auxiliary Supervision):
   └── 在 d2, d3 输出辅助分割头，用深监督加速收敛
```

**与已有方法的区别**:
| 方面 | SegFormer Head | SAM2-UNet (RFB+UNet) | HGSOAD (Ours) |
|------|---------------|----------------------|---------------|
| 多尺度利用 | 简单 concat + MLP | RFB 在编码端扩感受野 | 门控在解码端逐级控制 |
| 小目标处理 | 无专门机制 | 多膨胀率感受野 | SAGU 门控增强高分辨率层 |
| 特征选择 | 所有尺度等权融合 | 所有尺度等权融合 | 门控动态选择有用特征 |

---

#### 辅助创新: SAM2 Hiera 参数高效适配

**内容**: 首次在 SAM2 Hiera 骨干上系统对比 LoRA 和 Bottleneck Adapter 两种 PEFT 策略在多模态语义分割任务上的效果。

**实验价值**:
- SAM2-UNet/SAM2-UNeXT 使用 Bottleneck Adapter，但目标是二值分割
- SAMed 使用 LoRA，但目标是 SAM v1 的医学分割
- 尚无工作在 SAM2 Hiera + 多类别语义分割 + 多模态融合场景下对比这两种策略

**这不作为论文的主要创新点，而是作为实验分析的工程贡献。**

---

## 3. 模型架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    输入: RGB (3ch) + Aux (3ch)                   │
│                  (Thermal for FLM / LiDAR for DELIVER)          │
└───────────┬──────────────────────────────────┬──────────────────┘
            │                                  │
            ▼                                  ▼
┌──────────────────────┐          ┌──────────────────────────┐
│  SAM2 Hiera-L        │          │  辅助模态编码器            │
│  (frozen + Adapter)  │          │  (ConvNeXt-Tiny, 训练)    │
│                      │          │                          │
│  输出 4 级特征:       │          │  输出 4 级特征:            │
│  x1: stride 4        │          │  y1: stride 4             │
│  x2: stride 8        │          │  y2: stride 8             │
│  x3: stride 16       │          │  y3: stride 16            │
│  x4: stride 32       │          │  y4: stride 32            │
└───────┬──────────────┘          └───────┬────────────────────┘
        │                                 │
        │         ┌──────────┐            │
        └────────►│  CACAF   │◄───────────┘
                  │  (×4级)  │
                  │          │
                  │ 模态质量  │
                  │ 评估     │
                  │ +        │
                  │ 跨模态   │
                  │ 增强     │
                  │ +        │
                  │ 自适应   │
                  │ 融合     │
                  └────┬─────┘
                       │
                  融合特征 [f1, f2, f3, f4]
                       │
                       ▼
              ┌────────────────┐
              │    HGSOAD      │
              │    解码器      │
              │                │
              │  SAGU 门控     │
              │  + U-Net 上采样│
              │  + 辅助监督    │
              └───────┬────────┘
                      │
                      ▼
               分割掩码输出
```

### 3.2 编码器选择理由

**主编码器: SAM2 Hiera-L（冻结 + PEFT）**

| 选择理由 | 详细 |
|----------|------|
| 层级特征 | Hiera 天然输出 4 级多尺度特征，无需额外 FPN |
| 预训练质量 | SA-1B 数据集预训练，分割能力强 |
| PEFT 友好 | 冻结主干 + Adapter/LoRA，可训练参数少 |

**辅助编码器: ConvNeXt-Tiny（全量训练）**

| 选择理由 | 详细 |
|----------|------|
| 轻量 | 参数量 ~28M，远小于 DINOv2 ViT-L (~300M) |
| 4 级输出 | 天然输出 4 级特征 [96, 192, 384, 768] |
| ImageNet 预训练 | 有良好的初始化 |
| 独立训练 | 辅助模态（热成像/LiDAR）的分布与 RGB 不同，需要单独学习 |

**为什么去掉 DINOv2**:
- SAM2 Hiera-L 本身特征表示已足够强大
- DINOv2 ViT-L 增加 ~2-3GB 显存，但提升未必明显
- 审稿人会质疑三编码器的必要性 → 增加模块需要更多消融实验来证明
- 如实验发现精度不够，可作为"增强版"出现在消融实验中

### 3.3 模块详细设计

#### 3.3.1 CACAF 模块

**文件**: `segmentation/models/fusion/cacaf.py`

```python
class ModalQualityAssessment(nn.Module):
    """模态质量评估分支"""
    def __init__(self, in_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, 1)
        )

    def forward(self, x):
        # x: [B, C, H, W] → quality_score: [B, 1]
        out = self.gap(x).flatten(1)       # [B, C]
        score = self.mlp(out)              # [B, 1]
        return score


class CrossModalEnhancement(nn.Module):
    """轻量级跨模态交叉注意力"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.rgb2aux = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.aux2rgb = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_rgb = nn.LayerNorm(dim)
        self.norm_aux = nn.LayerNorm(dim)

    def forward(self, rgb_feat, aux_feat):
        # reshape: [B, C, H, W] → [B, HW, C]
        B, C, H, W = rgb_feat.shape
        rgb_seq = rgb_feat.flatten(2).permute(0, 2, 1)
        aux_seq = aux_feat.flatten(2).permute(0, 2, 1)

        # 双向交叉注意力
        rgb_enhanced = rgb_seq + self.rgb2aux(rgb_seq, aux_seq, aux_seq)[0]
        aux_enhanced = aux_seq + self.aux2rgb(aux_seq, rgb_seq, rgb_seq)[0]

        rgb_enhanced = self.norm_rgb(rgb_enhanced).permute(0,2,1).view(B,C,H,W)
        aux_enhanced = self.norm_aux(aux_enhanced).permute(0,2,1).view(B,C,H,W)
        return rgb_enhanced, aux_enhanced


class CACAF(nn.Module):
    """Condition-Aware Cross-modal Adaptive Fusion"""
    def __init__(self, rgb_channels, aux_channels, out_channels, num_heads=4):
        super().__init__()
        # 通道对齐
        self.align_rgb = nn.Conv2d(rgb_channels, out_channels, 1) if rgb_channels != out_channels else nn.Identity()
        self.align_aux = nn.Conv2d(aux_channels, out_channels, 1) if aux_channels != out_channels else nn.Identity()

        # 质量评估
        self.mqa_rgb = ModalQualityAssessment(out_channels)
        self.mqa_aux = ModalQualityAssessment(out_channels)

        # 跨模态增强
        self.cross_enhance = CrossModalEnhancement(out_channels, num_heads)

    def forward(self, rgb_feat, aux_feat):
        # 1. 通道对齐
        rgb_feat = self.align_rgb(rgb_feat)
        aux_feat = self.align_aux(aux_feat)

        # 2. 质量评估 → 自适应权重
        q_rgb = self.mqa_rgb(rgb_feat)         # [B, 1]
        q_aux = self.mqa_aux(aux_feat)         # [B, 1]
        weights = F.softmax(torch.cat([q_rgb, q_aux], dim=1), dim=1)  # [B, 2]
        w_rgb = weights[:, 0:1, None, None]    # [B, 1, 1, 1]
        w_aux = weights[:, 1:2, None, None]    # [B, 1, 1, 1]

        # 3. 跨模态增强
        rgb_enhanced, aux_enhanced = self.cross_enhance(rgb_feat, aux_feat)

        # 4. 自适应融合
        fused = w_rgb * rgb_enhanced + w_aux * aux_enhanced
        return fused, weights.detach()  # 返回权重用于可视化分析
```

**设计要点**:
- `ModalQualityAssessment` 极其轻量（两层 MLP），不增加显著计算
- 端到端训练，无需额外质量标注
- 返回 `weights` 用于论文中的可视化分析（不同天气条件下权重分布）

#### 3.3.2 HGSOAD 解码器

**文件**: `segmentation/models/decode_heads/hgsoad_head.py`

```python
class ScaleAwareGatingUnit(nn.Module):
    """尺度感知门控单元 SAGU"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        gate = self.sigmoid(avg_out + max_out)
        return x * gate


class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpBlock(nn.Module):
    """上采样 + skip connection + DoubleConv"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # 处理尺寸不对齐
        diffH = skip.size(2) - x.size(2)
        diffW = skip.size(3) - x.size(3)
        x = F.pad(x, [diffW // 2, diffW - diffW // 2,
                       diffH // 2, diffH - diffH // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class HGSOAD(nn.Module):
    """Hierarchical-Guided Small Object Aware Decoder"""
    def __init__(self, in_channels_list, num_classes, decode_channels=256):
        """
        Args:
            in_channels_list: 各级融合特征的通道数 [f1_ch, f2_ch, f3_ch, f4_ch]
            num_classes: 分割类别数
            decode_channels: 解码器内部通道数
        """
        super().__init__()
        c1, c2, c3, c4 = in_channels_list

        # 通道压缩到统一维度
        self.reduce4 = nn.Conv2d(c4, decode_channels, 1)
        self.reduce3 = nn.Conv2d(c3, decode_channels, 1)
        self.reduce2 = nn.Conv2d(c2, decode_channels, 1)
        self.reduce1 = nn.Conv2d(c1, decode_channels, 1)

        # 尺度感知门控
        self.sagu4 = ScaleAwareGatingUnit(decode_channels)
        self.sagu3 = ScaleAwareGatingUnit(decode_channels)
        self.sagu2 = ScaleAwareGatingUnit(decode_channels)
        self.sagu1 = ScaleAwareGatingUnit(decode_channels)

        # U-Net 上采样路径
        self.up3 = UpBlock(decode_channels, decode_channels, decode_channels)
        self.up2 = UpBlock(decode_channels, decode_channels, decode_channels)
        self.up1 = UpBlock(decode_channels, decode_channels, decode_channels)

        # 分类头
        self.cls_head = nn.Conv2d(decode_channels, num_classes, 1)

        # 辅助监督头（训练时使用，推理时关闭）
        self.aux_head2 = nn.Conv2d(decode_channels, num_classes, 1)
        self.aux_head3 = nn.Conv2d(decode_channels, num_classes, 1)

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, f4] 各级融合特征
        Returns:
            main_out: 主分割输出
            aux_outs: 辅助监督输出 (仅训练时)
        """
        f1, f2, f3, f4 = features

        # 通道压缩 + 门控
        f4 = self.sagu4(self.reduce4(f4))
        f3 = self.sagu3(self.reduce3(f3))
        f2 = self.sagu2(self.reduce2(f2))
        f1 = self.sagu1(self.reduce1(f1))

        # 层级上采样
        d3 = self.up3(f4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)

        # 主输出
        main_out = self.cls_head(d1)

        if self.training:
            aux_out2 = self.aux_head2(d2)
            aux_out3 = self.aux_head3(d3)
            return main_out, [aux_out2, aux_out3]
        return main_out
```

**设计要点**:
- SAGU 对高分辨率层（f1, f2）的门控会自动学习增强小目标响应
- SAGU 对低分辨率层（f3, f4）的门控会保留全局上下文
- 辅助监督加速收敛，推理时无额外开销
- 解码器通道统一为 `decode_channels=256`，平衡精度和效率

#### 3.3.3 SAM2 Hiera 适配器

**文件**: `segmentation/models/backbones/sam2_adapter.py`

```python
# 方案 A: Bottleneck Adapter（来自 SAM2-UNeXT/SAM2-UNet）
class BottleneckAdapter(nn.Module):
    def __init__(self, blk, bottleneck_dim=32):
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        return self.block(x + prompt)


# 方案 B: LoRA Adapter（来自 SAMed，适配 SAM2 Hiera）
class LoRA_HieraBlock(nn.Module):
    def __init__(self, qkv_layer, dim, r=4):
        super().__init__()
        self.qkv = qkv_layer
        self.dim = dim
        self.w_a_q = nn.Linear(dim, r, bias=False)
        self.w_b_q = nn.Linear(r, dim, bias=False)
        self.w_a_v = nn.Linear(dim, r, bias=False)
        self.w_b_v = nn.Linear(r, dim, bias=False)
        nn.init.zeros_(self.w_b_q.weight)
        nn.init.zeros_(self.w_b_v.weight)

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.w_b_q(self.w_a_q(x))
        new_v = self.w_b_v(self.w_a_v(x))
        qkv[:, :, :, :self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv
```

**论文中的消融实验**:
| 方案 | 可训练参数 | 预期 mIoU | 训练显存 |
|------|-----------|-----------|---------|
| Full Fine-tuning | ~300M | 最高但过拟合风险 | >24GB |
| Bottleneck Adapter (d=32) | ~2M | 与 Full FT 接近 | ~16GB |
| LoRA (r=4) | ~1M | 略低 | ~14GB |

---

## 4. 代码架构设计

### 4.1 项目文件结构

```
Drone-SAM-Adapter/
├── README.md
├── requirements.txt
├── docs/
│   ├── PLAN.md              (v1 原方案，保留作为参考)
│   ├── PLAN_v2.md           (本文档)
│   ├── ANALYSIS.md          (v1 分析，保留)
│   ├── ANALYSIS_v2.md       (新分析)
│   └── RESOURCE_INDEX.md    (资源索引，保留)
│
├── segmentation/
│   ├── models/
│   │   ├── backbones/
│   │   │   ├── sam2_adapter.py        # SAM2 Hiera + Adapter/LoRA
│   │   │   └── aux_encoder.py         # 辅助模态编码器 (ConvNeXt-Tiny)
│   │   ├── fusion/
│   │   │   └── cacaf.py               # [核心创新1] CACAF 融合模块
│   │   ├── decode_heads/
│   │   │   └── hgsoad_head.py         # [核心创新2] HGSOAD 解码器
│   │   └── segmentors/
│   │       └── cacaf_segmentor.py     # 整体分割器 (组装各模块)
│   │
│   ├── datasets/
│   │   ├── flm_dataset.py             # FLM (RGB+Thermal) 数据集
│   │   ├── deliver_dataset.py         # DELIVER (RGB+LiDAR) 数据集
│   │   └── pipelines/
│   │       ├── loading.py             # 多模态数据加载
│   │       └── transforms.py          # 数据增强
│   │
│   ├── train.py                       # 训练入口
│   ├── test.py                        # 测试入口
│   └── visualize.py                   # CACAF 权重可视化工具
│
├── configs/
│   ├── flm/
│   │   ├── cacaf_sam2_flm_rgb_thermal.py
│   │   └── baseline_mmsam_flm.py      # 基线复现配置
│   └── deliver/
│       ├── cacaf_sam2_deliver_rgb_lidar.py
│       └── baseline_mmsam_deliver.py   # 基线复现配置
│
├── checkpoints/                        # 预训练权重（不入库）
│   ├── sam2_hiera_large.pt
│   └── convnext_tiny_22k.pth
│
├── tools/
│   ├── convert_sam2_checkpoint.py      # SAM2 权重转换
│   └── analyze_fusion_weights.py       # 融合权重分析脚本
│
└── data/ → /home/jl/dataset            # 软链接到数据集
```

### 4.2 框架选择

**推荐: 基于 Multimodal-SAM-Adapter 的 MMSegmentation 框架**

| 理由 | 详细 |
|------|------|
| 代码复用 | 直接复用基线的训练/测试流程、数据管道、分布式训练 |
| 公平对比 | 使用相同框架确保实验对比公平 |
| 论文可信度 | 审稿人更信任标准框架下的对比实验 |

**但需注意**: MM SAM-Adapter 使用的 MMSeg 版本较旧 (0.20.2)，我们需要评估是否升级。考虑到兼容性，建议**保持与基线相同的 MMSeg 版本**。

### 4.3 依赖需求

```
# requirements.txt
# ---- 核心框架 ----
torch>=2.0.0
torchvision>=0.15.0
mmcv-full==1.4.2
mmsegmentation==0.20.2
mmengine>=0.8.0

# ---- 模型 ----
timm>=0.9.0                  # ConvNeXt-Tiny 预训练权重
einops>=0.6.0                # tensor 操作

# ---- 数据 ----
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
albumentations>=1.3.0        # 数据增强

# ---- 工具 ----
tensorboard                  # 训练监控
prettytable                  # 评估结果展示
scipy                        # 统计分析
matplotlib                   # 可视化
```

**SAM2 依赖**: 不通过 pip 安装 SAM2 完整包，而是直接提取 Hiera 编码器代码（参考 SAM2-UNeXT 的做法），避免依赖冲突。

### 4.4 预训练权重

| 模型 | 来源 | 大小 | 用途 |
|------|------|------|------|
| SAM2 Hiera-L | Meta (dl.fbaipublicfiles.com) | ~890MB | 主编码器，冻结 |
| ConvNeXt-Tiny (ImageNet-22K) | timm hub | ~110MB | 辅助编码器，训练 |

---

## 5. 实验设计

### 5.1 主实验

#### 数据集配置

| 数据集 | 模态对 | 类别 | 训练集 | 验证/测试集 | 输入尺寸 | 角色 |
|--------|--------|------|--------|-------------|---------|------|
| **FLM** | RGB + Thermal | 14 | 1,220 | 280 (test) | 800×600 → crop 到 512×512 或 pad 到 800×800 | **主实验** |
| **DELIVER** | RGB + LiDAR | 25 | ~4,000 | ~2,000 / ~1,900 | 1042×1042 → crop 到 1024×1024 | **泛化验证** |

#### 对比方法

| 方法 | 类型 | 来源 |
|------|------|------|
| SegFormer-B2 (RGB only) | 单模态基线 | 公开预训练 |
| CMX | 多模态融合 | TPAMI 2023 |
| CMNeXt | 多模态融合 | NeurIPS 2023 |
| MM SAM-Adapter | SAM 多模态适配 | **直接基线** |
| Ours (CACAF-Net) | 本文方法 | - |

### 5.2 消融实验（论文关键部分）

#### 模块消融

| 实验ID | CACAF | HGSOAD | Adapter | 预期结论 |
|--------|-------|--------|---------|----------|
| A1 | ✗ (简单 concat) | ✗ (SegFormer Head) | Bottleneck | 基线级别 |
| A2 | ✓ | ✗ (SegFormer Head) | Bottleneck | CACAF 的独立贡献 |
| A3 | ✗ (简单 concat) | ✓ | Bottleneck | HGSOAD 的独立贡献 |
| A4 | ✓ | ✓ | Bottleneck | **完整方法（最优）** |

#### CACAF 内部消融

| 实验ID | MQA (质量评估) | 跨模态增强 | 预期结论 |
|--------|---------------|-----------|----------|
| C1 | ✗ (等权 0.5/0.5) | ✓ | 证明自适应权重的价值 |
| C2 | ✓ | ✗ (简单加法) | 证明跨模态增强的价值 |
| C3 | ✓ | ✓ | **完整 CACAF** |

#### HGSOAD 内部消融

| 实验ID | SAGU 门控 | 辅助监督 | 预期结论 |
|--------|----------|---------|----------|
| H1 | ✗ | ✗ | 朴素 U-Net 解码 |
| H2 | ✓ | ✗ | 门控的价值 |
| H3 | ✗ | ✓ | 辅助监督的价值 |
| H4 | ✓ | ✓ | **完整 HGSOAD** |

#### PEFT 策略对比

| 方案 | 可训练参数 | mIoU | 训练显存 |
|------|-----------|------|---------|
| Full Fine-tuning | ~300M | - | - |
| Bottleneck Adapter (d=32) | ~2M | - | - |
| Bottleneck Adapter (d=64) | ~4M | - | - |
| LoRA (r=4) | ~1M | - | - |
| LoRA (r=8) | ~2M | - | - |

### 5.3 分析实验（增强论文深度）

#### 5.3.1 CACAF 融合权重可视化

**实验设计**:
1. 在 DELIVER 测试集上按天气条件分组 (cloud/fog/night/rain/sun)
2. 统计每组的平均 w_rgb 和 w_aux
3. 绘制箱线图或热力图

**预期结果**:
```
天气条件    w_rgb (RGB权重)    w_aux (辅助模态权重)
sun         0.65 ± 0.08       0.35 ± 0.08      ← RGB 主导
cloud       0.58 ± 0.10       0.42 ± 0.10
rain        0.45 ± 0.12       0.55 ± 0.12      ← 趋向均衡
fog         0.35 ± 0.10       0.65 ± 0.10      ← 辅助模态主导
night       0.30 ± 0.12       0.70 ± 0.12      ← 辅助模态主导
```

**这张图将是论文的亮点图之一**，直观展示"条件感知"的效果。

#### 5.3.2 Per-Condition 性能分析

在 DELIVER 上按天气条件报告 mIoU:

| 方法 | Sun | Cloud | Rain | Fog | Night | 平均 |
|------|-----|-------|------|-----|-------|------|
| MM SAM-Adapter | - | - | - | - | - | 57.14 |
| Ours | - | - | - | - | - | - |
| Δ (改进) | - | - | - | - | - | - |

**预期**: 在 Fog 和 Night 条件下的改进幅度 > Sun 和 Cloud。

#### 5.3.3 小目标类别分析

在 FLM 上报告小目标类别（Person, Motorcycle, Bicycle）的 IoU:

| 方法 | Person | Motorcycle | Bicycle | 其他类平均 |
|------|--------|-----------|---------|-----------|
| MM SAM-Adapter | - | - | - | - |
| Ours | - | - | - | - |

### 5.4 评估指标

| 指标 | 说明 | 使用场景 |
|------|------|---------|
| **mIoU** | 所有类别的平均交并比 | 主指标 |
| **mAcc** | 所有类别的平均准确率 | 辅助指标 |
| **aAcc** | 全局像素准确率 | 辅助指标 |
| **per-class IoU** | 每个类别的 IoU | 小目标分析 |
| **FPS** | 推理帧率 | 效率对比 |
| **Params** | 可训练参数量 | PEFT 对比 |
| **GFLOPs** | 计算量 | 效率对比 |

---

## 6. 训练配置

### 6.1 FLM 数据集训练配置

```python
# configs/flm/cacaf_sam2_flm_rgb_thermal.py

# 模型
model = dict(
    type='CacafSegmentor',
    backbone=dict(
        type='SAM2HieraAdapter',
        checkpoint='checkpoints/sam2_hiera_large.pt',
        adapter_type='bottleneck',  # 或 'lora'
        adapter_dim=32,
        freeze_backbone=True,
    ),
    aux_backbone=dict(
        type='ConvNeXtTiny',
        pretrained='checkpoints/convnext_tiny_22k.pth',
        freeze=False,
    ),
    fusion=dict(
        type='CACAF',
        num_heads=4,
    ),
    decode_head=dict(
        type='HGSOAD',
        num_classes=14,
        decode_channels=256,
        aux_loss_weight=0.4,
    ),
)

# 数据
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(type='FLMDataset', split='train', modalities=['rgb', 'thermal']),
    val=dict(type='FLMDataset', split='test', modalities=['rgb', 'thermal']),
)

# 优化器
optimizer = dict(
    type='AdamW', lr=2e-4, weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),      # SAM2 主干低学习率
            'aux_backbone': dict(lr_mult=1.0),   # 辅助编码器正常学习率
            'fusion': dict(lr_mult=1.0),
            'decode_head': dict(lr_mult=1.0),
        }
    )
)

# 学习率策略
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=True)

# 训练
runner = dict(type='EpochBasedRunner', max_epochs=100)
optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=2)
# 有效 batch size = 2 × 2 (累积) = 4

# 混合精度
fp16 = dict(loss_scale='dynamic')
```

### 6.2 显存估算 (4090, 24GB)

| 组件 | 参数量 | 显存估算 (FP16) |
|------|--------|-----------------|
| SAM2 Hiera-L (frozen) | ~300M | ~600MB (仅前向) |
| SAM2 Adapter 参数 | ~2M | ~50MB |
| ConvNeXt-Tiny (trainable) | ~28M | ~2GB (前向+梯度+优化器) |
| CACAF (×4 级) | ~8M | ~1GB |
| HGSOAD | ~10M | ~1GB |
| 特征图 + 激活值 | - | ~6-8GB |
| **合计** | ~348M (可训练 ~48M) | **~12-14GB** |

**结论**: 4090 (24GB) 充裕，batch_size=2 + 梯度累积=2 可行。

---

## 7. 论文大纲

### 建议的论文结构

```
Title: Condition-Aware Adaptive Multimodal Semantic Segmentation
       under Adverse Conditions

1. Introduction
   - 恶劣条件对语义分割的挑战
   - 多模态融合的必要性
   - 现有方法的局限（静态融合、小目标不敏感）
   - 本文贡献（3 点）

2. Related Work
   2.1 多模态语义分割 (CMX, CMNeXt, MM SAM-Adapter, ...)
   2.2 SAM/SAM2 在分割中的应用
   2.3 恶劣条件下的感知

3. Method
   3.1 Overall Architecture
   3.2 SAM2 Hiera Backbone with PEFT
   3.3 CACAF: Condition-Aware Cross-modal Adaptive Fusion
       3.3.1 Modal Quality Assessment
       3.3.2 Cross-Modal Enhancement
       3.3.3 Adaptive Fusion
   3.4 HGSOAD: Hierarchical-Guided Small Object Aware Decoding
       3.4.1 Scale-Aware Gating Unit
       3.4.2 Hierarchical Upsampling with Auxiliary Supervision
   3.5 Loss Function

4. Experiments
   4.1 Datasets and Implementation Details
   4.2 Comparison with State-of-the-Art
       - FLM (RGB+Thermal)
       - DELIVER (RGB+LiDAR)
   4.3 Ablation Studies
       - 模块消融
       - CACAF 内部消融
       - HGSOAD 内部消融
       - PEFT 策略对比
   4.4 Analysis
       - 融合权重可视化（per-condition）
       - 小目标类别分析
       - 定性结果（分割可视化）

5. Conclusion
```

### 论文贡献总结 (Introduction 最后一段)

```
本文的主要贡献如下:
1. 提出了 CACAF 模块，通过模态质量评估分支实现条件感知的自适应
   多模态融合，使模型在不同退化条件下自动调整模态贡献权重。
2. 提出了 HGSOAD 解码器，通过尺度感知门控单元增强小目标的分割
   精度，同时利用辅助监督加速训练收敛。
3. 在 FLM 和 DELIVER 两个数据集上进行了广泛实验，验证了方法在
   RGB+热成像和 RGB+LiDAR 两种多模态场景下的有效性和泛化能力。
```

---

## 8. 实施计划（修订版）

| 阶段 | 时间 | 核心任务 | 产出 | 验收标准 |
|------|------|---------|------|---------|
| **环境搭建** | Week 1 | 跑通 MM SAM-Adapter 基线 (FLM+DELIVER) | 基线复现结果 | mIoU 与论文报告接近 |
| **核心模块开发** | Week 2-3 | 实现 CACAF + HGSOAD + SAM2 Adapter 集成 | 可运行的完整模型 | 单次前向传播无错误 |
| **FLM 训练** | Week 4 | FLM 数据集训练 + 调参 | 训练好的模型 | mIoU ≥ 基线 |
| **DELIVER 训练** | Week 5 | DELIVER 数据集训练 | 训练好的模型 | mIoU ≥ 基线 |
| **消融实验** | Week 6 | 全部消融实验 | 消融结果表格 | 每个模块有正贡献 |
| **分析实验** | Week 7 | 权重可视化 + per-condition 分析 + 定性结果 | 分析图表 | 支撑论文叙事 |
| **论文写作** | Week 8-9 | 撰写论文 + 画图 | 论文初稿 | 完整可投稿 |

### 风险与对策（修订版）

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| 基线复现结果不达标 | 中 | 高 | 严格对齐基线代码和配置，必要时联系原作者 |
| CACAF 权重无法学到条件感知 | 中 | 高 | 增加 MQA 分支容量，或引入显式退化检测预训练 |
| HGSOAD 对小目标改进不明显 | 中 | 中 | 调整 SAGU 结构，或引入 Focal Loss 辅助 |
| FLM 数据集太小 (1220 张) | 低 | 中 | 加强数据增强 (RandomCrop, ColorJitter, MixUp) |
| 显存超限 | 低 | 中 | 降低 decode_channels 或 batch_size |

---

## 9. 成功标准

### MVP (最低可发表标准)

- [ ] FLM 上 mIoU ≥ MM SAM-Adapter 基线
- [ ] DELIVER 上 mIoU ≥ MM SAM-Adapter 基线
- [ ] 消融实验证明 CACAF 和 HGSOAD 各自有正贡献
- [ ] CACAF 权重可视化能展示条件感知趋势

### 目标标准

- [ ] FLM 上 mIoU 超过基线 2%+
- [ ] DELIVER 上 mIoU 超过基线 1%+
- [ ] 恶劣条件子集 (fog/night) 上优势明显 (3%+)
- [ ] 小目标类别 IoU 提升 3%+
- [ ] 可训练参数 < 50M

---

## 10. 与 v1 方案的关键变更对照

| 方面 | v1 方案 | v2 方案 | 变更理由 |
|------|--------|--------|---------|
| **论文定位** | 无人机搜救 | 恶劣条件多模态分割 | 数据集不支撑"搜救"定位 |
| **创新点数量** | 4 个（分散） | 2 核心 + 1 辅助（聚焦） | 审稿人偏好深度 > 广度 |
| **主编码器** | SAM2 Hiera-L | SAM2 Hiera-L | 保持 |
| **辅助编码器** | DINOv2 ViT-L (~300M) | ConvNeXt-Tiny (~28M) | 省显存，降复杂度 |
| **第三编码器** | 无（但暗示 TwinConvNeXt） | 无 | 去掉不必要的模块 |
| **融合模块** | 简单双向注意力 | CACAF（条件感知自适应） | 更有原创性和论文故事 |
| **解码器** | 轻量 U-Net（深度可分离卷积） | HGSOAD（门控 + U-Net + 辅助监督） | 从工程优化改为方法创新 |
| **边缘部署** | 作为创新点 | 不作为创新点 | 实验无法验证 |
| **数据集** | 未确定 | FLM（主）+ DELIVER（泛化） | 明确且可执行 |
| **框架** | 从零搭建 | 基于 MM SAM-Adapter 改进 | 减少工程量，确保公平对比 |
| **时间规划** | 10 周 | 9 周（更紧凑） | 去掉部署优化阶段 |


核心问题诊断
你现在最大的短板不是分数，是故事完整性。CCFC 审稿人通常是领域内的，会追问：

为什么选 FMB 这个数据集？有没有在其他数据集上验证？
CACAF 和基线融合方法（如 CMX、TokenFusion 等）相比优势在哪？
参数量/推理速度 vs 性能的 tradeoff 是否合理？
提升中稿率的具体策略
最高优先级：加一个数据集

单数据集结果在 CCFC 审稿中很容易被质疑泛化性。FMB 之外，RGB-Thermal 分割常用的还有：

数据集	规模	说明
MFNet	1569张，9类	最常用，几乎所有对比方法都有结果
PST900	894张，5类	较新，竞争对手少
MFNet 最值得加——基准方法齐全，好对比，而且你的模型架构不需要改，直接换数据集训练即可。代价是再跑一次训练（约6h）+ 消融（可以只跑 Full/NoCACAF/NoSAGU 三个）。

次优先级：补充 SOTA 对比表

论文里要有一张 State-of-the-Art 对比表，列出近3年的方法（CMX、TokenFusion、GMNet、EGFNet 等），这些结果可以直接从对应论文里抄（FMB 有公开排行榜）。不需要自己跑，只需要引用。

写作层面：突出无人机场景特殊性

很多 RGB-Thermal 融合论文针对地面摄像头，你的无人机场景有两个可以强调的差异点：

俯视角导致目标尺度更小（支撑 SAGU 的必要性）
光照变化更剧烈（支撑 CACAF 质量感知的必要性）
这两点要在 Introduction 和 Ablation 分析里明确呼应。

我的判断
条件	中稿概率估计
现状（单数据集，无 SOTA 对比）	~20-25%
加 MFNet 结果 + SOTA 对比表	~45-55%
再加消融自洽（V6 重跑）	~55-65%
加 MFNet 是性价比最高的一步，一次训练就能把中稿率从两成涨到接近五成。

你有兴趣先在 MFNet 上跑一次看看结果吗？还是先等 V5-NoCACAF-NoSAGU 跑完再做决定？