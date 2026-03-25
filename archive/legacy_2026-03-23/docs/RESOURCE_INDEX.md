# SAM项目代码资源索引

> **用途**: 为大模型提供结构化的代码资源索引，便于快速定位和复用各SAM项目的代码模块

**更新日期**: 2026-03-04
**版本**: v1.0

---

## 目录结构

```
├── 1. 项目概览表
├── 2. 功能模块索引
│   ├── 2.1 编码器模块 (Encoders)
│   ├── 2.2 解码器模块 (Decoders)
│   ├── 2.3 融合模块 (Fusion)
│   ├── 2.4 适配器模块 (Adapters)
│   ├── 2.5 注意力机制 (Attention)
│   └── 2.6 工具函数 (Utils)
├── 3. 文件路径映射表
├── 4. 代码片段引用指南
├── 5. 快速查找功能
└── 6. 类继承关系图
```

---

## 1. 项目概览表

| 项目 | 路径 | 主要特性 | 适用场景 |
|------|------|----------|----------|
| **SAM2-UNeXT** | `~/SAM2-UNeXT/` | SAM2+DINOv2双编码器 | 多模态分割 |
| **SAM2-UNet** | `~/SAM2-UNet/` | SAM2+RFB模块 | 小目标检测 |
| **SAMed** | `~/SAMed/` | LoRA微调 | 医学图像 |
| **Mult-scale-SAM** | `~/Mult-scale-SAM/` | 多尺度融合 | 遥感图像 |
| **MMSAM-Adapter** | `~/Multimodal-SAM-Adapter/` | 多模态适配器 | RGB+LiDAR/Depth |

---

## 2. 功能模块索引

### 2.1 编码器模块 (Encoders)

#### SAM2 Hiera Encoder

| 项目 | 文件路径 | 类名 | 功能描述 |
|------|----------|------|----------|
| SAM2-UNeXT | `SAM2UNeXT.py:82-110` | `SAM2UNeXT.__init__` | SAM2编码器初始化 + Adapter包装 |
| SAM2-UNet | `SAM2UNet.py:124-151` | `SAM2UNet.__init__` | SAM2编码器 + 删除无关模块 |
| SAM2-UNeXT | `sam2/modeling/backbones/hieradet.py` | `Hiera` | SAM2 Hiera主干网络 |

**关键代码片段** (SAM2-UNeXT):
```python
# 路径: ~/SAM2-UNeXT/SAM2UNeXT.py:82-110
model = build_sam2("sam2_hiera_l.yaml", checkpoint_path)
del model.sam_mask_decoder
del model.sam_prompt_encoder
del model.memory_encoder
del model.memory_attention
del model.mask_downsample
del model.obj_ptr_tpos_proj
del model.obj_ptr_proj
del model.image_encoder.neck
self.sam = model.image_encoder.trunk
```

#### DINOv2 Encoder

| 项目 | 文件路径 | 类名 | 功能描述 |
|------|----------|------|----------|
| SAM2-UNeXT | `SAM2UNeXT.py:112-124` | `SAM2UNeXT.__init__` | DINOv2辅助编码器初始化 |

**关键代码片段**:
```python
# 路径: ~/SAM2-UNeXT/SAM2UNeXT.py:112-124
self.dino = timm.create_model('vit_large_patch14_dinov2',
                              features_only=True,
                              img_size=(448, 448),
                              pretrained=True)
for param in self.dino.parameters():
    param.requires_grad = False
```

#### SAM ViT Encoder (原始SAM)

| 项目 | 文件路径 | 类名 | 功能描述 |
|------|----------|------|----------|
| SAMed | `segment_anything/modeling/image_encoder.py` | `ImageEncoderViT` | SAM ViT图像编码器 |
| MMSAM-Adapter | `mmseg_custom/models/backbones/image_encoder.py` | `ImageEncoderViT` | 带窗口注意力的ViT |

---

### 2.2 解码器模块 (Decoders)

#### U-Net Decoder

| 项目 | 文件路径 | 类名 | 功能描述 |
|------|----------|------|----------|
| SAM2-UNet | `SAM2UNet.py:7-48` | `DoubleConv` | 双卷积块 |
| SAM2-UNet | `SAM2UNet.py:27-48` | `Up` | 上采样模块 |
| SAM2-UNeXT | `SAM2UNeXT.py:40-79` | `Up` | 上采样模块 (支持可选跳跃连接) |

**关键代码片段** (SAM2-UNet):
```python
# 路径: ~/SAM2-UNet/SAM2UNet.py:27-48
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
```

#### Segformer Head

| 项目 | 文件路径 | 类名 | 功能描述 |
|------|----------|------|----------|
| MMSAM-Adapter | `mmseg_custom/models/decode_heads/segformer_head.py` | `SegformerHead` | Segformer MLP解码头 |

---

### 2.3 融合模块 (Fusion)

#### 多模态融合

| 项目 | 文件路径 | 类名/函数 | 功能描述 |
|------|----------|-----------|----------|
| MMSAM-Adapter | `mmseg_custom/models/backbones/adapter_modules_multimodal_mix_mod_new_in_twin_convnext_new.py` | `TwinConvNeXt` | 双流ConvNeXt融合 |
| MMSAM-Adapter | 同上文件 | `MSDeformAttn` | 多尺度变形注意力 |
| Mult-scale-SAM | `segment_anything/modeling_CNN/image_encoder_prompt.py:398-452` | `PoolingAttention` | 多尺度池化注意力 |

**多尺度池化注意力代码**:
```python
# 路径: ~/Mult-scale-SAM/segment_anything/modeling_CNN/image_encoder_prompt.py
class PoolingAttention(nn.Module):
    # 多尺度池化 [1, 2, 4, 8]
    def __init__(self, dim, pool_ratios=[1,2,4,8], ...):
        self.pool_ratios = pool_ratios
        # 实现多尺度自适应池化和注意力聚合
```

#### 特征对齐与融合

| 项目 | 文件路径 | 类名 | 功能描述 |
|------|----------|------|----------|
| SAM2-UNeXT | `SAM2UNeXT.py:126-134` | `align1-4`, `reduce1-4` | 双编码器特征对齐 |
| SAM2-UNeXT | `SAM2UNeXT.py:143-160` | `forward` | 双编码器特征融合 |

**关键代码片段**:
```python
# 路径: ~/SAM2-UNeXT/SAM2UNeXT.py:143-160
def forward(self, x):
    # SAM2特征
    x1_s, x2_s, x3_s, x4_s = self.sam(x)
    # DINOv2特征
    x_d = self.dino(F.interpolate(x, size=(448, 448), mode='bilinear'))[-1]

    # 特征对齐
    x1_d = F.interpolate(self.align1(x_d), size=x1_s.shape[-2:], mode='bilinear')
    x2_d = F.interpolate(self.align2(x_d), size=x2_s.shape[-2:], mode='bilinear')
    x3_d = F.interpolate(self.align3(x_d), size=x3_s.shape[-2:], mode='bilinear')
    x4_d = F.interpolate(self.align4(x_d), size=x4_s.shape[-2:], mode='bilinear')

    # 特征融合
    x1, x2, x3, x4 = torch.cat([x1_s,x1_d], dim=1), torch.cat([x2_s,x2_d], dim=1), ...
```

---

### 2.4 适配器模块 (Adapters)

#### Lightweight Adapter (SAM2-UNeXT/SAM2-UNet)

| 项目 | 文件路径 | 类名 | 功能描述 |
|------|----------|------|----------|
| SAM2-UNeXT | `SAM2UNeXT.py:9-37` | `Adapter` | 轻量化提示学习适配器 |
| SAM2-UNet | `SAM2UNet.py:51-67` | `Adapter` | 同上 (相同实现) |

**关键代码片段**:
```python
# 路径: ~/SAM2-UNeXT/SAM2UNeXT.py:9-37
class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
```

**应用方式**:
```python
# 路径: ~/SAM2-UNeXT/SAM2UNeXT.py:103-110
blocks = []
for block in self.sam.blocks:
    blocks.append(Adapter(block))
self.sam.blocks = nn.Sequential(*blocks)
```

#### LoRA Adapter (SAMed)

| 项目 | 文件路径 | 类名 | 功能描述 |
|------|----------|------|----------|
| SAMed | `sam_lora_image_encoder.py:17-48` | `_LoRA_qkv` | LoRA QKV包装器 |
| SAMed | `sam_lora_image_encoder.py:51-187` | `LoRA_Sam` | SAM模型的LoRA包装 |

**LoRA关键代码**:
```python
# 路径: ~/SAMed/sam_lora_image_encoder.py:17-48
class _LoRA_qkv(nn.Module):
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv
```

**LoRA应用方式**:
```python
# 路径: ~/SAMed/sam_lora_image_encoder.py:88-108
for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
    if t_layer_i not in self.lora_layer:
        continue
    w_qkv_linear = blk.attn.qkv
    self.dim = w_qkv_linear.in_features
    w_a_linear_q = nn.Linear(self.dim, r, bias=False)
    w_b_linear_q = nn.Linear(r, self.dim, bias=False)
    w_a_linear_v = nn.Linear(self.dim, r, bias=False)
    w_b_linear_v = nn.Linear(r, self.dim, bias=False)
    blk.attn.qkv = _LoRA_qkv(w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v)
```

---

### 2.5 注意力机制 (Attention)

#### 多尺度变形注意力 (MSDeformAttn)

| 项目 | 文件路径 | 用途 |
|------|----------|------|
| MMSAM-Adapter | `mmseg_custom/models/backbones/adapter_modules_multimodal_mix_mod_new_in_twin_convnext_new.py:7` | 跨模态注意力 |
| Mult-scale-SAM | `segment_anything/modeling_CNN/transformer.py` | 可变形Transformer |

**导入方式**:
```python
# 路径: ~/Multimodal-SAM-Adapter/.../adapter_modules_multimodal_mix_mod_new_in_twin_convnext_new.py
from ops.modules import MSDeformAttn
```

#### 窗口注意力

| 项目 | 文件路径 | 类名 | 功能描述 |
|------|----------|------|----------|
| MMSAM-Adapter | `mmseg_custom/models/backbones/image_encoder.py` | `WindowAttention` | 基于窗口的多头注意力 |

---

### 2.6 小目标增强模块

#### RFB (Receptive Field Block)

| 项目 | 文件路径 | 类名 | 功能描述 |
|------|----------|------|----------|
| SAM2-UNet | `SAM2UNet.py:85-121` | `RFB_modified` | 多感受野模块 |

**关键代码片段**:
```python
# 路径: ~/SAM2-UNet/SAM2UNet.py:85-121
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        # 分支0: 1x1
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        # 分支1: 1x1 + 3x3 (dilation=3)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        # 分支2: dilation=5
        self.branch2 = nn.Sequential(...)
        # 分支3: dilation=7
        self.branch3 = nn.Sequential(...)
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x
```

**RFB应用**:
```python
# 路径: ~/SAM2-UNet/SAM2UNet.py:152-155
self.rfb1 = RFB_modified(144, 64)
self.rfb2 = RFB_modified(288, 64)
self.rfb3 = RFB_modified(576, 64)
self.rfb4 = RFB_modified(1152, 64)
```

---

### 2.7 工具函数 (Utils)

#### 模型构建

| 项目 | 文件路径 | 函数/类 | 功能 |
|------|----------|---------|------|
| SAM2-UNeXT | `sam2/build_sam.py` | `build_sam2` | 构建SAM2模型 |
| SAM2-UNet | `sam2/build_sam.py` | `build_sam2` | 同上 |
| SAMed | `segment_anything/build_sam.py` | `build_sam` | 构建SAM模型 |
| SAMed | `segment_anything/sam_model_registry.py` | `sam_model_registry` | SAM模型注册表 |

#### 数据加载与增强

| 项目 | 文件路径 | 类/函数 | 功能 |
|------|----------|---------|------|
| MMSAM-Adapter | `mmseg_custom/datasets/DELIVER.py` | `DELIVER` | 多模态数据集 |
| MMSAM-Adapter | `mmseg_custom/datasets/pipelines/loading.py` | `LoadImageandModalities3ch` | 多模态数据加载 |
| MMSAM-Adapter | `mmseg_custom/datasets/pipelines/transform.py` | 数据增强类 | 多模态数据增强 |
| Mult-scale-SAM | `remote_dataset/transforms.py` | 各种Transform | 遥感图像增强 |

#### 训练/测试脚本

| 项目 | 文件路径 | 功能 |
|------|----------|------|
| SAM2-UNeXT | `train.py`, `test.py`, `eval.py` | 训练、测试、评估 |
| SAM2-UNet | `train.py`, `test.py`, `eval.py` | 同上 |
| SAMed | `train.py` | 训练脚本 |
| MMSAM-Adapter | `segmentation/train.py`, `test.py`, `infer_test.py` | 同上 |

---

## 3. 文件路径映射表

### 3.1 按项目分类

#### SAM2-UNeXT
```
~/SAM2-UNeXT/
├── SAM2UNeXT.py                    # 主模型文件 ⭐
├── train.py                        # 训练脚本
├── test.py                         # 测试脚本
├── eval.py                         # 评估脚本
├── dataset.py                      # 数据集
├── sam2_configs/                   # SAM2配置
└── sam2/
    ├── build_sam.py                # SAM2构建函数
    ├── modeling/
    │   ├── backbones/
    │   │   └── hieradet.py         # Hiera主干
    │   └── sam2_base.py            # SAM2基础模型
    └── utils/
```

#### SAM2-UNet
```
~/SAM2-UNet/
├── SAM2UNet.py                     # 主模型文件 ⭐
├── train.py                        # 训练脚本
├── test.py                         # 测试脚本
├── eval.py                         # 评估脚本
├── eval_noncam.py                  # 非camo数据集评估
├── dataset.py                      # 数据集
└── sam2/                           # SAM2代码 (同SAM2-UNeXT)
```

#### SAMed
```
~/SAMed/
├── sam_lora_image_encoder.py       # LoRA实现 ⭐
├── sam_lora_image_encoder_mask_decoder.py  # LoRA+掩码解码器
├── train.py                        # 训练脚本
├── utils.py                        # 工具函数
├── preprocess/
│   └── preprocess_data.py          # 数据预处理
└── segment_anything/
    ├── build_sam.py                # SAM构建
    ├── modeling/
    │   ├── image_encoder.py        # ViT编码器
    │   ├── mask_decoder.py         # 掩码解码器
    │   └── sam.py                  # SAM主模型
    └── utils/
```

#### Mult-scale-SAM
```
~/Mult-scale-SAM/
├── train.py                        # 训练脚本
├── loss.py                         # 损失函数
├── remote_dataset/                 # 遥感数据集
│   ├── transforms.py               # 数据增强
│   └── build_dataset.py            # 数据集构建
└── segment_anything/
    └── modeling_CNN/
        ├── image_encoder_prompt.py # 多尺度编码器 ⭐
        ├── ppm_sam.py              # 金字塔池化模块
        └── transformer.py          # Transformer组件
```

#### Multimodal-SAM-Adapter
```
~/Multimodal-SAM-Adapter/
└── segmentation/
    ├── train.py                    # 训练脚本
    ├── test.py                     # 测试脚本
    ├── infer_test.py               # 推理脚本
    ├── mmseg_custom/
    │   ├── models/
    │   │   ├── backbones/
    │   │   │   ├── adapter_modules_multimodal_mix_mod_new_in_twin_convnext_new.py  # 多模态适配器 ⭐
    │   │   │   ├── image_encoder.py  # ViT编码器
    │   │   │   └── twin_convnext.py   # 双流ConvNeXt
    │   │   ├── decode_heads/
    │   │   │   └── segformer_head.py  # Segformer解码头
    │   │   └── segmentors/
    │   │       └── encoder_decoder.py  # 编码器-解码器框架
    │   └── datasets/
    │       ├── DELIVER.py             # DELIVER数据集 ⭐
    │       └── pipelines/
    │           ├── loading.py         # 数据加载
    │           └── transform.py       # 数据增强
    └── configs/
        └── DELIVER/
            └── Segformer_MMSAM_adapter_large_DELIVER_1024x1024_ss_RGBLIDAR.py  # 配置文件 ⭐
```

### 3.2 按功能分类

| 功能 | 文件路径 | 项目 |
|------|----------|------|
| **SAM2编码器** | `SAM2UNeXT.py:82-110` | SAM2-UNeXT |
| **DINOv2编码器** | `SAM2UNeXT.py:112-124` | SAM2-UNeXT |
| **SAM ViT编码器** | `segment_anything/modeling/image_encoder.py` | SAMed |
| **Lightweight Adapter** | `SAM2UNeXT.py:9-37` | SAM2-UNeXT |
| **LoRA Adapter** | `sam_lora_image_encoder.py:17-187` | SAMed |
| **RFB模块** | `SAM2UNet.py:85-121` | SAM2-UNet |
| **U-Net解码器** | `SAM2UNet.py:7-48` | SAM2-UNet |
| **多尺度池化注意力** | `segment_anything/modeling_CNN/image_encoder_prompt.py:398-452` | Mult-scale-SAM |
| **多模态数据集** | `mmseg_custom/datasets/DELIVER.py` | MMSAM-Adapter |
| **变形注意力** | `adapter_modules_multimodal_mix_mod_new_in_twin_convnext_new.py` | MMSAM-Adapter |

---

## 4. 代码片段引用指南

### 4.1 如何复用SAM2编码器

```python
# 来源: ~/SAM2-UNeXT/SAM2UNeXT.py:82-110
from sam2.build_sam import build_sam2

# 1. 构建SAM2模型
model_cfg = "sam2_hiera_l.yaml"
model = build_sam2(model_cfg, checkpoint_path)

# 2. 删除不需要的模块
del model.sam_mask_decoder
del model.sam_prompt_encoder
del model.memory_encoder
del model.memory_attention
del model.mask_downsample
del model.obj_ptr_tpos_proj
del model.obj_ptr_proj
del model.image_encoder.neck

# 3. 提取编码器主干
encoder = model.image_encoder.trunk

# 4. 冻结参数
for param in encoder.parameters():
    param.requires_grad = False
```

### 4.2 如何添加Adapter

```python
# 来源: ~/SAM2-UNeXT/SAM2UNeXT.py:9-37 + 103-110
class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

# 应用到编码器
blocks = []
for block in encoder.blocks:
    blocks.append(Adapter(block))
encoder.blocks = nn.Sequential(*blocks)
```

### 4.3 如何集成DINOv2

```python
# 来源: ~/SAM2-UNeXT/SAM2UNeXT.py:112-124
import timm
import torch.nn.functional as F

# 创建DINOv2模型
dino = timm.create_model('vit_large_patch14_dinov2',
                         features_only=True,
                         img_size=(448, 448),
                         pretrained=True)

# 冻结参数
for param in dino.parameters():
    param.requires_grad = False

# 使用时需要调整输入尺寸
def forward(self, x):
    # 调整输入到448x448
    x_448 = F.interpolate(x, size=(448, 448), mode='bilinear')
    dino_feat = self.dino(x_448)[-1]  # 取最后一层特征
    return dino_feat
```

### 4.4 如何实现RFB模块

```python
# 来源: ~/SAM2-UNet/SAM2UNet.py:85-121
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        # 分支0: 1x1
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        # 分支1: dilation=3
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        # 分支2: dilation=5
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        # 分支3: dilation=7
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x
```

### 4.5 如何实现U-Net上采样

```python
# 来源: ~/SAM2-UNet/SAM2UNet.py:27-48
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
```

### 4.6 如何加载多模态数据

```python
# 来源: ~/Multimodal-SAM-Adapter/segmentation/mmseg_custom/datasets/DELIVER.py
@DATASETS.register_module()
class DELIVER(CustomDataset):
    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='_rgb_front.png',
                 modalities_name=['rgb', 'lidar'],  # 模态名称
                 modalities_ch=[3, 3],               # 各模态通道数
                 mod_dir=['samples/depth/training', 'samples/lidar/training'],
                 mod_suffix=['_depth_front.png', '_lidar_front.png'],
                 **kwargs):

        self.modalities_name = modalities_name
        self.modalities_ch = modalities_ch
        self.mod_dir = mod_dir
        self.mod_suffix = mod_suffix
        # ... 其他初始化

    def load_annotations_modalities(self, img_dir, img_suffix, mod_dir_dict,
                                    mod_suffix_dict, modalities_name, ...):
        """加载多模态数据标注"""
        img_infos = []
        # ... 实现多模态文件路径加载
        return img_infos
```

---

## 5. 快速查找功能

### 5.1 按需求查找

| 需求 | 推荐项目 | 文件路径 |
|------|----------|----------|
| **双编码器架构** | SAM2-UNeXT | `SAM2UNeXT.py:82-160` |
| **小目标检测** | SAM2-UNet | `SAM2UNet.py:85-173` |
| **参数高效微调** | SAMed | `sam_lora_image_encoder.py:51-187` |
| **多模态融合** | MMSAM-Adapter | `adapter_modules_multimodal_mix_mod_new_in_twin_convnext_new.py` |
| **多尺度特征** | Mult-scale-SAM | `segment_anything/modeling_CNN/image_encoder_prompt.py:398-452` |
| **U-Net解码器** | SAM2-UNet | `SAM2UNet.py:7-48` |
| **轻量化Adapter** | SAM2-UNeXT | `SAM2UNeXT.py:9-37` |
| **数据集实现** | MMSAM-Adapter | `mmseg_custom/datasets/DELIVER.py` |
| **训练脚本** | SAM2-UNeXT | `train.py` |
| **评估脚本** | SAM2-UNet | `eval.py` |

### 5.2 按模块查找

| 模块 | 包含项目 |
|------|----------|
| **SAM2编码器** | SAM2-UNeXT, SAM2-UNet |
| **SAM ViT编码器** | SAMed, MMSAM-Adapter |
| **DINOv2** | SAM2-UNeXT |
| **Adapter** | SAM2-UNeXT, SAM2-UNet, SAMed |
| **RFB** | SAM2-UNet |
| **U-Net** | SAM2-UNeXT, SAM2-UNet |
| **多模态数据集** | MMSAM-Adapter |
| **多尺度注意力** | Mult-scale-SAM |

### 5.3 关键类索引

| 类名 | 项目 | 文件 |
|------|------|------|
| `SAM2UNeXT` | SAM2-UNeXT | `SAM2UNeXT.py:82` |
| `SAM2UNet` | SAM2-UNet | `SAM2UNet.py:124` |
| `Adapter` | SAM2-UNeXT/SAM2-UNet | `SAM2UNeXT.py:9` |
| `LoRA_Sam` | SAMed | `sam_lora_image_encoder.py:51` |
| `RFB_modified` | SAM2-UNet | `SAM2UNet.py:85` |
| `Up` | SAM2-UNeXT/SAM2-UNet | `SAM2UNeXT.py:40` |
| `DoubleConv` | SAM2-UNeXT/SAM2-UNet | `SAM2UNeXT.py:7` |
| `DELIVER` | MMSAM-Adapter | `mmseg_custom/datasets/DELIVER.py:22` |
| `PoolingAttention` | Mult-scale-SAM | `segment_anything/modeling_CNN/image_encoder_prompt.py:398` |

---

## 6. 类继承关系图

```
编码器继承关系:
├── SAM2 Hiera (SAM2-UNeXT/SAM2-UNet)
│   └── build_sam2() -> image_encoder.trunk
│
├── SAM ViT (SAMed/MMSAM-Adapter)
│   └── ImageEncoderViT
│       ├── PatchEmbed
│       └── Transformer blocks
│
└── DINOv2 (SAM2-UNeXT)
    └── timm.create_model('vit_large_patch14_dinov2')


解码器继承关系:
├── U-Net Decoder (SAM2-UNeXT/SAM2-UNet)
│   ├── DoubleConv
│   └── Up
│
└── Segformer Head (MMSAM-Adapter)
    └── SegformerHead


适配器继承关系:
├── Lightweight Adapter (SAM2-UNeXT/SAM2-UNet)
│   └── Adapter(nn.Module)
│       └── prompt_learn (Sequential)
│
└── LoRA Adapter (SAMed)
    └── LoRA_Sam(nn.Module)
        └── _LoRA_qkv(nn.Module)
            ├── linear_a_q/v
            └── linear_b_q/v


增强模块:
└── RFB (SAM2-UNet)
    └── RFB_modified(nn.Module)
        ├── branch0-3 (多感受野分支)
        ├── conv_cat (特征聚合)
        └── conv_res (残差连接)
```

---

## 7. 配置文件索引

| 项目 | 配置文件路径 | 用途 |
|------|-------------|------|
| SAM2-UNeXT | `sam2_configs/sam2_hiera_l.yaml` | SAM2 L配置 |
| MMSAM-Adapter | `configs/DELIVER/Segformer_MMSAM_adapter_large_DELIVER_1024x1024_ss_RGBLIDAR.py` | 完整训练配置 |
| MMSAM-Adapter | `configs/_base_/models/segformer_mit-b0.py` | 基础模型配置 |
| MMSAM-Adapter | `configs/_base_/datasets/DELIVER_MM.py` | 数据集配置 |

---

## 8. 预训练权重索引

| 模型 | 下载链接 | 存放位置 |
|------|----------|----------|
| SAM2 Hiera-L | [link](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt) | `checkpoints/` |
| SAM ViT-H | [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | `checkpoints/` |
| DINOv2 ViT-L | [link](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) | `checkpoints/` |

---

## 9. 使用建议

### 9.1 组合推荐

**无人机多模态分割推荐组合**:
```
编码器: SAM2-UNeXT 的双编码器
├── SAM2 Hiera (SAM2-UNeXT)
├── DINOv2 辅助 (SAM2-UNeXT)
└── Adapter包装 (SAM2-UNeXT)

增强: RFB模块
└── RFB_modified (SAM2-UNet)

解码器: U-Net
└── Up + DoubleConv (SAM2-UNet)

微调: LoRA (可选)
└── LoRA_Sam (SAMed)

数据: 多模态数据集
└── DELIVER类 (MMSAM-Adapter)
```

### 9.2 优先级排序

**如果时间有限，按以下优先级参考**:
1. **SAM2-UNeXT** - 最接近目标架构，双编码器设计完善
2. **SAM2-UNet** - RFB模块和U-Net解码器实现清晰
3. **MMSAM-Adapter** - 多模态数据管道和融合实现
4. **SAMed** - LoRA微调方案（如果需要参数高效）

---

## 附录: 项目依赖关系

```
依赖关系图:

sam2 (官方)
├── SAM2-UNeXT ─────┐
└── SAM2-UNet ──────┤
                      ├── Drone-SAM-Adapter (新项目)
                      │
segment_anything (官方)│
├── SAMed ───────────┤
│                     │
timm (DINOv2) ────────┘
└── SAM2-UNeXT ──────┘

MMSegmentation框架
└── MMSAM-Adapter ───┘
```

---

**文档维护**: 请在添加新代码或修改现有代码时更新此索引
**反馈**: 如发现错误或遗漏，请及时修正
