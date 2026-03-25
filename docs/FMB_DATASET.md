# FMB 数据集说明
> 实测整理，2026-03-05
> 最近更新，2026-03-10（训练/评估参数同步）

---

## 基本信息

| 项目 | 值 |
|------|-----|
| 数据集路径 | `/home/jl/dataset/FMB/` |
| 模态 | RGB + Thermal（红外，3-ch RGB 格式存储） |
| 类别数 | 14（0-indexed，无 void/ignore 类） |
| 原始分辨率 | 800 × 600 |
| 训练用尺寸 | 512 × 512（随机裁剪） |
| 评估用尺寸 | 512 × 512（默认 letterbox 保比例；可切换 stretch） |

---

## 数据划分

| 划分 | Easy | Hard | 合计 |
|------|------|------|------|
| train | 752 | 308 | **1060** |
| val   | 80  | 80  | **160** |
| test  | 183 | 97  | **280** |
| **总计** | | | **1500** |

> 注：快照中"训练 1220 张"= train(1060) + val(160)，test 280 张单独用于最终评估。

---

## 实际目录结构

```
/home/jl/dataset/FMB/
├── train_easy_files.txt        # 文件名列表，如 00001.png（根目录）
├── train_hard_files.txt
├── val_easy_files.txt
├── val_hard_files.txt
├── test_easy_files.txt
├── test_hard_files.txt
├── train/
│   ├── Visible/
│   │   ├── easy/  *.png        # RGB（800×600，3-ch）
│   │   ├── hard/  *.png
│   │   ├── train_easy_files.txt  # 同根目录，重复
│   │   └── train_hard_files.txt
│   ├── Infrared/  *.png        # Thermal（800×600，3-ch，灰度以 RGB 存储，flat）
│   └── Label/     *.png        # GT mask（800×600，L mode，uint8）
├── val/
│   ├── Visible/{easy,hard}/
│   ├── Infrared/               # flat
│   └── Label/                  # flat
└── test/
    ├── Visible/{easy,hard}/
    ├── Infrared/               # flat
    └── Label/                  # flat
```

**关键点**：
- RGB 有 `easy/hard` 子目录；Infrared 和 Label **没有**，直接 flat 存放
- split 文件在**根目录**和 `{split}/Visible/` 下各有一份，内容相同
- `FMBDataset` 使用根目录的 split 文件

---

## 图像格式

| 类型 | PIL mode | 通道 | 数值范围 | 说明 |
|------|----------|------|----------|------|
| RGB | `'RGB'` | 3 | 0-255 uint8 | 标准彩色图像 |
| Thermal | `'RGB'` | 3 | 0-255 uint8 | 灰度信息以 3-ch RGB 存储（三通道相同） |
| Label | `'L'` | 1 | 0-13 uint8 | 直接类别索引，**无** 255 ignore |

---

## 14 类别定义

| ID | 类别 | 说明 |
|----|------|------|
| 0 | Road | 道路 |
| 1 | Sidewalk | 人行道 |
| 2 | Building | 建筑 |
| 3 | Traffic Light | 交通灯 |
| 4 | Traffic Sign | 交通标志 |
| 5 | Vegetation | 植被 |
| 6 | Sky | 天空 |
| 7 | Person | 行人 |
| 8 | Car | 小汽车 |
| 9 | Truck | 卡车 |
| 10 | Bus | 公交车 |
| 11 | Motorcycle | 摩托车 |
| 12 | Bicycle | 自行车 ← 基线 IoU=0.0，核心难点 |
| 13 | Pole | 杆柱 |

---

## 归一化参数

```python
# RGB：ImageNet 标准
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]

# Thermal：均值归一化到 0
THM_MEAN = [0.5, 0.5, 0.5]
THM_STD  = [0.5, 0.5, 0.5]
```

---

## 数据增强（训练）

| 步骤 | 操作 | 参数 |
|------|------|------|
| 1 | Random Scale Resize | 原图 × scale，scale ∈ [0.5, 2.0]；resize 后 min(H,W) ≥ 512 |
| 2 | RandomGaussianBlur（可开关） | 默认 p=0.2，RGB+Thm 同步 |
| 3 | Pad if needed | 不足 512 的边补 0（RGB/Thm）/ 255（Label） |
| 4 | RandomCrop + 类别占比约束（可开关） | 512 × 512，`cat_max_ratio` 默认 0.75 |
| 5 | RandomHorizontalFlip | p=0.5，RGB+Thm+Label 同步 |
| 6 | PhotoMetricDistortion（可开关） | 默认开启（brightness/contrast/saturation/hue） |
| 7 | Normalize | 各自 mean/std |

评估：
- `letterbox`（默认）：保持长宽比 resize，再 pad 到 512×512
- `stretch`：直接 resize 到 512×512（旧行为）

---

## 2026-03-10：新增训练/评估参数

### `FMBDataset` 新增参数

```python
FMBDataset(
    ...,
    eval_resize_mode="letterbox",  # "letterbox" | "stretch"
    cat_max_ratio=0.75,            # 随机裁剪中单类最大占比约束，1.0 表示关闭
    blur_prob=0.2,                 # 高斯模糊概率
    photo_distort=True,            # 强 photometric distortion
)
```

### `train_cacaf.py` 对应新增参数

```bash
--use-ohem --ohem-thresh 0.7 --ohem-min-kept 100000
--eval-resize-mode letterbox
--cat-max-ratio 0.75
--blur-prob 0.2
--photo-distort
```

---

## Loader 接口

```python
from segmentation.datasets.fmb_dataset import FMBDataset, build_fmb_dataloaders

# 单 Dataset
ds = FMBDataset(
    root='/home/jl/dataset/FMB',
    split='train',
    crop_size=512,
    augment=True,
    eval_resize_mode='letterbox',
    cat_max_ratio=0.75,
    blur_prob=0.2,
    photo_distort=True,
)
rgb_t, thm_t, lbl_t = ds[0]
# rgb_t: [3,512,512] float32   thm_t: [3,512,512] float32   lbl_t: [512,512] int64

# 一次性获取三个 DataLoader
train_loader, val_loader, test_loader = build_fmb_dataloaders(
    root='/home/jl/dataset/FMB',
    crop_size=512,
    batch_size=4,
    num_workers=4,
)
```

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `segmentation/datasets/fmb_dataset.py` | FMB Dataset + DataLoader 工厂，已验证 ✅ |
| `segmentation/train_cacaf.py` | 独立训练脚本 |
| `segmentation/mmseg_custom/datasets/FMB_val.py` | 基线 MMSeg 数据集类（参考） |
