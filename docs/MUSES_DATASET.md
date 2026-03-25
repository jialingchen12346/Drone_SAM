# MUSES 数据集探索记录
> 探索时间: 2026-03-06
> 状态: 待训练（数据集已分析，训练环境待搭建）

---

## 数据集概况

**MUSES** (Multi-Sensor Event-based Semantic Segmentation) 是一个基于事件相机的多模态语义分割数据集，面向自动驾驶场景。

| 项目 | 值 |
|------|-----|
| 数据集路径 | `/home/jl/dataset/muses` |
| 分辨率 | 1920×1080 |
| 数据量 | train=1500, val=250, test=750 |
| 模态 | RGB (frame_camera) + Event Camera (h5) |
| 标注标准 | Cityscapes（19 类，trainId 0-18） |
| 场景类型 | 街景平视（vs FMB 无人机俯视） |
| 天气条件 | clear, fog, rain, snow |
| 时间 | day, night |

---

## 目录结构

```
/home/jl/dataset/muses/
├── frame_camera/          # RGB 图像
│   ├── train/
│   │   ├── clear/day/*.png
│   │   ├── clear/night/*.png
│   │   ├── fog/day/*.png
│   │   ├── rain/day/*.png
│   │   └── snow/day/*.png
│   ├── val/
│   └── test/
├── event_camera/          # 事件相机数据
│   ├── train/
│   │   └── {weather}/{time}/*.h5
│   ├── val/
│   └── test/
├── gt_semantic/           # 语义分割标签
│   ├── train/
│   │   └── {weather}/{time}/*_gt_labelIds.png
│   └── val/
└── muses/
    ├── calib.json         # 相机标定
    └── meta.json          # 元数据（3.3MB）
```

**文件命名规则**:
- RGB: `REC{id}_frame_{timestamp}_frame_camera.png`
- Event: `REC{id}_frame_{timestamp}_event_camera.h5`
- Label: `REC{id}_frame_{timestamp}_gt_labelIds.png`

---

## 数据格式

### RGB 图像
- 格式: PNG, 3 通道
- 尺寸: 1920×1080
- 色彩空间: RGB

### Event Camera (h5)
Event Camera 数据以 HDF5 格式存储，包含事件流：

```python
# h5 文件结构
events/
  ├── x: [N] uint16      # 事件 x 坐标 (0-1919)
  ├── y: [N] uint16      # 事件 y 坐标 (0-1079)
  ├── t: [N] uint32      # 时间戳 (微秒)
  └── p: [N] uint8       # 极性 (0/1)
ms_to_idx: [30] uint64   # 毫秒到事件索引的映射
t_offset: int64          # 时间偏移
```

**示例**（`REC0241_frame_503673_event_camera.h5`）:
- 事件数量: 292,645 个事件
- 时间窗口: ~30ms

### 语义标签
- 格式: PNG, 单通道 uint8
- 尺寸: 1920×1080
- 标注标准: **Cityscapes labelId**（原始 ID，0-255）
- 实际出现的 ID: 58 个不同值（需映射到 19 类 trainId）

**Val 集出现的类别 ID**:
```
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 35, 60, 64, 70, 80, 100, 102, 107, 119, 128, 130, 142, 152, 153, 156, 170, 180, 190, 220, 230, 232, 244, 250, 251, 255]
```

**Cityscapes trainId 映射**（需实现）:
- 0-18: 有效类别（road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle）
- 255: ignore

---

## 与 FMB 数据集对比

| 项目 | FMB | MUSES |
|------|-----|-------|
| 分辨率 | 800×600 | 1920×1080 |
| 类别数 | 14 | 19 (Cityscapes) |
| 训练集 | 1060 | 1500 |
| 验证集 | 160 | 250 |
| 测试集 | 280 | 750 |
| 辅助模态 | Thermal (3-ch PNG) | Event Camera (h5) |
| 场景 | 无人机俯视 | 街景平视 |
| 标签格式 | 1-indexed (1-14) | Cityscapes labelId |

**关键差异**:
1. **分辨率更高**: 1920×1080 vs 800×600，显存需求更大
2. **类别不同**: 19 类 vs 14 类，**FMB 模型无法直接迁移**
3. **模态不同**: Event Camera (事件流) vs Thermal (密集图像)
4. **视角不同**: 平视街景 vs 俯视无人机，语义分布差异大

---

## 训练适配要点

### 1. Dataset Loader 需求
- 解析 h5 事件流 → 转换为密集表示（event frame/voxel grid）
- Cityscapes labelId → trainId 映射（参考 Cityscapes 官方映射表）
- 处理 4 种天气 × 2 种时间的子目录结构
- 数据增强：需考虑 1920×1080 → 512×512 的 resize/crop 策略

### 2. 模型适配
- `num_classes=19`（vs FMB 的 14）
- 辅助编码器输入：Event Camera 密集表示（需设计转换方法）
- 推理分辨率：SAM2 仍限制为 512×512，预测后 upsample 到 1920×1080

### 3. 训练配置建议
- Batch size: 2-3（1920×1080 显存需求大）
- 或使用 crop 训练（512×512 或 1024×1024）
- Epochs: 50-100
- 预计训练时间: 3-4 小时（1500 样本，batch=4，512×512 crop）

### 4. Event Camera 处理方案
**方案 A**: Event Frame（推荐，简单）
```python
# 累积事件到 2D 图像
event_frame = np.zeros((H, W), dtype=np.float32)
for x, y, p in zip(events['x'], events['y'], events['p']):
    event_frame[y, x] += (1 if p else -1)
# 归一化 → 3 通道（复制）
```

**方案 B**: Voxel Grid（更精细）
```python
# 时间分箱到 B 个 bins
voxel_grid = np.zeros((B, H, W))
for x, y, t, p in events:
    bin_idx = int((t - t_min) / (t_max - t_min) * B)
    voxel_grid[bin_idx, y, x] += (1 if p else -1)
# 降维到 3 通道（如取前 3 个 bins）
```

---

## 待实现文件清单

### 必需文件
- [ ] `segmentation/datasets/muses_dataset.py` — MUSES Dataset Loader
  - Event Camera h5 解析
  - Cityscapes labelId → trainId 映射
  - 天气/时间子目录遍历
- [ ] `segmentation/train_muses.py` — MUSES 训练脚本
  - 复用 `CACafSegmentor`（改 num_classes=19）
  - 调整 batch size/crop size
- [ ] `scripts/eval_muses.py` — MUSES 评估脚本
  - 全分辨率推理（512×512 → upsample 1920×1080）
  - Cityscapes 标准指标（mIoU, fwIoU）

### 可选文件
- [ ] `scripts/visualize_muses.py` — 可视化（RGB/Event/GT/Pred）
- [ ] `scripts/analyze_weather.py` — 按天气条件分析性能

---

## Cityscapes 类别映射表（参考）

| trainId | name | labelId (部分) |
|---------|------|----------------|
| 0 | road | 7 |
| 1 | sidewalk | 8 |
| 2 | building | 11 |
| 3 | wall | 12 |
| 4 | fence | 13 |
| 5 | pole | 17, 18 |
| 6 | traffic light | 19 |
| 7 | traffic sign | 20 |
| 8 | vegetation | 21 |
| 9 | terrain | 22 |
| 10 | sky | 23 |
| 11 | person | 24 |
| 12 | rider | 25 |
| 13 | car | 26 |
| 14 | truck | 27 |
| 15 | bus | 28 |
| 16 | train | 31 |
| 17 | motorcycle | 32 |
| 18 | bicycle | 33 |
| 255 | ignore | 0, 1, 2, ... |

完整映射表见 Cityscapes 官方仓库：
https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

---

## 依赖安装

```bash
conda activate sam2-unet
pip install h5py  # 已安装（2026-03-06）
```

---

## 参考资料

- MUSES 论文: [待补充]
- Cityscapes 数据集: https://www.cityscapes-dataset.com/
- Event Camera 综述: [待补充]

---

## 备注

- **当前状态**: 数据集已探索，训练环境待搭建
- **优先级**: 低（FMB 项目优先，MUSES 作为扩展实验）
- **预期收益**: 验证 CACAF 在街景 + Event Camera 上的泛化能力
- **风险**: Event Camera 数据处理可能需要调试，显存需求更高
