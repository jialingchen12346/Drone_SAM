# 架构调研笔记

日期：2026-03-23

主题：在旧结果统计口径不一致的前提下，RGB-T 语义分割新模型应如何重设计

## 调研问题

在以下约束下，什么样的新结构最值得做第一轮重训：

- 保留 SAM2 / SAM 风格 RGB 主编码器的优势
- 工程改动控制在可迭代范围内
- 增强模态质量变化下的鲁棒性
- 改善小目标、细结构和边界表现

## 核心结论

文献整体支持三个判断。

第一，RGB-T 语义分割的融合设计已经从“更强地融合”转向“更有选择地融合”。现在更被重视的问题是模态污染、模态偏置和对单一传感器的过度依赖，而不是单纯堆更多注意力模块。

第二，解码器仍然很重要。多篇分割论文都表明，细节流和语义流解耦、边界引导融合，在小目标和细结构场景下仍然是很强且实用的设计模式。

第三，SAM / SAM2 适配类方法更适合把创新放在“模态注入、融合过滤、任务解码”上，而不是马上推翻整个 backbone。因此，新一轮最合理的创新核心应是“区域级可靠性融合 + 细节/语义解耦解码”，而不是再做一版全局标量权重融合。

## 关键发现

### 1. RGB-T 融合研究的主线已经变成“选择性融合”

- **CSRPNet** 的出发点是：朴素双向融合会把一条模态的噪声污染到另一条模态，因此先传播共享的通道/空间关系，再做增强。
- **VPFNet** 将融合特征视为随机变量，强调对模态噪声、类别不平衡和模态偏置的鲁棒性。
- **CRM** 用互补随机遮挡和自蒸馏，直接解决模型过度依赖单一模态的问题。
- **RTFDNet** 则进一步把 fusion 和 decoupling 放到同一个鲁棒性框架内考虑。

直接含义：

- 新模型最好显式建模“模态可靠性”或“共享/特有信息分解”，而不是只再加一条 cross-attention。

### 2. 高分辨率细节仍然是决定成败的瓶颈

- **FEANet** 直接强调 RGB-T 实时分割常常损失空间细节，因此重点增强高分辨率特征。
- **PIDNet** 虽然不是 RGB-T 论文，但它有一个非常有用的结论：高分辨率细节和低频语义如果直接粗暴相加，细节会被压制，因此需要独立分支和额外引导。
- **DDRNet** 说明双分辨率处理加多次双向交互，是精度与效率之间很稳的一种折中。

直接含义：

- 你当前的 HGSOAD 方向是对的，但还不够强。下一版应该把“细节”和“语义上下文”明确拆开，而不是只做 channel gate。

### 3. SAM / SAM2 适配方法支持“保留 backbone，创新放在注入和解码”

- **MM SAM-Adapter** 的核心是把多模态特征注入到 SAM 的 RGB 特征里，同时尽量保留 RGB 预训练泛化能力。
- **SAM2-Adapter** 说明轻量适配在复杂下游分割任务上依然有效。
- **ClassWise-SAM-Adapter** 说明 decoder / task head 的设计，在 foundation model 适配里依旧是主要增益来源之一。
- **SHIFNet** 则进一步引入 text-guided balancing 和异构 prompting decoder，说明最新 RGB-T + SAM2 研究正在往“语义引导式模态平衡”发展。

直接含义：

- 第一轮新研究继续保留 SAM2 + adapter 是合理约束。
- 新颖性应主要放在 fusion 和 decoder，而不是马上重写 encoder。

### 4. 最新 RGB-T 工作进一步强化了两个方向

- **TUNI** 倾向于统一式多模态特征提取，并加入局部跨模态相似性融合。
- **SGFNet** 提出从频域上区分高频细节和低频上下文，再分别建模交互。
- **RTFDNet** 则强调部分传感器失效场景下的鲁棒性，需要 decoupling regularization。

直接含义：

- 当前最值得押注的两条线是：
  1. 区域级/局部级可靠性融合
  2. 细节与上下文解耦

## 方法脉络

### A. 直接融合或简单融合

- 早期代表如 **RTFNet**，证明 RGB+thermal 双流优于简单四通道输入，但融合方式仍较直接。

结论：

- 这一类已经不再足够构成论文创新点。

### B. 注意力/增强型融合

- **FEANet**：通过增强式注意力保留空间细节。
- **CSRPNet**：通过关系传播提取共享信息。
- **MMSFormer**：用 transformer 做多模态融合。

结论：

- 单纯“用了注意力”已经没有足够区分度。
- 关键在于模块是否能清晰解释：如何避免模态污染，或如何利用局部互补性。

### C. 鲁棒性导向的多模态学习

- **VPFNet**：概率式融合
- **CRM**：模态遮挡 + 自蒸馏
- **RTFDNet**：融合与解耦联合正则

结论：

- “对模态质量变化更稳”是当前最有说服力的叙事之一。

### D. Foundation model 适配

- **MM SAM-Adapter**：将多模态特征注入 SAM
- **SAM2-Adapter**：adapter 形式的 SAM2 下游适配
- **CWSAM**：PEFT + classwise decoder + 低频注入
- **SHIFNet**：SAM2 + 语义引导 + 语言指导

结论：

- backbone 已经能作为可信基线，真正的差异化来自如何注入辅助模态、如何过滤噪声、如何做任务解码。

## 对本项目最值得做的方向

## 方向 1：区域级可靠性融合

核心思想：

- 用每个尺度上的空间可靠性图，替换当前“每尺度一个 RGB/T 标量权重”。

为什么值得做：

- 更贴近 FMB 的真实场景：thermal 不一定整张图都更好，它往往只在局部区域、小目标或低照区域更可靠。
- 也更符合当前文献对“模态偏置、污染、鲁棒性”的关注。

最低风险实现：

- 对 RGB / T 特征先通道对齐
- 预测一个 `2-channel spatial reliability map`
- 在空间位置上对两模态做 softmax 加权
- 加一层轻量局部交互模块，而不是重型全局 attention

研究价值：

- 比当前全局标量式 MQA 更强，也更容易自圆其说。

## 方向 2：细节-语义解耦解码器

核心思想：

- 解码时分出 detail branch 和 semantic branch，并让语义或边界信息反向引导细节恢复。

为什么值得做：

- FEANet 指向空间细节保留的重要性。
- PIDNet / DDRNet 说明细节和语义不该过早、过粗地合并。

最低风险实现：

- `f1` 保留为细节主流
- `f4 -> f3 -> f2` 组成语义主流
- 在 stride 4 处做 guided merge
- 最后输出分类头

研究价值：

- 对 pole、traffic light、person、边界类目标的解释力比 SAGU 更强。

## 方向 3：不扩大结构规模的鲁棒训练

核心思想：

- 训练期加入模态随机遮挡或 branch dropout，减少 RGB 主导，提升退化情况下的稳定性。

为什么值得做：

- 直接对应 CRM 和 RTFDNet 的思路。

最低风险实现：

- 训练时随机遮挡 thermal、遮挡 RGB，或只遮挡局部区域
- 可选再加 clean / masked prediction consistency loss

研究价值：

- 可以和方向 1 天然组合，而且消融成本低。

## 当前不建议优先做的方向

### 1. 再做一版全局标量质量融合

- 与现有 CACAF 太近。
- 文献方向已经往局部可靠性、鲁棒性和解耦走了。

### 2. 第一轮就做完整语言引导式重构

- SHIFNet 很有意思，但 text encoder、prompting、语义嵌入会显著扩大工程范围。
- 更适合做第二轮扩展，而不是第一轮主线。

### 3. 立刻替换整个 encoder 栈

- TUNI 的统一式结构值得关注，但改动太大，不利于快速迭代和归因。

## 推荐的新一轮最小方案

1. 保留 `SAM2 Hiera + adapter` 作为 RGB encoder
2. 保留 `ConvNeXt-Tiny` 作为 thermal encoder
3. 用以下结构替换 CACAF：
   - 通道对齐
   - 局部交互
   - 空间可靠性图
4. 用以下结构替换 HGSOAD：
   - detail branch
   - semantic branch
   - guided merge head
5. 训练阶段加入可开关的模态 masking

## 论文层面的创新点表述

- **创新点 1**：区域级可靠性融合，实现按位置自适应的模态选择
- **创新点 2**：细节-语义解耦解码器，强化小目标和细结构恢复
- **辅助训练策略**：互补模态遮挡，提升退化条件下的鲁棒性

这套表述明显强于：

- 仅做全局质量权重
- 仅做通道注意力门控
- 再堆一个泛化的 cross-attention 融合块

## 当前开放问题

- 可靠性图应当每尺度独立预测，还是由深层语义做自顶向下调制
- FMB 上是否值得加显式 boundary supervision
- 局部交互模块应优先用卷积、窗口注意力，还是相似度驱动混合

## 参考来源

[1] CSRPNet: https://arxiv.org/abs/2308.12534  
[2] VPFNet: https://arxiv.org/abs/2307.08536  
[3] CRM: https://arxiv.org/abs/2303.17386  
[4] RTFDNet: https://arxiv.org/abs/2603.09149  
[5] FEANet: https://arxiv.org/abs/2110.08988  
[6] PIDNet: https://arxiv.org/abs/2206.02066  
[7] DDRNet: https://arxiv.org/abs/2101.06085  
[8] MM SAM-Adapter repo: https://github.com/iacopo97/Multimodal-SAM-Adapter  
[9] MM SAM-adapter paper: https://arxiv.org/abs/2509.10408  
[10] SAM2-Adapter: https://arxiv.org/abs/2408.04579  
[11] ClassWise-SAM-Adapter: https://arxiv.org/abs/2401.02326  
[12] SHIFNet: https://arxiv.org/abs/2503.02581  
[13] TUNI: https://arxiv.org/abs/2509.10005  
[14] SGFNet: https://arxiv.org/abs/2505.15491  
