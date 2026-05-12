# U-Net 与改进 U-Net 模型总结

## 1. 任务定义

本项目面向二维截面流场到三维截面流场的预测任务。模型输入为二维速度场中的两个速度分量，输出为目标三维截面上的三个速度分量。

输入：

```text
X ∈ R^(2 × 48 × 48)
```

其中两个通道分别表示二维截面速度分量 `u` 和 `v`。

输出：

```text
Y_hat ∈ R^(3 × 64 × 48)
```

其中三个通道分别表示目标截面速度分量 `u`、`v` 和 `w`。

项目中已将旧版 U-Net 保存为 `models/unet_model_V0.py`，当前实际训练与预测调用的改进模型为 `models/unet_model.py`。当前改进模型可概括为 **MedNeXt-UNet-Lift**：它保留 U-Net 的编码器-解码器和跳跃连接框架，但将基础卷积模块、下采样方式、归一化方式和输出高度映射方式都进行了改进。

## 2. 原始 U-Net V0 结构

旧版模型位于 `models/unet_model_V0.py`。其结构为标准 U-Net，主要由 `DoubleConv`、`Down`、`Up` 和输出卷积组成。

整体流程：

```text
Input 2×48×48
→ DoubleConv(2, 64)
→ Down(64, 128)
→ Down(128, 256)
→ Down(256, 512)
→ Down(512, 512)
→ Up(1024, 256)
→ Up(512, 128)
→ Up(256, 64)
→ Up(128, 64)
→ 1×1 Conv(64, 3)
→ Bilinear Upsample to 3×64×48
```

V0 模型的 `DoubleConv` 为两次连续的 `3×3 Conv + BatchNorm + ReLU`。下采样使用 `MaxPool2d(2)`，上采样使用双线性插值，编码器与解码器之间通过通道拼接形式的 skip connection 连接。输出端先通过 `1×1 Conv2d` 得到 `3×48×48`，再用固定双线性插值调整为 `3×64×48`。

V0 的优点是结构清晰、参数路径稳定，适合作为基线模型。但它也存在几个局限：基础卷积感受野较小；BatchNorm 对小 batch 训练较敏感；MaxPool 下采样不可学习；输出高度方向仅依赖固定插值，难以学习从 `48` 到 `64` 的非线性空间映射。

## 3. 当前改进 U-Net：MedNeXt-UNet-Lift

当前模型位于 `models/unet_model.py`，文件顶部标注为 `MedNeXt_UNet_Lift`。它在 U-Net 主框架上引入了 MedNeXt 风格卷积块和可学习高度提升头，整体结构如下：

```text
Input 2×48×48
→ Stem: Conv 3×3 + GroupNorm + GELU + MedNeXtBlock ×2
→ Down1: learned residual downsample, 64 → 128, 48×48 → 24×24
→ Down2: learned residual downsample, 128 → 256, 24×24 → 12×12
→ Down3: learned residual downsample, 256 → 512, 12×12 → 6×6
→ Down4: learned residual downsample, 512 → 512, 6×6 → 3×3
→ Bottleneck: MedNeXtBlock ×3
→ Up1: bilinear upsample + skip fusion, 512 + 512 → 256, 3×3 → 6×6
→ Up2: bilinear upsample + skip fusion, 256 + 256 → 128, 6×6 → 12×12
→ Up3: bilinear upsample + skip fusion, 128 + 128 → 64, 12×12 → 24×24
→ Up4: bilinear upsample + skip fusion, 64 + 64 → 64, 24×24 → 48×48
→ HeightLiftingHead: learned height mapping, 64×48×48 → 3×64×48
```

### 3.1 MedNeXtBlock2D

改进模型将旧版 `DoubleConv` 内部的普通卷积替换为 MedNeXt 风格模块。单个 `MedNeXtBlock2D` 的结构为：

```text
Depthwise large-kernel Conv
→ GroupNorm
→ 1×1 pointwise expansion
→ GELU
→ Global Response Normalization
→ 1×1 pointwise projection
→ Residual connection
```

该模块的主要特点包括：

- 使用 depthwise large-kernel convolution 扩大局部感受野，同时降低卷积计算量。
- 使用 `GroupNorm` 替代 `BatchNorm`，降低小 batch 训练时统计量不稳定的问题。
- 使用 `GELU` 替代 `ReLU`，使激活函数更平滑，有利于连续流场回归。
- 引入 `GRN`，即 Global Response Normalization，用于增强通道响应的全局竞争与特征校准。
- 使用残差连接缓解深层网络训练中的梯度传播问题。

### 3.2 GRN 全局响应归一化

当前模型实现了 `GRN` 模块。其思想来自 ConvNeXt V2，并适配到二维特征图。对于输入特征 `x ∈ R^(B×C×H×W)`，GRN 首先计算每个通道的空间 L2 范数：

```text
G_x = ||x||_2 over spatial dimensions
```

然后根据通道间均值进行归一化，最后通过可学习参数 `gamma` 和 `beta` 对输入响应进行调制：

```text
y = x + gamma · (x · N_x) + beta
```

在流场预测任务中，GRN 有助于突出强响应的流动结构，并抑制冗余或弱相关通道特征。

### 3.3 可学习残差下采样

V0 模型中下采样为：

```text
MaxPool2d → DoubleConv
```

改进模型中下采样改为可学习残差下采样：

```text
Conv 3×3, stride=2
+ Conv 1×1, stride=2
→ MedNeXtBlock
→ MedNeXtBlock
```

这种结构相比固定池化有两个优势：一是下采样过程具有可学习参数，能够根据流场数据自适应保留关键结构；二是主分支和捷径分支相加形成残差式下采样，有利于特征稳定传递。

### 3.4 解码器跳跃融合

改进模型仍然保留 U-Net 的 skip connection，但融合方式更明确：

```text
decoder feature
→ bilinear upsample to skip feature size
→ 1×1 reduce channels
→ concatenate with encoder skip feature
→ 1×1 fuse channels
→ MedNeXtBlock
→ MedNeXtBlock
```

这种设计在保留浅层空间细节的同时，利用 `1×1` 卷积对拼接后的特征进行通道压缩和融合，避免直接拼接后通道冗余过大。

### 3.5 Bottleneck 深层特征精炼

在最深层 `3×3` 尺度处，模型额外加入了 3 个 `MedNeXtBlock2D` 作为 bottleneck refinement：

```text
MedNeXtBlock2D(512, 512, kernel_size=3) ×3
```

该部分用于增强全局低分辨率特征表达，使网络在解码前获得更充分的抽象流动模式表征。

### 3.6 HeightLiftingHead 可学习高度提升头

这是当前改进模型中非常适合写进论文的关键改动。V0 模型使用固定双线性插值将输出从 `48×48` 调整为 `64×48`，而当前模型使用 `HeightLiftingHead` 学习高度方向的映射：

```text
[B, 64, 48, 48]
→ Conv 3×3 + GroupNorm + GELU
→ Conv 3×3 + GroupNorm + GELU
→ shared MLP along height dimension: 48 → 96 → 64
→ Conv 3×3 + GroupNorm + GELU
→ Conv 1×1
→ [B, 3, 64, 48]
```

其核心是对高度维度应用共享 MLP：

```text
height_proj: Linear(48, 96) → GELU → Linear(96, 64)
```

这意味着模型不再依赖固定插值，而是能够学习从二维输入截面高度到三维目标截面高度的非线性映射关系。对于 `48 → 64` 这类输入输出空间分辨率不一致的流场预测任务，该模块比固定上采样更有表达能力。

## 4. V0 与改进模型对比

| 对比项 | U-Net V0 | 改进 U-Net, MedNeXt-UNet-Lift |
|---|---|---|
| 基础模块 | `3×3 Conv + BN + ReLU ×2` | MedNeXtBlock2D |
| 卷积形式 | 普通卷积 | depthwise large-kernel conv + pointwise conv |
| 归一化 | BatchNorm | GroupNorm + GRN |
| 激活函数 | ReLU | GELU |
| 下采样 | MaxPool2d | 可学习残差下采样 |
| bottleneck | 单个 U-Net bottleneck | 额外 3 个 MedNeXtBlock 深层精炼 |
| skip fusion | 直接拼接后 DoubleConv | 上采样、降维、拼接、1×1 融合、MedNeXtBlock |
| 输出尺寸调整 | 固定 bilinear resize | learned HeightLiftingHead |
| 适合论文强调点 | 基线 U-Net | 大感受野、残差学习、全局响应归一化、可学习高度提升 |

## 5. 损失函数

训练中使用自定义 `FlowFieldLoss`。该损失函数由基础重构误差、速度分量加权误差、速度模长误差和涡量约束组成。

基础重构损失：

```text
L_base = MSE(Y_hat, Y)
```

速度分量加权损失：

```text
W = diag(w_u, w_v, w_w)
L_comp = MSE(W · Y_hat, W · Y)
```

当前默认权重为：

```text
w_u = 1.1, w_v = 1.0, w_w = 1.0
```

基础误差与分量加权误差融合：

```text
L_data = 0.7 · L_base + 0.3 · L_comp
```

速度模长损失：

```text
|V_hat| = sqrt(u_hat^2 + v_hat^2 + w_hat^2 + ε)
|V| = sqrt(u^2 + v^2 + w^2 + ε)
L_mag = MSE(|V_hat|, |V|)
```

加入速度模长约束后：

```text
L_data = 0.7 · L_data + 0.3 · L_mag
```

涡量约束项：

```text
ω = ∂v/∂x - ∂u/∂y
L_vort = mean((∂ω/∂x)^2 + (∂ω/∂y)^2)
```

最终损失：

```text
L_total = L_data + λ_vort · L_vort
```

当前默认：

```text
λ_vort = 0.1
```

该复合损失兼顾逐点数值误差、速度矢量整体强度和局部旋涡结构平滑性，适合用于连续流场回归任务。

## 6. 可写入论文的改进模型要点

### 6.1 方法描述

本文在传统 U-Net 编码器-解码器结构基础上，提出一种面向流场预测任务的 MedNeXt-UNet-Lift 网络。该网络保留 U-Net 的多尺度特征提取与跳跃连接机制，同时引入 MedNeXt 风格卷积块、全局响应归一化、可学习残差下采样和高度方向可学习提升头，以增强模型对复杂流动结构的表达能力。

### 6.2 结构改进

相比传统 U-Net 中的普通双卷积模块，本文采用由深度可分离大核卷积、GroupNorm、点卷积扩展、GELU 激活、GRN 和残差连接组成的 MedNeXtBlock。该模块能够在较低计算开销下扩大感受野，并通过残差路径提高深层特征学习稳定性。

### 6.3 输出映射改进

传统 U-Net V0 使用固定双线性插值将输出高度从 `48` 调整至 `64`。本文改进模型使用 HeightLiftingHead，通过共享 MLP 显式学习高度方向上的 `48 → 64` 映射。该设计使模型能够学习输入二维截面到目标三维截面之间更复杂的空间对应关系，而不是依赖固定插值规则。

### 6.4 物理一致性

在训练目标上，本文使用复合流场损失函数。除基本 MSE 外，损失函数进一步引入速度分量加权、速度模长约束和涡量正则项。该设计不仅关注速度分量的逐点误差，也约束速度矢量强度和局部旋涡结构，从而提升预测结果的物理合理性。

### 6.5 推荐论文贡献表述

- 提出 MedNeXt-UNet-Lift 网络，用于二维截面速度场到三维截面速度场的端到端预测。
- 将传统 U-Net 的普通卷积块替换为 MedNeXt 风格残差块，提高模型感受野和连续流场特征表达能力。
- 使用 GroupNorm 与 GRN 替代单一 BatchNorm，增强小批量训练稳定性和通道响应校准能力。
- 设计可学习残差下采样模块，替代固定 MaxPool 下采样，以自适应保留关键流动结构。
- 提出 HeightLiftingHead，通过高度方向共享 MLP 学习 `48 → 64` 的空间映射，改善固定插值带来的表达能力限制。
- 构建融合 MSE、分量加权、速度模长和涡量约束的复合损失函数，提高预测流场的数值精度和物理一致性。

## 7. 改进模型结构图 Prompt

下面 prompt 用于绘制当前 `models/unet_model.py` 中的改进 U-Net，即 MedNeXt-UNet-Lift，而不是旧版 V0 U-Net。

```text
请绘制一张适合学术论文方法章节使用的深度学习模型结构图，主题为“MedNeXt-UNet-Lift for 2D-to-3D Flow Field Prediction”。整体采用白色背景、蓝色编码器、橙色解码器、绿色输出头，风格简洁、专业、层级清晰。

图像从左到右展示网络。最左侧输入框标注为“Input 2D velocity field, 2 channels, 48×48, u and v”。输入进入 Stem 模块，Stem 标注为“Conv 3×3 + GroupNorm + GELU + MedNeXtBlock ×2”，输出为“64 channels, 48×48”。

编码器向下排列 4 个下采样阶段，每个阶段用蓝色模块表示，并标注为“learned residual downsampling + MedNeXtBlock ×2”。四个阶段分别为：Down1: 64→128, 48×48→24×24；Down2: 128→256, 24×24→12×12；Down3: 256→512, 12×12→6×6；Down4: 512→512, 6×6→3×3。每个下采样模块旁边用小注释写明“stride-2 Conv 3×3 + stride-2 Conv 1×1 residual branch”。

在网络最底部绘制 Bottleneck 模块，标注为“Bottleneck refinement: MedNeXtBlock ×3, 512 channels, 3×3”。在旁边添加一个小图例解释 MedNeXtBlock 的内部结构：“Depthwise large-kernel Conv → GroupNorm → 1×1 expansion → GELU → GRN → 1×1 projection → Residual add”。

右侧绘制解码器，使用橙色模块。解码器包含四个 Up 阶段：Up1: 512 + skip512 → 256, 3×3→6×6；Up2: 256 + skip256 → 128, 6×6→12×12；Up3: 128 + skip128 → 64, 12×12→24×24；Up4: 64 + skip64 → 64, 24×24→48×48。每个 Up 模块标注为“bilinear upsample + 1×1 reduce + skip concatenation + 1×1 fusion + MedNeXtBlock ×2”。

在编码器和解码器相同分辨率层之间画出横向 skip connection 箭头，分别连接 48×48、24×24、12×12 和 6×6 特征图。箭头标注为“skip connection: spatial detail fusion”。

在解码器末端绘制绿色的 HeightLiftingHead 模块，标注为“Learned HeightLiftingHead”。该模块内部显示三个步骤：1. “Conv 3×3 + GN + GELU ×2”；2. “shared height MLP: Linear 48→96 → GELU → Linear 96→64”；3. “Conv 3×3 + GN + GELU + Conv 1×1”。输出框标注为“Predicted 3D velocity slice, 3 channels, 64×48, u/v/w”。

图底部添加一个损失函数框，标题为“Composite flow-field loss”，内容为“MSE reconstruction + component weighted loss + velocity magnitude loss + vorticity regularization”，公式写为“L_total = L_data + 0.1 L_vort”。整体布局要突出：MedNeXtBlock、learned residual downsampling、skip fusion、learned height lifting head 四个创新点。
```

