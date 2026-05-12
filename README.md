# 流场预测模型

基于深度学习的2D到3D流场预测系统，支持U-Net、GAN和Transformer三种模型架构。

## 项目简介

本项目实现了从2D截面速度场数据预测3D截面速度场的深度学习模型。系统支持三种不同的模型架构：

- **U-Net**: 编码器-解码器结构，适合图像到图像的转换任务
- **GAN**: 生成对抗网络，可以生成更真实的速度场
- **Transformer**: 基于自注意力机制，能够捕获长距离依赖关系

## 数据说明

- **输入数据**: 2D xy截面速度场 `(2, 90, 48, 48)`
  - 2个通道：u和v速度分量
  - 90个工况
  - 每个工况的截面大小为 48x48

- **输出数据**: 3D zy截面速度场 `(3, 90, 64, 48)`
  - 3个通道：u、v和w速度分量
  - 90个工况
  - 每个工况的截面大小为 64x48

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

#### 训练U-Net模型
```bash
python train.py --model unet --batch_size 8 --epochs 100 --lr 1e-4
```

#### 训练GAN模型
```bash
python train.py --model gan --batch_size 4 --epochs 200 --lr 2e-4
```

#### 训练Transformer模型
```bash
python train.py --model transformer --batch_size 4 --epochs 100 --lr 1e-4
```

### 2. 预测结果

#### 在测试集上评估模型
```bash
python predict.py --model unet --evaluate --visualize
```

#### 对单个2D数据进行预测
```bash
python predict.py --model unet --input data/test_2d.npy --output prediction.npy --visualize
```

### 3. 查看训练结果

使用TensorBoard查看训练过程：
```bash
tensorboard --logdir results/{model_name}/logs
```

## 项目结构

```
Flow-Field-Prediction/
├── data/                          # 数据目录
│   ├── cxp_2d_uv.npy            # 2D输入数据
│   └── cxp_3d_uvw.npy           # 3D目标数据
├── models/                        # 模型定义
│   ├── __init__.py
│   ├── unet_model.py             # U-Net模型
│   ├── gan_model.py              # GAN模型
│   └── transformer_model.py     # Transformer模型
├── results/                       # 训练结果
│   ├── unet/                    # U-Net训练结果
│   │   ├── checkpoints/         # 模型权重文件
│   │   ├── logs/                # TensorBoard日志
│   │   ├── visualizations/      # 预测结果可视化
│   │   ├── config.json          # 训练配置
│   │   └── normalizer.npz       # 数据归一化参数
│   ├── gan/                     # GAN训练结果
│   └── transformer/             # Transformer训练结果
├── train.py                       # 训练脚本
├── predict.py                     # 预测脚本
├── data_loader.py                 # 数据加载模块
├── example_usage.py              # 使用示例
└── requirements.txt               # 依赖包
```

## 训练参数说明

### train.py 参数

- `--model`: 模型类型 (unet/gan/transformer)
- `--batch_size`: 批量大小 (默认: 8)
- `--epochs`: 训练轮数 (默认: 100)
- `--lr`: 学习率 (默认: 1e-4)
- `--num_workers`: 数据加载线程数 (默认: 0)
- `--save_interval`: 保存检查点间隔 (默认: 10)
- `--resume`: 从检查点恢复训练

### predict.py 参数

- `--model`: 模型类型 (unet/gan/transformer)
- `--checkpoint`: 模型检查点路径
- `--input`: 输入2D数据路径
- `--output`: 预测结果保存路径
- `--visualize`: 可视化预测结果
- `--evaluate`: 在测试集上评估
- `--save_dir`: 结果保存目录

## 模型性能对比

| 模型 | 训练时间 | 推理速度 | 预测精度 | 推荐场景 |
|------|---------|---------|---------|---------|
| U-Net | 快 | 快 | 高 | 通用场景，推荐首选 |
| GAN | 慢 | 中 | 中高 | 需要生成更真实的结果 |
| Transformer | 中 | 中 | 高 | 需要捕获长距离依赖 |

## 常见问题

### 1. 训练过程中显存不足

减小批量大小：
```bash
python train.py --model unet --batch_size 4
```

### 2. 如何使用已训练的模型进行预测

```bash
python predict.py --model unet --evaluate --visualize
```

### 3. 如何恢复中断的训练

```bash
python train.py --model unet --resume
```

### 4. 如何查看训练曲线

```bash
tensorboard --logdir results/unet/logs
```

## 技术细节

### 数据预处理

- 数据归一化：使用z-score标准化
- 数据划分：训练集70%，验证集15%，测试集15%
- 数据增强：支持随机旋转、翻转等操作

### 模型架构

#### U-Net
- 编码器-解码器结构
- 跳跃连接保留空间细节
- 适合处理连续的物理量

#### GAN
- 生成器：U-Net架构
- 判别器：PatchGAN结构
- 损失函数：对抗损失 + 像素损失

#### Transformer
- Patch Embedding将图像转换为序列
- 多头自注意力机制
- 位置编码增强空间信息

### 训练策略

- 优化器：Adam
- 学习率调度：余弦退火
- 早停策略：基于验证集性能
- 检查点保存：保存最佳模型

## 引用

如果您使用了本项目的代码或模型，请引用：

```bibtex
@software{flow_field_prediction,
  title={Flow Field Prediction using Deep Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Flow-Field-Prediction}
}
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或Pull Request。

## 更新日志

### v1.0.0 (2024-03-07)
- 初始版本发布
- 支持U-Net、GAN、Transformer三种模型
- 完整的训练和预测流程
- 可视化功能