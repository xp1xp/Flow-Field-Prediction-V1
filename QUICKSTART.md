# 快速开始指南

## 系统要求

- Python 3.7+
- CUDA (可选，用于GPU加速)
- 8GB+ RAM

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python test_basic.py
```

## 快速开始

### 方法1: 使用批处理脚本 (Windows用户)

#### 训练模型
双击运行 `train_models.bat`，选择要训练的模型。

#### 预测结果
双击运行 `predict_models.bat`，选择要执行的操作。

### 方法2: 使用命令行

#### 训练U-Net模型 (推荐)
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

#### 评估模型
```bash
python predict.py --model unet --evaluate --visualize
```

#### 预测单个数据
```bash
python predict.py --model unet --input data/test_2d.npy --output prediction.npy --visualize
```

## 查看训练进度

使用TensorBoard查看训练过程：

```bash
tensorboard --logdir results/unet/logs
```

然后在浏览器中打开 `http://localhost:6006`

## 项目结构说明

```
Flow-Field-Prediction/
├── data/                    # 数据目录
│   ├── cxp_2d_uv.npy      # 2D输入数据 (2, 90, 48, 48)
│   └── cxp_3d_uvw.npy     # 3D目标数据 (3, 90, 64, 48)
├── models/                  # 模型定义
│   ├── unet_model.py       # U-Net模型
│   ├── gan_model.py        # GAN模型
│   └── transformer_model.py # Transformer模型
├── results/                 # 训练结果 (自动创建)
│   ├── unet/              # U-Net训练结果
│   │   ├── checkpoints/   # 模型权重
│   │   ├── logs/          # TensorBoard日志
│   │   ├── visualizations/ # 可视化结果
│   │   ├── config.json    # 训练配置
│   │   └── normalizer.npz # 数据归一化参数
│   ├── gan/               # GAN训练结果
│   └── transformer/       # Transformer训练结果
├── train.py                # 训练脚本
├── predict.py              # 预测脚本
├── data_loader.py          # 数据加载模块
├── test_basic.py           # 基础测试脚本
├── train_models.bat        # 训练批处理脚本
├── predict_models.bat      # 预测批处理脚本
└── requirements.txt        # 依赖包
```

## 常见问题

### Q1: 训练时显存不足怎么办？

减小批量大小：
```bash
python train.py --model unet --batch_size 4
```

### Q2: 如何恢复中断的训练？

```bash
python train.py --model unet --resume
```

### Q3: 如何使用已训练的模型？

```bash
python predict.py --model unet --evaluate --visualize
```

### Q4: 训练需要多长时间？

- U-Net: 约10-30分钟 (取决于硬件)
- GAN: 约30-60分钟
- Transformer: 约20-40分钟

### Q5: 哪个模型效果最好？

推荐使用U-Net模型，它在速度场预测任务上表现最好且训练稳定。

## 训练参数调优

### 批量大小
- GPU显存充足: 8-16
- GPU显存有限: 2-4

### 学习率
- U-Net: 1e-4
- GAN: 2e-4
- Transformer: 1e-4

### 训练轮数
- 快速测试: 10-20轮
- 正式训练: 100-200轮

## 结果解读

### 训练日志
- `train/loss`: 训练损失
- `val/loss`: 验证损失
- 损失越低越好

### 预测结果
- 可视化结果保存在 `results/{model}/visualizations/`
- 包含输入2D、预测3D、真实3D和误差图

### 性能指标
- MSE (均方误差): 越低越好
- MAE (平均绝对误差): 越低越好

## 进阶使用

### 自定义数据

准备符合格式的数据：
- 2D数据: `(2, N, H, W)` - N个样本，H×W大小
- 3D数据: `(3, N, H', W')` - N个样本，H'×W'大小

### 修改模型架构

编辑 `models/` 目录下的模型文件，修改网络结构。

### 添加新的损失函数

在 `train.py` 的 `Trainer` 类中修改损失函数。

## 获取帮助

- 查看README.md获取详细信息
- 运行 `python example_usage.py` 查看使用示例
- 提交Issue报告问题

## 许可证

MIT License