# predict.py 配置说明

## 快速开始

现在你可以直接在代码中修改参数，无需使用命令行！

## 配置方法

打开 `predict.py` 文件，在文件顶部的 **配置区域** 修改参数：

```python
# ==================== 配置区域 ====================
# 在这里修改参数，无需使用命令行

# 模型选择: 'unet', 'gan', 'transformer'
MODEL_TYPE = 'unet'

# 模型检查点路径 (留空则使用默认路径 results/{model}/checkpoints/best_model.pth)
CHECKPOINT_PATH = ''

# 测试集数据路径
TEST_DATA_PATH = 'data/cxp_2d_uv.npy'
TEST_TARGET_PATH = 'data/cxp_3d_uvw.npy'

# 输入数据路径 (用于单样本预测，留空则进行测试集评估)
INPUT_DATA_PATH = ''

# 输出路径 (用于保存预测结果，留空则不保存)
OUTPUT_PATH = ''

# 是否可视化结果
VISUALIZE = True

# 是否在测试集上评估
EVALUATE = True

# 结果保存目录
SAVE_DIR = 'results'

# ==================== 配置区域结束 ====================
```

## 使用场景

### 场景1: 在测试集上评估模型

```python
MODEL_TYPE = 'unet'           # 选择模型
EVALUATE = True              # 启用测试集评估
VISUALIZE = True             # 生成可视化结果
INPUT_DATA_PATH = ''          # 留空
OUTPUT_PATH = ''             # 留空
```

**运行**: `python predict.py`

### 场景2: 预测单个数据

```python
MODEL_TYPE = 'unet'           # 选择模型
EVALUATE = False             # 禁用测试集评估
INPUT_DATA_PATH = 'data/cxp_2d_uv.npy'  # 输入数据路径
OUTPUT_PATH = 'results/prediction.npy'     # 输出路径
VISUALIZE = True             # 生成可视化结果
```

**运行**: `python predict.py`

### 场景3: 使用自定义测试集

```python
MODEL_TYPE = 'unet'           # 选择模型
TEST_DATA_PATH = 'data/my_test_2d.npy'      # 自定义测试数据
TEST_TARGET_PATH = 'data/my_test_3d.npy'    # 自定义测试目标
EVALUATE = True              # 启用测试集评估
VISUALIZE = True             # 生成可视化结果
```

**运行**: `python predict.py`

### 场景4: 使用特定的模型检查点

```python
MODEL_TYPE = 'unet'
CHECKPOINT_PATH = 'results/unet/checkpoints/checkpoint_epoch_50.pth'  # 指定检查点
EVALUATE = True
VISUALIZE = True
```

**运行**: `python predict.py`

## 参数说明

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `MODEL_TYPE` | str | 模型类型，可选: 'unet', 'gan', 'transformer' | `'unet'` |
| `CHECKPOINT_PATH` | str | 模型检查点路径，留空则使用默认路径 | `''` 或 `'results/unet/checkpoints/best_model.pth'` |
| `TEST_DATA_PATH` | str | 测试集2D数据路径 | `'data/cxp_2d_uv.npy'` |
| `TEST_TARGET_PATH` | str | 测试集3D目标数据路径 | `'data/cxp_3d_uvw.npy'` |
| `INPUT_DATA_PATH` | str | 单样本预测的输入数据路径 | `'data/test_2d.npy'` |
| `OUTPUT_PATH` | str | 预测结果保存路径 | `'results/prediction.npy'` |
| `VISUALIZE` | bool | 是否生成可视化结果 | `True` 或 `False` |
| `EVALUATE` | bool | 是否在测试集上评估 | `True` 或 `False` |
| `SAVE_DIR` | str | 结果保存目录 | `'results'` |

## 命令行参数（可选）

虽然代码中已经配置了参数，但你仍然可以使用命令行参数覆盖配置：

```bash
# 覆盖模型类型
python predict.py --model gan

# 覆盖输入输出路径
python predict.py --input my_data.npy --output my_prediction.npy

# 禁用可视化
python predict.py --no-visualize

# 指定测试数据
python predict.py --test_data custom_test.npy --test_target custom_target.npy
```

## 输出说明

### 测试集评估输出

```
使用模型: unet
检查点路径: results/unet/checkpoints/best_model.pth
在测试集上评估...
Predicting on test set: 100%|████████| 15/15 [00:02<00:00,  6.01it/s]

Test Set Evaluation:
MSE: 0.123456
MAE: 0.234567

预测结果已保存到 results/unet/predictions.npy
```

### 可视化输出

- 测试集评估: 生成 5 个样本的可视化图
  - `results/{model}/visualizations/prediction_0.png`
  - `results/{model}/visualizations/prediction_1.png`
  - ...
  - `results/{model}/visualizations/prediction_4.png`

- 单样本预测: 生成 1 个可视化图
  - `results/{model}/visualizations/prediction.png`

## 常见问题

### Q1: 如何切换到其他模型？
修改 `MODEL_TYPE` 参数：
```python
MODEL_TYPE = 'gan'  # 或 'transformer'
```

### Q2: 如何使用自己的测试数据？
修改 `TEST_DATA_PATH` 和 `TEST_TARGET_PATH`：
```python
TEST_DATA_PATH = 'my_data/test_2d.npy'
TEST_TARGET_PATH = 'my_data/test_3d.npy'
```

### Q3: 如何只预测不评估？
```python
EVALUATE = False
INPUT_DATA_PATH = 'data/cxp_2d_uv.npy'
OUTPUT_PATH = 'results/my_prediction.npy'
```

### Q4: 如何不生成可视化？
```python
VISUALIZE = False
```

### Q5: 如何使用特定的训练轮次模型？
```python
CHECKPOINT_PATH = 'results/unet/checkpoints/checkpoint_epoch_50.pth'
```

## 完整示例

### 示例1: 评估U-Net模型
```python
MODEL_TYPE = 'unet'
CHECKPOINT_PATH = ''
TEST_DATA_PATH = 'data/cxp_2d_uv.npy'
TEST_TARGET_PATH = 'data/cxp_3d_uvw.npy'
INPUT_DATA_PATH = ''
OUTPUT_PATH = ''
VISUALIZE = True
EVALUATE = True
SAVE_DIR = 'results'
```

### 示例2: 评估GAN模型
```python
MODEL_TYPE = 'gan'
CHECKPOINT_PATH = ''
TEST_DATA_PATH = 'data/cxp_2d_uv.npy'
TEST_TARGET_PATH = 'data/cxp_3d_uvw.npy'
INPUT_DATA_PATH = ''
OUTPUT_PATH = ''
VISUALIZE = True
EVALUATE = True
SAVE_DIR = 'results'
```

### 示例3: 预测单个样本
```python
MODEL_TYPE = 'unet'
CHECKPOINT_PATH = ''
TEST_DATA_PATH = 'data/cxp_2d_uv.npy'
TEST_TARGET_PATH = 'data/cxp_3d_uvw.npy'
INPUT_DATA_PATH = 'data/cxp_2d_uv.npy'
OUTPUT_PATH = 'results/my_prediction.npy'
VISUALIZE = True
EVALUATE = False
SAVE_DIR = 'results'
```

## 提示

1. **路径分隔符**: Windows使用反斜杠 `\`，Linux/Mac使用正斜杠 `/`
2. **相对路径**: 可以使用相对路径（如 `data/...`）或绝对路径
3. **空值**: 留空字符串 `''` 表示使用默认值或禁用该功能
4. **布尔值**: 使用 `True` 或 `False`（首字母大写）
5. **字符串**: 使用单引号或双引号括起来

现在你可以直接修改代码中的配置区域，然后运行 `python predict.py` 即可！