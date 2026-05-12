#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
流场预测模型使用示例
演示如何使用训练好的模型进行预测
"""

import numpy as np
import matplotlib.pyplot as plt

def example_1_load_and_predict():
    """
    示例1: 加载2D数据并预测3D流场
    """
    print("=" * 60)
    print("示例1: 加载2D数据并预测3D流场")
    print("=" * 60)
    
    # 步骤1: 加载2D输入数据
    print("\n步骤1: 加载2D输入数据")
    input_2d = np.load('data/cxp_2d_uv.npy')
    print(f"2D数据形状: {input_2d.shape}")
    print(f"  - 通道数: {input_2d.shape[0]} (u, v)")
    print(f"  - 工况数: {input_2d.shape[1]}")
    print(f"  - 截面大小: {input_2d.shape[2]}×{input_2d.shape[3]}")
    
    # 步骤2: 选择一个工况进行预测
    print("\n步骤2: 选择一个工况进行预测")
    case_idx = 0  # 选择第一个工况
    single_input = input_2d[:, case_idx, :, :]
    print(f"选择的工况: {case_idx}")
    print(f"输入形状: {single_input.shape}")
    
    # 步骤3: 使用模型进行预测
    print("\n步骤3: 使用模型进行预测")
    print("命令: python predict.py --model unet --input data/cxp_2d_uv.npy --output prediction.npy --visualize")
    print("\n或者在Python代码中:")
    print("""
from predict import ModelPredictor

# 加载模型
predictor = ModelPredictor('unet', 'results/unet/checkpoints/best_model.pth')

# 进行预测
prediction = predictor.predict(single_input)

print(f"预测结果形状: {prediction.shape}")
print(f"  - 通道数: {prediction.shape[0]} (u, v, w)")
print(f"  - 截面大小: {prediction.shape[1]}×{prediction.shape[2]}")
    """)
    
    return True

def example_2_evaluate_on_test_set():
    """
    示例2: 在测试集上评估模型性能
    """
    print("\n" + "=" * 60)
    print("示例2: 在测试集上评估模型性能")
    print("=" * 60)
    
    print("\n步骤1: 训练模型")
    print("命令: python train.py --model unet --epochs 100")
    
    print("\n步骤2: 在测试集上评估")
    print("命令: python predict.py --model unet --evaluate --visualize")
    
    print("\n或者在Python代码中:")
    print("""
from predict import ModelPredictor

# 加载模型
predictor = ModelPredictor('unet', 'results/unet/checkpoints/best_model.pth')

# 在测试集上评估
predictions, targets, metrics = predictor.evaluate_on_test_set()

print(f"测试集评估结果:")
print(f"  - MSE: {metrics['mse']:.6f}")
print(f"  - MAE: {metrics['mae']:.6f}")
    """)
    
    return True

def example_3_visualize_results():
    """
    示例3: 可视化预测结果
    """
    print("\n" + "=" * 60)
    print("示例3: 可视化预测结果")
    print("=" * 60)
    
    print("\n使用命令行可视化:")
    print("命令: python predict.py --model unet --evaluate --visualize")
    
    print("\n或者在Python代码中:")
    print("""
from predict import ModelPredictor
import numpy as np

# 加载模型
predictor = ModelPredictor('unet', 'results/unet/checkpoints/best_model.pth')

# 加载测试数据
input_2d = np.load('data/cxp_2d_uv.npy')
input_3d = np.load('data/cxp_3d_uvw.npy')

# 选择一个测试样本
test_idx = 0
input_data = input_2d[:, test_idx, :, :]
target_data = input_3d[:, test_idx, :, :]

# 进行预测
prediction = predictor.predict(input_data)

# 可视化结果
predictor.visualize_prediction(
    input_data, 
    prediction, 
    target_data, 
    save_path='results/unet/visualizations/example.png'
)
    """)
    
    return True

def example_4_compare_models():
    """
    示例4: 比较不同模型的性能
    """
    print("\n" + "=" * 60)
    print("示例4: 比较不同模型的性能")
    print("=" * 60)
    
    print("\n步骤1: 训练所有模型")
    print("命令:")
    print("  python train.py --model unet --epochs 50")
    print("  python train.py --model gan --epochs 50")
    print("  python train.py --model transformer --epochs 50")
    
    print("\n步骤2: 评估所有模型")
    print("命令:")
    print("  python predict.py --model unet --evaluate")
    print("  python predict.py --model gan --evaluate")
    print("  python predict.py --model transformer --evaluate")
    
    print("\n步骤3: 比较性能指标")
    print("""
from predict import ModelPredictor

models = ['unet', 'gan', 'transformer']
results = {}

for model_name in models:
    predictor = ModelPredictor(model_name, f'results/{model_name}/checkpoints/best_model.pth')
    predictions, targets, metrics = predictor.evaluate_on_test_set()
    results[model_name] = metrics

# 打印比较结果
print("模型性能比较:")
print(f"{'模型':<15} {'MSE':<12} {'MAE':<12}")
print("-" * 40)
for model_name, metrics in results.items():
    print(f"{model_name:<15} {metrics['mse']:<12.6f} {metrics['mae']:<12.6f}")
    """)
    
    return True

def example_5_custom_prediction():
    """
    示例5: 使用自定义数据进行预测
    """
    print("\n" + "=" * 60)
    print("示例5: 使用自定义数据进行预测")
    print("=" * 60)
    
    print("\n准备自定义数据:")
    print("""
# 自定义数据应该符合以下格式:
# - 形状: (2, H, W) 或 (N, 2, H, W)
# - 数据类型: float32 或 float64
# - 归一化: 建议进行归一化处理

import numpy as np

# 创建示例数据 (2个通道, 48×48)
custom_data = np.random.randn(2, 48, 48).astype(np.float32)

# 保存数据
np.save('custom_input.npy', custom_data)
    """)
    
    print("\n使用自定义数据进行预测:")
    print("命令: python predict.py --model unet --input custom_input.npy --output custom_output.npy --visualize")
    
    print("\n或者在Python代码中:")
    print("""
from predict import ModelPredictor
import numpy as np

# 加载模型
predictor = ModelPredictor('unet', 'results/unet/checkpoints/best_model.pth')

# 加载自定义数据
custom_data = np.load('custom_input.npy')

# 进行预测
prediction = predictor.predict(custom_data)

# 保存预测结果
np.save('custom_output.npy', prediction)

print(f"预测结果已保存，形状: {prediction.shape}")
    """)
    
    return True

def main():
    """
    主函数：运行所有示例
    """
    print("\n" + "=" * 60)
    print("流场预测模型使用示例")
    print("=" * 60)
    
    examples = [
        ("加载2D数据并预测3D流场", example_1_load_and_predict),
        ("在测试集上评估模型性能", example_2_evaluate_on_test_set),
        ("可视化预测结果", example_3_visualize_results),
        ("比较不同模型的性能", example_4_compare_models),
        ("使用自定义数据进行预测", example_5_custom_prediction),
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\n" + "=" * 60)
    print("提示: 这些示例展示了如何使用流场预测系统")
    print("请根据需要选择相应的示例进行参考")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    main()