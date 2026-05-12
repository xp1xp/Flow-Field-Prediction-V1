import os
import numpy as np

def example_usage():
    print("=" * 60)
    print("流场预测模型使用示例")
    print("=" * 60)
    
    print("\n1. 训练模型")
    print("-" * 60)
    print("训练U-Net模型:")
    print("python train.py --model unet --batch_size 8 --epochs 100 --lr 1e-4")
    
    print("\n训练GAN模型:")
    print("python train.py --model gan --batch_size 4 --epochs 200 --lr 2e-4")
    
    print("\n训练Transformer模型:")
    print("python train.py --model transformer --batch_size 4 --epochs 100 --lr 1e-4")
    
    print("\n2. 预测结果")
    print("-" * 60)
    print("在测试集上评估模型:")
    print("python predict.py --model unet --evaluate --visualize")
    
    print("\n对单个2D数据进行预测:")
    print("python predict.py --model unet --input data/test_2d.npy --output prediction.npy --visualize")
    
    print("\n3. 查看训练结果")
    print("-" * 60)
    print("训练结果保存在 results/{model_name}/ 目录下:")
    print("  - checkpoints/: 模型权重文件")
    print("  - logs/: TensorBoard日志文件")
    print("  - visualizations/: 预测结果可视化")
    print("  - config.json: 训练配置")
    print("  - normalizer.npz: 数据归一化参数")
    
    print("\n使用TensorBoard查看训练过程:")
    print("tensorboard --logdir results/{model_name}/logs")
    
    print("\n4. 项目结构")
    print("-" * 60)
    print("Flow-Field-Prediction/")
    print("├── data/")
    print("│   ├── cxp_2d_uv.npy      # 2D输入数据")
    print("│   └── cxp_3d_uvw.npy     # 3D目标数据")
    print("├── models/")
    print("│   ├── unet_model.py       # U-Net模型")
    print("│   ├── gan_model.py        # GAN模型")
    print("│   └── transformer_model.py # Transformer模型")
    print("├── results/")
    print("│   ├── unet/              # U-Net训练结果")
    print("│   ├── gan/               # GAN训练结果")
    print("│   └── transformer/       # Transformer训练结果")
    print("├── train.py               # 训练脚本")
    print("├── predict.py             # 预测脚本")
    print("├── data_loader.py         # 数据加载模块")
    print("└── requirements.txt       # 依赖包")

def quick_start():
    print("\n" + "=" * 60)
    print("快速开始指南")
    print("=" * 60)
    
    print("\n步骤1: 安装依赖")
    print("pip install -r requirements.txt")
    
    print("\n步骤2: 训练模型 (推荐从U-Net开始)")
    print("python train.py --model unet --epochs 50")
    
    print("\n步骤3: 评估模型")
    print("python predict.py --model unet --evaluate --visualize")
    
    print("\n步骤4: 查看结果")
    print("可视化结果保存在 results/unet/visualizations/ 目录下")

if __name__ == '__main__':
    example_usage()
    quick_start()
    
    print("\n" + "=" * 60)
    print("更多详细信息请参考README.md")
    print("=" * 60)