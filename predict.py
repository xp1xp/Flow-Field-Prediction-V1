import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from normalization import DataNormalizer
from models.unet_model import get_unet_model
from models.gan_model import get_gan_models
from models.transformer_model import get_transformer_model

# ==================== 配置区域 ====================
# 在这里修改参数，无需使用命令行

# 模型选择: 'unet', 'gan', 'transformer'
MODEL_TYPE = 'unet'
# MODEL_TYPE = 'transformer'

# 模型检查点路径 (留空则使用默认路径 results/{model}/checkpoints/best_model.pth)
CHECKPOINT_PATH = 'results/unet/checkpoints/best_model.pth'
# CHECKPOINT_PATH = 'results/transformer/checkpoints/best_model.pth'
# V1=base, V3=base_attn_comp_mag, V4=base_attn_comp, V5=base_attn_mag

# 测试集数据路径
# TEST_DATA_PATH = 'data/cxp_2d_uv.npy'
# TEST_TARGET_PATH = 'data/cxp_3d_uvw.npy'
TEST_DATA_PATH = 'data/cxp_2c_uv_test.npy'
TEST_TARGET_PATH = 'data/cxp_3c_uvw_test.npy'

# 输入数据路径 (用于单样本预测，留空则进行测试集评估)
# INPUT_DATA_PATH = ''
INPUT_DATA_PATH = 'data\data_real2c.npy'

# 输出路径 (用于保存预测结果，留空则不保存)
OUTPUT_PATH = ''

# 是否可视化结果
VISUALIZE = True

# 是否在测试集上评估
EVALUATE = False

# 结果保存目录
SAVE_DIR = 'results'

REAL_INPUT_COLOR_LIMITS = {  #调整实验数据输入下，输出的色标范围
    'u': (50, 190),
    'v': (-55, 45),
    'w': (-30, 30),
}

# ==================== 配置区域结束 ====================

class ModelPredictor:
    def __init__(self, model_type, checkpoint_path, device=None):
        self.model_type = model_type
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_model(checkpoint_path)
        self.load_normalizer()
        
        print(f"Model {model_type} loaded on {self.device}")
    
    def load_model(self, checkpoint_path):
        if self.model_type == 'unet':
            self.model = get_unet_model()
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        elif self.model_type == 'gan':
            self.generator, _ = get_gan_models()
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.generator.eval()
            self.model = self.generator
        elif self.model_type == 'transformer':
            self.model = get_transformer_model()
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
    
    def load_normalizer(self):
        normalizer_path = os.path.join('results', self.model_type, 'normalizer.npz')
        self.normalizer = DataNormalizer()
        self.normalizer.load(normalizer_path)
    
    def normalize_input(self, input_2d):
        if isinstance(input_2d, torch.Tensor):
            input_2d = input_2d.detach().cpu().numpy()
        return self.normalizer.transform_2d(input_2d)

    def predict(self, input_2d, input_is_normalized=False):
        self.model.eval()
        with torch.no_grad():
            if isinstance(input_2d, np.ndarray):
                model_input = input_2d
            else:
                model_input = input_2d.detach().cpu().numpy()

            if not input_is_normalized:
                model_input = self.normalize_input(model_input)

            input_tensor = torch.from_numpy(model_input.astype(np.float32))
            
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.unsqueeze(0)
            
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            
            output = output.cpu().numpy()
            
            if output.shape[0] == 1:
                output = output.squeeze(0)
            
            output = self.normalizer.inverse_transform_3d(output)
            
            return output
    
    def predict_batch(self, input_2d_batch, input_is_normalized=False):
        self.model.eval()
        with torch.no_grad():
            if isinstance(input_2d_batch, np.ndarray):
                model_input = input_2d_batch
            else:
                model_input = input_2d_batch.detach().cpu().numpy()

            if not input_is_normalized:
                model_input = self.normalize_input(model_input)

            input_tensor = torch.from_numpy(model_input.astype(np.float32))
            
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            
            output = output.cpu().numpy()
            
            output = self.normalizer.inverse_transform_3d(output)
            
            return output
    
    def evaluate_on_test_set(self, test_data_path='data/cxp_2d_uv.npy',     # 默认测试集位置
                             test_target_path='data/cxp_3d_uvw.npy'):
        data_2d = np.load(test_data_path)
        data_3d = np.load(test_target_path)
        print("原始数据形状:")
        print(data_2d.shape, data_3d.shape)
        
        total_samples = data_2d.shape[1]
        train_size = int(total_samples * 0) #0.8
        val_size = int(total_samples * 0) #0.1
        
        test_2d = data_2d[:, train_size+val_size:]
        test_3d = data_3d[:, train_size+val_size:]
        
        predictions = []
        targets = []
        inputs = []
        input_raws = []
        
        for i in tqdm(range(test_2d.shape[1]), desc="Predicting on test set"):
            input_2d_raw = test_2d[:, i, :, :]
            input_2d = self.normalize_input(input_2d_raw)
            target_3d = test_3d[:, i, :, :]
            
            prediction = self.predict(input_2d, input_is_normalized=True)
            
            predictions.append(prediction)
            targets.append(target_3d)
            inputs.append(input_2d)
            input_raws.append(input_2d_raw)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        inputs = np.array(inputs)
        input_raws = np.array(input_raws)
        
        print("测试集输入数据形状:")
        print(input_raws.shape,inputs.shape,targets.shape)
        print("测试集输出数据形状:")
        print(predictions.shape)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        mse_per_sample = np.mean((predictions - targets) ** 2, axis=(1, 2, 3))  # 形状 (n,)  n 表示样本个数
        mae_per_sample = np.mean(np.abs(predictions - targets), axis=(1, 2, 3))  # 形状 (n,)
        # print(mse_per_sample.shape, mae_per_sample.shape)
        mse_per_sample_row = mse_per_sample.reshape(1, -1) # 形状 (n,)-->(1, n)
        mae_per_sample_row = mae_per_sample.reshape(1, -1)
        # print("行向量形状:", mse_per_sample_row.shape)   # (1, n)

        print(f"\nTest Set Evaluation:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"单个样本MSE: {mse_per_sample_row}")
        print(f"单个样本MAE: {mae_per_sample_row}")
        
        return input_raws, inputs, predictions, targets, {'mse': mse, 'mae': mae}
    
    def visualize_prediction(self, input_2d=None, prediction_3d=None, target_3d=None,
                             save_path=None, color_limits=None):
        fig = plt.figure(figsize=(18, 12))
        
        if input_2d is not None and target_3d is not None:
            gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
            velocity_components = ['u', 'v', 'w']
            
            for i, comp in enumerate(['u', 'v']):
                ax = fig.add_subplot(gs[i, 0])
                im = ax.imshow(input_2d[i], cmap='jet', origin='lower')
                ax.set_title(f'Input 2D {comp.upper()} Velocity')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax)
            
            for i, comp in enumerate(velocity_components):
                vmin = min(prediction_3d[i].min(), target_3d[i].min())
                vmax = max(prediction_3d[i].max(), target_3d[i].max())
                
                ax = fig.add_subplot(gs[i, 1])
                im = ax.imshow(prediction_3d[i], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                ax.set_title(f'Predicted 3D {comp.upper()} Velocity')
                ax.set_xlabel('Z')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax)
            
                ax = fig.add_subplot(gs[i, 2])
                im = ax.imshow(target_3d[i], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                ax.set_title(f'Target 3D {comp.upper()} Velocity')
                ax.set_xlabel('Z')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax)
            
                ax = fig.add_subplot(gs[i, 3])
                diff = abs(prediction_3d[i] - target_3d[i])
                im = ax.imshow(diff, cmap='jet', origin='lower')
                ax.set_title(f'Error {comp.upper()} Velocity')
                ax.set_xlabel('Z')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax)
        elif input_2d is not None:
            gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
            velocity_components = ['u', 'v', 'w']
            
            for i, comp in enumerate(['u', 'v']):
                ax = fig.add_subplot(gs[i, 0])
                im = ax.imshow(input_2d[i], cmap='jet', origin='lower')
                ax.set_title(f'Input 2D {comp.upper()} Velocity')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax)
            
            for i, comp in enumerate(velocity_components):
                if color_limits and comp in color_limits:
                    vmin, vmax = color_limits[comp]
                elif target_3d is not None:
                    vmin = min(prediction_3d[i].min(), target_3d[i].min())
                    vmax = max(prediction_3d[i].max(), target_3d[i].max())
                else:
                    vmin, vmax = None, None
                
                ax = fig.add_subplot(gs[i, 1])
                im = ax.imshow(prediction_3d[i], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                ax.set_title(f'Predicted 3D {comp.upper()} Velocity')
                ax.set_xlabel('Z')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax)
            
                if target_3d is not None:
                    ax = fig.add_subplot(gs[i, 2])
                    im = ax.imshow(target_3d[i], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                    ax.set_title(f'Target 3D {comp.upper()} Velocity')
                    ax.set_xlabel('Z')
                    ax.set_ylabel('Y')
                    plt.colorbar(im, ax=ax)
        else:
            gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
            velocity_components = ['u', 'v', 'w']
            
            for i, comp in enumerate(velocity_components):
                vmin = min(prediction_3d[i].min(), target_3d[i].min())
                vmax = max(prediction_3d[i].max(), target_3d[i].max())
                
                ax = fig.add_subplot(gs[i, 0])
                im = ax.imshow(prediction_3d[i], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                ax.set_title(f'Predicted 3D {comp.upper()} Velocity')
                ax.set_xlabel('Z')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax)
            
                ax = fig.add_subplot(gs[i, 1])
                im = ax.imshow(target_3d[i], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                ax.set_title(f'Target 3D {comp.upper()} Velocity')
                ax.set_xlabel('Z')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    # 优先使用配置区域的参数，如果命令行提供了参数则覆盖配置
    parser = argparse.ArgumentParser(description='Predict 3D flow field from 2D input')
    parser.add_argument('--model', type=str, default=MODEL_TYPE, choices=['unet', 'gan', 'transformer'],
                       help='Model type to use')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                       help='Path to model checkpoint (default: results/{model}/checkpoints/best_model.pth)')
    parser.add_argument('--input', type=str, default=INPUT_DATA_PATH,
                       help='Path to input 2D data (.npy file)')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH,
                       help='Path to save prediction output (.npy file)')
    parser.add_argument('--visualize', action='store_true', default=VISUALIZE,
                       help='Visualize prediction results')
    parser.add_argument('--evaluate', action='store_true', default=EVALUATE,
                       help='Evaluate on test set')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR,
                       help='Directory to save results')
    parser.add_argument('--test_data', type=str, default=TEST_DATA_PATH,
                       help='Path to test data (.npy file)')
    parser.add_argument('--test_target', type=str, default=TEST_TARGET_PATH,
                       help='Path to test target data (.npy file)')
    
    args = parser.parse_args()
    
    # 如果checkpoint路径为空，使用默认路径
    if not args.checkpoint:
        args.checkpoint = os.path.join('results', args.model, 'checkpoints', 'best_model.pth')
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    print(f"使用模型: {args.model}")
    print(f"检查点路径: {args.checkpoint}")
    
    predictor = ModelPredictor(args.model, args.checkpoint)
    
    if args.evaluate or (not args.input and not args.output):
        print("在测试集上评估...")
        input_raws, inputs, predictions, targets, metrics = predictor.evaluate_on_test_set(args.test_data, args.test_target)
        # input_raws为原始输入数据，inputs为归一化后的输入数据，predictions为预测结果，targets为目标对照数据，metrics为评估指标（MSE和MAE）

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        # 保存评估结果（目前用于测试集数据）
        output_dir = os.path.join(args.save_dir, args.model, 'evaluation_results')
        os.makedirs(output_dir, exist_ok=True)
        # 保存原始输入数据
        np.save(os.path.join(output_dir, 'input_raws.npy'), input_raws)
        print(f"原始输入数据已保存到 {os.path.join(output_dir, 'input_raws.npy')}")
        # 保存归一化后的输入数据
        np.save(os.path.join(output_dir, 'inputs_normalized.npy'), inputs)
        print(f"归一化输入数据已保存到 {os.path.join(output_dir, 'inputs_normalized.npy')}")
        # 保存预测结果
        np.save(os.path.join(output_dir, 'predictions.npy'), predictions)
        print(f"预测结果已保存到 {os.path.join(output_dir, 'predictions.npy')}")
        # 保存目标数据
        np.save(os.path.join(output_dir, 'targets.npy'), targets)
        print(f"目标数据已保存到 {os.path.join(output_dir, 'targets.npy')}")
        

        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            np.save(args.output, predictions)
            print(f"预测结果已保存到 {args.output}")
        
        if args.visualize:
            output_dir = os.path.join(args.save_dir, args.model, 'visualizations')
            os.makedirs(output_dir, exist_ok=True)
            
            for i in range(min(5, len(predictions))):
                save_path = os.path.join(output_dir, f'prediction_{i}.png')
                predictor.visualize_prediction(input_raws[i], predictions[i], 
                                               targets[i], 
                                               save_path=save_path)
    
    elif args.input:  # 用于单样本数据，INPUT_DATA_PATH
        print(f"从 {args.input} 加载输入数据...")
        input_data = np.load(args.input)
        
        print("进行预测...")
        prediction = predictor.predict(input_data)
        np.save("./results/unet/real_prediction.npy", prediction)#预测结果已保存
        print(f"预测结果已保存到/results/unet/real_prediction.npy")
        
        
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            np.save(args.output, prediction)
            print(f"预测结果已保存到 {args.output}")
        
        if args.visualize:
            output_dir = os.path.join(args.save_dir, args.model, 'visualizations')
            os.makedirs(output_dir, exist_ok=True)
            
            save_path = os.path.join(output_dir, 'prediction.png')
            input_is_real_data = os.path.normcase(os.path.normpath(args.input)) == os.path.normcase(
                os.path.normpath('data/data_real2c.npy')
            )
            predictor.visualize_prediction(
                input_data,
                prediction,
                save_path=save_path,
                color_limits=REAL_INPUT_COLOR_LIMITS if input_is_real_data else None,
            )
    
    else:
        print("请提供 --input 进行单样本预测，或者使用 --evaluate 进行测试集评估")
        print("示例:")
        print("  单样本预测: python predict.py --model unet --input data/test_2d.npy --output prediction.npy --visualize")
        print("  测试集评估: python predict.py --model unet --evaluate --visualize")

if __name__ == '__main__':
    main()
