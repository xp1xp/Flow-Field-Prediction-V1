import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# 改进的流场损失函数
import torch.nn.functional as F

class FlowFieldLoss(nn.Module):
    """
    精简版物理信息损失函数（仅包含涡量约束）
    包含：
    - 基础数据损失（MSE/MAE，分量加权）
    - 速度大小损失
    - 涡量约束（增强旋涡结构捕捉）
    """
    def __init__(self,
                 weight_u=1.1,                     # u分量权重
                 weight_v=1.0,                     # v分量权重
                 weight_w=1.0,                     # w分量权重（若为3D）
                 use_mae=False,                    # 是否使用MAE（否则MSE）
                 use_velocity_magnitude=True,      # 是否启用速度大小损失
                 weight_vorticity=0.1,             # 涡量约束权重
                 dx=1.0, dy=1.0):                  # 网格间距（dz,dy）
        super().__init__()
        self.weight_u = weight_u
        self.weight_v = weight_v
        self.weight_w = weight_w
        self.use_mae = use_mae
        self.use_velocity_magnitude = use_velocity_magnitude
        self.weight_vorticity = weight_vorticity
        self.dz = dx
        self.dy = dy

    def forward(self, pred, target):
        # 数据损失
        if self.use_mae:
            data_loss = F.l1_loss(pred, target)
        else:
            data_loss = F.mse_loss(pred, target)

        n_channels = pred.shape[1]
        if n_channels >= 2 and (self.weight_u != 1.0 or self.weight_v != 1.0 or (n_channels>2 and self.weight_w != 1.0)):
            weights_list = [self.weight_u, self.weight_v]
            if n_channels > 2:
                weights_list.append(self.weight_w)
            weights = torch.tensor(weights_list, device=pred.device).view(1, n_channels, 1, 1)
            component_loss = F.mse_loss(pred * weights, target * weights)
            data_loss = 0.7 * data_loss + 0.3 * component_loss

        if self.use_velocity_magnitude:
            pred_mag = torch.sqrt(torch.sum(pred**2, dim=1, keepdim=True) + 1e-8)
            target_mag = torch.sqrt(torch.sum(target**2, dim=1, keepdim=True) + 1e-8)
            mag_loss = F.mse_loss(pred_mag, target_mag)
            data_loss = 0.7 * data_loss + 0.3 * mag_loss

        # 涡量约束（使用填充导数，保持尺寸）
        # Vorticity regularization on the predicted ZY plane.
        # Layout: pred[:, channel, z, y]. The stored Y coordinate decreases as
        # tensor axis 3 increases, so d/dy is the negative axis-3 derivative.
        v = pred[:, 1:2, ...]
        w = pred[:, 2:3, ...] if n_channels >= 3 else None
        vorticity_loss = 0.0
        if self.weight_vorticity > 0 and w is not None:
            # 计算涡量分量
            dw_dy = -self._derivative(w, axis=3, delta=self.dy, use_padding=True)
            dv_dz = self._derivative(v, axis=2, delta=self.dz, use_padding=True)
            vorticity = dw_dy - dv_dz
            dvort_dz = self._derivative(vorticity, axis=2, delta=self.dz, use_padding=True)
            dvort_dy = self._derivative(vorticity, axis=3, delta=self.dy, use_padding=True)
            vorticity_loss = torch.mean(dvort_dz**2 + dvort_dy**2)

        total_loss = data_loss + self.weight_vorticity * vorticity_loss
        return total_loss

    def _derivative(self, tensor, axis, delta=1.0, use_padding=False):
        """计算中心差分导数。若use_padding=True，返回与输入相同尺寸；否则尺寸减2。"""
        if use_padding:
            if axis == 2:  # x方向（高度维度）
                padded = F.pad(tensor, (0, 0, 1, 1), mode='replicate')
                return (padded[:, :, 2:, :] - padded[:, :, :-2, :]) / (2 * delta)
            elif axis == 3:  # y方向（宽度维度）
                padded = F.pad(tensor, (1, 1, 0, 0), mode='replicate')
                return (padded[:, :, :, 2:] - padded[:, :, :, :-2]) / (2 * delta)
            elif axis == 4:  # z方向（3D流场）
                padded = F.pad(tensor, (0, 0, 0, 0, 1, 1), mode='replicate')
                return (padded[:, :, :, :, 2:] - padded[:, :, :, :, :-2]) / (2 * delta)
            else:
                raise ValueError("Axis must be 2,3,4")
        else:
            if axis == 2:
                return (tensor[:, :, 2:, :] - tensor[:, :, :-2, :]) / (2 * delta)
            elif axis == 3:
                return (tensor[:, :, :, 2:] - tensor[:, :, :, :-2]) / (2 * delta)
            elif axis == 4:
                return (tensor[:, :, :, :, 2:] - tensor[:, :, :, :, :-2]) / (2 * delta)
            else:
                raise ValueError("Axis must be 2,3,4")



from data_loader import get_data_loaders
from models.unet_model import get_unet_model
from models.gan_model import get_gan_models
from models.transformer_model import get_transformer_model

class Trainer:
    def __init__(self, model_type, config):
        self.model_type = model_type
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.setup_directories()
        self.setup_model()
        self.setup_data()
        self.setup_training()
        
    def setup_directories(self):
        self.save_dir = os.path.join('results', self.model_type)
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        self.log_dir = os.path.join(self.save_dir, 'logs')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"Results will be saved to: {self.save_dir}")
    
    def setup_model(self): # 损失函数定义
        if self.model_type == 'unet':
            self.model = get_unet_model().to(self.device)
            # 使用改进的流场损失函数
            self.criterion = FlowFieldLoss(
                # weight_u=1.1, 
                # weight_v=1.0,
                # weight_w=1.0,  # w分量权重更高
                use_mae=False,
                # use_velocity_magnitude=False
            )
        elif self.model_type == 'gan':
            self.generator, self.discriminator = get_gan_models()
            self.generator = self.generator.to(self.device)
            self.discriminator = self.discriminator.to(self.device)
            self.criterion_gan = nn.BCELoss()
            # GAN的像素损失也使用改进的流场损失
            self.criterion_pixel = FlowFieldLoss(
                # weight_u=1.1, 
                # weight_v=1.0,
                # weight_w=1.0,
                use_mae=False,
                # use_velocity_magnitude=True
            )
        elif self.model_type == 'transformer':
            self.model = get_transformer_model().to(self.device)
            # Transformer也使用改进的流场损失函数
            self.criterion = FlowFieldLoss(
                # weight_u=1.1, 
                # weight_v=1.0,
                # weight_w=1.0,
                use_mae=False,
                # use_velocity_magnitude=False
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Model {self.model_type} initialized on {self.device}")
        print(f"Using FlowFieldLoss with weight_u={1.1}, velocity_magnitude_loss=False")
    
    def setup_data(self):
        self.train_loader, self.val_loader, self.test_loader, self.normalizer = get_data_loaders(
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers']
        )
        
        self.normalizer.save(os.path.join(self.save_dir, 'normalizer.npz'))
        print("Data loaded and normalizer saved")
    
    def setup_training(self):
        if self.model_type == 'gan':
            self.optimizer_G = optim.Adam(self.generator.parameters(), 
                                         lr=self.config['lr'], betas=(0.5, 0.999))
            self.optimizer_D = optim.Adam(self.discriminator.parameters(), 
                                         lr=self.config['lr'], betas=(0.5, 0.999))
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        
        self.writer = SummaryWriter(self.log_dir)
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        
        # 存储损失值
        self.train_losses = []
        self.val_losses = []
        if self.model_type == 'gan':
            self.g_losses = []
            self.d_losses = []
        
        if self.config['resume']:
            self.load_checkpoint()
    
    def train_epoch(self, epoch):
        if self.model_type == 'gan':
            return self.train_gan_epoch(epoch)
        else:
            return self.train_standard_epoch(epoch)
    
    def train_standard_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), 
                                     epoch * len(self.train_loader) + batch_idx)
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def train_gan_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            real = torch.ones(batch_size, 1, 1, 1).to(self.device)
            fake = torch.zeros(batch_size, 1, 1, 1).to(self.device)
            
            self.optimizer_D.zero_grad()
            real_output = self.discriminator(targets)
            d_loss_real = self.criterion_gan(real_output, real)
            
            fake_output = self.discriminator(self.generator(inputs).detach())
            d_loss_fake = self.criterion_gan(fake_output, fake)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            self.optimizer_D.step()
            
            self.optimizer_G.zero_grad()
            fake_images = self.generator(inputs)
            fake_output = self.discriminator(fake_images)
            
            g_loss_gan = self.criterion_gan(fake_output, real)
            g_loss_pixel = self.criterion_pixel(fake_images, targets)
            g_loss = g_loss_gan + 100 * g_loss_pixel
            
            g_loss.backward()
            self.optimizer_G.step()
            
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            pbar.set_postfix({'G_loss': g_loss.item(), 'D_loss': d_loss.item()})
            
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/G_loss', g_loss.item(), 
                                     epoch * len(self.train_loader) + batch_idx)
                self.writer.add_scalar('train/D_loss', d_loss.item(), 
                                     epoch * len(self.train_loader) + batch_idx)
        
        avg_g_loss = total_g_loss / len(self.train_loader)
        avg_d_loss = total_d_loss / len(self.train_loader)
        self.g_losses.append(avg_g_loss)
        self.d_losses.append(avg_d_loss)
        self.train_losses.append(avg_g_loss)  # 存储G loss作为训练loss
        
        return {'G_loss': avg_g_loss, 
                'D_loss': avg_d_loss}
    
    def validate(self, epoch):
        if self.model_type == 'gan':
            return self.validate_gan(epoch)
        else:
            return self.validate_standard(epoch)
    
    def validate_standard(self, epoch):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        
        return avg_loss
    
    def validate_gan(self, epoch):
        self.generator.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.generator(inputs)
                loss = self.criterion_pixel(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.model_type == 'gan':
            checkpoint['generator_state_dict'] = self.generator.state_dict()
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            checkpoint['optimizer_G_state_dict'] = self.optimizer_G.state_dict()
            checkpoint['optimizer_D_state_dict'] = self.optimizer_D.state_dict()
        else:
            checkpoint['model_state_dict'] = self.model.state_dict()
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        # torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved with validation loss: {val_loss:.4f}")
    
    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            
            if self.model_type == 'gan':
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
                self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    def train(self):
        print(f"Starting training for {self.config['epochs']} epochs...")
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"{'='*50}")
            
            train_loss = self.train_epoch(epoch)
            
            if self.model_type == 'gan':
                print(f"Train G Loss: {train_loss['G_loss']:.4f}, Train D Loss: {train_loss['D_loss']:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f}")
            
            val_loss = self.validate(epoch)
            print(f"Validation Loss: {val_loss:.4f}")
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, val_loss, is_best=False)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        self.writer.close()
        
        # 保存损失数据
        self.save_loss_data()
        # 生成损失可视化图
        self.plot_losses()
    
    def save_loss_data(self):
        """保存损失数据到文件"""
        loss_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if self.model_type == 'gan':
            loss_data['g_losses'] = self.g_losses
            loss_data['d_losses'] = self.d_losses
        
        loss_path = os.path.join(self.save_dir, 'loss_data.npy')
        np.save(loss_path, loss_data)
        print(f"Loss data saved to: {loss_path}")
        
        # 保存为JSON格式，便于查看
        loss_json_path = os.path.join(self.save_dir, 'loss_data.json')
        with open(loss_json_path, 'w') as f:
            # 转换为Python原生类型
            json_loss_data = {k: [float(v) for v in values] for k, values in loss_data.items()}
            json.dump(json_loss_data, f, indent=4)
        print(f"Loss data (JSON) saved to: {loss_json_path}")
    
    def plot_losses(self):
        """生成损失可视化图"""
        plt.figure(figsize=(12, 8))
        
        if self.model_type == 'gan':
            # GAN模型的损失图
            epochs = range(1, len(self.train_losses) + 1)
            
            plt.subplot(2, 1, 1)
            plt.plot(epochs, self.g_losses, 'b-', label='Generator Loss')
            plt.plot(epochs, self.d_losses, 'r-', label='Discriminator Loss')
            plt.title('GAN Losses')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(epochs, self.val_losses, 'g-', label='Validation Loss')
            plt.title('Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        else:
            # 标准模型的损失图
            epochs = range(1, len(self.train_losses) + 1)
            plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
            plt.plot(epochs, self.val_losses, 'g-', label='Validation Loss')
            plt.title('Training and Validation Losses')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'loss_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Train flow field prediction models')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'gan', 'transformer'],
                       help='Model type to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'num_workers': args.num_workers,
        'save_interval': args.save_interval,
        'resume': args.resume
    }
    
    config_path = os.path.join('results', args.model, 'config.json')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    trainer = Trainer(args.model, config)
    trainer.train()

if __name__ == '__main__':
    main()
