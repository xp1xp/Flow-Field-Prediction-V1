import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

class CrossSection3DVisualizer:
    def __init__(self, data_dir='.'):
        """初始化3D可视化器"""
        self.data_dir = data_dir
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"当前工作目录: {os.getcwd()}")
        print(f"脚本所在目录: {current_dir}")
        
        # 加载速度场数据
        data_2d_path = os.path.join(current_dir, 'cxp_2d_uv.npy')
        data_3d_path = os.path.join(current_dir, 'cxp_3d_uvw.npy')
        coords_2d_path = os.path.join(current_dir, 'cxp_2d_coords.npy')
        coords_3d_path = os.path.join(current_dir, 'cxp_3d_coords.npy')
        
        print(f"2D数据路径: {data_2d_path}")
        print(f"3D数据路径: {data_3d_path}")
        
        # 检查文件是否存在
        for path in [data_2d_path, data_3d_path, coords_2d_path, coords_3d_path]:
            if os.path.exists(path):
                print(f"✓ 文件存在: {os.path.basename(path)}")
            else:
                print(f"✗ 文件不存在: {path}")
        
        self.data_2d = np.load(data_2d_path)  # (2, 90, 48, 48) [u, v, 工况, Y, X]
        self.data_3d = np.load(data_3d_path)  # (3, 90, 64, 48) [u, v, w, 工况, Z, Y]
        
        # 加载坐标数据
        self.coords_2d = np.load(coords_2d_path)  # (2, 48, 48) [X, Y]
        self.coords_3d = np.load(coords_3d_path)  # (2, 64, 48) [Z, Y]
        
        print(f"2D数据形状: {self.data_2d.shape}")
        print(f"3D数据形状: {self.data_3d.shape}")
        print(f"2D坐标形状: {self.coords_2d.shape}")
        print(f"3D坐标形状: {self.coords_3d.shape}")
    
    def get_3d_coordinates(self, z_2d_position=0.0, x_3d_position=0.0):
        """获取3D坐标
        
        Args:
            z_2d_position: 2D截面（XY平面）的Z坐标位置
            x_3d_position: 3D截面（ZY平面）的X坐标位置
        """
        # 2D截面 (XY平面)
        x_2d = self.coords_2d[0, :, :]  # (48, 48)
        y_2d = self.coords_2d[1, :, :]  # (48, 48)
        z_2d = np.full_like(x_2d, z_2d_position)  # Z=自定义位置
        
        # 3D截面 (ZY平面)
        z_3d = self.coords_3d[0, :, :]  # (64, 48)
        y_3d = self.coords_3d[1, :, :]  # (64, 48)
        x_3d = np.full_like(z_3d, x_3d_position)  # X=自定义位置
        
        return x_2d, y_2d, z_2d, x_3d, y_3d, z_3d
    
    def visualize_3d_cross_sections(self, case_idx=0, component='u', z_2d_position=0.0, x_3d_position=0.0):
        """在3D空间中可视化正交截面
        
        Args:
            case_idx: 工况索引 (0-89)
            component: 速度分量 ('u', 'v', 'w')
            z_2d_position: 2D截面（XY平面）的Z坐标位置
            x_3d_position: 3D截面（ZY平面）的X坐标位置
        """
        print(f"\n在3D空间中可视化工况 {case_idx} 的 {component} 速度分量...")
        
        # 确定分量索引
        comp_idx = {'u': 0, 'v': 1, 'w': 2}[component]
        
        # 检查分量是否存在
        if component == 'w' and comp_idx >= self.data_2d.shape[0]:
            print("警告: 2D数据中没有w分量，只显示3D数据")
            show_2d = False
        else:
            show_2d = True
        
        # 提取数据
        if show_2d:
            data_2d_comp = self.data_2d[comp_idx, case_idx, :, :]  # (48, 48)
        data_3d_comp = self.data_3d[comp_idx, case_idx, :, :]  # (64, 48)
        
        # 获取3D坐标
        x_2d, y_2d, z_2d, x_3d, y_3d, z_3d = self.get_3d_coordinates(z_2d_position, x_3d_position)
        
        # 确定颜色范围
        if show_2d:
            vmin = min(data_2d_comp.min(), data_3d_comp.min())
            vmax = max(data_2d_comp.max(), data_3d_comp.max())
        else:
            vmin = data_3d_comp.min()
            vmax = data_3d_comp.max()
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'3D-flow-field-Case {case_idx} - {component.upper()} ', fontsize=14)
        
        # 绘制2D截面 (XY平面)
        if show_2d:
            surf1 = ax.plot_surface(x_2d, y_2d, z_2d, 
                                  facecolors=cm.jet((data_2d_comp - vmin) / (vmax - vmin)),
                                  alpha=0.8, 
                                  rstride=1, cstride=1, 
                                  antialiased=True)
            # ax.text(x_2d.max() * 1.1, y_2d.mean(), z_2d_position, 'XY', fontsize=10)
        
        # 绘制3D截面 (ZY平面)
        surf2 = ax.plot_surface(x_3d, y_3d, z_3d, 
                              facecolors=cm.jet((data_3d_comp - vmin) / (vmax - vmin)),
                              alpha=0.8, 
                              rstride=1, cstride=1, 
                              antialiased=True)
        # ax.text(x_3d_position, y_3d.mean(), z_3d.max() * 1.1, 'ZY', fontsize=10)
        
        # 添加颜色条
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array([vmin, vmax])
        cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label(f'{component.upper()} ', rotation=270, labelpad=20)
        
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 调整视角
        ax.view_init(elev=30, azim=45)
        
        # 保存结果
        save_path = os.path.join('data', f'3d_cross_section_case{case_idx}_{component}.png')
        # save_path = f'3d_cross_section_case{case_idx}_{component}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D可视化结果已保存到: {save_path}")
        
        # 显示图形
        plt.show()
    
    def visualize_3d_scatter(self, case_idx=0, component='u', sample_rate=2, z_2d_position=0.0, x_3d_position=0.0):
        """使用散点图在3D空间中可视化正交截面
        
        Args:
            case_idx: 工况索引 (0-89)
            component: 速度分量 ('u', 'v', 'w')
            sample_rate: 采样率，减少点的数量以提高性能
            z_2d_position: 2D截面（XY平面）的Z坐标位置
            x_3d_position: 3D截面（ZY平面）的X坐标位置
        """
        print(f"\n在3D空间中用散点图可视化工况 {case_idx} 的 {component} 速度分量...")
        
        # 确定分量索引
        comp_idx = {'u': 0, 'v': 1, 'w': 2}[component]
        
        # 提取数据
        data_2d_comp = self.data_2d[comp_idx, case_idx, ::sample_rate, ::sample_rate]  # 降采样
        data_3d_comp = self.data_3d[comp_idx, case_idx, ::sample_rate, ::sample_rate]  # 降采样
        
        # 获取3D坐标
        x_2d, y_2d, z_2d, x_3d, y_3d, z_3d = self.get_3d_coordinates(z_2d_position, x_3d_position)
        x_2d = x_2d[::sample_rate, ::sample_rate]
        y_2d = y_2d[::sample_rate, ::sample_rate]
        z_2d = z_2d[::sample_rate, ::sample_rate]
        x_3d = x_3d[::sample_rate, ::sample_rate]
        y_3d = y_3d[::sample_rate, ::sample_rate]
        z_3d = z_3d[::sample_rate, ::sample_rate]
        
        # 确定颜色范围
        vmin = min(data_2d_comp.min(), data_3d_comp.min())
        vmax = max(data_2d_comp.max(), data_3d_comp.max())
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'3D散点图 - 工况 {case_idx} - {component.upper()} 速度', fontsize=14)
        
        # 绘制2D截面散点
        scatter1 = ax.scatter(x_2d.flatten(), y_2d.flatten(), z_2d.flatten(), 
                            c=data_2d_comp.flatten(), 
                            cmap='jet', 
                            vmin=vmin, vmax=vmax, 
                            alpha=0.6, s=10)
        
        # 绘制3D截面散点
        scatter2 = ax.scatter(x_3d.flatten(), y_3d.flatten(), z_3d.flatten(), 
                            c=data_3d_comp.flatten(), 
                            cmap='jet', 
                            vmin=vmin, vmax=vmax, 
                            alpha=0.6, s=10)
        
        # 添加颜色条
        cbar = fig.colorbar(scatter1, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label(f'{component.upper()} 速度', rotation=270, labelpad=20)
        
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 调整视角
        ax.view_init(elev=30, azim=45)
        
        # 保存结果
        save_path = f'3d_scatter_case{case_idx}_{component}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D散点图结果已保存到: {save_path}")
        
        # 显示图形
        plt.show()

def main():
    """主函数"""
    print("=====================================")
    print("3D正交截面可视化工具")
    print("=====================================")
    
    # 创建可视化器
    visualizer = CrossSection3DVisualizer()
    
    # 可视化指定工况和分量
    # 这里选择第0个工况，u分量
    # 自定义位置：XY平面在Z=5.0位置，ZY平面在X=2.5位置
    visualizer.visualize_3d_cross_sections(
        case_idx=0, 
        component='u',
        z_2d_position=0.0105,  # XY平面的Z坐标位置
        x_3d_position=0.03566   # ZY平面的X坐标位置
    )
    
    # 可选：使用散点图可视化
    # visualizer.visualize_3d_scatter(
    #     case_idx=0, 
    #     component='u', 
    #     sample_rate=2,
    #     z_2d_position=5.0,
    #     x_3d_position=2.5
    # )
    
    print("\n3D可视化完成！")

if __name__ == '__main__':
    main()