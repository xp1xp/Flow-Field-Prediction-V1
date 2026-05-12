import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path


class CrossSectionVisualizer:
    def __init__(self, data_dir=None):
        """
        初始化可视化器

        data_dir:
            数据文件夹路径。
            如果不指定，默认使用当前项目下的 data 文件夹。
        """

        if data_dir is None:
            self.data_dir = Path.cwd() / 'data'
        else:
            self.data_dir = Path(data_dir)

        self.load_data()

    def load_data(self):
        """加载数据"""
        print("正在加载数据...")

        path_2d = self.data_dir / 'cxp_2d_uv.npy'
        path_3d = self.data_dir / 'cxp_3d_uvw.npy'
        path_coords_2d = self.data_dir / 'cxp_2d_coords.npy'
        path_coords_3d = self.data_dir / 'cxp_3d_coords.npy'

        for path in [path_2d, path_3d, path_coords_2d, path_coords_3d]:
            if not path.exists():
                raise FileNotFoundError(f"找不到文件: {path}")

        self.data_2d = np.load(path_2d)
        self.data_3d = np.load(path_3d)

        self.coords_2d = np.load(path_coords_2d)
        self.coords_3d = np.load(path_coords_3d)

        print(f"2D数据形状: {self.data_2d.shape}")
        print(f"3D数据形状: {self.data_3d.shape}")
        print(f"2D坐标形状: {self.coords_2d.shape}")
        print(f"3D坐标形状: {self.coords_3d.shape}")

    def imshow_square_pixels(
        self,
        ax,
        data,
        x_coord,
        y_coord,
        cmap='jet',
        vmin=None,
        vmax=None,
        rotate90=True,
        rotate_k=1,
        swap_axes=True
    ):
        """
        绘制图像，并保证每个像素格为标准方格

        rotate90:
            是否旋转90度

        rotate_k:
            1  : 逆时针旋转90度
            -1 : 顺时针旋转90度

        swap_axes:
            是否对调横纵坐标
        """

        x_min, x_max = x_coord.min(), x_coord.max()
        y_min, y_max = y_coord.min(), y_coord.max()

        data_plot = data.copy()

        # Step 1: 图像旋转90度
        if rotate90:
            data_plot = np.rot90(data_plot, k=rotate_k)

        # Step 2: 坐标对调
        if swap_axes:
            # 横轴使用原来的 y，纵轴使用原来的 x
            extent = [y_min, y_max, x_min, x_max]
        else:
            # 横轴使用原来的 x，纵轴使用原来的 y
            extent = [x_min, x_max, y_min, y_max]

        nrows, ncols = data_plot.shape

        im = ax.imshow(
            data_plot,
            cmap=cmap,
            origin='lower',
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )

        # 保证像素格为标准方格
        cell_dx = abs(extent[1] - extent[0]) / ncols
        cell_dy = abs(extent[3] - extent[2]) / nrows

        ax.set_aspect(cell_dx / cell_dy, adjustable='box')

        return im

    def visualize_cross_sections(
        self,
        case_idx=0,
        component='u',
        rotate90=True,
        rotate_k=1,
        swap_axes=True
    ):
        """
        可视化正交截面

        case_idx:
            工况索引

        component:
            'u', 'v', 'w'

        rotate90:
            是否旋转90度

        rotate_k:
            1  逆时针旋转90度
            -1 顺时针旋转90度

        swap_axes:
            是否对调每张图的横纵坐标
        """

        print(f"\n可视化工况 {case_idx} 的 {component} 速度分量...")

        if component not in ['u', 'v', 'w']:
            raise ValueError("component 只能是 'u', 'v', 或 'w'")

        comp_idx = {'u': 0, 'v': 1, 'w': 2}[component]

        if case_idx < 0 or case_idx >= self.data_3d.shape[1]:
            raise ValueError(f"case_idx 超出范围，应在 0 到 {self.data_3d.shape[1] - 1} 之间")

        if component == 'w':
            print("提示: 2D数据中没有 w 分量，只显示 3D 截面")
            show_2d = False
        else:
            show_2d = True

        if show_2d:
            data_2d_comp = self.data_2d[comp_idx, case_idx, :, :]
            x_2d = self.coords_2d[0, :, :]
            y_2d = self.coords_2d[1, :, :]

        data_3d_comp = self.data_3d[comp_idx, case_idx, :, :]
        z_3d = self.coords_3d[0, :, :]
        y_3d = self.coords_3d[1, :, :]

        if show_2d:
            vmin = min(data_2d_comp.min(), data_3d_comp.min())
            vmax = max(data_2d_comp.max(), data_3d_comp.max())
        else:
            vmin = data_3d_comp.min()
            vmax = data_3d_comp.max()

        if show_2d:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            fig.suptitle(
                f'Case {case_idx} - {component.upper()} Velocity Cross Sections',
                fontsize=14
            )
        else:
            fig, ax2 = plt.subplots(1, 1, figsize=(7, 7))
            fig.suptitle(
                f'Case {case_idx} - {component.upper()} Velocity 3D Cross Section',
                fontsize=14
            )

        # =========================
        # 2D XY 截面
        # =========================
        if show_2d:
            im1 = self.imshow_square_pixels(
                ax=ax1,
                data=data_2d_comp,
                x_coord=x_2d,
                y_coord=y_2d,
                cmap='jet',
                vmin=vmin,
                vmax=vmax,
                rotate90=rotate90,
                rotate_k=rotate_k,
                swap_axes=swap_axes
            )

            ax1.set_title('XY Section')

            if swap_axes:
                ax1.set_xlabel('Y')
                ax1.set_ylabel('X')
            else:
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')

            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax1, label=f'{component.upper()} Velocity')

        # =========================
        # 3D ZY 截面
        # =========================
        im2 = self.imshow_square_pixels(
            ax=ax2,
            data=data_3d_comp,
            x_coord=z_3d,
            y_coord=y_3d,
            cmap='jet',
            vmin=vmin,
            vmax=vmax,
            rotate90=rotate90,
            rotate_k=rotate_k,
            swap_axes=swap_axes
        )

        ax2.set_title('ZY Section')

        if swap_axes:
            ax2.set_xlabel('Y')
            ax2.set_ylabel('Z')
        else:
            ax2.set_xlabel('Z')
            ax2.set_ylabel('Y')

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax2, label=f'{component.upper()} Velocity')

        if show_2d:
            ax1.tick_params(direction='in')
        ax2.tick_params(direction='in')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = f'cross_section_case{case_idx}_{component}_rot90_swap_axes_square.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"可视化结果已保存到: {save_path}")

        plt.show()


def main():
    print("=====================================")
    print("正交截面可视化工具")
    print("=====================================")

    visualizer = CrossSectionVisualizer()

    visualizer.visualize_cross_sections(
        case_idx=0,
        component='u',
        rotate90=True,
        rotate_k=1,
        swap_axes=True
    )

    print("\n可视化完成！")


if __name__ == '__main__':
    main()