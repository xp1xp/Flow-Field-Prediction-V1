import numpy as np
import matplotlib.pyplot as plt

def main():
    data_2d = np.load("./data/cxp_2c_uv_train.npy")
    
    # =========================
    # 2. 看看形状
    print("data_2d.shape:", data_2d.shape)
    # print("data_2d[0, 0, :, 0]:", data_2d[0, 0, :, 0])

    coords_2d = np.load("./data/cxp_2d_coords.npy")
    # coords_3d = np.load(path_coords_3d)

    print("2D数据形状: ",coords_2d.shape)
    # print("coords_2d[0, :, 0]:", coords_2d[0, :, 0])
    # print(f"3D数据形状: {self.data_3d.shape}")    

    # =========================
    # 3. 绘制曲线
    x = coords_2d[0, :, 0]      # 对应坐标
    u = data_2d[0, 0, :, 47]     # 第0个速度分量，第0个工况，第0列
    print("x.shape:", x.shape)
    print("u.shape:", u.shape)
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(
        x,
        u,
        marker='o',
        markersize=4,
        linewidth=1.5,
    )
    plt.xlabel('Coordinate')
    plt.ylabel('Velocity u')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    
    # =========================
    # 4. 绘制云图
    U = data_2d[0, 0, :, :]      # 云图数据，shape = (48, 48)
    X = coords_2d[0, :, :]
    Y = coords_2d[1, :, :]
    plt.figure(figsize=(6, 5), dpi=300)
    contour = plt.contourf(X, Y, U, levels=50, cmap='jet')
    plt.colorbar(contour, label='Velocity u')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
