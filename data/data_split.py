import numpy as np

# 加载数据
data_2d = np.load('data\cxp_2d_uv.npy')      # (2, 90, 48, 48)
data_3d = np.load('data\cxp_3d_uvw.npy')     # (3, 90, 64, 48)

# 第一组索引（test set）：从第5个开始，每隔9个提取，共10个
# indices_test = list(range(4, 90, 9))    # [4, 13, 22, 31, 40, 49, 58, 67, 76, 85]
indices_test = [4, 13, 22, 31, 40, 48, 57, 66, 74, 83]

# 第二组索引（train set）：剩余的80个
all_indices = set(range(90))
indices_train = sorted(list(all_indices - set(indices_test)))

# 提取数据
data_2d_test = data_2d[:, indices_test, :, :]    # (2, 10, 48, 48)
data_2d_train = data_2d[:, indices_train, :, :]  # (2, 80, 48, 48)

data_3d_test = data_3d[:, indices_test, :, :]    # (3, 10, 64, 48)
data_3d_train = data_3d[:, indices_train, :, :]  # (3, 80, 64, 48)

# 保存为 npy 文件
np.save('data\cxp_2c_uv_train.npy', data_2d_train)
np.save('data\cxp_2c_uv_test.npy', data_2d_test)
np.save('data\cxp_3c_uvw_train.npy', data_3d_train)
np.save('data\cxp_3c_uvw_test.npy', data_3d_test)

print("保存完成！")
print(f"2D Train: {data_2d_train.shape}, 2D Test: {data_2d_test.shape}")
print(f"3D Train: {data_3d_train.shape}, 3D Test: {data_3d_test.shape}")