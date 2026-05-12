
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


data = loadmat('data\data_real2c.mat')
data_real2c = data['data_real2c']
print(data_real2c.shape)
fangzhen = np.load('data\cxp_2c_uv_test.npy')
print(fangzhen.shape)



data_real2c = np.rot90(data_real2c, k=-1, axes=(1, 2))
# data_real2c = np.flip(data_real2c, axis=2)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(data_real2c[0,:,:], cmap='jet')
plt.title('u')
plt.subplot(1,2,2)
plt.imshow(fangzhen[0,0,:,:], cmap='jet')
plt.title('fangzhen')



plt.show()
np.save('data\data_real2c.npy', data_real2c)