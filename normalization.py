import numpy as np


class DataNormalizer:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean_2d = None
        self.std_2d = None
        self.mean_3d = None
        self.std_3d = None

    def fit(self, data_2d, data_3d):
        self.mean_2d = data_2d.mean(axis=(1, 2, 3), keepdims=True)
        self.std_2d = np.maximum(data_2d.std(axis=(1, 2, 3), keepdims=True), self.eps)
        self.mean_3d = data_3d.mean(axis=(1, 2, 3), keepdims=True)
        self.std_3d = np.maximum(data_3d.std(axis=(1, 2, 3), keepdims=True), self.eps)
        return self

    def transform_2d(self, data_2d):
        if self.mean_2d is None or self.std_2d is None:
            raise ValueError("Normalizer has not been fitted yet")

        if len(data_2d.shape) == 3:
            mean_2d = self.mean_2d.reshape(-1, 1, 1)
            std_2d = self.std_2d.reshape(-1, 1, 1)
            return (data_2d - mean_2d) / std_2d
        return (data_2d - self.mean_2d) / self.std_2d

    def transform_3d(self, data_3d):
        if self.mean_3d is None or self.std_3d is None:
            raise ValueError("Normalizer has not been fitted yet")

        if len(data_3d.shape) == 3:
            mean_3d = self.mean_3d.reshape(-1, 1, 1)
            std_3d = self.std_3d.reshape(-1, 1, 1)
            return (data_3d - mean_3d) / std_3d
        return (data_3d - self.mean_3d) / self.std_3d

    def inverse_transform_3d(self, data_3d_normalized):
        if self.mean_3d is None or self.std_3d is None:
            raise ValueError("Normalizer has not been fitted yet")

        if len(data_3d_normalized.shape) == 3:
            mean_3d = self.mean_3d.reshape(-1, 1, 1)
            std_3d = self.std_3d.reshape(-1, 1, 1)
            return data_3d_normalized * std_3d + mean_3d
        return data_3d_normalized * self.std_3d + self.mean_3d

    def save(self, filepath):
        np.savez(
            filepath,
            mean_2d=self.mean_2d,
            std_2d=self.std_2d,
            mean_3d=self.mean_3d,
            std_3d=self.std_3d,
        )

    def load(self, filepath):
        data = np.load(filepath)
        self.mean_2d = data["mean_2d"]
        self.std_2d = data["std_2d"]
        self.mean_3d = data["mean_3d"]
        self.std_3d = data["std_3d"]
        return self
