import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from normalization import DataNormalizer


class FlowDataset(Dataset):
    def __init__(self, data_2d, data_3d, transform=None):
        self.data_2d = data_2d
        self.data_3d = data_3d
        self.transform = transform
        self.length = data_2d.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data_2d[:, idx, :, :].astype(np.float32))
        y = torch.from_numpy(self.data_3d[:, idx, :, :].astype(np.float32))

        if self.transform:
            x, y = self.transform(x, y)

        return x, y


def load_flow_arrays(
    data_dir="data",
    train_2d_name="cxp_2c_uv_train.npy",
    train_3d_name="cxp_3c_uvw_train.npy",
    test_2d_name="cxp_2c_uv_test.npy",
    test_3d_name="cxp_3c_uvw_test.npy",
):
    train_2d = np.load(os.path.join(data_dir, train_2d_name))
    train_3d = np.load(os.path.join(data_dir, train_3d_name))

    test_2d_path = os.path.join(data_dir, test_2d_name)
    test_3d_path = os.path.join(data_dir, test_3d_name)
    test_2d = np.load(test_2d_path) if os.path.exists(test_2d_path) else None
    test_3d = np.load(test_3d_path) if os.path.exists(test_3d_path) else None

    return train_2d, train_3d, test_2d, test_3d


def split_train_val_data(data_2d, data_3d, train_ratio=0.9, val_ratio=0.1):
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must both be positive")

    total_ratio = train_ratio + val_ratio
    total_samples = data_2d.shape[1]
    train_size = int(total_samples * train_ratio / total_ratio)
    val_size = total_samples - train_size

    if train_size <= 0 or val_size <= 0:
        raise ValueError("train_ratio and val_ratio must produce non-empty train and validation sets")

    train_2d = data_2d[:, :train_size]
    train_3d = data_3d[:, :train_size]
    val_2d = data_2d[:, train_size:train_size + val_size]
    val_3d = data_3d[:, train_size:train_size + val_size]
    return train_2d, train_3d, val_2d, val_3d


def build_normalized_datasets(train_2d, train_3d, val_2d, val_3d, test_2d=None, test_3d=None):
    normalizer = DataNormalizer()
    normalizer.fit(train_2d, train_3d)

    train_dataset = FlowDataset(
        normalizer.transform_2d(train_2d),
        normalizer.transform_3d(train_3d),
    )
    val_dataset = FlowDataset(
        normalizer.transform_2d(val_2d),
        normalizer.transform_3d(val_3d),
    )

    if test_2d is not None and test_3d is not None:
        test_dataset = FlowDataset(
            normalizer.transform_2d(test_2d),
            normalizer.transform_3d(test_3d),
        )
    else:
        test_dataset = FlowDataset(
            np.empty((train_2d.shape[0], 0, train_2d.shape[2], train_2d.shape[3]), dtype=train_2d.dtype),
            np.empty((train_3d.shape[0], 0, train_3d.shape[2], train_3d.shape[3]), dtype=train_3d.dtype),
        )

    return train_dataset, val_dataset, test_dataset, normalizer


def load_and_preprocess_data(data_dir="data", train_ratio=0.9, val_ratio=0.1, test_ratio=0):
    del test_ratio

    raw_train_2d, raw_train_3d, raw_test_2d, raw_test_3d = load_flow_arrays(data_dir)

    print(f"Train 2D data shape: {raw_train_2d.shape}")
    print(f"Train 3D data shape: {raw_train_3d.shape}")
    if raw_test_2d is not None and raw_test_3d is not None:
        print(f"External test 2D data shape: {raw_test_2d.shape}")
        print(f"External test 3D data shape: {raw_test_3d.shape}")

    train_2d, train_3d, val_2d, val_3d = split_train_val_data(
        raw_train_2d,
        raw_train_3d,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    train_dataset, val_dataset, test_dataset, normalizer = build_normalized_datasets(
        train_2d,
        train_3d,
        val_2d,
        val_3d,
        raw_test_2d,
        raw_test_3d,
    )
    
    print("\nDataset split:")
    print(f"Train: {train_2d.shape[1]} samples")
    print(f"Validation: {val_2d.shape[1]} samples")
    print(f"Test: {0 if raw_test_2d is None else raw_test_2d.shape[1]} samples")

    return train_dataset, val_dataset, test_dataset, normalizer


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=8, num_workers=0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_data_loaders(data_dir="data", batch_size=8, num_workers=0):
    train_dataset, val_dataset, test_dataset, normalizer = load_and_preprocess_data(data_dir)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, normalizer
