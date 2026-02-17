"""
DCN-v2 Data Loading Utilities
==============================

Dataset and DataLoader utilities for DCN-v2 with native categorical support.

Key Components:
    - DCNv2Dataset: PyTorch Dataset yielding (num_dict, cat_dict, label)
    - dcnv2_collate_fn: Custom collate function for Dict-based batching
    - create_dcn_dataloaders: Helper to create train/val/test loaders

Usage:
    from models.dcn_v2.data_utils import DCNv2Dataset, dcnv2_collate_fn, create_dcn_dataloaders

    train_loader, val_loader, test_loader = create_dcn_dataloaders(
        data_path='data/processed',
        batch_size=1024
    )

    for num_vals, cat_vals, labels in train_loader:
        probs = model(num_vals, cat_vals)
        loss = criterion(probs, labels)

# Purpose: Standardized data loading for DCN-v2
# Why: Handles both numerical and categorical features correctly
# Approved: 2026-02-16
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class DCNv2Dataset(Dataset):
    """
    PyTorch Dataset for DCN-v2 with separate numerical and categorical features.

    Args:
        X_num: DataFrame or path to numerical features parquet
        X_cat: DataFrame or path to categorical features parquet
        y: DataFrame or path to labels parquet (None for inference)
        num_features: List of numerical feature names (inferred if None)
        cat_features: List of categorical feature names (inferred if None)

    Yields:
        Tuple of (numerical_dict, categorical_dict, label) or
        Tuple of (numerical_dict, categorical_dict) if y is None

    # Purpose: Load DCN-v2 data with separate num/cat handling
    # Why: DCN-v2 expects Dict[str, Tensor] for both feature types
    # Approved: 2026-02-16
    """

    def __init__(
        self,
        X_num: pd.DataFrame,
        X_cat: pd.DataFrame,
        y: Optional[pd.DataFrame] = None,
        num_features: Optional[List[str]] = None,
        cat_features: Optional[List[str]] = None,
    ):
        # Store as numpy arrays for fast access
        self.X_num = X_num.values.astype(np.float32)
        self.X_cat = X_cat.values.astype(np.int64)  # Long for embedding lookup

        # Feature names
        self.num_features = num_features or list(X_num.columns)
        self.cat_features = cat_features or list(X_cat.columns)

        # Labels (optional for inference)
        if y is not None:
            self.y = y.values.astype(np.float32).flatten()
            self.has_labels = True
        else:
            self.y = None
            self.has_labels = False

    def __len__(self) -> int:
        return len(self.X_num)

    def __getitem__(self, idx: int):
        # Numerical features dict
        num_row = self.X_num[idx]
        numerical_values = {
            name: torch.tensor(num_row[i], dtype=torch.float32)
            for i, name in enumerate(self.num_features)
        }

        # Categorical features dict
        cat_row = self.X_cat[idx]
        categorical_values = {
            name: torch.tensor(cat_row[i], dtype=torch.long)
            for i, name in enumerate(self.cat_features)
        }

        if self.has_labels:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return numerical_values, categorical_values, label
        else:
            return numerical_values, categorical_values


def dcnv2_collate_fn(batch):
    """
    Custom collate function for DCN-v2 batching.

    Handles both labeled (train/val/test) and unlabeled (inference) data.

    Args:
        batch: List of tuples from DCNv2Dataset

    Returns:
        If labeled: (batched_num_dict, batched_cat_dict, batched_labels)
        If unlabeled: (batched_num_dict, batched_cat_dict)

    # Purpose: Stack Dict features into batched tensors
    # Why: Default collate doesn't handle Dict[str, Tensor] inputs
    # Approved: 2026-02-16
    """
    # Check if batch has labels (3-tuple) or not (2-tuple)
    has_labels = len(batch[0]) == 3

    if has_labels:
        num_dicts, cat_dicts, labels = zip(*batch)
    else:
        num_dicts, cat_dicts = zip(*batch)
        labels = None

    # Batch numerical features
    num_features = list(num_dicts[0].keys())
    batched_numerical = {
        name: torch.stack([d[name] for d in num_dicts])
        for name in num_features
    }

    # Batch categorical features
    cat_features = list(cat_dicts[0].keys())
    batched_categorical = {
        name: torch.stack([d[name] for d in cat_dicts])
        for name in cat_features
    }

    if has_labels:
        batched_labels = torch.stack(labels).unsqueeze(1)  # Shape: (batch, 1)
        return batched_numerical, batched_categorical, batched_labels
    else:
        return batched_numerical, batched_categorical


def load_dcn_data(data_path: str, split: str = 'train'):
    """
    Load DCN-v2 preprocessed data.

    Args:
        data_path: Path to data/processed directory
        split: One of 'train', 'val', 'test', 'reject'

    Returns:
        Tuple of (X_num, X_cat, y) DataFrames
        y is None for 'reject' split

    # Purpose: Load preprocessed DCN-v2 data files
    # Why: Centralizes file loading logic
    # Approved: 2026-02-16
    """
    data_path = Path(data_path)

    X_num = pd.read_parquet(data_path / f"dcn_X_{split}_num.parquet")
    X_cat = pd.read_parquet(data_path / f"dcn_X_{split}_cat.parquet")

    if split == 'reject':
        y = None
    else:
        y = pd.read_parquet(data_path / f"dcn_y_{split}.parquet")

    return X_num, X_cat, y


def load_cardinalities(data_path: str) -> Dict[str, int]:
    """
    Load categorical cardinalities from JSON.

    Args:
        data_path: Path to data/processed directory

    Returns:
        Dict mapping feature name to cardinality

    # Purpose: Load cardinalities for model initialization
    # Why: DCN-v2 needs cardinalities to create embedding layers
    # Approved: 2026-02-16
    """
    data_path = Path(data_path)
    with open(data_path / "dcn_cardinalities.json", 'r') as f:
        return json.load(f)


def create_dcn_dataloaders(
    data_path: str,
    batch_size: int = 1024,
    num_workers: int = 0,
    include_reject: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for DCN-v2 training.

    Args:
        data_path: Path to data/processed directory
        batch_size: Batch size for training
        num_workers: Number of worker processes
        include_reject: Whether to include reject loader

    Returns:
        Tuple of (train_loader, val_loader, test_loader, reject_loader)
        reject_loader is None if include_reject=False

    # Purpose: One-call DataLoader creation for DCN-v2
    # Why: Simplifies notebook setup code
    # Approved: 2026-02-16
    """
    # Load data
    X_train_num, X_train_cat, y_train = load_dcn_data(data_path, 'train')
    X_val_num, X_val_cat, y_val = load_dcn_data(data_path, 'val')
    X_test_num, X_test_cat, y_test = load_dcn_data(data_path, 'test')

    # Get feature names
    num_features = list(X_train_num.columns)
    cat_features = list(X_train_cat.columns)

    # Create datasets
    train_dataset = DCNv2Dataset(X_train_num, X_train_cat, y_train, num_features, cat_features)
    val_dataset = DCNv2Dataset(X_val_num, X_val_cat, y_val, num_features, cat_features)
    test_dataset = DCNv2Dataset(X_test_num, X_test_cat, y_test, num_features, cat_features)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dcnv2_collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dcnv2_collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dcnv2_collate_fn,
        num_workers=num_workers,
    )

    # Reject loader (optional)
    reject_loader = None
    if include_reject:
        X_reject_num, X_reject_cat, _ = load_dcn_data(data_path, 'reject')
        reject_dataset = DCNv2Dataset(X_reject_num, X_reject_cat, None, num_features, cat_features)
        reject_loader = DataLoader(
            reject_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dcnv2_collate_fn,
            num_workers=num_workers,
        )

    return train_loader, val_loader, test_loader, reject_loader
