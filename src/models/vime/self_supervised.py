"""
VIME Self-Supervised Training Loop (Step 2).

Training on the full population (D_L ∪ D_R) using:
  - Mask Cross-Entropy loss  (does the model know which features were corrupted?)
  - Feature Reconstruction MSE  (can the model reconstruct original values?)
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from .encoder import VIMEModel


def corrupt_features(
    x: torch.Tensor,
    corruption_rate: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask a fraction of features.

    Returns:
        x_corrupted: input with masked features replaced by shuffled values
        mask:        binary tensor (1 = corrupted, 0 = original)
    """
    mask = torch.bernoulli(torch.full_like(x, corruption_rate))
    # Replace corrupted positions with values from random rows (shuffle)
    shuffle_idx = torch.randperm(x.size(0))
    x_corrupted = x.clone()
    x_corrupted[mask.bool()] = x[shuffle_idx][mask.bool()]
    return x_corrupted, mask


def train_vime(
    model: VIMEModel,
    data: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    corruption_rate: float = 0.3,
    mask_loss_weight: float = 1.0,
    recon_loss_weight: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """
    Train the VIME model in self-supervised mode.

    Args:
        model:  VIMEModel instance
        data:   numpy array of shape (n_samples, n_features) — full population
        ...

    Returns:
        history: dict with loss curves {'total', 'mask', 'recon'}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mask_criterion = nn.BCELoss()
    recon_criterion = nn.MSELoss()

    tensor_data = torch.FloatTensor(data)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"total": [], "mask": [], "recon": []}

    for epoch in range(epochs):
        model.train()
        epoch_losses = {"total": 0.0, "mask": 0.0, "recon": 0.0}

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            x_corrupted, mask = corrupt_features(batch_x, corruption_rate)
            x_corrupted = x_corrupted.to(device)
            mask = mask.to(device)

            # Forward pass
            z, m_hat, x_hat = model(x_corrupted)

            # Losses
            loss_mask = mask_criterion(m_hat, mask)
            loss_recon = recon_criterion(x_hat, batch_x)
            loss = mask_loss_weight * loss_mask + recon_loss_weight * loss_recon

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses["total"] += loss.item()
            epoch_losses["mask"] += loss_mask.item()
            epoch_losses["recon"] += loss_recon.item()

        n_batches = len(loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
            history[key].append(epoch_losses[key])

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} — "
                f"Total: {epoch_losses['total']:.4f}  "
                f"Mask: {epoch_losses['mask']:.4f}  "
                f"Recon: {epoch_losses['recon']:.4f}"
            )

    return history
