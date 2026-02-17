"""
DCN-v2 Training Utilities
==========================

Training and evaluation functions for DCN-v2.
Includes Focal Loss and Diamond-specific metrics (Precision@k).

# Purpose: Centralize training logic for clean notebooks
# Why: Minimize notebook code, maximize reusability
# Approved: 2026-02-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple, List, Optional
import os


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    Focuses on hard examples by down-weighting easy ones.
    FL(p) = -alpha * (1-p)^gamma * log(p)

    Args:
        alpha: Weight for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard examples

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)

    # Purpose: Better loss for imbalanced data
    # Why: Standard BCE weights all samples equally; Focal Loss focuses on hard ones
    # Approved: 2026-02-16
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model outputs (before sigmoid), shape (batch, 1)
            targets: Binary labels, shape (batch, 1)

        Returns:
            Scalar loss
        """
        probs = torch.sigmoid(logits)

        # Compute focal weights
        # For positive samples (target=1): weight = (1-p)^gamma
        # For negative samples (target=0): weight = p^gamma
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Combined focal loss
        loss = alpha_weight * focal_weight * bce

        return loss.mean()


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.20) -> Tuple[float, int]:
    """
    Compute Precision@k for diamond recovery.

    Among the top k% lowest-risk predictions (sorted by ascending P(default)),
    what fraction are truly good borrowers (label=0)?

    Args:
        y_true: True labels (1=default, 0=good)
        y_pred: Predicted P(default)
        k: Fraction of samples to consider (default: 0.20 = top 20%)

    Returns:
        (precision, n_diamonds): Precision and count of diamonds found

    # Purpose: Measure diamond recovery quality
    # Why: AUC doesn't capture what we care about - finding good borrowers
    # Approved: 2026-02-16
    """
    n_samples = len(y_true)
    n_top_k = int(n_samples * k)

    # Sort by predicted probability (ascending = lowest risk first)
    sorted_indices = np.argsort(y_pred)
    top_k_indices = sorted_indices[:n_top_k]

    # Count diamonds (label=0) in top k
    top_k_labels = y_true[top_k_indices]
    n_diamonds = (top_k_labels == 0).sum()
    precision = n_diamonds / n_top_k

    return precision, int(n_diamonds)


def evaluate_full(
    model: nn.Module,
    loader,
    device: torch.device,
    k: float = 0.20,
) -> Dict[str, float]:
    """
    Evaluate model with multiple metrics.

    Args:
        model: DCNv2 model
        loader: DataLoader yielding (num_vals, cat_vals, labels)
        device: torch device
        k: Fraction for Precision@k

    Returns:
        Dict with 'auc', 'precision_at_k', 'n_diamonds'

    # Purpose: Comprehensive evaluation for diamonds task
    # Why: Need both AUC and Precision@k to track model quality
    # Approved: 2026-02-16
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for num_vals, cat_vals, labels in loader:
            num_vals = {k: v.to(device) for k, v in num_vals.items()}
            cat_vals = {k: v.to(device) for k, v in cat_vals.items()}

            probs = model(num_vals, cat_vals)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    auc = roc_auc_score(y_true, y_pred)
    prec, n_diamonds = precision_at_k(y_true, y_pred, k)

    return {
        'auc': auc,
        'precision_at_k': prec,
        'n_diamonds': n_diamonds,
    }


# Keep old evaluate for backwards compatibility
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    """Evaluate model and compute AUC-ROC (backwards compatible)."""
    return evaluate_full(model, loader, device)['auc']


def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train for one epoch.

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for num_vals, cat_vals, labels in loader:
        num_vals = {k: v.to(device) for k, v in num_vals.items()}
        cat_vals = {k: v.to(device) for k, v in cat_vals.items()}
        labels = labels.to(device)

        # Forward pass - extract logits
        x0 = model.embedding(num_vals, cat_vals)
        cross_out = model.cross_network(x0)
        deep_out = model.deep_network(x0)
        combined = torch.cat([cross_out, deep_out], dim=1)
        logits = model.head(combined)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train_dcn_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    n_epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    loss_type: str = 'focal',
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: Optional[float] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Complete training loop with early stopping.

    Uses Precision@20% as the primary metric for early stopping (diamond recovery).

    Args:
        model: DCNv2 model (already on device)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: torch device
        n_epochs: Maximum epochs
        patience: Early stopping patience
        lr: Learning rate
        weight_decay: L2 regularization
        loss_type: 'focal' or 'bce'
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
        pos_weight: Positive class weight (only for BCE)
        checkpoint_path: Path to save best model
        verbose: Print progress

    Returns:
        Dict with training history and best metrics
    """
    # Setup loss function
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        if verbose:
            print(f'Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})')
    else:
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            criterion = nn.BCEWithLogitsLoss()
        if verbose:
            print(f'Using BCE Loss (pos_weight={pos_weight})')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training state - track Precision@20% as primary metric
    best_val_prec = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_auc': [], 'train_prec': [],
        'val_auc': [], 'val_prec': [],
    }
    best_epoch = 0

    if verbose:
        print(f'\n{"Epoch":>5} | {"Loss":>7} | {"Train AUC":>9} | {"Train P@20":>10} | {"Val AUC":>9} | {"Val P@20":>10}')
        print('-' * 70)

    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate both sets
        train_metrics = evaluate_full(model, train_loader, device)
        val_metrics = evaluate_full(model, val_loader, device)

        # Log
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_metrics['auc'])
        history['train_prec'].append(train_metrics['precision_at_k'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_prec'].append(val_metrics['precision_at_k'])

        if verbose:
            print(f'{epoch+1:5d} | {train_loss:7.4f} | {train_metrics["auc"]:9.4f} | {train_metrics["precision_at_k"]:10.4f} | {val_metrics["auc"]:9.4f} | {val_metrics["precision_at_k"]:10.4f}', end='')

        # Checkpointing based on Precision@20% (diamond recovery)
        if val_metrics['precision_at_k'] > best_val_prec:
            best_val_prec = val_metrics['precision_at_k']
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            patience_counter = 0

            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_metrics['auc'],
                    'val_precision': val_metrics['precision_at_k'],
                }, checkpoint_path)
            if verbose:
                print(' *best*')
        else:
            if verbose:
                print()
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f'\nEarly stopping at epoch {epoch+1}')
                break

    if verbose:
        print(f'\nBest: Epoch {best_epoch+1} | Val AUC: {best_val_auc:.4f} | Val P@20%: {best_val_prec:.4f}')

    return {
        'history': history,
        'best_val_auc': best_val_auc,
        'best_val_prec': best_val_prec,
        'best_epoch': best_epoch + 1,
    }
