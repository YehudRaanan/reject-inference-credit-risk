"""
VIME-Wide Training Script
==========================

Route G-Wide: Train VIME with 512-dim encoder (no bottleneck).

Uses all 3 preprocessing fixes from Phase 5.6:
1. ClippedRobustScaler
2. Partial Reconstruction Loss
3. Target Encoding

Training pipeline:
1. Self-supervised pretraining (reconstruction + mask prediction)
2. Classification head training with Focal Loss (same as DCN)

Evaluation metrics (same as DCN):
- AUC-ROC
- Precision@20%
- Diamonds found

Created: 2026-02-17
"""

import logging
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

import config
from src.models.vime.vime_wide import VIMEWide
from src.models.vime.vime_utils import compute_mask_probabilities, vime_corruption
from src.models.vime.vime_trainer import masked_mse_loss

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Focal Loss (same as DCN) ===
class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification (same as DCN)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


# === Precision@k metric (same as DCN) ===
def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.20):
    """Compute Precision@k for diamond recovery."""
    n_samples = len(y_true)
    n_top_k = int(n_samples * k)
    sorted_indices = np.argsort(y_pred)
    top_k_indices = sorted_indices[:n_top_k]
    top_k_labels = y_true[top_k_indices]
    n_diamonds = (top_k_labels == 0).sum()
    precision = n_diamonds / n_top_k
    return precision, int(n_diamonds)


# === Classification Head (MLP) ===
class ClassificationHead(nn.Module):
    """MLP classification head for VIME-Wide embeddings."""

    def __init__(self, embedding_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        return self.fc3(x)  # Returns logits


# === Full Model (Encoder + Head) ===
class VIMEWideClassifier(nn.Module):
    """VIME-Wide encoder + classification head."""

    def __init__(self, encoder: VIMEWide, head: ClassificationHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        z = self.encoder.encode(x)
        return self.head(z)


# === Evaluation Function (same as DCN) ===
def evaluate_full(model, X, y, device, k: float = 0.20):
    """Evaluate model with AUC and Precision@k."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        # Process in chunks
        preds = []
        for i in range(0, len(X_t), 4096):
            logits = model(X_t[i:i + 4096])
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds.append(probs)
        y_pred = np.concatenate(preds)

    auc = roc_auc_score(y, y_pred)
    prec, n_diamonds = precision_at_k(y, y_pred, k)

    return {
        'auc': auc,
        'precision_at_k': prec,
        'n_diamonds': n_diamonds,
    }


# === Self-Supervised Pretraining ===
def pretrain_vime_wide(
    X_all_tensor: torch.Tensor,
    full_mask: torch.Tensor,
    p_mask: torch.Tensor,
    input_dim: int,
    embedding_dim: int = 512,
    hidden_dim: int = 512,
    dropout: float = 0.3,
    batch_size: int = 512,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    alpha: float = 1.0,
    use_partial_loss: bool = True,
):
    """Pretrain VIME-Wide encoder with self-supervised learning."""
    logger.info("\n" + "=" * 70)
    logger.info("Phase 1: Self-Supervised Pretraining")
    logger.info("=" * 70)

    # Create DataLoader
    if use_partial_loss:
        dataset = TensorDataset(X_all_tensor, full_mask)
    else:
        dataset = TensorDataset(X_all_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = VIMEWide(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Architecture: {input_dim} -> {hidden_dim} -> {embedding_dim} (latent)")
    logger.info(f"Parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_mask = nn.BCELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mask_loss = 0.0
        epoch_recon_loss = 0.0
        start_time = time.time()

        for batch in dataloader:
            if use_partial_loss:
                x_clean, missing_mask_batch = batch
                x_clean = x_clean.to(device)
                missing_mask_batch = missing_mask_batch.to(device)
            else:
                x_clean = batch[0].to(device)
                missing_mask_batch = None

            x_tilde, m = vime_corruption(x_clean, p_mask)
            m_hat, x_hat = model(x_tilde)

            loss_m = criterion_mask(m_hat, m)
            if use_partial_loss and missing_mask_batch is not None:
                loss_r = masked_mse_loss(x_hat, x_clean, missing_mask_batch)
            else:
                loss_r = F.mse_loss(x_hat, x_clean)

            loss = alpha * loss_m + loss_r

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mask_loss += loss_m.item()
            epoch_recon_loss += loss_r.item()

        avg_loss = epoch_loss / len(dataloader)
        avg_m = epoch_mask_loss / len(dataloader)
        avg_r = epoch_recon_loss / len(dataloader)
        elapsed = time.time() - start_time

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1:2d}/{epochs}] "
                f"Loss: {avg_loss:.4f} (Mask: {avg_m:.4f}, Recon: {avg_r:.4f}) "
                f"- {elapsed:.1f}s"
            )

    # Save encoder
    save_path = config.MODEL_DIR / "vime_wide_encoder.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Encoder saved to {save_path}")

    return model


# === Classification Training with Focal Loss ===
def train_classifier(
    encoder: VIMEWide,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    embedding_dim: int = 512,
    dropout: float = 0.3,
    batch_size: int = 1024,
    epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
):
    """Train classification head with Focal Loss (same as DCN)."""
    logger.info("\n" + "=" * 70)
    logger.info("Phase 2: Classification Training (Focal Loss)")
    logger.info("=" * 70)

    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    # Create classification head
    head = ClassificationHead(embedding_dim=embedding_dim, dropout=dropout)
    model = VIMEWideClassifier(encoder, head).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters (head only): {trainable_params:,}")

    # Setup
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    logger.info(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")

    optimizer = optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop with evaluation
    best_val_prec = 0.0
    best_val_auc = 0.0
    patience_counter = 0
    best_epoch = 0
    history = {'train_loss': [], 'train_auc': [], 'train_prec': [], 'val_auc': [], 'val_prec': []}

    logger.info(f"\n{'Epoch':>5} | {'Loss':>7} | {'Train AUC':>9} | {'Train P@20':>10} | {'Val AUC':>9} | {'Val P@20':>10}")
    logger.info("-" * 70)

    for epoch in range(epochs):
        # Train
        model.train()
        # Only head is trainable (encoder frozen)
        head.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        train_metrics = evaluate_full(model, X_train, y_train, device)
        val_metrics = evaluate_full(model, X_val, y_val, device)

        history['train_loss'].append(avg_loss)
        history['train_auc'].append(train_metrics['auc'])
        history['train_prec'].append(train_metrics['precision_at_k'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_prec'].append(val_metrics['precision_at_k'])

        # Early stopping based on Precision@20%
        if val_metrics['precision_at_k'] > best_val_prec:
            best_val_prec = val_metrics['precision_at_k']
            best_val_auc = val_metrics['auc']
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model
            save_path = config.MODEL_DIR / "vime_wide_classifier.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'head_state_dict': head.state_dict(),
                'val_auc': val_metrics['auc'],
                'val_precision': val_metrics['precision_at_k'],
            }, save_path)
            marker = " *best*"
        else:
            marker = ""
            patience_counter += 1

        # Log
        logger.info(
            f"{epoch+1:5d} | {avg_loss:7.4f} | {train_metrics['auc']:9.4f} | "
            f"{train_metrics['precision_at_k']:10.4f} | {val_metrics['auc']:9.4f} | "
            f"{val_metrics['precision_at_k']:10.4f}{marker}"
        )
        if patience_counter >= patience:
            logger.info(f"\nEarly stopping at epoch {epoch+1}")
            break

    logger.info(f"\nBest: Epoch {best_epoch} | Val AUC: {best_val_auc:.4f} | Val P@20%: {best_val_prec:.4f}")

    return model, {
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'best_val_prec': best_val_prec,
        'history': history,
    }


# === Main Training Pipeline ===
def train_vime_wide_full(
    # Self-supervised params
    ssl_epochs: int = 30,
    ssl_batch_size: int = 512,
    ssl_lr: float = 1e-3,
    # Architecture params
    embedding_dim: int = 512,
    hidden_dim: int = 512,
    dropout: float = 0.3,
    # Classification params
    cls_epochs: int = 100,
    cls_batch_size: int = 1024,
    cls_lr: float = 1e-3,
    weight_decay: float = 1e-3,
    patience: int = 10,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
):
    """Full VIME-Wide training pipeline: SSL pretraining + classification."""
    logger.info("=" * 70)
    logger.info("VIME-Wide Full Training Pipeline (Route G-Wide)")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Architecture: 253 -> {hidden_dim} -> {embedding_dim} (no bottleneck)")

    # 1. Load Data
    logger.info("\nLoading data...")
    X_train = pd.read_parquet(config.DATA_PROCESSED / "vime_X_train_fixed.parquet")
    X_val = pd.read_parquet(config.DATA_PROCESSED / "vime_X_val_fixed.parquet")
    X_test = pd.read_parquet(config.DATA_PROCESSED / "vime_X_test_fixed.parquet")
    X_reject = pd.read_parquet(config.DATA_PROCESSED / "vime_X_reject_fixed.parquet")

    y_train = pd.read_parquet(config.DATA_PROCESSED / "vime_y_train.parquet").values.ravel()
    y_val = pd.read_parquet(config.DATA_PROCESSED / "vime_y_val.parquet").values.ravel()
    y_test = pd.read_parquet(config.DATA_PROCESSED / "vime_y_test.parquet").values.ravel()

    # Load D_R ground truth
    dr_truth = pd.read_parquet(config.SPLIT_DR_TRUTH)
    y_reject = dr_truth[config.TARGET_COL].values

    # Load missing masks
    mask_train = pd.read_parquet(config.DATA_PROCESSED / "vime_missing_mask_train.parquet")
    mask_val = pd.read_parquet(config.DATA_PROCESSED / "vime_missing_mask_val.parquet")
    mask_test = pd.read_parquet(config.DATA_PROCESSED / "vime_missing_mask_test.parquet")
    mask_reject = pd.read_parquet(config.DATA_PROCESSED / "vime_missing_mask_reject.parquet")

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Reject: {X_reject.shape}")

    # Concatenate for SSL
    X_all_df = pd.concat([X_train, X_val, X_test, X_reject], axis=0, ignore_index=True)
    mask_all_df = pd.concat([mask_train, mask_val, mask_test, mask_reject], axis=0, ignore_index=True)

    X_all_tensor = torch.tensor(X_all_df.values, dtype=torch.float32)

    # Create missing mask tensor
    full_mask = torch.zeros_like(X_all_tensor)
    feature_names = X_all_df.columns.tolist()
    for i, mask_col in enumerate(mask_all_df.columns):
        original_col = mask_col.replace("missing_", "")
        if original_col in feature_names:
            feat_idx = feature_names.index(original_col)
            full_mask[:, feat_idx] = torch.tensor(mask_all_df.iloc[:, i].values, dtype=torch.float32)

    # Compute mask probabilities
    from src.models.vime.column_manager import identify_feature_groups
    feature_groups = identify_feature_groups(
        columns=X_all_df.columns.tolist(),
        high_missing_rate_threshold=0.5,
        df_for_stats=X_all_df
    )
    p_mask = compute_mask_probabilities(
        X_all_df,
        critical_cols=feature_groups['critical'],
        high_missing_cols=feature_groups['high_missing'],
    ).to(device)

    input_dim = X_all_tensor.shape[1]

    # 2. Self-Supervised Pretraining
    encoder = pretrain_vime_wide(
        X_all_tensor=X_all_tensor,
        full_mask=full_mask,
        p_mask=p_mask,
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        batch_size=ssl_batch_size,
        epochs=ssl_epochs,
        learning_rate=ssl_lr,
    )

    # 3. Classification Training
    model, cls_results = train_classifier(
        encoder=encoder,
        X_train=X_train.values,
        y_train=y_train,
        X_val=X_val.values,
        y_val=y_val,
        embedding_dim=embedding_dim,
        dropout=dropout,
        batch_size=cls_batch_size,
        epochs=cls_epochs,
        patience=patience,
        learning_rate=cls_lr,
        weight_decay=weight_decay,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )

    # 4. Load best model and evaluate
    logger.info("\n" + "=" * 70)
    logger.info("Final Evaluation")
    logger.info("=" * 70)

    ckpt = torch.load(config.MODEL_DIR / "vime_wide_classifier.pth", weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_metrics = evaluate_full(model, X_test.values, y_test, device)
    reject_metrics = evaluate_full(model, X_reject.values, y_reject, device)

    logger.info(f"\n=== Test Set (D_L) Results ===")
    logger.info(f"AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Precision@20%: {test_metrics['precision_at_k']:.4f}")
    logger.info(f"Diamonds Found: {test_metrics['n_diamonds']:,}")

    logger.info(f"\n=== Reject Set (D_R) Results ===")
    logger.info(f"AUC: {reject_metrics['auc']:.4f}")
    logger.info(f"Precision@20%: {reject_metrics['precision_at_k']:.4f}")
    logger.info(f"Diamonds Found: {reject_metrics['n_diamonds']:,}")

    logger.info(f"\n--- Comparison ---")
    logger.info(f"Route B:        D_L AUC=0.7391, D_R AUC=0.7206, D_R P@20%=0.965")
    logger.info(f"A-Fixed (128):  D_L AUC=0.7166, D_R AUC=0.6873")
    logger.info(f"DCN-v2:         D_L AUC=0.7236, D_R AUC=0.6966, D_R P@20%=0.9567")

    # 5. Save results
    results = {
        'route': 'G-Wide (VIME-512)',
        'architecture': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'embedding_dim': embedding_dim,
            'dropout': dropout,
        },
        'training': {
            'ssl_epochs': ssl_epochs,
            'cls_epochs': cls_results['best_epoch'],
            'focal_alpha': focal_alpha,
            'focal_gamma': focal_gamma,
        },
        'test_metrics': {
            'auc': test_metrics['auc'],
            'precision_at_k': test_metrics['precision_at_k'],
            'n_diamonds': test_metrics['n_diamonds'],
        },
        'reject_metrics': {
            'auc': reject_metrics['auc'],
            'precision_at_k': reject_metrics['precision_at_k'],
            'n_diamonds': reject_metrics['n_diamonds'],
        },
        'best_val_auc': cls_results['best_val_auc'],
        'best_val_prec': cls_results['best_val_prec'],
    }

    results_path = config.RESULT_DIR / "vime_wide_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    return model, results


if __name__ == "__main__":
    train_vime_wide_full()
