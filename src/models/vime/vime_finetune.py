"""
End-to-end fine-tuning of VIME encoder + classification head.

Phase 5.7 — Dynamic Embeddings.

Instead of using frozen VIME embeddings fed to CatBoost, this script
fine-tunes the VIME encoder jointly with a classification head, allowing
the encoder to adapt its representations for the downstream classification
task (predicting loan defaults).

Architecture:
    Input (253 fixed features)
    → VIME Encoder (pre-trained, UNFROZEN) → 128-dim latent
    → ClassificationHead (128→64→32→1) → P(default)

Design decisions (approved by owner 2026-02-15):
    - Two hidden layers in head (128→64→32→1)
    - ReLU activation, Dropout=0.3
    - Same LR for encoder and head
    - Early stopping with patience=10
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

import config
from src.models.vime.vime_model import VIME
from src.models.vime.classification_head import ClassificationHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Wrapper Model ────────────────────────────────────────────
class VIMEClassifier(nn.Module):
    """End-to-end VIME encoder + classification head.

    Combines the pre-trained VIME encoder with a new classification
    head. During fine-tuning, gradients flow through BOTH components,
    allowing the encoder to adapt its representations for classification.
    """

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder   # Pre-trained VIME encoder (will be unfrozen)
        self.head = head         # New ClassificationHead (128→64→32→1)

    def forward(self, x):
        z = self.encoder.encode(x)   # Raw features → 128-dim latent
        return self.head(z)          # 128-dim → P(default)


# ── Data Loading ─────────────────────────────────────────────
def load_data():
    """Load fixed-preprocessed features and labels for fine-tuning.

    Returns raw fixed features (253-dim), NOT frozen embeddings.
    The encoder processes them end-to-end during fine-tuning.
    """
    logger.info("Loading fixed-preprocessed data...")

    X_train = pd.read_parquet(config.DATA_PROCESSED / "vime_X_train_fixed.parquet").values
    X_val = pd.read_parquet(config.DATA_PROCESSED / "vime_X_val_fixed.parquet").values
    X_test = pd.read_parquet(config.DATA_PROCESSED / "vime_X_test_fixed.parquet").values
    X_reject = pd.read_parquet(config.DATA_PROCESSED / "vime_X_reject_fixed.parquet").values

    y_train = pd.read_parquet(config.DATA_PROCESSED / "vime_y_train.parquet").values.ravel()
    y_val = pd.read_parquet(config.DATA_PROCESSED / "vime_y_val.parquet").values.ravel()
    y_test = pd.read_parquet(config.DATA_PROCESSED / "vime_y_test.parquet").values.ravel()

    # Reject ground truth is stored separately (labels hidden during training)
    dr_truth = pd.read_parquet(config.SPLIT_DR_TRUTH)
    y_reject = dr_truth[config.TARGET_COL].values

    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Reject: {X_reject.shape}")
    logger.info(f"  Input dim: {X_train.shape[1]}, Default rate: {y_train.mean():.4f}")

    return X_train, y_train, X_val, y_val, X_test, y_test, X_reject, y_reject


# ── Training Loop ────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch. Returns average loss."""
    model.train()   # Enable dropout
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float().unsqueeze(1)  # Shape: (batch, 1)

        optimizer.zero_grad()               # Clear previous gradients
        y_pred = model(X_batch)             # Forward: raw → encoder → head → P(default)
        loss = criterion(y_pred, y_batch)   # BCE loss
        loss.backward()                     # Backprop through head AND encoder
        optimizer.step()                    # Update ALL weights

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, X, y, device):
    """Evaluate model on a dataset. Returns AUC-ROC."""
    model.eval()   # Disable dropout
    with torch.no_grad():   # No gradient computation (faster, saves memory)
        X_t = torch.FloatTensor(X).to(device)
        # Process in chunks to avoid OOM on large datasets
        chunk_size = 4096
        preds = []
        for i in range(0, len(X_t), chunk_size):
            chunk = X_t[i:i + chunk_size]
            pred = model(chunk).cpu().numpy().ravel()
            preds.append(pred)
        y_pred = np.concatenate(preds)
    return roc_auc_score(y, y_pred)


# ── Main Fine-Tuning Pipeline ───────────────────────────────
def finetune_vime(
    epochs=config.FINETUNE_EPOCHS,
    lr=config.FINETUNE_LR,
    batch_size=config.FINETUNE_BATCH_SIZE,
    patience=config.FINETUNE_PATIENCE,
    dropout=config.FINETUNE_HEAD_DROPOUT,
):
    """
    Fine-tune the pre-trained VIME encoder with a classification head.

    This is Step 2.2A: Supervised fine-tuning on D_L only.

    Args:
        epochs: Maximum training epochs
        lr: Learning rate (same for encoder and head)
        batch_size: Training batch size
        patience: Early stopping patience
        dropout: Classification head dropout rate
    """
    logger.info("=" * 70)
    logger.info("Phase 5.7: VIME End-to-End Fine-Tuning (Dynamic Embeddings)")
    logger.info("=" * 70)

    # ── 1. Load data ──────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test, X_reject, y_reject = load_data()
    input_dim = X_train.shape[1]

    # ── 2. Load pre-trained VIME encoder ──────────────────
    logger.info("\nLoading pre-trained VIME encoder...")
    encoder = VIME(input_dim=input_dim)
    encoder_path = config.MODEL_DIR / "vime_encoder_fixed.pth"
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    logger.info(f"  Loaded from: {encoder_path}")

    # ── 3. Create classification head ─────────────────────
    head = ClassificationHead(embedding_dim=config.VIME_EMBEDDING_DIM, dropout=dropout)
    logger.info(f"  Head: {config.VIME_EMBEDDING_DIM} → 64 → 32 → 1, dropout={dropout}")

    # ── 4. Combine into end-to-end model ──────────────────
    model = VIMEClassifier(encoder, head).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    logger.info(f"  Total parameters: {total_params:,} (encoder: {encoder_params:,}, head: {head_params:,})")

    # ── 5. Optimizer and loss ─────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    logger.info(f"  Optimizer: Adam, lr={lr}, same for all parameters")

    # ── 6. DataLoader ─────────────────────────────────────
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ── 7. Training with early stopping ───────────────────
    logger.info(f"\nTraining for up to {epochs} epochs (patience={patience})...\n")
    best_val_auc = 0
    patience_counter = 0
    best_epoch = 0
    history = {"train_loss": [], "val_auc": []}

    start_time = time.time()

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auc = evaluate(model, X_val, y_val, device)

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)

        improved = ""
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_DIR / "vime_finetuned.pth")
            improved = " ★ NEW BEST"
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or improved or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {train_loss:.5f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Best: {best_val_auc:.4f} (ep {best_epoch})"
                f"{improved}"
            )

        if patience_counter >= patience:
            logger.info(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    logger.info(f"\nTraining complete in {elapsed:.1f}s ({best_epoch} best epoch)")

    # ── 8. Load best model and evaluate ───────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Evaluating best model...")
    logger.info("=" * 70)

    model.load_state_dict(torch.load(config.MODEL_DIR / "vime_finetuned.pth", map_location=device))
    model.to(device)

    val_auc = evaluate(model, X_val, y_val, device)
    test_auc = evaluate(model, X_test, y_test, device)
    reject_auc = evaluate(model, X_reject, y_reject, device)

    logger.info(f"  Val AUC:    {val_auc:.4f}")
    logger.info(f"  Test AUC:   {test_auc:.4f}")
    logger.info(f"  D_R AUC:    {reject_auc:.4f}")

    # ── 9. Comparison with baselines ──────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Comparison with Baselines")
    logger.info("=" * 70)

    baselines = {
        "Route A (frozen emb + CatBoost)": 0.6912,
        "Route A-Fixed (frozen fixed emb + CatBoost)": 0.7021,
        "Route B (raw features + CatBoost)": 0.7391,
        "Route C-Fixed Teacher (raw + fixed emb)": 0.7362,
    }

    for name, baseline in baselines.items():
        delta = test_auc - baseline
        sign = "+" if delta >= 0 else ""
        logger.info(f"  vs {name}: {sign}{delta:.4f}")

    # ── 10. Save results ──────────────────────────────────
    results = {
        "route": "Dynamic (Fine-tuned)",
        "step": "2.2A",
        "description": "End-to-end VIME fine-tuning (supervised on D_L only)",
        "val_auc": val_auc,
        "test_auc": test_auc,
        "reject_auc": reject_auc,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "lr": lr,
        "batch_size": batch_size,
        "dropout": dropout,
        "total_params": total_params,
        "encoder_params": encoder_params,
        "head_params": head_params,
        "baselines": baselines,
        "training_history": history,
    }

    results_path = config.RESULT_DIR / "finetune_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    return results


# ── Step 2.2B: Fine-Tune with Pseudo-Labels ─────────────────
def finetune_with_pseudo_labels(
    epochs=config.FINETUNE_EPOCHS,
    lr=config.FINETUNE_LR,
    batch_size=config.FINETUNE_BATCH_SIZE,
    patience=config.FINETUNE_PATIENCE,
    dropout=config.FINETUNE_HEAD_DROPOUT,
):
    """
    Step 2.2B: Fine-tune on D_L + pseudo-labeled D_R.

    Uses the Teacher model (from 2.2A) to generate pseudo-labels for D_R,
    then retrains a fresh encoder+head on the combined dataset.
    """
    logger.info("=" * 70)
    logger.info("Step 2.2B: Fine-Tuning with Pseudo-Labels (D_L + D_R)")
    logger.info("=" * 70)

    # ── 1. Load data ──────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test, X_reject, y_reject = load_data()
    input_dim = X_train.shape[1]

    # ── 2. Load Teacher model (from 2.2A) to generate pseudo-labels
    logger.info("\nLoading Teacher model (2.2A best) for pseudo-labeling...")
    teacher_encoder = VIME(input_dim=input_dim)
    teacher_head = ClassificationHead(embedding_dim=config.VIME_EMBEDDING_DIM, dropout=dropout)
    teacher = VIMEClassifier(teacher_encoder, teacher_head).to(device)
    teacher.load_state_dict(torch.load(config.MODEL_DIR / "vime_finetuned.pth", map_location=device))
    teacher.eval()

    # ── 3. Generate pseudo-labels for D_R ─────────────────
    logger.info("Generating pseudo-labels for D_R...")
    with torch.no_grad():
        X_r_t = torch.FloatTensor(X_reject).to(device)
        preds = []
        for i in range(0, len(X_r_t), 4096):
            pred = teacher(X_r_t[i:i + 4096]).cpu().numpy().ravel()
            preds.append(pred)
        pseudo_probs = np.concatenate(preds)

    pseudo_labels = (pseudo_probs >= config.PSEUDO_LABEL_THRESHOLD).astype(float)
    pseudo_default_rate = pseudo_labels.mean()
    logger.info(f"  Pseudo-label default rate: {pseudo_default_rate:.4f}")
    logger.info(f"  D_R size: {len(pseudo_labels)}, Pseudo defaults: {int(pseudo_labels.sum())}")

    # ── 4. Combine D_L + pseudo-labeled D_R ───────────────
    X_combined = np.concatenate([X_train, X_reject], axis=0)
    y_combined = np.concatenate([y_train, pseudo_labels], axis=0)
    logger.info(f"  Combined training set: {X_combined.shape[0]} samples")

    # ── 5. Train fresh model on combined data ─────────────
    logger.info("\nTraining Student model on combined data...")
    encoder = VIME(input_dim=input_dim)
    encoder.load_state_dict(torch.load(config.MODEL_DIR / "vime_encoder_fixed.pth", map_location=device))
    head = ClassificationHead(embedding_dim=config.VIME_EMBEDDING_DIM, dropout=dropout)
    model = VIMEClassifier(encoder, head).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_dataset = TensorDataset(torch.FloatTensor(X_combined), torch.FloatTensor(y_combined))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_auc = 0
    patience_counter = 0
    best_epoch = 0

    start_time = time.time()
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auc = evaluate(model, X_val, y_val, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_DIR / "vime_finetuned_pseudo.pth")
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.5f} | Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f} (ep {best_epoch})")

        if patience_counter >= patience:
            logger.info(f"\n  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start_time
    logger.info(f"\nTraining complete in {elapsed:.1f}s ({best_epoch} best epoch)")

    # ── 6. Evaluate best Student ──────────────────────────
    model.load_state_dict(torch.load(config.MODEL_DIR / "vime_finetuned_pseudo.pth", map_location=device))
    model.to(device)

    val_auc = evaluate(model, X_val, y_val, device)
    test_auc = evaluate(model, X_test, y_test, device)
    reject_auc = evaluate(model, X_reject, y_reject, device)

    logger.info(f"\n  Student Val AUC:  {val_auc:.4f}")
    logger.info(f"  Student Test AUC: {test_auc:.4f}")
    logger.info(f"  Student D_R AUC:  {reject_auc:.4f}")

    results = {
        "route": "Dynamic (Fine-tuned + Pseudo-Labels)",
        "step": "2.2B",
        "description": "Fine-tuned VIME with pseudo-labeled D_R (Student model)",
        "val_auc": val_auc,
        "test_auc": test_auc,
        "reject_auc": reject_auc,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "pseudo_default_rate": pseudo_default_rate,
        "combined_train_size": len(y_combined),
    }

    results_path = config.RESULT_DIR / "finetune_pseudo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    return results


# ── Step 2.2C: Frozen Encoder Ablation ───────────────────────
def finetune_frozen_ablation(
    epochs=config.FINETUNE_EPOCHS,
    lr=config.FINETUNE_LR,
    batch_size=config.FINETUNE_BATCH_SIZE,
    patience=config.FINETUNE_PATIENCE,
    dropout=config.FINETUNE_HEAD_DROPOUT,
):
    """
    Step 2.2C: Ablation — freeze encoder, train ONLY the classification head.

    Same architecture as 2.2A, but the encoder weights are frozen.
    This isolates the contribution of end-to-end fine-tuning:
    - If frozen << dynamic: fine-tuning genuinely helps the encoder adapt
    - If frozen ≈ dynamic: the head alone is sufficient, encoder changes don't matter
    """
    logger.info("=" * 70)
    logger.info("Step 2.2C: Frozen Encoder Ablation")
    logger.info("=" * 70)

    # ── 1. Load data ──────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test, X_reject, y_reject = load_data()
    input_dim = X_train.shape[1]

    # ── 2. Load encoder and FREEZE it ─────────────────────
    logger.info("\nLoading pre-trained VIME encoder (FROZEN)...")
    encoder = VIME(input_dim=input_dim)
    encoder.load_state_dict(torch.load(config.MODEL_DIR / "vime_encoder_fixed.pth", map_location=device))

    # FREEZE all encoder parameters — no gradients will flow through
    for param in encoder.parameters():
        param.requires_grad = False

    frozen_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"  Frozen encoder parameters: {frozen_params:,} (all requires_grad=False)")

    # ── 3. Create trainable head only ─────────────────────
    head = ClassificationHead(embedding_dim=config.VIME_EMBEDDING_DIM, dropout=dropout)
    model = VIMEClassifier(encoder, head).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable parameters: {trainable_params:,} (head only)")

    # Only optimize head parameters (encoder is frozen)
    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ── 4. Train ──────────────────────────────────────────
    logger.info(f"\nTraining (head only) for up to {epochs} epochs...\n")
    best_val_auc = 0
    patience_counter = 0
    best_epoch = 0

    start_time = time.time()
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auc = evaluate(model, X_val, y_val, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_DIR / "vime_frozen_head.pth")
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.5f} | Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f} (ep {best_epoch})")

        if patience_counter >= patience:
            logger.info(f"\n  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start_time
    logger.info(f"\nTraining complete in {elapsed:.1f}s ({best_epoch} best epoch)")

    # ── 5. Evaluate ───────────────────────────────────────
    model.load_state_dict(torch.load(config.MODEL_DIR / "vime_frozen_head.pth", map_location=device))
    model.to(device)

    val_auc = evaluate(model, X_val, y_val, device)
    test_auc = evaluate(model, X_test, y_test, device)
    reject_auc = evaluate(model, X_reject, y_reject, device)

    logger.info(f"\n  Frozen Val AUC:  {val_auc:.4f}")
    logger.info(f"  Frozen Test AUC: {test_auc:.4f}")
    logger.info(f"  Frozen D_R AUC:  {reject_auc:.4f}")

    results = {
        "route": "Frozen Encoder + MLP Head",
        "step": "2.2C",
        "description": "Frozen VIME encoder + classification head (ablation)",
        "val_auc": val_auc,
        "test_auc": test_auc,
        "reject_auc": reject_auc,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }

    results_path = config.RESULT_DIR / "finetune_frozen_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        step = sys.argv[1].upper()
        if step == "2.2B":
            finetune_with_pseudo_labels()
        elif step == "2.2C":
            finetune_frozen_ablation()
        elif step == "ALL":
            finetune_vime()
            finetune_with_pseudo_labels()
            finetune_frozen_ablation()
        else:
            finetune_vime()
    else:
        finetune_vime()
