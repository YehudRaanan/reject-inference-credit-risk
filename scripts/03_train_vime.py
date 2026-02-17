"""
A-Fixed (128-dim) with SAME training setup as G-Wide for fair comparison.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

import config
from models.vime.vime_model import VIME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cpu')

print('=' * 70)
print('A-Fixed (128-dim) with SAME training setup as G-Wide')
print('=' * 70)

# Focal Loss (same as G-Wide)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (alpha_weight * focal_weight * bce).mean()

def precision_at_k(y_true, y_pred, k=0.20):
    n_top_k = int(len(y_true) * k)
    sorted_indices = np.argsort(y_pred)
    top_k_labels = y_true[sorted_indices[:n_top_k]]
    n_diamonds = (top_k_labels == 0).sum()
    return n_diamonds / n_top_k, int(n_diamonds)

# Classification head (same structure, 128-dim input)
class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim=128, dropout=0.3):
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
        return self.fc3(x)

class VIMEClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        z = self.encoder.encode(x)
        return self.head(z)

def evaluate_full(model, X, y, device, k=0.20):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        preds = []
        for i in range(0, len(X_t), 4096):
            logits = model(X_t[i:i + 4096])
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds.append(probs)
        y_pred = np.concatenate(preds)
    auc = roc_auc_score(y, y_pred)
    prec, n_diamonds = precision_at_k(y, y_pred, k)
    return {'auc': auc, 'precision_at_k': prec, 'n_diamonds': n_diamonds}

# Load data
logger.info('Loading data...')
X_train = pd.read_parquet(config.DATA_PROCESSED / 'vime_X_train_fixed.parquet').values
X_val = pd.read_parquet(config.DATA_PROCESSED / 'vime_X_val_fixed.parquet').values
X_test = pd.read_parquet(config.DATA_PROCESSED / 'vime_X_test_fixed.parquet').values
X_reject = pd.read_parquet(config.DATA_PROCESSED / 'vime_X_reject_fixed.parquet').values

y_train = pd.read_parquet(config.DATA_PROCESSED / 'vime_y_train.parquet').values.ravel()
y_val = pd.read_parquet(config.DATA_PROCESSED / 'vime_y_val.parquet').values.ravel()
y_test = pd.read_parquet(config.DATA_PROCESSED / 'vime_y_test.parquet').values.ravel()
y_reject = pd.read_parquet(config.SPLIT_DR_TRUTH)[config.TARGET_COL].values

logger.info(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Reject: {X_reject.shape}')

# Load pre-trained A-Fixed encoder (128-dim)
logger.info('Loading pre-trained A-Fixed encoder (128-dim)...')
input_dim = X_train.shape[1]
encoder = VIME(input_dim=input_dim, embedding_dim=128, hidden_dim=256)
encoder.load_state_dict(torch.load(config.MODEL_DIR / 'vime_encoder_fixed.pth', map_location=device))
encoder.to(device)

# Freeze encoder
for param in encoder.parameters():
    param.requires_grad = False
encoder.eval()

# Create classification head
head = ClassificationHead(embedding_dim=128, dropout=0.3)
model = VIMEClassifier(encoder, head).to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'Architecture: {input_dim} -> 256 -> 256 -> 128 (latent)')
logger.info(f'Trainable parameters (head only): {trainable_params:,}')

# Setup training (SAME as G-Wide)
criterion = FocalLoss(alpha=0.25, gamma=2.0)
logger.info('Using Focal Loss (alpha=0.25, gamma=2.0)')
optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# Training loop
best_val_prec = 0.0
best_val_auc = 0.0
patience_counter = 0
best_epoch = 0
patience = 10
epochs = 100

print()
print(f'{"Epoch":>5} | {"Loss":>7} | {"Train AUC":>9} | {"Train P@20":>10} | {"Val AUC":>9} | {"Val P@20":>10}')
print('-' * 70)

for epoch in range(epochs):
    model.train()
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
    train_metrics = evaluate_full(model, X_train, y_train, device)
    val_metrics = evaluate_full(model, X_val, y_val, device)

    if val_metrics['precision_at_k'] > best_val_prec:
        best_val_prec = val_metrics['precision_at_k']
        best_val_auc = val_metrics['auc']
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'head_state_dict': head.state_dict(),
            'val_auc': val_metrics['auc'],
            'val_precision': val_metrics['precision_at_k'],
        }, config.MODEL_DIR / 'vime_128_classifier.pth')
        marker = ' *best*'
    else:
        marker = ''
        patience_counter += 1

    print(f'{epoch+1:5d} | {avg_loss:7.4f} | {train_metrics["auc"]:9.4f} | {train_metrics["precision_at_k"]:10.4f} | {val_metrics["auc"]:9.4f} | {val_metrics["precision_at_k"]:10.4f}{marker}')

    if patience_counter >= patience:
        print(f'\nEarly stopping at epoch {epoch+1}')
        break

print(f'\nBest: Epoch {best_epoch} | Val AUC: {best_val_auc:.4f} | Val P@20%: {best_val_prec:.4f}')

# Load best and evaluate
print('\n' + '=' * 70)
print('Final Evaluation')
print('=' * 70)

ckpt = torch.load(config.MODEL_DIR / 'vime_128_classifier.pth', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])

test_metrics = evaluate_full(model, X_test, y_test, device)
reject_metrics = evaluate_full(model, X_reject, y_reject, device)

print(f'\n=== Test Set (D_L) Results ===')
print(f'AUC: {test_metrics["auc"]:.4f}')
print(f'Precision@20%: {test_metrics["precision_at_k"]:.4f}')
print(f'Diamonds Found: {test_metrics["n_diamonds"]:,}')

print(f'\n=== Reject Set (D_R) Results ===')
print(f'AUC: {reject_metrics["auc"]:.4f}')
print(f'Precision@20%: {reject_metrics["precision_at_k"]:.4f}')
print(f'Diamonds Found: {reject_metrics["n_diamonds"]:,}')

print(f'\n--- Fair Comparison (Same Training Setup) ---')
print(f'A-Fixed (128):  D_L AUC={test_metrics["auc"]:.4f}, D_R AUC={reject_metrics["auc"]:.4f}, D_R P@20%={reject_metrics["precision_at_k"]:.4f}')
print(f'G-Wide  (512):  D_L AUC=0.7030, D_R AUC=0.6807, D_R P@20%=0.9551')
print(f'Route B:        D_L AUC=0.7391, D_R AUC=0.7206, D_R P@20%=0.9650')
print(f'DCN-v2:         D_L AUC=0.7236, D_R AUC=0.6966, D_R P@20%=0.9567')

# Save results
results = {
    'route': 'A-Fixed (VIME-128) - Same setup as G-Wide',
    'test_metrics': test_metrics,
    'reject_metrics': reject_metrics,
    'best_val_auc': best_val_auc,
    'best_val_prec': best_val_prec,
    'best_epoch': best_epoch,
}
with open(config.RESULT_DIR / 'vime_128_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved.')
