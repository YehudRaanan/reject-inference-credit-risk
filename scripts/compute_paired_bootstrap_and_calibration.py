"""
Paired Bootstrap Comparison + Calibration Metrics

This script computes:
1. Paired bootstrap CIs for DELTA metrics (proper model comparison)
2. Calibration metrics: Brier score, ECE, reliability diagrams

Paired bootstrap: Uses SAME indices for all models in each iteration,
computes delta within iteration, then reports CI for the delta.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from catboost import CatBoostClassifier
import torch
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import config


def precision_at_k(y_true, y_pred, k=0.20):
    """Compute precision at top k% (lowest risk = lowest predicted probability)."""
    n_top_k = int(len(y_true) * k)
    sorted_indices = np.argsort(y_pred)
    top_k_indices = sorted_indices[:n_top_k]
    top_k_labels = y_true[top_k_indices]
    n_diamonds = (top_k_labels == 0).sum()
    precision = n_diamonds / n_top_k
    return precision, int(n_diamonds)


def expected_calibration_error(y_true, y_pred, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum over bins of: (bin_size / total) * |accuracy - confidence|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find predictions in this bin
        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Average confidence in this bin
            avg_confidence = y_pred[in_bin].mean()
            # Actual accuracy (fraction of positives)
            avg_accuracy = y_true[in_bin].mean()
            # Weighted absolute difference
            ece += prop_in_bin * np.abs(avg_accuracy - avg_confidence)

    return ece


def compute_calibration_metrics(y_true, y_pred, model_name):
    """Compute all calibration metrics for a model."""
    brier = brier_score_loss(y_true, y_pred)
    ece = expected_calibration_error(y_true, y_pred, n_bins=10)

    # Calibration curve for reliability diagram
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')

    return {
        'brier_score': float(brier),
        'ece': float(ece),
        'calibration_curve': {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }
    }


def paired_bootstrap_comparison(y_true, predictions_dict, n_bootstrap=1000, seed=42):
    """
    Compute paired bootstrap CIs for delta metrics between models.

    Uses SAME indices for all models in each bootstrap iteration.
    Returns CI for DELTA, not separate CIs.
    """
    np.random.seed(seed)
    n_samples = len(y_true)

    model_names = list(predictions_dict.keys())

    # Store deltas for each pair
    results = {}

    # Also store raw metrics for each model
    model_metrics = {name: {'auc': [], 'p20': []} for name in model_names}

    print(f"\nRunning {n_bootstrap} paired bootstrap iterations...")
    for _ in tqdm(range(n_bootstrap)):
        # SAME indices for all models
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_true[indices]

        # Skip if only one class
        if len(np.unique(y_boot)) < 2:
            continue

        # Compute metrics for each model on SAME bootstrap sample
        boot_metrics = {}
        for name, pred in predictions_dict.items():
            pred_boot = pred[indices]
            auc = roc_auc_score(y_boot, pred_boot)
            prec, _ = precision_at_k(y_boot, pred_boot, k=0.20)
            boot_metrics[name] = {'auc': auc, 'p20': prec}
            model_metrics[name]['auc'].append(auc)
            model_metrics[name]['p20'].append(prec)

    # Compute deltas for key comparisons
    comparisons = [
        ('CatBoost', 'VIME-128'),
        ('CatBoost', 'VIME-512'),
        ('CatBoost', 'DCN-v2'),
        ('CatBoost', 'CatBoost-Ensemble'),
        ('VIME-128', 'VIME-512'),
        ('VIME-128-Fair-v2', 'VIME-512'),
        ('VIME-128-Fair-v2', 'VIME-128'),
        ('VIME-512-Fair-v3', 'VIME-128-Fair-v2'),
        ('VIME-512-Fair-v3', 'VIME-512'),
    ]

    for model_a, model_b in comparisons:
        if model_a in model_metrics and model_b in model_metrics:
            # Compute delta arrays
            auc_a = np.array(model_metrics[model_a]['auc'])
            auc_b = np.array(model_metrics[model_b]['auc'])
            p20_a = np.array(model_metrics[model_a]['p20'])
            p20_b = np.array(model_metrics[model_b]['p20'])

            delta_auc = auc_a - auc_b
            delta_p20 = p20_a - p20_b

            # CI for delta
            results[f'{model_a} vs {model_b}'] = {
                'delta_auc': {
                    'mean': float(np.mean(delta_auc)),
                    'ci_low': float(np.percentile(delta_auc, 2.5)),
                    'ci_high': float(np.percentile(delta_auc, 97.5)),
                    'std': float(np.std(delta_auc)),
                    'significant': bool(np.percentile(delta_auc, 2.5) > 0 or np.percentile(delta_auc, 97.5) < 0)
                },
                'delta_p20': {
                    'mean': float(np.mean(delta_p20)),
                    'ci_low': float(np.percentile(delta_p20, 2.5)),
                    'ci_high': float(np.percentile(delta_p20, 97.5)),
                    'std': float(np.std(delta_p20)),
                    'significant': bool(np.percentile(delta_p20, 2.5) > 0 or np.percentile(delta_p20, 97.5) < 0)
                }
            }

    # Also return individual model CIs (for reference)
    individual_cis = {}
    for name in model_names:
        if model_metrics[name]['auc']:
            aucs = np.array(model_metrics[name]['auc'])
            p20s = np.array(model_metrics[name]['p20'])
            individual_cis[name] = {
                'auc': {
                    'mean': float(np.mean(aucs)),
                    'ci_low': float(np.percentile(aucs, 2.5)),
                    'ci_high': float(np.percentile(aucs, 97.5))
                },
                'p20': {
                    'mean': float(np.mean(p20s)),
                    'ci_low': float(np.percentile(p20s, 2.5)),
                    'ci_high': float(np.percentile(p20s, 97.5))
                }
            }

    return results, individual_cis


def load_model_predictions(D_R_features, y_DR):
    """Load all model predictions on D_R."""
    predictions = {}

    # 1. CatBoost
    print("Loading CatBoost predictions...")
    model_path = config.MODEL_DIR / 'catboost_fair.cbm'
    if model_path.exists():
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        X_copy = D_R_features.copy()
        for col in X_copy.select_dtypes(include=['object', 'category']).columns:
            X_copy[col] = X_copy[col].fillna('_MISSING_').astype(str)
        predictions['CatBoost'] = model.predict_proba(X_copy)[:, 1]

    # 2. CatBoost-Ensemble
    print("Loading CatBoost-Ensemble predictions...")
    ext_path = config.MODEL_DIR / 'catboost_ensemble_ext_fair.cbm'
    other_path = config.MODEL_DIR / 'catboost_ensemble_other_fair.cbm'
    if ext_path.exists() and other_path.exists():
        ext_cols = [c for c in D_R_features.columns if 'EXT_SOURCE' in c or 'EXT_SOURCES' in c]
        other_cols = [c for c in D_R_features.columns if c not in ext_cols]

        X_ext = D_R_features[ext_cols].copy()
        X_other = D_R_features[other_cols].copy()
        for col in X_other.select_dtypes(include=['object', 'category']).columns:
            X_other[col] = X_other[col].fillna('_MISSING_').astype(str)

        model_ext = CatBoostClassifier()
        model_ext.load_model(str(ext_path))
        model_other = CatBoostClassifier()
        model_other.load_model(str(other_path))

        pred_ext = model_ext.predict_proba(X_ext)[:, 1]
        pred_other = model_other.predict_proba(X_other)[:, 1]
        predictions['CatBoost-Ensemble'] = 0.3 * pred_ext + 0.7 * pred_other

    # 3. VIME-128
    print("Loading VIME-128 predictions...")
    vime_X_path = config.DATA_PROCESSED / 'vime_X_reject_fixed.parquet'
    vime_encoder_path = config.MODEL_DIR / 'vime_encoder_fixed.pth'
    vime_classifier_path = config.MODEL_DIR / 'vime_128_classifier.pth'

    if vime_X_path.exists() and vime_encoder_path.exists() and vime_classifier_path.exists():
        try:
            from models.vime.vime_model import VIME
            import torch.nn.functional as F

            vime_X = pd.read_parquet(vime_X_path)
            X_tensor = torch.tensor(vime_X.values, dtype=torch.float32)

            input_dim = X_tensor.shape[1]
            vime_model = VIME(input_dim, embedding_dim=128, hidden_dim=256)
            vime_model.load_state_dict(torch.load(vime_encoder_path, map_location='cpu', weights_only=False))
            vime_model.eval()

            class ClassificationHead128(torch.nn.Module):
                def __init__(self, embedding_dim=128, dropout=0.3):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(embedding_dim, 256)
                    self.bn1 = torch.nn.BatchNorm1d(256)
                    self.drop1 = torch.nn.Dropout(dropout)
                    self.fc2 = torch.nn.Linear(256, 64)
                    self.bn2 = torch.nn.BatchNorm1d(64)
                    self.drop2 = torch.nn.Dropout(dropout)
                    self.fc3 = torch.nn.Linear(64, 1)

                def forward(self, x):
                    x = F.relu(self.bn1(self.fc1(x)))
                    x = self.drop1(x)
                    x = F.relu(self.bn2(self.fc2(x)))
                    x = self.drop2(x)
                    return self.fc3(x)

            classifier = ClassificationHead128(128)
            checkpoint = torch.load(vime_classifier_path, map_location='cpu', weights_only=False)
            if 'head_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['head_state_dict'])
            else:
                classifier.load_state_dict(checkpoint)
            classifier.eval()

            with torch.no_grad():
                embeddings = vime_model.encode(X_tensor)
                logits = classifier(embeddings)
                predictions['VIME-128'] = torch.sigmoid(logits).numpy().flatten()
        except Exception as e:
            print(f"  Error loading VIME-128: {e}")

    # 3b. VIME-128-Fair-v2 (re-trained with identical setup to G-Wide)
    print("Loading VIME-128-Fair-v2 predictions...")
    vime128v2_encoder_path = config.MODEL_DIR / 'vime_128_fair_v2_encoder.pth'
    vime128v2_classifier_path = config.MODEL_DIR / 'vime_128_fair_v2_classifier.pth'

    if vime128v2_encoder_path.exists() and vime128v2_classifier_path.exists() and vime_X_path.exists():
        try:
            from models.vime.vime_model import VIME
            import torch.nn.functional as F

            if 'vime_X' not in dir():
                vime_X = pd.read_parquet(vime_X_path)
                X_tensor = torch.tensor(vime_X.values, dtype=torch.float32)

            input_dim = X_tensor.shape[1]
            vime_model_v2 = VIME(input_dim, embedding_dim=128, hidden_dim=256, dropout=0.3)
            vime_model_v2.load_state_dict(torch.load(vime128v2_encoder_path, map_location='cpu', weights_only=False))
            vime_model_v2.eval()

            class ClassificationHead128v2(torch.nn.Module):
                def __init__(self, embedding_dim=128, dropout=0.3):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(embedding_dim, 256)
                    self.bn1 = torch.nn.BatchNorm1d(256)
                    self.drop1 = torch.nn.Dropout(dropout)
                    self.fc2 = torch.nn.Linear(256, 64)
                    self.bn2 = torch.nn.BatchNorm1d(64)
                    self.drop2 = torch.nn.Dropout(dropout)
                    self.fc3 = torch.nn.Linear(64, 1)

                def forward(self, x):
                    x = F.relu(self.bn1(self.fc1(x)))
                    x = self.drop1(x)
                    x = F.relu(self.bn2(self.fc2(x)))
                    x = self.drop2(x)
                    return self.fc3(x)

            class VIMEClassifier128v2(torch.nn.Module):
                def __init__(self, encoder, head):
                    super().__init__()
                    self.encoder = encoder
                    self.head = head

                def forward(self, x):
                    z = self.encoder.encode(x)
                    return self.head(z)

            head_v2 = ClassificationHead128v2(128)
            model_128v2 = VIMEClassifier128v2(vime_model_v2, head_v2)
            checkpoint_v2 = torch.load(vime128v2_classifier_path, map_location='cpu', weights_only=False)

            if 'model_state_dict' in checkpoint_v2:
                model_128v2.load_state_dict(checkpoint_v2['model_state_dict'])
                print("  Loaded VIME-128-Fair-v2 from model_state_dict")
            elif 'head_state_dict' in checkpoint_v2:
                head_v2.load_state_dict(checkpoint_v2['head_state_dict'])
                print("  Loaded VIME-128-Fair-v2 from head_state_dict")

            model_128v2.eval()

            with torch.no_grad():
                logits_v2 = model_128v2(X_tensor)
                predictions['VIME-128-Fair-v2'] = torch.sigmoid(logits_v2).numpy().flatten()
        except Exception as e:
            print(f"  Error loading VIME-128-Fair-v2: {e}")
            import traceback
            traceback.print_exc()

    # 3c. VIME-512-Fair-v3 (512-dim using SAME VIME class as 128, matched layers)
    print("Loading VIME-512-Fair-v3 predictions...")
    vime512v3_encoder_path = config.MODEL_DIR / 'vime_512_fair_v3_encoder.pth'
    vime512v3_classifier_path = config.MODEL_DIR / 'vime_512_fair_v3_classifier.pth'

    if vime512v3_encoder_path.exists() and vime512v3_classifier_path.exists() and vime_X_path.exists():
        try:
            from models.vime.vime_model import VIME
            import torch.nn.functional as F

            if 'vime_X' not in dir():
                vime_X = pd.read_parquet(vime_X_path)
                X_tensor = torch.tensor(vime_X.values, dtype=torch.float32)

            input_dim = X_tensor.shape[1]
            vime_model_v3 = VIME(input_dim, embedding_dim=512, hidden_dim=512, dropout=0.3)
            vime_model_v3.load_state_dict(torch.load(vime512v3_encoder_path, map_location='cpu', weights_only=False))
            vime_model_v3.eval()

            class ClassificationHead512v3(torch.nn.Module):
                def __init__(self, embedding_dim=512, dropout=0.3):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(embedding_dim, 256)
                    self.bn1 = torch.nn.BatchNorm1d(256)
                    self.drop1 = torch.nn.Dropout(dropout)
                    self.fc2 = torch.nn.Linear(256, 64)
                    self.bn2 = torch.nn.BatchNorm1d(64)
                    self.drop2 = torch.nn.Dropout(dropout)
                    self.fc3 = torch.nn.Linear(64, 1)

                def forward(self, x):
                    x = F.relu(self.bn1(self.fc1(x)))
                    x = self.drop1(x)
                    x = F.relu(self.bn2(self.fc2(x)))
                    x = self.drop2(x)
                    return self.fc3(x)

            class VIMEClassifier512v3(torch.nn.Module):
                def __init__(self, encoder, head):
                    super().__init__()
                    self.encoder = encoder
                    self.head = head

                def forward(self, x):
                    z = self.encoder.encode(x)
                    return self.head(z)

            head_v3 = ClassificationHead512v3(512)
            model_512v3 = VIMEClassifier512v3(vime_model_v3, head_v3)
            checkpoint_v3 = torch.load(vime512v3_classifier_path, map_location='cpu', weights_only=False)

            if 'model_state_dict' in checkpoint_v3:
                model_512v3.load_state_dict(checkpoint_v3['model_state_dict'])
                print("  Loaded VIME-512-Fair-v3 from model_state_dict")
            elif 'head_state_dict' in checkpoint_v3:
                head_v3.load_state_dict(checkpoint_v3['head_state_dict'])
                print("  Loaded VIME-512-Fair-v3 from head_state_dict")

            model_512v3.eval()

            with torch.no_grad():
                logits_v3 = model_512v3(X_tensor)
                predictions['VIME-512-Fair-v3'] = torch.sigmoid(logits_v3).numpy().flatten()
        except Exception as e:
            print(f"  Error loading VIME-512-Fair-v3: {e}")
            import traceback
            traceback.print_exc()

    # 4. VIME-512 - Load FULL fine-tuned model from model_state_dict
    print("Loading VIME-512 predictions...")
    vime512_classifier_path = config.MODEL_DIR / 'vime_wide_classifier.pth'

    if vime512_classifier_path.exists() and vime_X_path.exists():
        try:
            from models.vime.vime_wide import VIMEWide
            import torch.nn.functional as F

            if 'vime_X' not in dir():
                vime_X = pd.read_parquet(vime_X_path)
                X_tensor = torch.tensor(vime_X.values, dtype=torch.float32)

            input_dim = X_tensor.shape[1]

            # Define the full model class matching the training script
            class ClassificationHead512(torch.nn.Module):
                def __init__(self, embedding_dim=512, dropout=0.3):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(embedding_dim, 256)
                    self.bn1 = torch.nn.BatchNorm1d(256)
                    self.drop1 = torch.nn.Dropout(dropout)
                    self.fc2 = torch.nn.Linear(256, 64)
                    self.bn2 = torch.nn.BatchNorm1d(64)
                    self.drop2 = torch.nn.Dropout(dropout)
                    self.fc3 = torch.nn.Linear(64, 1)

                def forward(self, x):
                    x = F.relu(self.bn1(self.fc1(x)))
                    x = self.drop1(x)
                    x = F.relu(self.bn2(self.fc2(x)))
                    x = self.drop2(x)
                    return self.fc3(x)

            class VIMEClassifierWide(torch.nn.Module):
                def __init__(self, input_dim, hidden_dim=512, embedding_dim=512, dropout=0.3):
                    super().__init__()
                    self.encoder = VIMEWide(input_dim, hidden_dim, embedding_dim, dropout)
                    self.head = ClassificationHead512(embedding_dim, dropout)

                def forward(self, x):
                    z = self.encoder.encode(x)
                    return self.head(z)

            # Load full fine-tuned model from model_state_dict
            model_wide = VIMEClassifierWide(input_dim, hidden_dim=512, embedding_dim=512, dropout=0.3)
            checkpoint_wide = torch.load(vime512_classifier_path, map_location='cpu', weights_only=False)

            if 'model_state_dict' in checkpoint_wide:
                model_wide.load_state_dict(checkpoint_wide['model_state_dict'])
                print("  Loaded VIME-512 from model_state_dict (full fine-tuned model)")
            else:
                # Fallback to old method if model_state_dict not available
                model_wide.encoder.load_state_dict(torch.load(config.MODEL_DIR / 'vime_wide_encoder.pth', map_location='cpu', weights_only=False))
                model_wide.head.load_state_dict(checkpoint_wide.get('head_state_dict', checkpoint_wide))
                print("  Loaded VIME-512 from separate encoder + head (fallback)")

            model_wide.eval()

            with torch.no_grad():
                logits_wide = model_wide(X_tensor)
                predictions['VIME-512'] = torch.sigmoid(logits_wide).numpy().flatten()
        except Exception as e:
            print(f"  Error loading VIME-512: {e}")
            import traceback
            traceback.print_exc()

    # 5. DCN-v2 (load from saved predictions if available)
    print("Loading DCN-v2 predictions...")
    dcn_pred_path = config.RESULT_DIR / 'dcn_v2_predictions_DR.npy'
    if dcn_pred_path.exists():
        predictions['DCN-v2'] = np.load(dcn_pred_path)
    else:
        # Try to load model and predict
        dcn_model_path = config.MODEL_DIR / 'dcn_v2_cat_best.pth'
        if dcn_model_path.exists():
            try:
                from models.dcn_v2.dcn_v2 import DCNv2
                from models.dcn_v2.embedding_engine import EmbeddingEngine

                # Load DCN config
                dcn_config_path = config.RESULT_DIR / 'dcn_v2_config.json'
                if dcn_config_path.exists():
                    with open(dcn_config_path, 'r') as f:
                        dcn_config = json.load(f)
                    print("  DCN-v2 model loading not implemented - using saved results")
            except Exception as e:
                print(f"  Error loading DCN-v2: {e}")

    return predictions


def plot_reliability_diagrams(calibration_results, output_path):
    """Plot reliability diagrams for all models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (model_name, metrics) in enumerate(calibration_results.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]

        curve = metrics['calibration_curve']
        prob_true = np.array(curve['prob_true'])
        prob_pred = np.array(curve['prob_pred'])

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        # Model calibration
        ax.plot(prob_pred, prob_true, 's-', label=model_name)

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{model_name}\nBrier={metrics["brier_score"]:.4f}, ECE={metrics["ece"]:.4f}')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(calibration_results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reliability diagrams saved to {output_path}")


def main():
    print("=" * 70)
    print("PAIRED BOOTSTRAP COMPARISON + CALIBRATION METRICS")
    print("=" * 70)

    # Load D_R data
    print("\nLoading D_R data...")
    D_R_features = pd.read_parquet(config.SPLIT_DR_FEATURES)
    D_R_truth = pd.read_parquet(config.SPLIT_DR_TRUTH)
    y_DR = D_R_truth['TARGET'].values

    print(f"D_R size: {len(y_DR):,} samples")
    print(f"D_R default rate: {y_DR.mean():.2%}")

    # Load all model predictions
    predictions = load_model_predictions(D_R_features, y_DR)
    print(f"\nLoaded predictions for: {list(predictions.keys())}")

    # ==========================================
    # 1. PAIRED BOOTSTRAP COMPARISON
    # ==========================================
    print("\n" + "=" * 70)
    print("PAIRED BOOTSTRAP COMPARISON (n=1000)")
    print("=" * 70)

    paired_results, individual_cis = paired_bootstrap_comparison(
        y_DR, predictions, n_bootstrap=1000, seed=42
    )

    # Print results
    print("\n--- DELTA METRICS (Model A - Model B) ---")
    print(f"{'Comparison':<30} {'Delta-AUC [95% CI]':<35} {'Significant?':<12}")
    print("-" * 80)

    for comparison, metrics in paired_results.items():
        delta = metrics['delta_auc']
        ci_str = f"{delta['mean']:+.4f} [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]"
        sig = "YES" if delta['significant'] else "NO"
        print(f"{comparison:<30} {ci_str:<35} {sig:<12}")

    print("\n--- DELTA P@20% ---")
    print(f"{'Comparison':<30} {'Delta-P@20% [95% CI]':<35} {'Significant?':<12}")
    print("-" * 80)

    for comparison, metrics in paired_results.items():
        delta = metrics['delta_p20']
        ci_str = f"{delta['mean']:+.4f} [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]"
        sig = "YES" if delta['significant'] else "NO"
        print(f"{comparison:<30} {ci_str:<35} {sig:<12}")

    # ==========================================
    # 2. CALIBRATION METRICS
    # ==========================================
    print("\n" + "=" * 70)
    print("CALIBRATION METRICS")
    print("=" * 70)

    calibration_results = {}
    for model_name, pred in predictions.items():
        calibration_results[model_name] = compute_calibration_metrics(y_DR, pred, model_name)

    print(f"\n{'Model':<20} {'Brier Score':<15} {'ECE':<15} {'Interpretation'}")
    print("-" * 70)

    for model_name, metrics in calibration_results.items():
        brier = metrics['brier_score']
        ece = metrics['ece']
        # Interpretation
        if ece < 0.05:
            interp = "Well calibrated"
        elif ece < 0.10:
            interp = "Moderately calibrated"
        else:
            interp = "Poorly calibrated"
        print(f"{model_name:<20} {brier:<15.4f} {ece:<15.4f} {interp}")

    # ==========================================
    # 3. SAVE RESULTS
    # ==========================================
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Save paired bootstrap results
    all_results = {
        'paired_bootstrap': paired_results,
        'individual_cis': individual_cis,
        'calibration': calibration_results,
        'methodology': {
            'n_bootstrap': 1000,
            'seed': 42,
            'paired': True,
            'description': 'Paired bootstrap using same indices for all models in each iteration'
        }
    }

    output_path = config.RESULT_DIR / 'paired_bootstrap_and_calibration.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Plot reliability diagrams
    plot_path = config.FIGURE_DIR / 'reliability_diagrams.png'
    plot_reliability_diagrams(calibration_results, plot_path)

    # ==========================================
    # 4. SUMMARY FOR REPORT
    # ==========================================
    print("\n" + "=" * 70)
    print("SUMMARY FOR REPORT UPDATE")
    print("=" * 70)

    print("\n### Paired Bootstrap Results (ΔAUC):")
    for comparison, metrics in paired_results.items():
        delta = metrics['delta_auc']
        status = "SIGNIFICANT" if delta['significant'] else "NOT SIGNIFICANT"
        print(f"- {comparison}: ΔAUC = {delta['mean']:+.4f} [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] - {status}")

    print("\n### Calibration Summary:")
    for model_name, metrics in calibration_results.items():
        print(f"- {model_name}: Brier={metrics['brier_score']:.4f}, ECE={metrics['ece']:.4f}")


if __name__ == '__main__':
    main()
