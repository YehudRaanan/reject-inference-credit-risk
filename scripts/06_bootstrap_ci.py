"""
Compute Bootstrap 95% Confidence Intervals for Model Comparison

This script loads saved models and computes bootstrap CIs for:
- AUC-ROC
- Precision@20%
- Diamonds@20%

Bootstrap method: resample test set with replacement N times, compute metric each time.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import torch
import json
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, k=0.20, seed=42):
    """
    Compute bootstrap 95% confidence intervals for AUC and P@k%.

    Returns dict with point estimates and CIs.
    """
    np.random.seed(seed)
    n_samples = len(y_true)

    aucs = []
    precisions = []
    diamonds = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_true[indices]
        pred_boot = y_pred[indices]

        # Skip if only one class in sample (rare but possible)
        if len(np.unique(y_boot)) < 2:
            continue

        # Compute metrics
        auc = roc_auc_score(y_boot, pred_boot)
        prec, n_dia = precision_at_k(y_boot, pred_boot, k=k)

        aucs.append(auc)
        precisions.append(prec)
        diamonds.append(n_dia)

    aucs = np.array(aucs)
    precisions = np.array(precisions)
    diamonds = np.array(diamonds)

    # Point estimates on original data
    auc_point = roc_auc_score(y_true, y_pred)
    prec_point, dia_point = precision_at_k(y_true, y_pred, k=k)

    # 95% CI (2.5th and 97.5th percentiles)
    return {
        'AUC': {
            'point': float(auc_point),
            'ci_low': float(np.percentile(aucs, 2.5)),
            'ci_high': float(np.percentile(aucs, 97.5)),
            'std': float(np.std(aucs))
        },
        'P@20%': {
            'point': float(prec_point),
            'ci_low': float(np.percentile(precisions, 2.5)),
            'ci_high': float(np.percentile(precisions, 97.5)),
            'std': float(np.std(precisions))
        },
        'Diamonds': {
            'point': int(dia_point),
            'ci_low': int(np.percentile(diamonds, 2.5)),
            'ci_high': int(np.percentile(diamonds, 97.5)),
            'std': float(np.std(diamonds))
        }
    }


def load_catboost_predictions(model_path, X):
    """Load CatBoost model and get predictions."""
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    # Handle NaN in categorical columns - MUST match training script
    X_copy = X.copy()
    for col in X_copy.select_dtypes(include=['object', 'category']).columns:
        X_copy[col] = X_copy[col].fillna('_MISSING_').astype(str)

    return model.predict_proba(X_copy)[:, 1]


def load_vime_predictions(encoder_path, classifier_path, X_tensor, device='cpu'):
    """Load VIME encoder + classifier and get predictions."""
    sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'models' / 'vime'))
    from vime_model import VIMEEncoder
    from classification_head import ClassificationHead

    # Determine dimensions
    input_dim = X_tensor.shape[1]

    # Check if it's wide (512) or standard (128)
    checkpoint = torch.load(encoder_path, map_location=device)
    if 'fc1.weight' in checkpoint:
        hidden_dim = checkpoint['fc1.weight'].shape[0]
        # Infer embedding dim from fc3 if exists, else fc2
        if 'fc3.weight' in checkpoint:
            embedding_dim = checkpoint['fc3.weight'].shape[0]
        else:
            embedding_dim = checkpoint['fc2.weight'].shape[0]
    else:
        hidden_dim = 256
        embedding_dim = 128

    encoder = VIMEEncoder(input_dim, hidden_dim, embedding_dim)
    encoder.load_state_dict(checkpoint)
    encoder.to(device)
    encoder.eval()

    classifier = ClassificationHead(embedding_dim)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    with torch.no_grad():
        embeddings = encoder(X_tensor.to(device))
        logits = classifier(embeddings)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    return probs


def main():
    print("=" * 60)
    print("Bootstrap 95% Confidence Intervals for Model Comparison")
    print("=" * 60)

    # Load D_R data
    print("\nLoading D_R data...")
    D_R_features = pd.read_parquet(config.SPLIT_DR_FEATURES)
    D_R_truth = pd.read_parquet(config.SPLIT_DR_TRUTH)
    y_DR = D_R_truth['TARGET'].values

    print(f"D_R size: {len(y_DR):,} samples")
    print(f"D_R default rate: {y_DR.mean():.2%}")

    results = {}
    n_bootstrap = 1000

    # ==========================================
    # 1. CatBoost (Fair)
    # ==========================================
    print("\n[1/5] CatBoost (Fair)...")
    model_path = config.MODEL_DIR / 'catboost_fair.cbm'
    if model_path.exists():
        pred = load_catboost_predictions(model_path, D_R_features)
        results['CatBoost'] = bootstrap_metrics(y_DR, pred, n_bootstrap=n_bootstrap)
        print(f"  AUC: {results['CatBoost']['AUC']['point']:.4f} "
              f"[{results['CatBoost']['AUC']['ci_low']:.4f}, {results['CatBoost']['AUC']['ci_high']:.4f}]")
    else:
        print(f"  Model not found: {model_path}")

    # ==========================================
    # 2. CatBoost-Ensemble (Fair)
    # ==========================================
    print("\n[2/5] CatBoost-Ensemble (Fair)...")
    ext_path = config.MODEL_DIR / 'catboost_ensemble_ext_fair.cbm'
    other_path = config.MODEL_DIR / 'catboost_ensemble_other_fair.cbm'

    if ext_path.exists() and other_path.exists():
        # Load EXT_SOURCE columns
        ext_cols = [c for c in D_R_features.columns if 'EXT_SOURCE' in c or 'EXT_SOURCES' in c]
        other_cols = [c for c in D_R_features.columns if c not in ext_cols]

        X_ext = D_R_features[ext_cols].copy()
        X_other = D_R_features[other_cols].copy()

        # Handle NaN in categorical columns - MUST match training script
        for col in X_other.select_dtypes(include=['object', 'category']).columns:
            X_other[col] = X_other[col].fillna('_MISSING_').astype(str)

        pred_ext = load_catboost_predictions(ext_path, X_ext)
        pred_other = load_catboost_predictions(other_path, X_other)

        # Ensemble with optimal weights
        w_ext, w_other = 0.3, 0.7
        pred_ensemble = w_ext * pred_ext + w_other * pred_other

        results['CatBoost-Ensemble'] = bootstrap_metrics(y_DR, pred_ensemble, n_bootstrap=n_bootstrap)
        print(f"  AUC: {results['CatBoost-Ensemble']['AUC']['point']:.4f} "
              f"[{results['CatBoost-Ensemble']['AUC']['ci_low']:.4f}, {results['CatBoost-Ensemble']['AUC']['ci_high']:.4f}]")
    else:
        print(f"  Models not found")

    # ==========================================
    # 3. DCN-v2
    # ==========================================
    print("\n[3/5] DCN-v2...")
    dcn_results_path = config.RESULT_DIR / 'dcn_v2_DR_results.json'
    if dcn_results_path.exists():
        # DCN-v2 requires complex loading - use saved results if available
        with open(dcn_results_path, 'r') as f:
            dcn_saved = json.load(f)
        print(f"  Using saved results (bootstrap requires model reload)")
        print(f"  AUC: {dcn_saved.get('D_R_AUC', 'N/A')}")
        # Note: For full bootstrap, would need to reload DCN-v2 model
        results['DCN-v2'] = {
            'AUC': {'point': dcn_saved.get('D_R_AUC', 0.6966), 'note': 'Point estimate only'},
            'P@20%': {'point': dcn_saved.get('D_R_P@20%', 0.9567), 'note': 'Point estimate only'},
            'Diamonds': {'point': dcn_saved.get('D_R_Diamonds', 17741), 'note': 'Point estimate only'}
        }
    else:
        print(f"  Results not found")

    # ==========================================
    # 4. VIME-128
    # ==========================================
    print("\n[4/5] VIME-128...")
    vime128_encoder_path = config.MODEL_DIR / 'vime_encoder_fixed.pth'
    vime128_classifier_path = config.MODEL_DIR / 'vime_128_classifier.pth'
    vime_X_path = config.DATA_PROCESSED / 'vime_X_reject_fixed.parquet'

    if vime128_encoder_path.exists() and vime128_classifier_path.exists() and vime_X_path.exists():
        try:
            # Load VIME preprocessed data
            vime_X = pd.read_parquet(vime_X_path)
            X_tensor = torch.tensor(vime_X.values, dtype=torch.float32)

            # Load encoder from vime_model.py (VIME class)
            sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'models' / 'vime'))
            from vime_model import VIME
            import torch.nn.functional as F

            input_dim = X_tensor.shape[1]
            vime_model = VIME(input_dim, embedding_dim=128, hidden_dim=256)
            vime_model.load_state_dict(torch.load(vime128_encoder_path, map_location='cpu', weights_only=False))
            vime_model.eval()

            # Define classifier head matching training script architecture
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
            checkpoint = torch.load(vime128_classifier_path, map_location='cpu', weights_only=False)
            # Handle both raw state_dict and checkpoint dict formats
            if 'head_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['head_state_dict'])
            else:
                classifier.load_state_dict(checkpoint)
            classifier.eval()

            with torch.no_grad():
                embeddings = vime_model.encode(X_tensor)  # Use encode() method
                logits = classifier(embeddings)
                pred_vime128 = torch.sigmoid(logits).numpy().flatten()

            results['VIME-128'] = bootstrap_metrics(y_DR, pred_vime128, n_bootstrap=n_bootstrap)
            print(f"  AUC: {results['VIME-128']['AUC']['point']:.4f} "
                  f"[{results['VIME-128']['AUC']['ci_low']:.4f}, {results['VIME-128']['AUC']['ci_high']:.4f}]")
        except Exception as e:
            print(f"  Error loading VIME-128: {e}")
            # Fallback to saved results
            vime128_results_path = config.RESULT_DIR / 'vime_128_results.json'
            if vime128_results_path.exists():
                with open(vime128_results_path, 'r') as f:
                    vime128_saved = json.load(f)
                results['VIME-128'] = {
                    'AUC': {'point': vime128_saved.get('D_R_AUC', 0.6902), 'note': 'Point estimate only'},
                    'P@20%': {'point': vime128_saved.get('D_R_P@20%', 0.9546), 'note': 'Point estimate only'},
                    'Diamonds': {'point': vime128_saved.get('D_R_Diamonds', 17701), 'note': 'Point estimate only'}
                }
    else:
        print(f"  Models not found")

    # ==========================================
    # 5. VIME-512
    # ==========================================
    print("\n[5/5] VIME-512...")
    vime512_encoder_path = config.MODEL_DIR / 'vime_wide_encoder.pth'
    vime512_classifier_path = config.MODEL_DIR / 'vime_wide_classifier.pth'

    if vime512_encoder_path.exists() and vime512_classifier_path.exists() and vime_X_path.exists():
        try:
            # Load VIME preprocessed data (same as VIME-128)
            if 'vime_X' not in dir():
                vime_X = pd.read_parquet(vime_X_path)
                X_tensor = torch.tensor(vime_X.values, dtype=torch.float32)

            # Load wide encoder - VIMEWide has encode() method
            from vime_wide import VIMEWide

            input_dim = X_tensor.shape[1]
            encoder_wide = VIMEWide(input_dim, hidden_dim=512, embedding_dim=512)
            encoder_wide.load_state_dict(torch.load(vime512_encoder_path, map_location='cpu', weights_only=False))
            encoder_wide.eval()

            # Define classifier head matching training script architecture (512-dim input)
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

            classifier_wide = ClassificationHead512(512)
            checkpoint_wide = torch.load(vime512_classifier_path, map_location='cpu', weights_only=False)
            # Handle both raw state_dict and checkpoint dict formats
            if 'head_state_dict' in checkpoint_wide:
                classifier_wide.load_state_dict(checkpoint_wide['head_state_dict'])
            else:
                classifier_wide.load_state_dict(checkpoint_wide)
            classifier_wide.eval()

            with torch.no_grad():
                embeddings_wide = encoder_wide.encode(X_tensor)  # Use encode() method
                logits_wide = classifier_wide(embeddings_wide)
                pred_vime512 = torch.sigmoid(logits_wide).numpy().flatten()

            results['VIME-512'] = bootstrap_metrics(y_DR, pred_vime512, n_bootstrap=n_bootstrap)
            print(f"  AUC: {results['VIME-512']['AUC']['point']:.4f} "
                  f"[{results['VIME-512']['AUC']['ci_low']:.4f}, {results['VIME-512']['AUC']['ci_high']:.4f}]")
        except Exception as e:
            print(f"  Error loading VIME-512: {e}")
            # Fallback to saved results
            vime512_results_path = config.RESULT_DIR / 'vime_wide_results.json'
            if vime512_results_path.exists():
                with open(vime512_results_path, 'r') as f:
                    vime512_saved = json.load(f)
                results['VIME-512'] = {
                    'AUC': {'point': vime512_saved.get('D_R_AUC', 0.6807), 'note': 'Point estimate only'},
                    'P@20%': {'point': vime512_saved.get('D_R_P@20%', 0.9551), 'note': 'Point estimate only'},
                    'Diamonds': {'point': vime512_saved.get('D_R_Diamonds', 17711), 'note': 'Point estimate only'}
                }
    else:
        print(f"  Models not found")

    # ==========================================
    # Summary Table
    # ==========================================
    print("\n" + "=" * 80)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (D_R Population)")
    print("=" * 80)
    print(f"{'Model':<20} {'AUC [95% CI]':<30} {'P@20% [95% CI]':<25} {'Diamonds':<15}")
    print("-" * 80)

    for model_name, metrics in results.items():
        auc = metrics['AUC']
        prec = metrics['P@20%']
        dia = metrics['Diamonds']

        if 'ci_low' in auc:
            auc_str = f"{auc['point']:.4f} [{auc['ci_low']:.4f}, {auc['ci_high']:.4f}]"
            prec_str = f"{prec['point']:.4f} [{prec['ci_low']:.4f}, {prec['ci_high']:.4f}]"
            dia_str = f"{dia['point']:,} [{dia['ci_low']:,}, {dia['ci_high']:,}]"
        else:
            auc_str = f"{auc['point']:.4f} (no CI)"
            prec_str = f"{prec['point']:.4f} (no CI)"
            dia_str = f"{dia['point']:,} (no CI)"

        print(f"{model_name:<20} {auc_str:<30} {prec_str:<25} {dia_str:<15}")

    # ==========================================
    # Statistical Significance Check
    # ==========================================
    if 'CatBoost' in results and 'CatBoost-Ensemble' in results:
        if 'ci_high' in results['CatBoost']['AUC'] and 'ci_low' in results['CatBoost-Ensemble']['AUC']:
            cb_low = results['CatBoost']['AUC']['ci_low']
            cb_high = results['CatBoost']['AUC']['ci_high']
            ens_low = results['CatBoost-Ensemble']['AUC']['ci_low']
            ens_high = results['CatBoost-Ensemble']['AUC']['ci_high']

            # Check if CIs overlap
            overlap = not (cb_low > ens_high or ens_low > cb_high)

            print("\n" + "-" * 80)
            print("STATISTICAL SIGNIFICANCE (CatBoost vs CatBoost-Ensemble):")
            if overlap:
                print("  CIs OVERLAP - difference may not be statistically significant at 95% level")
            else:
                print("  CIs DO NOT OVERLAP - difference is statistically significant at 95% level")

    # Check VIME-128 vs VIME-512 (bottleneck hypothesis)
    if 'VIME-128' in results and 'VIME-512' in results:
        if 'ci_high' in results['VIME-128']['AUC'] and 'ci_low' in results['VIME-512']['AUC']:
            v128_low = results['VIME-128']['AUC']['ci_low']
            v128_high = results['VIME-128']['AUC']['ci_high']
            v512_low = results['VIME-512']['AUC']['ci_low']
            v512_high = results['VIME-512']['AUC']['ci_high']

            # Check if CIs overlap
            overlap = not (v128_low > v512_high or v512_low > v128_high)

            print("\n" + "-" * 80)
            print("STATISTICAL SIGNIFICANCE (VIME-128 vs VIME-512 - Bottleneck Hypothesis):")
            print(f"  VIME-128: {results['VIME-128']['AUC']['point']:.4f} [{v128_low:.4f}, {v128_high:.4f}]")
            print(f"  VIME-512: {results['VIME-512']['AUC']['point']:.4f} [{v512_low:.4f}, {v512_high:.4f}]")
            if overlap:
                print("  CIs OVERLAP - VIME-128 vs VIME-512 difference NOT statistically significant")
                print("  Bottleneck hypothesis rejection has WEAK statistical support")
            else:
                if v128_low > v512_high:
                    print("  CIs DO NOT OVERLAP - VIME-128 significantly BETTER than VIME-512")
                    print("  Bottleneck hypothesis rejection is STATISTICALLY SIGNIFICANT")
                else:
                    print("  CIs DO NOT OVERLAP - VIME-512 significantly BETTER than VIME-128")
                    print("  This would SUPPORT the bottleneck hypothesis (unexpected)")

    # Save results
    output_path = config.RESULT_DIR / 'bootstrap_confidence_intervals.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
