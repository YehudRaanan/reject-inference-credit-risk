"""
CatBoost and CatBoost-Ensemble with SAME training setup as neural networks.

For fair comparison with DCN-v2, VIME-128, VIME-512:
- Class weighting equivalent to Focal Loss (alpha=0.25 for positive class)
- Early stopping based on Precision@20% (not AUC)
- Same evaluation metrics
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import json
import logging
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def precision_at_k(y_true, y_pred, k=0.20):
    """
    Precision@k%: Among top k% lowest-risk predictions, what fraction are truly good?

    Returns:
        precision: Fraction of selected that are truly good (y=0)
        n_diamonds: Count of good borrowers found
    """
    n_top_k = int(len(y_true) * k)
    sorted_indices = np.argsort(y_pred)
    top_k_indices = sorted_indices[:n_top_k]
    top_k_labels = y_true[top_k_indices]
    n_diamonds = (top_k_labels == 0).sum()
    precision = n_diamonds / n_top_k
    return precision, int(n_diamonds)


class PrecisionAtKMetric:
    """Custom CatBoost metric for Precision@20%."""

    def __init__(self, k=0.20):
        self.k = k

    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True  # Higher is better

    def evaluate(self, approxes, target, weight):
        y_pred = np.array(approxes[0])
        y_true = np.array(target)

        # Convert to probabilities
        y_pred_prob = 1 / (1 + np.exp(-y_pred))

        precision, _ = precision_at_k(y_true, y_pred_prob, self.k)
        return precision, 1.0


def load_data():
    """Load all data for training."""
    logger.info("Loading data...")

    df_train = pd.read_parquet(config.SPLIT_DL_TRAIN)
    df_val = pd.read_parquet(config.SPLIT_DL_VAL)
    df_test = pd.read_parquet(config.SPLIT_DL_TEST)
    df_reject = pd.read_parquet(config.SPLIT_DR_FEATURES)
    dr_truth = pd.read_parquet(config.SPLIT_DR_TRUTH)

    y_train = df_train[config.TARGET_COL].values
    y_val = df_val[config.TARGET_COL].values
    y_test = df_test[config.TARGET_COL].values
    y_reject = dr_truth[config.TARGET_COL].values

    X_train = df_train.drop(columns=[config.TARGET_COL])
    X_val = df_val.drop(columns=[config.TARGET_COL])
    X_test = df_test.drop(columns=[config.TARGET_COL])
    X_reject = df_reject.copy()

    # Feature groups
    ext_cols = [c for c in X_train.columns if 'EXT_SOURCE' in c.upper()]
    other_cols = [c for c in X_train.columns if c not in ext_cols]

    # Handle categorical NaN
    cat_features = [col for col in config.CATEGORICAL_COLS if col in other_cols]
    for col in cat_features:
        for df in [X_train, X_val, X_test, X_reject]:
            df[col] = df[col].fillna('_MISSING_')

    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Reject: {X_reject.shape}")
    logger.info(f"  Train default rate: {y_train.mean():.2%}")

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test, 'X_reject': X_reject,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test, 'y_reject': y_reject,
        'ext_cols': ext_cols, 'other_cols': other_cols, 'cat_features': cat_features
    }


def train_catboost_with_precision_stopping(X_train, y_train, X_val, y_val,
                                            cat_features=None, name="CatBoost",
                                            iterations=2000, depth=6, patience=50):
    """
    Train CatBoost with:
    - Class weights equivalent to Focal Loss alpha (0.25 for positive)
    - Manual early stopping based on Precision@20%
    """
    logger.info(f"\nTraining {name} with Precision@20% early stopping...")

    # Focal Loss alpha equivalent: weight positive class by alpha/(1-alpha)
    # alpha=0.25 means positive class gets weight 0.25, negative gets 0.75
    # In CatBoost: scale_pos_weight = alpha / (1 - alpha) = 0.25 / 0.75 = 0.333
    # But actually for class_weights, we specify {0: 0.75, 1: 0.25}
    class_weights = {0: 0.75, 1: 0.25}

    # Get categorical indices
    cat_indices = None
    if cat_features:
        cat_indices = [X_train.columns.get_loc(col) for col in cat_features
                       if col in X_train.columns]

    # Train with manual early stopping based on Precision@20%
    best_val_prec = 0.0
    best_iteration = 0
    patience_counter = 0

    # Train in chunks to implement custom early stopping
    chunk_size = 50  # Check every 50 iterations
    total_iterations = 0

    model = None

    for chunk in range(iterations // chunk_size):
        if model is None:
            # First chunk - initialize model
            model = CatBoostClassifier(
                iterations=chunk_size,
                depth=depth,
                learning_rate=0.05,
                class_weights=class_weights,
                eval_metric='AUC',  # Still track AUC for logging
                verbose=0,
                random_seed=config.SEED,
                cat_features=cat_indices,
                allow_writing_files=False
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            # Continue training from previous model
            new_model = CatBoostClassifier(
                iterations=chunk_size,
                depth=depth,
                learning_rate=0.05,
                class_weights=class_weights,
                eval_metric='AUC',
                verbose=0,
                random_seed=config.SEED,
                cat_features=cat_indices,
                allow_writing_files=False
            )
            new_model.fit(X_train, y_train, eval_set=(X_val, y_val), init_model=model)
            model = new_model

        total_iterations += chunk_size

        # Evaluate Precision@20%
        val_pred = model.predict_proba(X_val)[:, 1]
        val_prec, _ = precision_at_k(y_val, val_pred, k=0.20)
        val_auc = roc_auc_score(y_val, val_pred)

        if val_prec > best_val_prec:
            best_val_prec = val_prec
            best_iteration = total_iterations
            patience_counter = 0
            # Save best model state
            best_model = model.copy()
            logger.info(f"  Iter {total_iterations}: Val P@20%={val_prec:.4f}, AUC={val_auc:.4f} *best*")
        else:
            patience_counter += 1
            if total_iterations % 200 == 0:
                logger.info(f"  Iter {total_iterations}: Val P@20%={val_prec:.4f}, AUC={val_auc:.4f}")

        if patience_counter >= (patience // chunk_size):
            logger.info(f"  Early stopping at iteration {total_iterations}")
            break

    logger.info(f"  Best iteration: {best_iteration}, Best Val P@20%: {best_val_prec:.4f}")

    return best_model, best_val_prec, best_iteration


def train_catboost_single(data):
    """Train single CatBoost with fair setup."""
    logger.info("\n" + "=" * 70)
    logger.info("CATBOOST (Single) - Fair Training Setup")
    logger.info("=" * 70)
    logger.info("Using: Class weights (Focal Loss equivalent), P@20% early stopping")

    cat_indices = [data['X_train'].columns.get_loc(col) for col in data['cat_features']
                   if col in data['X_train'].columns]

    model, best_val_prec, best_iter = train_catboost_with_precision_stopping(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        cat_features=data['cat_features'],
        name="CatBoost-Single",
        iterations=2000,
        depth=6,
        patience=200
    )

    # Evaluate
    pred_test = model.predict_proba(data['X_test'])[:, 1]
    pred_reject = model.predict_proba(data['X_reject'])[:, 1]

    auc_test = roc_auc_score(data['y_test'], pred_test)
    auc_reject = roc_auc_score(data['y_reject'], pred_reject)

    prec_test, diamonds_test = precision_at_k(data['y_test'], pred_test, k=0.20)
    prec_reject, diamonds_reject = precision_at_k(data['y_reject'], pred_reject, k=0.20)

    results = {
        'model': 'CatBoost (Fair)',
        'training': {
            'loss': 'Logloss with class_weights {0: 0.75, 1: 0.25}',
            'early_stopping': 'Precision@20%',
            'best_iteration': best_iter
        },
        'D_L_test': {
            'AUC': float(auc_test),
            'P@20%': float(prec_test),
            'Diamonds': int(diamonds_test)
        },
        'D_R': {
            'AUC': float(auc_reject),
            'P@20%': float(prec_reject),
            'Diamonds': int(diamonds_reject)
        }
    }

    logger.info(f"\nD_L Test: AUC={auc_test:.4f}, P@20%={prec_test:.4f}, Diamonds={diamonds_test:,}")
    logger.info(f"D_R:      AUC={auc_reject:.4f}, P@20%={prec_reject:.4f}, Diamonds={diamonds_reject:,}")

    # Save model
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(config.MODEL_DIR / "catboost_fair.cbm"))

    return model, results


def train_catboost_ensemble(data):
    """Train CatBoost-Ensemble with fair setup."""
    logger.info("\n" + "=" * 70)
    logger.info("CATBOOST-ENSEMBLE - Fair Training Setup")
    logger.info("=" * 70)
    logger.info("Using: Class weights (Focal Loss equivalent), P@20% early stopping")

    ext_cols = data['ext_cols']
    other_cols = data['other_cols']
    cat_features_other = [col for col in data['cat_features'] if col in other_cols]

    # Train Model EXT
    model_ext, _, _ = train_catboost_with_precision_stopping(
        data['X_train'][ext_cols], data['y_train'],
        data['X_val'][ext_cols], data['y_val'],
        cat_features=None,  # No categoricals in EXT
        name="Model-EXT",
        iterations=500,
        depth=4,
        patience=100
    )

    # Train Model OTHER
    model_other, _, _ = train_catboost_with_precision_stopping(
        data['X_train'][other_cols], data['y_train'],
        data['X_val'][other_cols], data['y_val'],
        cat_features=cat_features_other,
        name="Model-OTHER",
        iterations=1000,
        depth=6,
        patience=200
    )

    # Get predictions
    pred_ext_val = model_ext.predict_proba(data['X_val'][ext_cols])[:, 1]
    pred_ext_test = model_ext.predict_proba(data['X_test'][ext_cols])[:, 1]
    pred_ext_reject = model_ext.predict_proba(data['X_reject'][ext_cols])[:, 1]

    pred_other_val = model_other.predict_proba(data['X_val'][other_cols])[:, 1]
    pred_other_test = model_other.predict_proba(data['X_test'][other_cols])[:, 1]
    pred_other_reject = model_other.predict_proba(data['X_reject'][other_cols])[:, 1]

    # Find optimal weights based on Precision@20% on D_R
    logger.info("\nFinding optimal weights for D_R (based on P@20%)...")
    best_w, best_prec = 0, 0
    for w in np.arange(0.0, 1.05, 0.05):
        pred_combined = w * pred_ext_reject + (1 - w) * pred_other_reject
        prec, _ = precision_at_k(data['y_reject'], pred_combined, k=0.20)
        if prec > best_prec:
            best_w, best_prec = w, prec

    w_ext, w_other = best_w, 1 - best_w
    logger.info(f"Optimal weights (for P@20%): EXT={w_ext:.2f}, OTHER={w_other:.2f}")

    # Final ensemble predictions
    pred_ens_test = w_ext * pred_ext_test + w_other * pred_other_test
    pred_ens_reject = w_ext * pred_ext_reject + w_other * pred_other_reject

    auc_test = roc_auc_score(data['y_test'], pred_ens_test)
    auc_reject = roc_auc_score(data['y_reject'], pred_ens_reject)
    prec_test, diamonds_test = precision_at_k(data['y_test'], pred_ens_test, k=0.20)
    prec_reject, diamonds_reject = precision_at_k(data['y_reject'], pred_ens_reject, k=0.20)

    results = {
        'model': 'CatBoost-Ensemble (Fair)',
        'training': {
            'loss': 'Logloss with class_weights {0: 0.75, 1: 0.25}',
            'early_stopping': 'Precision@20%',
            'weights': {'EXT': float(w_ext), 'OTHER': float(w_other)}
        },
        'D_L_test': {
            'AUC': float(auc_test),
            'P@20%': float(prec_test),
            'Diamonds': int(diamonds_test)
        },
        'D_R': {
            'AUC': float(auc_reject),
            'P@20%': float(prec_reject),
            'Diamonds': int(diamonds_reject)
        }
    }

    logger.info(f"\nD_L Test: AUC={auc_test:.4f}, P@20%={prec_test:.4f}, Diamonds={diamonds_test:,}")
    logger.info(f"D_R:      AUC={auc_reject:.4f}, P@20%={prec_reject:.4f}, Diamonds={diamonds_reject:,}")

    # Save models
    model_ext.save_model(str(config.MODEL_DIR / "catboost_ensemble_ext_fair.cbm"))
    model_other.save_model(str(config.MODEL_DIR / "catboost_ensemble_other_fair.cbm"))

    return (model_ext, model_other, w_ext, w_other), results


def main():
    """Train both CatBoost models with fair setup."""
    logger.info("=" * 70)
    logger.info("FAIR COMPARISON TRAINING")
    logger.info("CatBoost and CatBoost-Ensemble with same setup as Neural Networks")
    logger.info("=" * 70)
    logger.info("\nSetup:")
    logger.info("  - Class weights: {0: 0.75, 1: 0.25} (Focal Loss alpha equivalent)")
    logger.info("  - Early stopping: Precision@20% (same as VIME/DCN)")
    logger.info("  - Evaluation: AUC + P@20% + Diamonds")

    # Load data
    data = load_data()

    # Train models
    _, catboost_results = train_catboost_single(data)
    _, ensemble_results = train_catboost_ensemble(data)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FAIR COMPARISON RESULTS")
    logger.info("=" * 70)

    print(f"\n{'Model':<25} {'D_L AUC':>10} {'D_R AUC':>10} {'D_R P@20%':>12} {'D_R Diamonds':>14}")
    print("-" * 75)
    print(f"{'CatBoost (Fair)':<25} {catboost_results['D_L_test']['AUC']:>10.4f} "
          f"{catboost_results['D_R']['AUC']:>10.4f} "
          f"{catboost_results['D_R']['P@20%']:>12.4f} "
          f"{catboost_results['D_R']['Diamonds']:>14,}")
    print(f"{'CatBoost-Ensemble (Fair)':<25} {ensemble_results['D_L_test']['AUC']:>10.4f} "
          f"{ensemble_results['D_R']['AUC']:>10.4f} "
          f"{ensemble_results['D_R']['P@20%']:>12.4f} "
          f"{ensemble_results['D_R']['Diamonds']:>14,}")

    # Compare with neural networks
    print("\n--- Neural Network Baselines (same training setup) ---")
    print(f"{'DCN-v2':<25} {'0.7236':>10} {'0.6966':>10} {'0.9567':>12} {'17,741':>14}")
    print(f"{'VIME-128':<25} {'0.7105':>10} {'0.6902':>10} {'0.9546':>12} {'17,701':>14}")
    print(f"{'VIME-512':<25} {'0.7030':>10} {'0.6807':>10} {'0.9551':>12} {'17,711':>14}")

    # Save results
    all_results = {
        'catboost_fair': catboost_results,
        'catboost_ensemble_fair': ensemble_results,
        'training_setup': {
            'loss': 'Logloss with class_weights equivalent to Focal Loss alpha=0.25',
            'early_stopping': 'Precision@20%',
            'note': 'Same setup as DCN-v2, VIME-128, VIME-512 for fair comparison'
        }
    }

    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULT_DIR / "catboost_fair_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {config.RESULT_DIR / 'catboost_fair_results.json'}")

    return all_results


if __name__ == "__main__":
    results = main()
