"""
Diamond Recovery Analysis

Business Question: How many good borrowers ("diamonds") did we correctly
identify among the rejected population (D_R)?

Diamonds = Non-defaulters in D_R (TARGET=0 in rejected population)
These are good borrowers who were wrongly rejected by the simple scorecard.

Metrics:
- Precision@k: If we approve top k% of D_R, what % are truly good?
- Recall@k: What fraction of all diamonds do we capture in top k%?
- Lift: How much better than random selection?
"""
import logging
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data():
    """Load embeddings and D_R ground truth."""
    X_reject = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_reject.parquet").values

    # Ground truth for D_R
    dr_truth = pd.read_parquet(config.SPLIT_DR_TRUTH)
    y_reject_true = dr_truth["TARGET"].values

    return X_reject, y_reject_true


def diamond_metrics_at_k(y_true, y_pred_proba, k_percent):
    """
    Calculate diamond recovery metrics at top k%.

    Diamonds are NON-defaulters (TARGET=0).
    Lower predicted probability = more likely to be a diamond.

    Args:
        y_true: Ground truth (1=default, 0=good/diamond)
        y_pred_proba: Predicted P(default)
        k_percent: Top k% to consider (e.g., 10 means top 10%)

    Returns:
        Dict with precision, recall, lift, and counts
    """
    n_total = len(y_true)
    n_select = int(n_total * k_percent / 100)

    # Sort by predicted probability (ascending = most likely to be good)
    sorted_indices = np.argsort(y_pred_proba)
    top_k_indices = sorted_indices[:n_select]

    # Ground truth for selected samples
    y_selected = y_true[top_k_indices]

    # Diamonds are TARGET=0 (non-defaulters)
    diamonds_in_selection = (y_selected == 0).sum()
    total_diamonds = (y_true == 0).sum()

    # Precision: What fraction of selected are diamonds?
    precision = diamonds_in_selection / n_select if n_select > 0 else 0

    # Recall: What fraction of all diamonds did we capture?
    recall = diamonds_in_selection / total_diamonds if total_diamonds > 0 else 0

    # Baseline (random selection)
    baseline_precision = total_diamonds / n_total

    # Lift: How much better than random?
    lift = precision / baseline_precision if baseline_precision > 0 else 0

    return {
        "k_percent": k_percent,
        "n_selected": n_select,
        "diamonds_found": int(diamonds_in_selection),
        "total_diamonds": int(total_diamonds),
        "precision": float(precision),
        "recall": float(recall),
        "baseline_precision": float(baseline_precision),
        "lift": float(lift),
    }


def run_diamond_analysis():
    """Run diamond recovery analysis on D_R."""
    logger.info("=" * 70)
    logger.info("Diamond Recovery Analysis")
    logger.info("=" * 70)
    logger.info("Diamonds = Good borrowers (non-defaulters) wrongly rejected")
    logger.info("")

    # Load data
    X_reject, y_reject_true = load_data()

    # D_R statistics
    n_total = len(y_reject_true)
    n_defaults = y_reject_true.sum()
    n_diamonds = n_total - n_defaults
    default_rate = n_defaults / n_total

    logger.info(f"D_R Population: {n_total:,} rejected applicants")
    logger.info(f"  Defaults (bad):     {n_defaults:,} ({default_rate:.1%})")
    logger.info(f"  Diamonds (good):    {n_diamonds:,} ({1-default_rate:.1%})")
    logger.info("")

    # Load trained model
    model_path = config.MODEL_DIR / "route_a_final.cbm"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    logger.info(f"Loaded model: {model_path.name}")

    # Predict on D_R
    y_pred_proba = model.predict_proba(X_reject)[:, 1]

    # AUC on D_R (using ground truth)
    auc_dr = roc_auc_score(y_reject_true, y_pred_proba)
    logger.info(f"AUC on D_R (ground truth): {auc_dr:.4f}")
    logger.info("")

    # Calculate metrics at various k%
    k_values = [5, 10, 15, 20, 25, 30, 40, 50]

    logger.info("Diamond Recovery at Top k%:")
    logger.info("-" * 70)
    logger.info(f"{'k%':>5} | {'Selected':>8} | {'Diamonds':>10} | {'Precision':>10} | {'Recall':>8} | {'Lift':>6}")
    logger.info("-" * 70)

    results = {
        "d_r_stats": {
            "total": n_total,
            "defaults": int(n_defaults),
            "diamonds": int(n_diamonds),
            "default_rate": float(default_rate),
        },
        "auc_on_d_r": float(auc_dr),
        "recovery_at_k": {}
    }

    for k in k_values:
        metrics = diamond_metrics_at_k(y_reject_true, y_pred_proba, k)
        results["recovery_at_k"][str(k)] = metrics

        logger.info(f"{k:>5}% | {metrics['n_selected']:>8,} | {metrics['diamonds_found']:>10,} | "
                   f"{metrics['precision']:>10.1%} | {metrics['recall']:>8.1%} | {metrics['lift']:>6.2f}x")

    logger.info("-" * 70)
    logger.info(f"Baseline (random): {results['d_r_stats']['diamonds']/results['d_r_stats']['total']:.1%} precision, 1.00x lift")

    # Business interpretation
    logger.info("")
    logger.info("=" * 70)
    logger.info("Business Interpretation")
    logger.info("=" * 70)

    # Example: Top 20%
    k20 = results["recovery_at_k"]["20"]
    logger.info(f"\nIf we approve the top 20% of rejected applicants ({k20['n_selected']:,} people):")
    logger.info(f"  - {k20['diamonds_found']:,} would be diamonds (good borrowers)")
    logger.info(f"  - {k20['n_selected'] - k20['diamonds_found']:,} would be defaults (bad)")
    logger.info(f"  - Precision: {k20['precision']:.1%} (vs {k20['baseline_precision']:.1%} random)")
    logger.info(f"  - Lift: {k20['lift']:.2f}x better than random selection")
    logger.info(f"  - We capture {k20['recall']:.1%} of all diamonds in D_R")

    # Save results
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULT_DIR / "diamond_recovery_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    results = run_diamond_analysis()
