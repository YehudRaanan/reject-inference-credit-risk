"""
Diamond Recovery Comparison: Route A vs Route B

Compares both routes on the business metric: finding diamonds (good borrowers)
in the rejected population D_R.
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


def diamond_metrics_at_k(y_true, y_pred_proba, k_percent):
    """Calculate diamond recovery metrics at top k%."""
    n_total = len(y_true)
    n_select = int(n_total * k_percent / 100)

    sorted_indices = np.argsort(y_pred_proba)
    top_k_indices = sorted_indices[:n_select]
    y_selected = y_true[top_k_indices]

    diamonds_in_selection = (y_selected == 0).sum()
    total_diamonds = (y_true == 0).sum()

    precision = diamonds_in_selection / n_select if n_select > 0 else 0
    recall = diamonds_in_selection / total_diamonds if total_diamonds > 0 else 0
    baseline_precision = total_diamonds / n_total
    lift = precision / baseline_precision if baseline_precision > 0 else 0

    return {
        "k_percent": k_percent,
        "n_selected": n_select,
        "diamonds_found": int(diamonds_in_selection),
        "precision": float(precision),
        "recall": float(recall),
        "lift": float(lift),
    }


def run_comparison():
    """Compare Route A and Route B diamond recovery."""
    logger.info("=" * 70)
    logger.info("Diamond Recovery Comparison: Route A vs Route B")
    logger.info("=" * 70)

    # Load ground truth
    dr_truth = pd.read_parquet(config.SPLIT_DR_TRUTH)
    y_true = dr_truth["TARGET"].values

    n_total = len(y_true)
    n_diamonds = (y_true == 0).sum()
    n_defaults = y_true.sum()

    logger.info(f"D_R Population: {n_total:,} rejected applicants")
    logger.info(f"  Diamonds (good): {n_diamonds:,} ({n_diamonds/n_total:.1%})")
    logger.info(f"  Defaults (bad):  {n_defaults:,} ({n_defaults/n_total:.1%})")
    logger.info("")

    # === Route A (VIME embeddings) ===
    logger.info("Loading Route A model (VIME embeddings)...")
    X_reject_vime = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_reject.parquet").values

    model_a = CatBoostClassifier()
    model_a.load_model(str(config.MODEL_DIR / "route_a_final.cbm"))

    proba_a = model_a.predict_proba(X_reject_vime)[:, 1]
    auc_a = roc_auc_score(y_true, proba_a)
    logger.info(f"Route A AUC on D_R: {auc_a:.4f}")

    # === Route B (raw features) ===
    logger.info("Loading Route B model (raw features)...")
    df_reject = pd.read_parquet(config.SPLIT_DR_FEATURES)

    # Handle categorical NaN (same as training)
    cat_features = [col for col in config.CATEGORICAL_COLS if col in df_reject.columns]
    for col in cat_features:
        df_reject[col] = df_reject[col].fillna("_MISSING_")

    model_b = CatBoostClassifier()
    model_b.load_model(str(config.MODEL_DIR / "route_b_final.cbm"))

    proba_b = model_b.predict_proba(df_reject)[:, 1]
    auc_b = roc_auc_score(y_true, proba_b)
    logger.info(f"Route B AUC on D_R: {auc_b:.4f}")
    logger.info("")

    # === Comparison at various k% ===
    k_values = [5, 10, 15, 20, 25, 30, 40, 50]

    logger.info("Diamond Recovery Comparison at Top k%:")
    logger.info("-" * 80)
    logger.info(f"{'k%':>5} | {'Route A Prec':>12} | {'Route A Lift':>12} | {'Route B Prec':>12} | {'Route B Lift':>12} | {'Winner':>8}")
    logger.info("-" * 80)

    results = {
        "d_r_stats": {
            "total": n_total,
            "diamonds": int(n_diamonds),
            "defaults": int(n_defaults),
        },
        "route_a": {"auc_on_d_r": float(auc_a), "recovery_at_k": {}},
        "route_b": {"auc_on_d_r": float(auc_b), "recovery_at_k": {}},
    }

    for k in k_values:
        metrics_a = diamond_metrics_at_k(y_true, proba_a, k)
        metrics_b = diamond_metrics_at_k(y_true, proba_b, k)

        results["route_a"]["recovery_at_k"][str(k)] = metrics_a
        results["route_b"]["recovery_at_k"][str(k)] = metrics_b

        winner = "A" if metrics_a["precision"] > metrics_b["precision"] else "B"
        if abs(metrics_a["precision"] - metrics_b["precision"]) < 0.001:
            winner = "TIE"

        logger.info(f"{k:>5}% | {metrics_a['precision']:>11.1%} | {metrics_a['lift']:>11.2f}x | "
                   f"{metrics_b['precision']:>11.1%} | {metrics_b['lift']:>11.2f}x | {winner:>8}")

    logger.info("-" * 80)
    logger.info(f"Baseline (random): {n_diamonds/n_total:.1%} precision, 1.00x lift")

    # === Summary ===
    logger.info("")
    logger.info("=" * 70)
    logger.info("Summary Comparison")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<30} | {'Route A':>15} | {'Route B':>15} | {'Δ (A-B)':>10}")
    logger.info("-" * 70)
    logger.info(f"{'AUC on D_R':<30} | {auc_a:>15.4f} | {auc_b:>15.4f} | {auc_a-auc_b:>+10.4f}")

    k20_a = results["route_a"]["recovery_at_k"]["20"]
    k20_b = results["route_b"]["recovery_at_k"]["20"]
    logger.info(f"{'Precision@20%':<30} | {k20_a['precision']:>14.1%} | {k20_b['precision']:>14.1%} | {(k20_a['precision']-k20_b['precision'])*100:>+9.1f}pp")
    logger.info(f"{'Lift@20%':<30} | {k20_a['lift']:>14.2f}x | {k20_b['lift']:>14.2f}x | {k20_a['lift']-k20_b['lift']:>+10.2f}")
    logger.info(f"{'Diamonds Found@20%':<30} | {k20_a['diamonds_found']:>15,} | {k20_b['diamonds_found']:>15,} | {k20_a['diamonds_found']-k20_b['diamonds_found']:>+10,}")

    # Save results
    output_path = config.RESULT_DIR / "diamond_recovery_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    results = run_comparison()
