"""
Phase 6: Comprehensive Evaluation & Comparison

Evaluates Route A vs Route B across three populations:
1. D_L_test (approved population)
2. D_R_ground_truth (rejected population - diamond recovery)
3. Full population (D_L_test + D_R)

Metrics:
- AUC-ROC, PR-AUC
- Precision@k, Recall@k, Lift
- Calibration curves
- SHAP analysis (Route B only)
"""

import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, brier_score_loss
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_models_and_data():
    """Load trained models and test data."""
    logger.info("Loading models and data...")

    # Load Route A model (VIME embeddings)
    model_a = CatBoostClassifier()
    model_a.load_model(str(config.MODEL_DIR / "route_a_final.cbm"))

    # Load Route B model (raw features)
    model_b = CatBoostClassifier()
    model_b.load_model(str(config.MODEL_DIR / "route_b_final.cbm"))

    # Load D_L_test data
    df_test = pd.read_parquet(config.SPLIT_DL_TEST)
    y_test = df_test[config.TARGET_COL].values
    X_test_raw = df_test.drop(columns=[config.TARGET_COL])

    # Load VIME embeddings for test
    X_test_vime = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_test.parquet").values

    # Load D_R data
    df_reject = pd.read_parquet(config.SPLIT_DR_FEATURES)
    dr_truth = pd.read_parquet(config.SPLIT_DR_TRUTH)
    y_reject = dr_truth[config.TARGET_COL].values
    X_reject_raw = df_reject.copy()
    X_reject_vime = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_reject.parquet").values

    # Handle categorical NaN for Route B
    cat_features = [col for col in config.CATEGORICAL_COLS if col in X_test_raw.columns]
    for col in cat_features:
        X_test_raw[col] = X_test_raw[col].fillna("_MISSING_")
        X_reject_raw[col] = X_reject_raw[col].fillna("_MISSING_")

    logger.info(f"  D_L_test: {len(y_test)} samples")
    logger.info(f"  D_R: {len(y_reject)} samples")

    return {
        "model_a": model_a,
        "model_b": model_b,
        "X_test_vime": X_test_vime,
        "X_test_raw": X_test_raw,
        "y_test": y_test,
        "X_reject_vime": X_reject_vime,
        "X_reject_raw": X_reject_raw,
        "y_reject": y_reject,
    }


def evaluate_population(model, X, y_true, population_name):
    """Compute comprehensive metrics for a population."""
    y_pred = model.predict_proba(X)[:, 1]

    # Core metrics
    auc_roc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)

    # Curves
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    logger.info(f"  {population_name}: AUC-ROC={auc_roc:.4f}, PR-AUC={pr_auc:.4f}, Brier={brier:.4f}")

    return {
        "auc_roc": float(auc_roc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "n_samples": len(y_true),
        "default_rate": float(y_true.mean()),
        "curves": {
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr": {"precision": precision.tolist(), "recall": recall.tolist()},
        },
        "y_pred": y_pred,
    }


def diamond_recovery_at_k(y_true, y_pred, k_percent):
    """Calculate diamond recovery metrics at top k%."""
    n_total = len(y_true)
    n_select = int(n_total * k_percent / 100)

    # Sort by predicted probability (ascending = most likely to be good/diamond)
    sorted_indices = np.argsort(y_pred)
    top_k_indices = sorted_indices[:n_select]
    y_selected = y_true[top_k_indices]

    # Diamonds are TARGET=0
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
        "total_diamonds": int(total_diamonds),
        "precision": float(precision),
        "recall": float(recall),
        "lift": float(lift),
    }


def evaluate_diamond_recovery(y_true, y_pred, route_name):
    """Full diamond recovery analysis."""
    k_values = [5, 10, 15, 20, 25, 30, 40, 50]
    results = {}

    for k in k_values:
        results[str(k)] = diamond_recovery_at_k(y_true, y_pred, k)

    return results


def plot_roc_comparison(results, output_path):
    """Plot ROC curves for both routes on all populations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    populations = ["D_L_test", "D_R", "Full"]

    for ax, pop in zip(axes, populations):
        for route in ["A", "B"]:
            key = f"route_{route.lower()}_{pop.lower().replace('_', '')}"
            if pop == "D_L_test":
                key = f"route_{route.lower()}_dl_test"
            elif pop == "D_R":
                key = f"route_{route.lower()}_dr"
            else:
                key = f"route_{route.lower()}_full"

            if key in results:
                curves = results[key]["curves"]["roc"]
                auc = results[key]["auc_roc"]
                ax.plot(curves["fpr"], curves["tpr"], label=f"Route {route} (AUC={auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {pop}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC comparison: {output_path}")


def plot_pr_comparison(results, output_path):
    """Plot PR curves for both routes on D_R (diamond recovery focus)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for route in ["A", "B"]:
        key = f"route_{route.lower()}_dr"
        if key in results:
            curves = results[key]["curves"]["pr"]
            pr_auc = results[key]["pr_auc"]
            ax.plot(curves["recall"], curves["precision"], label=f"Route {route} (PR-AUC={pr_auc:.4f})")

    baseline = results["route_a_dr"]["default_rate"] if "route_a_dr" in results else 0.13
    ax.axhline(y=1-baseline, color="gray", linestyle="--", label=f"Baseline ({1-baseline:.1%} diamonds)")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - D_R (Diamond Recovery)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved PR comparison: {output_path}")


def plot_lift_comparison(diamond_results_a, diamond_results_b, output_path):
    """Plot lift curves for diamond recovery."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    k_values = [5, 10, 15, 20, 25, 30, 40, 50]

    # Precision@k
    ax = axes[0]
    prec_a = [diamond_results_a[str(k)]["precision"] * 100 for k in k_values]
    prec_b = [diamond_results_b[str(k)]["precision"] * 100 for k in k_values]

    ax.plot(k_values, prec_a, "o-", label="Route A (VIME)", color="blue")
    ax.plot(k_values, prec_b, "s-", label="Route B (Raw)", color="green")
    ax.axhline(y=86.6, color="gray", linestyle="--", label="Baseline (86.6%)")
    ax.set_xlabel("Top k% Selected")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Diamond Precision @ Top k%")
    ax.legend()
    ax.grid(alpha=0.3)

    # Lift@k
    ax = axes[1]
    lift_a = [diamond_results_a[str(k)]["lift"] for k in k_values]
    lift_b = [diamond_results_b[str(k)]["lift"] for k in k_values]

    ax.plot(k_values, lift_a, "o-", label="Route A (VIME)", color="blue")
    ax.plot(k_values, lift_b, "s-", label="Route B (Raw)", color="green")
    ax.axhline(y=1.0, color="gray", linestyle="--", label="Random (1.0x)")
    ax.set_xlabel("Top k% Selected")
    ax.set_ylabel("Lift")
    ax.set_title("Lift over Random @ Top k%")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved lift comparison: {output_path}")


def run_shap_analysis(model, X, feature_names, output_path, n_samples=1000):
    """Run SHAP analysis for Route B."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed, skipping SHAP analysis")
        return None

    logger.info(f"Running SHAP analysis on {n_samples} samples...")

    # Sample for efficiency
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    else:
        X_sample = X

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP summary: {output_path}")

    # Feature importance
    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    return importance_df


def run_phase6_evaluation():
    """Run full Phase 6 evaluation."""
    logger.info("=" * 70)
    logger.info("Phase 6: Comprehensive Evaluation & Comparison")
    logger.info("=" * 70)

    # Load data
    data = load_models_and_data()

    results = {}

    # === 6.1 Evaluate on D_L_test ===
    logger.info("\n[6.1] Evaluating on D_L_test (approved population)...")

    results["route_a_dl_test"] = evaluate_population(
        data["model_a"], data["X_test_vime"], data["y_test"], "Route A"
    )
    results["route_b_dl_test"] = evaluate_population(
        data["model_b"], data["X_test_raw"], data["y_test"], "Route B"
    )

    # === 6.2 Evaluate on D_R (diamond recovery) ===
    logger.info("\n[6.2] Evaluating on D_R (rejected population - diamond recovery)...")

    results["route_a_dr"] = evaluate_population(
        data["model_a"], data["X_reject_vime"], data["y_reject"], "Route A"
    )
    results["route_b_dr"] = evaluate_population(
        data["model_b"], data["X_reject_raw"], data["y_reject"], "Route B"
    )

    # Diamond recovery metrics
    diamond_a = evaluate_diamond_recovery(
        data["y_reject"], results["route_a_dr"]["y_pred"], "Route A"
    )
    diamond_b = evaluate_diamond_recovery(
        data["y_reject"], results["route_b_dr"]["y_pred"], "Route B"
    )
    results["diamond_recovery_a"] = diamond_a
    results["diamond_recovery_b"] = diamond_b

    # === 6.3 Evaluate on full population ===
    logger.info("\n[6.3] Evaluating on full population (D_L_test + D_R)...")

    # Combine populations
    X_full_vime = np.vstack([data["X_test_vime"], data["X_reject_vime"]])
    X_full_raw = pd.concat([data["X_test_raw"], data["X_reject_raw"]], ignore_index=True)
    y_full = np.concatenate([data["y_test"], data["y_reject"]])

    results["route_a_full"] = evaluate_population(
        data["model_a"], X_full_vime, y_full, "Route A"
    )
    results["route_b_full"] = evaluate_population(
        data["model_b"], X_full_raw, y_full, "Route B"
    )

    # === 6.4 SHAP Analysis (Route B only) ===
    logger.info("\n[6.4] Running SHAP analysis for Route B...")

    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shap_output = config.FIGURE_DIR / "shap_route_b.png"

    shap_importance = run_shap_analysis(
        data["model_b"],
        data["X_test_raw"],
        data["X_test_raw"].columns.tolist(),
        shap_output
    )

    if shap_importance is not None:
        results["shap_top_features"] = shap_importance.head(20).to_dict("records")

    # === 6.5 Generate comparison figures ===
    logger.info("\n[6.5] Generating comparison figures...")

    # Remove y_pred from results before saving (not JSON serializable in full)
    results_to_save = {}
    for k, v in results.items():
        if isinstance(v, dict) and "y_pred" in v:
            v_copy = {kk: vv for kk, vv in v.items() if kk != "y_pred"}
            # Also simplify curves for JSON
            if "curves" in v_copy:
                v_copy["curves"] = {
                    "roc_auc": v_copy.get("auc_roc"),
                    "pr_auc": v_copy.get("pr_auc"),
                }
            results_to_save[k] = v_copy
        else:
            results_to_save[k] = v

    # ROC comparison
    plot_roc_comparison(results, config.FIGURE_DIR / "roc_comparison.png")

    # PR comparison
    plot_pr_comparison(results, config.FIGURE_DIR / "pr_comparison_dr.png")

    # Lift comparison
    plot_lift_comparison(diamond_a, diamond_b, config.FIGURE_DIR / "lift_comparison.png")

    # === Summary ===
    logger.info("\n" + "=" * 70)
    logger.info("Phase 6 Evaluation Summary")
    logger.info("=" * 70)

    summary_table = """
| Population | Metric | Route A (VIME) | Route B (Raw) | Winner |
|------------|--------|----------------|---------------|--------|
| D_L_test | AUC-ROC | {a_dl_auc:.4f} | {b_dl_auc:.4f} | {dl_winner} |
| D_L_test | PR-AUC | {a_dl_pr:.4f} | {b_dl_pr:.4f} | {dl_pr_winner} |
| D_R | AUC-ROC | {a_dr_auc:.4f} | {b_dr_auc:.4f} | {dr_winner} |
| D_R | PR-AUC | {a_dr_pr:.4f} | {b_dr_pr:.4f} | {dr_pr_winner} |
| Full | AUC-ROC | {a_full_auc:.4f} | {b_full_auc:.4f} | {full_winner} |
| D_R | Precision@20% | {a_prec20:.1%} | {b_prec20:.1%} | {prec_winner} |
| D_R | Diamonds@20% | {a_dia20:,} | {b_dia20:,} | {dia_winner} |
""".format(
        a_dl_auc=results["route_a_dl_test"]["auc_roc"],
        b_dl_auc=results["route_b_dl_test"]["auc_roc"],
        dl_winner="B" if results["route_b_dl_test"]["auc_roc"] > results["route_a_dl_test"]["auc_roc"] else "A",
        a_dl_pr=results["route_a_dl_test"]["pr_auc"],
        b_dl_pr=results["route_b_dl_test"]["pr_auc"],
        dl_pr_winner="B" if results["route_b_dl_test"]["pr_auc"] > results["route_a_dl_test"]["pr_auc"] else "A",
        a_dr_auc=results["route_a_dr"]["auc_roc"],
        b_dr_auc=results["route_b_dr"]["auc_roc"],
        dr_winner="B" if results["route_b_dr"]["auc_roc"] > results["route_a_dr"]["auc_roc"] else "A",
        a_dr_pr=results["route_a_dr"]["pr_auc"],
        b_dr_pr=results["route_b_dr"]["pr_auc"],
        dr_pr_winner="B" if results["route_b_dr"]["pr_auc"] > results["route_a_dr"]["pr_auc"] else "A",
        a_full_auc=results["route_a_full"]["auc_roc"],
        b_full_auc=results["route_b_full"]["auc_roc"],
        full_winner="B" if results["route_b_full"]["auc_roc"] > results["route_a_full"]["auc_roc"] else "A",
        a_prec20=diamond_a["20"]["precision"],
        b_prec20=diamond_b["20"]["precision"],
        prec_winner="B" if diamond_b["20"]["precision"] > diamond_a["20"]["precision"] else "A",
        a_dia20=diamond_a["20"]["diamonds_found"],
        b_dia20=diamond_b["20"]["diamonds_found"],
        dia_winner="B" if diamond_b["20"]["diamonds_found"] > diamond_a["20"]["diamonds_found"] else "A",
    )

    logger.info(summary_table)

    # Save results
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULT_DIR / "phase6_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")

    # Save summary table as markdown
    summary_path = config.RESULT_DIR / "phase6_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Phase 6 Evaluation Summary\n\n")
        f.write(summary_table)
    logger.info(f"Summary saved to {summary_path}")

    return results


if __name__ == "__main__":
    results = run_phase6_evaluation()
