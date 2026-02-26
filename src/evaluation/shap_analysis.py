"""
SHAP Analysis (Step 5).

Compare feature importances between Route A (hybrid with embeddings)
and Route B (classical tabular only).
"""
import numpy as np
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from pathlib import Path


def compute_shap_values(
    model: CatBoostClassifier,
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> shap.Explanation:
    """Compute SHAP values for a CatBoost model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    if feature_names is not None:
        shap_values.feature_names = feature_names

    return shap_values


def plot_comparison(
    shap_a: shap.Explanation,
    shap_b: shap.Explanation,
    save_dir: Path | None = None,
    top_n: int = 15,
) -> None:
    """
    Side-by-side SHAP summary plots for Route A vs Route B.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    plt.sca(axes[0])
    shap.summary_plot(shap_a, show=False, max_display=top_n)
    axes[0].set_title("Route A — Hybrid (Features + Embeddings)", fontsize=12)

    plt.sca(axes[1])
    shap.summary_plot(shap_b, show=False, max_display=top_n)
    axes[1].set_title("Route B — Classical (Features Only)", fontsize=12)

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "shap_comparison.png", dpi=150, bbox_inches="tight")
        print(f"Saved SHAP comparison plot to {save_dir / 'shap_comparison.png'}")

    plt.show()
