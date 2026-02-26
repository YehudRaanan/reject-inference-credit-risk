"""
Evaluation Metrics (Step 5).

Provides:
- AUC-ROC
- Precision-Recall AUC
- AUK (Area Under the Kickout curve)
- PSI (Population Stability Index)
- Comparison table generation
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


def compute_auc_roc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute Area Under the ROC Curve."""
    return roc_auc_score(y_true, y_proba)


def compute_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve."""
    return average_precision_score(y_true, y_proba)


def compute_auk(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Compute Area Under the Kickout (AUK) curve.
    
    The kickout curve measures: as we reject the riskiest X% of applicants,
    what fraction of actual defaults do we catch?
    Higher AUK = better ability to identify risky 'grey area' applicants.
    """
    # Sort by predicted risk (descending)
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    total_defaults = y_sorted.sum()

    if total_defaults == 0:
        return 0.0

    kickout_fractions = np.linspace(0, 1, n_bins + 1)[1:]
    caught_fractions = []

    for frac in kickout_fractions:
        n_kicked = int(frac * len(y_sorted))
        caught = y_sorted[:n_kicked].sum() / total_defaults
        caught_fractions.append(caught)

    # Area under the curve via trapezoidal rule
    auk = np.trapz(caught_fractions, kickout_fractions)
    return auk


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index (PSI).
    
    Measures distribution shift between accepted population (expected)
    and pseudo-labeled rejected population (actual).
    
    PSI < 0.10: no significant shift
    PSI 0.10-0.25: moderate shift (investigate)
    PSI > 0.25: significant shift (action needed)
    """
    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to proportions (with small epsilon to avoid log(0))
    eps = 1e-6
    expected_pct = expected_counts / expected_counts.sum() + eps
    actual_pct = actual_counts / actual_counts.sum() + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def comparison_table(
    y_true: np.ndarray,
    route_a_proba: np.ndarray,
    route_b_proba: np.ndarray,
) -> pd.DataFrame:
    """
    Generate a head-to-head comparison table of Route A vs Route B.
    """
    results = {
        "Metric": ["AUC-ROC", "PR-AUC", "AUK"],
        "Route A (Hybrid)": [
            compute_auc_roc(y_true, route_a_proba),
            compute_pr_auc(y_true, route_a_proba),
            compute_auk(y_true, route_a_proba),
        ],
        "Route B (Classical)": [
            compute_auc_roc(y_true, route_b_proba),
            compute_pr_auc(y_true, route_b_proba),
            compute_auk(y_true, route_b_proba),
        ],
    }

    df = pd.DataFrame(results)
    df["Δ (A − B)"] = df["Route A (Hybrid)"] - df["Route B (Classical)"]
    return df
