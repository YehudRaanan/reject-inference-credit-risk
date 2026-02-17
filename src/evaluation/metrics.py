"""
Evaluation Metrics for Credit Risk Models.

Implements:
- AUC-ROC: Overall discrimination ability
- Precision@K%: Among top K% lowest-risk, what fraction are good?
- Diamond Recovery: Count of good borrowers found in top K%
- Bootstrap Confidence Intervals
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def precision_at_k(y_true, y_pred, k=0.20):
    """
    Precision@k%: Among top k% lowest-risk predictions, what fraction are truly good?

    For credit risk, we want to identify applicants LEAST likely to default.
    Lower predicted probability = lower risk = select these as "good".

    Args:
        y_true: True labels (1=default, 0=non-default)
        y_pred: Predicted probabilities of default
        k: Top k% to evaluate (default 20%)

    Returns:
        precision: Fraction of selected that are truly good (y=0)
        n_diamonds: Count of good borrowers found ("diamonds in the mud")
    """
    n_top_k = int(len(y_true) * k)
    # Sort by predicted probability (ascending = lowest risk first)
    sorted_indices = np.argsort(y_pred)
    top_k_indices = sorted_indices[:n_top_k]
    top_k_labels = y_true[top_k_indices]

    # Count "good" borrowers (y=0) in top k%
    n_diamonds = (top_k_labels == 0).sum()
    precision = n_diamonds / n_top_k

    return precision, int(n_diamonds)


def evaluate_model(y_true, y_pred, k=0.20, name="Model"):
    """
    Complete evaluation of a credit risk model.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        k: Top k% for precision evaluation
        name: Model name for logging

    Returns:
        metrics: Dict with AUC, P@K, and Diamonds
    """
    auc = roc_auc_score(y_true, y_pred)
    precision, n_diamonds = precision_at_k(y_true, y_pred, k)

    metrics = {
        'AUC': float(auc),
        f'P@{int(k*100)}%': float(precision),
        'Diamonds': int(n_diamonds)
    }

    return metrics


def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, k=0.20, seed=42):
    """
    Compute bootstrap 95% confidence intervals for model metrics.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        n_bootstrap: Number of bootstrap resamples
        k: Top k% for precision evaluation
        seed: Random seed for reproducibility

    Returns:
        results: Dict with point estimates and CIs for each metric
    """
    np.random.seed(seed)
    n = len(y_true)

    # Point estimates
    auc_point = roc_auc_score(y_true, y_pred)
    prec_point, diamonds_point = precision_at_k(y_true, y_pred, k)

    # Bootstrap samples
    auc_samples = []
    prec_samples = []
    diamonds_samples = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]

        # Skip samples with single class
        if len(np.unique(y_true_sample)) < 2:
            continue

        auc_samples.append(roc_auc_score(y_true_sample, y_pred_sample))
        prec, diamonds = precision_at_k(y_true_sample, y_pred_sample, k)
        prec_samples.append(prec)
        diamonds_samples.append(diamonds)

    # 95% confidence intervals
    results = {
        'AUC': {
            'point': float(auc_point),
            'ci_low': float(np.percentile(auc_samples, 2.5)),
            'ci_high': float(np.percentile(auc_samples, 97.5)),
            'std': float(np.std(auc_samples))
        },
        f'P@{int(k*100)}%': {
            'point': float(prec_point),
            'ci_low': float(np.percentile(prec_samples, 2.5)),
            'ci_high': float(np.percentile(prec_samples, 97.5)),
            'std': float(np.std(prec_samples))
        },
        'Diamonds': {
            'point': int(diamonds_point),
            'ci_low': int(np.percentile(diamonds_samples, 2.5)),
            'ci_high': int(np.percentile(diamonds_samples, 97.5)),
            'std': float(np.std(diamonds_samples))
        }
    }

    return results
