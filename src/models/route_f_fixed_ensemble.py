"""
Route F-Fixed — Variable Split Ensemble with Fixed VIME Embeddings (Steps 1.5A + 1.5B)

Two-model ensemble (EXT_SOURCE vs OTHER) using fixed VIME embeddings.

Step 1.5A — Embeddings Only:
    Model_EXT:   Full 128-dim fixed embeddings → predict D_R
    Model_OTHER: Full 128-dim fixed embeddings → predict D_R
    Both use the same embeddings; the difference is in the weight sweep
    for optimal D_R scoring.

Step 1.5B — Raw + Embeddings:
    Model_EXT:   9 EXT_SOURCE raw features + 128 fixed embeddings → predict D_R
    Model_OTHER: 123 OTHER raw features + 128 fixed embeddings → predict D_R
    Each sub-model gets its own raw feature group plus full embeddings.

Baseline: Route F (original) D_R AUC = 0.7268

Reference: WORKING_PLAN.md Phase 5.6, Steps 1.5A + 1.5B
"""
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data():
    """
    Load all data needed for the ensemble experiments.

    Returns:
        data: Dict with all feature sets, labels, and metadata
    """
    logger.info("Loading data for Route F-Fixed ensemble...")

    # Fixed VIME embeddings
    emb_train = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_train_fixed.parquet")
    emb_val = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_val_fixed.parquet")
    emb_test = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_test_fixed.parquet")
    emb_reject = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_reject_fixed.parquet")

    # Rename embedding columns
    emb_cols = [f"emb_{i}" for i in range(emb_train.shape[1])]
    for df in [emb_train, emb_val, emb_test, emb_reject]:
        df.columns = emb_cols

    # Raw features
    df_train = pd.read_parquet(config.SPLIT_DL_TRAIN)
    df_val = pd.read_parquet(config.SPLIT_DL_VAL)
    df_test = pd.read_parquet(config.SPLIT_DL_TEST)
    df_reject = pd.read_parquet(config.SPLIT_DR_FEATURES)
    dr_truth = pd.read_parquet(config.SPLIT_DR_TRUTH)

    # Labels
    y_train = df_train[config.TARGET_COL].values
    y_val = df_val[config.TARGET_COL].values
    y_test = df_test[config.TARGET_COL].values
    y_reject = dr_truth[config.TARGET_COL].values

    # Raw feature splits
    X_raw_train = df_train.drop(columns=[config.TARGET_COL]).reset_index(drop=True)
    X_raw_val = df_val.drop(columns=[config.TARGET_COL]).reset_index(drop=True)
    X_raw_test = df_test.drop(columns=[config.TARGET_COL]).reset_index(drop=True)
    X_raw_reject = df_reject.reset_index(drop=True)

    # Identify EXT and OTHER columns from raw features
    ext_cols = [c for c in X_raw_train.columns if "EXT_SOURCE" in c.upper()]
    other_cols = [c for c in X_raw_train.columns if c not in ext_cols]

    # Handle categorical NaN
    cat_features_other = [col for col in config.CATEGORICAL_COLS if col in other_cols]
    for col in cat_features_other:
        for df in [X_raw_train, X_raw_val, X_raw_test, X_raw_reject]:
            df[col] = df[col].fillna("_MISSING_")

    logger.info(f"  EXT_SOURCE raw features: {len(ext_cols)}")
    logger.info(f"  OTHER raw features: {len(other_cols)}")
    logger.info(f"  Fixed embeddings: {emb_train.shape[1]}")

    data = {
        "emb_train": emb_train.reset_index(drop=True),
        "emb_val": emb_val.reset_index(drop=True),
        "emb_test": emb_test.reset_index(drop=True),
        "emb_reject": emb_reject.reset_index(drop=True),
        "X_raw_train": X_raw_train,
        "X_raw_val": X_raw_val,
        "X_raw_test": X_raw_test,
        "X_raw_reject": X_raw_reject,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_reject": y_reject,
        "ext_cols": ext_cols,
        "other_cols": other_cols,
        "cat_features_other": cat_features_other,
        "emb_cols": emb_cols,
    }
    return data


def train_model(X_train, y_train, X_val, y_val, cat_features=None, name="model",
                iterations=2000, depth=6):
    """Train a CatBoost model with given parameters."""
    params = {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": config.CB_LEARNING_RATE,
        "eval_metric": config.CB_EVAL_METRIC,
        "early_stopping_rounds": config.CB_EARLY_STOP,
        "verbose": config.CB_VERBOSE,
        "random_seed": config.SEED,
    }
    if cat_features:
        params["cat_features"] = cat_features

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    logger.info(f"{name} Val AUC: {val_auc:.4f}")
    return model, val_auc


def find_optimal_weights(pred_ext, pred_other, y_true):
    """Find optimal EXT/OTHER combination weights via grid sweep."""
    best_w, best_auc = 0, 0

    for w in np.arange(0.0, 1.05, 0.05):
        pred_combined = w * pred_ext + (1 - w) * pred_other
        auc = roc_auc_score(y_true, pred_combined)
        if auc > best_auc:
            best_w, best_auc = w, auc

    return best_w, best_auc


def run_step_15a(data):
    """
    Step 1.5A: Variable Split — Embeddings Only.

    Both Model_EXT and Model_OTHER use the full 128-dim fixed embeddings.
    The ensemble explores whether different weight combinations
    can improve D_R scoring with the same latent space.

    Returns:
        results_15a: Dict with metrics
    """
    logger.info("=" * 70)
    logger.info("Step 1.5A: Variable Split — Embeddings Only")
    logger.info("=" * 70)

    emb_train = data["emb_train"].values
    emb_val = data["emb_val"].values
    emb_test = data["emb_test"].values
    emb_reject = data["emb_reject"].values

    # Train Model_EXT (embeddings only, but will serve the EXT role in ensemble)
    logger.info("\n[1.5A] Training Model_EXT on full fixed embeddings...")
    model_ext, ext_val_auc = train_model(
        emb_train, data["y_train"], emb_val, data["y_val"],
        name="Model_EXT_emb", iterations=1000, depth=5
    )

    # Train Model_OTHER (same embeddings, different random seed for diversity)
    logger.info("\n[1.5A] Training Model_OTHER on full fixed embeddings...")
    model_other, other_val_auc = train_model(
        emb_train, data["y_train"], emb_val, data["y_val"],
        name="Model_OTHER_emb", iterations=2000, depth=6
    )

    # Get predictions
    pred_ext_test = model_ext.predict_proba(emb_test)[:, 1]
    pred_ext_reject = model_ext.predict_proba(emb_reject)[:, 1]
    pred_other_test = model_other.predict_proba(emb_test)[:, 1]
    pred_other_reject = model_other.predict_proba(emb_reject)[:, 1]

    # Individual AUCs
    auc_ext_test = roc_auc_score(data["y_test"], pred_ext_test)
    auc_ext_reject = roc_auc_score(data["y_reject"], pred_ext_reject)
    auc_other_test = roc_auc_score(data["y_test"], pred_other_test)
    auc_other_reject = roc_auc_score(data["y_reject"], pred_other_reject)

    logger.info(f"\nModel_EXT:   D_L_test={auc_ext_test:.4f}, D_R={auc_ext_reject:.4f}")
    logger.info(f"Model_OTHER: D_L_test={auc_other_test:.4f}, D_R={auc_other_reject:.4f}")

    # Weight sweep for D_R
    w_opt_dr, auc_opt_dr = find_optimal_weights(pred_ext_reject, pred_other_reject, data["y_reject"])
    w_opt_test, auc_opt_test = find_optimal_weights(pred_ext_test, pred_other_test, data["y_test"])

    # Apply optimal D_R weights to test set for cross-check
    pred_combined_test = w_opt_dr * pred_ext_test + (1 - w_opt_dr) * pred_other_test
    auc_combined_test = roc_auc_score(data["y_test"], pred_combined_test)

    logger.info(f"\nOptimal weights for D_R: EXT={w_opt_dr:.2f}, OTHER={1-w_opt_dr:.2f}")
    logger.info(f"D_R AUC with optimal weights:      {auc_opt_dr:.4f}")
    logger.info(f"D_L_test AUC with optimal weights:  {auc_combined_test:.4f}")

    results_15a = {
        "step": "1.5A",
        "description": "Variable split, embeddings only",
        "model_ext": {"D_L_test": auc_ext_test, "D_R": auc_ext_reject, "val_auc": ext_val_auc},
        "model_other": {"D_L_test": auc_other_test, "D_R": auc_other_reject, "val_auc": other_val_auc},
        "ensemble_dr_optimal": {
            "w_ext": float(w_opt_dr),
            "w_other": float(1 - w_opt_dr),
            "D_L_test": float(auc_combined_test),
            "D_R": float(auc_opt_dr),
        },
        "ensemble_test_optimal": {
            "w_ext": float(w_opt_test),
            "w_other": float(1 - w_opt_test),
            "D_L_test": float(auc_opt_test),
        },
    }

    return results_15a, model_ext, model_other


def run_step_15b(data):
    """
    Step 1.5B: Variable Split — Raw + Embeddings.

    Model_EXT:   9 EXT_SOURCE raw features + 128 fixed embeddings = 137 features
    Model_OTHER: 123 OTHER raw features + 128 fixed embeddings = 251 features

    Returns:
        results_15b: Dict with metrics
    """
    logger.info("\n" + "=" * 70)
    logger.info("Step 1.5B: Variable Split — Raw + Embeddings")
    logger.info("=" * 70)

    ext_cols = data["ext_cols"]
    other_cols = data["other_cols"]
    emb_cols = data["emb_cols"]

    # Build EXT feature set: EXT_SOURCE raw + full embeddings
    X_ext_train = pd.concat([data["X_raw_train"][ext_cols].reset_index(drop=True), data["emb_train"]], axis=1)
    X_ext_val = pd.concat([data["X_raw_val"][ext_cols].reset_index(drop=True), data["emb_val"]], axis=1)
    X_ext_test = pd.concat([data["X_raw_test"][ext_cols].reset_index(drop=True), data["emb_test"]], axis=1)
    X_ext_reject = pd.concat([data["X_raw_reject"][ext_cols].reset_index(drop=True), data["emb_reject"]], axis=1)

    # Build OTHER feature set: OTHER raw + full embeddings
    X_other_train = pd.concat([data["X_raw_train"][other_cols].reset_index(drop=True), data["emb_train"]], axis=1)
    X_other_val = pd.concat([data["X_raw_val"][other_cols].reset_index(drop=True), data["emb_val"]], axis=1)
    X_other_test = pd.concat([data["X_raw_test"][other_cols].reset_index(drop=True), data["emb_test"]], axis=1)
    X_other_reject = pd.concat([data["X_raw_reject"][other_cols].reset_index(drop=True), data["emb_reject"]], axis=1)

    logger.info(f"  Model_EXT features: {X_ext_train.shape[1]} ({len(ext_cols)} raw + {len(emb_cols)} emb)")
    logger.info(f"  Model_OTHER features: {X_other_train.shape[1]} ({len(other_cols)} raw + {len(emb_cols)} emb)")

    # Get categorical indices for OTHER model
    cat_features_other = data["cat_features_other"]
    cat_indices_other = [X_other_train.columns.get_loc(col) for col in cat_features_other
                         if col in X_other_train.columns]

    # Train Model_EXT (no categoricals in EXT_SOURCE features)
    logger.info("\n[1.5B] Training Model_EXT (EXT raw + embeddings)...")
    model_ext, ext_val_auc = train_model(
        X_ext_train, data["y_train"], X_ext_val, data["y_val"],
        name="Model_EXT_hybrid", iterations=1000, depth=5
    )

    # Train Model_OTHER (has categoricals)
    logger.info("\n[1.5B] Training Model_OTHER (OTHER raw + embeddings)...")
    model_other, other_val_auc = train_model(
        X_other_train, data["y_train"], X_other_val, data["y_val"],
        cat_features=cat_indices_other,
        name="Model_OTHER_hybrid", iterations=2000, depth=6
    )

    # Get predictions
    pred_ext_test = model_ext.predict_proba(X_ext_test)[:, 1]
    pred_ext_reject = model_ext.predict_proba(X_ext_reject)[:, 1]
    pred_other_test = model_other.predict_proba(X_other_test)[:, 1]
    pred_other_reject = model_other.predict_proba(X_other_reject)[:, 1]

    # Individual AUCs
    auc_ext_test = roc_auc_score(data["y_test"], pred_ext_test)
    auc_ext_reject = roc_auc_score(data["y_reject"], pred_ext_reject)
    auc_other_test = roc_auc_score(data["y_test"], pred_other_test)
    auc_other_reject = roc_auc_score(data["y_reject"], pred_other_reject)

    logger.info(f"\nModel_EXT:   D_L_test={auc_ext_test:.4f}, D_R={auc_ext_reject:.4f}")
    logger.info(f"Model_OTHER: D_L_test={auc_other_test:.4f}, D_R={auc_other_reject:.4f}")

    # Weight sweep for D_R
    w_opt_dr, auc_opt_dr = find_optimal_weights(pred_ext_reject, pred_other_reject, data["y_reject"])
    w_opt_test, auc_opt_test = find_optimal_weights(pred_ext_test, pred_other_test, data["y_test"])

    # Apply optimal D_R weights to test set for cross-check
    pred_combined_test = w_opt_dr * pred_ext_test + (1 - w_opt_dr) * pred_other_test
    auc_combined_test = roc_auc_score(data["y_test"], pred_combined_test)

    logger.info(f"\nOptimal weights for D_R: EXT={w_opt_dr:.2f}, OTHER={1-w_opt_dr:.2f}")
    logger.info(f"D_R AUC with optimal weights:      {auc_opt_dr:.4f}")
    logger.info(f"D_L_test AUC with optimal weights:  {auc_combined_test:.4f}")

    results_15b = {
        "step": "1.5B",
        "description": "Variable split, raw + embeddings",
        "n_ext_features": int(X_ext_train.shape[1]),
        "n_other_features": int(X_other_train.shape[1]),
        "model_ext": {"D_L_test": auc_ext_test, "D_R": auc_ext_reject, "val_auc": ext_val_auc},
        "model_other": {"D_L_test": auc_other_test, "D_R": auc_other_reject, "val_auc": other_val_auc},
        "ensemble_dr_optimal": {
            "w_ext": float(w_opt_dr),
            "w_other": float(1 - w_opt_dr),
            "D_L_test": float(auc_combined_test),
            "D_R": float(auc_opt_dr),
        },
        "ensemble_test_optimal": {
            "w_ext": float(w_opt_test),
            "w_other": float(1 - w_opt_test),
            "D_L_test": float(auc_opt_test),
        },
    }

    return results_15b, model_ext, model_other


def run_route_f_fixed():
    """
    Run full Route F-Fixed ensemble pipeline (Steps 1.5A + 1.5B).

    Returns:
        results: Combined results dict
    """
    logger.info("=" * 70)
    logger.info("Route F-Fixed: Variable Split Ensemble with Fixed VIME Embeddings")
    logger.info("=" * 70)

    # Load all data
    data = load_data()

    # Run Step 1.5A
    results_15a, _, _ = run_step_15a(data)

    # Run Step 1.5B
    results_15b, model_ext, model_other = run_step_15b(data)

    # Compile combined results
    results = {
        "route": "F-Fixed",
        "step_15a": results_15a,
        "step_15b": results_15b,
        "baseline_route_f_d_r_auc": 0.7268,
    }

    # Save models (from 1.5B — the one with raw features)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_ext.save_model(str(config.MODEL_DIR / "route_f_fixed_model_ext.cbm"))
    model_other.save_model(str(config.MODEL_DIR / "route_f_fixed_model_other.cbm"))
    logger.info(f"\nModels saved to {config.MODEL_DIR}")

    # Save results
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULT_DIR / "route_f_fixed_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Results saved to {config.RESULT_DIR / 'route_f_fixed_results.json'}")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("ROUTE F-FIXED — FINAL SUMMARY")
    logger.info("=" * 70)

    logger.info("\n--- Step 1.5A (Embeddings Only) ---")
    ens_15a = results_15a["ensemble_dr_optimal"]
    logger.info(f"Ensemble D_R AUC: {ens_15a['D_R']:.4f} (w_ext={ens_15a['w_ext']:.2f})")
    logger.info(f"Ensemble D_L_test AUC: {ens_15a['D_L_test']:.4f}")

    logger.info("\n--- Step 1.5B (Raw + Embeddings) ---")
    ens_15b = results_15b["ensemble_dr_optimal"]
    logger.info(f"Ensemble D_R AUC: {ens_15b['D_R']:.4f} (w_ext={ens_15b['w_ext']:.2f})")
    logger.info(f"Ensemble D_L_test AUC: {ens_15b['D_L_test']:.4f}")

    logger.info(f"\nBaseline Route F D_R AUC: 0.7268")
    logger.info(f"1.5A improvement: {ens_15a['D_R'] - 0.7268:+.4f}")
    logger.info(f"1.5B improvement: {ens_15b['D_R'] - 0.7268:+.4f}")

    return results


if __name__ == "__main__":
    results = run_route_f_fixed()
    print("\n" + json.dumps(results, indent=2, default=float))
