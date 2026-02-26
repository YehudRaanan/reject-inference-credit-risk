"""
Reject Inference Methodological Comparison — Main Experiment Runner.

End-to-end pipeline:
  1. Load & preprocess data (shared)
  2. Fork into Route A / Route B preprocessing
  3. Train VIME (self-supervised) & extract embeddings
  4. Run Route A (hybrid) and Route B (classical)
  5. Evaluate & compare on hold-out set
"""
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.utils import set_seed, get_device, log_step
from src.preprocessing.shared_pipeline import run_shared_pipeline, split_financed_rejected
from src.preprocessing.route_a_prep import prepare_route_a
from src.preprocessing.route_b_prep import prepare_route_b
from src.models.vime.encoder import VIMEModel
from src.models.vime.self_supervised import train_vime
from src.models.vime.embeddings import extract_embeddings
from src.models.route_a_hybrid import run_route_a
from src.models.route_b_classical import run_route_b
from src.evaluation.metrics import comparison_table, compute_psi


def main():
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_seed(config.SEED)
    config.ensure_dirs()
    device = get_device()
    log_step("Setup", f"Seed={config.SEED}, Device={device}")

    # ------------------------------------------------------------------
    # Step 1: Load raw data
    # ------------------------------------------------------------------
    log_step("Data Loading", "Looking for CSV files in data/raw/")

    # TODO: Update filename to match your actual dataset file
    raw_path = config.DATA_RAW / "kowope_mart.csv"
    if not raw_path.exists():
        print(f"ERROR: Dataset not found at {raw_path}")
        print("Please place the raw Kowope Mart CSV in data/raw/")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    log_step("Data Loading", f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ------------------------------------------------------------------
    # Step 1.1–1.2: Shared preprocessing
    # ------------------------------------------------------------------
    log_step("Shared Preprocessing")
    df = run_shared_pipeline(df)

    # ------------------------------------------------------------------
    # Step 1.3: Split D_L / D_R
    # ------------------------------------------------------------------
    log_step("Data Splitting")
    d_l, d_r = split_financed_rejected(df)
    log_step("Data Splitting", f"D_L={len(d_l)}, D_R={len(d_r)}")

    # Hold-out from D_L
    d_l_train, d_l_test = train_test_split(
        d_l, test_size=config.TEST_SIZE, random_state=config.SEED,
        stratify=d_l["Good_Bad"]
    )
    log_step("Hold-out", f"Train={len(d_l_train)}, Test={len(d_l_test)}")

    # ------------------------------------------------------------------
    # Identify column types (TODO: adjust after EDA)
    # ------------------------------------------------------------------
    target_col = "Good_Bad"
    exclude_cols = [target_col]
    feature_cols = [c for c in d_l_train.columns if c not in exclude_cols]
    numerical_cols = d_l_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numerical_cols]

    y_train = d_l_train[target_col].values
    y_test = d_l_test[target_col].values

    # ==================================================================
    # ROUTE A: VIME Hybrid
    # ==================================================================
    log_step("Route A", "Preparing data for VIME")

    X_a_train, scaler = prepare_route_a(d_l_train[feature_cols], numerical_cols, fit=True)
    X_a_test, _ = prepare_route_a(d_l_test[feature_cols], numerical_cols, scaler=scaler, fit=False)
    X_a_rejected, _ = prepare_route_a(d_r[feature_cols], numerical_cols, scaler=scaler, fit=False)

    # Combine D_L + D_R for self-supervised VIME training
    all_route_a = np.vstack([X_a_train.values, X_a_rejected.values])

    # Train VIME
    log_step("Route A", "Training VIME (self-supervised)")
    input_dim = all_route_a.shape[1]
    vime_model = VIMEModel(input_dim, embed_dim=config.VIME["encoder_dim"])
    history = train_vime(
        vime_model, all_route_a,
        epochs=config.VIME["epochs"],
        batch_size=config.VIME["batch_size"],
        learning_rate=config.VIME["learning_rate"],
        corruption_rate=config.VIME["corruption_rate"],
        device=device,
    )

    # Extract embeddings
    log_step("Route A", "Extracting embeddings")
    emb_train = extract_embeddings(vime_model, X_a_train.values, device=device)
    emb_test = extract_embeddings(vime_model, X_a_test.values, device=device)
    emb_rejected = extract_embeddings(vime_model, X_a_rejected.values, device=device)

    # Run Route A pipeline
    log_step("Route A", "Training hybrid model")
    model_a = run_route_a(X_a_train.values, y_train, X_a_rejected.values, emb_train, emb_rejected)

    # Predict on test set
    X_test_full = np.hstack([X_a_test.values, emb_test])
    proba_a = model_a.predict_proba(X_test_full)[:, 1]

    # ==================================================================
    # ROUTE B: Classical Baseline
    # ==================================================================
    log_step("Route B", "Preparing data for classical baseline")
    X_b_train = prepare_route_b(d_l_train[feature_cols], y_train, categorical_cols)
    X_b_test = prepare_route_b(d_l_test[feature_cols], categorical_cols=categorical_cols)
    X_b_rejected = prepare_route_b(d_r[feature_cols], categorical_cols=categorical_cols)

    log_step("Route B", "Training classical model")
    model_b = run_route_b(X_b_train.values, y_train, X_b_rejected.values)
    proba_b = model_b.predict_proba(X_b_test.values)[:, 1]

    # ==================================================================
    # STEP 5: Evaluation & Comparison
    # ==================================================================
    log_step("Evaluation", "Computing metrics")

    results = comparison_table(y_test, proba_a, proba_b)
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 60)
    print(results.to_string(index=False))

    # PSI check
    psi_a = compute_psi(proba_a, model_a.predict_proba(
        np.hstack([X_a_rejected.values, emb_rejected]))[:, 1])
    psi_b = compute_psi(proba_b, model_b.predict_proba(X_b_rejected.values)[:, 1])
    print(f"\nPSI — Route A: {psi_a:.4f}, Route B: {psi_b:.4f}")

    # Save results
    results.to_csv(config.RESULTS_DIR / "comparison_results.csv", index=False)
    log_step("Done", f"Results saved to {config.RESULTS_DIR / 'comparison_results.csv'}")


if __name__ == "__main__":
    main()
