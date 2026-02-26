# Reject Inference Methodological Comparison

**Deep Learning Final Exercise** — Comparing a VIME-based hybrid reject inference approach against a classical CatBoost baseline on the **Home Credit Default Risk** dataset.

## The Problem

In traditional banking, rejected applicants have no observed outcome — their creditworthiness is unknown. **Reject inference** attempts to recover this information and identify **"diamonds in the mud"**: good borrowers who were wrongly declined by a simple score threshold.

## Approach

| Route | Approach | Key Technique |
|-------|----------|---------------|
| **A** (Hybrid) | VIME self-supervised encoder → embeddings + CatBoost teacher → Isolation Forest filtering | Deep Learning |
| **B** (Classical) | CatBoost on raw features → direct pseudo-labeling | Baseline |

## Dataset

- **Source**: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle)
- **Size**: 307,511 application records, 122 features
- **Reject simulation**: Bottom 30% by `EXT_SOURCE_2` score → "rejected" (D_R)
- **Evaluation**: Three populations — approved, rejected (💎 diamond recovery), full

## Data & Preprocessing

**Source Repository**: [js-aguiar/home-credit-default-competition](https://github.com/js-aguiar/home-credit-default-competition) (7th Place Solution)
- We strictly follow the feature engineering (ratios, aggregations) from this repository to ensure robust signal recovery.

**VIME Specifics (Phase 2)**:
- **Imputation**: Median/Mode (Must be complete input for VIME masking).
- **Encoding**: One-Hot Encoding (No target leakage allowed).
- **Scaling**: Min-Max Scaler [0, 1] (Required for VIME reconstruction loss).

## Setup

```bash
pip install -r requirements.txt
```

Place `application_train.csv` (and optionally `bureau.csv`) into `data/raw/`.

## Project Structure

```
config.py              # Central configuration (paths, splits, hyperparams)
run_experiment.py      # End-to-end pipeline
src/
  preprocessing/       # Shared + route-specific data pipelines
  models/
    vime/              # Self-supervised encoder (Route A)
    route_a_hybrid.py  # Hybrid reject inference
    route_b_classical.py # Classical baseline
  evaluation/          # Metrics, SHAP analysis
notebooks/             # EDA
outputs/               # Models, figures, results
docs/                  # Working plan, decision log
```

## How to Run

```bash
python run_experiment.py
```

## Evaluation Metrics

- **D_L_test**: AUC-ROC, PR-AUC (approved population)
- **D_R_ground_truth**: AUC-ROC, Precision@k, Recall of good borrowers, Lift (💎 diamond recovery)
- **Full population**: AUC-ROC, AUK, PSI, SHAP comparison

## Reference

- **Data Splitting Strategy**: See `docs/data_splitting_proposal.md`
- **Working Plan**: See `docs/WORKING_PLAN.md`
- **Decision Log**: See `docs/decision_log.md`
