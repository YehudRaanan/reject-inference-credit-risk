# Phase 3 Report: VIME Self-Supervised Training

> **Phase 3.5**: Learning Walkthrough
> **Phase 3.6**: Owner Code Review #4
> **Date**: 2026-02-10
> **Status**: Complete

---

## Part 1: Learning Walkthrough (Phase 3.5)

### 1.1 What is VIME?

**VIME** (Value Imputation and Mask Estimation) is a self-supervised learning framework for tabular data, introduced by Yoon et al. at NeurIPS 2020.

**Core Insight**: Unlike images (where augmentations like rotation/cropping work well), tabular data lacks natural augmentations. VIME creates a pretext task by **corrupting** features and training the model to:
1. **Detect** which features were corrupted (Mask Estimation)
2. **Reconstruct** the original values (Value Imputation)

This forces the encoder to learn meaningful feature relationships without needing labels.

### 1.2 Why VIME for Reject Inference?

In our reject inference problem:
- **D_L** (approved): Has labels, but only ~215k samples
- **D_R** (rejected): No labels, but ~93k samples with valuable signal

VIME allows us to:
1. Pre-train on **ALL** data (D_L + D_R) without labels
2. Learn representations that capture feature relationships in both populations
3. Transfer this knowledge to improve downstream classification

**Key Advantage**: The encoder sees the rejected population during pre-training, learning patterns that a purely supervised model on D_L would miss.

### 1.3 Architecture Deep Dive

```
Input (328 features)
        │
        ▼
┌───────────────────┐
│   ENCODER         │
│  FC1: 328 → 256   │──→ ReLU
│  FC2: 256 → 256   │──→ ReLU + Dropout(0.1)
│  FC3: 256 → 128   │──→ Latent Vector (z)
└───────────────────┘
        │
        ├──────────────────────┐
        ▼                      ▼
┌───────────────────┐  ┌───────────────────┐
│  MASK ESTIMATOR   │  │ RECONSTRUCTOR     │
│  FC: 128 → 328    │  │  FC: 128 → 328    │
│  Sigmoid          │  │  (Linear)         │
└───────────────────┘  └───────────────────┘
        │                      │
        ▼                      ▼
   m_hat ∈ [0,1]^328      x_hat ∈ ℝ^328
   (corruption prob)      (reconstructed)
```

**Why 128-dim latent space?**
- 328 → 128 is a ~2.5x compression
- Forces the model to learn compact, meaningful representations
- Empirically: larger dims = overfitting, smaller = underfitting

### 1.4 Self-Supervised Loss Design

The total loss combines two objectives:

```
L_total = α · L_mask + L_recon
```

Where:
- **L_mask** = BCE(m_hat, m_true) — Binary cross-entropy for mask prediction
- **L_recon** = MSE(x_hat, x_original) — Mean squared error for reconstruction
- **α = 2.0** — Weights mask loss higher (detecting corruption is harder)

**Why weight mask loss higher?**
- Reconstruction is "easier" — the model can just learn identity mapping
- Mask estimation requires understanding feature relationships
- α=2.0 forces the model to focus on the harder task

### 1.5 The Corruption Process

Standard VIME corrupts features by **shuffling** values within each column:

```python
# For each feature j:
x_tilde[:, j] = shuffle(x[:, j])  # if masked
```

**Our 3-Level Masking Strategy**:

| Level | Features | Mask Prob | Rationale |
|-------|----------|-----------|-----------|
| **1 (Critical)** | `AMT_INCOME_TOTAL` | 0.5 | Force learning of income relationships |
| **2 (High Missing)** | 62 cols with >50% missing | 0.1 | Already have missing indicators |
| **3 (Regular)** | 265 other cols | 0.3 | Standard VIME rate |

**Why not uniform masking?**
- Income is critical for credit risk — we want the model to learn it well
- High-missing columns have separate `missingindicator_*` features — no need to mask often
- This customization improved convergence stability

### 1.6 The Sparsity Problem & Smart Labeling

**Problem Discovered**: Our data is **64.6% sparse** (zeros from One-Hot encoding + missing indicators).

Standard corruption:
```
Original: [0, 0, 1, 0, 0]  (one-hot)
Shuffled: [0, 0, 0, 0, 1]
Result:   Often 0 → 0 (no actual change!)
```

If we label this as "corrupted" (m=1), the model gets confused — the value didn't actually change.

**Solution (Smart Labeling)**:
```python
# Only label as corrupted if value actually changed
is_changed = (abs(x - x_corrupted) > 1e-5).float()
mask = mask * is_changed
```

**Impact**:
| Metric | Before | After |
|--------|--------|-------|
| Mask Loss | 0.54 | 0.18 |
| Mask AUC | 0.638 | **0.9315** |

### 1.7 Why It Works for Reject Inference

1. **Shared Feature Space**: D_L and D_R have the same features. Pre-training on both teaches the encoder what "normal" looks like across the full population.

2. **Missing Pattern Learning**: VIME learns from missingness indicators — crucial since D_R may have different missing patterns than D_L.

3. **Representation Transfer**: The 128-dim embedding captures compressed feature relationships. The downstream CatBoost trains on these embeddings, not raw features.

4. **Domain Adaptation Effect**: By seeing D_R during pre-training, the encoder is less likely to produce out-of-distribution embeddings for rejected applicants.

---

## Part 2: Code Review (Phase 3.6)

### 2.1 Files Reviewed

| File | Purpose | Lines |
|------|---------|-------|
| `vime_model.py` | VIME architecture | 62 |
| `vime_trainer.py` | Training loop | 166 |
| `vime_utils.py` | Corruption & masking | 83 |
| `column_manager.py` | Feature grouping | 67 |
| `vime_embeddings.py` | Embedding extraction | 87 |
| `vime_evaluator.py` | Validation metrics | 113 |
| `route_a_prep.py` | VIME preprocessing | 165 |

### 2.2 Architecture Review (`vime_model.py`)

**Strengths**:
- Clean separation of `encode()` and `forward()` methods
- Proper use of Dropout for regularization
- Architecture matches VIME paper specification
- Reference to paper included in docstring

**Observations**:
- ~~Hidden dim (256) and embedding dim (128) were hardcoded~~ **FIXED**: Now uses `config.VIME_HIDDEN_DIM` and `config.VIME_EMBEDDING_DIM`
- Config updated to reflect actual trained model values (256/128)

### 2.3 Training Loop Review (`vime_trainer.py`)

**Strengths**:
- Proper fit-on-train-only philosophy (though currently trains on ALL data for SSL, which is correct)
- Logging at each epoch with loss breakdown
- Model saved after training
- Supports subsampling for debugging (`sample_size`)

**Observations**:
- No early stopping implemented — runs for fixed `epochs`
- No validation loss tracking during training (only final evaluation)
- ~~`alpha=2.0` hardcoded~~ **FIXED**: Now uses `config.VIME_ALPHA`

**Recommendation**: Add validation loss tracking and early stopping:
```python
# After each epoch, compute val loss
# If val_loss doesn't improve for VIME_PATIENCE epochs, stop
```

### 2.4 Corruption Logic Review (`vime_utils.py`)

**Strengths**:
- Smart labeling fix is elegant and effective
- Per-column mask probabilities correctly implemented
- Efficient tensor operations (no Python loops)

**The Smart Labeling Fix** (lines 75-80):
```python
is_changed = (torch.abs(x - x_corrupted) > 1e-5).float()
mask = mask * is_changed
```
This is the key fix that improved AUC from 0.638 → 0.9315.

**Observation**: Threshold of `1e-5` is reasonable for MinMax scaled [0,1] data. Could be parameterized if needed.

### 2.5 Column Manager Review (`column_manager.py`)

**Strengths**:
- Data-driven detection of high-missing columns
- Explicit handling of missing indicators
- Clear separation of groups

**Observation**: ~~Critical columns hardcoded~~ **FIXED**: Now uses `config.VIME_CRITICAL_FEATURES`

### 2.6 Embeddings Extraction Review (`vime_embeddings.py`)

**Strengths**:
- Batch processing to avoid OOM
- Preserves DataFrame index for traceability
- Clear column naming (`emb_0`, `emb_1`, ...)

**Observation**: Batch size hardcoded to 1024 — could come from config.

### 2.7 Preprocessing Review (`route_a_prep.py`)

**Strengths**:
- Comprehensive docstring explaining VIME requirements
- Correct fit-on-train-only approach
- Uses `add_indicator=True` for missingness signal
- One-Hot encoding prevents label leakage
- MinMax [0,1] scaling matches VIME sigmoid

**Observation**: No preprocessor persistence (pickle). If re-running only Phase 4+, would need to re-fit. Consider saving:
```python
with open(config.MODEL_DIR / "route_a_preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)
```

### 2.8 Summary of Code Quality

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Correctness** | Excellent | Smart labeling fix is key |
| **Documentation** | Good | Docstrings present, could add more inline comments |
| **Modularity** | Good | Clear separation of concerns |
| **Config Usage** | Excellent | All hyperparameters now from config (fixed 2026-02-10) |
| **Error Handling** | Fair | Basic file existence checks, could be more robust |
| **Testing** | Missing | No unit tests for VIME components |

### 2.9 Recommendations for Phase 4+

1. ~~**Config Consistency**~~: FIXED (2026-02-10) — All VIME hyperparameters now centralized in `config.py`

2. **Preprocessor Persistence**: Save fitted preprocessor for reproducibility.

3. **Early Stopping**: Add validation-based early stopping to training loop.

4. **Unit Tests**: Create `tests/test_vime.py` with:
   - Test corruption produces expected mask rates
   - Test smart labeling correctly handles sparse data
   - Test embedding dimensions match config

5. **Logging**: Save training logs to file for reproducibility.

---

## Part 3: Validation Results Summary

### Final VIME Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Reconstruction MSE** | 0.0157 | Excellent — low reconstruction error |
| **Mask Prediction AUC** | 0.9315 | Excellent — near-perfect corruption detection |

### Data Statistics

| Split | Rows | Features | Embeddings Generated |
|-------|------|----------|---------------------|
| Train | 150,357 | 328 | 128-dim |
| Val | 32,219 | 328 | 128-dim |
| Test | 32,220 | 328 | 128-dim |
| Reject (D_R) | 92,715 | 328 | 128-dim |

### Artifacts Produced

- `outputs/models/vime_encoder.pth` — Trained VIME model
- `data/processed/vime_emb_*.parquet` — Embeddings for all splits

---

## Phase 3 Checklist

- [x] **3.1** Implement VIME architecture
- [x] **3.2** Implement self-supervised training loop
- [x] **3.3** Implement embedding extraction
- [x] **3.4** QA Checks (reconstruction loss, embedding variance)
- [x] **3.5** Learning Walkthrough (this document)
- [x] **3.6** Owner Code Review #4 (this document)

---

## Next Steps

With Phase 3 complete, proceed to:

1. **Phase 4.1**: Train Teacher CatBoost on VIME embeddings (D_L_train)
2. **Phase 4.2**: Pseudo-label D_R using teacher predictions
3. **Phase 4.3**: Optional Isolation Forest filtering
4. **Phase 4.4**: Train final Route A model

---

*Report generated: 2026-02-10*
