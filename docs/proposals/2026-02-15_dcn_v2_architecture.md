# Proposal: DCN-v2 Architecture for Credit Risk Assessment

**Date**: 2026-02-15
**Status**: APPROVED
**Approved by**: Owner (2026-02-15)

---

## What

Implement Deep & Cross Network v2 (DCN-v2) to replace VIME as the neural network approach for reject inference, addressing the 128-dim information bottleneck that limited VIME to AUC ~0.716.

## Why

### Problem Statement
Phase 5.7 experiments conclusively demonstrated that VIME's performance ceiling (~0.716 AUC) is architectural, not methodological:
- Frozen encoder: 0.7166 AUC
- Fine-tuned encoder: 0.7163 AUC
- Route B (CatBoost, raw features): 0.7391 AUC

The 253→128 dimensional compression discards task-relevant information that cannot be recovered through fine-tuning.

### Why DCN-v2 Solves This
1. **No bottleneck**: Original features persist via residual connection x₀ at every cross layer
2. **Explicit interactions**: Cross Network efficiently computes polynomial feature interactions
3. **Native missingness**: Dedicated "Missing Bin" preserves missingness as a learnable signal
4. **Proven tabular performance**: State-of-the-art on CTR prediction and tabular benchmarks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DCN-v2 Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input: 253 features (numerical + categorical)             │
│                          ↓                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              SOFT BINNING LAYER                      │   │
│   │  • Learnable cutpoints per numerical feature        │   │
│   │  • Temperature decay (cold→hot during training)     │   │
│   │  • Dedicated Missing Bin (Bin 0) for NaN values     │   │
│   └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              EMBEDDING ENGINE                        │   │
│   │  • Categorical: Embedding(cardinality, sqrt(card))  │   │
│   │  • Numerical: Weighted sum of bin embeddings        │   │
│   │  • LayerNorm on concatenated x₀                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│                    x₀ (embedding vector)                     │
│                          ↓                                   │
│   ┌────────────────────┬────────────────────────────────┐   │
│   │   CROSS NETWORK    │       DEEP NETWORK             │   │
│   │   (3 layers)       │       (512→256→128)            │   │
│   │                    │                                 │   │
│   │   x_{l+1} =        │       Linear + GELU            │   │
│   │   x₀ ⊙ (W·x_l + b) │       + Dropout(0.15)          │   │
│   │   + x_l            │                                 │   │
│   │                    │                                 │   │
│   │   ↑ x₀ residual    │                                 │   │
│   │   at every layer   │                                 │   │
│   └────────────────────┴────────────────────────────────┘   │
│                          ↓                                   │
│                    [concatenate]                             │
│                          ↓                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              PREDICTION HEAD                         │   │
│   │  • Linear → Sigmoid → P(default)                    │   │
│   │  • Platt Scaling for calibration                    │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| **Wider VIME (512-dim)** | Minimal code changes | Still compresses, SSL objective unchanged | Rejected |
| **FT-Transformer** | Strong tabular performance | Complex attention, extensive tuning needed | Out of scope |
| **TabNet** | Interpretable attention | Less stable training, complex | Rejected |
| **NODE** | Tree-like NN | Very complex implementation | Rejected |
| **DCN-v2** | Preserves features, proven, moderate complexity | New implementation required | **Selected** |

## Impact

### Files to Create
```
src/models/dcn_v2/
├── __init__.py
├── soft_binning.py
├── embedding_engine.py
├── cross_network.py
├── deep_network.py
├── dcn_v2_model.py
├── dcn_v2_trainer.py
└── dcn_v2_reject_inference.py
```

### Config Changes
New section `DCN_V2_CONFIG` in `config.py` with all hyperparameters.

### No Breaking Changes
- Existing VIME code preserved for comparison
- Same evaluation metrics and data splits
- Same reject inference methodology (teacher-student)

## Success Criteria

| Metric | Target | Current Best |
|--------|--------|--------------|
| D_L_test AUC | ≥ 0.7391 | Route B: 0.7391 |
| D_R AUC | ≥ 0.7268 | Route F: 0.7268 |
| Diamonds@20% | ≥ 17,885 | Route B: 17,885 |

## Implementation Protocol

**Pair-programming required** for all code (AI_GUIDE.md Section 2.1):
1. EXPLAIN each line
2. DISCUSS with owner
3. APPROVE before writing
4. DOCUMENT purpose

---

**STATUS: APPROVED** (2026-02-15)
