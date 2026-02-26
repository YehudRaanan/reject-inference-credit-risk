# DCN-v2 Implementation Plan — Enriched with Project Guidelines

> **Status**: PENDING APPROVAL
> **Date**: 2026-02-15
> **Purpose**: Complete implementation roadmap for DCN-v2 architecture, aligned with all project guidelines and pair-programming requirements.

---

## Executive Summary

This plan transitions the project from VIME (128-dim bottleneck, AUC ~0.716) to **DCN-v2** (Deep & Cross Network v2), addressing the information compression problem identified in Phase 5.7.

**Key Insight**: The VIME ceiling is architectural (253→128 dim compression). DCN-v2 preserves original signals through residual connections to source at every layer.

---

## 1. Approval Gates (from AI_GUIDE.md)

> [!CAUTION]
> **The following require explicit written approval before implementation:**

| Gate | Items Requiring Approval | Status |
|------|-------------------------|--------|
| **Architecture** | DCN-v2 layers, CrossLayer, DeepLayer, SoftBinning module designs | PENDING |
| **Hyperparameters** | lr=1e-3, wd=1e-4, dropout=0.1-0.2, embedding dims, bin counts | PENDING |
| **Data Pipeline** | Soft binning preprocessing, missing bin (Bin 0), LayerNorm | PENDING |
| **Evaluation** | Same metrics as previous phases (AUC, PR-AUC, Diamond Recovery) | PENDING |
| **Dependencies** | No new packages required (PyTorch, CatBoost already installed) | PENDING |

---

## 2. Pair-Programming Protocol

> [!IMPORTANT]
> **MANDATORY FOR ALL CODE IN THIS PHASE**

### 2.1 Code Writing Process

**Every line of code** must follow this process:

1. **EXPLAIN** — Agent describes what the line does and why it's needed
2. **DISCUSS** — Owner asks questions, suggests alternatives if desired
3. **APPROVE** — Owner explicitly approves (e.g., "approved", "yes", "ok")
4. **WRITE** — Only then does the agent write the code to the file

### 2.2 Documentation Requirements

For each code block:
```python
# Purpose: [What this block accomplishes]
# Why: [Why this approach was chosen]
# Alternatives: [What else was considered]
# Approved: [Date] by Owner
```

### 2.3 Review Checkpoints

| Checkpoint | Trigger | Action |
|------------|---------|--------|
| **Module Complete** | Each nn.Module finished | Owner reviews full file |
| **Integration Point** | Modules connect | Test with dummy data |
| **Training Loop** | Before first run | Owner approves hyperparams |
| **Evaluation** | Before metrics | Owner confirms test setup |

---

## 3. Implementation Phases

### Phase 8.0: Project Setup for DCN-v2

- [ ] **8.0.1** Create proposal document (`docs/proposals/2026-02-15_dcn_v2_architecture.md`)
- [ ] **8.0.2** Owner reviews and approves architecture proposal
- [ ] **8.0.3** Create folder structure: `src/models/dcn_v2/`
- [ ] **8.0.4** Add decision log entry for DCN-v2 transition

**Artifacts**: Proposal doc, decision log entry

---

### Phase 8.1: Soft Binning Module (PAIR-PROGRAMMING)

**Goal**: Implement differentiable soft binning for numerical features with dedicated missing bin.

#### 8.1.1 Module Design Discussion
- [ ] Explain SoftBinning architecture concept
- [ ] Discuss temperature decay schedule (cold→hot)
- [ ] Discuss quantile initialization strategy
- [ ] Owner approves design

#### 8.1.2 Implementation (Line-by-Line)
- [ ] `src/models/dcn_v2/__init__.py` — Package init
- [ ] `src/models/dcn_v2/soft_binning.py` — SoftBinning class
  - Class definition and `__init__`
  - Learnable cutpoints (nn.Parameter)
  - Forward pass (membership calculation)
  - Missing value handling (Bin 0)
  - Temperature scheduling

#### 8.1.3 QA
- [ ] Unit test: binning produces valid memberships (sum to 1)
- [ ] Unit test: NaN values map to Bin 0
- [ ] Unit test: temperature decay works

**Artifacts**: `soft_binning.py`, unit tests

---

### Phase 8.2: Embedding Engine (PAIR-PROGRAMMING)

**Goal**: Translate raw features into high-dimensional embeddings.

#### 8.2.1 Module Design Discussion
- [ ] Explain embedding lookup for categorical features
- [ ] Explain weighted bin embedding for numerical features
- [ ] Discuss embedding dimension strategy (sqrt cardinality vs fixed)
- [ ] Owner approves design

#### 8.2.2 Implementation (Line-by-Line)
- [ ] `src/models/dcn_v2/embedding_engine.py` — EmbeddingEngine class
  - Categorical embedding tables
  - Numerical bin embeddings
  - Concatenation logic
  - LayerNorm on final x₀

#### 8.2.3 QA
- [ ] Unit test: output dimension matches expected
- [ ] Unit test: categorical embeddings work with variable cardinality
- [ ] Unit test: LayerNorm applied correctly

**Artifacts**: `embedding_engine.py`, unit tests

---

### Phase 8.3: Cross Network (PAIR-PROGRAMMING)

**Goal**: Implement explicit polynomial feature interactions.

#### 8.3.1 Module Design Discussion
- [ ] Explain CrossLayer formula: x_{l+1} = x₀ ⊙ (W · x_l + b) + x_l
- [ ] Discuss residual connection to source x₀
- [ ] Discuss number of cross layers (start with 3)
- [ ] Owner approves design

#### 8.3.2 Implementation (Line-by-Line)
- [ ] `src/models/dcn_v2/cross_network.py` — CrossLayer and CrossNetwork classes
  - Single CrossLayer implementation
  - Stacked CrossNetwork with configurable depth
  - Efficient implementation with torch.addmm

#### 8.3.3 QA
- [ ] Unit test: output dimension preserved
- [ ] Unit test: gradient flows through all layers
- [ ] Unit test: x₀ residual correctly applied

**Artifacts**: `cross_network.py`, unit tests

---

### Phase 8.4: Deep Network (PAIR-PROGRAMMING)

**Goal**: Implement MLP path for implicit non-linear transformations.

#### 8.4.1 Module Design Discussion
- [ ] Explain MLP architecture (512→256→128)
- [ ] Discuss GELU vs ReLU activation
- [ ] Discuss dropout strategy (only on Deep path)
- [ ] Owner approves design

#### 8.4.2 Implementation (Line-by-Line)
- [ ] `src/models/dcn_v2/deep_network.py` — DeepNetwork class
  - Layer definitions
  - GELU activation
  - Dropout layers
  - Forward pass

#### 8.4.3 QA
- [ ] Unit test: output dimension correct
- [ ] Unit test: dropout applied during training only

**Artifacts**: `deep_network.py`, unit tests

---

### Phase 8.5: DCN-v2 Complete Model (PAIR-PROGRAMMING)

**Goal**: Integrate all components into full DCN-v2 model.

#### 8.5.1 Module Design Discussion
- [ ] Explain fusion strategy (concatenate Cross + Deep outputs)
- [ ] Discuss prediction head (Linear → Sigmoid)
- [ ] Discuss Platt Scaling for calibration
- [ ] Owner approves design

#### 8.5.2 Implementation (Line-by-Line)
- [ ] `src/models/dcn_v2/dcn_v2_model.py` — DCNv2 class
  - Combine SoftBinning + EmbeddingEngine + CrossNetwork + DeepNetwork
  - Fusion layer
  - Prediction head
  - Forward pass

#### 8.5.3 QA
- [ ] Unit test: end-to-end forward pass works
- [ ] Unit test: output is valid probability [0, 1]
- [ ] Integration test: model trains on dummy data

**Artifacts**: `dcn_v2_model.py`, integration tests

---

### Phase 8.6: Training Pipeline (PAIR-PROGRAMMING)

**Goal**: Implement training loop with proper loss and optimization.

#### 8.6.1 Training Design Discussion
- [ ] Explain AdamW optimizer choice
- [ ] Explain cost-sensitive loss (class weight ~11.5x)
- [ ] Discuss Focal Loss as alternative
- [ ] Discuss early stopping strategy
- [ ] Owner approves design

#### 8.6.2 Implementation (Line-by-Line)
- [ ] `src/models/dcn_v2/dcn_v2_trainer.py` — Training loop
  - Data loading (reuse existing parquet infrastructure)
  - Loss function (BCE with class weights)
  - Optimizer setup
  - Training loop with validation
  - Checkpoint saving
  - Gradient norm monitoring

#### 8.6.3 QA
- [ ] Verify loss decreases during training
- [ ] Verify gradient norms are reasonable
- [ ] Verify checkpoints save correctly

**Artifacts**: `dcn_v2_trainer.py`, training logs

---

### Phase 8.7: Reject Inference (PAIR-PROGRAMMING)

**Goal**: Apply DCN-v2 for diamond recovery using teacher-student approach.

#### 8.7.1 Design Discussion
- [ ] Explain teacher model scoring on D_R
- [ ] Explain Isolation Forest OOD filtering
- [ ] Explain pseudo-label threshold selection
- [ ] Explain student model training
- [ ] Owner approves design

#### 8.7.2 Implementation (Line-by-Line)
- [ ] `src/models/dcn_v2/dcn_v2_reject_inference.py`
  - Teacher training on D_L
  - D_R scoring
  - IF filtering
  - Pseudo-label generation
  - Student training on D_L + D_R_filtered

#### 8.7.3 QA
- [ ] Compare Diamond Recovery vs Route B baseline
- [ ] Document filtering statistics

**Artifacts**: `dcn_v2_reject_inference.py`, results JSON

---

### Phase 8.8: Evaluation & Comparison

**Goal**: Fair comparison with all previous routes.

- [ ] **8.8.1** Evaluate on D_L_test (AUC-ROC, PR-AUC)
- [ ] **8.8.2** Evaluate on D_R (Diamond Recovery, Precision@k, Lift)
- [ ] **8.8.3** Evaluate on full population
- [ ] **8.8.4** Generate comparison table (all 12+ routes)
- [ ] **8.8.5** SHAP analysis for interpretability
- [ ] **8.8.6** Write phase report (`docs/reports/phase8_dcn_v2_report.md`)

**Success Criteria**:
- [ ] DCN-v2 Test AUC ≥ 0.7391 (match Route B)
- [ ] DCN-v2 D_R AUC ≥ 0.7206 (match Route B)
- [ ] No information bottleneck (preserve all 253 features through network)

---

## 4. Config Updates Required

Add to `config.py` (requires approval):

```python
# DCN-v2 Configuration
DCN_V2_CONFIG = {
    # Soft Binning
    'num_bins': 10,
    'temperature_init': 0.1,
    'temperature_final': 10.0,

    # Embedding
    'numerical_embed_dim': 16,
    'categorical_embed_formula': 'sqrt',  # sqrt(cardinality)

    # Cross Network
    'num_cross_layers': 3,

    # Deep Network
    'deep_layers': [512, 256, 128],
    'activation': 'gelu',
    'dropout': 0.15,

    # Training
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 100,
    'batch_size': 1024,
    'early_stopping_patience': 10,

    # Class Imbalance
    'pos_class_weight': 11.5,  # Based on ~8% default rate
    'use_focal_loss': False,
    'focal_gamma': 2.0,
}
```

---

## 5. File Structure

```
src/models/dcn_v2/
├── __init__.py
├── soft_binning.py        # Phase 8.1
├── embedding_engine.py    # Phase 8.2
├── cross_network.py       # Phase 8.3
├── deep_network.py        # Phase 8.4
├── dcn_v2_model.py        # Phase 8.5
├── dcn_v2_trainer.py      # Phase 8.6
└── dcn_v2_reject_inference.py  # Phase 8.7
```

---

## 6. Handover Protocol

At the end of each session:

1. **Create handover doc** (`docs/handovers/2026-02-XX_dcn_v2_session_N.md`)
2. **List completed code** with approval status
3. **Document any open questions**
4. **Specify exact next step** for next session

---

## 7. Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| D_L_test AUC | ≥ 0.7391 | Match Route B baseline |
| D_R AUC | ≥ 0.7268 | Match Route F (best D_R) |
| Diamonds@20% | ≥ 17,885 | Match Route B |
| No bottleneck | 253 features preserved | Core architectural goal |
| Code quality | All lines approved | Learning objective |

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| DCN-v2 underperforms | Ablation studies (remove components) |
| Training instability | Gradient clipping, LR scheduling |
| Overfitting | Dropout, early stopping, weight decay |
| Slow iteration | Smaller model for prototyping first |

---

## Approval Request

**Owner**: Please review this enriched plan and confirm:

1. [ ] Architecture approach approved
2. [ ] Pair-programming protocol understood and accepted
3. [ ] Phase structure approved
4. [ ] Ready to begin Phase 8.0

---

*This plan follows all guidelines from `AI_GUIDE.md` and incorporates lessons learned from Phases 1-7.*
