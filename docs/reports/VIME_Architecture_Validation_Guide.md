# VIME Architecture Validation Guide

This document details the architecture of the **VIME (Value Imputation and Mask Estimation)** models implemented in the project. It explicitly breaks down the two versions tested: the **Baseline 128-dim (Standard)** and the **Expanded 512-dim (Wide)**, to enable external validation of the architectures and experimental assumptions.

---

## 1. VIME Methodology Overview

VIME is a self-supervised neural network architecture adapted specifically for tabular data. The training process operates in two distinct phases:

1.  **Self-Supervised Pretraining (SSL):** The model receives corrupted tabular data (randomly shuffled values derived from the empirical distribution of each feature) and is trained to:
    *   **Mask Estimation:** Predict which specific features were artificially corrupted (Binary Classification using Binary Cross-Entropy Loss).
    *   **Feature Reconstruction:** Reconstruct the original, uncorrupted feature values (Regression using Mean Squared Error / Masked MSE Loss).
2.  **Supervised Fine-Tuning / Classification:** The pretrained encoder backbone is frozen (or fine-tuned), and a Multi-Layer Perceptron (MLP) classification head is attached to the latent embedding to predict the target label (e.g., Default Risk).

---

## 2. Model Variations

The project tests a foundational hypothesis about representational capacity in tabular deep learning: *Is a narrow bottleneck limiting the model's ability to learn complex tabular interactions?*

To test this, two specific architectures were implemented. A third variant (**VIME-128-Fair-v2**) was later created to eliminate confounding training variables and produce a fair architectural comparison.

### 2.1 Baseline VIME (128-dim Bottleneck)
**Implementation files:** `src/models/vime/vime_model.py`, `src/models/vime/encoder.py`

This version closely follows standard implementations with a strict dimensional bottleneck to force feature compression.

*   **Encoder Backbone:**
    *   Input $\rightarrow$ Linear(Input Dim, 256) $\rightarrow$ BatchNorm1d $\rightarrow$ ReLU $\rightarrow$ Dropout(0.1 or 0.3)
    *   Linear(256, 256) $\rightarrow$ ReLU (Present in `vime_model.py`, simplified in `encoder.py`)
    *   Linear(256, **128**) $\rightarrow$ BatchNorm1d $\rightarrow$ ReLU $\rightarrow$ **Latent Embedding (128-dim)**
*   **Decoders (SSL Phase):** Both decoders are shallow, mapping directly from the compressed space back to the input space.
    *   **Mask Estimator Head:** Linear(128, Input Dim) $\rightarrow$ Sigmoid
    *   **Feature Reconstructor Head:** Linear(128, Input Dim)

### 2.2 Fair-v2 VIME-128 (Controlled Comparison)
**Implementation file:** `src/models/vime/train_vime_128_fair_v2.py`

This variant was created to eliminate confounding training variables between VIME-128 and VIME-512. It uses the **same encoder architecture** as the baseline VIME-128 (Section 2.1) but adopts all of VIME-512's training settings:

*   **Encoder Backbone:** Identical to baseline VIME-128 (253 → 256 → 256 → **128**), but with `Dropout(0.3)` matching VIME-512.
*   **Decoders (SSL Phase):** Same shallow decoders as baseline (`Linear(128, Input Dim)`).
*   **Training Unification (matches VIME-512):**
    *   Encoder dropout: `0.3` (was `0.1`)
    *   SSL batch size: `512` (was `128`)
    *   SSL epochs: `30` (was `20`)
    *   Mask loss weight (α): `1.0` (was `2.0`)
    *   Reconstruction loss: **Masked MSE** — ignores NaN positions (was full MSE)
    *   Optimizer: **AdamW** (was Adam)
    *   Supervised loss: **Focal Loss** (α=0.25, γ=2.0) (was BCE)
    *   Early stopping metric: **P@20%** (was AUC)

> With these changes, the **only** remaining differences between Fair-v2 and VIME-512 are architectural: latent dim (128 vs 512), hidden dim (256 vs 512), and decoder depth (shallow vs mirror).

### 2.3 Fair-v3 VIME-512 (Fully Controlled Comparison)
**Implementation file:** `src/models/vime/train_vime_512_fair_v3.py`

This variant uses the **same VIME model class** as VIME-128 (Section 2.1), instantiated with 512-dim widths. This eliminates **all** confounders — same model class, same number of layers, same training settings — leaving only width as the variable.

*   **Encoder Backbone:** Same 3-layer structure as VIME-128: `253 → 512 → 512 → 512`, with `Dropout(0.3)`.
*   **Decoders (SSL Phase):** Same shallow 1-layer decoders: `Linear(512, Input Dim)`.
*   **Training:** 100% identical to Fair-v2 (dropout 0.3, masked MSE, focal loss, AdamW, same hyperparameters).

> The **only** difference between Fair-v3 and Fair-v2 is the width: 512/512 vs 128/256. Everything else — model class, layer count, training settings — is identical.

### 2.4 Expanded VIME-Wide (512-dim No-Bottleneck)
**Implementation file:** `src/models/vime/vime_wide.py`

This version (Route G-Wide) tests if removing the bottleneck improves performance, hypothesizing that 128 dimensions restrict the representation of 250+ tabular features.

*   **Encoder Backbone (Wider & Deeper):**
    *   Input $\rightarrow$ Linear(Input Dim, **512**) $\rightarrow$ BatchNorm1d $\rightarrow$ ReLU $\rightarrow$ Dropout(0.3)
    *   Linear(512, **512**) $\rightarrow$ BatchNorm1d $\rightarrow$ ReLU $\rightarrow$ Dropout(0.3) $\rightarrow$ **Latent Embedding (512-dim)**
*   **Mirror Decoders (SSL Phase):** Unlike the baseline, the decoders in the Wide version are deep and mirror the encoder's structure, allowing for more complex decoding logic before reaching the input dimension.
    *   **Mask Estimator Head:** Linear(512, 512) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Linear(512, Input Dim) $\rightarrow$ Sigmoid
    *   **Feature Reconstructor Head:** Linear(512, 512) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Linear(512, Input Dim)

---

## 3. Key Nuances and Structural Differences

For reproducibility and validation, note the following critical differences between the architectures:

1.  **Latent Capacity:** The `VIMEWide` model completely removes the dimensionality reduction step. Instead of shrinking ~253 features into 128 latent parameters, it projects them into a 512-dimensional space, preserving up to 4x more capacity.
2.  **Decoder Depth:** The `VIMEWide` decoders use an intermediate hidden layer of 512 dimensions (`Linear(512, 512) -> Transformed -> Linear(512, Input)`). The baseline model uses a direct mapping (`Linear(128, Input)`). This means `VIMEWide` has significantly deeper reconstruction expressiveness.
3.  **Regularization (Dropout):** The `VIMEWide` model enforces a stricter, uniform dropout rate (`0.3`) across all encoder and decoder layers. The original baseline used lower dropout (`0.1`) only on specific layers. **Fair-v2 unifies this to `0.3`**, removing dropout as a confounding variable.
4.  **Classification Head:** `VIMEWide` utilizes a standardized internal classification head for phase 2 (`ClassificationHead` class in `train_vime_wide.py`):
    *   512 (Emb) $\rightarrow$ 256 (BN+ReLU+Drop 0.3) $\rightarrow$ 64 (BN+ReLU+Drop 0.3) $\rightarrow$ 1 (Logit)
    *   It also natively implements a **Focal Loss** function (`alpha=0.25`, `gamma=2.0`) to combat class imbalance. **Fair-v2 also uses Focal Loss**, removing the loss function as a confounding variable.
5.  **Reconstruction Loss Mechanism:** `VIMEWide` utilizes a `Partial Reconstruction Loss` methodology (`masked_mse_loss`). It selectively ignores naturally occurring `NaN` values (passed via a pre-calculated missing mask from preprocessing) during the MSE calculation, forcing the network to penalize *only* the artificial corruption it introduced. **Fair-v2 also uses Masked MSE**, removing this as a confounding variable.

### 3.1 Confounding Variables Identified and Resolved

The original VIME-128 vs VIME-512 comparison conflated 6 training differences with architectural differences. Fair-v2 resolves all of them:

| Variable | VIME-128 (original) | VIME-128-Fair-v2 | VIME-512 |
| :--- | :--- | :--- | :--- |
| Encoder dropout | 0.1 | **0.3** | 0.3 |
| SSL batch size | 128 | **512** | 512 |
| SSL epochs | 20 | **30** | 30 |
| Mask loss weight (α) | 2.0 | **1.0** | 1.0 |
| Recon loss | Full MSE | **Masked MSE** | Masked MSE |
| Supervised loss | BCE | **Focal** | Focal |

---

## 4. Summary Table

| Parameter/Feature | Baseline VIME (128) | Fair-v2 VIME (128) | Fair-v3 VIME (512) | VIME-Wide (512) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Class** | VIME | VIME | **VIME** | VIMEWide |
| **Encoder Architecture** | `In→256→(256)→128` | `In→256→(256)→128` | `In→512→(512)→512` | `In→512→512` |
| **Encoder Layers** | 3 | 3 | **3** | 2 |
| **Latent Embedding Size** | **128-dim** | **128-dim** | **512-dim** | **512-dim** |
| **Decoder Architecture** | Shallow (`128→In`) | Shallow (`128→In`) | **Shallow (`512→In`)** | Deep Mirror (`512→512→In`) |
| **Decoder Layers** | 1 | 1 | **1** | 2 |
| **Dropout** | `0.1` (variable) | `0.3` (uniform) | `0.3` (uniform) | `0.3` (uniform) |
| **Loss Strategy** | Standard MSE + BCE | Masked MSE + Focal | Masked MSE + Focal | Masked MSE + Focal |
| **Purpose** | Original baseline | Fair training comparison | **Fully controlled** (only width differs from Fair-v2) | Original capacity test |

---

## 5. Experimental Results

| Model | Model Class | D_R AUC | D_R P@20% | D_T AUC | Best Val AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VIME-512-Fair-v3** | VIME | **0.7058** | **0.9591** | **0.7188** | **0.7285** |
| VIME-128-Fair-v2 | VIME | 0.6956 | 0.9588 | 0.7171 | 0.7310 |
| VIME-128 (original) | VIME | 0.6856 | 0.9522 | 0.7079 | 0.7213 |
| VIME-512 (Wide) | VIMEWide | 0.6822 | 0.9548 | 0.7125 | 0.7217 |

### Paired Bootstrap Significance (n=1000 resamples)

| Comparison | ΔAUC (D_R) | 95% CI | Significant? |
| :--- | :--- | :--- | :--- |
| **VIME-512-Fair-v3 vs VIME-128-Fair-v2** | **+0.0102** | [+0.0080, +0.0125] | **YES** |
| VIME-512-Fair-v3 vs VIME-512 (Wide) | +0.0236 | [+0.0209, +0.0264] | YES |
| VIME-128-Fair-v2 vs VIME-512 (Wide) | +0.0134 | [+0.0107, +0.0162] | YES |
| VIME-128-Fair-v2 vs VIME-128 (old) | +0.0100 | [+0.0075, +0.0125] | YES |
| VIME-128 (old) vs VIME-512 (Wide) | +0.0034 | [−0.0001, +0.0068] | No (marginal) |

### Conclusion

A three-stage controlled comparison reveals:
1. **Fair-v2 (confounded by architecture):** 128-dim > 512-dim (ΔAUC = +0.0134) — but 512 used a different model class (VIMEWide: 2-layer encoder, mirror decoder)
2. **Fair-v3 (fully controlled):** 512-dim > 128-dim (ΔAUC = +0.0102) — when using the **same VIME model class**, more capacity helps
3. **Architecture effect:** The VIMEWide mirror decoder hurts 512-dim by ΔAUC = −0.0236, far more than the capacity helps (+0.0102)

**Bottom line:** More latent capacity does help, but the VIMEWide architecture (2-layer encoder + mirror decoder) is harmful. Simpler is better for VIME on this dataset.
