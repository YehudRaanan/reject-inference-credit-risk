# Critique: "Neural Networks vs Gradient Boosting for Tabular Data"

**Date**: 2026-02-11
**Related Document**: `docs/reports/conclusion_nn_vs_gbdt_missing_data.md`

## Executive Summary

The referenced conclusion correctly identifies that **Gradient Boosted Decision Trees (GBDTs)** like CatBoost currently outperform the implemented **Neural Network (NN)** approaches (VIME) on the Home Credit dataset.

However, the document makes **sweeping theoretical generalizations** that are scientifically inaccurate in the context of modern Deep Learning research (2021-2024). Specifically, the claim that NNs have a "structural limitation" preventing them from handling missing data is **false** for state-of-the-art architectures.

While the *recommendation* to use GBDTs for this specific project is sound (due to efficiency and performance on medium-sized tabular data), the *justification* relies on outdated views of Neural Network capabilities.

---

## 1. The "NNs Cannot Process NaN" Fallacy

**Original Claim**: *"Neural networks cannot process NaN values... This forces a preprocessing step that necessarily destroys information."*

**Critique**: This applies only to standard Multi-Layer Perceptrons (MLPs) using simple imputation (mean/median). It ignores:

1.  **Tokenization of Missingness**: Modern Transformer-based architectures for tabular data (e.g., **FT-Transformer**, **TabTransformer**) treat features as tokens. A missing value is simply assigned a special `<UNK>` or `<NAN>` learnable embedding vector, preserving 100% of the "missingness signal" exactly as a tree does.
2.  **Attention Mechanisms**: Mechanisms like **TabNet** (comparable to GBDTs) uses attentive masks to select features. Missing values can be masked out dynamically, effectively "routing" around them, similar to a decision tree branch.
3.  **Learnable Imputation**: Even in simple MLPs, concatenating a binary mask `[Value, Is_Missing]` to the input allows the network to learn the interaction $f(x, m)$ theoretically approximating any tree split rule, given sufficient depth and data.

**Correction**: NNs *can* process missing data natively if the architecture supports it (e.g., via embeddings or masking), without "destroying" information.

## 2. Critique of the VIME Implementation

**Original Claim**: *"VIME... lose[s] information in preprocessing."*

**Critique**: The underperformance of VIME in this project (Route A) is likely due to **implementation choices**, not a fundamental flaw in Self-Supervised Learning (SSL):

1.  **Frozen Embeddings**: The project used VIME as a *fixed feature extractor* for CatBoost. This "bottlenecks" all 328 features into a limiting 128-dimension vector.
    *   *Correction*: A true comparison would involve **fine-tuning** the VIME encoder with a classification head (End-to-End training). This allows the model to adjust how it handles missing values specifically to minimize the *downstream* classification loss, rather than just the *reconstruction* loss.
2.  **Reconstruction vs. Prediction**: VIME's pretext task (recovering the value) forces the model to learn the *correlations* of the missing value. For credit risk, the *fact* that it is missing might be more important than *what the value would have been*. By forcing reconstruction, the model might be "denoising" the very signal (missingness) that predicts default.

## 3. Updated Literature Context (2021-2025)

The conclusion cites **Grinsztajn et al. (2022)**, a strong paper supporting GBDTs. However, more recent work provides nuance:

*   **Gorishniy et al. (2021, "Revisiting Deep Learning...")**: Demonstrated that **FT-Transformer** (Feature Tokenizer Transformer) performs on par with GBDTs on most datasets, specifically by handling categorical and numerical features more intelligently (embeddings).
*   **Hollmann et al. (2023, "TabPFN")**: Showed that Transformers can solve small tabular datasets better than GBDTs by acting as a "Prior-Data Fitted Network".
*   **McElfresh et al. (2023)**: Highlight that while GBDTs win on "out-of-the-box" benchmarks, NNs often win when:
    1.  Data scale is massive (>1M rows).
    2.  Multi-modal signals (text/graphs) are integrated (which this project mentions but dismisses).

## 4. Nuance on "Structural Superiority"

**Original Claim**: *"Tree-based models are structurally superior."*

**Critique**: "Structurally fit" is a better term.
*   **GBDTs** are structurally fit for *axis-aligned decision boundaries* and *discrete/missing handling* on medium-sized structured data.
*   **NNs** are structurally fit for *complex interactions*, *differentiable optimization*, and *representation learning*.

The "structural superiority" of trees implies NNs *cannot* reach that performance. In reality, NNs *can* reach it (universal approximation theorem), but they are **less data-efficient** and **harder to tune** to get there. For a 300k row dataset, the "effort-to-accuracy ratio" heavily favors GBDTs, but not because of an impossible structural barrier.

## Final Verdict

**The Decision to usage CatBoost is Correct**:
For the Home Credit Default Risk task (300k rows, purely tabular, informative missingness), **CatBoost** is practically superior. It achieves state-of-the-art results with a fraction of the compute and tuning time.

**The Theory is Flawed**:
The document incorrectly conflates "Standard MLP with Median Imputation" with "All Neural Networks".
*   **Do not** cite "structural impossibility" as the reason.
*   **Do** cite "pragmatic efficiency", "induction bias suitability", and "complexity of implementation" as the reasons.

**Recommendation for Future Reports**:
Soften the language from "Neural Networks cannot..." to "Standard Neural Network pipelines require complex workarounds (like specialized architectures or embeddings) to match the native efficiency of GBDTs on this type of data."
