# DCN-v2 Architecture (Deep & Cross Network)

## Full Architecture

```mermaid
flowchart TB
    subgraph INPUT["Input Layer"]
        NUM[("Numerical Features<br/>106 dimensions")]
        CAT[("Categorical Features<br/>16 columns")]
    end

    subgraph EMBEDDING["🔷 Embedding Engine"]
        subgraph SOFTBIN["SoftBinning (Numerical)"]
            SB1["Learnable Cutpoints<br/>(10 bins per feature)"]
            SB2["Sigmoid → Cumulative Probs"]
            SB3["Bin Membership Probs"]
            SB4["Weighted Sum of<br/>Bin Embeddings (16-dim)"]
            SB5["Missing → Index 0"]

            SB1 --> SB2 --> SB3 --> SB4
            SB5 --> SB4
        end

        subgraph CATEMB["Categorical Embeddings"]
            CE1["Lookup Table<br/>per category"]
            CE2["dim = √cardinality"]
            CE3["Missing → Index 0<br/>(learnable)"]

            CE1 --> CE2
            CE3 --> CE2
        end

        CONCAT[("Concatenate<br/>x₀ ≈ 700-dim")]
    end

    subgraph PARALLEL["Parallel Processing"]
        subgraph CROSS["🔶 Cross Network"]
            direction TB
            CL1["CrossLayer 1<br/>x₁ = x₀ ⊙ (V(U(x₀)) + b) + x₀"]
            CL2["CrossLayer 2<br/>x₂ = x₀ ⊙ (V(U(x₁)) + b) + x₁"]
            CL3["CrossLayer 3<br/>x₃ = x₀ ⊙ (V(U(x₂)) + b) + x₂"]
            CO[("Cross Output<br/>~700-dim")]

            CL1 --> CL2 --> CL3 --> CO
        end

        subgraph DEEP["🔷 Deep Network"]
            direction TB
            D1["Linear(d → 512)<br/>ReLU, Dropout(0.1)"]
            D2["Linear(512 → 256)<br/>ReLU, Dropout(0.1)"]
            D3["Linear(256 → 128)<br/>ReLU, Dropout(0.1)"]
            DO[("Deep Output<br/>128-dim")]

            D1 --> D2 --> D3 --> DO
        end
    end

    subgraph OUTPUT["Output Layer"]
        MERGE["Concatenate<br/>(~828-dim)"]
        HEAD["Linear(828 → 1)"]
        SIG["Sigmoid"]
        PRED[("P(default)<br/>∈ [0, 1]")]
    end

    NUM --> SOFTBIN --> CONCAT
    CAT --> CATEMB --> CONCAT
    CONCAT --> CROSS
    CONCAT --> DEEP
    CO --> MERGE
    DO --> MERGE
    MERGE --> HEAD --> SIG --> PRED

    style INPUT fill:#e3f2fd
    style EMBEDDING fill:#fff3e0
    style CROSS fill:#ffecb3
    style DEEP fill:#c8e6c9
    style OUTPUT fill:#f3e5f5
```

## Cross Layer Detail

```mermaid
flowchart LR
    subgraph CROSSLAYER["CrossLayer (Low-Rank)"]
        X0["x₀<br/>(initial embedding)"]
        XL["x_l<br/>(current layer)"]

        U["U: (d, 64)<br/>Project down"]
        V["V: (64, d)<br/>Project up"]
        B["b: bias"]

        HADAMARD["⊙<br/>Element-wise<br/>multiply"]
        ADD["+ x_l<br/>Residual"]

        XL --> U --> V
        V --> B_ADD["+ b"]
        B_ADD --> HADAMARD
        X0 --> HADAMARD
        HADAMARD --> ADD
        XL --> ADD
        ADD --> XL1["x_{l+1}"]
    end

    style CROSSLAYER fill:#fff3e0
```

## SoftBinning Detail

```mermaid
flowchart TB
    subgraph SOFTBIN["SoftBinning Layer"]
        INPUT["Raw numerical value x"]

        subgraph CUTPOINTS["Learnable Cutpoints"]
            C1["c₁ (10th percentile init)"]
            C2["c₂ (20th percentile init)"]
            C3["..."]
            C9["c₉ (90th percentile init)"]
        end

        DIST["Compute distances:<br/>x - c_i"]
        TEMP["Apply temperature τ=5.0"]
        SIG["Sigmoid → Cumulative probs"]
        DIFF["Difference → Bin probs"]
        EMB["Bin Embeddings<br/>E ∈ ℝ^{10 × 16}"]
        WEIGHTED["Weighted sum<br/>Σ p_i × E_i"]
        OUTPUT["16-dim embedding<br/>per feature"]
    end

    INPUT --> DIST --> TEMP --> SIG --> DIFF --> WEIGHTED --> OUTPUT
    EMB --> WEIGHTED

    style SOFTBIN fill:#e8f5e9
```

## Parameter Breakdown

```mermaid
pie title DCN-v2 Parameters (~800K-1M)
    "SoftBinning Embeddings (106×10×16)" : 169600
    "Categorical Embeddings" : 50000
    "Cross Network (3 layers, rank 64)" : 270000
    "Deep Network (d→512→256→128)" : 450000
    "Output Head" : 829
```

## Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| SoftBinning Bins | 10 | Learnable bin boundaries |
| SoftBinning Temperature | 5.0 | Soft-to-hard transition |
| Bin Embedding Dim | 16 | Per-feature embedding size |
| Cross Network Layers | 3 | 2nd to 4th order interactions |
| Cross Network Rank | 64 | Low-rank factorization |
| Deep Network | 512→256→128 | MLP layers |
| Dropout | 0.1 | Regularization |
| Loss | Focal (α=0.25, γ=2.0) | Class imbalance |
| Optimizer | AdamW | Weight decay 1e-5 |
| Batch Size | 1024 | |
| Early Stopping | P@20%, patience=10 | |
