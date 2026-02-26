# VIME-512 Architecture (Wide Variant)

## Full Architecture (SSL + Classification)

```mermaid
flowchart TB
    subgraph INPUT["Input Layer"]
        I[("Input Features<br/>253 dimensions<br/>(122 original + 10 engineered<br/>+ one-hot + masks)")]
    end

    subgraph ENCODER["🔷 Wide Encoder (No Bottleneck)"]
        E1["Linear(253 → 512)"]
        BN1["BatchNorm1d(512)"]
        R1["ReLU"]
        D1["Dropout(0.3)"]

        E2["Linear(512 → 512)"]
        BN2["BatchNorm1d(512)"]
        R2["ReLU"]
        D2["Dropout(0.3)"]

        LATENT[("Latent Space<br/>512-dim<br/>NO compression")]
    end

    subgraph SSL["🔶 SSL Decoder (Mirror Architecture)"]
        subgraph MASK_HEAD["Mask Prediction Head"]
            M1["Linear(512 → 512)"]
            MBN["BatchNorm1d(512)"]
            MR["ReLU"]
            MD["Dropout(0.3)"]
            M2["Linear(512 → 253)"]
            MS["Sigmoid"]
            MO[("Mask Output<br/>253-dim<br/>P(corrupted)")]
        end

        subgraph RECON_HEAD["Reconstruction Head"]
            R_1["Linear(512 → 512)"]
            RBN["BatchNorm1d(512)"]
            RR["ReLU"]
            RD["Dropout(0.3)"]
            R_2["Linear(512 → 253)"]
            RO[("Reconstruction<br/>253-dim<br/>Feature values")]
        end
    end

    subgraph CLS["🔷 Classification Head (Fine-tuning)"]
        C1["Linear(512 → 256)"]
        CBN1["BatchNorm1d(256)"]
        CR1["ReLU"]
        CD1["Dropout(0.3)"]

        C2["Linear(256 → 64)"]
        CBN2["BatchNorm1d(64)"]
        CR2["ReLU"]
        CD2["Dropout(0.3)"]

        C3["Linear(64 → 1)"]
        SIG["Sigmoid"]
        OUT[("P(default)<br/>∈ [0, 1]")]
    end

    I --> E1 --> BN1 --> R1 --> D1
    D1 --> E2 --> BN2 --> R2 --> D2 --> LATENT

    LATENT --> M1 --> MBN --> MR --> MD --> M2 --> MS --> MO
    LATENT --> R_1 --> RBN --> RR --> RD --> R_2 --> RO

    LATENT --> C1 --> CBN1 --> CR1 --> CD1
    CD1 --> C2 --> CBN2 --> CR2 --> CD2
    CD2 --> C3 --> SIG --> OUT

    style INPUT fill:#e3f2fd
    style ENCODER fill:#bbdefb
    style SSL fill:#fff3e0
    style CLS fill:#c8e6c9
    style LATENT fill:#e1bee7,stroke:#9c27b0,stroke-width:3px
```

## VIME-128 vs VIME-512 Comparison

```mermaid
flowchart LR
    subgraph V128["VIME-128 (Bottleneck)"]
        direction TB
        V128_IN["253-dim"]
        V128_H1["256"]
        V128_H2["256"]
        V128_L["128"]
        V128_OUT["1"]

        V128_IN --> V128_H1 --> V128_H2 --> V128_L --> V128_OUT
    end

    subgraph V512["VIME-512 (Wide)"]
        direction TB
        V512_IN["253-dim"]
        V512_H1["512"]
        V512_H2["512"]
        V512_L["512"]
        V512_OUT["1"]

        V512_IN --> V512_H1 --> V512_H2 --> V512_L --> V512_OUT
    end

    V128 -.->|"Bottleneck<br/>hypothesis"| COMPARE
    V512 -.->|"test"| COMPARE

    subgraph COMPARE["Results (Fair v3 Comparison)"]
        RES["VIME-512-Fair-v3 (VIME class): AUC 0.7058<br/>VIME-128-Fair-v2 (VIME class): AUC 0.6956<br/>VIME-512 G-Wide (VIMEWide class): AUC 0.6822<br/>Capacity helps (+0.0102), but<br/>mirror decoder hurts (−0.0236)"]
    end

    style V128 fill:#ffecb3
    style V512 fill:#c8e6c9
    style COMPARE fill:#f3e5f5
```

## Parameter Count Breakdown

```mermaid
pie title VIME-512 Parameters (~850K total)
    "Encoder Layer 1 (253→512)" : 130048
    "Encoder Layer 2 (512→512)" : 262656
    "Mask Head (512→512→253)" : 392957
    "Recon Head (512→512→253)" : 392957
    "Cls Head (512→256→64→1)" : 148801
```

## Architecture Design Rationale

```mermaid
flowchart TB
    subgraph HYPOTHESIS["Bottleneck Hypothesis"]
        H1["VIME-128 compresses 253 → 128<br/>(~1.9× reduction)"]
        H2["Information may be lost<br/>in compression"]
        H3["Wider latent space<br/>should preserve more info"]
    end

    subgraph DESIGN["VIME-512 Design Choices"]
        D1["No compression<br/>253 → 512 (expansion)"]
        D2["Stronger dropout (0.3)<br/>to prevent overfitting"]
        D3["Mirror decoder architecture<br/>to avoid decoder bottleneck"]
        D4["Matching capacity<br/>in encoder and decoder"]
    end

    subgraph RESULT["Experimental Result (Fair v3)"]
        R1["512 > 128 when same class (ΔAUC +0.0102)"]
        R2["But VIMEWide hurts (ΔAUC −0.0236)"]
        R3["Conclusion: Capacity helps,<br/>but mirror decoder architecture<br/>hurts more. Pretext task<br/>misalignment is dominant issue."]
    end

    HYPOTHESIS --> DESIGN --> RESULT

    style HYPOTHESIS fill:#fff3e0
    style DESIGN fill:#e3f2fd
    style RESULT fill:#ffcdd2
```

## Hyperparameters

| Parameter | VIME-128 (old) | Fair-v2 (128) | Fair-v3 (512) | G-Wide (512) |
|-----------|----------------|---------------|---------------|--------------|
| Model Class | VIME | VIME | **VIME** | VIMEWide |
| Encoder Layers | 3 | 3 | **3** | 2 |
| Decoder Layers | 1 (shallow) | 1 (shallow) | **1 (shallow)** | 2 (mirror) |
| Hidden Dimension | 256 | 256 | **512** | 512 |
| Latent Dimension | 128 | 128 | **512** | 512 |
| Encoder Dropout | 0.1 | 0.3 | **0.3** | 0.3 |
| Total Parameters | ~193K | ~193K | **~915K** | ~850K |
| SSL Epochs | 20 | 30 | **30** | 30 |
| SSL Alpha | 2.0 | 1.0 | **1.0** | 1.0 |
| SSL Batch Size | 128 | 512 | **512** | 512 |
| Cls Epochs (best) | 6 | 6 | **11** | 8 |
| D_R AUC | 0.6856 | 0.6956 | **0.7058** | 0.6822 |

*Fair-v3 uses the SAME VIME model class as Fair-v2, with only width changed. This eliminates all confounders except latent capacity.*
