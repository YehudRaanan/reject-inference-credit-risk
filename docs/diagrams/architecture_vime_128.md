# VIME-128 Architecture

> **Note:** Two variants exist. The **original** VIME-128 used different training
> settings than VIME-512 (dropout 0.1, full MSE, small batches). The **Fair-v2**
> variant unifies all training variables with VIME-512, isolating architecture as
> the only difference. Fair-v2 is the recommended variant for comparison purposes.

## Full Architecture (SSL + Classification)

```mermaid
flowchart TB
    subgraph INPUT["Input Layer"]
        I[("Input Features<br/>253 dimensions<br/>(122 original + 10 engineered<br/>+ one-hot + masks)")]
    end

    subgraph ENCODER["🔷 Encoder (Bottleneck)"]
        E1["Linear(253 → 256)"]
        BN1["BatchNorm1d(256)"]
        R1["ReLU"]

        E2["Linear(256 → 256)"]
        BN2["BatchNorm1d(256)"]
        R2["ReLU"]
        D1["Dropout(0.1)"]

        E3["Linear(256 → 128)"]
        BN3["BatchNorm1d(128)"]
        R3["ReLU"]

        LATENT[("Latent Space<br/>128-dim<br/>~1.9× compression")]
    end

    subgraph SSL["🔶 SSL Decoder (Pretraining)"]
        subgraph MASK_HEAD["Mask Prediction Head"]
            M1["Linear(128 → 253)"]
            MS["Sigmoid"]
            MO[("Mask Output<br/>253-dim<br/>P(corrupted)")]
        end

        subgraph RECON_HEAD["Reconstruction Head"]
            R_1["Linear(128 → 253)"]
            RO[("Reconstruction<br/>253-dim<br/>Feature values")]
        end
    end

    subgraph CLS["🔷 Classification Head (Fine-tuning)"]
        C1["Linear(128 → 64)"]
        CR1["ReLU"]
        CD1["Dropout(0.3)"]

        C2["Linear(64 → 32)"]
        CR2["ReLU"]
        CD2["Dropout(0.3)"]

        C3["Linear(32 → 1)"]
        SIG["Sigmoid"]
        OUT[("P(default)<br/>∈ [0, 1]")]
    end

    I --> E1 --> BN1 --> R1
    R1 --> E2 --> BN2 --> R2 --> D1
    D1 --> E3 --> BN3 --> R3 --> LATENT

    LATENT --> M1 --> MS --> MO
    LATENT --> R_1 --> RO

    LATENT --> C1 --> CR1 --> CD1
    CD1 --> C2 --> CR2 --> CD2
    CD2 --> C3 --> SIG --> OUT

    style INPUT fill:#e3f2fd
    style ENCODER fill:#bbdefb
    style SSL fill:#fff3e0
    style CLS fill:#c8e6c9
    style LATENT fill:#f3e5f5,stroke:#9c27b0,stroke-width:3px
```

## Parameter Count Breakdown

```mermaid
pie title VIME-128 Parameters (~193K total)
    "Encoder Layer 1 (253→256)" : 65024
    "Encoder Layer 2 (256→256)" : 65792
    "Encoder Layer 3 (256→128)" : 32896
    "Mask Head (128→253)" : 32637
    "Recon Head (128→253)" : 32637
    "Cls Head (128→64→32→1)" : 10369
```

## Training Pipeline

```mermaid
flowchart LR
    subgraph PHASE1["Phase 1: Self-Supervised Pretraining"]
        direction TB
        P1_IN["Input X"]
        P1_CORRUPT["Corrupt 30%<br/>(shuffle values)"]
        P1_ENC["Encoder"]
        P1_DEC["Decoder Heads"]
        P1_LOSS["Loss = 2×BCE_mask + MSE_recon"]

        P1_IN --> P1_CORRUPT --> P1_ENC --> P1_DEC --> P1_LOSS
    end

    subgraph PHASE2["Phase 2: Supervised Fine-tuning"]
        direction TB
        P2_IN["Input X"]
        P2_ENC["Encoder<br/>(pretrained, frozen initially)"]
        P2_CLS["Classification Head"]
        P2_LOSS["Focal Loss<br/>α=0.25, γ=2.0"]

        P2_IN --> P2_ENC --> P2_CLS --> P2_LOSS
    end

    PHASE1 -->|"Transfer<br/>Encoder Weights"| PHASE2

    style PHASE1 fill:#fff3e0
    style PHASE2 fill:#e8f5e9
```

## Hyperparameters

### Original VIME-128

| Parameter | SSL Phase | Classification Phase |
|-----------|-----------|---------------------|
| Learning Rate | 1e-3 | 1e-3 |
| Batch Size | 128 | 256 |
| Epochs | 20 | 50 (early stop ~11) |
| Optimizer | Adam | Adam |
| Dropout (encoder) | 0.1 | 0.1 |
| Dropout (head) | - | 0.3 |
| Early Stopping | - | 10 epochs patience |
| Recon Loss | Full MSE | - |
| Mask Loss weight | 2.0 | - |
| Sup Loss | BCE | - |

### Fair-v2 VIME-128 (Recommended for Comparison)

| Parameter | SSL Phase | Classification Phase |
|-----------|-----------|---------------------|
| Learning Rate | 1e-3 | 1e-3 |
| Batch Size | 512 | 256 |
| Epochs | 30 | 50 (early stop ~6) |
| Optimizer | AdamW | AdamW |
| Dropout (encoder) | 0.3 | 0.3 |
| Dropout (head) | - | 0.3 |
| Early Stopping | - | 10 epochs patience (P@20%) |
| Recon Loss | Masked MSE | - |
| Mask Loss weight (α) | 1.0 | - |
| Sup Loss | - | Focal (α=0.25, γ=2.0) |

> **Why Fair-v2?** The original VIME-128 differed from VIME-512 in 6 training
> variables, making it impossible to attribute performance differences to
> architecture alone. Fair-v2 unifies all training settings, isolating only
> latent dim (128 vs 512), hidden dim (256 vs 512), and decoder depth.

## Results Comparison

| Variant | Model Class | D_R AUC | D_R P@20% | D_T AUC | Best Val AUC |
|---------|------------|---------|-----------|---------|--------------|
| **VIME-512-Fair-v3** | VIME | **0.7058** | **0.9591** | **0.7188** | **0.7285** |
| VIME-128-Fair-v2 | VIME | 0.6956 | 0.9588 | 0.7171 | 0.7310 |
| VIME-128 (original) | VIME | 0.6856 | 0.9522 | 0.7079 | 0.7213 |
| VIME-512 (G-Wide) | VIMEWide | 0.6822 | 0.9548 | 0.7125 | 0.7217 |

**Key findings (paired bootstrap, n=1000):**
- **512 > 128 (same class):** VIME-512-Fair-v3 significantly outperforms VIME-128-Fair-v2 (ΔAUC = +0.0102, 95% CI [+0.0080, +0.0125]). More capacity helps.
- **VIMEWide hurts:** VIME-512-Fair-v3 (simple VIME) outperforms VIME-512/G-Wide (VIMEWide) by ΔAUC = +0.0236. The mirror decoder architecture is harmful.
- **Architecture > capacity:** The mirror decoder effect (−0.0236) is 2× the capacity effect (+0.0102).
