# CatBoost Architecture

## Single Model Architecture

```mermaid
flowchart TB
    subgraph INPUT["Input Features"]
        NUM[("Numerical Features<br/>116 columns<br/>(106 + 10 engineered)")]
        CAT[("Categorical Features<br/>16 columns<br/>raw string values")]
    end

    subgraph PREPROCESS["Preprocessing"]
        NAN_NUM["Numerical NaN → Median"]
        NAN_CAT["Categorical NaN → '_MISSING_'"]
    end

    subgraph CATBOOST["🌲 CatBoost Model"]
        subgraph ENCODING["Native Categorical Handling"]
            OTS["Ordered Target Statistics<br/>(prevents target leakage)"]
            SPLITS["Optimal Category Splits<br/>(learned during training)"]
        end

        subgraph TREES["Gradient Boosted Trees"]
            T1["Tree 1<br/>depth=6"]
            T2["Tree 2<br/>depth=6"]
            T3["..."]
            TN["Tree ~750<br/>depth=6<br/>(early stopped)"]

            T1 --> T2 --> T3 --> TN
        end

        subgraph AGGREGATE["Prediction Aggregation"]
            SUM["Σ tree predictions<br/>(with learning rate)"]
            SIGMOID["Sigmoid"]
        end
    end

    subgraph OUTPUT["Output"]
        PRED[("P(default)<br/>∈ [0, 1]")]
    end

    NUM --> NAN_NUM --> CATBOOST
    CAT --> NAN_CAT --> CATBOOST
    ENCODING --> TREES
    TREES --> SUM --> SIGMOID --> PRED

    style INPUT fill:#e3f2fd
    style PREPROCESS fill:#fff3e0
    style CATBOOST fill:#c8e6c9
    style OUTPUT fill:#f3e5f5
```

## Ensemble Architecture

```mermaid
flowchart TB
    subgraph INPUT["Input Features (132 total)"]
        EXT[("EXT_SOURCE Features<br/>9 columns")]
        OTHER[("OTHER Features<br/>123 columns")]
    end

    subgraph MODEL_EXT["🌲 Model EXT"]
        EXT_TREES["500 trees<br/>depth=4"]
        EXT_PRED[("P_EXT")]
    end

    subgraph MODEL_OTHER["🌲 Model OTHER"]
        OTHER_TREES["1000 trees<br/>depth=6"]
        OTHER_PRED[("P_OTHER")]
    end

    subgraph ENSEMBLE["Ensemble Combination"]
        WEIGHT["Weighted Average"]
        FORMULA["P = 0.3 × P_EXT + 0.7 × P_OTHER"]
    end

    subgraph OUTPUT["Output"]
        FINAL[("P(default)<br/>∈ [0, 1]")]
    end

    EXT --> MODEL_EXT --> EXT_PRED --> WEIGHT
    OTHER --> MODEL_OTHER --> OTHER_PRED --> WEIGHT
    WEIGHT --> FORMULA --> FINAL

    style INPUT fill:#e3f2fd
    style MODEL_EXT fill:#ffecb3
    style MODEL_OTHER fill:#c8e6c9
    style ENSEMBLE fill:#f3e5f5
```

## Training Process

```mermaid
flowchart LR
    subgraph TRAINING["Gradient Boosting Process"]
        direction TB
        INIT["Initialize: f₀(x) = 0"]
        ITER1["Iteration 1:<br/>Fit tree to residuals"]
        ITER2["Iteration 2:<br/>Fit tree to new residuals"]
        ITERN["Iteration n:<br/>Continue until<br/>early stopping"]

        INIT --> ITER1 --> ITER2 --> ITERN
    end

    subgraph EARLY_STOP["P@20% Early Stopping"]
        direction TB
        CHUNK["Train 50 iterations"]
        EVAL["Evaluate P@20%<br/>on validation"]
        CHECK{"P@20%<br/>improved?"}
        CONT["Continue"]
        STOP["Stop & restore<br/>best weights"]

        CHUNK --> EVAL --> CHECK
        CHECK -->|Yes| CONT --> CHUNK
        CHECK -->|No, patience exceeded| STOP
    end

    TRAINING --> EARLY_STOP

    style TRAINING fill:#e8f5e9
    style EARLY_STOP fill:#fff3e0
```

## Hyperparameters

### Single Model
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Iterations | 2000 (max) | Maximum trees |
| Actual Iterations | ~750 | After early stopping |
| Depth | 6 | Tree depth |
| Learning Rate | 0.05 | Shrinkage |
| Loss | Logloss | Binary cross-entropy |
| Class Weights | {0: 0.75, 1: 0.25} | Imbalance handling |
| Early Stopping | P@20%, 200 rounds | |
| Categorical Features | 16 columns | Native handling |

### Ensemble
| Component | Model EXT | Model OTHER |
|-----------|-----------|-------------|
| Features | 9 | 123 |
| Iterations | 500 | 1000 |
| Depth | 4 | 6 |
| Early Stopping | 50 rounds | 100 rounds |
| Ensemble Weight | 0.3 | 0.7 |

## Why CatBoost Wins

```mermaid
flowchart TB
    subgraph ADVANTAGES["CatBoost Advantages"]
        A1["Native categorical handling<br/>(no information loss)"]
        A2["Ordered target statistics<br/>(prevents leakage)"]
        A3["No dimensionality reduction<br/>(all 132 features available)"]
        A4["Exact threshold splits<br/>(optimal for axis-aligned boundaries)"]
        A5["Feature interactions<br/>(via tree depth)"]
        A6["Sample efficiency<br/>(~200K samples sufficient)"]
    end

    subgraph RESULTS["Performance"]
        R1["D_R AUC: 0.7154"]
        R2["P@20%: 96.6%"]
        R3["Brier: 0.1152"]
        R4["ECE: 0.0724"]
    end

    ADVANTAGES --> RESULTS

    style ADVANTAGES fill:#c8e6c9
    style RESULTS fill:#e8f5e9
```
