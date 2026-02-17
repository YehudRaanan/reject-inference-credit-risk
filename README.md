# Reject Inference for Credit Risk: Neural Networks vs GBDTs

A methodological comparison of neural network approaches (VIME, DCN-v2) against gradient boosted decision trees (CatBoost) for reject inference in credit risk assessment.

## Key Findings

| Model | D_R AUC | D_R P@20% | Diamonds Found |
|-------|---------|-----------|----------------|
| **CatBoost** | **0.7149** [0.7103, 0.7195] | **96.6%** | **17,914** |
| CatBoost-Ensemble | 0.7096 [0.7050, 0.7141] | 96.4% | 17,878 |
| DCN-v2 | 0.6966 | 95.7% | 17,741 |
| VIME-128 | 0.6902 [0.6852, 0.6952] | 95.5% | 17,701 |
| VIME-512 | 0.6891 [0.6845, 0.6938] | 95.8% | 17,760 |

*95% bootstrap confidence intervals in brackets*

**CatBoost wins across all metrics.** Neural networks consistently underperform GBDTs on this tabular credit scoring task.

## Problem: Reject Inference

Credit institutions only observe outcomes (default/repayment) for approved applicants. Rejected applicants represent an information gap. **Reject inference** attempts to find "diamonds in the mud" - good borrowers among those rejected by simple score thresholds.

### Experiment Setup

- **Dataset:** Home Credit Default Risk (307K applications, 122 features)
- **D_L (Approved):** Top 70% by EXT_SOURCE_2 score (labels visible)
- **D_R (Rejected):** Bottom 30% by score (labels hidden during training)
- **Goal:** Identify good borrowers (non-defaulters) in the rejected population

## Models Compared

### CatBoost (Baseline)
- Native categorical handling
- Native missing value handling
- No dimensionality reduction

### VIME (Self-Supervised)
- VIME-128: 253 -> 256 -> 256 -> 128 bottleneck
- VIME-512: 253 -> 512 -> 512 (no bottleneck)
- Self-supervised pretraining + classification head

### DCN-v2 (Deep & Cross Network)
- Explicit polynomial interactions + deep network
- SoftBinning for numerical features
- Native categorical embeddings

## Key Insights

1. **VIME Bottleneck Hypothesis Rejected:** VIME-512 performs *worse* than VIME-128, ruling out the 128-dim bottleneck as the limitation

2. **Self-Supervised Mismatch:** Reconstruction pretext task doesn't align with classification objective

3. **GBDTs Handle Tabular Data Better:** More data-efficient, native handling of categoricals and missing values

4. **Early Stopping Metric Matters:** AUC favors ensembles (smoothed predictions), P@20% favors single models (concentrated predictions)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Data Setup

1. Download `application_train.csv` from [Kaggle Home Credit](https://www.kaggle.com/c/home-credit-default-risk/data)
2. Place in `data/raw/`

### Load Pre-trained Models

```python
from catboost import CatBoostClassifier

# Load CatBoost
model = CatBoostClassifier()
model.load_model('outputs/models/catboost/catboost_fair.cbm')
```

```python
import torch
from src.models.vime.model import VIME
from src.models.vime.classifier import ClassificationHead128

# Load VIME-128
encoder = VIME(input_dim=253, embedding_dim=128, hidden_dim=256)
encoder.load_state_dict(torch.load('outputs/models/vime/vime_encoder_fixed.pth'))

checkpoint = torch.load('outputs/models/vime/vime_128_classifier.pth')
classifier = ClassificationHead128()
classifier.load_state_dict(checkpoint['head_state_dict'])
```

## Project Structure

```
reject-inference-credit-risk/
├── config.py                 # Central configuration
├── requirements.txt          # Dependencies
├── data/
│   ├── raw/                  # Download instructions
│   └── processed/            # Preprocessed parquet files
│       ├── splits/           # D_L_train, D_L_val, D_L_test, D_R
│       ├── vime/             # VIME preprocessing
│       ├── dcn/              # DCN-v2 preprocessing
│       └── metadata/         # JSON metadata
├── src/
│   ├── preprocessing/        # Data preprocessing
│   ├── models/
│   │   ├── catboost/         # CatBoost trainer
│   │   ├── vime/             # VIME model + classifier
│   │   └── dcn_v2/           # DCN-v2 model
│   └── evaluation/           # Metrics and bootstrap CI
├── outputs/
│   ├── models/               # Trained weights (~100MB)
│   └── results/              # JSON evaluation results
└── scripts/                  # Training scripts
```

## Fair Comparison Setup

All models trained with equivalent setup:
- **Loss:** Focal Loss (alpha=0.25) or equivalent class weights
- **Early Stopping:** Precision@20% on validation set
- **Evaluation:** AUC, P@20%, Diamond count on D_R

## Citation

If you use this code, please cite:

```
@misc{reject-inference-2026,
  title={Reject Inference for Credit Risk: Neural Networks vs GBDTs},
  year={2026},
  url={https://github.com/[username]/reject-inference-credit-risk}
}
```

## References

- [VIME: Extending Self- and Semi-supervised Learning to Tabular Domain](https://arxiv.org/abs/2006.08856) (NeurIPS 2020)
- [DCN V2: Improved Deep & Cross Network](https://arxiv.org/abs/2008.13535) (WWW 2021)
- [CatBoost: Gradient Boosting with Categorical Features](https://arxiv.org/abs/1706.09516)

## License

MIT License
