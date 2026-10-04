# Reject inference for credit risk

Declining applicants likely to default protects a lending portfolio, but declining someone who would repay is a missed opportunity. Rejected applicants have no observed repayment outcome, making it difficult to assess whether a different decision rule would admit more good borrowers.

Developed as a deep learning course project, this study compares neural network approaches with CatBoost in a simulated reject inference setting. Its documented decisions cover the dataset, rejection rule, evaluation measures and a more controlled VIME comparison. The repository contains the experiments, saved results and a [final report](docs/reports/final_report.md).

## Study design

The study uses [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) application data. Applicants below the 30th percentile of `EXT_SOURCE_2` are treated as rejected. Their `TARGET` labels are hidden during training and retained for evaluation; rows without `EXT_SOURCE_2` are excluded from the study's split.

The dataset choice followed a concrete limitation: the earlier Kowope Mart reference data used anonymized fields. Home Credit has named application fields such as income, credit amount and occupation, so model inputs and explanations can be discussed in lending terms. Its `EXT_SOURCE_2` field also supports a transparent, score-like rejection rule. The [decision log](docs/decision_log.md) records the choice and the alternatives considered.

The cutoff does not represent observed bank decisions. It gives every evaluated applicant a known outcome, while allowing the training pipeline to treat the simulated rejected group as unlabeled.

## What was compared

- **CatBoost:** a tabular baseline using its native categorical handling, plus an ensemble variant.
- **VIME:** self-supervised representations followed by a default-risk classifier, including 128- and 512-dimensional variants.
- **DCN-v2:** a second neural network approach for tabular features.

CatBoost provides a tabular reference for judging whether the neural approaches add value. VIME can learn representations from simulated rejected applications without using their outcome labels; DCN-v2 tests a different way to model tabular feature interactions.

The evaluation uses AUC to measure ranking across the rejected group. **Precision@20%** measures the share of non-defaulting borrowers among the 20% predicted to have the lowest risk. That fixed selection share connects model performance to the practical question of which rejected applicants might merit another look. The [final report](docs/reports/final_report.md) contains the model tables, confidence intervals and calibration analysis.

## A result that changed the interpretation

The first comparison appeared to favor 128-dimensional VIME over 512-dimensional VIME. On February 26, the project records that these versions also differed in encoder and decoder structure, so the result could not isolate width. Fair v3 responded by using the same VIME model class, layer depths and training setup for both configurations. The wider configuration still changed both hidden and latent dimensions. In the [saved results](outputs/results/vime_512_fair_v3_results.json) and [paired comparison](outputs/results/paired_bootstrap_and_calibration.json), it performs better on rejected-group AUC than the 128-dimensional Fair v2 version. The revised conclusion is a better result for the wider configuration within this more closely matched architecture.

CatBoost remains the strongest of the reported approaches on rejected-group AUC. These findings apply to this dataset, simulated cutoff and recorded training runs; the neural network comparisons use a single seed. See the [February 26 decision record](docs/decision_log.md#2026-02-26--fair-v3-fully-controlled-comparison-same-model-class-for-128-and-512) for the sequence of decisions.

## Data flow and existing entry points

The current Home Credit code uses `data/raw/application_train.csv`, with `TARGET` as the outcome. `application_test.csv` is not used for this study; `bureau.csv` is optional in configuration. From the repository root, these are the existing entry points and their file dependencies:

| Stage | Entry point | Reads or writes |
|---|---|---|
| Shared split | `src/preprocessing/shared_pipeline.py` | Reads `application_train.csv`; writes approved train, validation and test splits plus rejected features and held-out ground truth under `data/processed/` |
| VIME preparation | `src/preprocessing/route_a_prep.py` | Reads those splits; writes `vime_*_fixed.parquet`, missingness masks and target files |
| CatBoost comparison | `scripts/train_catboost_fair.py` | Reads shared splits; writes `outputs/results/catboost_fair_results.json` |
| Controlled VIME comparison | `src/models/vime/train_vime_128_fair_v2.py` and `src/models/vime/train_vime_512_fair_v3.py` | Read fixed VIME features, masks and rejected ground truth for evaluation; write their respective result JSON files |

The top-level `run_experiment.py` belongs to an older Kowope Mart workflow: it looks for `kowope_mart.csv` and `Good_Bad`. It is not the entry point for the Home Credit results described here.

## Read the evidence

- [Final report](docs/reports/final_report.md): study design, results and interpretation.
- [Decision log](docs/decision_log.md): dataset choice, evaluation choices and the February 26 VIME controls.
- [Saved result files](outputs/results/): model metrics and paired bootstrap calculations.

The reported comparison comes from a single dataset and a single-feature rejection cutoff. Different lending populations or rejection mechanisms may give different results.
