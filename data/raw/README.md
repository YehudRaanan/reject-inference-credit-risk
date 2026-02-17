# Raw Data

## Data Download Instructions

The raw data is not included in this repository due to its size. Please download it from Kaggle:

### Steps:

1. Go to the [Home Credit Default Risk competition page](https://www.kaggle.com/c/home-credit-default-risk/data)
2. Sign in to your Kaggle account (create one if needed)
3. Accept the competition rules
4. Download `application_train.csv` (307,511 rows, ~150MB)
5. Place it in this folder (`data/raw/`)

### Alternative: Kaggle CLI

```bash
# Install Kaggle CLI
pip install kaggle

# Configure API token (see: https://www.kaggle.com/docs/api)
# Then download:
kaggle competitions download -c home-credit-default-risk -f application_train.csv
unzip application_train.csv.zip -d data/raw/
```

### File Details

| File | Rows | Columns | Size |
|------|------|---------|------|
| `application_train.csv` | 307,511 | 122 | ~150MB |

### Data Description

The Home Credit Default Risk dataset contains loan applications with:
- **Target**: Binary classification (1 = default, 0 = non-default)
- **Features**: 16 categorical + 106 numerical
- **Class imbalance**: ~8% default rate
