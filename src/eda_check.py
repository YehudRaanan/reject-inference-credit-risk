"""
EDA Check Script — Home Credit Default Risk
===========================================

Performs initial validation of the dataset structure, features,
and target distribution to ensure readiness for the pipeline.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
DATA_DIR = Path("data/raw")
TRAIN_PATH = DATA_DIR / "application_train.csv"

def run_eda():
    print(f"Loading {TRAIN_PATH}...")
    df = pd.read_csv(TRAIN_PATH)
    print(f"Shape: {df.shape}")
    
    # 1. Target Distribution
    target_counts = df["TARGET"].value_counts(normalize=True)
    print("\n--- Target Distribution ---")
    print(target_counts)
    default_rate = target_counts.get(1, 0)
    print(f"Default Rate: {default_rate:.2%}")
    
    # 2. EXT_SOURCE_2 Analysis (for Reject Simulation)
    print("\n--- EXT_SOURCE_2 Analysis ---")
    if "EXT_SOURCE_2" in df.columns:
        desc = df["EXT_SOURCE_2"].describe()
        print(desc)
        missing = df["EXT_SOURCE_2"].isnull().mean()
        print(f"Missing Rate: {missing:.2%}")
        
        # Check 30th percentile cutoff
        cutoff = df["EXT_SOURCE_2"].quantile(0.30)
        print(f"30th Percentile (Proposed Cutoff): {cutoff:.4f}")
        
        # Visualize
        # plt.figure(figsize=(10, 6))
        # sns.histplot(data=df, x="EXT_SOURCE_2", hue="TARGET", kde=True, common_norm=False)
        # plt.title(f"EXT_SOURCE_2 Distribution by Target (Cutoff: {cutoff:.4f})")
        # plt.axvline(cutoff, color='r', linestyle='--', label=f'Cutoff ({cutoff:.4f})')
        # plt.legend()
        # plt.savefig("outputs/figures/ext_source_2_dist.png")
        # print("Saved distribution plot to outputs/figures/ext_source_2_dist.png")
    else:
        print("WARNING: EXT_SOURCE_2 not found!")

    # 3. Column checks for shared_pipeline
    print("\n--- Column Availability Check ---")
    required_cols = [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "OWN_CAR_AGE", 
        "EXT_SOURCE_1", "EXT_SOURCE_3"
    ]
    for col in required_cols:
        present = col in df.columns
        print(f"{col}: {'OK' if present else 'MISSING'}")

    # 4. Missing Value Patterns
    print("\n--- Top Missing Columns ---")
    missing_rates = df.isnull().mean().sort_values(ascending=False)
    print(missing_rates.head(10))

if __name__ == "__main__":
    try:
        run_eda()
    except Exception as e:
        print(f"EDA Failed: {e}")
