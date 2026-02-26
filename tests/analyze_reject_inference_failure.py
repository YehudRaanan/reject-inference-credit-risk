
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.dcn_v2 import DCNv2

# Define Paths
PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(PROJECT_PATH, 'data', 'processed')
MODELS_PATH = os.path.join(PROJECT_PATH, 'outputs', 'models')
RESULTS_PATH = os.path.join(PROJECT_PATH, 'outputs', 'results')
TEACHER_PATH = os.path.join(MODELS_PATH, 'dcn_v2_best.pth')

def run_analysis():
    print("=== DCN-v2 Failure Analysis (Local CPU) ===")
    
    # 1. Load Data
    print("1. Loading Data...")
    try:
        X_train = pd.read_parquet(os.path.join(DATA_PATH, 'vime_X_train_fixed.parquet'))
        X_reject = pd.read_parquet(os.path.join(DATA_PATH, 'vime_X_reject_fixed.parquet'))
        df_reject_gt = pd.read_parquet(os.path.join(DATA_PATH, 'D_R_ground_truth.parquet'))
        y_reject_gt = df_reject_gt['TARGET'].values
        NUMERICAL_FEATURES = X_train.columns.tolist()
        print(f"   Loaded {len(X_reject)} rejected samples.")
    except Exception as e:
        print(f"ERROR: Could not load data. {e}")
        return

    # 2. Load Model
    print("2. Loading Teacher Model...")
    try:
        # Try to load HP from results, else hardcode
        with open(os.path.join(RESULTS_PATH, 'dcn_v2_results.json'), 'r') as f:
            results = json.load(f)
            hp = results['hyperparameters']
            
        model = DCNv2(
            numerical_features=NUMERICAL_FEATURES,
            categorical_cardinalities={},
            num_bins=hp['num_bins'],
            numerical_embed_dim=hp['numerical_embed_dim'],
            cross_layers=hp['cross_layers'],
            cross_rank=hp['cross_rank'],
            deep_hidden=hp['deep_hidden'],
            dropout=hp['dropout']
        )
        
        checkpoint = torch.load(TEACHER_PATH, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("   Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load model. {e}")
        return

    # 3. Generate Predictions (CPU)
    print("3. Generating Predictions (CPU)...")
    
    class DCNv2Dataset(Dataset):
        def __init__(self, X_df, feature_names):
            self.X = X_df.values.astype(np.float32)
            self.feature_names = feature_names
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            row = self.X[idx]
            return {name: torch.tensor(row[i]) for i, name in enumerate(self.feature_names)}

    def collate(batch):
        numerical_dicts = batch
        feature_names = numerical_dicts[0].keys()
        return {name: torch.stack([d[name] for d in numerical_dicts]) for name in feature_names}

    dataset = DCNv2Dataset(X_reject, NUMERICAL_FEATURES)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate)
    
    all_preds = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"   Batch {i}/{len(loader)}...", end='\r')
            probs = model(batch, {})
            all_preds.extend(probs.numpy().flatten())
    
    preds = np.array(all_preds)
    print("\n   Predictions complete.")

    # 4. Analysis
    print("\n4. Root Cause Analysis")
    
    # 4.1 Actual vs Assumed Default Rate
    actual_rate = y_reject_gt.mean()
    print(f"   ACTUAL Default Rate in D_R: {actual_rate:.1%}")
    
    # 4.2 Replicate Pseudo-Labeling (Top 30%)
    threshold = np.quantile(preds, 0.70)
    pseudo_labels = (preds >= threshold).astype(int)
    print(f"   ASSUMED Default Rate (Pseudo): {pseudo_labels.mean():.1%} (Threshold={threshold:.4f})")
    
    # 4.3 Confusion Matrix
    cm = confusion_matrix(y_reject_gt, pseudo_labels)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n   --- Confusion Matrix (Pseudo-Labels vs Reality) ---")
    print(f"   True Positives (Correctly labeled Bad): {tp}")
    print(f"   False Negatives (ACTUAL BAD LABELED GOOD): {fn}")
    print(f"   -> We missed {fn} bad borrowers ({fn/(fn+tp):.1%} of all bads)!")
    
    if actual_rate > 0.4:
        print("\n   CONCLUSION: The Rejected population is MUCH riskier than assumed.")
        print("   By forcing a 30% default rate, we filtered out valid bad borrowers.")
    elif fn > tp:
        print("\n   CONCLUSION: The model ranking is poor in the rejected region.")
    else:
        print("\n   CONCLUSION: The strategy itself introduced noise.")

if __name__ == "__main__":
    run_analysis()
