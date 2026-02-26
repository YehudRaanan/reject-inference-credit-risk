
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import IsolationForest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from models.dcn_v2 import DCNv2, compute_quantiles_for_binning

# --- Mock Data Generation ---
def generate_mock_data(n_samples=100, n_features=253):
    feature_names = [f'f_{i}' for i in range(n_features)]
    # Random floats resembling standardized/target-encoded data
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Random binary labels
    y = np.random.randint(0, 2, size=n_samples).astype(np.float32)
    
    df_X = pd.DataFrame(X, columns=feature_names)
    df_y = pd.Series(y, name='TARGET')
    return df_X, df_y, feature_names

# --- Replicating Notebook Classes ---
class DCNv2Dataset(Dataset):
    def __init__(self, X_df, y_df=None, feature_names=None):
        self.X = X_df.values.astype(np.float32)
        if y_df is not None:
            self.y = y_df.values.astype(np.float32).flatten()
        else:
            self.y = None
        self.feature_names = feature_names
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        row = self.X[idx]
        numerical_values = {name: torch.tensor(row[i]) for i, name in enumerate(self.feature_names)}
        if self.y is not None:
            return numerical_values, torch.tensor(self.y[idx])
        else:
            return numerical_values

def dcnv2_collate_fn(batch):
    has_labels = isinstance(batch[0], tuple)
    if has_labels:
        numerical_dicts, labels = zip(*batch)
    else:
        numerical_dicts = batch
        labels = None
    
    feature_names = numerical_dicts[0].keys()
    batched_numerical = {
        name: torch.stack([d[name] for d in numerical_dicts])
        for name in feature_names
    }
    
    if labels is not None:
        batched_labels = torch.stack(labels).unsqueeze(1)
        return batched_numerical, batched_labels
    else:
        return batched_numerical

# --- Test Execution ---
def test_pipeline():
    print("=== DCN-v2 Pipeline Dry Run ===\n")
    
    # 1. Generate Data
    print("1. Generating mock data (253 features)...")
    X_train, y_train, features = generate_mock_data(n_samples=50)
    X_reject, _, _ = generate_mock_data(n_samples=20)
    print(f"   Train shape: {X_train.shape}, Reject shape: {X_reject.shape}")

    # 2. Model Instantiation
    print("\n2. Instantiating DCN-v2 Model...")
    model = DCNv2(
        numerical_features=features,
        categorical_cardinalities={},
        num_bins=5, # Small bins for quick test
        numerical_embed_dim=4,
        cross_layers=2,
        cross_rank=16,
        deep_hidden=[32, 16, 8],
        dropout=0.1
    )
    
    # Compute Quantiles (Mock)
    quantiles_dict = {}
    for feat in features:
        values = X_train[feat].values
        # Simple quantiles
        q_vals = torch.tensor(np.quantile(values, np.linspace(0, 1, 5 + 1)[1:-1]), dtype=torch.float32)
        model.embedding.numerical_binning[feat].cutpoints.data = q_vals
        
    print("   Model created. Params:", sum(p.numel() for p in model.parameters()))

    # 3. Training Loop Test (Teacher)
    print("\n3. Testing Training Step (Teacher)...")
    dataset = DCNv2Dataset(X_train, y_train, features)
    loader = DataLoader(dataset, batch_size=10, collate_fn=dcnv2_collate_fn)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    model.train()
    for batch_idx, (num_vals, labels) in enumerate(loader):
        # Forward
        x0 = model.embedding(num_vals, {})
        cross = model.cross_network(x0)
        deep = model.deep_network(x0)
        combined = torch.cat([cross, deep], dim=1)
        logits = model.head(combined)
        
        # Loss
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")
        break # Just one batch needed
        
    # 4. Reject Inference Flow
    print("\n4. Testing Reject Inference Mock Flow...")
    
    # Embedding Generation
    model.eval()
    reject_dataset = DCNv2Dataset(X_reject, feature_names=features)
    reject_loader = DataLoader(reject_dataset, batch_size=10, collate_fn=dcnv2_collate_fn)
    
    train_embeddings = []
    reject_embeddings = []
    reject_preds = []
    
    with torch.no_grad():
        # Train embeddings (for IF)
        for num_vals, _ in loader:
            x0 = model.embedding(num_vals, {})
            train_embeddings.append(x0.numpy())
            
        # Reject embeddings & preds
        for num_vals in reject_loader:
            x0 = model.embedding(num_vals, {})
            probs = model(num_vals, {})
            reject_embeddings.append(x0.numpy())
            reject_preds.append(probs.numpy())
            
    emb_train_flat = np.vstack(train_embeddings)
    emb_reject_flat = np.vstack(reject_embeddings)
    preds_reject_flat = np.vstack(reject_preds).flatten()
    
    print(f"   Collected Embeddings: Train {emb_train_flat.shape}, Reject {emb_reject_flat.shape}")
    
    # Isolation Forest
    print("\n5. Testing Isolation Forest...")
    clf = IsolationForest(n_estimators=10, random_state=42)
    clf.fit(emb_train_flat)
    inlier_mask = clf.predict(emb_reject_flat) == 1
    print(f"   Inliers found: {inlier_mask.sum()}/{len(inlier_mask)}")
    
    # Pseudo Labeling
    print("\n6. Testing Pseudo-Labeling...")
    # Just mock threshold
    pseudo_labels = (preds_reject_flat[inlier_mask] > 0.5).astype(np.float32)
    print(f"   Pseudo-labels generated: {pseudo_labels.shape}")
    
    print("\n=== SUCCESS: Pipeline Verification Complete ===")

if __name__ == "__main__":
    test_pipeline()
