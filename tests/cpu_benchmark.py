
import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from models.dcn_v2 import DCNv2

# Dataset & Collate (Reuse)
class DCNv2Dataset(Dataset):
    def __init__(self, X_vals, y_vals, feature_names):
        self.X = X_vals
        self.y = y_vals
        self.feature_names = feature_names
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        row = self.X[idx]
        vals = {name: torch.tensor(row[i]) for i, name in enumerate(self.feature_names)}
        label = torch.tensor(self.y[idx])
        return vals, label

def dcnv2_collate_fn(batch):
    numerical_dicts, labels = zip(*batch)
    feature_names = numerical_dicts[0].keys()
    batched_numerical = {
        name: torch.stack([d[name] for d in numerical_dicts])
        for name in feature_names
    }
    batched_labels = torch.stack(labels).unsqueeze(1)
    return batched_numerical, batched_labels

def run_benchmark():
    print("=== DCN-v2 CPU Benchmark ===")
    
    # Parameters
    N_SAMPLES_BENCHMARK = 5_000   # Run on 5k samples
    N_SAMPLES_FULL = 246_000      # Approx size of Home Credit Train set (80% of 307k)
    BATCH_SIZE = 128              # CPU batch size usually smaller than GPU (1024)
    N_FEATURES = 253
    
    # Check Device
    if torch.cuda.is_available():
        print("WARNING: CUDA is available but forcing CPU for this benchmark.")
    device = torch.device('cpu')
    
    # 1. Generate Mock Data
    print(f"\n1. Generating mock data ({N_SAMPLES_BENCHMARK} samples, {N_FEATURES} features)...")
    feature_names = [f'f_{i}' for i in range(N_FEATURES)]
    X = np.random.randn(N_SAMPLES_BENCHMARK, N_FEATURES).astype(np.float32)
    y = np.random.randint(0, 2, size=N_SAMPLES_BENCHMARK).astype(np.float32)
    
    dataset = DCNv2Dataset(X, y, feature_names)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dcnv2_collate_fn)
    
    # 2. Model
    print("2. Instantiating Model...")
    model = DCNv2(
        numerical_features=feature_names,
        categorical_cardinalities={},
        num_bins=10, 
        numerical_embed_dim=16,
        cross_layers=3, 
        cross_rank=64, 
        deep_hidden=[512, 256, 128], 
        dropout=0.1
    ).to(device)
    
    for feat in feature_names:
        # cutpoints should have size (num_bins - 1)
        n_cutpoints = model.embedding.numerical_binning[feat].num_bins - 1
        model.embedding.numerical_binning[feat].cutpoints.data = torch.randn(n_cutpoints).sort()[0]

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 3. Training Loop Benchmark
    print("\n3. Starting Benchmark (1 Epoch on 5k samples)...")
    model.train()
    
    start_time = time.time()
    
    n_batches = 0
    for batch_idx, (num_vals, labels) in enumerate(loader):
        num_vals = {k: v.to(device) for k, v in num_vals.items()}
        labels = labels.to(device)
        
        # Forward
        x0 = model.embedding(num_vals, {})
        cross = model.cross_network(x0)
        deep = model.deep_network(x0)
        combined = torch.cat([cross, deep], dim=1)
        logits = model.head(combined)
        
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        n_batches += 1
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx}/{len(loader)}...", end='\r')
            
    end_time = time.time()
    duration = end_time - start_time
    
    # 4. Results
    samples_per_sec = N_SAMPLES_BENCHMARK / duration
    estimated_full_epoch = N_SAMPLES_FULL / samples_per_sec
    
    print(f"\n\nResults:")
    print(f"   Time for {N_SAMPLES_BENCHMARK} samples: {duration:.2f} seconds")
    print(f"   Throughput: {samples_per_sec:.1f} samples/sec")
    print(f"   Estimated Time for Full Epoch ({N_SAMPLES_FULL} samples): {estimated_full_epoch/60:.2f} minutes")
    print(f"   Estimated Time for 50 Epochs: {estimated_full_epoch*50/3600:.2f} hours")

if __name__ == "__main__":
    run_benchmark()
