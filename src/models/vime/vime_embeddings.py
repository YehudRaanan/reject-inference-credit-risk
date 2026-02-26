"""
Script to generate VIME embeddings for all data splits.
Loads the trained VIME encoder and processes 'train', 'val', 'test', and 'reject' splits.
Saves latent representations (z) to `data/processed/vime_emb_*.parquet`.
"""
import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import config
from src.models.vime.vime_model import VIME

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_embeddings():
    """
    Generates latent embeddings using the trained VIME encoder.
    Iterates through all data splits, runs inference, and saves the output.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Generating VIME Embeddings on {device}...")

    # 1. Load Model
    # We find input_dim by loading one file first
    train_path = config.DATA_PROCESSED / "vime_X_train.parquet"
    if not train_path.exists():
        logger.error("Training data not found.")
        return

    df_sample = pd.read_parquet(train_path)
    input_dim = df_sample.shape[1]
    
    model_path = config.MODEL_DIR / "vime_encoder.pth"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    model = VIME(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("VIME Encoder loaded successfully.")

    # 2. Process All Splits
    splits = ["train", "val", "test", "reject"]
    
    for split in splits:
        input_file = config.DATA_PROCESSED / f"vime_X_{split}.parquet"
        output_file = config.DATA_PROCESSED / f"vime_emb_{split}.parquet"
        
        if not input_file.exists():
            logger.warning(f"Input file {input_file} not found. Skipping.")
            continue
            
        logger.info(f"Processing {split} split...")
        
        # Load Data
        df = pd.read_parquet(input_file)
        X = torch.tensor(df.values, dtype=torch.float32).to(device)
        
        # Inference (Batch processing to avoid OOM)
        batch_size = 1024
        embeddings_list = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                z = model.encode(batch) # Returns latent vector only
                embeddings_list.append(z.cpu().numpy())
        
        # Concatenate
        embeddings = np.concatenate(embeddings_list, axis=0)
        
        # Save as DataFrame
        # Column names: emb_0, emb_1, ...
        emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
        df_emb = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
        
        df_emb.to_parquet(output_file)
        logger.info(f"Saved {output_file.name} ({df_emb.shape})")

    logger.info("All embeddings generated successfully.")

def generate_embeddings_fixed():
    """
    Generates latent embeddings using the VIME-Fixed encoder.
    Uses fixed-preprocessed data (vime_X_*_fixed.parquet) and the
    fixed model (vime_encoder_fixed.pth).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Generating VIME-Fixed Embeddings on {device}...")

    # 1. Load Model
    train_path = config.DATA_PROCESSED / "vime_X_train_fixed.parquet"
    if not train_path.exists():
        logger.error("Fixed training data not found. Run prepare_route_a_fixed() first.")
        return

    df_sample = pd.read_parquet(train_path)
    input_dim = df_sample.shape[1]

    model_path = config.MODEL_DIR / "vime_encoder_fixed.pth"
    if not model_path.exists():
        logger.error(f"Fixed model not found at {model_path}. Run train_vime_fixed() first.")
        return

    model = VIME(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"VIME-Fixed Encoder loaded successfully. Input dim: {input_dim}")

    # 2. Process All Splits
    splits = ["train", "val", "test", "reject"]

    for split in splits:
        input_file = config.DATA_PROCESSED / f"vime_X_{split}_fixed.parquet"
        output_file = config.DATA_PROCESSED / f"vime_emb_{split}_fixed.parquet"

        if not input_file.exists():
            logger.warning(f"Input file {input_file} not found. Skipping.")
            continue

        logger.info(f"Processing {split} split...")

        # Load Data
        df = pd.read_parquet(input_file)
        X = torch.tensor(df.values, dtype=torch.float32).to(device)

        # Inference (Batch processing to avoid OOM)
        batch_size = 1024
        embeddings_list = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                z = model.encode(batch)  # Returns latent vector only
                embeddings_list.append(z.cpu().numpy())

        # Concatenate
        embeddings = np.concatenate(embeddings_list, axis=0)

        # Save as DataFrame
        emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
        df_emb = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)

        df_emb.to_parquet(output_file)
        logger.info(f"Saved {output_file.name} ({df_emb.shape})")

    logger.info("All fixed embeddings generated successfully.")


if __name__ == "__main__":
    # Use fixed version by default
    generate_embeddings_fixed()
