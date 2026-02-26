"""
Embedding Extraction (Step 2 — final stage).

After self-supervised training, pass all data through the encoder
to generate latent vectors for downstream tasks.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .encoder import VIMEModel


def extract_embeddings(
    model: VIMEModel,
    data: np.ndarray,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Extract latent embeddings from a trained VIME encoder.

    Args:
        model:  Trained VIMEModel
        data:   numpy array (n_samples, n_features)

    Returns:
        embeddings: numpy array (n_samples, embed_dim)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    tensor_data = torch.FloatTensor(data)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            z = model.encoder(batch_x)
            all_embeddings.append(z.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)
