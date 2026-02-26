"""
VIME Encoder Architecture (Step 2).

Contains:
- VIMEEncoder:         Shared encoder backbone → latent embeddings
- MaskEstimator:       Head that predicts which features were corrupted
- FeatureReconstructor: Head that reconstructs original feature values
"""
import torch
import torch.nn as nn


class VIMEEncoder(nn.Module):
    """
    Shared encoder backbone.
    Maps raw tabular features → latent embedding of dimension `embed_dim`.
    """

    def __init__(self, input_dim: int, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MaskEstimator(nn.Module):
    """
    Predicts the binary corruption mask from the latent embedding.
    Output dimension = input_dim (one probability per original feature).
    """

    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class FeatureReconstructor(nn.Module):
    """
    Reconstructs the original feature values from the latent embedding.
    Output dimension = input_dim.
    """

    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class VIMEModel(nn.Module):
    """
    Complete VIME model combining encoder + both heads.
    """

    def __init__(self, input_dim: int, embed_dim: int = 128):
        super().__init__()
        self.encoder = VIMEEncoder(input_dim, embed_dim)
        self.mask_estimator = MaskEstimator(embed_dim, input_dim)
        self.feature_reconstructor = FeatureReconstructor(embed_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            z:      latent embedding
            m_hat:  predicted corruption mask
            x_hat:  reconstructed features
        """
        z = self.encoder(x)
        m_hat = self.mask_estimator(z)
        x_hat = self.feature_reconstructor(z)
        return z, m_hat, x_hat
