"""
VIME (Value Imputation and Mask Estimation) Autoencoder.

Architecture:
- Encoder: Input (d) -> Hidden (d) -> Embedding (z)
- Mask Estimator: Embedding (z) -> Mask Prediction (d) [Sigmoid]
- Feature Reconstructor: Embedding (z) -> Feature Reconstruction (d)

Reference: Jinsung Yoon et al., "VIME: Extending the Success of Self- and
Semi-supervised Learning to Tabular Domain", NeurIPS 2020.
"""
import torch
import torch.nn as nn


class VIME(nn.Module):
    """
    VIME 128-dim encoder architecture.

    Architecture:
        Encoder: input_dim -> 256 -> 256 -> 128 (bottleneck)
        Mask head: 128 -> input_dim (sigmoid)
        Recon head: 128 -> input_dim
    """

    def __init__(self, input_dim=253, embedding_dim=128, hidden_dim=256):
        super(VIME, self).__init__()

        # --- Encoder Layers ---
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_relu1 = nn.ReLU()

        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_relu2 = nn.ReLU()

        self.enc_dropout = nn.Dropout(0.1)

        # Latent layer (bottleneck)
        self.enc_fc3 = nn.Linear(hidden_dim, embedding_dim)

        # --- Mask Estimator Head ---
        self.mask_fc = nn.Linear(embedding_dim, input_dim)

        # --- Feature Reconstructor Head ---
        self.recon_fc = nn.Linear(embedding_dim, input_dim)

    def encode(self, x):
        """Returns the latent vector (z)."""
        x = self.enc_relu1(self.enc_fc1(x))
        x = self.enc_relu2(self.enc_fc2(x))
        x = self.enc_dropout(x)
        latent_vector = self.enc_fc3(x)
        return latent_vector

    def forward(self, x):
        """Full forward pass returning mask and reconstruction."""
        latent_vector = self.encode(x)
        m_hat = torch.sigmoid(self.mask_fc(latent_vector))
        x_hat = self.recon_fc(latent_vector)
        return m_hat, x_hat


class VIMEWide(nn.Module):
    """
    VIME 512-dim encoder architecture (no bottleneck).

    Architecture:
        Encoder: input_dim -> 512 -> 512 (no compression)
        Mask head: 512 -> 512 -> input_dim (mirror)
        Recon head: 512 -> 512 -> input_dim (mirror)
    """

    def __init__(
        self,
        input_dim: int = 253,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # === Encoder ===
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_bn1 = nn.BatchNorm1d(hidden_dim)
        self.enc_relu1 = nn.ReLU()
        self.enc_drop1 = nn.Dropout(dropout)

        self.enc_fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.enc_bn2 = nn.BatchNorm1d(embedding_dim)
        self.enc_relu2 = nn.ReLU()
        self.enc_drop2 = nn.Dropout(dropout)

        # === Mask Estimator (mirror architecture) ===
        self.mask_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.mask_bn1 = nn.BatchNorm1d(hidden_dim)
        self.mask_relu1 = nn.ReLU()
        self.mask_drop1 = nn.Dropout(dropout)
        self.mask_fc2 = nn.Linear(hidden_dim, input_dim)

        # === Feature Reconstructor (mirror architecture) ===
        self.recon_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.recon_bn1 = nn.BatchNorm1d(hidden_dim)
        self.recon_relu1 = nn.ReLU()
        self.recon_drop1 = nn.Dropout(dropout)
        self.recon_fc2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features to latent representation."""
        h = self.enc_fc1(x)
        h = self.enc_bn1(h)
        h = self.enc_relu1(h)
        h = self.enc_drop1(h)

        z = self.enc_fc2(h)
        z = self.enc_bn2(z)
        z = self.enc_relu2(z)
        z = self.enc_drop2(z)

        return z

    def forward(self, x: torch.Tensor):
        """Full forward pass returning mask and reconstruction."""
        z = self.encode(x)

        # Mask head
        h = self.mask_fc1(z)
        h = self.mask_bn1(h)
        h = self.mask_relu1(h)
        h = self.mask_drop1(h)
        m_hat = torch.sigmoid(self.mask_fc2(h))

        # Recon head
        h = self.recon_fc1(z)
        h = self.recon_bn1(h)
        h = self.recon_relu1(h)
        h = self.recon_drop1(h)
        x_hat = self.recon_fc2(h)

        return m_hat, x_hat
