"""
VIME-Wide: No-Bottleneck VIME Architecture
============================================

Route G-Wide: Tests if removing the 128-dim bottleneck improves performance.

Architecture comparison:
    Original VIME:  253 → 256 → 256 → 128 (bottleneck) → heads
    VIME-Wide:      253 → 512 → 512 (no bottleneck) → mirror heads

Key differences:
    1. Latent dim: 512 (vs 128) - 4x wider, no compression
    2. Hidden dim: 512 (vs 256) - wider hidden layers
    3. Mirror decoder: 512 → 512 → 253 (vs single layer 128 → 253)
    4. Dropout: 0.3 (vs 0.1) - more regularization

Hypothesis: If the 128-dim bottleneck was the limiting factor,
VIME-Wide should approach Route B's 0.7391 AUC.

Created: 2026-02-17
Phase: Route G-Wide implementation
"""

import torch
import torch.nn as nn


class VIMEWide(nn.Module):
    """
    VIME-Wide: No-bottleneck VIME architecture.

    Architecture:
        Encoder:  input_dim → 512 → 512 (latent)
        Mask head:    512 → 512 → input_dim (mirror)
        Recon head:   512 → 512 → input_dim (mirror)

    Args:
        input_dim: Number of input features (253 for fixed preprocessing)
        embedding_dim: Latent dimension (default: 512, no bottleneck)
        hidden_dim: Hidden layer dimension (default: 512)
        dropout: Dropout rate (default: 0.3)
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
        # Layer 1: input_dim → hidden_dim (253 → 512)
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_bn1 = nn.BatchNorm1d(hidden_dim)
        self.enc_relu1 = nn.ReLU()
        self.enc_drop1 = nn.Dropout(dropout)

        # Layer 2: hidden_dim → embedding_dim (512 → 512)
        self.enc_fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.enc_bn2 = nn.BatchNorm1d(embedding_dim)
        self.enc_relu2 = nn.ReLU()
        self.enc_drop2 = nn.Dropout(dropout)

        # === Mask Estimator (mirror architecture) ===
        # Predicts which features were corrupted
        # Layer 1: embedding_dim → hidden_dim (512 → 512)
        self.mask_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.mask_bn1 = nn.BatchNorm1d(hidden_dim)
        self.mask_relu1 = nn.ReLU()
        self.mask_drop1 = nn.Dropout(dropout)

        # Layer 2: hidden_dim → input_dim (512 → 253)
        self.mask_fc2 = nn.Linear(hidden_dim, input_dim)
        # Sigmoid applied in forward()

        # === Feature Reconstructor (mirror architecture) ===
        # Reconstructs original feature values
        # Layer 1: embedding_dim → hidden_dim (512 → 512)
        self.recon_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.recon_bn1 = nn.BatchNorm1d(hidden_dim)
        self.recon_relu1 = nn.ReLU()
        self.recon_drop1 = nn.Dropout(dropout)

        # Layer 2: hidden_dim → input_dim (512 → 253)
        self.recon_fc2 = nn.Linear(hidden_dim, input_dim)
        # No activation (regression output)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features to latent representation.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Latent tensor of shape (batch, embedding_dim)
        """
        # Layer 1
        h = self.enc_fc1(x)
        h = self.enc_bn1(h)
        h = self.enc_relu1(h)
        h = self.enc_drop1(h)

        # Layer 2 (latent)
        z = self.enc_fc2(h)
        z = self.enc_bn2(z)
        z = self.enc_relu2(z)
        z = self.enc_drop2(z)

        return z

    def decode_mask(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict corruption mask from latent representation.

        Args:
            z: Latent tensor of shape (batch, embedding_dim)

        Returns:
            Mask probabilities of shape (batch, input_dim)
        """
        h = self.mask_fc1(z)
        h = self.mask_bn1(h)
        h = self.mask_relu1(h)
        h = self.mask_drop1(h)

        m_hat = torch.sigmoid(self.mask_fc2(h))
        return m_hat

    def decode_features(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct original features from latent representation.

        Args:
            z: Latent tensor of shape (batch, embedding_dim)

        Returns:
            Reconstructed features of shape (batch, input_dim)
        """
        h = self.recon_fc1(z)
        h = self.recon_bn1(h)
        h = self.recon_relu1(h)
        h = self.recon_drop1(h)

        x_hat = self.recon_fc2(h)
        return x_hat

    def forward(self, x: torch.Tensor):
        """
        Full forward pass: encode → decode mask + features.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Tuple of (mask_prediction, feature_reconstruction)
            - mask_prediction: shape (batch, input_dim), values in [0, 1]
            - feature_reconstruction: shape (batch, input_dim)
        """
        # Encode
        z = self.encode(x)

        # Decode both heads
        m_hat = self.decode_mask(z)
        x_hat = self.decode_features(z)

        return m_hat, x_hat


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = VIMEWide(input_dim=253, embedding_dim=512, hidden_dim=512, dropout=0.3)
    print(f"VIMEWide architecture:")
    print(f"  Input dim: 253")
    print(f"  Hidden dim: 512")
    print(f"  Embedding dim: 512")
    print(f"  Total parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(32, 253)
    m_hat, x_hat = model(x)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Mask output shape: {m_hat.shape}")
    print(f"  Recon output shape: {x_hat.shape}")

    # Test encode
    z = model.encode(x)
    print(f"  Latent shape: {z.shape}")
