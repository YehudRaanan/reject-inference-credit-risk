"""
Classification head for end-to-end VIME fine-tuning.

Attaches to the pre-trained VIME encoder's 128-dim latent
vector and outputs P(default). Used in Phase 5.7.

Architecture:
    [z] (128) → Linear(128→64) → ReLU → Dropout(0.3)
              → Linear(64→32)  → ReLU → Dropout(0.3)
              → Linear(32→1)   → Sigmoid → P(default)

Total head parameters: 10,369
    - fc1: 128×64 + 64 = 8,256
    - fc2: 64×32 + 32  = 2,080
    - out: 32×1 + 1     = 33

Design decisions (approved by owner 2026-02-15):
    - Two hidden layers for sufficient expressivity
    - ReLU activation (same as encoder, simple and proven)
    - Dropout=0.3 (moderate regularization)
    - Same LR for encoder and head as starting point
"""
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """MLP classification head that attaches to the VIME encoder output.

    Args:
        embedding_dim: Size of the VIME encoder's latent vector (default: 128)
        dropout: Dropout probability for regularization (default: 0.3)
    """

    def __init__(self, embedding_dim=128, dropout=0.3):
        super().__init__()

        # Hidden layer 1: Reduce 128 → 64 dimensions
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        # Hidden layer 2: Reduce 64 → 32 dimensions
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # Output layer: 32 → 1 (single default probability)
        self.output = nn.Linear(32, 1)

    def forward(self, z):
        """
        Args:
            z: VIME encoder output (batch_size, 128)
        Returns:
            P(default) as tensor (batch_size, 1)
        """
        z = self.drop1(self.relu1(self.fc1(z)))
        z = self.drop2(self.relu2(self.fc2(z)))
        return torch.sigmoid(self.output(z))
