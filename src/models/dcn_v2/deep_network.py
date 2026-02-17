"""
Deep Network for DCN-v2.

Standard MLP that learns implicit non-linear feature interactions.
Complements the Cross Network which learns explicit polynomial interactions.

# Purpose: Learn implicit non-linear transformations
# Why: Cross Network handles explicit interactions; Deep Network adds non-linearity
# Approved: 2026-02-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DeepNetwork(nn.Module):
    """
    Simple MLP with raw layer definitions.

    Architecture: Linear → ReLU → Dropout (repeated)

    Args:
        input_dim: Input dimension (embedding size)
        hidden_dims: List of hidden layer dimensions. Default: [512, 256, 128]
        dropout: Dropout probability. Default: 0.1

    # Purpose: Learn implicit non-linear feature transformations
    # Why: Complements Cross Network's explicit interactions with non-linear capacity
    # Approved: 2026-02-15
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.1
    ):
        super().__init__()

        # Raw layer definitions - no Sequential, no loops
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.dropout = nn.Dropout(dropout)

        # Store output dimension for downstream concatenation
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP layers.

        Args:
            x: Input tensor, shape (batch, input_dim)

        Returns:
            Output tensor, shape (batch, output_dim)

        # Purpose: Transform input through 3 hidden layers with ReLU and dropout
        # Why: Simple, explicit forward pass for clarity
        # Approved: 2026-02-15
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return x
