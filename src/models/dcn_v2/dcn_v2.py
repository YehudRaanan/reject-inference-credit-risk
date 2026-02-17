"""
DCN-v2 (Deep & Cross Network v2) for Credit Risk Assessment.

Combines explicit polynomial interactions (Cross Network) with implicit
non-linear transformations (Deep Network) for tabular data.

Architecture:
    Raw Features → EmbeddingEngine → x₀
    x₀ → CrossNetwork (explicit) + DeepNetwork (implicit) → concat → head → P(default)

Reference: Wang et al., "DCN V2: Improved Deep & Cross Network" (2020)

# Purpose: Complete DCN-v2 model for binary classification
# Why: Addresses VIME's 128-dim bottleneck by preserving all features
# Approved: 2026-02-15
"""

import torch
import torch.nn as nn
from typing import List, Dict

from .embedding_engine import EmbeddingEngine
from .cross_network import CrossNetwork
from .deep_network import DeepNetwork


class DCNv2(nn.Module):
    """
    Deep & Cross Network v2 for credit risk binary classification.

    Combines:
    - EmbeddingEngine: Converts raw features to dense x₀
    - CrossNetwork: Learns explicit polynomial interactions
    - DeepNetwork: Learns implicit non-linear transformations
    - Classification head: Combines both outputs for final prediction

    Args:
        numerical_features: List of numerical feature names
        categorical_cardinalities: Dict mapping categorical names to cardinality
        num_bins: Number of bins for numerical soft binning. Default: 10
        numerical_embed_dim: Embedding dim per numerical feature. Default: 16
        cross_layers: Number of cross layers. Default: 3
        cross_rank: Low-rank dimension for cross layers. Default: 64
        deep_hidden: Hidden dims for deep network. Default: [512, 256, 128]
        dropout: Dropout probability. Default: 0.1

    # Purpose: Complete DCN-v2 model combining all components
    # Why: Single unified module for training and inference
    # Approved: 2026-02-15
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_cardinalities: Dict[str, int],
        num_bins: int = 10,
        numerical_embed_dim: int = 16,
        cross_layers: int = 3,
        cross_rank: int = 64,
        deep_hidden: List[int] = [512, 256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()

        # --- Embedding Engine ---
        # Raw features → dense x₀ vector
        self.embedding = EmbeddingEngine(
            numerical_features=numerical_features,
            categorical_cardinalities=categorical_cardinalities,
            num_bins=num_bins,
            numerical_embed_dim=numerical_embed_dim,
        )
        embed_dim = self.embedding.output_dim

        # --- Cross Network ---
        # x₀ → explicit polynomial interactions (same dim out)
        self.cross_network = CrossNetwork(
            input_dim=embed_dim,
            num_layers=cross_layers,
            rank=cross_rank,
        )

        # --- Deep Network ---
        # x₀ → implicit non-linear transformations
        self.deep_network = DeepNetwork(
            input_dim=embed_dim,
            hidden_dims=deep_hidden,
            dropout=dropout,
        )

        # --- Classification Head ---
        # concat(cross_out, deep_out) → logit → sigmoid
        head_input_dim = embed_dim + self.deep_network.output_dim
        self.head = nn.Linear(head_input_dim, 1)

    def forward(
        self,
        numerical_values: Dict[str, torch.Tensor],
        categorical_values: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass: raw features → P(default).

        Args:
            numerical_values: Dict mapping feature name to (batch,) tensor
            categorical_values: Dict mapping feature name to (batch,) tensor

        Returns:
            Probability of default, shape (batch, 1)

        # Purpose: End-to-end forward pass through all components
        # Why: Single forward call for training and inference
        # Approved: 2026-02-15
        """
        # Step 1: Embed raw features → x₀
        x0 = self.embedding(numerical_values, categorical_values)

        # Step 2: Cross Network (explicit interactions)
        cross_out = self.cross_network(x0)

        # Step 3: Deep Network (implicit transformations)
        deep_out = self.deep_network(x0)

        # Step 4: Concatenate outputs
        combined = torch.cat([cross_out, deep_out], dim=1)

        # Step 5: Classification head → probability
        logits = self.head(combined)
        prob = torch.sigmoid(logits)

        return prob
