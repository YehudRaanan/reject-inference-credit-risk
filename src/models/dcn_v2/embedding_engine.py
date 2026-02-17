"""
Embedding Engine for DCN-v2.

Converts raw features (numerical + categorical) into a dense embedding vector x₀.
This is the "source of truth" that feeds into both Cross and Deep networks.

Key components:
- Numerical: SoftBinning → weighted bin embeddings
- Categorical: Standard embedding lookup tables
- LayerNorm: Ensures balanced feature contributions

# Purpose: Create unified dense representation from heterogeneous features
# Why: Cross Network needs dense vectors; raw features are mixed types
# Approved: 2026-02-15
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional

from .soft_binning import SoftBinning, compute_quantiles_for_binning


class EmbeddingEngine(nn.Module):
    """
    Transforms raw tabular features into dense embeddings for DCN-v2.

    For each numerical feature:
        value → SoftBinning → membership weights → weighted sum of bin embeddings

    For each categorical feature:
        category_id → Embedding lookup → embedding vector

    All embeddings are concatenated and normalized via LayerNorm.

    Args:
        numerical_features: List of numerical feature names
        categorical_cardinalities: Dict mapping categorical feature names to their cardinality
        num_bins: Number of bins for numerical features (default: 10)
        numerical_embed_dim: Embedding dimension for numerical bins (default: 16)
        temperature: Temperature for soft binning (default: 5.0)

    # Purpose: Unified embedding layer for mixed feature types
    # Why: DCN-v2 Cross Network operates on dense vectors, not raw mixed-type features
    # Approved: 2026-02-15
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_cardinalities: Dict[str, int],
        num_bins: int = 10,
        numerical_embed_dim: int = 16,
        temperature: float = 5.0,
    ):
        super().__init__()

        self.numerical_features = numerical_features
        self.categorical_features = list(categorical_cardinalities.keys())
        self.num_bins = num_bins
        self.numerical_embed_dim = numerical_embed_dim

        # --- Numerical Feature Processing ---
        # Each numerical feature gets:
        # 1. A SoftBinning layer (converts value → bin memberships)
        # 2. An embedding table for bins (num_bins+1 embeddings, including missing bin)

        self.numerical_binning = nn.ModuleDict()
        self.numerical_bin_embeddings = nn.ModuleDict()

        for feat in numerical_features:
            # SoftBinning: outputs (batch, num_bins+1) membership weights
            self.numerical_binning[feat] = SoftBinning(
                num_bins=num_bins,
                temperature=temperature
            )
            # Bin embeddings: (num_bins+1) vectors of size numerical_embed_dim
            # +1 for the missing bin at index 0
            self.numerical_bin_embeddings[feat] = nn.Embedding(
                num_embeddings=num_bins + 1,
                embedding_dim=numerical_embed_dim
            )

        # --- Categorical Feature Processing ---
        # Each categorical feature gets an embedding table
        # Dimension = min(50, ceil(sqrt(cardinality))) - balances expressiveness vs parameters

        self.categorical_embeddings = nn.ModuleDict()
        self.categorical_embed_dims = {}

        for feat, cardinality in categorical_cardinalities.items():
            embed_dim = min(50, int(cardinality ** 0.5) + 1)
            self.categorical_embed_dims[feat] = embed_dim
            # +1 for unknown/missing category (index 0 reserved)
            self.categorical_embeddings[feat] = nn.Embedding(
                num_embeddings=cardinality + 1,
                embedding_dim=embed_dim,
                padding_idx=0  # Index 0 = unknown/missing, gets zero vector
            )

        # --- Compute total embedding dimension ---
        numerical_total_dim = len(numerical_features) * numerical_embed_dim
        categorical_total_dim = sum(self.categorical_embed_dims.values())
        self.output_dim = numerical_total_dim + categorical_total_dim

        # --- LayerNorm on concatenated x₀ ---
        # Ensures no single feature dominates the representation
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        numerical_values: Dict[str, torch.Tensor],
        categorical_values: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Transform raw features into dense embedding vector x₀.

        Args:
            numerical_values: Dict mapping feature name to tensor of shape (batch_size,)
            categorical_values: Dict mapping feature name to tensor of shape (batch_size,)
                               with integer category indices (0 = unknown/missing)

        Returns:
            x₀: Dense embedding tensor of shape (batch_size, output_dim)

        # Purpose: Create unified dense representation from heterogeneous features
        # Why: Cross Network needs dense vectors; raw features are mixed types
        # Approved: 2026-02-15
        """
        embeddings = []

        # --- Process Numerical Features ---
        # For each feature: value → SoftBinning → weighted sum of bin embeddings
        for feat in self.numerical_features:
            # Get raw values: (batch_size,)
            values = numerical_values[feat]

            # SoftBinning: (batch_size,) → (batch_size, num_bins+1) membership weights
            memberships = self.numerical_binning[feat](values)

            # Get all bin embeddings: (num_bins+1, embed_dim)
            bin_embeds = self.numerical_bin_embeddings[feat].weight

            # Weighted sum: memberships @ bin_embeds → (batch_size, embed_dim)
            # Each sample gets a smooth combination of bin embeddings
            feat_embed = torch.matmul(memberships, bin_embeds)
            embeddings.append(feat_embed)

        # --- Process Categorical Features ---
        # For each feature: category_id → embedding lookup
        for feat in self.categorical_features:
            # Get category indices: (batch_size,) with integer values
            indices = categorical_values[feat]

            # Embedding lookup: (batch_size,) → (batch_size, embed_dim)
            # Index 0 (padding_idx) returns zeros for unknown/missing
            feat_embed = self.categorical_embeddings[feat](indices)
            embeddings.append(feat_embed)

        # --- Concatenate all embeddings ---
        # Shape: (batch_size, output_dim)
        x0 = torch.cat(embeddings, dim=1)

        # --- Apply LayerNorm ---
        # Normalizes across feature dimension, preventing dominance
        x0 = self.layer_norm(x0)

        return x0
