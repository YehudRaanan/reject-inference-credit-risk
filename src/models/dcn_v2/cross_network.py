"""
Cross Network for DCN-v2.

Implements explicit polynomial feature interactions through stacked cross layers.
Key innovation: Each layer increases interaction order while preserving all lower orders.

Formula per layer:
    x_{l+1} = x₀ ⊙ (W @ x_l + b) + x_l

Where ⊙ is element-wise product and W uses low-rank factorization for efficiency.

Reference: Wang et al., "DCN V2: Improved Deep & Cross Network" (2020)

# Purpose: Learn explicit feature interactions of arbitrary order
# Why: MLPs learn implicit interactions poorly; Cross Network makes them explicit
# Approved: 2026-02-15
"""

import torch
import torch.nn as nn


class CrossLayer(nn.Module):
    """
    Single cross layer with low-rank weight factorization.

    Computes: x_{l+1} = x₀ ⊙ (V(U(x_l)) + b) + x_l

    Where:
    - x₀: Original input embedding (constant reference)
    - x_l: Current layer input
    - U: (d, r) projects down to rank r
    - V: (r, d) projects back up
    - ⊙: Element-wise multiplication
    - +x_l: Residual connection

    Args:
        input_dim: Dimension of input (d)
        rank: Low-rank dimension (r). Default: 64
              Reduces params from d² to 2dr

    # Purpose: Single cross interaction step with efficient parameterization
    # Why: Low-rank avoids O(d²) parameters while capturing key interactions
    # Approved: 2026-02-15
    """

    def __init__(self, input_dim: int, rank: int = 64):
        super().__init__()

        # Low-rank factorization: W = V @ U instead of full (d, d) matrix
        # U: (d, r) - compress to low-rank space
        # V: (r, d) - expand back to original dimension
        self.U = nn.Linear(input_dim, rank, bias=False)
        self.V = nn.Linear(rank, input_dim, bias=False)

        # Bias term for the cross interaction
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        """
        Compute one cross layer step.

        Args:
            x0: Original input embedding, shape (batch, d)
            xl: Current layer input, shape (batch, d)

        Returns:
            x_{l+1}: Next layer output, shape (batch, d)

        # Purpose: Compute x₀ ⊙ (V(U(x_l)) + b) + x_l
        # Why: Creates feature interactions between x₀ and transformed x_l
        # Approved: 2026-02-15
        """
        # Low-rank transformation: U compresses, V expands
        # (batch, d) → (batch, r) → (batch, d)
        transformed = self.V(self.U(xl))

        # Cross interaction: x₀ ⊙ (transformed + bias)
        # This creates interactions between original features and transformed current state
        cross = x0 * (transformed + self.bias)

        # Residual connection: preserve all previous information
        return cross + xl


class CrossNetwork(nn.Module):
    """
    Stack of CrossLayers for learning high-order feature interactions.

    Each layer increases the interaction order:
    - Layer 1: 2nd-order interactions (x₀ ⊙ x₀)
    - Layer 2: 3rd-order interactions
    - Layer L: (L+1)-order interactions

    All lower-order interactions are preserved via residual connections.

    Args:
        input_dim: Dimension of input embedding (d)
        num_layers: Number of cross layers to stack. Default: 3
        rank: Low-rank dimension for each layer. Default: 64

    # Purpose: Learn explicit polynomial feature interactions up to order (num_layers+1)
    # Why: MLPs struggle with explicit interactions; Cross Network captures them directly
    # Approved: 2026-02-15
    """

    def __init__(self, input_dim: int, num_layers: int = 3, rank: int = 64):
        super().__init__()

        # Stack of CrossLayers - each adds one order of interaction
        self.layers = nn.ModuleList([
            CrossLayer(input_dim, rank) for _ in range(num_layers)
        ])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Apply stacked cross layers.

        Args:
            x0: Input embedding, shape (batch, d)
                This is both the initial state AND the constant reference

        Returns:
            Output with accumulated interactions, shape (batch, d)

        # Purpose: Pass x₀ through all cross layers, accumulating interactions
        # Why: x₀ stays constant as reference; xl evolves with each layer
        # Approved: 2026-02-15
        """
        # x0 is used as both the constant reference and initial state
        xl = x0

        # Each layer: xl = x0 ⊙ (V(U(xl)) + b) + xl
        for layer in self.layers:
            xl = layer(x0, xl)

        return xl
