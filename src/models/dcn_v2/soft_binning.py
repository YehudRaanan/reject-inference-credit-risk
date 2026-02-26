"""
Soft Binning Layer for DCN-v2.

Converts numerical features into soft bin memberships with learnable cutpoints.
Key features:
- Differentiable: gradients flow through to learn optimal bin boundaries
- Temperature-controlled: soft (blurry) → hard (sharp) during training
- Missing-aware: Bin 0 is reserved for NaN values

Reference: Inspired by "Deep Embedding Learning for Tabular Data" concepts.

# Purpose: Transform numerical features into learnable bin memberships
# Why: Standard NNs struggle with non-monotonic relationships and thresholds in tabular data
# Approved: 2026-02-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SoftBinning(nn.Module):
    """
    Differentiable soft binning for a single numerical feature.

    Converts a scalar value into soft membership weights across bins.
    Key features:
    - Learnable cutpoints (bin boundaries) optimized during training
    - Fixed temperature controls sharpness (higher = sharper boundaries)
    - Dedicated missing bin (index 0) for NaN values

    Args:
        num_bins: Number of bins (excluding missing bin). Default: 10
        temperature: Softmax temperature (fixed). Higher = sharper bins. Default: 5.0
        quantiles: Optional tensor of shape (num_bins-1,) with data quantiles
                   for initialization. If None, uses uniform spacing.

    Output shape: (batch_size, num_bins + 1)
        - Index 0: Missing bin (1.0 if input is NaN, else 0.0)
        - Index 1 to num_bins: Regular bin memberships (sum to 1.0)

    Note:
        Temperature is fixed (not annealed) because soft bins work well for
        embedding-based architectures. The downstream Cross/Deep networks
        learn sharp boundaries if needed.

    # Purpose: Transform numerical features into learnable bin memberships
    # Why: Captures non-linear thresholds that matter for credit risk (e.g., score cutoffs)
    # Approved: 2026-02-15
    """

    def __init__(
        self,
        num_bins: int = 10,
        temperature: float = 5.0,
        quantiles: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature

        # Learnable bin boundaries (cutpoints)
        # We need (num_bins - 1) boundaries to create num_bins bins
        # Example: 3 boundaries [0.25, 0.5, 0.75] create 4 bins: [0-0.25], [0.25-0.5], [0.5-0.75], [0.75-1]
        if quantiles is not None:
            # Initialize from data quantiles for balanced bins
            assert len(quantiles) == num_bins - 1, \
                f"Expected {num_bins - 1} quantiles, got {len(quantiles)}"
            self.cutpoints = nn.Parameter(quantiles.clone().float())
        else:
            # Fallback: uniform spacing in [0.1, 0.9] to avoid edge bins
            self.cutpoints = nn.Parameter(torch.linspace(0.1, 0.9, num_bins - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute soft bin memberships for input values.

        Args:
            x: Input tensor of shape (batch_size,) or (batch_size, 1)
               Values should be scaled to [0, 1] range for best results.

        Returns:
            Tensor of shape (batch_size, num_bins + 1) with membership weights.
            Row sums to 1.0. Index 0 is the missing bin.

        # Purpose: Convert raw numerical values to soft bin membership vectors
        # Why: Enables learnable, differentiable discretization with missing value handling
        # Approved: 2026-02-15
        """
        # Flatten to 1D: (batch_size,)
        x = x.view(-1)
        batch_size = x.shape[0]

        # Step 1: Identify missing values
        is_missing = torch.isnan(x)

        # Step 2: Replace NaN with sentinel value for safe computation
        # Using -999 (outside [0,1] range) to clearly indicate "not a real value"
        # The bin_probs computed for these will be masked out in Step 6
        MISSING_SENTINEL = -999.0
        x_safe = torch.where(is_missing, torch.full_like(x, MISSING_SENTINEL), x)

        # Step 3: Compute distance from each cutpoint
        # Shape: (batch_size, num_bins - 1)
        distances = x_safe.unsqueeze(1) - self.cutpoints.unsqueeze(0)

        # Step 4: Sigmoid gives P(x > cutpoint) - smooth, differentiable step
        # Higher temperature = sharper transition (more like hard binning)
        cumulative_probs = torch.sigmoid(self.temperature * distances)

        # Step 5: Convert cumulative probabilities to per-bin probabilities
        # P(bin_k) = P(x > cutpoint_{k-1}) - P(x > cutpoint_k)
        ones = torch.ones(batch_size, 1, device=x.device)
        zeros = torch.zeros(batch_size, 1, device=x.device)
        cumulative_extended = torch.cat([ones, cumulative_probs, zeros], dim=1)
        bin_probs = cumulative_extended[:, :-1] - cumulative_extended[:, 1:]

        # Step 6: Build output with missing bin at index 0
        output = torch.zeros(batch_size, self.num_bins + 1, device=x.device)

        # Missing bin: 1.0 if NaN, else 0.0
        output[:, 0] = is_missing.float()

        # Regular bins: only for non-missing values (masked to 0 for NaN rows)
        output[:, 1:] = bin_probs * (~is_missing).float().unsqueeze(1)

        return output


def compute_quantiles_for_binning(
    data: torch.Tensor,
    num_bins: int = 10
) -> torch.Tensor:
    """
    Compute quantile cutpoints from data for balanced bin initialization.

    Args:
        data: 1D tensor of feature values (NaN values are ignored)
        num_bins: Number of bins to create

    Returns:
        Tensor of shape (num_bins - 1,) with quantile values

    Example:
        >>> data = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        >>> compute_quantiles_for_binning(data, num_bins=3)
        tensor([0.25, 0.60])  # 33rd and 67th percentiles

    # Purpose: Initialize bin cutpoints so each bin has ~equal samples
    # Why: Prevents dead bins and ensures balanced learning signal
    # Approved: 2026-02-15
    """
    # Remove NaN values - can't compute quantiles on missing data
    valid_data = data[~torch.isnan(data)]

    if len(valid_data) == 0:
        # Fallback if all NaN: use uniform spacing
        return torch.linspace(0.1, 0.9, num_bins - 1)

    # Compute interior percentiles: for 10 bins we need 9 cutpoints at 10%, 20%, ..., 90%
    # linspace(0, 100, 11) = [0, 10, 20, ..., 100], then take [1:-1] = [10, 20, ..., 90]
    percentiles = torch.linspace(0, 100, num_bins + 1)[1:-1]
    quantiles = torch.quantile(valid_data.float(), percentiles / 100.0)

    return quantiles
