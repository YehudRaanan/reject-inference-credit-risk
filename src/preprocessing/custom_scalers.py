"""
Custom Scalers for Route A Preprocessing
========================================

Provides outlier-robust scaling that maintains [0, 1] range required by VIME.

Key Class:
    ClippedRobustScaler: RobustScaler followed by percentile-based clipping to [0, 1].

Why This Matters:
    MinMaxScaler with extreme outliers (AMT_INCOME_TOTAL max=117M vs median=135K)
    compresses 99% of values into a tiny range [0, 0.005].
    RobustScaler + percentile clipping spreads values across the full [0, 1] range.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler


class ClippedRobustScaler(BaseEstimator, TransformerMixin):
    """
    RobustScaler followed by percentile-based clipping and normalization to [0, 1].

    Process:
        1. Apply RobustScaler (centers on median, scales by IQR)
        2. Clip at specified percentiles (default: 1st and 99th)
        3. Normalize clipped values to [0, 1]

    Why This Approach:
        - RobustScaler handles outliers gracefully (uses median, not mean)
        - Percentile clipping removes extreme outliers before final scaling
        - Final normalization ensures VIME-compatible [0, 1] range
        - Values within the percentile range get spread across full [0, 1]

    Example:
        >>> scaler = ClippedRobustScaler(clip_percentiles=(1, 99))
        >>> X_train_scaled = scaler.fit_transform(X_train)
        >>> X_test_scaled = scaler.transform(X_test)  # May clip extremes
    """

    def __init__(self, clip_percentiles=(1.0, 99.0), quantile_range=(25.0, 75.0)):
        """
        Initialize the scaler.

        Args:
            clip_percentiles: Tuple (lower, upper) percentiles for clipping.
                              Default (1, 99) means clip bottom 1% and top 1%.
            quantile_range: Tuple for RobustScaler IQR computation. Default (25, 75).
        """
        self.clip_percentiles = clip_percentiles
        self.quantile_range = quantile_range
        self.robust_scaler = RobustScaler(quantile_range=quantile_range)
        self.clip_lower_ = None
        self.clip_upper_ = None

    def fit(self, X, y=None):
        """
        Fit the scaler on training data.

        Args:
            X: Training data array of shape (n_samples, n_features)
            y: Ignored (for sklearn compatibility)

        Returns:
            self
        """
        # Step 1: Fit and transform with RobustScaler
        X_robust = self.robust_scaler.fit_transform(X)

        # Step 2: Compute percentile thresholds for clipping
        self.clip_lower_ = np.percentile(X_robust, self.clip_percentiles[0], axis=0)
        self.clip_upper_ = np.percentile(X_robust, self.clip_percentiles[1], axis=0)

        # Handle edge case: if lower == upper, expand slightly
        mask = self.clip_lower_ == self.clip_upper_
        self.clip_upper_[mask] = self.clip_lower_[mask] + 1.0

        return self

    def transform(self, X):
        """
        Transform data using fitted scaler.

        Args:
            X: Data array of shape (n_samples, n_features)

        Returns:
            Scaled data in [0, 1] range
        """
        # Step 1: Apply robust scaling
        X_robust = self.robust_scaler.transform(X)

        # Step 2: Clip at percentile thresholds
        X_clipped = np.clip(X_robust, self.clip_lower_, self.clip_upper_)

        # Step 3: Normalize to [0, 1] using the percentile bounds
        # Now the clipped range [lower, upper] maps to [0, 1]
        range_ = self.clip_upper_ - self.clip_lower_
        X_normalized = (X_clipped - self.clip_lower_) / range_

        # Final clip to handle numerical precision issues
        X_normalized = np.clip(X_normalized, 0.0, 1.0)

        return X_normalized

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Return input feature names (passthrough)."""
        return input_features
