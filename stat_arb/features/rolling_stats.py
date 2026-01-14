"""
Rolling statistics calculator for feature generation.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class RollingStatsCalculator:
    """Comprehensive rolling statistics for signal generation."""

    @staticmethod
    def rolling_mean(
        data: pd.DataFrame,
        window: int,
        min_periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute rolling mean."""
        if min_periods is None:
            min_periods = max(1, window // 2)
        return data.rolling(window=window, min_periods=min_periods).mean()

    @staticmethod
    def rolling_std(
        data: pd.DataFrame,
        window: int,
        min_periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute rolling standard deviation."""
        if min_periods is None:
            min_periods = max(1, window // 2)
        return data.rolling(window=window, min_periods=min_periods).std()

    @staticmethod
    def rolling_zscore(
        data: pd.DataFrame,
        window: int,
        min_periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute rolling z-score."""
        if min_periods is None:
            min_periods = max(1, window // 2)
        rolling_mean = data.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = data.rolling(window=window, min_periods=min_periods).std()
        return (data - rolling_mean) / (rolling_std + 1e-8)

    @staticmethod
    def rolling_vol(
        returns: pd.DataFrame,
        window: int = 20,
        annualize: bool = True,
    ) -> pd.DataFrame:
        """Compute rolling volatility."""
        vol = returns.rolling(window=window, min_periods=max(1, window // 2)).std()
        if annualize:
            vol = vol * np.sqrt(365)
        return vol

    @staticmethod
    def rolling_sharpe(
        returns: pd.DataFrame,
        window: int = 60,
        annualize: bool = True,
    ) -> pd.DataFrame:
        """Compute rolling Sharpe ratio (assuming zero risk-free rate)."""
        rolling_mean = returns.rolling(window=window, min_periods=20).mean()
        rolling_std = returns.rolling(window=window, min_periods=20).std()
        sharpe = rolling_mean / (rolling_std + 1e-8)
        if annualize:
            sharpe = sharpe * np.sqrt(365)
        return sharpe

    @staticmethod
    def rolling_beta(
        returns: pd.DataFrame,
        benchmark: pd.Series,
        window: int = 60,
    ) -> pd.DataFrame:
        """Compute rolling beta to benchmark."""
        betas = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

        for col in returns.columns:
            # Rolling covariance and variance
            cov = returns[col].rolling(window=window, min_periods=20).cov(benchmark)
            var = benchmark.rolling(window=window, min_periods=20).var()
            betas[col] = cov / (var + 1e-8)

        return betas

    @staticmethod
    def rolling_correlation(
        returns: pd.DataFrame,
        benchmark: pd.Series,
        window: int = 60,
    ) -> pd.DataFrame:
        """Compute rolling correlation to benchmark."""
        corrs = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

        for col in returns.columns:
            corrs[col] = returns[col].rolling(window=window, min_periods=20).corr(benchmark)

        return corrs

    @staticmethod
    def compute_signal_zscore(
        returns: pd.DataFrame,
        H: int = 5,      # Momentum lookback (rolling sum window)
        L: int = 60,     # Normalization window
        lag: int = 1,    # Signal lag to avoid lookahead
    ) -> pd.DataFrame:
        """
        Compute the z-score signal for mean reversion strategy.

        Signal construction (NO LOOKAHEAD):
        1. s_i,t = sum_{j=0..H-1} r_i,t-j  (rolling sum of returns)
        2. z_i,t = (s_i,t - rolling_mean_L(s)) / (rolling_std_L(s) + eps)
        3. Return z_i,t-lag (lagged to avoid lookahead)

        Parameters
        ----------
        returns : pd.DataFrame
            Return series (typically ETH-excess or residualized returns)
        H : int
            Rolling sum window (momentum lookback)
        L : int
            Rolling normalization window for z-score
        lag : int
            Number of periods to lag signal (default 1 for t-1 signal)

        Returns
        -------
        pd.DataFrame
            Z-score signals, lagged by `lag` periods
        """
        # Step 1: Rolling sum of returns
        rolling_sum = returns.rolling(window=H, min_periods=1).sum()

        # Step 2: Z-score normalization
        rolling_mean = rolling_sum.rolling(window=L, min_periods=20).mean()
        rolling_std = rolling_sum.rolling(window=L, min_periods=20).std()
        zscore = (rolling_sum - rolling_mean) / (rolling_std + 1e-8)

        # Step 3: Lag to avoid lookahead
        zscore_lagged = zscore.shift(lag)

        return zscore_lagged

    @staticmethod
    def cross_sectional_rank(
        data: pd.DataFrame,
        pct: bool = True,
    ) -> pd.DataFrame:
        """Compute cross-sectional rank (within each day)."""
        return data.rank(axis=1, pct=pct)

    @staticmethod
    def rolling_quantile(
        data: pd.DataFrame,
        window: int,
        quantile: float,
    ) -> pd.DataFrame:
        """Compute rolling quantile."""
        return data.rolling(window=window, min_periods=max(1, window // 2)).quantile(quantile)
