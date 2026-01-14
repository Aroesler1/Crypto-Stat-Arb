"""
Return calculation utilities.
"""
import pandas as pd
import numpy as np
from typing import Optional


class ReturnsCalculator:
    """Static methods for computing various return transformations."""

    @staticmethod
    def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns from prices."""
        return np.log(prices / prices.shift(1))

    @staticmethod
    def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Compute simple returns from prices."""
        return prices.pct_change()

    @staticmethod
    def excess_returns(
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> pd.DataFrame:
        """Compute excess returns over benchmark."""
        return returns.sub(benchmark_returns, axis=0)

    @staticmethod
    def eth_excess_returns(
        returns: pd.DataFrame,
        eth_returns: pd.Series,
    ) -> pd.DataFrame:
        """Compute ETH-excess returns."""
        return ReturnsCalculator.excess_returns(returns, eth_returns)

    @staticmethod
    def cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
        """Compute cumulative returns (assuming log returns)."""
        return returns.cumsum()

    @staticmethod
    def rolling_sum_returns(
        returns: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """Compute rolling sum of returns (momentum signal)."""
        return returns.rolling(window=window, min_periods=1).sum()

    @staticmethod
    def demeaned_returns(
        returns: pd.DataFrame,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """Demean returns (cross-sectional or rolling)."""
        if window is None:
            # Cross-sectional demean
            return returns.sub(returns.mean(axis=1), axis=0)
        else:
            # Rolling demean
            rolling_mean = returns.rolling(window=window, min_periods=1).mean()
            return returns - rolling_mean

    @staticmethod
    def standardized_returns(
        returns: pd.DataFrame,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """Standardize returns (z-score, cross-sectional or rolling)."""
        if window is None:
            # Cross-sectional standardization
            mean = returns.mean(axis=1)
            std = returns.std(axis=1)
            return returns.sub(mean, axis=0).div(std + 1e-8, axis=0)
        else:
            # Rolling standardization
            rolling_mean = returns.rolling(window=window, min_periods=1).mean()
            rolling_std = returns.rolling(window=window, min_periods=1).std()
            return (returns - rolling_mean) / (rolling_std + 1e-8)

    @staticmethod
    def volatility(
        returns: pd.DataFrame,
        window: int = 20,
        annualize: bool = True,
    ) -> pd.DataFrame:
        """Compute rolling volatility."""
        vol = returns.rolling(window=window, min_periods=1).std()
        if annualize:
            vol = vol * np.sqrt(365)  # Crypto trades 365 days
        return vol
