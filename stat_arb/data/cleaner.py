"""
Data cleaning utilities for crypto stat-arb backtest.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataCleaner:
    """Clean and validate data for backtesting."""

    def __init__(
        self,
        max_daily_return: float = 2.0,  # 200% max daily return
        max_missing_pct: float = 0.20,  # Max 20% missing data
        outlier_zscore: float = 5.0,    # Z-score threshold for outliers
    ):
        self.max_daily_return = max_daily_return
        self.max_missing_pct = max_missing_pct
        self.outlier_zscore = outlier_zscore

    def clean_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Clean return series by winsorizing extremes and handling missing data."""
        df = returns.copy()

        # Winsorize extreme returns
        df = df.clip(lower=-self.max_daily_return, upper=self.max_daily_return)

        # Remove tokens with too much missing data
        missing_pct = df.isna().mean()
        valid_cols = missing_pct[missing_pct <= self.max_missing_pct].index
        df = df[valid_cols]

        return df

    def clean_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Clean price series."""
        df = prices.copy()

        # Remove tokens with too much missing data
        missing_pct = df.isna().mean()
        valid_cols = missing_pct[missing_pct <= self.max_missing_pct].index
        df = df[valid_cols]

        # Forward fill small gaps (up to 3 days)
        df = df.ffill(limit=3)

        return df

    def detect_outliers(self, returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """Detect outliers using rolling z-scores."""
        rolling_mean = returns.rolling(window=window, min_periods=20).mean()
        rolling_std = returns.rolling(window=window, min_periods=20).std()

        zscore = (returns - rolling_mean) / (rolling_std + 1e-8)
        outliers = zscore.abs() > self.outlier_zscore

        return outliers

    def winsorize_outliers(self, returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """Winsorize outliers to rolling mean +/- threshold*std."""
        df = returns.copy()
        rolling_mean = df.rolling(window=window, min_periods=20).mean()
        rolling_std = df.rolling(window=window, min_periods=20).std()

        upper = rolling_mean + self.outlier_zscore * rolling_std
        lower = rolling_mean - self.outlier_zscore * rolling_std

        df = df.clip(lower=lower, upper=upper, axis=1)
        return df

    def compute_data_quality_report(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Generate data quality report for each token."""
        report = pd.DataFrame({
            'n_obs': returns.count(),
            'missing_pct': returns.isna().mean(),
            'mean': returns.mean(),
            'std': returns.std(),
            'min': returns.min(),
            'max': returns.max(),
            'skew': returns.skew(),
            'kurtosis': returns.kurtosis(),
        })
        return report.sort_values('missing_pct')

    def align_and_clean_all(
        self,
        excess_returns: pd.DataFrame,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Align and clean all data sources."""
        # Clean each
        excess_returns = self.clean_returns(excess_returns)
        prices = self.clean_prices(prices)

        # Get common tokens
        ret_tokens = set(c.replace('_returns', '') for c in excess_returns.columns)
        price_tokens = set(prices.columns)
        vol_tokens = set(volumes.columns)
        common_tokens = ret_tokens.intersection(price_tokens).intersection(vol_tokens)

        # Get common dates
        common_dates = excess_returns.index.intersection(prices.index).intersection(volumes.index)

        # Filter to common
        excess_returns = excess_returns.loc[common_dates, [f"{t}_returns" for t in common_tokens if f"{t}_returns" in excess_returns.columns]]
        prices = prices.loc[common_dates, [t for t in common_tokens if t in prices.columns]]
        volumes = volumes.loc[common_dates, [t for t in common_tokens if t in volumes.columns]]

        return excess_returns, prices, volumes
