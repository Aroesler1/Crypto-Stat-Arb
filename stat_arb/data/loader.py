"""
Data loading utilities for crypto stat-arb backtest.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

# Tokens to exclude (stablecoins and wrapped assets)
EXCLUDED_TOKENS = {
    # Stablecoins
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD', 'GUSD', 'FRAX',
    'LUSD', 'SUSD', 'CUSD', 'UST', 'MIM', 'FEI', 'TRIBE', 'RAI', 'DOLA',
    'OUSD', 'MUSD', 'HUSD', 'EURS', 'EURT', 'EUROC', 'XSGD', 'BIDR',
    'FDUSD', 'PYUSD', 'USDJ', 'RSV', 'VAI', 'CEUR', 'JPYC', 'XIDR',
    'IDRT', 'BKRW', 'BGBP', 'XAUT', 'PAXG', 'USTC',
    # Wrapped assets
    'WBTC', 'WETH', 'WBNB', 'WMATIC', 'WAVAX', 'WFTM', 'WONE', 'WCRO',
    'RENBTC', 'HBTC', 'BTCB', 'TBTC', 'SBTC', 'OBTC', 'BBTC',
    'STETH', 'RETH', 'CBETH', 'SFRXETH', 'WSTETH', 'BETH', 'ANKRETH',
}


class DataLoader:
    """Load and prepare data for backtesting."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._cache = {}

    def load_excess_returns(self, exclude_stablecoins: bool = True) -> pd.DataFrame:
        """Load ETH-excess log returns."""
        if 'excess_returns' not in self._cache:
            df = pd.read_csv(self.data_dir / 'excess_log_returns.csv', index_col=0, parse_dates=True)
            self._cache['excess_returns'] = df

        df = self._cache['excess_returns'].copy()

        if exclude_stablecoins:
            # Extract token names from column names (remove '_returns' suffix)
            cols_to_keep = []
            for col in df.columns:
                token = col.replace('_returns', '').upper()
                if token not in EXCLUDED_TOKENS:
                    cols_to_keep.append(col)
            df = df[cols_to_keep]

        return df

    def load_raw_prices(self, exclude_stablecoins: bool = True) -> pd.DataFrame:
        """Load raw OHLCV data and extract close prices."""
        if 'raw_data' not in self._cache:
            df = pd.read_csv(self.data_dir / 'all_tokens_24mo_daily.csv', index_col=0, parse_dates=True)
            self._cache['raw_data'] = df

        df = self._cache['raw_data'].copy()

        # Extract close price columns
        close_cols = [c for c in df.columns if c.endswith('_close')]
        prices = df[close_cols].copy()
        prices.columns = [c.replace('_close', '') for c in prices.columns]

        if exclude_stablecoins:
            cols_to_keep = [c for c in prices.columns if c.upper() not in EXCLUDED_TOKENS]
            prices = prices[cols_to_keep]

        return prices

    def load_volumes(self, exclude_stablecoins: bool = True) -> pd.DataFrame:
        """Load trading volumes."""
        if 'raw_data' not in self._cache:
            df = pd.read_csv(self.data_dir / 'all_tokens_24mo_daily.csv', index_col=0, parse_dates=True)
            self._cache['raw_data'] = df

        df = self._cache['raw_data'].copy()

        # Extract volume columns
        vol_cols = [c for c in df.columns if c.endswith('_volume')]
        volumes = df[vol_cols].copy()
        volumes.columns = [c.replace('_volume', '') for c in volumes.columns]

        if exclude_stablecoins:
            cols_to_keep = [c for c in volumes.columns if c.upper() not in EXCLUDED_TOKENS]
            volumes = volumes[cols_to_keep]

        return volumes

    def load_market_caps(self, exclude_stablecoins: bool = True) -> pd.DataFrame:
        """Load market caps (computed from close * circulating supply proxy via volume/price)."""
        if 'raw_data' not in self._cache:
            df = pd.read_csv(self.data_dir / 'all_tokens_24mo_daily.csv', index_col=0, parse_dates=True)
            self._cache['raw_data'] = df

        df = self._cache['raw_data'].copy()

        # Extract market cap columns if available, else compute from close * (volume/close) as proxy
        mcap_cols = [c for c in df.columns if c.endswith('_market_cap')]

        if mcap_cols:
            mcaps = df[mcap_cols].copy()
            mcaps.columns = [c.replace('_market_cap', '') for c in mcaps.columns]
        else:
            # Use volume as market cap proxy (volume ~ price * quantity traded)
            prices = self.load_raw_prices(exclude_stablecoins=False)
            volumes = self.load_volumes(exclude_stablecoins=False)
            # Rough proxy: higher volume tokens tend to have higher market cap
            mcaps = volumes.copy()

        if exclude_stablecoins:
            cols_to_keep = [c for c in mcaps.columns if c.upper() not in EXCLUDED_TOKENS]
            mcaps = mcaps[cols_to_keep]

        return mcaps

    def load_eth_data(self) -> pd.DataFrame:
        """Load ETH OHLCV data."""
        if 'eth_data' not in self._cache:
            df = pd.read_csv(self.data_dir / 'eth_ohlcv.csv', index_col=0, parse_dates=True)
            self._cache['eth_data'] = df
        return self._cache['eth_data'].copy()

    def load_token_summary(self) -> pd.DataFrame:
        """Load token summary/metadata."""
        if 'token_summary' not in self._cache:
            df = pd.read_csv(self.data_dir / 'token_summary.csv', index_col=0)
            self._cache['token_summary'] = df
        return self._cache['token_summary'].copy()

    def load_correlation_matrix(self, cleaned: bool = True) -> pd.DataFrame:
        """Load precomputed correlation matrix."""
        filename = 'corr_full_cleaned.csv' if cleaned else 'corr_full.csv'
        return pd.read_csv(self.data_dir / filename, index_col=0)

    def get_aligned_data(self, exclude_stablecoins: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get aligned excess returns, prices, volumes, and ETH data."""
        excess_returns = self.load_excess_returns(exclude_stablecoins)
        prices = self.load_raw_prices(exclude_stablecoins)
        volumes = self.load_volumes(exclude_stablecoins)
        eth_data = self.load_eth_data()

        # Find common dates
        common_dates = excess_returns.index.intersection(prices.index).intersection(volumes.index).intersection(eth_data.index)

        # Find common tokens (normalize column names).
        # sorted() is load-bearing, not tidiness: iterating the raw set puts the
        # columns in Python's per-process string hash order, so the panel came
        # out in a different column order on every run. That reorders the k-NN
        # graph and the k-means embedding, which flips cluster labels and the
        # noisy-cluster pick, and moved baseline gross Sharpe over roughly
        # 2.96-3.23 across identical runs. The checked-in results could not be
        # reproduced from a fresh process until this was pinned.
        excess_tokens = set(c.replace('_returns', '') for c in excess_returns.columns)
        price_tokens = set(prices.columns)
        vol_tokens = set(volumes.columns)
        common_tokens = sorted(excess_tokens & price_tokens & vol_tokens)

        # Align everything
        excess_returns = excess_returns.loc[common_dates, [f"{t}_returns" for t in common_tokens if f"{t}_returns" in excess_returns.columns]]
        prices = prices.loc[common_dates, [t for t in common_tokens if t in prices.columns]]
        volumes = volumes.loc[common_dates, [t for t in common_tokens if t in volumes.columns]]
        eth_data = eth_data.loc[common_dates]

        return excess_returns, prices, volumes, eth_data
