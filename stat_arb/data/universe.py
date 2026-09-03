"""
Universe management for crypto stat-arb backtest.
Handles survivorship bias and dynamic universe membership.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


class UniverseManager:
    """
    Manage tradable universe with survivorship controls.

    Universe membership based on:
    - ETH-relative market cap band (e.g., bottom 30% relative to ETH)
    - Minimum trading volume
    - Minimum price history
    - Monthly reconstitution
    """

    def __init__(
        self,
        mcap_percentile_low: float = 0.0,    # Lower bound percentile
        mcap_percentile_high: float = 0.30,  # Upper bound percentile (bottom 30%)
        min_volume_usd: float = 100_000,     # Minimum daily volume
        min_history_days: int = 90,          # Minimum days of history required
        reconstitution_freq: str = 'M',      # Monthly reconstitution
        volume_in_usd: bool = False,         # See note below
    ):
        self.mcap_percentile_low = mcap_percentile_low
        self.mcap_percentile_high = mcap_percentile_high
        self.min_volume_usd = min_volume_usd
        self.min_history_days = min_history_days
        self.reconstitution_freq = reconstitution_freq
        # The volume column of data/all_tokens_24mo_daily.csv is ALREADY in USD
        # (SNX $29M/day, ETH $8.7B/day on 2024-06-01), but the liquidity filter
        # below has always computed `volumes * prices`, i.e. USD x price. Every
        # published tier floor in this repository is therefore a floor on that
        # product rather than on notional traded, and the labels "$50k/day" and
        # "$1M/day" do not mean what they say.
        #
        # The default is left at the legacy behaviour so that no previously
        # published number silently moves. Pass volume_in_usd=True to filter on
        # notional traded, which is what the labels claim. The point-in-time
        # robustness run reports both conventions side by side.
        self.volume_in_usd = volume_in_usd

    def compute_daily_mcap_rank(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        eth_prices: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute daily market cap proxy rank (relative to cross-section).
        Uses volume * price as proxy when market cap not available.
        """
        # Market cap proxy: volume * price
        mcap_proxy = volumes * prices

        # Rank within each day (0 = smallest, 1 = largest)
        mcap_rank = mcap_proxy.rank(axis=1, pct=True)

        return mcap_rank

    def compute_eth_relative_mcap(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        eth_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute market cap relative to ETH market cap."""
        # Token market cap proxy
        token_mcap = volumes * prices

        # ETH market cap proxy (use volume if available, else just price)
        if 'volume' in eth_data.columns:
            eth_mcap = eth_data['close'] * eth_data['volume']
        else:
            eth_mcap = eth_data['close'] * 1e9  # Assume large volume

        # Relative market cap
        relative_mcap = token_mcap.div(eth_mcap, axis=0)

        return relative_mcap

    def get_universe_membership(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        eth_data: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute universe membership mask (True = in universe).

        Returns DataFrame of same shape as prices with boolean values.
        Reconstituted monthly to avoid excessive turnover.
        """
        # Align indices
        common_idx = prices.index.intersection(volumes.index)
        if returns is not None:
            common_idx = common_idx.intersection(returns.index)

        prices = prices.loc[common_idx]
        volumes = volumes.loc[common_idx]

        # Compute market cap rank
        mcap_rank = self.compute_daily_mcap_rank(
            prices, volumes,
            eth_data.loc[common_idx, 'close'] if 'close' in eth_data.columns else eth_data.loc[common_idx].iloc[:, 0]
        )

        # Market cap band filter
        in_band = (mcap_rank >= self.mcap_percentile_low) & (mcap_rank <= self.mcap_percentile_high)

        # Volume filter (rolling average notional); see volume_in_usd in __init__
        notional = volumes if self.volume_in_usd else volumes * prices
        avg_volume = notional.rolling(window=20, min_periods=5).mean()
        has_volume = avg_volume >= self.min_volume_usd

        # History filter
        has_history = prices.notna().rolling(window=self.min_history_days, min_periods=self.min_history_days).sum() >= self.min_history_days

        # Combine filters
        in_universe = in_band & has_volume & has_history

        # Monthly reconstitution: only update membership at month-end
        if self.reconstitution_freq == 'M':
            # Get month-end dates
            month_end = in_universe.index.to_period('M').to_timestamp('M')
            is_month_end = in_universe.index == month_end

            # Forward fill membership from month-end decisions
            reconstituted = in_universe.copy()
            for i in range(len(reconstituted)):
                if i == 0 or is_month_end[i]:
                    continue
                # Check if we crossed a month boundary
                if month_end[i] != month_end[i-1]:
                    continue
                reconstituted.iloc[i] = reconstituted.iloc[i-1]

            in_universe = reconstituted

        return in_universe

    def get_universe_tokens(
        self,
        universe_membership: pd.DataFrame,
        date: pd.Timestamp,
    ) -> List[str]:
        """Get list of tokens in universe on a specific date."""
        if date not in universe_membership.index:
            # Find closest prior date
            prior_dates = universe_membership.index[universe_membership.index <= date]
            if len(prior_dates) == 0:
                return []
            date = prior_dates[-1]

        membership = universe_membership.loc[date]
        return membership[membership].index.tolist()

    def compute_universe_stats(
        self,
        universe_membership: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute universe statistics over time."""
        stats = pd.DataFrame({
            'n_tokens': universe_membership.sum(axis=1),
            'pct_tokens': universe_membership.mean(axis=1),
        })
        return stats

    def save_universe_membership(
        self,
        universe_membership: pd.DataFrame,
        filepath: str,
    ):
        """Save universe membership to CSV."""
        universe_membership.to_csv(filepath)

    def apply_universe_mask(
        self,
        data: pd.DataFrame,
        universe_membership: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply universe mask to data (set out-of-universe values to NaN)."""
        # Align columns
        common_cols = data.columns.intersection(universe_membership.columns)
        common_idx = data.index.intersection(universe_membership.index)

        result = data.loc[common_idx, common_cols].copy()
        mask = universe_membership.loc[common_idx, common_cols]

        result[~mask] = np.nan
        return result
