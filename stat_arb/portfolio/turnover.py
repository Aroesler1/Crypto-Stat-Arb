"""
Turnover control and analysis.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class TurnoverController:
    """
    Control and analyze portfolio turnover.
    """

    def __init__(
        self,
        max_daily_turnover: float = 0.5,   # Max daily turnover (sum of |delta w|)
        turnover_penalty: float = 0.0,      # Penalty for turnover in objective
    ):
        self.max_daily_turnover = max_daily_turnover
        self.turnover_penalty = turnover_penalty

    def apply_turnover_limits(
        self,
        target_weights: pd.DataFrame,
        prev_weights: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Apply turnover limits to target weights.

        If change from previous weights exceeds limit, scale down.
        """
        if prev_weights is None:
            return target_weights

        result = target_weights.copy()

        for i in range(len(result)):
            date = result.index[i]

            if i == 0:
                if date in prev_weights.index:
                    prev = prev_weights.loc[date]
                else:
                    continue
            else:
                prev = result.iloc[i-1]

            target = result.loc[date]

            # Align
            common = prev.index.intersection(target.index)
            if len(common) == 0:
                continue

            delta = target.loc[common] - prev.loc[common].fillna(0)
            turnover = delta.abs().sum()

            if turnover > self.max_daily_turnover:
                scale = self.max_daily_turnover / turnover
                new_weights = prev.loc[common].fillna(0) + delta * scale
                result.loc[date, common] = new_weights

        return result

    def compute_turnover(
        self,
        weights: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute daily turnover.

        Turnover = sum of |w_t - w_{t-1}|
        """
        delta = weights.diff()
        turnover = delta.abs().sum(axis=1)
        return turnover

    def compute_one_way_turnover(
        self,
        weights: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute one-way turnover (buys only or sells only).

        One-way = 0.5 * two-way turnover
        """
        return self.compute_turnover(weights) / 2

    def estimate_transaction_costs(
        self,
        weights: pd.DataFrame,
        cost_bps: float = 50,
    ) -> pd.Series:
        """
        Estimate transaction costs.

        Cost = turnover * cost_bps / 10000
        """
        turnover = self.compute_turnover(weights)
        return turnover * cost_bps / 10000


class TurnoverAnalyzer:
    """
    Analyze turnover patterns and decomposition.
    """

    @staticmethod
    def decompose_turnover(
        weights: pd.DataFrame,
        cluster_labels: Optional[np.ndarray] = None,
        tokens: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Decompose turnover by source (rebalancing vs signal change).
        """
        delta = weights.diff()

        stats = pd.DataFrame(index=weights.index)
        stats['total_turnover'] = delta.abs().sum(axis=1)
        stats['n_trades'] = (delta.abs() > 1e-6).sum(axis=1)

        # Decompose by long/short
        stats['long_turnover'] = delta.clip(lower=0).sum(axis=1)
        stats['short_turnover'] = (-delta.clip(upper=0)).sum(axis=1)

        if cluster_labels is not None and tokens is not None:
            token_to_cluster = {token: cluster_labels[i] for i, token in enumerate(tokens)}

            for cluster_id in np.unique(cluster_labels):
                cluster_cols = [col for col in weights.columns
                                if col.replace('_returns', '') in token_to_cluster
                                and token_to_cluster[col.replace('_returns', '')] == cluster_id]

                if cluster_cols:
                    stats[f'cluster_{cluster_id}_turnover'] = delta[cluster_cols].abs().sum(axis=1)

        return stats

    @staticmethod
    def analyze_holding_periods(
        weights: pd.DataFrame,
        threshold: float = 0.01,
    ) -> pd.DataFrame:
        """
        Analyze average holding periods by position.
        """
        results = []

        for col in weights.columns:
            w = weights[col]

            # Find periods where position is active
            active = w.abs() > threshold
            changes = active.astype(int).diff()

            # Entry and exit points
            entries = changes[changes == 1].index
            exits = changes[changes == -1].index

            if len(entries) == 0:
                continue

            # Compute holding periods
            holding_periods = []
            for entry in entries:
                subsequent_exits = exits[exits > entry]
                if len(subsequent_exits) > 0:
                    exit_date = subsequent_exits[0]
                    holding_periods.append((exit_date - entry).days)

            if holding_periods:
                results.append({
                    'asset': col,
                    'n_trades': len(entries),
                    'avg_holding_days': np.mean(holding_periods),
                    'median_holding_days': np.median(holding_periods),
                    'max_holding_days': np.max(holding_periods),
                })

        return pd.DataFrame(results)
