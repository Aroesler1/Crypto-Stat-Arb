"""
Core backtest engine for P&L computation.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class BacktestEngine:
    """
    Core P&L computation with no lookahead bias.

    Execution convention:
    - Signals computed using data through t-1
    - Trades executed at day t close
    - P&L = sum(w_{t-1} * r_t) where w_{t-1} are weights determined from t-1 signal
    """

    def __init__(
        self,
        cost_bps: float = 50,
        execution: str = 'close',  # 'close' or 'open'
    ):
        self.cost_bps = cost_bps
        self.execution = execution

    def compute_returns(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute daily portfolio returns.

        IMPORTANT: Weights at time t are applied to returns at time t+1
        (weights determined at close of t, earn return of t+1).

        To avoid lookahead: weights should already be lagged in the signal generation.
        Here we assume weights[t] represents the position held going INTO day t,
        so return[t] = weights[t] * asset_returns[t].
        """
        # Align
        common_dates = weights.index.intersection(returns.index)
        common_cols = weights.columns.intersection(returns.columns)

        if len(common_cols) == 0:
            return pd.Series(0.0, index=common_dates)

        w = weights.loc[common_dates, common_cols]
        r = returns.loc[common_dates, common_cols]

        # Portfolio return = sum of (weight * asset return)
        # Weights at t earn returns at t (weights were set at t-1 close)
        port_ret = (w * r).sum(axis=1)

        return port_ret

    def compute_gross_returns(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """Compute returns before transaction costs."""
        return self.compute_returns(weights, returns)

    def compute_turnover(
        self,
        weights: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute daily turnover (sum of absolute weight changes).
        """
        delta = weights.diff()
        turnover = delta.abs().sum(axis=1)
        return turnover

    def compute_costs(
        self,
        turnover: pd.Series,
        cost_bps: Optional[float] = None,
    ) -> pd.Series:
        """
        Compute transaction costs.

        Cost = turnover * cost_bps / 10000
        """
        if cost_bps is None:
            cost_bps = self.cost_bps

        return turnover * cost_bps / 10000

    def compute_net_returns(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        cost_bps: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute net returns after transaction costs.

        Returns (net_returns, gross_returns, costs).
        """
        gross = self.compute_gross_returns(weights, returns)
        turnover = self.compute_turnover(weights)
        costs = self.compute_costs(turnover, cost_bps)

        # Align
        common_idx = gross.index.intersection(costs.index)
        gross = gross.loc[common_idx]
        costs = costs.loc[common_idx]

        net = gross - costs

        return net, gross, costs

    def compute_cumulative_returns(
        self,
        returns: pd.Series,
        log: bool = True,
    ) -> pd.Series:
        """
        Compute cumulative returns (equity curve).
        """
        if log:
            return returns.cumsum()
        else:
            return (1 + returns).cumprod() - 1

    def compute_drawdown(
        self,
        cumulative_returns: pd.Series,
    ) -> pd.Series:
        """
        Compute drawdown from equity curve.
        """
        rolling_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - rolling_max
        return drawdown

    def compute_exposure_stats(
        self,
        weights: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute exposure statistics over time.
        """
        stats = pd.DataFrame(index=weights.index)

        stats['gross_exposure'] = weights.abs().sum(axis=1)
        stats['net_exposure'] = weights.sum(axis=1)
        stats['long_exposure'] = weights.clip(lower=0).sum(axis=1)
        stats['short_exposure'] = (-weights.clip(upper=0)).sum(axis=1)
        stats['n_longs'] = (weights > 0.001).sum(axis=1)
        stats['n_shorts'] = (weights < -0.001).sum(axis=1)

        return stats
