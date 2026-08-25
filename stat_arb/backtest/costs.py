"""
Transaction cost modeling and analysis.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class TransactionCostModel:
    """
    BPS-per-side transaction cost model with sensitivity analysis.
    """

    def __init__(
        self,
        base_cost_bps: float = 50,
        cost_grid: List[float] = None,
    ):
        self.base_cost_bps = base_cost_bps
        self.cost_grid = cost_grid or [10, 25, 50, 100, 200]

    def compute_costs(
        self,
        turnover: pd.Series,
        cost_bps: Optional[float] = None,
    ) -> pd.Series:
        """
        Compute transaction costs from turnover.

        Cost = turnover * cost_bps / 10000

        Note: turnover is two-sided (buys + sells), so cost_bps is per-side.
        Total cost = turnover * 2 * cost_bps / 10000 for round-trip interpretation.
        Here we use single-sided: cost = turnover * cost_bps / 10000.
        """
        if cost_bps is None:
            cost_bps = self.base_cost_bps

        return turnover * cost_bps / 10000

    def compute_net_returns(
        self,
        gross_returns: pd.Series,
        turnover: pd.Series,
        cost_bps: Optional[float] = None,
    ) -> pd.Series:
        """Compute net returns after costs."""
        costs = self.compute_costs(turnover, cost_bps)
        return gross_returns - costs

    def sensitivity_analysis(
        self,
        gross_returns: pd.Series,
        turnover: pd.Series,
        cost_grid: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Run cost sensitivity analysis.

        Returns DataFrame with performance at each cost level.
        """
        if cost_grid is None:
            cost_grid = self.cost_grid

        results = []

        for cost_bps in cost_grid:
            net_returns = self.compute_net_returns(gross_returns, turnover, cost_bps)

            # Compute metrics
            ann_ret = net_returns.mean() * 365
            ann_vol = net_returns.std() * np.sqrt(365)
            sharpe = ann_ret / (ann_vol + 1e-8)

            cum_ret = net_returns.cumsum()
            max_dd = (cum_ret - cum_ret.expanding().max()).min()

            results.append({
                'cost_bps': cost_bps,
                'ann_return': ann_ret,
                'ann_vol': ann_vol,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'total_return': cum_ret.iloc[-1] if len(cum_ret) > 0 else 0,
            })

        return pd.DataFrame(results)

    def find_breakeven_cost(
        self,
        gross_returns: pd.Series,
        turnover: pd.Series,
        target_sharpe: float = 0.0,
        precision: float = 1.0,
    ) -> float:
        """
        Find cost level at which strategy breaks even (Sharpe = target).

        Uses binary search.
        """
        low, high = 0, 500

        while high - low > precision:
            mid = (low + high) / 2
            net_returns = self.compute_net_returns(gross_returns, turnover, mid)

            ann_ret = net_returns.mean() * 365
            ann_vol = net_returns.std() * np.sqrt(365)
            sharpe = ann_ret / (ann_vol + 1e-8)

            if sharpe > target_sharpe:
                low = mid
            else:
                high = mid

        return (low + high) / 2


class CostAnalyzer:
    """
    Detailed cost analysis and decomposition.
    """

    @staticmethod
    def decompose_pnl(
        gross_returns: pd.Series,
        costs: pd.Series,
    ) -> Dict:
        """
        Decompose P&L into gross and cost components.
        """
        net_returns = gross_returns - costs

        return {
            'gross_total': gross_returns.sum(),
            'cost_total': costs.sum(),
            'net_total': net_returns.sum(),
            'cost_drag_pct': costs.sum() / (gross_returns.sum() + 1e-8) * 100,
            'avg_daily_cost': costs.mean(),
            'max_daily_cost': costs.max(),
        }

    @staticmethod
    def compute_cost_efficiency(
        gross_returns: pd.Series,
        turnover: pd.Series,
    ) -> float:
        """
        Compute cost efficiency: gross return per unit of turnover.
        """
        return gross_returns.sum() / (turnover.sum() + 1e-8)

    @staticmethod
    def analyze_cost_by_period(
        gross_returns: pd.Series,
        costs: pd.Series,
        freq: str = 'ME',
    ) -> pd.DataFrame:
        """
        Analyze costs by time period.
        """
        net_returns = gross_returns - costs

        df = pd.DataFrame({
            'gross': gross_returns,
            'costs': costs,
            'net': net_returns,
        })

        grouped = df.groupby(pd.Grouper(freq=freq)).agg({
            'gross': 'sum',
            'costs': 'sum',
            'net': 'sum',
        })

        grouped['cost_ratio'] = grouped['costs'] / (grouped['gross'].abs() + 1e-8)

        return grouped
