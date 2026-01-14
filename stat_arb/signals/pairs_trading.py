"""
Pairs trading / cointegration strategy within clusters.
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from itertools import combinations
import warnings


class PairsTradingStrategy:
    """
    Cointegration-based pairs trading within clusters.

    For each cluster:
    1. Identify cointegrated pairs using Engle-Granger test
    2. Compute spread z-score
    3. Trade mean reversion on spread
    """

    def __init__(
        self,
        min_history: int = 90,          # Minimum history for cointegration test
        coint_pvalue: float = 0.05,     # Cointegration p-value threshold
        zscore_window: int = 20,        # Spread z-score window
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        max_pairs_per_cluster: int = 5,
        hedge_ratio_method: str = 'ols',  # 'ols' or 'tls'
    ):
        self.min_history = min_history
        self.coint_pvalue = coint_pvalue
        self.zscore_window = zscore_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_pairs_per_cluster = max_pairs_per_cluster
        self.hedge_ratio_method = hedge_ratio_method

    def test_cointegration(
        self,
        price1: pd.Series,
        price2: pd.Series,
    ) -> Tuple[bool, float, float]:
        """
        Test cointegration between two price series using Engle-Granger.

        Returns (is_cointegrated, p_value, hedge_ratio).
        """
        try:
            from statsmodels.tsa.stattools import coint
        except ImportError:
            # Fallback: simple correlation-based proxy
            corr = price1.corr(price2)
            return abs(corr) > 0.7, 1 - abs(corr), 1.0

        # Align
        common_idx = price1.index.intersection(price2.index)
        p1 = price1.loc[common_idx].dropna()
        p2 = price2.loc[common_idx].dropna()

        common_idx = p1.index.intersection(p2.index)
        p1 = p1.loc[common_idx]
        p2 = p2.loc[common_idx]

        if len(p1) < self.min_history:
            return False, 1.0, 1.0

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score, pvalue, _ = coint(p1.values, p2.values)

            # Compute hedge ratio via OLS
            if self.hedge_ratio_method == 'ols':
                hedge_ratio = np.polyfit(p2.values, p1.values, 1)[0]
            else:
                # Total least squares (orthogonal regression)
                from scipy import odr
                def f(B, x):
                    return B[0] * x + B[1]
                model = odr.Model(f)
                data = odr.Data(p2.values, p1.values)
                odr_obj = odr.ODR(data, model, beta0=[1., 0.])
                output = odr_obj.run()
                hedge_ratio = output.beta[0]

            return pvalue < self.coint_pvalue, pvalue, hedge_ratio

        except Exception:
            return False, 1.0, 1.0

    def find_cointegrated_pairs(
        self,
        prices: pd.DataFrame,
        cluster_labels: np.ndarray,
        tokens: list,
    ) -> Dict[int, List[Tuple[str, str, float]]]:
        """
        Find cointegrated pairs within each cluster.

        Returns dict mapping cluster_id -> list of (token1, token2, hedge_ratio).
        """
        token_to_cluster = {token: cluster_labels[i] for i, token in enumerate(tokens)}

        pairs_by_cluster = {}

        for cluster_id in np.unique(cluster_labels):
            cluster_tokens = [t for t in tokens if token_to_cluster.get(t) == cluster_id]

            if len(cluster_tokens) < 2:
                continue

            # Get prices for cluster tokens
            cluster_prices = prices[[t for t in cluster_tokens if t in prices.columns]]

            pairs = []
            for t1, t2 in combinations(cluster_prices.columns, 2):
                is_coint, pvalue, hedge_ratio = self.test_cointegration(
                    cluster_prices[t1], cluster_prices[t2]
                )
                if is_coint:
                    pairs.append((t1, t2, hedge_ratio, pvalue))

            # Sort by p-value and take top pairs
            pairs.sort(key=lambda x: x[3])
            pairs_by_cluster[cluster_id] = [(p[0], p[1], p[2]) for p in pairs[:self.max_pairs_per_cluster]]

        return pairs_by_cluster

    def compute_spread(
        self,
        price1: pd.Series,
        price2: pd.Series,
        hedge_ratio: float,
    ) -> pd.Series:
        """
        Compute spread: spread = price1 - hedge_ratio * price2
        """
        return price1 - hedge_ratio * price2

    def compute_spread_zscore(
        self,
        spread: pd.Series,
    ) -> pd.Series:
        """
        Compute z-score of spread.
        """
        rolling_mean = spread.rolling(window=self.zscore_window, min_periods=10).mean()
        rolling_std = spread.rolling(window=self.zscore_window, min_periods=10).std()
        return (spread - rolling_mean) / (rolling_std + 1e-8)

    def generate_pair_signals(
        self,
        prices: pd.DataFrame,
        pairs_by_cluster: Dict[int, List[Tuple[str, str, float]]],
    ) -> pd.DataFrame:
        """
        Generate trading signals for each pair.

        Returns DataFrame with columns for each pair showing position
        (-1 = short spread, 0 = flat, 1 = long spread).
        """
        all_signals = {}

        for cluster_id, pairs in pairs_by_cluster.items():
            for t1, t2, hedge_ratio in pairs:
                if t1 not in prices.columns or t2 not in prices.columns:
                    continue

                spread = self.compute_spread(prices[t1], prices[t2], hedge_ratio)
                zscore = self.compute_spread_zscore(spread)

                # Generate signals
                signal = pd.Series(0, index=prices.index)

                # Long spread when z < -entry_threshold
                signal[zscore < -self.entry_threshold] = 1
                # Short spread when z > entry_threshold
                signal[zscore > self.entry_threshold] = -1

                # Exit when z crosses exit_threshold (simplified: just set to 0)
                # In practice would need state tracking

                pair_name = f"{t1}_{t2}"
                all_signals[pair_name] = signal

        return pd.DataFrame(all_signals)

    def generate_target_weights(
        self,
        prices: pd.DataFrame,
        pairs_by_cluster: Dict[int, List[Tuple[str, str, float]]],
        weight_per_pair: float = 0.1,
    ) -> pd.DataFrame:
        """
        Generate target weights from pair signals.
        """
        tokens = set()
        for pairs in pairs_by_cluster.values():
            for t1, t2, _ in pairs:
                tokens.add(t1)
                tokens.add(t2)

        weights = pd.DataFrame(0.0, index=prices.index, columns=list(tokens))

        for cluster_id, pairs in pairs_by_cluster.items():
            for t1, t2, hedge_ratio in pairs:
                if t1 not in prices.columns or t2 not in prices.columns:
                    continue

                spread = self.compute_spread(prices[t1], prices[t2], hedge_ratio)
                zscore = self.compute_spread_zscore(spread)

                # Lagged signal (no lookahead)
                zscore_lagged = zscore.shift(1)

                for date in prices.index:
                    if pd.isna(zscore_lagged.get(date)):
                        continue

                    z = zscore_lagged[date]

                    if z < -self.entry_threshold:
                        # Long spread: long t1, short t2
                        weights.loc[date, t1] += weight_per_pair
                        weights.loc[date, t2] -= weight_per_pair * hedge_ratio
                    elif z > self.entry_threshold:
                        # Short spread: short t1, long t2
                        weights.loc[date, t1] -= weight_per_pair
                        weights.loc[date, t2] += weight_per_pair * hedge_ratio

        # Normalize to ensure dollar-neutral
        weights = weights.sub(weights.mean(axis=1), axis=0)

        return weights
