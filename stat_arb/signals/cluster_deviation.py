"""
Cluster deviation-to-mean strategy.

Trade token's deviation from its cluster composite return.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict


class ClusterDeviationStrategy:
    """
    Cluster deviation-to-mean strategy.

    For each token, compute its deviation from the cluster composite return
    and trade mean reversion on the deviation.
    """

    def __init__(
        self,
        lookback: int = 20,          # Cumulative deviation window
        zscore_window: int = 60,     # Z-score normalization window
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        composite_type: str = 'equal',  # 'equal' or 'vol_weighted'
        vol_lookback: int = 20,
    ):
        self.lookback = lookback
        self.zscore_window = zscore_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.composite_type = composite_type
        self.vol_lookback = vol_lookback

    def compute_cluster_composites(
        self,
        returns: pd.DataFrame,
        cluster_labels: np.ndarray,
        tokens: list,
    ) -> Dict[int, pd.Series]:
        """
        Compute composite return for each cluster.

        Returns dict mapping cluster_id -> composite return series.
        """
        token_to_cluster = {token: cluster_labels[i] for i, token in enumerate(tokens)}

        composites = {}
        for cluster_id in np.unique(cluster_labels):
            # Get columns for this cluster
            cluster_cols = []
            for col in returns.columns:
                token = col.replace('_returns', '')
                if token in token_to_cluster and token_to_cluster[token] == cluster_id:
                    cluster_cols.append(col)

            if not cluster_cols:
                continue

            cluster_returns = returns[cluster_cols]

            if self.composite_type == 'equal':
                composite = cluster_returns.mean(axis=1)
            else:
                # Inverse volatility weighted
                vol = cluster_returns.rolling(window=self.vol_lookback, min_periods=5).std()
                inv_vol = 1.0 / (vol + 1e-8)
                weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
                composite = (cluster_returns * weights).sum(axis=1)

            composites[cluster_id] = composite

        return composites

    def compute_deviations(
        self,
        returns: pd.DataFrame,
        composites: Dict[int, pd.Series],
        cluster_labels: np.ndarray,
        tokens: list,
    ) -> pd.DataFrame:
        """
        Compute token deviations from cluster composite.
        """
        token_to_cluster = {token: cluster_labels[i] for i, token in enumerate(tokens)}

        deviations = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

        for col in returns.columns:
            token = col.replace('_returns', '')
            if token not in token_to_cluster:
                continue

            cluster_id = token_to_cluster[token]
            if cluster_id not in composites:
                continue

            # Deviation = token return - cluster composite
            deviations[col] = returns[col] - composites[cluster_id]

        return deviations

    def compute_signals(
        self,
        returns: pd.DataFrame,
        cluster_labels: np.ndarray,
        tokens: list,
        lag: int = 1,
    ) -> pd.DataFrame:
        """
        Compute deviation z-score signals.
        """
        # Compute cluster composites
        composites = self.compute_cluster_composites(returns, cluster_labels, tokens)

        # Compute deviations
        deviations = self.compute_deviations(returns, composites, cluster_labels, tokens)

        # Cumulative deviation
        cum_dev = deviations.rolling(window=self.lookback, min_periods=1).sum()

        # Z-score
        rolling_mean = cum_dev.rolling(window=self.zscore_window, min_periods=20).mean()
        rolling_std = cum_dev.rolling(window=self.zscore_window, min_periods=20).std()
        zscore = (cum_dev - rolling_mean) / (rolling_std + 1e-8)

        # Lag
        return zscore.shift(lag)

    def generate_target_weights(
        self,
        signals: pd.DataFrame,
        cluster_labels: np.ndarray,
        tokens: list,
        returns: Optional[pd.DataFrame] = None,
        clusters_to_trade: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Generate target weights from deviation signals.

        Long tokens with negative z-score deviation (underperformed cluster).
        Short tokens with positive z-score deviation (outperformed cluster).
        """
        token_to_cluster = {token: cluster_labels[i] for i, token in enumerate(tokens)}

        if clusters_to_trade is None:
            clusters_to_trade = list(np.unique(cluster_labels))

        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for date in signals.index:
            date_signals = signals.loc[date].dropna()

            for cluster_id in clusters_to_trade:
                cluster_cols = [col for col in date_signals.index
                                if col.replace('_returns', '') in token_to_cluster
                                and token_to_cluster[col.replace('_returns', '')] == cluster_id]

                if len(cluster_cols) < 2:
                    continue

                cluster_signals = date_signals[cluster_cols]

                # Long negative z (underperformed), short positive z (outperformed)
                long_mask = cluster_signals < -self.entry_threshold
                short_mask = cluster_signals > self.entry_threshold

                n_long = long_mask.sum()
                n_short = short_mask.sum()

                w = pd.Series(0.0, index=cluster_cols)

                if n_long > 0:
                    w[long_mask] = 1.0 / n_long
                if n_short > 0:
                    w[short_mask] = -1.0 / n_short

                # Cluster-neutralize over the SELECTED names only; demeaning
                # all cluster members would open positions in names that never
                # crossed the entry threshold, inflating turnover
                selected = w != 0.0
                if selected.any():
                    w[selected] = w[selected] - w[selected].mean()

                weights.loc[date, cluster_cols] = w

        # Dollar-neutralize across traded positions only (untraded names
        # must stay at exactly zero weight)
        traded_mask = weights != 0.0
        n_traded = traded_mask.sum(axis=1)
        net = weights.sum(axis=1)
        adjust = (net / n_traded.replace(0, np.nan)).fillna(0.0)
        weights = weights.sub(adjust, axis=0).where(traded_mask, 0.0)

        return weights
