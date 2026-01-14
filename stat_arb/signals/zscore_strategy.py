"""
Cross-sectional z-score mean reversion strategy.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple


class ZScoreStrategy:
    """
    Cross-sectional z-score mean reversion strategy.

    Signal construction (NO LOOKAHEAD):
    1. x_i,t = residual return for token i on day t (after PCA removal)
    2. s_i,t = sum_{j=0..H-1} x_i,t-j  (rolling sum with window H)
    3. z_i,t-1 = (s_i,t-1 - rolling_mean_L(s)) / (rolling_std_L(s) + eps)

    Portfolio construction:
    - Within each cluster: long bottom quantile, short top quantile
    - Inverse volatility weighting
    - Cluster-neutral and dollar-neutral
    """

    def __init__(
        self,
        H: int = 5,           # Momentum lookback (rolling sum window)
        L: int = 60,          # Z-score normalization window
        entry_threshold: float = 2.0,    # Enter when |z| > threshold
        exit_threshold: float = 0.5,     # Exit when |z| < threshold
        quantile: float = 0.2,           # Top/bottom quantile for trading
        use_threshold: bool = False,     # Use threshold vs quantile method
        vol_lookback: int = 20,          # Volatility estimation window
    ):
        self.H = H
        self.L = L
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.quantile = quantile
        self.use_threshold = use_threshold
        self.vol_lookback = vol_lookback

    def compute_signals(
        self,
        returns: pd.DataFrame,
        lag: int = 1,
    ) -> pd.DataFrame:
        """
        Compute z-score signals from returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Return series (typically residualized/ETH-excess returns)
        lag : int
            Signal lag to avoid lookahead (default 1)

        Returns
        -------
        pd.DataFrame
            Z-score signals, lagged by `lag` periods
        """
        # Step 1: Rolling sum of returns
        rolling_sum = returns.rolling(window=self.H, min_periods=1).sum()

        # Step 2: Z-score normalization
        rolling_mean = rolling_sum.rolling(window=self.L, min_periods=20).mean()
        rolling_std = rolling_sum.rolling(window=self.L, min_periods=20).std()
        zscore = (rolling_sum - rolling_mean) / (rolling_std + 1e-8)

        # Step 3: Lag signal
        zscore_lagged = zscore.shift(lag)

        return zscore_lagged

    def compute_volatility_weights(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute inverse volatility weights.
        """
        vol = returns.rolling(window=self.vol_lookback, min_periods=10).std()
        inv_vol = 1.0 / (vol + 1e-8)

        # Normalize cross-sectionally
        inv_vol_norm = inv_vol.div(inv_vol.sum(axis=1), axis=0)

        return inv_vol_norm

    def generate_target_weights(
        self,
        signals: pd.DataFrame,
        cluster_labels: np.ndarray,
        tokens: list,
        returns: Optional[pd.DataFrame] = None,
        clusters_to_trade: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Generate target portfolio weights from signals.

        Parameters
        ----------
        signals : pd.DataFrame
            Z-score signals (rows = dates, columns = tokens)
        cluster_labels : np.ndarray
            Cluster assignment for each token
        tokens : list
            List of token names corresponding to cluster_labels
        returns : pd.DataFrame, optional
            Returns for volatility weighting
        clusters_to_trade : list, optional
            Subset of clusters to trade (e.g., [0, 2] to skip cluster 1)

        Returns
        -------
        pd.DataFrame
            Target weights (rows = dates, columns = tokens)
        """
        # Create token -> cluster mapping
        token_to_cluster = {token: cluster_labels[i] for i, token in enumerate(tokens)}

        # Determine which clusters to trade
        if clusters_to_trade is None:
            clusters_to_trade = list(np.unique(cluster_labels))

        # Initialize weights
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        # Compute volatility weights if returns provided
        if returns is not None:
            vol_weights = self.compute_volatility_weights(returns)
        else:
            vol_weights = None

        # Process each date
        for date in signals.index:
            date_signals = signals.loc[date].dropna()

            if len(date_signals) == 0:
                continue

            # Process each cluster
            for cluster_id in clusters_to_trade:
                # Get tokens in this cluster
                cluster_tokens = [t for t in date_signals.index
                                  if t.replace('_returns', '') in token_to_cluster
                                  and token_to_cluster[t.replace('_returns', '')] == cluster_id]

                if len(cluster_tokens) < 2:
                    continue

                cluster_signals = date_signals[cluster_tokens]

                if self.use_threshold:
                    # Threshold-based entry/exit
                    w = self._threshold_weights(cluster_signals)
                else:
                    # Quantile-based
                    w = self._quantile_weights(cluster_signals)

                # Apply volatility weighting
                if vol_weights is not None and date in vol_weights.index:
                    vol_w = vol_weights.loc[date, cluster_tokens].fillna(1.0 / len(cluster_tokens))
                    vol_w = vol_w / vol_w.sum()  # Re-normalize
                    w = w * vol_w * len(cluster_tokens)  # Scale

                # Ensure cluster-neutral (sum = 0)
                w = w - w.mean()

                weights.loc[date, cluster_tokens] = w

        # Ensure dollar-neutral across all positions
        weights = weights.sub(weights.mean(axis=1), axis=0)

        return weights

    def _quantile_weights(self, signals: pd.Series) -> pd.Series:
        """
        Generate weights based on quantile ranking.

        Long bottom quantile (most negative z), short top quantile (most positive z).
        """
        n = len(signals)
        n_each_side = max(1, int(n * self.quantile))

        weights = pd.Series(0.0, index=signals.index)

        # Sort by signal
        sorted_signals = signals.sort_values()

        # Long bottom quantile (most negative z -> mean reversion expects increase)
        long_tokens = sorted_signals.index[:n_each_side]
        weights[long_tokens] = 1.0 / n_each_side

        # Short top quantile (most positive z -> mean reversion expects decrease)
        short_tokens = sorted_signals.index[-n_each_side:]
        weights[short_tokens] = -1.0 / n_each_side

        return weights

    def _threshold_weights(self, signals: pd.Series) -> pd.Series:
        """
        Generate weights based on z-score thresholds.
        """
        weights = pd.Series(0.0, index=signals.index)

        # Long when z < -entry_threshold
        long_mask = signals < -self.entry_threshold
        # Short when z > entry_threshold
        short_mask = signals > self.entry_threshold

        n_long = long_mask.sum()
        n_short = short_mask.sum()

        if n_long > 0:
            weights[long_mask] = 1.0 / n_long
        if n_short > 0:
            weights[short_mask] = -1.0 / n_short

        return weights

    def compute_holdings(
        self,
        target_weights: pd.DataFrame,
        prev_holdings: Optional[pd.DataFrame] = None,
        turnover_limit: float = 1.0,
    ) -> pd.DataFrame:
        """
        Apply turnover constraints to get actual holdings.
        """
        if prev_holdings is None or turnover_limit >= 1.0:
            return target_weights

        holdings = target_weights.copy()

        for i in range(1, len(holdings)):
            prev = prev_holdings.iloc[i-1] if i <= len(prev_holdings) else holdings.iloc[i-1]
            target = target_weights.iloc[i]

            # Compute desired change
            delta = target - prev

            # L1 norm of change
            turnover = delta.abs().sum()

            if turnover > turnover_limit:
                # Scale down changes
                scale = turnover_limit / turnover
                holdings.iloc[i] = prev + delta * scale

        return holdings
