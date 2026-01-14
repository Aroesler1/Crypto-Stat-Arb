"""
Portfolio construction with risk constraints.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict


class PortfolioConstructor:
    """
    Portfolio construction with multiple constraints.

    Constraints:
    - Dollar neutral (sum of weights = 0)
    - Factor neutral (PC1, ETH beta)
    - Position caps
    - Cluster concentration limits
    - Gross leverage cap
    - Volatility targeting
    """

    def __init__(
        self,
        dollar_neutral: bool = True,
        pc1_neutral: bool = True,
        eth_neutral: bool = False,
        max_position: float = 0.10,       # Max weight per position
        max_cluster_weight: float = 0.40,  # Max total weight per cluster
        leverage_cap: float = 1.5,         # Max gross leverage
        vol_target: Optional[float] = None,  # Target annualized vol
    ):
        self.dollar_neutral = dollar_neutral
        self.pc1_neutral = pc1_neutral
        self.eth_neutral = eth_neutral
        self.max_position = max_position
        self.max_cluster_weight = max_cluster_weight
        self.leverage_cap = leverage_cap
        self.vol_target = vol_target

    def construct_portfolio(
        self,
        target_weights: pd.DataFrame,
        returns: pd.DataFrame,
        cluster_labels: Optional[np.ndarray] = None,
        tokens: Optional[list] = None,
        pc1_betas: Optional[pd.DataFrame] = None,
        eth_betas: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Apply all constraints to target weights.

        Parameters
        ----------
        target_weights : pd.DataFrame
            Raw target weights from signal
        returns : pd.DataFrame
            Historical returns for risk estimation
        cluster_labels : np.ndarray
            Cluster assignments
        tokens : list
            Token names
        pc1_betas : pd.DataFrame
            PC1 betas for neutralization
        eth_betas : pd.DataFrame
            ETH betas for neutralization

        Returns
        -------
        pd.DataFrame
            Constrained portfolio weights
        """
        weights = target_weights.copy()

        # Apply position caps
        weights = self._apply_position_caps(weights)

        # Apply cluster concentration caps
        if cluster_labels is not None and tokens is not None:
            weights = self._apply_cluster_caps(weights, cluster_labels, tokens)

        # Apply factor neutralization
        if self.pc1_neutral and pc1_betas is not None:
            weights = self._enforce_factor_neutral(weights, pc1_betas, 'PC1')

        if self.eth_neutral and eth_betas is not None:
            weights = self._enforce_factor_neutral(weights, eth_betas, 'ETH')

        # Apply dollar neutrality
        if self.dollar_neutral:
            weights = self._enforce_dollar_neutral(weights)

        # Apply leverage cap
        weights = self._apply_leverage_cap(weights)

        # Apply volatility targeting
        if self.vol_target is not None:
            weights = self._apply_vol_target(weights, returns)

        return weights

    def _apply_position_caps(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Cap individual position sizes."""
        return weights.clip(lower=-self.max_position, upper=self.max_position)

    def _apply_cluster_caps(
        self,
        weights: pd.DataFrame,
        cluster_labels: np.ndarray,
        tokens: list,
    ) -> pd.DataFrame:
        """Cap total weight per cluster."""
        token_to_cluster = {token: cluster_labels[i] for i, token in enumerate(tokens)}

        result = weights.copy()

        for date in result.index:
            for cluster_id in np.unique(cluster_labels):
                # Get columns in this cluster
                cluster_cols = []
                for col in result.columns:
                    token = col.replace('_returns', '')
                    if token in token_to_cluster and token_to_cluster[token] == cluster_id:
                        cluster_cols.append(col)

                if not cluster_cols:
                    continue

                # Check cluster weight
                cluster_weights = result.loc[date, cluster_cols]
                gross_cluster = cluster_weights.abs().sum()

                if gross_cluster > self.max_cluster_weight:
                    # Scale down
                    scale = self.max_cluster_weight / gross_cluster
                    result.loc[date, cluster_cols] = cluster_weights * scale

        return result

    def _enforce_dollar_neutral(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Ensure sum of weights = 0."""
        return weights.sub(weights.mean(axis=1), axis=0)

    def _enforce_factor_neutral(
        self,
        weights: pd.DataFrame,
        factor_betas: pd.DataFrame,
        factor_name: str,
    ) -> pd.DataFrame:
        """
        Neutralize to a factor (make sum(w_i * beta_i) = 0).
        """
        result = weights.copy()

        for date in result.index:
            w = result.loc[date].dropna()

            if len(w) == 0:
                continue

            # Get betas for these assets
            if date in factor_betas.index:
                betas = factor_betas.loc[date]
            else:
                # Use latest available
                prior = factor_betas.index[factor_betas.index <= date]
                if len(prior) == 0:
                    continue
                betas = factor_betas.loc[prior[-1]]

            # Align
            common = w.index.intersection(betas.index)
            if len(common) == 0:
                continue

            w_aligned = w.loc[common].values
            b_aligned = betas.loc[common].values

            # Neutralize: w_new = w - lambda * b
            # where lambda = (w @ b) / (b @ b)
            b_sq = b_aligned @ b_aligned
            if b_sq > 1e-8:
                lam = (w_aligned @ b_aligned) / b_sq
                w_new = w_aligned - lam * b_aligned
                result.loc[date, common] = w_new

        return result

    def _apply_leverage_cap(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Cap gross leverage."""
        result = weights.copy()

        for date in result.index:
            gross = result.loc[date].abs().sum()
            if gross > self.leverage_cap:
                scale = self.leverage_cap / gross
                result.loc[date] = result.loc[date] * scale

        return result

    def _apply_vol_target(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        lookback: int = 60,
    ) -> pd.DataFrame:
        """Scale weights to target portfolio volatility."""
        result = weights.copy()

        # Compute rolling portfolio volatility
        for i in range(lookback, len(result)):
            date = result.index[i]
            w = result.loc[date].dropna()

            # Historical returns for vol estimation
            hist_start = max(0, i - lookback)
            hist_returns = returns.iloc[hist_start:i]

            # Align columns
            common_cols = w.index.intersection(hist_returns.columns)
            if len(common_cols) < 2:
                continue

            w_aligned = w.loc[common_cols].values
            ret_aligned = hist_returns[common_cols].values

            # Portfolio return series
            port_ret = ret_aligned @ w_aligned

            # Annualized vol
            port_vol = np.std(port_ret) * np.sqrt(365)

            if port_vol > 1e-8:
                scale = self.vol_target / port_vol
                # Cap scaling to avoid extreme leverage
                scale = min(scale, self.leverage_cap / (w.abs().sum() + 1e-8))
                result.loc[date, common_cols] = w_aligned * scale

        return result
