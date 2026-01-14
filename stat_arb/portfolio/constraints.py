"""
Neutrality constraints for portfolio construction.
"""
import numpy as np
import pandas as pd
from typing import Optional, List


class NeutralityConstraints:
    """
    Multi-factor neutralization constraints.
    """

    def __init__(
        self,
        factors: List[str] = None,
        tolerance: float = 0.01,
    ):
        """
        Parameters
        ----------
        factors : list
            List of factor names to neutralize to
        tolerance : float
            Allowed deviation from zero exposure
        """
        self.factors = factors or []
        self.tolerance = tolerance

    def neutralize(
        self,
        weights: pd.Series,
        factor_exposures: pd.DataFrame,
    ) -> pd.Series:
        """
        Neutralize weights to multiple factors simultaneously.

        Uses least-squares adjustment to minimize change while
        achieving zero exposure to all factors.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights
        factor_exposures : pd.DataFrame
            Factor exposures (rows = assets, columns = factors)

        Returns
        -------
        pd.Series
            Neutralized weights
        """
        # Align
        common = weights.index.intersection(factor_exposures.index)
        if len(common) == 0:
            return weights

        w = weights.loc[common].values
        F = factor_exposures.loc[common].values  # (n_assets, n_factors)

        # Current factor exposures
        current_exposure = F.T @ w  # (n_factors,)

        # Adjust weights to neutralize
        # min ||w_new - w||^2 s.t. F' w_new = 0
        # Solution: w_new = w - F @ (F'F)^{-1} @ F' @ w

        FtF = F.T @ F
        if np.linalg.cond(FtF) < 1e10:
            FtF_inv = np.linalg.inv(FtF + 1e-8 * np.eye(len(FtF)))
            adjustment = F @ FtF_inv @ current_exposure
            w_new = w - adjustment
        else:
            # Fallback to sequential neutralization
            w_new = w.copy()
            for j in range(F.shape[1]):
                f = F[:, j]
                exposure = f @ w_new
                if np.abs(f @ f) > 1e-8:
                    lam = exposure / (f @ f)
                    w_new = w_new - lam * f

        return pd.Series(w_new, index=common)

    def check_neutrality(
        self,
        weights: pd.Series,
        factor_exposures: pd.DataFrame,
    ) -> pd.Series:
        """
        Check factor exposures of portfolio.

        Returns Series of exposures to each factor.
        """
        common = weights.index.intersection(factor_exposures.index)
        if len(common) == 0:
            return pd.Series(dtype=float)

        w = weights.loc[common].values
        F = factor_exposures.loc[common].values

        exposures = F.T @ w

        return pd.Series(exposures, index=factor_exposures.columns)

    def compute_factor_betas(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        window: int = 60,
    ) -> pd.DataFrame:
        """
        Compute rolling factor betas for each asset.

        Returns DataFrame (dates x assets) with betas.
        """
        betas = {}

        for factor_col in factor_returns.columns:
            factor_ret = factor_returns[factor_col]
            factor_betas = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

            for col in returns.columns:
                asset_ret = returns[col]

                # Rolling beta
                for i in range(window, len(returns)):
                    idx = returns.index[i]
                    hist_asset = asset_ret.iloc[i-window:i].values
                    hist_factor = factor_ret.iloc[i-window:i].values

                    # Handle NaNs
                    mask = ~(np.isnan(hist_asset) | np.isnan(hist_factor))
                    if mask.sum() < 20:
                        continue

                    cov = np.cov(hist_asset[mask], hist_factor[mask])[0, 1]
                    var = np.var(hist_factor[mask])

                    if var > 1e-8:
                        factor_betas.loc[idx, col] = cov / var

            betas[factor_col] = factor_betas

        return betas
