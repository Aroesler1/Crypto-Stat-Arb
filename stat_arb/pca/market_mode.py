"""
PCA market mode extraction and residualization.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class MarketModeExtractor:
    """
    Extract market modes (principal components) from returns
    and compute residuals.
    """

    def __init__(
        self,
        n_components: int = 1,
        standardize: bool = True,
    ):
        self.n_components = n_components
        self.standardize = standardize
        self.pca_ = None
        self.scaler_ = None
        self.loadings_ = None
        self.explained_variance_ratio_ = None

    def fit(
        self,
        returns: pd.DataFrame,
    ) -> 'MarketModeExtractor':
        """
        Fit PCA on returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Return series (rows = dates, columns = assets)

        Returns
        -------
        self
        """
        X = returns.dropna().values

        # Standardize if requested
        if self.standardize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        # Fit PCA
        self.pca_ = PCA(n_components=min(self.n_components, X.shape[1]))
        self.pca_.fit(X)

        self.loadings_ = self.pca_.components_
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_

        return self

    def get_factor_returns(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Get factor (PC) returns time series.

        Returns DataFrame with columns PC1, PC2, etc.
        """
        X = returns.values.copy()

        if self.standardize and self.scaler_ is not None:
            X = self.scaler_.transform(X)

        scores = X @ self.loadings_.T

        factor_cols = [f'PC{i+1}' for i in range(scores.shape[1])]
        return pd.DataFrame(scores, index=returns.index, columns=factor_cols)

    def residualize(
        self,
        returns: pd.DataFrame,
        n_components: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Remove first n_components PCs from returns.

        Returns residual returns.
        """
        if n_components is None:
            n_components = self.n_components

        X = returns.values.copy()

        if self.standardize and self.scaler_ is not None:
            X_std = self.scaler_.transform(X)
        else:
            X_std = X

        # Compute scores for components to remove
        scores = X_std @ self.loadings_[:n_components].T

        # Reconstruct contribution of these components
        reconstruction = scores @ self.loadings_[:n_components]

        # Residuals in standardized space
        residuals_std = X_std - reconstruction

        # Transform back to original space
        if self.standardize and self.scaler_ is not None:
            residuals = self.scaler_.inverse_transform(residuals_std)
        else:
            residuals = residuals_std

        return pd.DataFrame(residuals, index=returns.index, columns=returns.columns)

    def compute_betas(
        self,
        returns: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute factor betas for each asset.

        Returns DataFrame with beta to each PC.
        """
        if factor_returns is None:
            factor_returns = self.get_factor_returns(returns)

        betas = {}
        for col in returns.columns:
            asset_ret = returns[col].values
            for pc_col in factor_returns.columns:
                pc_ret = factor_returns[pc_col].values
                # Simple OLS beta
                cov = np.cov(asset_ret, pc_ret, ddof=1)[0, 1]
                var = np.var(pc_ret, ddof=1)
                betas[(col, pc_col)] = cov / (var + 1e-8)

        # Reshape to DataFrame
        beta_df = pd.DataFrame(betas, index=['beta']).T
        beta_df.index = pd.MultiIndex.from_tuples(beta_df.index, names=['asset', 'factor'])
        beta_df = beta_df.unstack('factor')
        beta_df.columns = beta_df.columns.droplevel(0)

        return beta_df

    def compute_factor_exposure(
        self,
        weights: pd.Series,
        factor_returns: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute portfolio's exposure to each factor.
        """
        if factor_returns is None and returns is not None:
            factor_returns = self.get_factor_returns(returns)

        # Factor exposure = sum of (weight * loading)
        exposures = {}
        for i, pc_col in enumerate(factor_returns.columns):
            loading = self.loadings_[i]
            # Align weights with loadings
            aligned_weights = weights.reindex(returns.columns if returns is not None else weights.index).fillna(0)
            exposures[pc_col] = (aligned_weights.values * loading).sum()

        return pd.Series(exposures)

    def neutralize_to_factor(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        factor_idx: int = 0,
    ) -> pd.Series:
        """
        Neutralize portfolio weights to a specific factor (e.g., PC1).

        Adjusts weights so that sum(w_i * beta_i) = 0 for the factor.
        """
        # Compute betas
        betas = self.compute_betas(returns)
        pc_col = f'PC{factor_idx + 1}'

        if pc_col not in betas.columns:
            return weights

        beta_vector = betas[pc_col]

        # Align
        common_assets = weights.index.intersection(beta_vector.index)
        w = weights.loc[common_assets].values
        b = beta_vector.loc[common_assets].values

        # Current exposure
        current_exposure = (w * b).sum()

        # Neutralize by adjusting weights proportionally to their betas
        # w_new = w - lambda * b, where lambda chosen so sum(w_new * b) = 0
        # sum(w * b) - lambda * sum(b * b) = 0
        # lambda = sum(w * b) / sum(b * b)
        if np.abs(b @ b) > 1e-8:
            lam = (w @ b) / (b @ b)
            w_new = w - lam * b
        else:
            w_new = w

        return pd.Series(w_new, index=common_assets)


class RollingPCAExtractor:
    """
    Rolling/walk-forward PCA for proper out-of-sample residualization.
    """

    def __init__(
        self,
        n_components: int = 1,
        window: int = 365,
        min_periods: int = 180,
        standardize: bool = True,
    ):
        self.n_components = n_components
        self.window = window
        self.min_periods = min_periods
        self.standardize = standardize

    def fit_transform_rolling(
        self,
        returns: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply rolling PCA and return residuals.

        No lookahead: PCA fit on [t-window, t-1], applied to day t.

        Returns
        -------
        residuals : pd.DataFrame
            Residualized returns
        factor_returns : pd.DataFrame
            Factor return time series
        explained_var : pd.DataFrame
            Explained variance ratio over time
        """
        dates = returns.index
        residuals = pd.DataFrame(index=dates, columns=returns.columns, dtype=float)
        factor_returns = pd.DataFrame(index=dates, columns=[f'PC{i+1}' for i in range(self.n_components)], dtype=float)
        explained_var = pd.DataFrame(index=dates, columns=[f'PC{i+1}_var' for i in range(self.n_components)], dtype=float)

        for i in range(self.min_periods, len(dates)):
            # Training window: [i-window, i-1]
            train_start = max(0, i - self.window)
            train_end = i  # exclusive
            test_idx = i

            train_data = returns.iloc[train_start:train_end].dropna(how='all')

            if len(train_data) < self.min_periods:
                continue

            # Fit PCA on training data
            extractor = MarketModeExtractor(
                n_components=self.n_components,
                standardize=self.standardize,
            )
            extractor.fit(train_data)

            # Apply to test point
            test_data = returns.iloc[[test_idx]]

            # Get factor returns
            factors = extractor.get_factor_returns(test_data)
            factor_returns.iloc[test_idx] = factors.iloc[0].values

            # Get residuals
            resid = extractor.residualize(test_data, self.n_components)
            residuals.iloc[test_idx] = resid.iloc[0].values

            # Store explained variance
            for j in range(min(self.n_components, len(extractor.explained_variance_ratio_))):
                explained_var.iloc[test_idx, j] = extractor.explained_variance_ratio_[j]

        return residuals, factor_returns, explained_var
