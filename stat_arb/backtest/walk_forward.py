"""
Walk-forward (rolling OOS) backtest framework.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtest."""
    train_window: int = 365        # Training window in days
    test_window: int = 28          # Test/OOS window in days
    refit_frequency: int = 28      # How often to refit (days)
    min_train_history: int = 180   # Minimum training history required
    gap_days: int = 0              # Gap between train and test (avoid lookahead)


class WalkForwardBacktest:
    """
    Rolling walk-forward out-of-sample backtest.

    Ensures no lookahead:
    - Training data: [t - train_window - gap, t - gap)
    - Test data: [t, t + test_window)
    - Refit every refit_frequency days
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()
        self.fold_results_ = []

    def generate_folds(
        self,
        dates: pd.DatetimeIndex,
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate train/test date folds.

        Returns list of (train_dates, test_dates) tuples.
        """
        folds = []
        n_dates = len(dates)

        # Start after minimum training period
        start_idx = self.config.min_train_history

        current_idx = start_idx
        while current_idx < n_dates:
            # Training window
            train_end = current_idx - self.config.gap_days
            train_start = max(0, train_end - self.config.train_window)

            train_dates = dates[train_start:train_end]

            # Test window
            test_start = current_idx
            test_end = min(n_dates, test_start + self.config.test_window)

            test_dates = dates[test_start:test_end]

            if len(train_dates) >= self.config.min_train_history and len(test_dates) > 0:
                folds.append((train_dates, test_dates))

            current_idx += self.config.refit_frequency

        return folds

    def run_backtest(
        self,
        returns: pd.DataFrame,
        signal_func: Callable,
        portfolio_func: Callable,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
        """
        Run complete walk-forward backtest.

        Parameters
        ----------
        returns : pd.DataFrame
            Full return series
        signal_func : Callable
            Function(train_returns, **kwargs) -> signals for test period
        portfolio_func : Callable
            Function(signals, returns, **kwargs) -> weights

        Returns
        -------
        weights : pd.DataFrame
            Portfolio weights over full period
        fold_info : List[Dict]
            Information about each fold (for diagnostics)
        """
        folds = self.generate_folds(returns.index)

        all_weights = []
        fold_info = []

        for fold_idx, (train_dates, test_dates) in enumerate(folds):
            # Extract training data
            train_returns = returns.loc[train_dates]

            # Generate signals/model on training data
            # Signal function should return signals for test period
            try:
                signals = signal_func(
                    train_returns=train_returns,
                    test_dates=test_dates,
                    full_returns=returns,
                    **kwargs,
                )

                # Generate portfolio weights
                test_returns = returns.loc[test_dates]
                weights = portfolio_func(
                    signals=signals,
                    returns=test_returns,
                    train_returns=train_returns,
                    **kwargs,
                )

                all_weights.append(weights)

                fold_info.append({
                    'fold': fold_idx,
                    'train_start': train_dates[0],
                    'train_end': train_dates[-1],
                    'test_start': test_dates[0],
                    'test_end': test_dates[-1],
                    'n_train': len(train_dates),
                    'n_test': len(test_dates),
                    'status': 'success',
                })

            except Exception as e:
                fold_info.append({
                    'fold': fold_idx,
                    'train_start': train_dates[0],
                    'train_end': train_dates[-1],
                    'test_start': test_dates[0],
                    'test_end': test_dates[-1],
                    'n_train': len(train_dates),
                    'n_test': len(test_dates),
                    'status': 'error',
                    'error': str(e),
                })

        # Combine all weights
        if all_weights:
            combined_weights = pd.concat(all_weights)
            # Remove duplicate indices (from overlapping test periods)
            combined_weights = combined_weights[~combined_weights.index.duplicated(keep='first')]
            combined_weights = combined_weights.sort_index()
        else:
            combined_weights = pd.DataFrame()

        return combined_weights, fold_info

    def run_fold(
        self,
        returns: pd.DataFrame,
        train_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex,
        clustering_func: Callable,
        signal_func: Callable,
        portfolio_func: Callable,
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run a single fold of walk-forward backtest.

        More granular control than run_backtest.
        """
        # Training phase
        train_returns = returns.loc[train_dates]

        # Fit clustering on training data
        cluster_result = clustering_func(train_returns, **kwargs)

        # Generate signals
        signals = signal_func(
            train_returns=train_returns,
            cluster_result=cluster_result,
            **kwargs,
        )

        # Apply to test period
        test_returns = returns.loc[test_dates]

        # Get signals for test dates (using model fitted on training)
        test_signals = signal_func(
            train_returns=train_returns,
            test_dates=test_dates,
            cluster_result=cluster_result,
            full_returns=returns,
            **kwargs,
        )

        # Generate weights
        weights = portfolio_func(
            signals=test_signals,
            returns=test_returns,
            cluster_result=cluster_result,
            **kwargs,
        )

        fold_result = {
            'train_start': train_dates[0],
            'train_end': train_dates[-1],
            'test_start': test_dates[0],
            'test_end': test_dates[-1],
            'cluster_result': cluster_result,
        }

        return weights, fold_result
