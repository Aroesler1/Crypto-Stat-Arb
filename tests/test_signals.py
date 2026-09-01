"""
Regression tests for signal -> weight construction.

Covers the 2026-08 fixes:
- neutralization must not open positions in names the selection rule skipped
- inverse-vol weights must be lagged (no same-day volatility information)
"""
import numpy as np
import pandas as pd
import pytest

from stat_arb.signals.zscore_strategy import ZScoreStrategy
from stat_arb.signals.cluster_deviation import ClusterDeviationStrategy


TOKENS = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff']
COLS = [f'{t}_returns' for t in TOKENS]
CLUSTERS = np.array([0, 0, 0, 1, 1, 1])


def make_returns(seed=7, n_days=40):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2024-01-01', periods=n_days, freq='D')
    return pd.DataFrame(rng.normal(0, 0.02, size=(n_days, len(COLS))), index=idx, columns=COLS)


def make_signals(returns):
    # Fixed cross-sectional ranking, constant through time: aaa most negative
    # z (long candidate), ccc most positive (short candidate), bbb neutral;
    # same pattern in cluster 1.
    row = pd.Series([-2.5, 0.0, 2.5, -2.5, 0.0, 2.5], index=COLS)
    return pd.DataFrame([row] * len(returns), index=returns.index)


def test_zscore_untraded_names_stay_at_zero():
    returns = make_returns()
    signals = make_signals(returns)
    strat = ZScoreStrategy(quantile=0.2)

    weights = strat.generate_target_weights(
        signals=signals,
        cluster_labels=CLUSTERS,
        tokens=TOKENS,
        returns=returns,
    )

    # quantile=0.2 with 3 names/cluster trades 1 long + 1 short per cluster;
    # the middle name must hold exactly zero weight on every date
    assert (weights['bbb_returns'] == 0.0).all()
    assert (weights['eee_returns'] == 0.0).all()
    # traded names must actually hold positions
    assert (weights['aaa_returns'].iloc[-1] != 0.0)


def test_zscore_traded_weights_dollar_neutral():
    returns = make_returns()
    signals = make_signals(returns)
    strat = ZScoreStrategy(quantile=0.2)

    weights = strat.generate_target_weights(
        signals=signals,
        cluster_labels=CLUSTERS,
        tokens=TOKENS,
        returns=returns,
    )

    net = weights.sum(axis=1)
    assert np.allclose(net.iloc[15:], 0.0, atol=1e-12)


def test_zscore_vol_weights_are_lagged():
    returns = make_returns()
    signals = make_signals(returns)
    strat = ZScoreStrategy(quantile=0.2)

    weights_base = strat.generate_target_weights(
        signals=signals, cluster_labels=CLUSTERS, tokens=TOKENS, returns=returns,
    )

    # Perturb ONLY the final day's returns; with properly lagged vol weights
    # the final day's positions cannot change
    bumped = returns.copy()
    bumped.iloc[-1] = bumped.iloc[-1] + 0.10
    weights_bumped = strat.generate_target_weights(
        signals=signals, cluster_labels=CLUSTERS, tokens=TOKENS, returns=bumped,
    )

    last = returns.index[-1]
    pd.testing.assert_series_equal(weights_base.loc[last], weights_bumped.loc[last])


def test_cluster_deviation_untraded_names_stay_at_zero():
    returns = make_returns()
    strat = ClusterDeviationStrategy(entry_threshold=1.5)
    signals = make_signals(returns)

    weights = strat.generate_target_weights(
        signals=signals,
        cluster_labels=CLUSTERS,
        tokens=TOKENS,
    )

    # |z|=0 names never cross the 1.5 threshold and must stay flat
    assert (weights['bbb_returns'] == 0.0).all()
    assert (weights['eee_returns'] == 0.0).all()
    # selected names trade, and the book is dollar-neutral
    assert (weights['aaa_returns'] > 0.0).all()
    assert np.allclose(weights.sum(axis=1), 0.0, atol=1e-12)
