"""Tests for backtest engine carry costs and PCA NaN robustness."""
import numpy as np
import pandas as pd

from stat_arb.backtest.engine import BacktestEngine
from stat_arb.pca.market_mode import MarketModeExtractor


def _panel(n_days=30, n_assets=4, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2024-01-01', periods=n_days, freq='D')
    cols = [f'tok{i}_returns' for i in range(n_assets)]
    return pd.DataFrame(rng.normal(0, 0.02, size=(n_days, n_assets)), index=idx, columns=cols)


def test_carry_cost_applied_to_gross_exposure():
    returns = _panel()
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    weights.iloc[:, 0] = 0.75
    weights.iloc[:, 1] = -0.75  # gross exposure 1.5

    engine = BacktestEngine(cost_bps=0, carry_bps_daily=10)
    net, gross, costs = engine.compute_net_returns(weights, returns)

    # after the first day (turnover cost 0 given cost_bps=0), daily cost
    # should equal 1.5 * 10bps = 0.0015
    assert np.allclose(costs.iloc[1:], 1.5 * 10 / 10000)
    assert np.allclose(net, gross - costs)


def test_carry_default_zero_keeps_backward_compatibility():
    returns = _panel()
    weights = pd.DataFrame(0.5, index=returns.index, columns=returns.columns)
    engine = BacktestEngine(cost_bps=50)
    net, gross, costs = engine.compute_net_returns(weights, returns)
    turnover = engine.compute_turnover(weights)
    assert np.allclose(costs.fillna(0), (turnover * 50 / 10000).fillna(0))


def test_market_mode_fit_survives_ragged_nans():
    returns = _panel(n_days=60)
    # staggered listing: one asset missing for the first 30 days
    ragged = returns.copy()
    ragged.iloc[:30, 3] = np.nan

    extractor = MarketModeExtractor(n_components=1)
    extractor.fit(ragged)

    # a plain dropna() would have discarded half the sample; the fit must
    # keep all rows and produce finite loadings
    assert np.isfinite(extractor.loadings_).all()
    resid = extractor.residualize(ragged.fillna(0.0))
    assert resid.shape == ragged.shape
