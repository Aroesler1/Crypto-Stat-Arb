"""Tests for the phase-3 execution controls (no-trade band, trade frequency)."""
import numpy as np
import pandas as pd

from stat_arb.run_phase3 import make_execution_portfolio_func


def _signals(n_days=6, cols=("a", "b")):
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D")
    data = {
        "a": [0.10, 0.11, 0.12, 0.30, 0.30, 0.30],
        "b": [-0.10, -0.11, -0.12, -0.30, -0.30, -0.30],
    }
    return pd.DataFrame(data, index=idx)


def test_no_trade_band_holds_sub_band_moves():
    holder = [pd.Series({"a": 0.10, "b": -0.10})]
    func = make_execution_portfolio_func(holder, weight_band=0.05, trade_frequency_days=1,
                                         max_turnover_per_day=10.0)
    weights = func(_signals(), returns=None)

    # days 0-2 move by 0.00-0.02 < band: held at prior weights
    assert np.allclose(weights.iloc[0], [0.10, -0.10])
    assert np.allclose(weights.iloc[1], [0.10, -0.10])
    assert np.allclose(weights.iloc[2], [0.10, -0.10])
    # day 3 moves by 0.20 > band: trades to target
    assert np.allclose(weights.iloc[3], [0.30, -0.30])


def test_trade_frequency_holds_between_rebalances():
    holder = [None]
    func = make_execution_portfolio_func(holder, weight_band=0.0, trade_frequency_days=3,
                                         max_turnover_per_day=10.0)
    weights = func(_signals(), returns=None)

    # trades on day 0 and day 3 only; days 1-2 and 4-5 hold
    assert np.allclose(weights.iloc[1], weights.iloc[0])
    assert np.allclose(weights.iloc[2], weights.iloc[0])
    assert np.allclose(weights.iloc[3], [0.30, -0.30])
    assert np.allclose(weights.iloc[4], weights.iloc[3])


def test_turnover_cap_still_applies():
    holder = [None]
    func = make_execution_portfolio_func(holder, weight_band=0.0, trade_frequency_days=1,
                                         max_turnover_per_day=0.10)
    weights = func(_signals(), returns=None)

    turnover_day0 = weights.iloc[0].abs().sum()  # from flat
    assert turnover_day0 <= 0.10 + 1e-12


def test_state_carries_across_folds():
    holder = [None]
    func = make_execution_portfolio_func(holder, weight_band=0.0, trade_frequency_days=1,
                                         max_turnover_per_day=10.0)
    first = func(_signals(), returns=None)
    assert holder[0] is not None
    pd.testing.assert_series_equal(holder[0], first.iloc[-1])
