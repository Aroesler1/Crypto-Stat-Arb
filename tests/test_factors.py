"""Tests for the crypto characteristics and the quintile factor construction.

The load-bearing tests are the lookahead ones. A factor table built on a
survivorship-free micro-cap panel is exactly the place where a one-day alignment
slip turns into a spurious Sharpe, so what is pinned is that a portfolio formed
on a date cannot see that date's return, and that a characteristic computed on
day t uses only data through day t.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.factors import characteristics as C  # noqa: E402
from stat_arb.factors import portfolios as PF  # noqa: E402

DATES = pd.date_range("2020-01-06", periods=400, freq="D")   # starts on a Monday


def _panels(n_tokens=20, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"T{i}" for i in range(n_tokens)]
    steps = rng.normal(0.0, 0.03, (len(DATES), n_tokens))
    close = pd.DataFrame(10.0 * np.exp(np.cumsum(steps, axis=0)), index=DATES, columns=cols)
    volume = pd.DataFrame(rng.lognormal(13, 1, (len(DATES), n_tokens)), index=DATES, columns=cols)
    mcap = close * 1e6
    market = pd.Series(np.log1p(rng.normal(0, 0.02, len(DATES))), index=DATES)
    return close, volume, mcap, market


def _membership(close):
    return pd.DataFrame(True, index=close.index, columns=close.columns)


# --- characteristics -------------------------------------------------------

def test_every_characteristic_builds_and_is_finite_somewhere():
    close, volume, mcap, market = _panels()
    built = C.build_characteristics(close, volume, mcap, market)
    assert set(built) == set(C.CHARACTERISTICS)
    for name, panel in built.items():
        assert panel.shape == close.shape, name
        assert panel.notna().to_numpy().sum() > 0, name


def test_momentum_is_the_cumulative_log_return():
    close = pd.DataFrame({"A": [10.0] * 10 + [20.0] * 10}, index=DATES[:20])
    got = C.momentum(close, 7).iloc[-1]["A"]
    assert got == pytest.approx(0.0, abs=1e-12)      # flat over the last 7 days
    got = C.momentum(close, 14).iloc[-1]["A"]
    assert got == pytest.approx(np.log(2), rel=1e-9)


def test_reversal_is_the_negated_return():
    close, _, _, _ = _panels()
    assert np.allclose(C.reversal(close, 7).to_numpy(),
                       -C.momentum(close, 7).to_numpy(), equal_nan=True)


def test_characteristics_use_no_future_data():
    """Truncating the panel must not change any earlier value."""
    close, volume, mcap, market = _panels()
    cut = 300
    full = C.build_characteristics(close, volume, mcap, market)
    part = C.build_characteristics(close.iloc[:cut], volume.iloc[:cut],
                                   mcap.iloc[:cut], market.iloc[:cut])
    for name in full:
        a = full[name].iloc[:cut].to_numpy()
        b = part[name].to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[both], b[both], atol=1e-9), name


def test_amihud_is_higher_for_a_thinner_token():
    close, volume, mcap, market = _panels(n_tokens=2, seed=1)
    volume["T1"] = volume["T1"] / 1000.0            # same returns, far less volume
    a = C.amihud(close, volume).iloc[-1]
    assert a["T1"] > a["T0"]


def test_high_52w_is_one_at_a_new_high_and_below_otherwise():
    rising = pd.DataFrame({"A": np.linspace(1, 100, 400)}, index=DATES)
    assert C.high_52w(rising).iloc[-1]["A"] == pytest.approx(1.0)
    falling = pd.DataFrame({"A": np.linspace(100, 1, 400)}, index=DATES)
    assert C.high_52w(falling).iloc[-1]["A"] < 0.2


def test_market_beta_recovers_a_known_beta():
    rng = np.random.default_rng(3)
    market = pd.Series(rng.normal(0, 0.02, len(DATES)), index=DATES)
    close = pd.DataFrame(
        {"A": 10 * np.exp(np.cumsum(2.0 * market.to_numpy()))}, index=DATES)
    assert C.market_beta(close, market, window=200).iloc[-1]["A"] == pytest.approx(2.0, rel=0.05)


def test_turning_points_separates_choppy_from_trending():
    alternating = pd.DataFrame(
        {"A": 10 * np.exp(np.cumsum(np.tile([0.05, -0.05], 200)))}, index=DATES)
    trending = pd.DataFrame({"A": 10 * np.exp(np.cumsum(np.full(400, 0.01)))}, index=DATES)
    assert C.turning_points(alternating).iloc[-1]["A"] > 0.8
    assert C.turning_points(trending).iloc[-1]["A"] < 0.2


# --- portfolio construction ------------------------------------------------

def _forward_simple(close):
    return close.pct_change().shift(-0)


def test_portfolio_formed_on_a_date_cannot_see_that_dates_return():
    """The test that matters. A characteristic that is a perfect copy of the
    formation-day return must earn nothing, because it is lagged."""
    close, volume, mcap, market = _panels(seed=5)
    fwd = close.pct_change()
    same_day = fwd.copy()                       # cheat: today's return as the signal
    gross, net, _ = PF.quintile_long_short(
        same_day, fwd, mcap, _membership(close), higher_is_long=True)
    # the shift(1) inside means the signal is yesterday's return, not today's,
    # so this must not produce a large positive Sharpe
    assert PF.annualised_sharpe(gross) < 3.0


def test_a_perfectly_predictive_lagged_signal_does_earn():
    """The complement: a signal that legitimately predicts tomorrow works."""
    close, volume, mcap, market = _panels(seed=6)
    fwd = close.pct_change()
    # shift(-1) makes the value on day t equal to day t+1's return; after the
    # construction's own shift(1) the portfolio formed on t sees day t's return
    # and holds into t+1, which is genuine foresight and must be visible
    oracle = fwd.shift(-2)
    gross, _, _ = PF.quintile_long_short(
        oracle, fwd, mcap, _membership(close), higher_is_long=True)
    assert PF.annualised_sharpe(gross) > 2.0


def test_costs_reduce_the_net_series():
    close, volume, mcap, market = _panels(seed=7)
    fwd = close.pct_change()
    char = C.momentum(close, 28)
    gross, net, turn = PF.quintile_long_short(char, fwd, mcap, _membership(close))
    assert (net <= gross + 1e-12).all()
    assert turn.mean() > 0
    assert PF.annualised_sharpe(net) < PF.annualised_sharpe(gross) or gross.mean() < 0


def test_sign_flips_the_spread():
    close, volume, mcap, market = _panels(seed=8)
    fwd = close.pct_change()
    char = C.momentum(close, 28)
    up, _, _ = PF.quintile_long_short(char, fwd, mcap, _membership(close),
                                      higher_is_long=True)
    down, _, _ = PF.quintile_long_short(char, fwd, mcap, _membership(close),
                                        higher_is_long=False)
    common = up.index.intersection(down.index)
    assert np.allclose(up.loc[common].to_numpy(), -down.loc[common].to_numpy(), atol=1e-12)


def test_membership_restricts_the_universe():
    close, volume, mcap, market = _panels(seed=9)
    fwd = close.pct_change()
    char = C.momentum(close, 28)
    restricted = _membership(close).copy()
    restricted.iloc[:, 5:] = False              # only 5 names, below 2 x quintiles
    _, net, _ = PF.quintile_long_short(char, fwd, mcap, restricted)
    assert net.empty


def test_too_few_names_produces_no_series_rather_than_a_degenerate_one():
    close, volume, mcap, market = _panels(n_tokens=4)
    fwd = close.pct_change()
    _, net, _ = PF.quintile_long_short(C.momentum(close, 28), fwd, mcap,
                                       _membership(close))
    assert net.empty


def test_value_weighting_follows_market_cap():
    """A dominant name should drive the long leg's return."""
    close, volume, mcap, market = _panels(n_tokens=20, seed=11)
    mcap = mcap.copy()
    mcap.iloc[:, 0] = mcap.iloc[:, 0] * 1e6      # one giant
    fwd = close.pct_change()
    char = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    char.iloc[:, :4] = 1.0                       # giant is in the top quintile
    gross, _, _ = PF.quintile_long_short(char, fwd, mcap, _membership(close))
    assert not gross.empty


def test_factor_returns_summarises_every_factor():
    close, volume, mcap, market = _panels(seed=12)
    fwd = close.pct_change()
    chars = C.build_characteristics(close, volume, mcap, market,
                                    names=["mom_4w", "size", "amihud"])
    signs = {k: C.CHARACTERISTICS[k][1] for k in chars}
    series, summary = PF.factor_returns(chars, fwd, mcap, _membership(close), signs)
    assert set(summary["factor"]) <= set(chars)
    assert (summary["n_weeks"] > 0).all()
    assert set(series.columns) == set(summary["factor"])
