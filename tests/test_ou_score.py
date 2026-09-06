"""Tests for the Avellaneda-Lee s-score and the other Step 3 signal extensions.

The load-bearing test is parameter recovery: an OU process simulated with a
known mean-reversion speed must come back with that speed, because every
downstream decision (the s-score scale, the half-life filter, the position
size) is a function of it. The rest pin the conditions the paper imposes and
this implementation must not quietly skip.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.signals.ou_score import (  # noqa: E402
    BetaAdjustedDeviation, ClusterMomentumOverlay, OUScoreStrategy,
    ewma_zscore, half_life_position_scale, ou_parameters, s_scores,
)


def simulate_ou(kappa, m, sigma, n=4000, seed=0):
    """Exact discretisation of dX = kappa (m - X) dt + sigma dW, dt = 1."""
    rng = np.random.default_rng(seed)
    b = np.exp(-kappa)
    # stationary innovation std so the simulated series matches the AR(1) fit
    sd = sigma * np.sqrt((1 - b ** 2) / (2 * kappa))
    x = np.empty(n)
    x[0] = m
    for t in range(1, n):
        x[t] = m + b * (x[t - 1] - m) + rng.normal(0, sd)
    return x


# --- parameter recovery ----------------------------------------------------

@pytest.mark.parametrize("kappa", [0.10, 0.25, 0.50])
def test_ou_fit_recovers_the_simulated_mean_reversion_speed(kappa):
    x = simulate_ou(kappa, m=0.0, sigma=1.0, n=6000, seed=1)
    got, m, sigma_eq = ou_parameters(x)
    assert got == pytest.approx(kappa, rel=0.12)
    assert m == pytest.approx(0.0, abs=0.12)
    assert sigma_eq > 0


def test_ou_fit_recovers_a_nonzero_long_run_mean():
    x = simulate_ou(0.3, m=2.5, sigma=1.0, n=6000, seed=2)
    _, m, _ = ou_parameters(x)
    assert m == pytest.approx(2.5, rel=0.10)


def test_half_life_matches_the_recovered_speed():
    kappa_true = 0.35
    x = simulate_ou(kappa_true, m=0.0, sigma=1.0, n=6000, seed=3)
    kappa, _, _ = ou_parameters(x)
    assert np.log(2) / kappa == pytest.approx(np.log(2) / kappa_true, rel=0.15)


# --- the conditions the paper imposes --------------------------------------

def test_a_random_walk_is_not_tradable():
    """A random walk has no reversion to harvest.

    In a finite sample the AR(1) coefficient of a walk comes back just under 1
    rather than exactly 1, so the fit itself does not always reject it. The
    protection that matters is the half-life condition: the implied half-life
    runs to hundreds of days, far beyond any holding horizon, so the filter
    drops it. Avellaneda and Lee impose the same condition as a floor on kappa.
    """
    rng = np.random.default_rng(4)
    walk = np.cumsum(rng.normal(size=3000))
    kappa, _, _ = ou_parameters(walk)
    assert np.isnan(kappa) or np.log(2) / kappa > 50.0


def test_an_alternating_series_is_rejected():
    """b <= 0 is alternation, not reversion."""
    x = np.array([1.0, -1.0] * 500) + np.random.default_rng(5).normal(0, 0.01, 1000)
    assert np.isnan(ou_parameters(x)[0])


def test_a_constant_series_is_rejected():
    assert np.isnan(ou_parameters(np.ones(500))[0])


def test_too_few_observations_is_rejected():
    assert np.isnan(ou_parameters(np.array([1.0, 2.0, 3.0]))[0])


# --- the half-life filter --------------------------------------------------

def _panel(kappa, n=400, cols=3, seed=0):
    data = {f"T{i}_returns": np.diff(simulate_ou(kappa, 0.0, 1.0, n + 1, seed + i))
            for i in range(cols)}
    return pd.DataFrame(data, index=pd.date_range("2020-01-01", periods=n))


def test_slow_reverting_tokens_are_dropped_by_the_half_life_filter():
    """A residual that takes weeks to revert cannot be held for three days."""
    slow = _panel(kappa=0.02, n=500)          # half-life ~35 days
    kept, _ = s_scores(slow, window=60, max_half_life=3.0)
    dropped, half_lives = s_scores(slow, window=60, max_half_life=None)
    assert kept.notna().to_numpy().sum() < dropped.notna().to_numpy().sum()
    assert half_lives.stack().median() > 3.0


def test_fast_reverting_tokens_survive_the_filter():
    fast = _panel(kappa=0.60, n=500)           # half-life ~1.2 days
    kept, half_lives = s_scores(fast, window=60, max_half_life=3.0)
    assert kept.notna().to_numpy().sum() > 0
    assert half_lives.stack().median() < 3.0


def test_s_scores_are_finite_and_centred():
    fast = _panel(kappa=0.5, n=600)
    s, _ = s_scores(fast, window=60, max_half_life=None)
    vals = s.stack().dropna()
    assert np.isfinite(vals).all()
    assert abs(float(vals.mean())) < 0.5       # standardised deviation, not a level


# --- strategy wiring -------------------------------------------------------

def _clustered(n_per=6, k=2, n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n)
    cols, labels, tokens = {}, [], []
    for c in range(k):
        common = rng.normal(0, 0.02, n)
        for j in range(n_per):
            name = f"C{c}T{j}"
            cols[f"{name}_returns"] = common + rng.normal(0, 0.01, n)
            labels.append(c)
            tokens.append(name)
    return pd.DataFrame(cols, index=dates), np.array(labels), tokens


def test_ou_strategy_produces_lagged_signals_only():
    returns, labels, tokens = _clustered()
    s = OUScoreStrategy(window=60).compute_signals(returns, labels, tokens, lag=1)
    assert s.index.equals(returns.index)
    assert s.iloc[0].isna().all()             # nothing formed on the first day


def test_ou_weights_are_dollar_neutral_within_each_cluster():
    returns, labels, tokens = _clustered()
    strat = OUScoreStrategy(window=60, entry_threshold=0.0)
    s = strat.compute_signals(returns, labels, tokens)
    w = strat.generate_target_weights(s, labels, tokens, returns)
    traded = w.loc[w.abs().sum(axis=1) > 0]
    assert len(traded) > 0
    # neutral overall, because every cluster leg is neutral on its own
    assert traded.sum(axis=1).abs().max() < 1e-8
    # each traded cluster contributes gross 1.0; the runner rescales the book
    # to its leverage target afterwards
    assert traded.abs().sum(axis=1).max() == pytest.approx(2.0, abs=1e-8)
    for cluster in (0, 1):
        cols = [c for c, lab in zip(w.columns, labels) if lab == cluster]
        assert traded[cols].sum(axis=1).abs().max() < 1e-8


def test_ou_weights_are_long_the_underperformer():
    """A negative s-score is a token below its cluster, so it is bought."""
    returns, labels, tokens = _clustered()
    strat = OUScoreStrategy(window=60, entry_threshold=0.0)
    s = strat.compute_signals(returns, labels, tokens)
    w = strat.generate_target_weights(s, labels, tokens)
    day = w.abs().sum(axis=1).idxmax()
    aligned = pd.concat([s.loc[day].rename("s"), w.loc[day].rename("w")], axis=1).dropna()
    aligned = aligned[aligned["w"] != 0]
    assert (np.sign(aligned["s"]) != np.sign(aligned["w"])).all()


def test_entry_threshold_suppresses_small_deviations():
    returns, labels, tokens = _clustered()
    strat = OUScoreStrategy(window=60)
    s = strat.compute_signals(returns, labels, tokens)
    loose = OUScoreStrategy(window=60, entry_threshold=0.0).generate_target_weights(
        s, labels, tokens)
    tight = OUScoreStrategy(window=60, entry_threshold=3.0).generate_target_weights(
        s, labels, tokens)
    assert (tight.abs().sum().sum()) < (loose.abs().sum().sum())


# --- beta adjustment -------------------------------------------------------

def test_beta_adjustment_estimates_a_high_beta_member_as_high_beta():
    rng = np.random.default_rng(7)
    n = 400
    dates = pd.date_range("2020-01-01", periods=n)
    common = rng.normal(0, 0.02, n)
    returns = pd.DataFrame({
        "A_returns": 2.0 * common + rng.normal(0, 0.001, n),
        "B_returns": 1.0 * common + rng.normal(0, 0.001, n),
        "C_returns": 0.5 * common + rng.normal(0, 0.001, n),
    }, index=dates)
    labels, tokens = np.array([0, 0, 0]), ["A", "B", "C"]
    strat = BetaAdjustedDeviation(beta_window=100)
    strat.compute_signals(returns, labels, tokens)
    final = strat.betas_.dropna().iloc[-1]
    assert final["A_returns"] > final["B_returns"] > final["C_returns"]


def test_betas_use_only_past_data():
    returns, labels, tokens = _clustered()
    strat = BetaAdjustedDeviation(beta_window=60, min_periods=20)
    strat.compute_signals(returns, labels, tokens)
    # shifted by one, so the first estimable row is min_periods, not min_periods-1
    assert strat.betas_.iloc[:20].isna().all().all()


# --- EWMA and half-life sizing ---------------------------------------------

def test_ewma_tracks_a_level_shift_faster_than_a_rolling_window():
    """The property that motivates using it: no 60-day memory of a stale level."""
    n = 300
    x = pd.DataFrame({"a": np.r_[np.zeros(n // 2), np.ones(n // 2) * 5.0]},
                     index=pd.date_range("2020-01-01", periods=n))
    at = n // 2 + 10
    ewma_mean = x.ewm(halflife=5, min_periods=5).mean().iloc[at]["a"]
    rolling_mean = x.rolling(60).mean().iloc[at]["a"]
    assert ewma_mean > rolling_mean          # closer to the new level of 5
    # after 11 days at halflife 5 the EWMA has closed ~78% of the gap to 5.0
    assert ewma_mean > 3.5 and rolling_mean < 1.5


def test_ewma_zscore_is_standardised_on_stationary_input():
    rng = np.random.default_rng(11)
    x = pd.DataFrame({"a": rng.normal(0, 1, 2000)},
                     index=pd.date_range("2020-01-01", periods=2000))
    z = ewma_zscore(x, halflife=20).dropna()
    assert abs(float(z["a"].mean())) < 0.15
    assert float(z["a"].std()) == pytest.approx(1.0, rel=0.25)


def test_position_scale_shrinks_with_a_longer_half_life():
    hl = pd.DataFrame({"a": [1.0, 3.0, 10.0, 30.0]})
    scale = half_life_position_scale(hl, target=3.0, floor=0.25, cap=1.0)
    assert scale["a"].is_monotonic_decreasing
    assert scale["a"].iloc[0] == pytest.approx(1.0)     # capped
    assert scale["a"].iloc[-1] == pytest.approx(0.25)   # floored


def test_position_scale_handles_missing_half_lives():
    hl = pd.DataFrame({"a": [np.nan, 0.0]})
    scale = half_life_position_scale(hl)
    assert (scale["a"] == 0.25).all()


# --- cluster momentum overlay ----------------------------------------------

def test_overlay_selects_the_trailing_winners():
    n = 200
    dates = pd.date_range("2020-01-01", periods=n)
    returns = pd.DataFrame({
        "W1_returns": np.full(n, 0.01), "W2_returns": np.full(n, 0.01),
        "L1_returns": np.full(n, -0.01), "L2_returns": np.full(n, -0.01),
    }, index=dates)
    labels, tokens = np.array([0, 0, 1, 1]), ["W1", "W2", "L1", "L2"]
    overlay = ClusterMomentumOverlay(momentum_window=28, top_frac=0.5)
    assert overlay.clusters_to_trade(returns, labels, tokens, dates[-1]) == [0]


def test_overlay_ranking_cannot_see_the_date_it_selects_for():
    n = 200
    dates = pd.date_range("2020-01-01", periods=n)
    # cluster 0 loses for the whole history and wins only on the final day
    r0 = np.r_[np.full(n - 1, -0.01), 10.0]
    returns = pd.DataFrame({
        "A_returns": r0, "B_returns": r0,
        "C_returns": np.full(n, 0.001), "D_returns": np.full(n, 0.001),
    }, index=dates)
    labels, tokens = np.array([0, 0, 1, 1]), ["A", "B", "C", "D"]
    overlay = ClusterMomentumOverlay(momentum_window=28, top_frac=0.5)
    # the final day's spike must not select cluster 0
    assert overlay.clusters_to_trade(returns, labels, tokens, dates[-1]) == [1]


def test_between_cluster_weights_are_dollar_neutral():
    returns, labels, tokens = _clustered(k=3)
    w = ClusterMomentumOverlay(momentum_window=20).between_cluster_weights(
        returns, labels, tokens)
    traded = w.loc[w.abs().sum(axis=1) > 0]
    assert len(traded) > 0
    assert traded.sum(axis=1).abs().max() < 1e-8
