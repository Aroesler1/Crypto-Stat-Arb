"""Tests for PSR / DSR statistics (Bailey & Lopez de Prado)."""
import numpy as np
import pandas as pd

from stat_arb.backtest.statistics import (
    per_period_sharpe,
    probabilistic_sharpe_ratio,
    expected_max_sharpe,
    deflated_sharpe_ratio,
)


def test_psr_of_zero_mean_noise_is_half():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, size=5000))
    psr = probabilistic_sharpe_ratio(r)
    assert 0.35 < psr < 0.65


def test_psr_increases_with_mean():
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.01, size=2000)
    low = probabilistic_sharpe_ratio(pd.Series(noise))
    high = probabilistic_sharpe_ratio(pd.Series(noise + 0.001))
    assert high > low


def test_expected_max_sharpe_monotone_in_trials():
    var = 0.02 ** 2
    vals = [expected_max_sharpe(n, var) for n in (1, 5, 25, 100)]
    assert vals[0] == 0.0
    assert vals == sorted(vals)


def test_dsr_below_psr_under_multiple_trials():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.001, 0.01, size=500))
    psr = probabilistic_sharpe_ratio(r)
    dsr = deflated_sharpe_ratio(r, n_trials=25, var_trial_sr=0.05 ** 2)['dsr']
    assert dsr < psr


def test_dsr_uses_trial_sharpes_for_variance():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.001, 0.01, size=500))
    trial_srs = [0.01, -0.02, 0.05, 0.00, 0.03]
    out = deflated_sharpe_ratio(r, n_trials=len(trial_srs), trial_sharpes=trial_srs)
    expected_var = float(np.var(trial_srs, ddof=1))
    assert abs(out['sr_benchmark'] - expected_max_sharpe(5, expected_var)) < 1e-12
    assert out['sr_observed'] == per_period_sharpe(r)
