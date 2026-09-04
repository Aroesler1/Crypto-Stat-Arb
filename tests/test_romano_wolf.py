"""Tests for the Romano-Wolf stepwise multiple test.

Ported from the VOO repo, so what is pinned here is that the port behaves: it
finds real alpha, it does not find alpha in noise, it controls the family-wise
error rate near the nominal level, and the two-sided variant added for the
survivorship factor table works in both directions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.backtest.statistics import romano_wolf_stepdown  # noqa: E402


def _family(n=500, n_noise=12, alpha_mean=0.004, sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    cols = {"real_alpha": rng.normal(alpha_mean, sd, n)}
    cols.update({f"noise{i}": rng.normal(0.0, sd, n) for i in range(n_noise)})
    return pd.DataFrame(cols, index=pd.date_range("2020-01-01", periods=n))


def test_finds_real_alpha_and_rejects_the_noise():
    out = romano_wolf_stepdown(_family(), n_boot=400)
    sig = set(out.loc[out["significant"], "strategy"])
    assert sig == {"real_alpha"}


def test_pure_noise_family_yields_no_rejections():
    rng = np.random.default_rng(1)
    noise = pd.DataFrame({f"s{i}": rng.normal(0, 0.01, 400) for i in range(10)},
                         index=pd.date_range("2020-01-01", periods=400))
    out = romano_wolf_stepdown(noise, n_boot=400)
    assert not out["significant"].any()


def test_family_wise_error_rate_is_near_nominal():
    """The property the procedure exists for: at most ~5% of pure-noise
    families should produce any rejection at all."""
    false_positive_families = 0
    trials = 40
    for seed in range(trials):
        rng = np.random.default_rng(1000 + seed)
        noise = pd.DataFrame({f"s{i}": rng.normal(0, 0.01, 300) for i in range(8)},
                             index=pd.date_range("2020-01-01", periods=300))
        out = romano_wolf_stepdown(noise, n_boot=250, seed=seed)
        false_positive_families += bool(out["significant"].any())
    # 5% nominal; allow sampling slack over 40 trials
    assert false_positive_families / trials <= 0.20


def test_more_alpha_is_rejected_at_an_earlier_step():
    rng = np.random.default_rng(2)
    n = 600
    df = pd.DataFrame({
        "strong": rng.normal(0.006, 0.01, n),
        "weak": rng.normal(0.0025, 0.01, n),
        **{f"noise{i}": rng.normal(0, 0.01, n) for i in range(6)},
    }, index=pd.date_range("2020-01-01", periods=n))
    out = romano_wolf_stepdown(df, n_boot=500).set_index("strategy")
    assert out.loc["strong", "significant"]
    assert out.loc["strong", "rejected_at_step"] <= out.loc["weak", "rejected_at_step"] \
        or not out.loc["weak", "significant"]


def test_adjusted_p_values_are_probabilities():
    out = romano_wolf_stepdown(_family(), n_boot=300)
    p = out["adjusted_p"].dropna()
    assert ((p >= 0) & (p <= 1)).all()


def test_one_sided_test_ignores_a_large_negative_mean():
    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame({
        "very_negative": rng.normal(-0.006, 0.01, n),
        **{f"noise{i}": rng.normal(0, 0.01, n) for i in range(6)},
    }, index=pd.date_range("2020-01-01", periods=n))
    out = romano_wolf_stepdown(df, n_boot=400, two_sided=False).set_index("strategy")
    assert not out.loc["very_negative", "significant"]


def test_two_sided_test_finds_it():
    """What the survivorship table needs: a difference either way is a finding."""
    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame({
        "very_negative": rng.normal(-0.006, 0.01, n),
        **{f"noise{i}": rng.normal(0, 0.01, n) for i in range(6)},
    }, index=pd.date_range("2020-01-01", periods=n))
    out = romano_wolf_stepdown(df, n_boot=400, two_sided=True).set_index("strategy")
    assert out.loc["very_negative", "significant"]
    assert out.loc["very_negative", "t_stat"] < 0


def test_serial_dependence_does_not_produce_spurious_rejections():
    """The block bootstrap is there for this; an i.i.d. resample would
    understate the variance of the maximum and over-reject."""
    rng = np.random.default_rng(4)
    n = 600
    cols = {}
    for i in range(8):
        e = rng.normal(0, 0.01, n)
        ar = np.zeros(n)
        for t in range(1, n):
            ar[t] = 0.6 * ar[t - 1] + e[t]      # strongly autocorrelated, zero mean
        cols[f"s{i}"] = ar
    df = pd.DataFrame(cols, index=pd.date_range("2020-01-01", periods=n))
    out = romano_wolf_stepdown(df, n_boot=400)
    assert not out["significant"].any()


def test_is_deterministic_for_a_fixed_seed():
    a = romano_wolf_stepdown(_family(), n_boot=200, seed=7)
    b = romano_wolf_stepdown(_family(), n_boot=200, seed=7)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"a": np.arange(10.0)}),                        # too few rows
    pd.DataFrame({"a": np.arange(100.0)}),                       # one strategy
])
def test_degenerate_input_raises_rather_than_returning_nonsense(frame):
    with pytest.raises(ValueError):
        romano_wolf_stepdown(frame)
