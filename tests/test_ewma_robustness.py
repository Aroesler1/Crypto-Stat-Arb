"""Tests for the EWMA robustness statistics.

These back the strongest claim in the repository, that EWMA standardisation
takes B3 point-in-time from -0.42 to +1.45, so the statistics that qualify it
have to be right. Checked against synthetic series whose answers are known by
construction.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.run_signal_ablation import (  # noqa: E402
    N_BEST_DAYS, best_days_stats, robustness_row, sharpe_by_year,
)

PY = 365


def _series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


# --- Sharpe by year --------------------------------------------------------

def test_by_year_splits_on_the_calendar_and_annualises():
    """A year of constant-mean noise must return its own analytic Sharpe."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.001, 0.01, 366)      # 2020 is a leap year
    b = rng.normal(-0.001, 0.01, 365)
    net = _series(np.r_[a, b], "2020-01-01")
    got = sharpe_by_year(net)
    assert set(got) == {2020, 2021}
    assert got[2020] == pytest.approx(a.mean() / a.std(ddof=1) * np.sqrt(PY), rel=1e-6)
    assert got[2021] == pytest.approx(b.mean() / b.std(ddof=1) * np.sqrt(PY), rel=1e-6)
    assert got[2020] > 0 > got[2021]


def test_by_year_skips_years_with_too_little_data():
    net = _series(np.random.default_rng(1).normal(0, 0.01, 20), "2020-12-20")
    got = sharpe_by_year(net)
    assert 2020 not in got          # only 12 days in 2020
    assert 2021 not in got          # only 8 in 2021


def test_by_year_skips_a_constant_year():
    """Zero variance is undefined, not infinitely good."""
    assert sharpe_by_year(_series(np.zeros(400))) == {}


def test_by_year_ignores_missing_days():
    net = _series(np.r_[np.random.default_rng(2).normal(0.001, 0.01, 200),
                        np.full(100, np.nan)])
    got = sharpe_by_year(net)
    assert np.isfinite(list(got.values())).all()


# --- best-days statistics --------------------------------------------------

def test_best_days_share_is_exact_by_construction():
    """One year of zeros plus ten days of 1.0: the ten days are all the return."""
    values = np.zeros(400)
    values[:N_BEST_DAYS] = 1.0
    stats = best_days_stats(_series(values))
    assert stats["best_days_share"] == pytest.approx(1.0)
    assert stats["sharpe_ex_best"] == pytest.approx(0.0) or np.isnan(stats["sharpe_ex_best"])


def test_best_days_share_is_half_when_the_spikes_are_half():
    rest = np.full(390, 1.0)             # 390 total
    spikes = np.full(N_BEST_DAYS, 39.0)  # 390 total
    stats = best_days_stats(_series(np.r_[rest, spikes]))
    assert stats["best_days_share"] == pytest.approx(0.5)


def test_removing_best_days_lowers_the_sharpe_of_a_spiky_series():
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 0.01, 400)
    values[:N_BEST_DAYS] = 0.5           # a few enormous days carry it
    net = _series(values)
    full = float(net.mean() / net.std(ddof=1) * np.sqrt(PY))
    assert best_days_stats(net)["sharpe_ex_best"] < full


def test_a_broad_based_series_barely_changes_when_best_days_are_removed():
    """The property the EWMA claim rests on."""
    rng = np.random.default_rng(4)
    net = _series(rng.normal(0.0015, 0.01, 3000))
    full = float(net.mean() / net.std(ddof=1) * np.sqrt(PY))
    stats = best_days_stats(net)
    assert abs(stats["sharpe_ex_best"] - full) < 0.35
    assert stats["best_days_share"] < 0.15


def test_best_days_needs_enough_observations():
    stats = best_days_stats(_series(np.random.default_rng(5).normal(0, 1, 20)))
    assert np.isnan(stats["sharpe_ex_best"])
    assert np.isnan(stats["best_days_share"])


def test_best_days_share_handles_a_zero_total_return():
    net = _series(np.r_[np.full(200, 1.0), np.full(200, -1.0)])
    assert np.isnan(best_days_stats(net)["best_days_share"])


# --- the assembled row -----------------------------------------------------

def test_robustness_row_carries_metadata_and_every_year():
    rng = np.random.default_rng(6)
    net = _series(rng.normal(0.001, 0.01, 1200), "2020-01-01")
    row = robustness_row(net, turnover=0.041, bracket="B3", treatment="pit",
                         arm="ewma", halflife=10.0, role="arm")
    assert row["bracket"] == "B3" and row["arm"] == "ewma"
    assert row["halflife"] == 10.0 and row["role"] == "arm"
    assert row["turnover"] == pytest.approx(0.041)
    assert row["n_days"] == 1200
    for year in (2020, 2021, 2022):
        assert f"sharpe_{year}" in row
    assert row["net_sharpe"] == pytest.approx(
        float(net.mean() / net.std(ddof=1) * np.sqrt(PY)), rel=1e-9)


def test_robustness_row_net_sharpe_matches_the_by_year_average_in_sign():
    rng = np.random.default_rng(7)
    net = _series(rng.normal(0.002, 0.01, 1100))
    row = robustness_row(net, 0.05, bracket="B3", treatment="pit", arm="ewma",
                         halflife=10.0, role="arm")
    years = [v for k, v in row.items() if k.startswith("sharpe_") and k != "sharpe_ex_best"]
    assert row["net_sharpe"] > 0
    assert np.mean(years) > 0
