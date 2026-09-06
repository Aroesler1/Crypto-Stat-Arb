"""Tests for the funding application and its coverage accounting.

The after-funding Sharpe is the number the verdict rests on, so what is pinned
is the sign convention (a short EARNS positive funding), the lag (funding
accrues on the position held, not the one being traded into), and the coverage
metric, whose naive form understates how much of the cost is actually measured.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.run_tradability import apply_funding, funding_coverage  # noqa: E402

DATES = pd.date_range("2021-01-01", periods=5, freq="D")


def _weights(values):
    return pd.DataFrame(values, index=DATES, columns=["A_returns", "B_returns"])


def _funding(values):
    return pd.DataFrame(values, index=DATES, columns=["A_returns", "B_returns"])


# --- sign convention -------------------------------------------------------

def test_a_short_earns_positive_funding():
    """Positive funding means longs pay shorts."""
    w = _weights([[-1.0, 0.0]] * 5)
    f = _funding([[0.01, 0.0]] * 5)
    pnl = apply_funding(w, f)
    assert (pnl.iloc[1:] > 0).all()
    assert pnl.iloc[1] == pytest.approx(0.01)


def test_a_long_pays_positive_funding():
    w = _weights([[1.0, 0.0]] * 5)
    f = _funding([[0.01, 0.0]] * 5)
    assert apply_funding(w, f).iloc[1] == pytest.approx(-0.01)


def test_negative_funding_reverses_both():
    w = _weights([[1.0, -1.0]] * 5)
    f = _funding([[-0.01, -0.01]] * 5)
    pnl = apply_funding(w, f)
    # long earns, short pays; they offset exactly on equal and opposite weights
    assert pnl.iloc[1] == pytest.approx(0.0)


# --- lagging ---------------------------------------------------------------

def test_funding_accrues_on_the_position_already_held():
    """A position opened today does not pay today's funding."""
    w = _weights([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    f = _funding([[0.01, 0.0]] * 5)
    pnl = apply_funding(w, f)
    assert pnl.iloc[1] == pytest.approx(0.0)     # opened today, pays nothing
    assert pnl.iloc[2] == pytest.approx(-0.01)   # held overnight, pays
    assert pnl.iloc[3] == pytest.approx(-0.01)   # still held at yesterday's close
    assert pnl.iloc[4] == pytest.approx(0.0)     # closed


def test_missing_rates_contribute_zero_not_nan():
    w = _weights([[1.0, 1.0]] * 5)
    f = _funding([[0.01, np.nan]] * 5)
    pnl = apply_funding(w, f)
    assert np.isfinite(pnl).all()
    assert pnl.iloc[1] == pytest.approx(-0.01)


# --- coverage --------------------------------------------------------------

def test_exposure_coverage_ignores_columns_the_book_never_holds():
    """The bug the column metric had.

    A bracket carries thousands of return columns, most of which the book never
    holds. Dividing covered columns by all columns reports a coverage far below
    what the traded book actually has.
    """
    cols = [f"T{i}_returns" for i in range(100)]
    w = pd.DataFrame(0.0, index=DATES, columns=cols)
    w[cols[0]] = 1.0                     # the book only ever holds one name
    f = pd.DataFrame(np.nan, index=DATES, columns=cols)
    f[cols[0]] = 0.01                    # and that name has a rate
    by_col, by_exp = funding_coverage(w, f, covered=[cols[0]])
    assert by_col == pytest.approx(0.01)     # 1 of 100 columns
    assert by_exp == pytest.approx(1.0)      # but all of the exposure


def test_exposure_coverage_is_a_weighted_share():
    cols = ["A_returns", "B_returns"]
    w = pd.DataFrame([[3.0, 1.0]] * 5, index=DATES, columns=cols)
    f = pd.DataFrame(np.nan, index=DATES, columns=cols)
    f["A_returns"] = 0.01
    _, by_exp = funding_coverage(w, f, covered=["A_returns"])
    assert by_exp == pytest.approx(0.75)     # 3 of every 4 units of position


def test_coverage_of_a_book_with_no_rates_is_zero():
    cols = ["A_returns", "B_returns"]
    w = pd.DataFrame(1.0, index=DATES, columns=cols)
    f = pd.DataFrame(np.nan, index=DATES, columns=cols)
    by_col, by_exp = funding_coverage(w, f, covered=[])
    assert by_col == 0.0 and by_exp == 0.0


def test_coverage_handles_an_empty_book():
    by_col, by_exp = funding_coverage(pd.DataFrame(), pd.DataFrame(), [])
    assert np.isnan(by_col) and np.isnan(by_exp)


def test_coverage_counts_only_days_the_rate_exists():
    """A name listed halfway through is only covered for the half it exists."""
    cols = ["A_returns"]
    w = pd.DataFrame(1.0, index=DATES, columns=cols)
    f = pd.DataFrame(np.nan, index=DATES, columns=cols)
    f.iloc[3:, 0] = 0.01                 # rate exists for the last two days
    _, by_exp = funding_coverage(w, f, covered=cols)
    # weights are lagged, so day 0 contributes nothing; of days 1-4, two covered
    assert by_exp == pytest.approx(0.5)
