"""Tests for perpetual funding and shortability.

No network. The fetch layer is exercised against stubbed responses so the
suite stays offline and deterministic; what is pinned is the parsing, the
pagination stop conditions, and the shortability accounting.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.data import funding as F  # noqa: E402


def test_shortability_partitions_the_universe(monkeypatch):
    monkeypatch.setattr(F, "list_perp_symbols", lambda: {"BTC", "ETH", "SNX"})
    rep = F.assess_shortability(["btc", "SNX", "SomeAltcoin", "eth"])
    assert set(rep.shortable) == {"BTC", "SNX", "ETH"}
    assert rep.missing == ["SOMEALTCOIN"]
    assert rep.coverage == pytest.approx(0.75)


def test_shortability_coverage_is_the_headline_number(monkeypatch):
    """A universe with almost no perps must report near-zero coverage.

    This is the constraint that decides whether a dollar-neutral book is
    implementable at all, so it must not be silently rounded away.
    """
    monkeypatch.setattr(F, "list_perp_symbols", lambda: {"BTC"})
    rep = F.assess_shortability([f"ALT{i}" for i in range(99)] + ["BTC"])
    assert rep.coverage == pytest.approx(0.01)


def _page(times, rate=0.0001):
    return [{"coin": "X", "fundingRate": str(rate), "premium": "0", "time": t} for t in times]


def test_fetch_stops_on_a_short_page(monkeypatch):
    """A page below the cap means the history is exhausted; stop, do not loop."""
    calls = {"n": 0}

    def fake_post(payload, timeout=25):
        calls["n"] += 1
        return _page([1685491200000 + i * F._HOURS_MS for i in range(3)])

    monkeypatch.setattr(F, "_post", fake_post)
    out = F.fetch_funding("X", 1685491200000, 1685491200000 + 10 * F._HOURS_MS, pause=0)
    assert calls["n"] == 1
    assert len(out) == 3
    assert list(out.columns) == ["time", "funding_rate"]


def test_fetch_stops_at_the_end_time(monkeypatch):
    start = 1685491200000
    full = _page([start + i * F._HOURS_MS for i in range(F._MAX_ROWS)])
    monkeypatch.setattr(F, "_post", lambda payload, timeout=25: full)
    out = F.fetch_funding("X", start, start + 5 * F._HOURS_MS, pause=0)
    # rows beyond the requested end are filtered out
    assert out["time"].max() <= pd.Timestamp(start + 5 * F._HOURS_MS, unit="ms")


def test_fetch_survives_a_network_failure(monkeypatch):
    def boom(payload, timeout=25):
        raise TimeoutError("network")

    monkeypatch.setattr(F, "_post", boom)
    out = F.fetch_funding("X", 0, 10, pause=0)
    assert out.empty and list(out.columns) == ["time", "funding_rate"]


def test_summarise_annualises_from_daily_rates():
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    panel = pd.DataFrame({"A": [0.001] * 100, "B": [-0.0005] * 100}, index=idx)
    s = F.summarise(panel)
    assert s.loc["A", "annualised_pct"] == pytest.approx(0.001 * 365 * 100)
    assert s.loc["B", "annualised_pct"] < 0
    # sorted best-to-worst so the most expensive short is visible first
    assert s.index[0] == "A"
