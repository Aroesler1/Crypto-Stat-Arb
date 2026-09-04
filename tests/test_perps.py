"""Tests for perpetual-venue symbol normalisation and the shortability join.

No network: the venue fetchers are stubbed. What is pinned is the normalisation,
because it decides whether a token counts as shortable, and getting it wrong in
either direction moves the tradability verdict this project ends on.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.data import perps as X  # noqa: E402


@pytest.mark.parametrize("raw,base", [
    ("BTCUSDT", "BTC"),                    # Binance USD-M
    ("1000BONKUSDT", "BONK"),              # Binance thousand-multiplier
    ("1000000MOGUSDT", "MOG"),             # Binance million-multiplier
    ("1000BONKUSDC", "BONK"),              # USDC-quoted
    ("kPEPE", "PEPE"),                     # Hyperliquid k-multiplier
    ("BTC-USD", "BTC"),                    # dYdX v4
    ("AAVE_USDC-PERPETUAL", "AAVE"),       # Deribit
    ("BTC-PERPETUAL", "BTC"),              # Deribit inverse
    ("ETH", "ETH"),                        # already a base
    ("eth", "ETH"),                        # case
    ("  SOL  ", "SOL"),                    # whitespace
])
def test_normalisation_reduces_a_contract_to_its_base(raw, base):
    assert X.normalize_base(raw) == base


@pytest.mark.parametrize("raw", ["", None, "   "])
def test_normalisation_handles_missing_symbols(raw):
    assert X.normalize_base(raw) == ""


def test_multiplier_stripping_does_not_eat_a_real_ticker():
    """A short ticker must not be mistaken for a multiplier prefix.

    The regex requires at least two characters to survive, so tickers that
    merely start with a digit run are left alone rather than truncated to
    nothing.
    """
    assert X.normalize_base("1INCH") == "1INCH"
    assert X.normalize_base("1INCHUSDT") == "1INCH"


def test_quote_suffix_is_stripped_once_not_repeatedly():
    # USDUSDT would otherwise strip down past its own base
    assert X.normalize_base("USDUSDT") == "USD"


def _perps():
    return pd.DataFrame([
        {"venue": "binance", "base": "BTC", "status": "ok"},
        {"venue": "binance", "base": "BONK", "status": "ok"},
        {"venue": "hyperliquid", "base": "BTC", "status": "ok"},
        {"venue": "dydx", "base": "SOL", "status": "ok"},
    ])


def test_shortability_reports_every_venue_that_lists_the_token():
    out = X.shortable(["BTC", "SOL", "BONK", "NOPERP"], _perps()).set_index("symbol")
    assert out.loc["BTC", "n_venues"] == 2
    assert out.loc["BTC", "venues"] == "binance,hyperliquid"
    assert out.loc["SOL", "shortable"]
    assert out.loc["BONK", "venues"] == "binance"
    assert not out.loc["NOPERP", "shortable"]
    assert out.loc["NOPERP", "n_venues"] == 0


def test_shortability_matches_through_the_normalised_base():
    """A CMC ticker joins to a multiplier-prefixed contract."""
    out = X.shortable(["BONK"], _perps()).set_index("symbol")
    assert out.loc["BONK", "shortable"]


def test_unreachable_venue_is_recorded_not_silently_empty(monkeypatch):
    def boom():
        raise TimeoutError("geo-blocked")
    monkeypatch.setitem(X._FETCHERS, "binance", boom)
    monkeypatch.setitem(X._FETCHERS, "dydx", lambda: {"SOL"})
    out = X.perp_universe(("binance", "dydx"))
    # the reachable venue contributes; the unreachable one contributes no rows
    assert set(out["venue"]) == {"dydx"}
    assert set(out["base"]) == {"SOL"}


def test_binance_funding_parses_the_static_archive(monkeypatch):
    import io
    import zipfile
    csv = (b"calc_time,funding_interval_hours,last_funding_rate\n"
           b"1704067200000,8,0.00037409\n"
           b"1704096000000,8,0.00027213\n"
           b"1704124800000,8,0.00033601\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("BTCUSDT-fundingRate-2024-01.csv", csv)
    monkeypatch.setattr(X, "_get", lambda url, timeout=30: buf.getvalue())

    df = X.binance_funding_monthly("BTCUSDT", "2024-01")
    assert len(df) == 3
    assert df["time"].iloc[0] == pd.Timestamp("2024-01-01")
    assert df["funding_rate"].iloc[0] == pytest.approx(0.00037409)

    daily = X.binance_funding_daily("BTCUSDT", "2024-01-01", "2024-01-01")
    # three 8-hourly rates on one day sum to the daily cost
    assert daily.iloc[0] == pytest.approx(0.00037409 + 0.00027213 + 0.00033601)


def test_missing_binance_month_is_empty_not_an_error(monkeypatch):
    import urllib.error

    def missing(url, timeout=30):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
    monkeypatch.setattr(X, "_get", missing)
    assert X.binance_funding_monthly("NEWCOINUSDT", "2019-01").empty
