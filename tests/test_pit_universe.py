"""Tests for the point-in-time universe: delisting returns and cmc_id keying.

No network. The fetch layer is exercised against stubbed HTTP responses so the
suite stays offline and deterministic; what is pinned is the two things that
would silently reintroduce the bias this universe exists to remove:

1. a token that dies must actually lose money, and the rule that decided how
   much must be recorded rather than assumed;
2. the panel must be keyed on cmc_id, because CMC reuses symbols and a
   symbol-keyed join splices two different assets into one price series.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.data import cmc_pit as C  # noqa: E402
from stat_arb.data import pit_universe as P  # noqa: E402

WINDOW_START = pd.Timestamp("2024-01-01")
WINDOW_END = pd.Timestamp("2024-03-31")


def _history(dates, closes, volumes):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume_usd": volumes,
        "market_cap_usd": [c * 1000 for c in closes],
    })


def _alive():
    days = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    return _history(days, np.linspace(10.0, 12.0, len(days)), [5e5] * len(days))


def _dies(last_day, last_volume):
    days = pd.date_range(WINDOW_START, last_day, freq="D")
    closes = list(np.linspace(10.0, 4.0, len(days)))
    vols = [5e5] * len(days)
    vols[-1] = last_volume
    return _history(days, closes, vols)


# --------------------------------------------------------------------------
# the delisting-return rule
# --------------------------------------------------------------------------

def test_token_trading_to_the_window_end_is_alive():
    rule, last = P.classify_delisting(_alive(), WINDOW_END)
    assert rule == P.RULE_ALIVE
    assert last is None


def test_final_bar_with_no_volume_is_a_total_loss():
    """A quote nobody traded at is not an exit price."""
    rule, last = P.classify_delisting(_dies("2024-02-10", last_volume=0.0), WINDOW_END)
    assert rule == P.RULE_TOTAL_LOSS
    assert last == pd.Timestamp("2024-02-10")


def test_final_bar_with_real_volume_is_a_documented_exit():
    rule, last = P.classify_delisting(_dies("2024-02-10", last_volume=1.2e5), WINDOW_END)
    assert rule == P.RULE_LAST_PRICE
    assert last == pd.Timestamp("2024-02-10")


def test_history_that_is_all_missing_prices_is_a_total_loss():
    empty = _history(pd.date_range(WINDOW_START, periods=5), [np.nan] * 5, [0.0] * 5)
    rule, last = P.classify_delisting(empty, WINDOW_END)
    assert rule == P.RULE_TOTAL_LOSS
    assert last is None


def test_non_positive_close_is_not_treated_as_a_price():
    hist = _history(pd.date_range(WINDOW_START, periods=5), [1.0, 1.0, 1.0, 0.0, 0.0],
                    [5e5, 5e5, 5e5, 5e5, 5e5])
    rule, last = P.classify_delisting(hist, WINDOW_END)
    assert rule == P.RULE_TOTAL_LOSS
    assert last == pd.Timestamp("2024-01-03")


def test_a_late_final_bar_is_staleness_not_death():
    """CMC bars lag; a token whose last bar is inside the tolerance is alive."""
    rule, _ = P.classify_delisting(_dies(WINDOW_END - pd.Timedelta(days=2), 0.0), WINDOW_END)
    assert rule == P.RULE_ALIVE


# --------------------------------------------------------------------------
# the delisting return actually reaching the return panel
# --------------------------------------------------------------------------

def _fixture_universe():
    hists = {101: _alive(), 202: _dies("2024-02-10", 0.0), 303: _dies("2024-02-10", 1.2e5)}
    snaps = pd.DataFrame([
        {"snapshot_date": pd.Timestamp("2024-01-31"), "cmc_id": i, "symbol": s,
         "name": s, "slug": s.lower(), "rank": 200, "price_usd": 1.0,
         "volume_24h_usd": 1e6, "market_cap_usd": 1e8}
        for i, s in [(101, "LIVE"), (202, "DEAD"), (303, "EXIT")]
    ])
    table = P.build_universe_table(hists, snaps, dead_ids={202},
                                   window_start=WINDOW_START, window_end=WINDOW_END)
    return hists, snaps, table


def test_delisting_rule_is_recorded_for_every_token():
    _, _, table = _fixture_universe()
    rules = dict(zip(table["cmc_id"], table["delisting_rule"]))
    assert rules == {101: P.RULE_ALIVE, 202: P.RULE_TOTAL_LOSS, 303: P.RULE_LAST_PRICE}
    assert table["delisting_rule"].notna().all()


def test_total_loss_writes_one_terminal_return_the_day_after_the_last_bar():
    hists, _, table = _fixture_universe()
    idx = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    close = P.panel(hists, "close", idx)
    eth = pd.Series(100.0, index=idx)  # flat benchmark isolates the token effect

    ret = P.excess_log_returns(close, eth, table)
    shock_date = pd.Timestamp("2024-02-11")

    assert ret.loc[shock_date, 202] == pytest.approx(np.log(P.TOTAL_LOSS_RESIDUAL))
    # and only that one day: the rest of the dead token's post-death panel is empty
    assert ret.loc[ret.index > shock_date, 202].notna().sum() == 0


def test_the_terminal_return_is_finite_because_the_panel_is_in_log_space():
    """log(0) is -inf and would poison every downstream statistic."""
    hists, _, table = _fixture_universe()
    idx = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    ret = P.excess_log_returns(P.panel(hists, "close", idx), pd.Series(100.0, index=idx), table)
    shock = ret.loc[pd.Timestamp("2024-02-11"), 202]
    assert np.isfinite(shock)
    assert shock < np.log(0.5)  # unambiguously a wipeout, not a bad day


def test_documented_exit_price_gets_no_extra_return():
    hists, _, table = _fixture_universe()
    idx = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    ret = P.excess_log_returns(P.panel(hists, "close", idx), pd.Series(100.0, index=idx), table)
    assert ret.loc[pd.Timestamp("2024-02-11"), 303] != ret.loc[pd.Timestamp("2024-02-11"), 202]
    assert pd.isna(ret.loc[pd.Timestamp("2024-02-11"), 303])


def test_the_shock_is_measured_against_the_benchmark_like_every_other_return():
    """The panel is ETH-excess, so a wipeout on a day ETH rallied is worse."""
    hists, _, table = _fixture_universe()
    idx = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    close = P.panel(hists, "close", idx)
    eth = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)
    ret = P.excess_log_returns(close, eth, table)
    eth_ret = float(np.log(eth / eth.shift(1)).loc[pd.Timestamp("2024-02-11")])
    assert ret.loc[pd.Timestamp("2024-02-11"), 202] == pytest.approx(
        np.log(P.TOTAL_LOSS_RESIDUAL) - eth_ret)


def test_residual_knob_moves_the_shock_and_is_recorded():
    hists, _, table = _fixture_universe()
    idx = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    close = P.panel(hists, "close", idx)
    eth = pd.Series(100.0, index=idx)
    harsh = P.excess_log_returns(close, eth, table, total_loss_residual=0.001)
    assert harsh.loc[pd.Timestamp("2024-02-11"), 202] == pytest.approx(np.log(0.001))
    recorded = table.set_index("cmc_id").loc[202, "total_loss_residual"]
    assert recorded == pytest.approx(P.TOTAL_LOSS_RESIDUAL)
    assert pd.isna(table.set_index("cmc_id").loc[101, "total_loss_residual"])


# --------------------------------------------------------------------------
# cmc_id keying
# --------------------------------------------------------------------------

def test_panel_columns_are_cmc_ids_not_symbols():
    hists, _, _ = _fixture_universe()
    idx = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    close = P.panel(hists, "close", idx)
    assert list(close.columns) == [101, 202, 303]
    assert all(isinstance(c, (int, np.integer)) for c in close.columns)


def test_symbol_reuse_does_not_splice_two_assets_into_one_series():
    """CMC reuses tickers. BTM, ERC20 and BTCU in this sample are all dead
    tokens whose symbols now belong to something else; keying on symbol would
    quietly concatenate the corpse and the successor and call it a return."""
    dead_btm = _dies("2024-02-10", 0.0)
    live_btm = _alive()
    hists = {1866: dead_btm, 999901: live_btm}
    snaps = pd.DataFrame([
        {"snapshot_date": pd.Timestamp("2024-01-31"), "cmc_id": i, "symbol": "BTM",
         "name": "BTM", "slug": "btm", "rank": 300, "price_usd": 1.0,
         "volume_24h_usd": 1e6, "market_cap_usd": 1e8}
        for i in (1866, 999901)
    ])
    table = P.build_universe_table(hists, snaps, dead_ids={1866},
                                   window_start=WINDOW_START, window_end=WINDOW_END)

    idx = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    close = P.panel(hists, "close", idx)

    assert set(close.columns) == {1866, 999901}
    assert (table["symbol"] == "BTM").all() and len(table) == 2
    assert table.set_index("cmc_id").loc[1866, "delisted"]
    assert not table.set_index("cmc_id").loc[999901, "delisted"]
    # the dead one stops; the live one does not. A symbol key would hide this.
    assert close[1866].last_valid_index() == pd.Timestamp("2024-02-10")
    assert close[999901].last_valid_index() == WINDOW_END


def test_delisted_flag_covers_both_death_modes():
    """CMC dropping a coin and a coin's prices stopping are both survivorship."""
    hists = {1: _alive(), 2: _dies("2024-02-10", 1.2e5)}
    snaps = pd.DataFrame([
        {"snapshot_date": pd.Timestamp("2024-01-31"), "cmc_id": i, "symbol": f"T{i}",
         "name": f"T{i}", "slug": f"t{i}", "rank": 300, "price_usd": 1.0,
         "volume_24h_usd": 1e6, "market_cap_usd": 1e8} for i in (1, 2)
    ])
    # id 1 still trades but CMC no longer carries it; id 2 trades then stops.
    table = P.build_universe_table(hists, snaps, dead_ids={1},
                                   window_start=WINDOW_START, window_end=WINDOW_END).set_index("cmc_id")
    assert table.loc[1, "delisted"] and table.loc[1, "cmc_untracked"]
    assert not table.loc[1, "prices_stop_in_window"]
    assert table.loc[2, "delisted"] and table.loc[2, "prices_stop_in_window"]
    assert not table.loc[2, "cmc_untracked"]


# --------------------------------------------------------------------------
# membership: point-in-time, and the survivor-only counterfactual
# --------------------------------------------------------------------------

def _two_snapshots():
    return pd.DataFrame([
        {"snapshot_date": pd.Timestamp("2024-01-31"), "cmc_id": 1, "rank": 200},
        {"snapshot_date": pd.Timestamp("2024-01-31"), "cmc_id": 2, "rank": 900},
        {"snapshot_date": pd.Timestamp("2024-02-29"), "cmc_id": 1, "rank": 200},
        {"snapshot_date": pd.Timestamp("2024-02-29"), "cmc_id": 2, "rank": 300},
    ])


def test_membership_never_uses_a_future_snapshot():
    idx = pd.date_range("2024-01-31", "2024-03-05", freq="D")
    mask = P.pit_membership(_two_snapshots(), idx, rank_lo=150, rank_hi=500)
    assert mask.loc[pd.Timestamp("2024-02-28"), 2] == False  # noqa: E712
    assert mask.loc[pd.Timestamp("2024-02-29"), 2] == True  # noqa: E712
    assert mask[1].all()


def test_membership_is_empty_before_the_first_snapshot():
    idx = pd.date_range("2024-01-01", "2024-02-05", freq="D")
    mask = P.pit_membership(_two_snapshots(), idx, rank_lo=150, rank_hi=500)
    assert not mask.loc[:pd.Timestamp("2024-01-30")].to_numpy().any()


def test_survivor_only_differs_from_pit_by_exactly_the_dead_tokens():
    idx = pd.date_range("2024-01-31", "2024-03-05", freq="D")
    pit = P.pit_membership(_two_snapshots(), idx, rank_lo=150, rank_hi=500)
    table = pd.DataFrame({"cmc_id": [1, 2], "delisted": [False, True]})
    survivor = P.drop_delisted(pit, table)
    assert survivor[1].equals(pit[1])
    assert not survivor[2].any()


# --------------------------------------------------------------------------
# fetch layer, stubbed
# --------------------------------------------------------------------------

def test_history_pages_backwards_because_the_api_ignores_time_start(monkeypatch):
    """The endpoint returns the last 400 bars before timeEnd and drops
    timeStart, so a longer span has to walk backwards or it silently truncates."""
    calls = []

    def fake_get(path, params, **kwargs):
        calls.append(params["timeEnd"])
        end = pd.Timestamp(params["timeEnd"], unit="s").normalize()
        days = pd.date_range(end - pd.Timedelta(days=C.MAX_HISTORY_ROWS - 1), end, freq="D")
        return {"data": {"quotes": [
            {"timeOpen": d.strftime("%Y-%m-%dT00:00:00.000Z"),
             "quote": {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 10.0, "marketCap": 100.0}}
            for d in days
        ]}}

    monkeypatch.setattr(C, "_get", fake_get)
    hist = C.fetch_history(7, "2023-05-01", "2025-06-01")

    assert len(calls) > 1, "a 760-day span must not be requested in one 400-bar call"
    assert calls == sorted(calls, reverse=True), "paging must walk backwards"
    assert hist["date"].min() == pd.Timestamp("2023-05-01")
    assert hist["date"].max() == pd.Timestamp("2025-06-01")
    assert not hist["date"].duplicated().any()


def test_history_stops_when_a_dead_coin_runs_out_of_bars(monkeypatch):
    def fake_get(path, params, **kwargs):
        return {"data": {"quotes": [
            {"timeOpen": "2024-02-09T00:00:00.000Z",
             "quote": {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 0.0, "marketCap": 100.0}}]}}

    monkeypatch.setattr(C, "_get", fake_get)
    hist = C.fetch_history(7, "2023-05-01", "2025-06-01")
    assert len(hist) == 1


def test_listing_snapshot_parses_the_point_in_time_rank_and_quote(monkeypatch):
    def fake_get(path, params, **kwargs):
        assert params["date"] == "2024-01-31"
        return {"data": [{
            "id": 1866, "symbol": "BTM", "name": "BytomDAO", "slug": "bytom",
            "cmcRank": 390,
            "quotes": [{"price": 0.05, "volume24h": 1.2e6, "marketCap": 5.0e7}],
        }]}

    monkeypatch.setattr(C, "_get", fake_get)
    snap = C.fetch_listing_snapshot("2024-01-31", depth=500)
    row = snap.iloc[0]
    assert row["cmc_id"] == 1866 and row["rank"] == 390
    assert row["market_cap_usd"] == pytest.approx(5.0e7)
    assert row["snapshot_date"] == pd.Timestamp("2024-01-31")


def test_map_pagination_stops_on_a_short_page(monkeypatch):
    pages = [[{"id": i, "symbol": f"S{i}", "name": f"N{i}", "slug": f"s{i}",
               "is_active": 0} for i in range(3)]]

    def fake_get(path, params, **kwargs):
        return {"data": pages.pop(0) if pages else []}

    monkeypatch.setattr(C, "_get", fake_get)
    df = C.fetch_map("inactive", page_size=5000)
    assert len(df) == 3
    assert (df["listing_status"] == "inactive").all()
    assert list(df["cmc_id"]) == [0, 1, 2]


# --------------------------------------------------------------------------
# price artifacts
# --------------------------------------------------------------------------

def test_scrub_drops_impossible_moves_and_counts_them():
    idx = pd.date_range(WINDOW_START, periods=4, freq="D")
    ret = pd.DataFrame({1: [0.01, np.log(80.0), -0.02, 0.0],   # vBNB-style bad print
                        2: [0.01, 0.02, 0.03, 0.04]}, index=idx)
    cleaned, n = P.scrub_extreme_returns(ret)
    assert n == 1
    assert pd.isna(cleaned.loc[idx[1], 1])
    assert cleaned[2].equals(ret[2])


def test_scrub_is_symmetric_and_leaves_ordinary_crypto_moves_alone():
    idx = pd.date_range(WINDOW_START, periods=3, freq="D")
    ret = pd.DataFrame({1: [np.log(2.5), -np.log(2.5), np.log(6.0)]}, index=idx)
    cleaned, n = P.scrub_extreme_returns(ret)
    assert n == 1                       # only the 6x, not the 2.5x either way
    assert cleaned.loc[idx[0], 1] == pytest.approx(np.log(2.5))
    assert cleaned.loc[idx[1], 1] == pytest.approx(-np.log(2.5))


def test_scrub_can_be_switched_off():
    idx = pd.date_range(WINDOW_START, periods=2, freq="D")
    ret = pd.DataFrame({1: [0.0, np.log(9999.0)]}, index=idx)
    cleaned, n = P.scrub_extreme_returns(ret, max_abs_log_return=None)
    assert n == 0 and cleaned.equals(ret)


def test_the_delisting_shock_survives_the_scrub():
    """Ordering regression: log(0.01) = -4.61 is far past the artifact
    threshold, so scrubbing after the shock instead of before would silently
    delete every delisting return and hand back the bias we set out to remove."""
    hists, _, table = _fixture_universe()
    idx = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    ret = P.excess_log_returns(P.panel(hists, "close", idx), pd.Series(100.0, index=idx), table)
    assert abs(np.log(P.TOTAL_LOSS_RESIDUAL)) > P.MAX_ABS_LOG_RETURN
    assert ret.loc[pd.Timestamp("2024-02-11"), 202] == pytest.approx(
        np.log(P.TOTAL_LOSS_RESIDUAL))
