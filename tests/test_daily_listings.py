"""Tests for the daily point-in-time listing panel.

No network. The panel replaces the per-token history endpoint as the price
source because that endpoint prunes dead coins (see
`build_daily_listings.py`), so what has to be pinned here is that the
replacement does not reintroduce the bias by a different route:

1. a token that dies must lose money, and the rule that decided how much must
   be recorded rather than assumed;
2. a token that merely falls below the pull depth must NOT lose money. It is
   alive; charging it -99% invents a loss that never happened. This is the
   distinction the older per-token panel never had to make, because a token
   that fell out of a rank band still had its own history series;
3. the reference asset is a parameter, because the residualization ablation
   runs the same panel against ETH, against BTC and against a value-weighted
   market.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.data import daily_listings as D  # noqa: E402
from stat_arb.data import pit_universe as P  # noqa: E402

START = pd.Timestamp("2016-01-01")
END = pd.Timestamp("2016-06-30")
INDEX = pd.date_range(START, END, freq="D")


def _rows(cmc_id, dates, prices, volumes=1e6, mcap=1e7, symbol=None):
    n = len(dates)
    vol = [volumes] * n if np.isscalar(volumes) else list(volumes)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "cmc_id": [int(cmc_id)] * n,
        "symbol": [symbol or f"T{cmc_id}"] * n,
        "name": [f"Token {cmc_id}"] * n,
        "slug": [f"token-{cmc_id}"] * n,
        "rank": list(range(1, n + 1)),
        "price_usd": list(prices),
        "volume_24h_usd": vol,
        "market_cap_usd": [mcap] * n,
    })


def _panel(*frames):
    return pd.concat(frames, ignore_index=True).sort_values(["date", "cmc_id"])


def _flat(cmc_id, dates, price=10.0, **kw):
    return _rows(cmc_id, dates, [price] * len(dates), **kw)


# --- wide panels -----------------------------------------------------------

def test_wide_is_sorted_and_int_keyed():
    p = _panel(_flat(30, INDEX[:5]), _flat(10, INDEX[:5]), _flat(20, INDEX[:5]))
    w = D.wide(p, "price_usd", INDEX[:5])
    assert list(w.columns) == [10, 20, 30]
    assert all(isinstance(c, int) for c in w.columns)


def test_wide_reindexes_to_the_requested_calendar():
    p = _flat(10, INDEX[:3])
    w = D.wide(p, "price_usd", INDEX[:10])
    assert len(w) == 10
    assert w[10].notna().sum() == 3


# --- death versus censoring ------------------------------------------------

def test_a_dead_token_is_a_death_and_gets_a_delisting_rule():
    p = _flat(10, INDEX[:60])                       # stops well before END
    t = D.build_universe_table(p, dead_ids={10}, window_end=END)
    row = t.iloc[0]
    assert row["exit_kind"] == D.EXIT_DEATH
    assert row["delisted"] and not row["censored"]
    assert row["delisting_date"] == INDEX[59]


def test_a_token_that_falls_out_of_the_ranking_alive_is_censored_not_killed():
    """The distinction the panel exists to get right."""
    p = _flat(10, INDEX[:60])
    t = D.build_universe_table(p, dead_ids=set(), window_end=END)  # not in dead map
    row = t.iloc[0]
    assert row["exit_kind"] == D.EXIT_CENSORED
    assert row["censored"] and not row["delisted"]
    assert row["delisting_rule"] == P.RULE_ALIVE
    assert pd.isna(row["delisting_date"])


def test_a_token_ranked_to_the_end_is_still_ranked():
    t = D.build_universe_table(_flat(10, INDEX), dead_ids=set(), window_end=END)
    assert t.iloc[0]["exit_kind"] == D.EXIT_NONE
    assert not t.iloc[0]["delisted"] and not t.iloc[0]["censored"]


def test_death_with_volume_exits_at_the_last_quote_without_volume_is_total_loss():
    with_vol = _flat(10, INDEX[:60], volumes=5e5)
    no_vol = _flat(11, INDEX[:60], volumes=0.0)
    t = D.build_universe_table(_panel(with_vol, no_vol), dead_ids={10, 11},
                               window_end=END).set_index("cmc_id")
    assert t.loc[10, "delisting_rule"] == P.RULE_LAST_PRICE
    assert t.loc[11, "delisting_rule"] == P.RULE_TOTAL_LOSS
    assert t.loc[11, "total_loss_residual"] == pytest.approx(P.TOTAL_LOSS_RESIDUAL)
    assert pd.isna(t.loc[10, "total_loss_residual"])


def test_a_token_that_reappears_is_classified_on_its_final_disappearance():
    gapped = _panel(_flat(10, INDEX[:20]), _flat(10, INDEX[40:60]))
    t = D.build_universe_table(gapped, dead_ids={10}, window_end=END)
    assert t.iloc[0]["delisting_date"] == INDEX[59]
    assert t.iloc[0]["n_obs"] == 40


def test_non_positive_prices_do_not_count_as_a_listing_life():
    p = _rows(10, INDEX[:5], [10.0, 10.0, 0.0, 0.0, 0.0])
    t = D.build_universe_table(p, dead_ids={10}, window_end=END)
    assert t.iloc[0]["last_date"] == INDEX[1]
    assert t.iloc[0]["n_obs"] == 2


# --- delisting returns -----------------------------------------------------

def _eth(index, level=100.0):
    return pd.Series(level, index=index, name="eth")


def test_total_loss_writes_the_shock_the_day_after_the_last_quote():
    p = _flat(10, INDEX[:60], volumes=0.0)
    t = D.build_universe_table(p, dead_ids={10}, window_end=END)
    r, _ = D.excess_log_returns(D.wide(p, "price_usd", INDEX), _eth(INDEX), t)
    shock_day = INDEX[60]
    assert r.loc[shock_day, 10] == pytest.approx(np.log(P.TOTAL_LOSS_RESIDUAL))
    assert r.loc[INDEX[61]:, 10].isna().all()


def test_last_price_death_gets_no_extra_return():
    """It exited at a real traded quote; the return into it is already there."""
    p = _flat(10, INDEX[:60], volumes=5e5)
    t = D.build_universe_table(p, dead_ids={10}, window_end=END)
    r, _ = D.excess_log_returns(D.wide(p, "price_usd", INDEX), _eth(INDEX), t)
    assert r.loc[INDEX[60]:, 10].isna().all()


def test_a_censored_token_is_never_charged_a_terminal_loss():
    """The counterpart of the total-loss test, and the reason it matters."""
    p = _flat(10, INDEX[:60], volumes=0.0)
    t = D.build_universe_table(p, dead_ids=set(), window_end=END)   # alive today
    r, _ = D.excess_log_returns(D.wide(p, "price_usd", INDEX), _eth(INDEX), t)
    assert r.loc[INDEX[60]:, 10].isna().all()
    assert (r[10].dropna() == 0).all()


def test_returns_are_in_excess_of_the_reference_asset():
    p = _rows(10, INDEX[:3], [10.0, 20.0, 20.0])
    eth = pd.Series([100.0, 200.0, 400.0], index=INDEX[:3])
    t = D.build_universe_table(p, dead_ids=set(), window_end=END)
    r, _ = D.excess_log_returns(D.wide(p, "price_usd", INDEX[:3]), eth, t)
    # day 1: token doubles, ETH doubles -> exactly zero excess
    assert r.loc[INDEX[1], 10] == pytest.approx(0.0, abs=1e-12)
    # day 2: token flat, ETH doubles -> -log 2
    assert r.loc[INDEX[2], 10] == pytest.approx(-np.log(2))


def test_the_reference_asset_is_a_parameter():
    """ETH-excess, BTC-excess and raw come from the same function."""
    p = _rows(10, INDEX[:2], [10.0, 20.0])
    t = D.build_universe_table(p, dead_ids=set(), window_end=END)
    close = D.wide(p, "price_usd", INDEX[:2])
    raw, _ = D.excess_log_returns(close, pd.Series(1.0, index=INDEX[:2]), t)
    btc, _ = D.excess_log_returns(close, pd.Series([50.0, 100.0], index=INDEX[:2]), t)
    assert raw.loc[INDEX[1], 10] == pytest.approx(np.log(2))
    assert btc.loc[INDEX[1], 10] == pytest.approx(0.0, abs=1e-12)


def test_extreme_prints_are_scrubbed_and_counted():
    p = _rows(10, INDEX[:4], [10.0, 10.0, 1000.0, 10.0])   # a x100 bad print
    t = D.build_universe_table(p, dead_ids=set(), window_end=END)
    r, n = D.excess_log_returns(D.wide(p, "price_usd", INDEX[:4]), _eth(INDEX[:4]), t)
    assert n == 2                       # the spike and its reversal
    assert r.loc[INDEX[2], 10] != r.loc[INDEX[2], 10]   # NaN


def test_scrubbing_never_removes_the_deliberate_delisting_shock():
    p = _flat(10, INDEX[:60], volumes=0.0)
    t = D.build_universe_table(p, dead_ids={10}, window_end=END)
    r, _ = D.excess_log_returns(D.wide(p, "price_usd", INDEX), _eth(INDEX), t)
    assert r.loc[INDEX[60], 10] == pytest.approx(np.log(P.TOTAL_LOSS_RESIDUAL))
    assert abs(r.loc[INDEX[60], 10]) > P.MAX_ABS_LOG_RETURN   # would have been scrubbed


# --- value-weighted market -------------------------------------------------

def test_value_weighted_market_uses_lagged_weights():
    """A token cannot weight itself by the size its own move gave it."""
    big = _rows(10, INDEX[:3], [10.0, 10.0, 10.0], mcap=1e9)
    small = _rows(11, INDEX[:3], [10.0, 20.0, 20.0], mcap=1e6)
    p = _panel(big, small)
    close = D.wide(p, "price_usd", INDEX[:3])
    mcap = D.wide(p, "market_cap_usd", INDEX[:3])
    mkt = D.value_weighted_market(close, mcap)
    day1 = np.log(mkt.iloc[1] / mkt.iloc[0])
    # the doubling name is 0.1% of the market, so the index barely moves
    assert 0 < day1 < 0.001 * np.log(2) * 1.5


def test_exit_summary_counts_deaths_and_censoring_separately():
    dead = _flat(10, INDEX[:60])
    cens = _flat(11, INDEX[:60])
    live = _flat(12, INDEX)
    t = D.build_universe_table(_panel(dead, cens, live), dead_ids={10}, window_end=END)
    s = D.exit_summary(t)
    assert s[D.EXIT_DEATH].sum() == 1
    assert s[D.EXIT_CENSORED].sum() == 1
    assert s[D.EXIT_NONE].sum() == 1


def test_validate_against_history_flags_disagreement():
    p = _rows(10, INDEX[:30], np.linspace(10, 20, 30))
    agree = pd.DataFrame({"date": INDEX[:30], "close": np.linspace(10, 20, 30)})
    disagree = pd.DataFrame({"date": INDEX[:30], "close": np.linspace(10, 20, 30) * 3})
    v = D.validate_against_history(p, {10: agree}).set_index("cmc_id")
    assert v.loc[10, "frac_disagree"] == 0.0
    v = D.validate_against_history(p, {10: disagree}).set_index("cmc_id")
    assert v.loc[10, "frac_disagree"] == 1.0


# --- redenomination check --------------------------------------------------

def test_redenomination_is_separated_from_a_genuine_move():
    """Price x10 with market cap flat is a corporate action, not a return."""
    idx = pd.date_range("2020-01-01", periods=4)
    close = pd.DataFrame({1: [1.0, 1.0, 10.0, 10.0], 2: [1.0, 1.0, 10.0, 10.0]}, index=idx)
    # token 1: supply absorbs the move (mcap flat). token 2: mcap moves with price.
    mcap = pd.DataFrame({1: [1e6, 1e6, 1e6, 1e6], 2: [1e6, 1e6, 1e7, 1e7]}, index=idx)
    f = D.redenomination_flags(close, mcap)
    assert f.loc[idx[2], 1]
    assert not f.loc[idx[2], 2]


def test_redenomination_ignores_small_price_moves():
    idx = pd.date_range("2020-01-01", periods=3)
    close = pd.DataFrame({1: [1.0, 1.05, 1.05]}, index=idx)
    mcap = pd.DataFrame({1: [1e6, 1e6, 1e6]}, index=idx)
    assert not D.redenomination_flags(close, mcap).to_numpy().any()


def test_implied_supply_is_market_cap_over_price():
    idx = pd.date_range("2020-01-01", periods=2)
    close = pd.DataFrame({1: [2.0, 4.0]}, index=idx)
    mcap = pd.DataFrame({1: [1e6, 1e6]}, index=idx)
    s = D.implied_supply(close, mcap)
    assert s.loc[idx[0], 1] == pytest.approx(5e5)
    assert s.loc[idx[1], 1] == pytest.approx(2.5e5)


def test_redenomination_flags_are_false_where_data_is_missing():
    idx = pd.date_range("2020-01-01", periods=3)
    close = pd.DataFrame({1: [1.0, np.nan, 10.0]}, index=idx)
    mcap = pd.DataFrame({1: [1e6, np.nan, 1e6]}, index=idx)
    f = D.redenomination_flags(close, mcap)
    assert not f.loc[idx[1], 1]


# --- the delisting policy is a parameter, not a silent choice ---------------

def test_volume_floor_treats_a_nil_volume_exit_as_a_total_loss():
    """A $7 final print is not an exit. Median final-day volume of a dying
    token in the real panel is $7.30, so this is the common case, not an edge."""
    p = _flat(10, INDEX[:60], volumes=7.30)
    t = D.build_universe_table(p, dead_ids={10}, window_end=END)
    assert t.iloc[0]["delisting_rule"] == P.RULE_TOTAL_LOSS
    assert t.iloc[0]["delisting_policy"] == "volume_floor"


def test_volume_floor_lets_a_genuinely_liquid_exit_through():
    p = _flat(10, INDEX[:60], volumes=5e6)
    t = D.build_universe_table(p, dead_ids={10}, window_end=END)
    assert t.iloc[0]["delisting_rule"] == P.RULE_LAST_PRICE


@pytest.mark.parametrize("policy,expected", [
    ("total_loss", P.RULE_TOTAL_LOSS),     # every death takes the shock
    ("last_price", P.RULE_LAST_PRICE),     # the naive treatment
])
def test_the_other_policies_override_the_volume_test(policy, expected):
    p = _flat(10, INDEX[:60], volumes=5e6)   # liquid, so the floor would pass it
    t = D.build_universe_table(p, dead_ids={10}, window_end=END,
                               delisting_policy=policy)
    assert t.iloc[0]["delisting_rule"] == expected


def test_no_policy_ever_charges_a_censored_token():
    for policy in D.DELISTING_POLICIES:
        t = D.build_universe_table(_flat(10, INDEX[:60], volumes=0.0),
                                   dead_ids=set(), window_end=END,
                                   delisting_policy=policy)
        assert t.iloc[0]["delisting_rule"] == P.RULE_ALIVE, policy


def test_an_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="delisting_policy"):
        D.build_universe_table(_flat(10, INDEX[:60]), dead_ids={10},
                               window_end=END, delisting_policy="optimistic")
