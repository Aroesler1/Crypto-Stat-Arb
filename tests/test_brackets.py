"""Tests for ETH-relative bracket assignment and the broader exclusion filter.

No network. What is pinned here is the three things that would quietly change
what "small cap" means:

1. bracket edges are half-open log decades of ``r = mcap_i / mcap_ETH``, so a
   token sitting exactly on an edge lands in one bracket and not both;
2. membership on any day is decided by the most recent *prior* snapshot, so a
   market cap from the future cannot select the universe that trades on it;
3. the exclusion filter removes stablecoins, wrappers, staking receipts and
   bridge claims, which are redundant quotes on assets already in the panel.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.data import brackets as B  # noqa: E402
from stat_arb.data.loader import is_derivative_or_stable  # noqa: E402

ETH = B.ETH_CMC_ID


def _snapshots(rows, dates=("2016-01-31", "2016-02-29"), eth_mcap=1e9):
    """Build a snapshot frame; `rows` is [(cmc_id, mcap_fraction_of_eth), ...]."""
    out = []
    for d in dates:
        out.append({"snapshot_date": pd.Timestamp(d), "cmc_id": ETH, "symbol": "ETH",
                    "name": "Ethereum", "slug": "ethereum", "rank": 2,
                    "price_usd": 100.0, "volume_24h_usd": 1e8,
                    "market_cap_usd": eth_mcap})
        for i, (cid, frac) in enumerate(rows):
            out.append({"snapshot_date": pd.Timestamp(d), "cmc_id": cid,
                        "symbol": f"T{cid}", "name": f"Token {cid}",
                        "slug": f"token-{cid}", "rank": 10 + i,
                        "price_usd": 1.0, "volume_24h_usd": 1e6,
                        "market_cap_usd": frac * eth_mcap})
    return pd.DataFrame(out)


# --- bracket assignment ----------------------------------------------------

def test_each_bracket_gets_its_decade():
    snaps = _snapshots([(10, 0.5), (11, 0.05), (12, 0.005), (13, 0.0005)])
    a = B.assign_brackets(snaps)
    got = a.drop_duplicates("cmc_id").set_index("cmc_id")["bracket"].to_dict()
    assert got == {10: "B0", 11: "B1", 12: "B2", 13: "B3"}


@pytest.mark.parametrize("ratio,expected", [
    (1.0, "B0"), (0.10, "B0"),          # lower edge is inclusive
    (0.0999, "B1"), (0.01, "B1"),
    (0.00999, "B2"), (0.001, "B2"),
    (0.00099, "B3"), (0.0001, "B3"),    # exactly at the drop line: kept
    (0.00009, None),                    # below it: dropped
    (0.0, None),
])
def test_bracket_edges_are_half_open(ratio, expected):
    assert B.bracket_of(ratio) == expected


def test_edges_partition_without_overlap():
    """Every bracket edge belongs to exactly one bracket."""
    for name in B.BRACKET_ORDER:
        lo, hi = B.BRACKET_EDGES[name]
        assert B.bracket_of(lo) == name
        if np.isfinite(hi):
            assert B.bracket_of(hi) != name


def test_below_drop_line_is_excluded_and_eth_is_never_a_member():
    snaps = _snapshots([(10, 0.5), (99, 1e-6)])
    a = B.assign_brackets(snaps)
    assert 99 not in set(a["cmc_id"])       # below 0.01% of ETH, dropped
    assert ETH not in set(a["cmc_id"])      # the reference is never traded


def test_non_positive_market_cap_is_dropped_not_assigned():
    snaps = _snapshots([(10, 0.5)])
    snaps.loc[snaps["cmc_id"] == 10, "market_cap_usd"] = 0.0
    assert B.assign_brackets(snaps).empty


def test_assignment_is_point_in_time_when_eth_moves():
    """The same dollar market cap changes bracket when ETH's size changes.

    This is the property that makes the definition worth having: a token worth
    $5M is a different animal against a $1B ETH than against a $100B one.
    """
    early = _snapshots([(10, 0.05)], dates=("2016-01-31",), eth_mcap=1e9)
    late = _snapshots([(10, 0.05)], dates=("2021-01-31",), eth_mcap=1e11)
    late.loc[late["cmc_id"] == 10, "market_cap_usd"] = 5e7   # the same $50M
    snaps = pd.concat([early, late], ignore_index=True)
    a = B.assign_brackets(snaps).set_index(["snapshot_date", "cmc_id"])["bracket"]
    assert a[(pd.Timestamp("2016-01-31"), 10)] == "B1"   # $50M vs $1B ETH
    assert a[(pd.Timestamp("2021-01-31"), 10)] == "B3"   # $50M vs $100B ETH


def test_start_date_defaults_to_eth_era():
    """No ETH-relative bracket exists before ETH does."""
    snaps = _snapshots([(10, 0.5)], dates=("2014-06-30", "2016-01-31"))
    a = B.assign_brackets(snaps)
    assert a["snapshot_date"].min() == pd.Timestamp("2016-01-31")
    assert (a["snapshot_date"] >= B.BRACKET_START).all()


def test_missing_eth_raises_rather_than_guessing():
    snaps = _snapshots([(10, 0.5)])
    with pytest.raises(ValueError, match="ETH"):
        B.assign_brackets(snaps[snaps["cmc_id"] != ETH])


# --- membership ------------------------------------------------------------

def test_membership_uses_the_most_recent_prior_snapshot_only():
    snaps = _snapshots([(10, 0.5)], dates=("2016-01-31", "2016-02-29"))
    # token 10 leaves B0 for B3 at the February reconstitution
    m = snaps["cmc_id"].eq(10) & snaps["snapshot_date"].eq(pd.Timestamp("2016-02-29"))
    snaps.loc[m, "market_cap_usd"] = 0.0005 * 1e9
    a = B.assign_brackets(snaps)
    index = pd.date_range("2016-01-25", "2016-03-05", freq="D")
    b0 = B.bracket_membership(a, index, "B0", columns=[10])

    assert not b0.loc[pd.Timestamp("2016-01-30"), 10]   # before the first snapshot
    assert b0.loc[pd.Timestamp("2016-01-31"), 10]       # on it
    assert b0.loc[pd.Timestamp("2016-02-28"), 10]       # still, until the next one
    assert not b0.loc[pd.Timestamp("2016-03-01"), 10]   # reconstituted out


def test_membership_columns_are_sorted_ints():
    a = B.assign_brackets(_snapshots([(30, 0.5), (10, 0.5), (20, 0.5)]))
    m = B.bracket_membership(a, pd.date_range("2016-02-01", periods=3), "B0")
    assert list(m.columns) == [10, 20, 30]
    assert all(isinstance(c, int) for c in m.columns)


def test_membership_of_an_empty_bracket_is_all_false():
    a = B.assign_brackets(_snapshots([(10, 0.5)]))
    m = B.bracket_membership(a, pd.date_range("2016-02-01", periods=3), "B2",
                             columns=[10])
    assert not m.to_numpy().any()


# --- dollar bounds and the rank cutoff -------------------------------------

def test_dollar_bounds_move_with_eth():
    snaps = pd.concat([
        _snapshots([(10, 0.5)], dates=("2016-01-31",), eth_mcap=1e9),
        _snapshots([(10, 0.5)], dates=("2021-01-31",), eth_mcap=1e11),
    ], ignore_index=True)
    b = B.bracket_dollar_bounds(snaps).set_index(["snapshot_date", "bracket"])
    assert b.loc[(pd.Timestamp("2016-01-31"), "B3"), "lo_usd"] == pytest.approx(1e5)
    assert b.loc[(pd.Timestamp("2021-01-31"), "B3"), "lo_usd"] == pytest.approx(1e7)
    assert np.isinf(b.loc[(pd.Timestamp("2016-01-31"), "B0"), "hi_usd"])


def test_rank_cutoff_binds_only_when_the_pull_runs_out_of_depth():
    """The deepest token pulled still being above the floor means names are missing."""
    eth_mcap = 1e9                     # B3 floor = 0.01% of ETH = $100k
    deep_rich = _snapshots([(10, 0.5)], dates=("2016-01-31",), eth_mcap=eth_mcap)
    deep_rich.loc[deep_rich["cmc_id"] == 10, "rank"] = 2000
    deep_rich.loc[deep_rich["cmc_id"] == 10, "market_cap_usd"] = 1e6   # > $100k floor
    assert B.rank_cutoff_binds(deep_rich, 2000, "B3")["binds"].all()

    deep_poor = deep_rich.copy()
    deep_poor.loc[deep_poor["cmc_id"] == 10, "market_cap_usd"] = 1e4   # < $100k floor
    assert not B.rank_cutoff_binds(deep_poor, 2000, "B3")["binds"].any()


def test_rank_cutoff_does_not_bind_when_cmc_never_ranked_that_deep():
    """A shallow month is CMC running out of coins, not the pull running out of
    depth. Reporting it as binding would blame the wrong constraint."""
    shallow = _snapshots([(10, 0.5)], dates=("2016-01-31",), eth_mcap=1e9)
    shallow.loc[shallow["cmc_id"] == 10, "rank"] = 400        # deepest rank is 400
    shallow.loc[shallow["cmc_id"] == 10, "market_cap_usd"] = 1e6   # above the floor
    out = B.rank_cutoff_binds(shallow, 2000, "B3")
    assert not out["binds"].any()
    assert not out["exhausted_depth"].any()
    assert out["n_ranked"].iloc[0] == 400


def test_zero_market_caps_in_the_tail_do_not_pass_for_a_deep_pull():
    """CMC prints null caps in the early-year tail; taking them at face value
    would report a $0 deepest cap and call the pull deep enough."""
    snaps = _snapshots([(10, 0.5), (11, 0.5)], dates=("2016-01-31",), eth_mcap=1e9)
    snaps.loc[snaps["cmc_id"] == 10, ["rank", "market_cap_usd"]] = [1999, 1e6]
    snaps.loc[snaps["cmc_id"] == 11, ["rank", "market_cap_usd"]] = [2000, 0.0]
    out = B.rank_cutoff_binds(snaps, 2000, "B3")
    assert out["deepest_market_cap_usd"].iloc[0] == 1e6   # the priced one, not the zero
    assert out["binds"].all()


def test_member_counts_flag_the_clustering_floor():
    rows = [(100 + i, 0.005) for i in range(B.MIN_MEMBERS_FOR_CLUSTERING - 1)]
    counts = B.member_counts(B.assign_brackets(_snapshots(rows)))
    assert not counts["clusterable"].any()

    rows.append((999, 0.005))
    counts = B.member_counts(B.assign_brackets(_snapshots(rows)))
    assert counts["clusterable"].all()
    assert (counts["n_members"] == B.MIN_MEMBERS_FOR_CLUSTERING).all()


# --- the exclusion filter --------------------------------------------------

@pytest.mark.parametrize("symbol,name", [
    ("USDT", "Tether"), ("WBTC", "Wrapped Bitcoin"), ("STETH", "Lido Staked Ether"),
    ("DAI", "Dai"), ("PAXG", "PAX Gold"),
])
def test_curated_ticker_list_still_governs(symbol, name):
    assert is_derivative_or_stable(symbol, name)


@pytest.mark.parametrize("symbol,name", [
    ("XYZ", "Wrapped Fantom"),            # wrapper the ticker list never heard of
    ("ABC", "Bridged USDC (Polygon)"),    # bridge claim
    ("QQQ", "Rocket Pool Staked ETH"),    # staking receipt
    ("PEG1", "Binance-Peg Cardano"),      # pegged wrapper
    ("NEW", "Some New USD Stablecoin"),   # stablecoin by name
    ("EUX", "Euro Coin"),                 # non-USD stablecoin
])
def test_name_patterns_catch_what_the_ticker_list_cannot(symbol, name):
    assert is_derivative_or_stable(symbol, name)


@pytest.mark.parametrize("symbol,name", [
    ("BTC", "Bitcoin"), ("SOL", "Solana"), ("LINK", "Chainlink"),
    ("UNI", "Uniswap"), ("DOGE", "Dogecoin"), ("AAVE", "Aave"),
    ("PUNDIX", "Pundi X"),                # contains "pundi x", an exception
])
def test_ordinary_assets_are_kept(symbol, name):
    assert not is_derivative_or_stable(symbol, name)


def test_exclusion_handles_missing_metadata():
    assert not is_derivative_or_stable(None, None)
    assert not is_derivative_or_stable("", "")
    assert is_derivative_or_stable("USDC", None)
