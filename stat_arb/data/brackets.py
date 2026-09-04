"""ETH-relative market-cap brackets, assigned point-in-time on log decades.

Why relative to ETH
-------------------
The pre-backtest report defined "small cap" not in dollars but as a band of
Ethereum's market cap: tokens between 0.001% and 0.1% of ETH. That choice is
what makes the definition survive the sample. Crypto market caps move two orders
of magnitude between 2016 and 2025, so a fixed dollar band is a different slice
of the cross-section in every year, and a rank band is a different slice again
whenever the number of listed tokens changes. A ratio to ETH is the same
economic statement in every year: "a hundredth the size of Ethereum" means the
same thing in 2017 and 2025 even though the dollar amount moves 20x.

The brackets
------------
At each month end, ``r_i = market cap of token i / market cap of ETH``, and the
token is assigned to a bracket on log decades of ``r``::

    B0 mega    r >= 10%      BTC sits above 100%; BNB, SOL, XRP and a few
                             others sit between 10% and 100%
    B1 large   1%  <= r < 10%
    B2 mid     0.1% <= r < 1%
    B3 small   0.01% <= r < 0.1%

Anything below 0.01% of ETH is **dropped by decision**, not for lack of data:
the pull reaches rank 2000 and those tokens are present, but below that line the
panel is dominated by names whose quoted price is a rounding artifact and whose
daily notional cannot support a position. B3 is the upper half of the report's
original 0.001%-0.1% band; the lower decade is exactly what is dropped.

ETH itself is the reference (``r = 1`` by construction) and is never traded.

Two properties worth stating because they drive later results:

1. **B0 will never reach 30 members.** There are only a handful of tokens above
   a tenth of ETH's size at any month end. It is not clusterable and is handled
   by a pairs book and a cross-sectional z-score without clusters.
2. **The bracket's dollar bounds move with ETH's price**, by construction. A
   report that does not show them invites the reader to assume they are fixed.
   `bracket_dollar_bounds` returns them per month end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# CoinMarketCap's id for Ethereum. The reference asset, never a tradable member.
ETH_CMC_ID = 1027

# Bracket edges as fractions of ETH's market cap, [lo, hi).
BRACKET_EDGES: dict[str, tuple[float, float]] = {
    "B0": (0.10, np.inf),
    "B1": (0.01, 0.10),
    "B2": (0.001, 0.01),
    "B3": (0.0001, 0.001),
}

BRACKET_LABELS: dict[str, str] = {
    "B0": "mega (r >= 10% of ETH)",
    "B1": "large (1% <= r < 10%)",
    "B2": "mid (0.1% <= r < 1%)",
    "B3": "small (0.01% <= r < 0.1%)",
}

BRACKET_ORDER = ("B0", "B1", "B2", "B3")

# Below this fraction of ETH, a token is dropped by decision. See the module
# docstring: this is a judgement about tradability, not a data limit.
DROP_BELOW = 0.0001

# ETH's own history starts 2015-08. Before that there is no reference asset, so
# no ETH-relative bracket can be assigned and no ETH-excess return exists. Every
# bracket panel therefore starts here rather than at the 2013 start of the pull,
# which exists only so BTC's early history and the rank snapshots are complete.
BRACKET_START = pd.Timestamp("2016-01-01")

# A bracket with fewer members than this at a month end is not clustered: a
# signed k-NN graph on fewer nodes than the smallest sensible cluster count
# times a few members is noise. The month is flagged and reported, not dropped
# silently. B1 hits this often, which is a result about the shape of the crypto
# cross-section, not a defect.
MIN_MEMBERS_FOR_CLUSTERING = 30


def eth_market_caps(snapshots: pd.DataFrame, eth_id: int = ETH_CMC_ID) -> pd.Series:
    """ETH's market cap at each snapshot date, indexed by snapshot date."""
    eth = snapshots[snapshots["cmc_id"].astype(int) == int(eth_id)]
    if eth.empty:
        raise ValueError(
            f"ETH (cmc_id={eth_id}) is absent from the snapshots; every bracket is "
            "defined relative to it, so this cannot be worked around"
        )
    s = (eth.assign(snapshot_date=pd.to_datetime(eth["snapshot_date"]))
            .sort_values("snapshot_date")
            .drop_duplicates("snapshot_date", keep="last")
            .set_index("snapshot_date")["market_cap_usd"]
            .astype(float))
    return s[s > 0]


def bracket_of(ratio: float) -> str | None:
    """Bracket label for one cap ratio, or None if it is below the drop line."""
    if not np.isfinite(ratio) or ratio < DROP_BELOW:
        return None
    for name in BRACKET_ORDER:
        lo, hi = BRACKET_EDGES[name]
        if lo <= ratio < hi:
            return name
    return None


def assign_brackets(
    snapshots: pd.DataFrame,
    eth_id: int = ETH_CMC_ID,
    start: pd.Timestamp | None = BRACKET_START,
) -> pd.DataFrame:
    """Point-in-time bracket assignment, one row per (snapshot_date, cmc_id).

    Returns the input columns plus ``cap_ratio`` and ``bracket``. Rows below
    `DROP_BELOW`, rows with a non-positive market cap, and ETH itself are
    removed: ETH is the reference, and a token with no market cap on the day has
    no ratio to assign.
    """
    snaps = snapshots.copy()
    snaps["snapshot_date"] = pd.to_datetime(snaps["snapshot_date"])
    snaps["cmc_id"] = snaps["cmc_id"].astype(int)

    eth_mcap = eth_market_caps(snaps, eth_id)
    if start is not None:
        snaps = snaps[snaps["snapshot_date"] >= pd.Timestamp(start)]

    snaps = snaps[snaps["snapshot_date"].isin(eth_mcap.index)]
    snaps = snaps[snaps["cmc_id"] != int(eth_id)]
    snaps = snaps[pd.to_numeric(snaps["market_cap_usd"], errors="coerce") > 0]

    ratio = (snaps["market_cap_usd"].astype(float).to_numpy()
             / snaps["snapshot_date"].map(eth_mcap).to_numpy())
    snaps = snaps.assign(cap_ratio=ratio)
    snaps = snaps[snaps["cap_ratio"] >= DROP_BELOW]

    edges = [BRACKET_EDGES[b][0] for b in reversed(BRACKET_ORDER)] + [np.inf]
    labels = list(reversed(BRACKET_ORDER))
    snaps["bracket"] = pd.cut(snaps["cap_ratio"], bins=edges, labels=labels,
                              right=False, include_lowest=True).astype(object)
    snaps = snaps[snaps["bracket"].notna()]
    return snaps.sort_values(["snapshot_date", "cap_ratio"], ascending=[True, False]).reset_index(drop=True)


def bracket_membership(
    assignments: pd.DataFrame,
    index: pd.DatetimeIndex,
    bracket: str,
    columns: list[int] | None = None,
) -> pd.DataFrame:
    """Daily membership of one bracket, reconstituted at each snapshot.

    Membership on any day is decided by the most recent snapshot on or before
    that day, so no future market cap leaks in. Same convention as
    `pit_universe.pit_membership`, which this deliberately mirrors: the two
    universes differ only in how membership is defined, never in when it is
    known.
    """
    a = assignments[assignments["bracket"] == bracket]
    ids = sorted(int(c) for c in (columns if columns is not None
                                  else assignments["cmc_id"].unique()))
    index = pd.DatetimeIndex(index)
    values = np.zeros((len(index), len(ids)), dtype=bool)
    if a.empty or not ids:
        return pd.DataFrame(values, index=index, columns=ids)

    id_pos = {c: i for i, c in enumerate(ids)}
    by_date = {pd.Timestamp(d): np.array(sorted(id_pos[c] for c in set(g["cmc_id"].astype(int))
                                                if c in id_pos), dtype=int)
               for d, g in a.groupby("snapshot_date")}
    # Reconstitution dates come from ALL snapshots, not only those where this
    # bracket happens to be non-empty. A bracket that is empty on a snapshot
    # date is empty from that date, and must not carry the previous month's
    # members forward: that would keep a token in B0 for years after it fell
    # out of it.
    for d in assignments["snapshot_date"].unique():
        by_date.setdefault(pd.Timestamp(d), np.empty(0, dtype=int))
    dates = np.array(sorted(by_date))
    if len(dates) == 0:
        return pd.DataFrame(values, index=index, columns=ids)

    # searchsorted gives, for each day, the most recent snapshot at or before it
    pos = np.searchsorted(dates, index.to_numpy(), side="right") - 1
    for row, p in enumerate(pos):
        if p < 0:
            continue
        cols = by_date[pd.Timestamp(dates[p])]
        if len(cols):
            values[row, cols] = True
    return pd.DataFrame(values, index=index, columns=ids)


def bracket_dollar_bounds(snapshots: pd.DataFrame, eth_id: int = ETH_CMC_ID,
                          start: pd.Timestamp | None = BRACKET_START) -> pd.DataFrame:
    """Each bracket's dollar bounds at each month end. They move with ETH.

    One row per (snapshot_date, bracket) with ``lo_usd`` and ``hi_usd``. This is
    the table that stops a reader assuming an ETH-relative bracket is a fixed
    dollar band.
    """
    eth_mcap = eth_market_caps(snapshots, eth_id)
    if start is not None:
        eth_mcap = eth_mcap[eth_mcap.index >= pd.Timestamp(start)]
    rows = []
    for day, mcap in eth_mcap.items():
        for name in BRACKET_ORDER:
            lo, hi = BRACKET_EDGES[name]
            rows.append({
                "snapshot_date": day,
                "bracket": name,
                "eth_market_cap_usd": float(mcap),
                "lo_usd": float(lo) * float(mcap),
                "hi_usd": (float(hi) * float(mcap)) if np.isfinite(hi) else np.inf,
            })
    return pd.DataFrame(rows)


def rank_cutoff_binds(
    snapshots: pd.DataFrame,
    rank_hi: int,
    bracket: str = "B3",
    eth_id: int = ETH_CMC_ID,
    start: pd.Timestamp | None = BRACKET_START,
) -> pd.DataFrame:
    """Does the rank-`rank_hi` pull depth cut into `bracket`'s floor?

    The pull reaches rank `rank_hi`. If the market cap of the deepest token
    pulled is still **above** the bracket's floor, then tokens that belong in the
    bracket exist below the pull depth and are missing: the rank cutoff binds and
    the bracket is incomplete that month. If the deepest token pulled is already
    below the floor, the pull is deep enough and the bracket is complete.

    Two ways the cutoff can fail to bind, and they are different statements:
    the pull went deep enough to pass under the floor, or CoinMarketCap simply
    did not rank `rank_hi` tokens that month, in which case the binding
    constraint is CMC's own coverage and not the depth chosen here. ``n_ranked``
    separates them.

    Returns one row per snapshot date with the deepest rank actually returned,
    its market cap, the bracket's floor in dollars, and ``binds``.
    """
    snaps = snapshots.copy()
    snaps["snapshot_date"] = pd.to_datetime(snaps["snapshot_date"])
    eth_mcap = eth_market_caps(snaps, eth_id)
    if start is not None:
        snaps = snaps[snaps["snapshot_date"] >= pd.Timestamp(start)]
    lo = BRACKET_EDGES[bracket][0]

    rows = []
    for day, g in snaps.groupby("snapshot_date"):
        if day not in eth_mcap.index:
            continue
        g = g[pd.to_numeric(g["rank"], errors="coerce").notna()]
        g = g[g["rank"] <= rank_hi]
        n_ranked = int(g["rank"].max()) if len(g) else 0
        # The deepest token with a *usable* market cap. CMC prints zero or null
        # market caps in the tail of the early-year rankings, and taking those
        # at face value reports a $0 deepest cap, which would say the pull is
        # deep enough when in fact nothing is known down there.
        g = g[pd.to_numeric(g["market_cap_usd"], errors="coerce") > 0]
        if g.empty:
            continue
        deepest = g.loc[g["rank"].idxmax()]
        floor_usd = lo * float(eth_mcap[day])
        deepest_mcap = float(pd.to_numeric(deepest["market_cap_usd"], errors="coerce"))
        # The cutoff binds only if the pull actually ran out of depth: CMC
        # ranked at least `rank_hi` tokens AND the deepest one priced is still
        # above the bracket's floor, so members exist below the pull.
        exhausted_depth = n_ranked >= rank_hi
        rows.append({
            "snapshot_date": day,
            "n_ranked": n_ranked,
            "deepest_rank": int(deepest["rank"]),
            "deepest_market_cap_usd": deepest_mcap,
            "bracket": bracket,
            "floor_usd": floor_usd,
            "exhausted_depth": exhausted_depth,
            "binds": bool(exhausted_depth and np.isfinite(deepest_mcap)
                          and deepest_mcap > floor_usd),
        })
    return pd.DataFrame(rows)


def member_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    """Members per (snapshot_date, bracket), with the clustering flag."""
    counts = (assignments.groupby(["snapshot_date", "bracket"])
              .size().rename("n_members").reset_index())
    counts["clusterable"] = counts["n_members"] >= MIN_MEMBERS_FOR_CLUSTERING
    return counts
