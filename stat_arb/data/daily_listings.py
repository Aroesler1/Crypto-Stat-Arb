"""Panels, deaths and return series from the daily point-in-time listings.

`build_daily_listings.py` pulls one CoinMarketCap ranking per calendar day. This
module turns that stack into the things a backtest needs: wide close / volume /
market-cap panels, a universe table recording how each token left the sample,
and ETH-excess log returns with delisting returns imposed.

Death versus censoring
----------------------
This is the one judgement that has to be right, because getting it wrong is
either survivorship bias or a fabricated loss.

A token stops appearing in a depth-2000 daily listing for two quite different
reasons:

``death``     the project stopped, and CoinMarketCap now carries it as inactive
              or untracked. The position could not be exited at any price and
              the delisting rule applies.
``censored``  the token fell below rank 2000 and is *still alive today*. Nothing
              happened to it that a holder would recognise as a loss; it simply
              left the observable window. Applying a delisting return here would
              invent a -99% that never happened.

The two are separated by CMC's own listing-status map, which is a present-day
fact and therefore safe to use: a token CMC still carries as active did not die,
whatever its rank. Censored tokens get no terminal return, their positions carry
at zero return until the next reconstitution drops them, and the count is
reported rather than buried, because a censoring rate that is large relative to
the death rate would undermine the whole measurement.

A token that reappears after a gap is not treated as having died at the gap. Only
the final disappearance is classified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.data.pit_universe import (  # reuse one definition of each rule
    MAX_ABS_LOG_RETURN, RULE_ALIVE, RULE_LAST_PRICE, RULE_TOTAL_LOSS,
    TOTAL_LOSS_RESIDUAL, scrub_extreme_returns,
)

ETH_CMC_ID = 1027

# A token still ranked within this many days of the sample end is alive, not
# gone. Absorbs the occasional missing day at the end of the pull.
DEFAULT_STALE_DAYS = 7

# How the token left the sample, when it was not still there at the end.
EXIT_DEATH = "death"
EXIT_CENSORED = "censored_by_rank"
EXIT_NONE = "still_ranked"

# Minimum final-day dollar volume for a dying token's last quote to count as an
# achievable exit rather than a printed number.
#
# The older per-token panel tested `volume > 0`, because a dying coin's own
# history series showed volume drying up to nothing. In the daily listing panel
# that signal does not exist: CMC keeps printing a nonzero volume right up to
# the day it drops the coin from the ranking. Measured on 836 deaths in the
# 2015-2018 panel, the final-day volume of a dying token has a **median of
# $7.30**; 737 of 836 are under $1,000 and only 8 clear $1M. A `> 0` test
# therefore classifies every one of them as a clean exit at the last quote,
# which is the survivorship assumption this project exists to remove, smuggled
# back in through the delisting rule.
#
# $10k sits above the 90th percentile of that distribution ($1,773), so the
# exact level is not load-bearing: anything between roughly $2k and $1M
# classifies the same ~99% of deaths as total losses. `DELISTING_POLICIES`
# reports the sensitivity rather than asking the reader to trust the number.
MIN_EXIT_VOLUME_USD = 10_000.0

# How a death is treated. The choice is a judgement, so it is a parameter and
# all three are reported rather than one being chosen silently.
#   volume_floor  exit at the last quote only if that quote had enough volume to
#                 exit into; otherwise the position is a total loss. Default.
#   total_loss    every death takes the terminal shock. Most conservative.
#   last_price    every death exits at its final quote, taking no terminal loss.
#                 The naive treatment, kept so the others can be read against it.
DELISTING_POLICIES = ("volume_floor", "total_loss", "last_price")


def wide(panel: pd.DataFrame, field: str, index: pd.DatetimeIndex | None = None,
         ids: list[int] | None = None) -> pd.DataFrame:
    """Wide daily panel of `field`, columns keyed on cmc_id (int), sorted.

    Columns are sorted explicitly. The repo has been bitten once by column order
    following a Python set's iteration order, which made clustering results
    depend on the process hash seed (see `tests/test_loader.py`).
    """
    p = panel
    if ids is not None:
        p = p[p["cmc_id"].isin([int(i) for i in ids])]
    out = (p.pivot_table(index="date", columns="cmc_id", values=field, aggfunc="last")
             .sort_index())
    out.columns = [int(c) for c in out.columns]
    out = out.sort_index(axis=1)
    if index is not None:
        out = out.reindex(pd.DatetimeIndex(index))
    return out


def build_universe_table(
    panel: pd.DataFrame,
    dead_ids: set[int],
    window_end: pd.Timestamp,
    stale_days: int = DEFAULT_STALE_DAYS,
    total_loss_residual: float = TOTAL_LOSS_RESIDUAL,
    delisting_policy: str = "volume_floor",
    min_exit_volume_usd: float = MIN_EXIT_VOLUME_USD,
) -> pd.DataFrame:
    """One row per cmc_id: listing life, how it left, and its delisting rule.

    `dead_ids` are the ids CMC now reports inactive or untracked. See the module
    docstring for why that present-day fact is what separates a death from a
    token that merely fell out of the top 2000, and `DELISTING_POLICIES` for why
    the treatment of a death is a parameter.
    """
    if delisting_policy not in DELISTING_POLICIES:
        raise ValueError(f"delisting_policy must be one of {DELISTING_POLICIES}")
    window_end = pd.Timestamp(window_end)
    cutoff = window_end - pd.Timedelta(days=stale_days)

    p = panel[pd.to_numeric(panel["price_usd"], errors="coerce") > 0]
    g = p.groupby("cmc_id")
    agg = g.agg(
        symbol=("symbol", "last"),
        name=("name", "last"),
        slug=("slug", "last"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_obs=("date", "size"),
        best_rank=("rank", "min"),
    )
    last_rows = (p.sort_values("date").groupby("cmc_id").last()
                 [["price_usd", "volume_24h_usd", "market_cap_usd"]]
                 .rename(columns={"price_usd": "last_price",
                                  "volume_24h_usd": "last_volume_usd",
                                  "market_cap_usd": "last_market_cap_usd"}))
    tab = agg.join(last_rows).reset_index()
    tab["cmc_id"] = tab["cmc_id"].astype(int)

    still_ranked = tab["last_date"] >= cutoff
    is_dead = tab["cmc_id"].isin(dead_ids)

    exit_kind = np.where(still_ranked, EXIT_NONE,
                         np.where(is_dead, EXIT_DEATH, EXIT_CENSORED))
    tab["exit_kind"] = exit_kind

    # Delisting rule. Only a death gets one; a censored token left the window
    # alive and is not charged a terminal loss.
    vol = pd.to_numeric(tab["last_volume_usd"], errors="coerce").fillna(0.0)
    if delisting_policy == "total_loss":
        exits_cleanly = pd.Series(False, index=tab.index)
    elif delisting_policy == "last_price":
        exits_cleanly = pd.Series(True, index=tab.index)
    else:  # volume_floor: could the position actually have been sold that day?
        exits_cleanly = vol >= float(min_exit_volume_usd)
    rule = np.where(exit_kind != EXIT_DEATH, RULE_ALIVE,
                    np.where(exits_cleanly, RULE_LAST_PRICE, RULE_TOTAL_LOSS))
    tab["delisting_rule"] = rule
    tab["delisting_date"] = tab["last_date"].where(tab["exit_kind"] == EXIT_DEATH)
    tab["delisted"] = tab["exit_kind"] == EXIT_DEATH
    tab["censored"] = tab["exit_kind"] == EXIT_CENSORED
    tab["cmc_untracked"] = is_dead
    tab["total_loss_residual"] = np.where(tab["delisting_rule"] == RULE_TOTAL_LOSS,
                                          float(total_loss_residual), np.nan)
    tab["delisting_policy"] = delisting_policy
    return tab.sort_values("cmc_id").reset_index(drop=True)


def excess_log_returns(
    close: pd.DataFrame,
    reference_close: pd.Series,
    universe_table: pd.DataFrame,
    total_loss_residual: float = TOTAL_LOSS_RESIDUAL,
    max_abs_log_return: float | None = MAX_ABS_LOG_RETURN,
) -> tuple[pd.DataFrame, int]:
    """Daily log returns in excess of `reference_close`, delisting returns imposed.

    Mirrors `pit_universe.excess_log_returns` but takes the reference series
    explicitly, because the residualization ablation needs to run this against
    ETH, against BTC and against a value-weighted market without three copies of
    the function. Pass a series of ones for raw (non-excess) returns.

    Returns the panel and the number of scrubbed observations, so the count can
    be reported instead of silently absorbed.
    """
    ref = pd.to_numeric(reference_close, errors="coerce").reindex(close.index)
    ref_ret = np.log(ref / ref.shift(1)).fillna(0.0)

    prices = close.where(close > 0)
    ret = np.log(prices / prices.shift(1)).sub(ref_ret, axis=0)

    # Scrub artifacts BEFORE the delisting shock is written, so the deliberate
    # terminal return is never mistaken for one of them.
    ret, n_scrubbed = scrub_extreme_returns(ret, max_abs_log_return)

    shock = float(np.log(total_loss_residual))
    losers = universe_table[universe_table["delisting_rule"] == RULE_TOTAL_LOSS]
    for row in losers.itertuples(index=False):
        col = int(row.cmc_id)
        if col not in ret.columns or pd.isna(row.delisting_date):
            continue
        after = ret.index[ret.index > pd.Timestamp(row.delisting_date)]
        if len(after) == 0:
            continue
        ret.loc[after[0], col] = shock - float(ref_ret.get(after[0], 0.0) or 0.0)
    return ret, n_scrubbed


def value_weighted_market(close: pd.DataFrame, mcap: pd.DataFrame) -> pd.Series:
    """Value-weighted market log return, lagged weights.

    Weights are the previous day's market caps, so the day's own move cannot
    weight itself. Returned as a cumulative price index so it can be passed to
    `excess_log_returns` wherever ETH's close would go.
    """
    prices = close.where(close > 0)
    r = np.log(prices / prices.shift(1))
    w = mcap.reindex(index=r.index, columns=r.columns).shift(1)
    w = w.where(w > 0)
    r = r.where(np.isfinite(r))
    num = (r * w).sum(axis=1, min_count=1)
    den = w.where(r.notna()).sum(axis=1, min_count=1)
    mkt = (num / den).fillna(0.0)
    return np.exp(mkt.cumsum()).rename("vw_market")


def exit_summary(universe_table: pd.DataFrame) -> pd.DataFrame:
    """Counts of how tokens left the sample, by year of exit.

    The censored column is the honesty check on the whole panel: it is the count
    of tokens that left the observable window alive, for which no delisting
    return is imposed.
    """
    t = universe_table.copy()
    t["exit_year"] = pd.to_datetime(t["last_date"]).dt.year
    out = (t.pivot_table(index="exit_year", columns="exit_kind",
                         values="cmc_id", aggfunc="count")
             .fillna(0).astype(int))
    for col in (EXIT_DEATH, EXIT_CENSORED, EXIT_NONE):
        if col not in out.columns:
            out[col] = 0
    return out[[EXIT_DEATH, EXIT_CENSORED, EXIT_NONE]]


def validate_against_history(
    panel: pd.DataFrame,
    histories: dict[int, pd.DataFrame],
    tol: float = 0.02,
) -> pd.DataFrame:
    """Cross-check listing closes against the per-token history endpoint.

    The two endpoints are independent views of the same quote, so agreeing on the
    tokens where both exist is evidence the listing panel is what it claims to
    be. Returns one row per token with the overlap count and the fraction of days
    where the two disagree by more than `tol` in relative terms.
    """
    rows = []
    for cmc_id, hist in histories.items():
        if hist is None or hist.empty or "close" not in hist.columns:
            continue
        a = (panel[panel["cmc_id"] == int(cmc_id)]
             .set_index("date")["price_usd"].astype(float))
        b = hist.set_index("date")["close"].astype(float)
        j = pd.concat([a.rename("listing"), b.rename("history")], axis=1,
                      join="inner").dropna()
        j = j[(j["listing"] > 0) & (j["history"] > 0)]
        if len(j) < 10:
            continue
        rel = (j["listing"] - j["history"]).abs() / j["history"]
        rows.append({
            "cmc_id": int(cmc_id),
            "n_overlap": int(len(j)),
            "median_rel_diff": float(rel.median()),
            "frac_disagree": float((rel > tol).mean()),
        })
    return pd.DataFrame(rows).sort_values("frac_disagree", ascending=False).reset_index(drop=True)


def implied_supply(close: pd.DataFrame, mcap: pd.DataFrame) -> pd.DataFrame:
    """Circulating supply implied by market cap and price.

    CMC's listing quote carries price and market cap but not supply, and their
    ratio is the supply CMC used. That is exactly the quantity a redenomination
    moves, which is what makes the check below possible without a second source.
    """
    p = close.where(close > 0)
    m = mcap.reindex(index=p.index, columns=p.columns).where(lambda x: x > 0)
    return m / p


def redenomination_flags(
    close: pd.DataFrame,
    mcap: pd.DataFrame,
    min_abs_price_move: float = float(np.log(2)),
    max_abs_mcap_move: float = 0.10,
) -> pd.DataFrame:
    """Days where price jumped but market cap did not: a redenomination.

    A token that redenominates (a split, a reverse split, a migration to a new
    contract with a different supply) prints a large price move with an
    offsetting supply move, leaving market cap roughly unchanged. PUPS in the
    existing sample is the clean example: price /10.4, supply x9.4, market cap
    flat. That is not a return, and a mean-reversion book that treats it as one
    "earns" the reversal of a corporate action.

    A bad print behaves differently: price moves and market cap moves with it,
    because supply did not change. So the two are separable, and only the
    redenomination is identified here. `scrub_extreme_returns` catches the
    remaining artifacts by magnitude.

    Returns a boolean panel, True where the day's price move looks like a
    redenomination rather than a return.
    """
    p = close.where(close > 0)
    m = mcap.reindex(index=p.index, columns=p.columns).where(lambda x: x > 0)
    price_move = np.log(p / p.shift(1))
    mcap_move = np.log(m / m.shift(1))
    flag = (price_move.abs() >= float(min_abs_price_move)) & \
           (mcap_move.abs() <= float(max_abs_mcap_move))
    return flag.fillna(False)
