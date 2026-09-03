"""Derived point-in-time universe tables, and the delisting-return rule.

`cmc_pit` fetches; this module turns the fetch into the two things the backtest
needs: a universe table saying which `cmc_id` was tradable when, and a return
panel in which tokens that died actually lose money.

The delisting-return rule
-------------------------
A survivorship-free universe is worth nothing if a dead token is allowed to
leave the panel quietly, because dropping its final period is precisely the bias
you set out to remove. Every token is classified into exactly one of three
rules, and the rule that applied is recorded per token so a reader can audit it
rather than trust it:

``alive``       Prices run to the end of the window. Nothing is imposed.
``last_price``  Trading stops inside the window, but the final bar is a
                documented, positive close on a day with non-zero reported
                volume. The position exits at that close; the return into it is
                already in the series, so no extra return is applied.
``total_loss``  Trading stops inside the window with no documented exit: the
                final close is missing or non-positive, or the last bar reports
                zero volume, meaning the quote was not achievable. A terminal
                return of -100% is applied on the day after the last bar.

One honest wrinkle. This book's panel is in **log** returns, and a -100% simple
return is ``log(0) = -inf``. The terminal shock is therefore applied as a
residual value (default 1% of the last close, i.e. -99%, ``log(0.01) = -4.61``)
rather than a true zero. `TOTAL_LOSS_RESIDUAL` is the knob; it is deliberately
conservative, biasing the measured survivorship effect *down*, and the value
used is written into the universe table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RULE_ALIVE = "alive"
RULE_LAST_PRICE = "last_price"
RULE_TOTAL_LOSS = "total_loss"

# Fraction of the last close assumed recoverable in a total-loss delisting.
# Not zero, because the return panel is in log space; see the module docstring.
TOTAL_LOSS_RESIDUAL = 0.01

# How close to the window end a final bar has to sit before the token counts as
# still trading rather than delisted. CMC bars can lag a day or two.
DEFAULT_STALE_DAYS = 7

# Daily |log return| beyond which a printed move is treated as a data artifact
# rather than a return, and dropped. log(5) is a same-day +400% / -80%.
#
# This is not cosmetic. A survivorship-free rank 150-500 universe is full of
# micro-caps whose CMC series contain redenominations and outright bad prints:
# vBNB prints 10.13 -> 812.27 -> 14.29 on flat supply, BTTOLD drops to 7.4e-07
# for one day and comes back, CAIR steps 1.0e-04 -> 0.79 and stays there. None
# of those are tradable returns, and a mean-reversion book "earns" the reversal
# of every one of them. Dropping the observation is the conservative choice: it
# removes the most profitable-looking reversals in the panel, so it biases the
# measured edge down, not up.
#
# Supply cannot separate these on its own. A redenomination shows an offsetting
# circulating-supply move (PUPS: price /10.4 as supply x9.4, market cap flat),
# but the bad prints above leave supply untouched, so a supply test alone misses
# them. The threshold catches both.
MAX_ABS_LOG_RETURN = float(np.log(5))


def classify_delisting(
    history: pd.DataFrame,
    window_end: pd.Timestamp,
    stale_days: int = DEFAULT_STALE_DAYS,
) -> tuple[str, pd.Timestamp | None]:
    """Which delisting rule applies to one token's daily history.

    Returns ``(rule, last_trade_date)``. `last_trade_date` is None for a token
    that never stopped, and otherwise the date of its final usable bar.
    """
    if history.empty:
        return RULE_TOTAL_LOSS, None

    hist = history.dropna(subset=["close"])
    hist = hist[hist["close"] > 0]
    if hist.empty:
        return RULE_TOTAL_LOSS, None

    hist = hist.sort_values("date")
    last_row = hist.iloc[-1]
    last_date = pd.Timestamp(last_row["date"])

    # A printed non-positive close after the last real price is a documented
    # move to zero, not a gap in the data. Checked before the staleness test so
    # a token that prints zeros right up to the window end is not called alive.
    tail = history[pd.to_datetime(history["date"]) > last_date]
    tail_closes = tail["close"].dropna() if not tail.empty else pd.Series(dtype=float)
    if (tail_closes <= 0).any():
        return RULE_TOTAL_LOSS, last_date

    if last_date >= pd.Timestamp(window_end) - pd.Timedelta(days=stale_days):
        return RULE_ALIVE, None

    last_volume = last_row.get("volume_usd", np.nan)
    if pd.notna(last_volume) and float(last_volume) > 0:
        return RULE_LAST_PRICE, last_date
    return RULE_TOTAL_LOSS, last_date


def build_universe_table(
    histories: dict[int, pd.DataFrame],
    snapshots: pd.DataFrame,
    dead_ids: set[int],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    stale_days: int = DEFAULT_STALE_DAYS,
    total_loss_residual: float = TOTAL_LOSS_RESIDUAL,
) -> pd.DataFrame:
    """One row per `cmc_id`: identity, listing life, and delisting treatment.

    `dead_ids` are the ids CMC now reports as inactive or untracked. A token
    counts as `delisted` if CMC has stopped carrying it *or* its prices stop
    inside the window; either way a present-day snapshot cannot see it, which is
    the bias being measured.
    """
    latest = (
        snapshots.sort_values("snapshot_date")
        .groupby("cmc_id")
        .agg(symbol=("symbol", "last"), name=("name", "last"), slug=("slug", "last"),
             best_rank=("rank", "min"), months_in_band=("rank", "size"))
    )

    rows = []
    for cmc_id, hist in histories.items():
        rule, last_trade = classify_delisting(hist, window_end, stale_days)
        usable = hist.dropna(subset=["close"]) if not hist.empty else hist
        usable = usable[usable["close"] > 0] if not usable.empty else usable

        first_date = pd.Timestamp(usable["date"].min()) if not usable.empty else pd.NaT
        last_date = pd.Timestamp(usable["date"].max()) if not usable.empty else pd.NaT
        stopped = pd.notna(last_date) and last_date < pd.Timestamp(window_end) - pd.Timedelta(days=stale_days)

        meta = latest.loc[cmc_id] if cmc_id in latest.index else None
        rows.append({
            "cmc_id": int(cmc_id),
            "symbol": (meta["symbol"] if meta is not None else None),
            "name": (meta["name"] if meta is not None else None),
            "slug": (meta["slug"] if meta is not None else None),
            "first_date": first_date,
            "last_date": last_date,
            "delisted": bool(cmc_id in dead_ids or stopped),
            "cmc_untracked": bool(cmc_id in dead_ids),
            "prices_stop_in_window": bool(stopped),
            "delisting_rule": rule,
            "delisting_date": last_trade,
            "total_loss_residual": float(total_loss_residual) if rule == RULE_TOTAL_LOSS else np.nan,
            "n_obs": int(len(usable)),
            "best_rank": (int(meta["best_rank"]) if meta is not None and pd.notna(meta["best_rank"]) else np.nan),
            "months_in_band": (int(meta["months_in_band"]) if meta is not None else 0),
            "window_start": pd.Timestamp(window_start),
            "window_end": pd.Timestamp(window_end),
        })

    return pd.DataFrame(rows).sort_values("cmc_id").reset_index(drop=True)


def panel(histories: dict[int, pd.DataFrame], field: str,
          index: pd.DatetimeIndex) -> pd.DataFrame:
    """Wide daily panel of `field`, columns keyed on cmc_id (int), not symbol."""
    cols = {}
    for cmc_id, hist in histories.items():
        if hist.empty or field not in hist.columns:
            continue
        s = hist.set_index("date")[field]
        s = s[~s.index.duplicated(keep="last")]
        cols[int(cmc_id)] = s
    if not cols:
        return pd.DataFrame(index=index)
    out = pd.DataFrame(cols)
    out.index = pd.DatetimeIndex(out.index)
    return out.reindex(index).sort_index(axis=1)


def scrub_extreme_returns(
    returns: pd.DataFrame,
    max_abs_log_return: float | None = MAX_ABS_LOG_RETURN,
) -> tuple[pd.DataFrame, int]:
    """Drop daily log returns too large to be a return. See MAX_ABS_LOG_RETURN.

    Returns the cleaned panel and the number of observations dropped, so the
    count can be reported rather than silently absorbed. The price level is left
    alone, so the following day's return is measured off the new level the way a
    split adjustment would leave it.
    """
    if max_abs_log_return is None:
        return returns, 0
    bad = returns.abs() > float(max_abs_log_return)
    n = int(bad.to_numpy().sum())
    if n == 0:
        return returns, 0
    return returns.mask(bad), n


def excess_log_returns(
    close: pd.DataFrame,
    eth_close: pd.Series,
    universe_table: pd.DataFrame,
    total_loss_residual: float = TOTAL_LOSS_RESIDUAL,
    max_abs_log_return: float | None = MAX_ABS_LOG_RETURN,
) -> pd.DataFrame:
    """ETH-excess daily log returns, with delisting returns imposed.

    Matches the convention of the committed `excess_log_returns.csv`:
    ``log(p_t / p_{t-1}) - log(eth_t / eth_{t-1})``.

    Tokens classified `total_loss` get their terminal shock written on the day
    after their last bar. Tokens classified `last_price` get nothing extra: they
    exited at a real traded price and the return into it is already here.

    Price artifacts are scrubbed first (see `scrub_extreme_returns`); pass
    max_abs_log_return=None to keep the raw panel.
    """
    eth = pd.to_numeric(eth_close, errors="coerce").reindex(close.index)
    eth_ret = np.log(eth / eth.shift(1))

    prices = close.where(close > 0)
    ret = np.log(prices / prices.shift(1)).sub(eth_ret, axis=0)

    # Scrub artifacts BEFORE the delisting shock is written, so the deliberate
    # -99% terminal return is never mistaken for one of them.
    ret, _ = scrub_extreme_returns(ret, max_abs_log_return)

    shock = float(np.log(total_loss_residual))
    losers = universe_table[universe_table["delisting_rule"] == RULE_TOTAL_LOSS]
    for row in losers.itertuples(index=False):
        col = int(row.cmc_id)
        if col not in ret.columns or pd.isna(row.delisting_date):
            continue
        after = ret.index[ret.index > pd.Timestamp(row.delisting_date)]
        if len(after) == 0:
            continue
        shock_date = after[0]
        eth_on_shock = eth_ret.get(shock_date, 0.0)
        ret.loc[shock_date, col] = shock - (0.0 if pd.isna(eth_on_shock) else eth_on_shock)

    return ret


def pit_membership(
    snapshots: pd.DataFrame,
    index: pd.DatetimeIndex,
    rank_lo: int,
    rank_hi: int,
    columns: list[int] | None = None,
) -> pd.DataFrame:
    """Rank-band membership as it stood, reconstituted at each snapshot date.

    On any day, membership is decided by the most recent snapshot on or before
    that day, so no future ranking leaks in.
    """
    snaps = snapshots.copy()
    snaps["snapshot_date"] = pd.to_datetime(snaps["snapshot_date"])
    in_band = snaps[(snaps["rank"] >= rank_lo) & (snaps["rank"] <= rank_hi)]

    ids = sorted(columns) if columns is not None else sorted(snaps["cmc_id"].unique())
    wide = pd.DataFrame(False, index=index, columns=[int(c) for c in ids])

    dates = sorted(in_band["snapshot_date"].unique())
    by_date = {d: set(g["cmc_id"]) for d, g in in_band.groupby("snapshot_date")}
    for i, day in enumerate(index):
        prior = [d for d in dates if d <= day]
        if not prior:
            continue
        members = by_date.get(prior[-1], set())
        hit = [c for c in wide.columns if c in members]
        if hit:
            wide.loc[day, hit] = True
    return wide


def drop_delisted(membership: pd.DataFrame, universe_table: pd.DataFrame) -> pd.DataFrame:
    """The survivor-only counterfactual: the same rule, blind to dead tokens.

    A present-day snapshot can only contain tokens that still exist today. This
    removes exactly those and changes nothing else, so the difference between
    the two books is survivorship and nothing but survivorship.
    """
    dead = set(universe_table.loc[universe_table["delisted"], "cmc_id"].astype(int))
    out = membership.copy()
    for col in out.columns:
        if int(col) in dead:
            out[col] = False
    return out
