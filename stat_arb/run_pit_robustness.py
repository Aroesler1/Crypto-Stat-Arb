"""Liquidity tiers on a point-in-time universe, against the survivor-only book.

`run_robustness.py` could only *bound* the survivorship problem, because the
committed universe is a present-day CoinMarketCap snapshot and a snapshot cannot
contain a token that died. This measures it instead, on a universe rebuilt from
CMC's point-in-time rankings including coins that no longer exist.

Identification
--------------
Three books are built from the same pull, over the same window, with the same
rank rule, the same filters and the same engine, so each step isolates one
selection:

``pit``           every token in the rank band at each reconstitution, dead ones
                  included, with delisting returns imposed.
``survivor-only`` the same, minus tokens that are dead today. A present-day
                  snapshot cannot contain them. pit -> survivor-only is
                  SURVIVORSHIP and nothing else.
``snapshot``      survivor-only, further restricted to tokens with an unbroken
                  price history across the whole window. This reproduces the
                  committed universe, whose 174 tokens all have exactly 100%
                  coverage -- a filter you can only apply once you know how the
                  sample ended. survivor-only -> snapshot is that LOOK-AHEAD
                  COMPLETE-HISTORY filter.

Reporting them separately matters because the two effects are not the same size
and point in different directions.

Both volume conventions are reported. The committed data's volume column is
already USD, but `UniverseManager` has always filtered on `volume * price`, so
the published tier labels ("$50k/day") are not notional traded. `usd` is the
corrected filter; `legacy` reproduces the published convention so the new table
can be read against the old one. See `UniverseManager.volume_in_usd`.

Usage:
    python stat_arb/build_pit_universe.py     # once, pulls the data
    python stat_arb/run_pit_robustness.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.data import pit_universe as P  # noqa: E402
from stat_arb.data.loader import EXCLUDED_TOKENS  # noqa: E402
from stat_arb.data.universe import UniverseManager  # noqa: E402
from stat_arb.run_phase3 import run_phase3_config  # noqa: E402
from stat_arb.build_pit_universe import (  # noqa: E402
    DEFAULT_END, DEFAULT_RANK_HI, DEFAULT_RANK_LO, DEFAULT_START, ETH_CMC_ID,
)

PERIODS_PER_YEAR = 365
# Best net configuration from the phase-3 grid, held fixed so the only thing
# moving between columns is the universe.
BEST_BAND = 0.02
BEST_FREQ = 3

# Published survivorship magnitudes (Ammann, Burdorf, Liebi, Stoeckl,
# SSRN 4287573; 3,904 coins, 2014-2021).
PUBLISHED_EW_BIAS = 0.6219
PUBLISHED_VW_BIAS = 0.0093

VOLUME_TIERS = {
    "baseline (>=$50k/day)": 50_000,
    "liquid (>=$1M/day)": 1_000_000,
    "very liquid (>=$5M/day)": 5_000_000,
}

# Coverage a token needs before the snapshot book will hold it. The committed
# universe's 174 tokens all sit at exactly 1.0.
COMPLETE_HISTORY_COVERAGE = 0.99


def complete_history_ids(close: pd.DataFrame, min_coverage: float) -> set[int]:
    """Tokens priced on essentially every day of the window."""
    coverage = close.notna().mean()
    return {int(c) for c in coverage[coverage >= min_coverage].index}


def restrict(membership: pd.DataFrame, keep: set[int]) -> pd.DataFrame:
    out = membership.copy()
    for col in out.columns:
        if int(col) not in keep:
            out[col] = False
    return out


def buy_and_hold_annualised(mask, simple_returns, weights=None) -> float:
    """Annualised return of a monthly-reconstituted buy-and-hold book.

    Buy-and-hold WITHIN each reconstitution period, not daily rebalanced. On a
    micro-cap panel the difference is not cosmetic: daily equal-weight
    rebalancing harvests the volatility of assets quoted at 1e-07 and reports
    four-figure annualised returns that no one could trade. This is the
    construction Ammann, Burdorf, Liebi and Stoeckl measure, so it is the one
    that can be set against their 62.19% equal-weighted figure.
    """
    r = simple_returns
    m = mask.reindex(index=r.index, columns=r.columns, fill_value=False)
    growth = []
    for _, idx in r.groupby(r.index.to_period("M")).groups.items():
        idx = pd.DatetimeIndex(sorted(idx))
        held = [c for c in r.columns if bool(m.loc[idx[0], c])]
        if not held:
            continue
        period = (1.0 + r.loc[idx, held].fillna(0.0)).prod()   # hold, do not rebalance
        if weights is None:
            growth.append(float(period.mean()))
        else:
            w = weights.loc[idx[0], held].astype(float)
            w = w.where(w > 0).fillna(0.0)
            growth.append(float((period * w).sum() / w.sum()) if w.sum() > 0
                          else float(period.mean()))
    if not growth:
        return float("nan")
    return float(np.prod(growth) ** (12.0 / len(growth)) - 1.0)


def paired_delta(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Annualised mean difference of two daily return series, and its standard
    error. Paired on date, because both books trade the same calendar."""
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return float("nan"), float("nan")
    d = joined.iloc[:, 0] - joined.iloc[:, 1]
    return (float(d.mean()) * PERIODS_PER_YEAR,
            float(d.std(ddof=1)) / np.sqrt(len(d)) * PERIODS_PER_YEAR)


def annualised_stats(returns: pd.Series) -> dict:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "n": len(r)}
    ann_return = float(r.mean()) * PERIODS_PER_YEAR
    ann_vol = float(r.std(ddof=1)) * np.sqrt(PERIODS_PER_YEAR)
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol > 0 else np.nan,
        "n": len(r),
    }


def load_pit(data_dir: Path, start: pd.Timestamp, end: pd.Timestamp):
    """Panels and tables from the derived parquet, keyed on cmc_id throughout."""
    table = pd.read_parquet(data_dir / "universe_pit.parquet")
    ohlcv = pd.read_parquet(data_dir / "universe_pit_ohlcv.parquet")
    ranks = pd.read_parquet(data_dir / "universe_pit_ranks.parquet")

    eth = (ohlcv[ohlcv["cmc_id"] == ETH_CMC_ID]
           .set_index("date")[["close", "volume_usd"]]
           .sort_index())
    index = pd.date_range(start, end, freq="D")

    # Drop stablecoins and wrapped assets, matching the committed loader. Symbol
    # is the right key for this one job: the exclusion list is a list of tickers.
    keep = table[~table["symbol"].fillna("").str.upper().isin(EXCLUDED_TOKENS)]
    keep_ids = set(keep["cmc_id"].astype(int)) - {ETH_CMC_ID}

    hist = {
        int(cid): g.reset_index(drop=True)
        for cid, g in ohlcv[ohlcv["cmc_id"].isin(keep_ids)].groupby("cmc_id")
    }
    close = P.panel(hist, "close", index)
    volume = P.panel(hist, "volume_usd", index)
    table = keep[keep["cmc_id"].astype(int).isin(close.columns)].reset_index(drop=True)
    return table, ranks, close, volume, eth.reindex(index)


def make_engine_inputs(close, volume, eth, table):
    """Adapt the cmc_id-keyed panels to the column names the engine expects.

    `run_phase3_config` splits return columns on a `_returns` suffix, so the
    identifier carried through the backtest is `"<cmc_id>_returns"`. It is still
    the cmc_id: no step of this pipeline ever joins on a symbol.
    """
    returns = P.excess_log_returns(close, eth["close"], table)
    returns.columns = [f"{int(c)}_returns" for c in returns.columns]
    prices = close.copy()
    prices.columns = [str(int(c)) for c in prices.columns]
    volumes = volume.copy()
    volumes.columns = [str(int(c)) for c in volumes.columns]
    eth_data = pd.DataFrame({"close": eth["close"], "volume": eth["volume_usd"]})
    return returns, prices, volumes, eth_data


def run_tier(returns, prices, volumes, eth_data, band_mask, min_volume_usd,
             volume_in_usd) -> dict | None:
    univ = UniverseManager(
        mcap_percentile_low=0.0,
        mcap_percentile_high=1.0,
        min_volume_usd=min_volume_usd,
        min_history_days=60,
        volume_in_usd=volume_in_usd,
    )
    filters = univ.get_universe_membership(prices, volumes, eth_data, returns)
    mask = filters & band_mask.reindex(index=filters.index, columns=filters.columns,
                                       fill_value=False)
    members = float(mask.sum(axis=1).mean()) if not mask.empty else 0.0

    result = run_phase3_config(returns, mask, weight_band=BEST_BAND,
                               trade_frequency_days=BEST_FREQ)
    if result is None:
        return None

    stats = annualised_stats(result["net_50"])
    stats["avg_members"] = members
    stats["gross_sharpe"] = annualised_stats(result["gross_returns"])["sharpe"]
    stats["avg_turnover"] = float(result["turnover"].mean())
    stats["breakeven_cost_bps"] = result["breakeven"]
    stats["net_50"] = result["net_50"]
    stats["mask"] = mask
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--rank-lo", type=int, default=DEFAULT_RANK_LO)
    ap.add_argument("--rank-hi", type=int, default=DEFAULT_RANK_HI)
    ap.add_argument("--conventions", default="usd,legacy",
                    help="volume filter conventions to run")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    if not (data_dir / "universe_pit.parquet").exists():
        print("data/universe_pit.parquet not found; run stat_arb/build_pit_universe.py first")
        return 1

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    table, ranks, close, volume, eth = load_pit(data_dir, start, end)
    returns, prices, volumes, eth_data = make_engine_inputs(close, volume, eth, table)

    ids = [int(c) for c in close.columns]
    pit_band = P.pit_membership(ranks, close.index, args.rank_lo, args.rank_hi, columns=ids)
    survivor_band = P.drop_delisted(pit_band, table)
    full_history = complete_history_ids(close, COMPLETE_HISTORY_COVERAGE)
    snapshot_band = restrict(survivor_band, full_history)

    n_dead = int(table["delisted"].sum())
    print(f"PIT universe: {len(table)} cmc_ids over {start.date()}..{end.date()}, "
          f"{n_dead} delisted ({n_dead / max(len(table), 1):.1%})")
    print(f"delisting rules: {table['delisting_rule'].value_counts().to_dict()}")
    print(f"complete-history tokens (>={COMPLETE_HISTORY_COVERAGE:.0%} coverage): "
          f"{len(full_history)} of {len(ids)}")

    # universe-level buy-and-hold, the measure the published figures are for
    simple = close.where(close > 0).pct_change()
    scrubbed, n_scrubbed = P.scrub_extreme_returns(
        np.log1p(simple.where(simple > -1)), P.MAX_ABS_LOG_RETURN)
    simple = np.expm1(scrubbed)
    for row in table[table["delisting_rule"] == P.RULE_TOTAL_LOSS].itertuples(index=False):
        cid = int(row.cmc_id)
        if cid not in simple.columns or pd.isna(row.delisting_date):
            continue
        after = simple.index[simple.index > pd.Timestamp(row.delisting_date)]
        if len(after):
            simple.loc[after[0], cid] = P.TOTAL_LOSS_RESIDUAL - 1.0
    print(f"price artifacts scrubbed (|log r| > log 5): {n_scrubbed:,} daily observations")

    mcap = P.panel(
        {int(cid): g.reset_index(drop=True)
         for cid, g in pd.read_parquet(data_dir / "universe_pit_ohlcv.parquet").groupby("cmc_id")},
        "market_cap_usd", close.index).reindex(columns=simple.columns)

    bh_books = {"pit": pit_band, "survivor-only": survivor_band, "snapshot": snapshot_band}
    print("\n=== universe-level buy-and-hold, monthly reconstitution ===")
    bh = {}
    for label, mask in bh_books.items():
        bh[label] = (buy_and_hold_annualised(mask, simple),
                     buy_and_hold_annualised(mask, simple, mcap))
        print(f"  {label:<14s} EW {bh[label][0]:+8.2%}   VW {bh[label][1]:+8.2%}   "
              f"avg members {mask.sum(axis=1).mean():6.1f}")
    print(f"  survivorship bias (survivor-only minus pit): "
          f"EW {bh['survivor-only'][0] - bh['pit'][0]:+.2%}, "
          f"VW {bh['survivor-only'][1] - bh['pit'][1]:+.2%}")
    print(f"  complete-history bias (snapshot minus survivor-only): "
          f"EW {bh['snapshot'][0] - bh['survivor-only'][0]:+.2%}, "
          f"VW {bh['snapshot'][1] - bh['survivor-only'][1]:+.2%}")
    print(f"  published (Ammann et al., 3,904 coins, 2014-2021): "
          f"EW {PUBLISHED_EW_BIAS:+.2%}, VW {PUBLISHED_VW_BIAS:+.2%}")

    for band in (pit_band, survivor_band, snapshot_band):
        band.columns = [str(c) for c in band.columns]
    books = {"pit": pit_band, "survivor-only": survivor_band, "snapshot": snapshot_band}
    rows, series = [], {}
    for convention in [c.strip() for c in args.conventions.split(",") if c.strip()]:
        volume_in_usd = convention == "usd"
        for book, band_mask in books.items():
            for tier, floor in VOLUME_TIERS.items():
                print(f"running {convention}/{book}/{tier} ...", flush=True)
                stats = run_tier(returns, prices, volumes, eth_data, band_mask,
                                 floor, volume_in_usd)
                if stats is None:
                    print("  produced no positions, skipped")
                    continue
                key = (convention, book, tier)
                series[key] = stats.pop("net_50")
                mask = stats.pop("mask")
                dead_ids = set(table.loc[table["delisted"], "cmc_id"].astype(int))
                held = [c for c in mask.columns if mask[c].any()]
                stats["dead_members"] = sum(1 for c in held if int(c) in dead_ids)
                stats.update(convention=convention, book=book, tier=tier)
                rows.append(stats)

    if not rows:
        print("no configuration produced a result")
        return 1

    out = pd.DataFrame(rows)
    out_dir = root / "stat_arb" / "reporting" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "pit_vs_survivor.csv", index=False)

    for convention in out["convention"].unique():
        sub = out[out["convention"] == convention]
        wide = sub.pivot(index="tier", columns="book",
                         values=["avg_members", "gross_sharpe", "sharpe",
                                 "breakeven_cost_bps", "ann_return", "dead_members"])
        print(f"\n=== volume convention: {convention} "
              f"({'notional traded' if convention == 'usd' else 'legacy volume x price'}) ===")
        print(f"config: band {BEST_BAND:.0%}, rebalance {BEST_FREQ}d, net of 50bps")
        print(wide.to_string(float_format=lambda v: f"{v:0.3f}"))

        print("\nrealised bias in the strategy's own net returns, annualised "
              "(+/- 1 s.e., paired by date):")
        for label, (rich, poor) in (("survivorship (survivor-only - pit)",
                                     ("survivor-only", "pit")),
                                    ("complete history (snapshot - survivor-only)",
                                     ("snapshot", "survivor-only"))):
            print(f"  {label}")
            for tier in VOLUME_TIERS:
                a, b = series.get((convention, rich, tier)), series.get((convention, poor, tier))
                if a is None or b is None:
                    continue
                delta, se = paired_delta(a, b)
                verdict = "" if abs(delta) > 2 * se else "   (not distinguishable from zero)"
                print(f"    {tier:<26s} {delta:+7.2%} +/- {se:.2%}{verdict}")
        print(f"  published buy-and-hold reference: EW {PUBLISHED_EW_BIAS:.2%}, "
              f"VW {PUBLISHED_VW_BIAS:.2%}")

    print(f"\nsaved -> {out_dir / 'pit_vs_survivor.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
