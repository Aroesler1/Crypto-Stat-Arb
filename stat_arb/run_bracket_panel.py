"""Step 0: build the ETH-relative bracket panel and report what is in it.

Reads the daily point-in-time listings (`build_daily_listings.py`), assigns every
token to an ETH-relative bracket at each month end (`data/brackets.py`), and
writes the derived tables the rest of the project runs on. Everything printed
here is a description of the panel, not a result about the strategy; the
strategy questions start at Step 1.

What it writes into `--out-dir`
-------------------------------
``bracket_universe.parquet``       one row per cmc_id: identity, listing life,
                                   how it left the sample, delisting rule
``bracket_assignments.parquet``    (month end, cmc_id) -> cap_ratio, bracket
``bracket_returns_<B>.parquet``    ETH-excess daily log returns per bracket
``bracket_membership_<B>.parquet`` daily membership, the three survivorship
                                   treatments stacked
``bracket_dollar_bounds.parquet``  each bracket's dollar bounds per month end
``bracket_shortability.parquet``   which venues list a perpetual per token

Rebuild from a clone (about 70 minutes of pulling, then a minute of compute):

    python stat_arb/build_daily_listings.py --start 2015-07-01 --end 2025-06-30 \
        --depth 2000 --workers 2 --sleep 0.3
    python stat_arb/run_bracket_panel.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.build_pit_universe import pull_dead_map  # noqa: E402
from stat_arb.data import brackets as B  # noqa: E402
from stat_arb.data import daily_listings as D  # noqa: E402
from stat_arb.data import perps as PERPS  # noqa: E402
from stat_arb.data import pit_universe as P  # noqa: E402
from stat_arb.data.loader import is_derivative_or_stable  # noqa: E402

BTC_CMC_ID = 1
RANK_DEPTH = 2000

# The named legacy universe. Kept so every published README number reproduces
# against the same code path the brackets use.
LEGACY_RANK_LO, LEGACY_RANK_HI = 150, 500
# The academic universe for the Step 5 factor table.
ACADEMIC_RANK_HI, ACADEMIC_MIN_MCAP = 1000, 1_000_000

SURVIVORSHIP_TREATMENTS = ("pit", "survivor-only", "snapshot")
COMPLETE_HISTORY_COVERAGE = 0.99


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Reconstitution dates: the last calendar day of each month in the panel."""
    return pd.DatetimeIndex(sorted(pd.Series(index).groupby(index.to_period("M")).max()))


def monthly_snapshots(panel: pd.DataFrame) -> pd.DataFrame:
    """The daily panel sampled at month ends, in the shape `brackets` expects."""
    ends = set(month_ends(pd.DatetimeIndex(panel["date"].unique())))
    snaps = panel[panel["date"].isin(ends)].copy()
    return snaps.rename(columns={"date": "snapshot_date",
                                 "volume_24h_usd": "volume_24h_usd"})


def restrict(membership: pd.DataFrame, keep: set[int]) -> pd.DataFrame:
    out = membership.copy()
    for col in out.columns:
        if int(col) not in keep:
            out[col] = False
    return out


def survivorship_books(membership: pd.DataFrame, table: pd.DataFrame,
                       close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The three universes that differ only in what they are allowed to see.

    Identical to `run_pit_robustness`'s construction, so a bracket result can be
    read against the published rank 150-500 one:

    ``pit``           everything in the bracket, dead tokens included
    ``survivor-only`` minus tokens that are dead today. A present-day snapshot
                      cannot contain them, so pit -> survivor-only is
                      survivorship and nothing else
    ``snapshot``      survivor-only, further restricted to tokens priced on
                      essentially every day of the window. That filter can only
                      be applied once you know how the sample ended
    """
    dead = set(table.loc[table["delisted"], "cmc_id"].astype(int))
    survivor = membership.copy()
    for col in survivor.columns:
        if int(col) in dead:
            survivor[col] = False
    coverage = close.notna().mean()
    complete = {int(c) for c in coverage[coverage >= COMPLETE_HISTORY_COVERAGE].index}
    return {"pit": membership, "survivor-only": survivor,
            "snapshot": restrict(survivor, complete)}


def per_year_table(assignments: pd.DataFrame, table: pd.DataFrame,
                   membership: dict[str, pd.DataFrame], short: pd.DataFrame,
                   bounds: pd.DataFrame) -> pd.DataFrame:
    """Per bracket per year: tokens, token-days, deaths, bounds, perp share."""
    sym = table.set_index("cmc_id")["symbol"].to_dict()
    shortable = set(short.loc[short["shortable"], "symbol"])
    exit_year = (table.assign(y=pd.to_datetime(table["last_date"]).dt.year)
                 .set_index("cmc_id"))

    rows = []
    for bracket in B.BRACKET_ORDER:
        a = assignments[assignments["bracket"] == bracket]
        if a.empty:
            continue
        m = membership[bracket]["pit"]
        # Only days that have a reconstitution behind them. Before the first
        # month end nobody is a member, and reporting that as "min 0" reads as
        # an empty bracket rather than a panel that has not started.
        first = assignments["snapshot_date"].min()
        m = m.loc[m.index >= first]
        for year, g in a.groupby(a["snapshot_date"].dt.year):
            ids = set(g["cmc_id"].astype(int))
            days = m.loc[m.index.year == year]
            deaths = sum(
                1 for c in ids
                if c in exit_year.index and bool(exit_year.loc[c, "delisted"])
                and exit_year.loc[c, "y"] == year)
            n_short = sum(1 for c in ids if sym.get(c) in shortable)
            bb = bounds[(bounds["bracket"] == bracket)
                        & (bounds["snapshot_date"].dt.year == year)]
            rows.append({
                "bracket": bracket,
                "year": int(year),
                "tokens": len(ids),
                "token_days": int(days.to_numpy().sum()),
                "deaths": deaths,
                "avg_members": float(days.sum(axis=1).mean()) if len(days) else np.nan,
                "min_members": int(days.sum(axis=1).min()) if len(days) else 0,
                "lo_usd": float(bb["lo_usd"].mean()) if len(bb) else np.nan,
                "hi_usd": float(bb["hi_usd"].replace(np.inf, np.nan).mean()) if len(bb) else np.nan,
                "perp_share": n_short / len(ids) if ids else np.nan,
            })
    return pd.DataFrame(rows)


def fmt_usd(v: float) -> str:
    if not np.isfinite(v):
        return "-"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(v) >= div:
            return f"${v / div:.1f}{unit}"
    return f"${v:.0f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=None, help="pit_daily_listings.parquet")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--start", default=str(B.BRACKET_START.date()))
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--no-perps", action="store_true",
                    help="skip the perpetual-venue fetch (offline runs)")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    panel_path = Path(args.panel) if args.panel else root / "data" / "pit_daily_listings.parquet"
    out_dir = Path(args.out_dir) if args.out_dir else root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not panel_path.exists():
        print(f"{panel_path} not found; run stat_arb/build_daily_listings.py first")
        return 1

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    print(f"loading {panel_path.name} ...", flush=True)
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[(panel["date"] >= start - pd.Timedelta(days=40)) & (panel["date"] <= end)]
    index = pd.DatetimeIndex(sorted(panel["date"].unique()))
    print(f"  {len(panel):,} rows, {panel['cmc_id'].nunique():,} cmc_ids, "
          f"{index.min().date()}..{index.max().date()}, {len(index):,} days")

    # --- exclusions ---------------------------------------------------------
    meta = (panel.sort_values("date").groupby("cmc_id")[["symbol", "name"]].last())
    excluded = {int(c) for c, r in meta.iterrows()
                if is_derivative_or_stable(r["symbol"], r["name"])}
    print(f"  excluded as stable/wrapped/staked/bridged: {len(excluded):,} cmc_ids")
    keep = panel[~panel["cmc_id"].isin(excluded | {B.ETH_CMC_ID})]

    # --- deaths versus censoring -------------------------------------------
    dead_map = pull_dead_map(root / "data" / "raw_cmc")
    dead_ids = set(dead_map["cmc_id"].astype(int))
    table = D.build_universe_table(keep, dead_ids, window_end=end)
    print(f"\n=== how tokens left the sample ===")
    print(D.exit_summary(table).to_string())
    n_death = int(table["delisted"].sum())
    n_cens = int(table["censored"].sum())
    print(f"  totals: {n_death:,} deaths, {n_cens:,} censored by rank, "
          f"{len(table) - n_death - n_cens:,} still ranked")
    print(f"  delisting rules: {table['delisting_rule'].value_counts().to_dict()}")
    dying = table[table["delisted"]]
    if len(dying):
        v = pd.to_numeric(dying["last_volume_usd"], errors="coerce").fillna(0.0)
        print(f"  final-day volume of a dying token: median ${v.median():,.2f}, "
              f"90th pct ${v.quantile(0.9):,.0f}, {int((v < 1000).sum())}/{len(v)} under $1k")
        print("  delisting-policy sensitivity (share of deaths taking a terminal loss):")
        for policy in D.DELISTING_POLICIES:
            alt = D.build_universe_table(keep, dead_ids, window_end=end,
                                         delisting_policy=policy)
            n_tl = int((alt["delisting_rule"] == P.RULE_TOTAL_LOSS).sum())
            print(f"    {policy:<13s} {n_tl:5d}/{len(dying)} "
                  f"({100 * n_tl / max(len(dying), 1):5.1f}%)")

    # --- panels -------------------------------------------------------------
    close = D.wide(keep, "price_usd", index)
    mcap = D.wide(keep, "market_cap_usd", index)
    volume = D.wide(keep, "volume_24h_usd", index)
    eth = D.wide(panel[panel["cmc_id"] == B.ETH_CMC_ID], "price_usd", index)[B.ETH_CMC_ID]
    btc = D.wide(panel[panel["cmc_id"] == BTC_CMC_ID], "price_usd", index)
    btc = btc[BTC_CMC_ID] if BTC_CMC_ID in btc.columns else pd.Series(np.nan, index=index)

    # --- data quality -------------------------------------------------------
    redenom = D.redenomination_flags(close, mcap)
    returns, n_scrubbed = D.excess_log_returns(close, eth, table)
    raw_ret = np.log(close.where(close > 0) / close.where(close > 0).shift(1))
    bad = raw_ret.abs() > P.MAX_ABS_LOG_RETURN
    qual = pd.DataFrame({
        "bad_prints": bad.groupby(bad.index.year).sum().sum(axis=1),
        "redenominations": redenom.groupby(redenom.index.year).sum().sum(axis=1),
        "observations": raw_ret.notna().groupby(raw_ret.index.year).sum().sum(axis=1),
    })
    qual["bad_print_pct"] = (100 * qual["bad_prints"] / qual["observations"]).round(3)
    print(f"\n=== data quality by year ===")
    print(qual.to_string())
    print(f"  total scrubbed |log r| > log 5: {n_scrubbed:,} "
          f"({100 * n_scrubbed / max(int(raw_ret.notna().to_numpy().sum()), 1):.3f}% of the panel)")

    # --- brackets -----------------------------------------------------------
    snaps = monthly_snapshots(keep)
    snaps_with_eth = pd.concat([snaps, monthly_snapshots(panel[panel["cmc_id"] == B.ETH_CMC_ID])],
                               ignore_index=True)
    assignments = B.assign_brackets(snaps_with_eth, start=start)
    bounds = B.bracket_dollar_bounds(snaps_with_eth, start=start)
    counts = B.member_counts(assignments)

    print(f"\n=== bracket sizes at month ends ({assignments['snapshot_date'].nunique()} reconstitutions) ===")
    size = counts.groupby("bracket")["n_members"].agg(["mean", "min", "max"]).round(1)
    size["months_clusterable"] = counts.groupby("bracket")["clusterable"].sum()
    size["months_total"] = counts.groupby("bracket")["clusterable"].size()
    size["unique_tokens"] = assignments.groupby("bracket")["cmc_id"].nunique()
    print(size.to_string())
    print(f"  clustering is skipped where members < {B.MIN_MEMBERS_FOR_CLUSTERING}")

    # --- does the rank-2000 depth bind? -------------------------------------
    binds = B.rank_cutoff_binds(snaps_with_eth, RANK_DEPTH, "B3", start=start)
    by_year = binds.assign(year=binds["snapshot_date"].dt.year).groupby("year").agg(
        months=("binds", "size"), binds=("binds", "sum"),
        deep=("exhausted_depth", "sum"), n_ranked=("n_ranked", "mean"),
        floor_usd=("floor_usd", "mean"), deepest_mcap=("deepest_market_cap_usd", "mean"))
    print(f"\n=== does the rank-{RANK_DEPTH} pull depth bind on B3's floor? ===")
    print("  a month binds if CMC ranked the full depth AND the deepest priced")
    print("  token is still above B3's floor, so members exist below the pull")
    print("  year  binds  full-depth  tokens ranked   B3 floor   deepest mcap")
    for y, r in by_year.iterrows():
        print(f"  {y}  {int(r['binds']):3d}/{int(r['months']):<2d}  "
              f"{int(r['deep']):3d}/{int(r['months']):<2d}       "
              f"{r['n_ranked']:8.0f}   {fmt_usd(r['floor_usd']):>9}   "
              f"{fmt_usd(r['deepest_mcap']):>9}")
    n_binds = int(binds["binds"].sum())
    print(f"  total: {n_binds}/{len(binds)} month-ends bind "
          f"({100 * n_binds / max(len(binds), 1):.0f}%)")

    # --- shortability -------------------------------------------------------
    if args.no_perps:
        short = pd.DataFrame(columns=["symbol", "base", "n_venues", "venues", "shortable"])
    else:
        print("\n=== perpetual venues ===")
        perps = PERPS.perp_universe()
        syms = sorted({s for s in table["symbol"].dropna().unique()})
        short = PERPS.shortable(syms, perps)
        perps.to_parquet(out_dir / "perp_universe.parquet", index=False)
    short.to_parquet(out_dir / "bracket_shortability.parquet", index=False)

    # --- per bracket per year ----------------------------------------------
    membership = {}
    ids = [int(c) for c in close.columns]
    for bracket in B.BRACKET_ORDER:
        m = B.bracket_membership(assignments, index, bracket, columns=ids)
        membership[bracket] = survivorship_books(m, table, close)

    year_tab = per_year_table(assignments, table, membership, short, bounds)
    print(f"\n=== per bracket per year ===")
    for bracket in B.BRACKET_ORDER:
        sub = year_tab[year_tab["bracket"] == bracket]
        if sub.empty:
            continue
        print(f"\n  {bracket}  {B.BRACKET_LABELS[bracket]}")
        print("  year  tokens  tok-days  deaths  avg  min   bounds (USD)        perp%")
        for _, r in sub.iterrows():
            rng = f"{fmt_usd(r['lo_usd'])}-{fmt_usd(r['hi_usd'])}" if np.isfinite(r["hi_usd"]) \
                  else f"{fmt_usd(r['lo_usd'])}+"
            print(f"  {int(r['year'])}  {int(r['tokens']):6d}  {int(r['token_days']):8d}  "
                  f"{int(r['deaths']):6d}  {r['avg_members']:4.0f} {int(r['min_members']):4d}   "
                  f"{rng:<18}  {100 * r['perp_share']:4.0f}%")

    # --- write derived tables ----------------------------------------------
    # The monthly rank snapshots at full depth: the point-in-time ranking that
    # every bracket is cut from, committed so the universe is reproducible
    # without repeating the pull.
    snaps_out = monthly_snapshots(panel)[
        ["snapshot_date", "cmc_id", "symbol", "name", "slug", "rank",
         "price_usd", "volume_24h_usd", "market_cap_usd"]]
    snaps_out.to_parquet(out_dir / "pit_monthly_ranks.parquet", index=False,
                         compression="zstd")

    table.to_parquet(out_dir / "bracket_universe.parquet", index=False, compression="zstd")
    assignments.to_parquet(out_dir / "bracket_assignments.parquet", index=False,
                           compression="zstd")
    bounds.to_parquet(out_dir / "bracket_dollar_bounds.parquet", index=False,
                      compression="zstd")
    counts.to_parquet(out_dir / "bracket_member_counts.parquet", index=False,
                      compression="zstd")
    qual.to_csv(out_dir / "bracket_data_quality.csv")
    binds.to_parquet(out_dir / "bracket_rank_binding.parquet", index=False, compression="zstd")
    year_tab.to_csv(out_dir / "bracket_by_year.csv", index=False)

    for bracket in B.BRACKET_ORDER:
        cols = sorted({int(c) for c in
                       assignments.loc[assignments["bracket"] == bracket, "cmc_id"]})
        cols = [c for c in cols if c in returns.columns]
        if not cols:
            continue
        r = returns[cols]
        r.columns = [str(c) for c in r.columns]
        # float32 halves the committed panel. Daily log returns live around
        # 1e-3 to 1e0 and float32 carries ~7 significant digits, so the error
        # accumulated over the whole 3,500-day sample is order 1e-5, far below
        # anything that moves a Sharpe ratio.
        r.astype("float32").to_parquet(
            out_dir / f"bracket_returns_{bracket}.parquet", compression="zstd")
        stacked = pd.concat(
            {k: membership[bracket][k][cols] for k in SURVIVORSHIP_TREATMENTS}, axis=1)
        stacked.columns = [f"{a}|{int(b)}" for a, b in stacked.columns]
        stacked.to_parquet(out_dir / f"bracket_membership_{bracket}.parquet",
                           compression="zstd")

    # reference series every later step residualises against
    pd.DataFrame({
        "eth": eth, "btc": btc,
        "vw_market": D.value_weighted_market(close, mcap),
    }).to_parquet(out_dir / "bracket_references.parquet", compression="zstd")

    written = (sorted(p.name for p in out_dir.glob("bracket_*"))
               + ["perp_universe.parquet", "pit_monthly_ranks.parquet"])
    total_mb = sum((out_dir / n).stat().st_size for n in written
                   if (out_dir / n).exists()) / 1e6
    print(f"\nwrote {len(written)} tables to {out_dir} ({total_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
