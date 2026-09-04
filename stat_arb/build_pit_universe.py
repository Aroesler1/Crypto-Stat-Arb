"""Build the point-in-time universe, including tokens that died.

One command, reproducible from a clone, no API key and no paid plan:

    python stat_arb/build_pit_universe.py

What it does
------------
1. Pulls a CoinMarketCap ranking snapshot at every month end in the sample
   window. These are point-in-time: a coin that has since died still appears,
   with the rank it actually held that month.
2. Pulls CMC's inactive and untracked coin maps, which is how a token that no
   longer exists gets identified as such.
3. Pulls full daily OHLCV for every `cmc_id` that entered the rank band at any
   reconstitution, plus ETH as the excess-return benchmark.
4. Writes two derived tables (see DATA.md):
     data/universe_pit.parquet        one row per cmc_id: symbol, first_date,
                                      last_date, delisted, delisting rule
     data/universe_pit_ohlcv.parquet  long daily OHLCV keyed on (cmc_id, date)

The raw pull is cached under `data/raw_cmc/` (gitignored) so a failure resumes
instead of restarting, and so the ~2,000 request pull happens once. Only the
derived tables are committed.

Rank band
---------
Defaults to CMC ranks 150-500, which is what the committed snapshot universe
actually is: matching `data/all_tokens_24mo_daily.csv` against a point-in-time
ranking on the snapshot date gives a minimum rank of 163, a 95th percentile of
495, and 96-99% coverage of ranks 201-500 against 0% of ranks 1-100.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.data import cmc_pit, pit_universe  # noqa: E402

ETH_CMC_ID = 1027
# The committed sample. Kept as the default so the PIT result is comparable to
# the published one rather than to a differently-dated experiment.
DEFAULT_START = "2023-05-29"
DEFAULT_END = "2025-05-29"
DEFAULT_RANK_LO = 150
DEFAULT_RANK_HI = 500
# Margin around the window so returns exist on the first day and a token that
# dies near the end is still seen to have died.
PAD_DAYS = 30


def _cache_path(raw_dir: Path, kind: str, key: str) -> Path:
    d = raw_dir / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.parquet"


def _history_ns(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Cache namespace for a history pull.

    The history endpoint returns only what was asked for, so a file cached for a
    2-year window is not a valid cache entry for a 12-year one. Namespacing on
    the window makes a wider pull refetch instead of silently reusing a series
    that stops short.
    """
    return f"history_{pd.Timestamp(start).date()}_{pd.Timestamp(end).date()}"


def _cached(path: Path, fetch, force: bool = False) -> pd.DataFrame:
    """Read `path` if present, else fetch and write it. Makes the pull resumable."""
    if path.exists() and not force:
        return pd.read_parquet(path)
    df = fetch()
    df.to_parquet(path, index=False)
    return df


def pull_snapshots(raw_dir: Path, start: pd.Timestamp, end: pd.Timestamp,
                   depth: int, force: bool = False) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="ME")
    if len(dates) == 0 or dates[0] > start:
        dates = pd.DatetimeIndex([pd.Timestamp(start)]).append(dates)
    frames = []
    for day in dates:
        key = day.strftime("%Y-%m-%d")
        # depth is part of the key: a snapshot cached at depth 500 is not a
        # valid cache hit for a depth-2000 pull.
        path = _cache_path(raw_dir, "snapshots", f"{key}_d{depth}")
        frames.append(_cached(path, lambda d=key: cmc_pit.fetch_listing_snapshot(d, depth), force))
        print(f"  snapshot {key}: {len(frames[-1])} ranks", flush=True)
    return pd.concat(frames, ignore_index=True)


def pull_dead_map(raw_dir: Path, force: bool = False) -> pd.DataFrame:
    frames = []
    for status in cmc_pit.DEAD_STATUSES:
        path = _cache_path(raw_dir, "map", status)
        df = _cached(path, lambda s=status: cmc_pit.fetch_map(s), force)
        print(f"  map {status}: {len(df)} coins", flush=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pull_histories(raw_dir: Path, cmc_ids: list[int], start: pd.Timestamp,
                   end: pd.Timestamp, workers: int = 4,
                   force: bool = False) -> dict[int, pd.DataFrame]:
    """Daily OHLCV for every id, cached one file per id so this resumes."""
    out: dict[int, pd.DataFrame] = {}
    todo = []
    ns = _history_ns(start, end)
    for cmc_id in cmc_ids:
        path = _cache_path(raw_dir, ns, str(int(cmc_id)))
        if path.exists() and not force:
            out[int(cmc_id)] = pd.read_parquet(path)
        else:
            todo.append((int(cmc_id), path))

    print(f"  histories: {len(out)} cached, {len(todo)} to fetch", flush=True)
    if not todo:
        return out

    def job(item):
        cmc_id, path = item
        df = cmc_pit.fetch_history(cmc_id, start, end)
        df.to_parquet(path, index=False)
        return cmc_id, df

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, item): item[0] for item in todo}
        for fut in as_completed(futures):
            cmc_id = futures[fut]
            try:
                cmc_id, df = fut.result()
                out[cmc_id] = df
            except Exception as exc:  # a single dead id must not sink the pull
                print(f"    id={cmc_id} failed: {exc}", flush=True)
                out[cmc_id] = pd.DataFrame(
                    columns=["date", "open", "high", "low", "close",
                             "volume_usd", "market_cap_usd"])
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(todo)} fetched", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--rank-lo", type=int, default=DEFAULT_RANK_LO)
    ap.add_argument("--rank-hi", type=int, default=DEFAULT_RANK_HI)
    ap.add_argument("--raw-dir", default=None, help="raw pull cache (gitignored)")
    ap.add_argument("--out-dir", default=None, help="where derived tables are written")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=None,
                    help="polite pause in seconds between CMC requests "
                         f"(default {cmc_pit.get_pause()})")
    ap.add_argument("--force", action="store_true", help="ignore the raw cache")
    args = ap.parse_args(argv)

    if args.sleep is not None:
        cmc_pit.set_pause(args.sleep)

    root = Path(__file__).resolve().parent.parent
    raw_dir = Path(args.raw_dir) if args.raw_dir else root / "data" / "raw_cmc"
    out_dir = Path(args.out_dir) if args.out_dir else root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    pull_start = start - pd.Timedelta(days=PAD_DAYS)
    pull_end = end + pd.Timedelta(days=PAD_DAYS)

    print(f"window {start.date()} .. {end.date()}  rank band [{args.rank_lo}, {args.rank_hi}]")

    print("1/4 point-in-time ranking snapshots")
    snapshots = pull_snapshots(raw_dir, start, end, args.rank_hi, args.force)

    print("2/4 dead-coin maps")
    dead_map = pull_dead_map(raw_dir, args.force)
    dead_ids = set(dead_map["cmc_id"].astype(int))

    band = snapshots[(snapshots["rank"] >= args.rank_lo) & (snapshots["rank"] <= args.rank_hi)]
    band_ids = sorted(band["cmc_id"].astype(int).unique())
    print(f"    rank band union: {len(band_ids)} cmc_ids, "
          f"{len(set(band_ids) & dead_ids)} of them dead today")

    print("3/4 daily OHLCV")
    histories = pull_histories(raw_dir, band_ids + [ETH_CMC_ID], pull_start, pull_end,
                               args.workers, args.force)
    eth_hist = histories.pop(ETH_CMC_ID, pd.DataFrame())
    if eth_hist.empty:
        print("ETH benchmark history is empty; cannot build excess returns")
        return 1

    print("4/4 derived tables")
    table = pit_universe.build_universe_table(
        histories, band, dead_ids, window_start=start, window_end=end)

    long = []
    for cmc_id, hist in histories.items():
        if hist.empty:
            continue
        h = hist.copy()
        h.insert(0, "cmc_id", int(cmc_id))
        long.append(h)
    ohlcv = (pd.concat(long, ignore_index=True)
             .sort_values(["cmc_id", "date"])
             .reset_index(drop=True))
    ohlcv = ohlcv[(ohlcv["date"] >= pull_start) & (ohlcv["date"] <= pull_end)]

    eth = eth_hist.copy()
    eth.insert(0, "cmc_id", ETH_CMC_ID)
    ohlcv = pd.concat([ohlcv, eth], ignore_index=True)

    # zstd so a rebuild reproduces the committed files byte-for-byte in spirit;
    # float64 price data barely compresses, so this is about consistency more
    # than size.
    table.to_parquet(out_dir / "universe_pit.parquet", index=False, compression="zstd")
    ohlcv.to_parquet(out_dir / "universe_pit_ohlcv.parquet", index=False, compression="zstd")
    snapshots.to_parquet(out_dir / "universe_pit_ranks.parquet", index=False, compression="zstd")

    n_dead = int(table["delisted"].sum())
    rules = table["delisting_rule"].value_counts().to_dict()
    print(f"\nuniverse_pit.parquet      {len(table)} cmc_ids, {n_dead} delisted "
          f"({n_dead / max(len(table), 1):.1%})")
    print(f"delisting rules applied   {rules}")
    print(f"universe_pit_ohlcv.parquet {len(ohlcv):,} daily bars "
          f"({ohlcv['date'].min().date()} .. {ohlcv['date'].max().date()})")
    print(f"universe_pit_ranks.parquet {len(snapshots):,} point-in-time ranks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
