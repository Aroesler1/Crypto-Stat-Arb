"""Pull CoinMarketCap's point-in-time ranking at DAILY frequency.

Why this exists, and why it is the primary source
-------------------------------------------------
`build_pit_universe.py` builds its price panel from the per-token history
endpoint. On a 24-month window that works. On a 2016-2025 window it does not,
and the way it fails is the exact failure this project exists to measure.

Measured on a stratified sample of 120 tokens that left the brackets, by the
year they left (see README, "the pruning problem"):

    death year   2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
    no daily     83%  75%  83%  83%  75%  42%  17%   0%   8%   0%
    history

CoinMarketCap prunes per-token daily history for coins that died roughly three
or more years ago. A panel built from that endpoint over 2016-2025 is therefore
close to survivor-only in its early years while being *labelled* point-in-time,
which is worse than a panel that is honestly survivor-only. The previous 24-month
run escaped this only because it sat entirely inside the region where coverage is
complete.

The ``listings/historical`` endpoint has no such gap. It accepts an arbitrary
date, not just month ends, and returns the full ranking as it stood on that day
including coins that have since been pruned from the history endpoint. Peculium
(cmc_id 2610) returns zero daily bars from the history endpoint, but prints
price 0.001386, volume $49,028 and market cap $2.74M in the listing for
2019-03-14. Consecutive days differ in every one of the top 50 prices, so this is
genuinely daily data and not a monthly stub repeated.

So this script pulls one listing per calendar day and stacks them. The result is
a daily panel that is point-in-time and survivorship-free by construction: a
token is in it on exactly the days CMC ranked it, and it disappears on the day it
stopped being ranked, which is the death event itself.

What it gives up
----------------
Close only, no OHLC. The listing quote carries price, 24h volume and market cap,
which is everything a close-to-close book needs, but a strategy wanting intraday
range has to fall back on the history endpoint for the tokens that still have it.
`validate_against_history` in `stat_arb/data/daily_listings.py` cross-checks the
two sources on the tokens where both exist.

Usage
-----
    python stat_arb/build_daily_listings.py --start 2015-07-01 --end 2025-06-30 \
        --depth 2000 --workers 2 --sleep 0.3

Days are cached one parquet per day under ``data/raw_cmc/listings_d<depth>/``
(gitignored), so the pull resumes rather than restarting.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.data import cmc_pit  # noqa: E402

# ETH's first listing is 2015-08-07. The daily panel starts a month earlier so
# there is a run-up before the first ETH-relative bracket can be assigned.
DEFAULT_START = "2015-07-01"
DEFAULT_END = "2025-06-30"
DEFAULT_DEPTH = 2000


def _day_path(raw_dir: Path, depth: int, day: pd.Timestamp) -> Path:
    d = raw_dir / f"listings_d{depth}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{day.strftime('%Y-%m-%d')}.parquet"


def pull_days(raw_dir: Path, start: pd.Timestamp, end: pd.Timestamp, depth: int,
              workers: int = 2, force: bool = False) -> list[Path]:
    """Fetch every calendar day in [start, end], one cached parquet per day."""
    days = pd.date_range(start, end, freq="D")
    todo = [d for d in days if force or not _day_path(raw_dir, depth, d).exists()]
    cached = len(days) - len(todo)
    print(f"  listings: {cached} cached, {len(todo)} to fetch "
          f"({start.date()}..{end.date()}, depth {depth})", flush=True)

    def job(day: pd.Timestamp) -> tuple[pd.Timestamp, int]:
        df = cmc_pit.fetch_listing_snapshot(day, depth)
        # An empty day is cached too: CMC has genuine gaps in the early years and
        # re-requesting them on every resume would triple the pull.
        df.to_parquet(_day_path(raw_dir, depth, day), index=False)
        return day, len(df)

    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(job, d): d for d in todo}
            for fut in as_completed(futures):
                day = futures[fut]
                try:
                    _, n = fut.result()
                except Exception as exc:  # one bad day must not sink the pull
                    print(f"    {day.date()} failed: {exc}", flush=True)
                    n = -1
                done += 1
                if done % 200 == 0:
                    print(f"    {done}/{len(todo)} days fetched", flush=True)
    return [_day_path(raw_dir, depth, d) for d in days]


def stack(paths: list[Path]) -> pd.DataFrame:
    """Concatenate cached day files into one long panel keyed on (date, cmc_id)."""
    frames = []
    for p in paths:
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "cmc_id", "symbol", "name", "slug",
                                     "rank", "price_usd", "volume_24h_usd",
                                     "market_cap_usd"])
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"snapshot_date": "date"})
    out["date"] = pd.to_datetime(out["date"])
    out["cmc_id"] = out["cmc_id"].astype("int32")
    for c in ("rank",):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    for c in ("price_usd", "volume_24h_usd", "market_cap_usd"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return (out.drop_duplicates(subset=["date", "cmc_id"], keep="last")
               .sort_values(["date", "rank"])
               .reset_index(drop=True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-write", action="store_true",
                    help="pull and cache only, do not stack or write the panel")
    args = ap.parse_args(argv)

    cmc_pit.set_pause(args.sleep)
    root = Path(__file__).resolve().parent.parent
    raw_dir = Path(args.raw_dir) if args.raw_dir else root / "data" / "raw_cmc"
    out_dir = Path(args.out_dir) if args.out_dir else root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    paths = pull_days(raw_dir, start, end, args.depth, args.workers, args.force)
    if args.no_write:
        return 0

    panel = stack(paths)
    if panel.empty:
        print("no listings pulled")
        return 1
    out_path = out_dir / "pit_daily_listings.parquet"
    panel.to_parquet(out_path, index=False, compression="zstd")
    size_mb = out_path.stat().st_size / 1e6
    print(f"\npit_daily_listings.parquet  {len(panel):,} rows, "
          f"{panel['cmc_id'].nunique():,} cmc_ids, "
          f"{panel['date'].min().date()}..{panel['date'].max().date()}, {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
