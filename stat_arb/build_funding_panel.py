"""Pull measured perpetual funding for every shortable bracket member.

The short leg of a market-neutral book pays (or receives) funding every eight
hours. The repo previously stressed that with a uniform carry knob and, later,
with real Hyperliquid rates for 11 tokens. This pulls the real thing for the
whole bracket universe.

Source is `data.binance.vision`, Binance's static archive. `fapi.binance.com`
returns HTTP 451 to a US IP, which is why the repo used Hyperliquid, but the
archive does not: its S3 listing enumerates 864 perpetual contracts and each
monthly file unzips to real 8-hourly funding rates. Only history is reachable
this way, never live state, which is all a backtest needs. Binance is the
deepest perpetual venue by a wide margin, so this is the difference between
funding for 11 tokens and funding for 533.

Sign convention: a positive funding rate means longs pay shorts, so a SHORT
position earns positive funding and a long pays it. `apply_to_short_leg` in
`run_tradability.py` is where that sign is turned into a cost.

Only (symbol, month) pairs where the token was actually a bracket member are
pulled, which is about 22,000 requests rather than the 100,000 a full
cross-product would need. Every response is cached one parquet per pair,
including the 404s that mean the contract did not exist yet, so a resume does
not re-ask for them.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.data import perps as PERPS  # noqa: E402


def _cache_path(raw_dir: Path, symbol: str, month: str) -> Path:
    d = raw_dir / "binance_funding" / symbol
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{month}.parquet"


def needed_pairs(assignments: pd.DataFrame, universe: pd.DataFrame,
                 symbol_map: dict[str, str]) -> list[tuple[str, str]]:
    """(contract symbol, YYYY-MM) pairs where a shortable member was in a bracket.

    A month of margin either side, so a position opened at a reconstitution and
    held into the next month has funding for every day it is on.
    """
    sym = universe.set_index("cmc_id")["symbol"].to_dict()
    a = assignments.copy()
    a["base"] = a["cmc_id"].map(sym).map(
        lambda s: PERPS.normalize_base(s) if isinstance(s, str) else "")
    a = a[a["base"].isin(symbol_map)]
    pairs = set()
    for base, day in zip(a["base"], pd.to_datetime(a["snapshot_date"])):
        contract = symbol_map[base]
        for offset in (0, 1):
            pairs.add((contract, str((day + pd.DateOffset(months=offset)).to_period("M"))))
    return sorted(pairs)


def pull(raw_dir: Path, pairs: list[tuple[str, str]], workers: int = 4,
         force: bool = False) -> None:
    todo = [p for p in pairs if force or not _cache_path(raw_dir, p[0], p[1]).exists()]
    print(f"  funding: {len(pairs) - len(todo)} cached, {len(todo)} to fetch", flush=True)
    if not todo:
        return

    def job(pair):
        symbol, month = pair
        df = PERPS.binance_funding_monthly(symbol, month)
        # An empty frame is cached too: it means the contract did not exist that
        # month, and re-asking on every resume would double the pull.
        df.to_parquet(_cache_path(raw_dir, symbol, month), index=False)
        return pair, len(df)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, p): p for p in todo}
        for fut in as_completed(futures):
            pair = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"    {pair} failed: {exc}", flush=True)
            done += 1
            if done % 1000 == 0:
                print(f"    {done}/{len(todo)} pulled", flush=True)


def stack(raw_dir: Path, symbol_map: dict[str, str]) -> pd.DataFrame:
    """Daily funding per base asset: date, base, funding_rate (daily sum)."""
    inverse = {v: k for k, v in symbol_map.items()}
    frames = []
    root = raw_dir / "binance_funding"
    for sym_dir in sorted(root.glob("*")):
        if not sym_dir.is_dir():
            continue
        parts = [pd.read_parquet(f) for f in sorted(sym_dir.glob("*.parquet"))]
        parts = [p for p in parts if not p.empty]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        daily = (df.set_index("time")["funding_rate"].sort_index()
                 .resample("D").sum().rename("funding_rate").reset_index())
        daily = daily.rename(columns={"time": "date"})
        daily["base"] = inverse.get(sym_dir.name, PERPS.normalize_base(sym_dir.name))
        daily["symbol"] = sym_dir.name
        frames.append(daily)
    if not frames:
        return pd.DataFrame(columns=["date", "base", "symbol", "funding_rate"])
    return (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["date", "base"], keep="last")
            .sort_values(["base", "date"]).reset_index(drop=True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    raw_dir = Path(args.raw_dir) if args.raw_dir else data_dir / "raw_cmc"

    assignments = pd.read_parquet(data_dir / "bracket_assignments.parquet")
    universe = pd.read_parquet(data_dir / "bracket_universe.parquet")

    print("enumerating Binance perpetual contracts ...", flush=True)
    symbol_map = PERPS.binance_symbol_map()
    print(f"  {len(symbol_map)} base assets with a contract")

    pairs = needed_pairs(assignments, universe, symbol_map)
    print(f"  {len({p[0] for p in pairs})} contracts, {len(pairs)} (symbol, month) pairs")

    pull(raw_dir, pairs, args.workers, args.force)

    panel = stack(raw_dir, symbol_map)
    if panel.empty:
        print("no funding pulled")
        return 1
    out = data_dir / "funding_panel.parquet"
    panel.to_parquet(out, index=False, compression="zstd")
    print(f"\nfunding_panel.parquet  {len(panel):,} daily rows, "
          f"{panel['base'].nunique()} bases, "
          f"{panel['date'].min().date()}..{panel['date'].max().date()}, "
          f"{out.stat().st_size / 1e6:.1f} MB")
    ann = panel.groupby("base")["funding_rate"].mean() * 365
    print(f"annualised funding, cross-sectional: median {ann.median():+.2%}, "
          f"10th {ann.quantile(0.1):+.2%}, 90th {ann.quantile(0.9):+.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
