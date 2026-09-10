"""Static perpetual-symbol subset audit, not historical executable returns.

A later venue snapshot has no historical listing dates and does not establish
borrow availability. Funding is measured only where a dated rate is present;
missing signed funding set to zero is neither a cost nor a performance bound.
Daily funding uses opening effective holdings as an intraday-notional proxy.
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

from stat_arb.backtest.costs import TransactionCostModel  # noqa: E402
from stat_arb.backtest.engine import BacktestEngine  # noqa: E402
from stat_arb.data import brackets as B  # noqa: E402
from stat_arb.data import perps as PERPS  # noqa: E402
from stat_arb.run_phase3 import annualized_sharpe  # noqa: E402
from stat_arb.run_residualization_ablation import (  # noqa: E402
    BEST_BAND, BEST_FREQ, load_inputs, run_arm,
)
from stat_arb.run_signal_ablation import BEST_REFERENCE, arm_kwargs, make_death_filter
from stat_arb.data import death_model as DM  # noqa: E402

PERIODS_PER_YEAR = 365


def shortable_ids(universe: pd.DataFrame, perps: pd.DataFrame) -> set[int]:
    """cmc_ids whose ticker matches a listed perpetual on any reachable venue."""
    bases = set(perps["base"])
    out = set()
    for row in universe.itertuples(index=False):
        sym = getattr(row, "symbol", None)
        if isinstance(sym, str) and PERPS.normalize_base(sym) in bases:
            out.add(int(row.cmc_id))
    return out


def funding_panel_wide(funding: pd.DataFrame, universe: pd.DataFrame,
                       index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    """Daily funding rate per return-column, aligned to the backtest calendar."""
    base_of = {}
    for row in universe.itertuples(index=False):
        sym = getattr(row, "symbol", None)
        if isinstance(sym, str):
            base_of[f"{int(row.cmc_id)}_returns"] = PERPS.normalize_base(sym)

    wide = funding.pivot_table(index="date", columns="base", values="funding_rate",
                               aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(pd.DatetimeIndex(index))
    # NaN, not 0.0, so a missing rate stays distinguishable from a genuine
    # zero one; `apply_funding` fills at the point of use.
    out = pd.DataFrame(np.nan, index=wide.index, columns=columns)
    covered = []
    for col in columns:
        base = base_of.get(col)
        if base is not None and base in wide.columns:
            out[col] = wide[base]
            covered.append(col)
    return out, covered


def funding_coverage(weights: pd.DataFrame, funding: pd.DataFrame,
                     covered: list[str]) -> tuple[float, float]:
    """How much of the book has a measured funding rate.

    Two numbers, because they say different things and the naive one misleads.
    ``by_column`` is the share of the bracket's return columns whose base appears
    in the funding panel at all; it is dragged down by every token that passed
    through the bracket years before any venue listed a perpetual on it,
    including ones the book never holds. ``by_exposure`` is the share of total
    absolute position sitting on a name with a rate on the day it is held, which
    is what actually determines how much of the funding cost is measured rather
    than assumed.
    """
    if weights.empty:
        return float("nan"), float("nan")
    by_column = len(covered) / max(len(weights.columns), 1)

    w = weights.abs().fillna(0.0)
    f = funding.reindex(index=w.index, columns=w.columns)
    present = f.notna()
    total = float(w.to_numpy().sum())
    if total <= 0:
        return by_column, float("nan")
    return by_column, float(w.where(present, 0.0).to_numpy().sum() / total)


def apply_funding(weights: pd.DataFrame, funding: pd.DataFrame) -> pd.Series:
    """Daily funding contribution to the book's return.

    Positive funding means longs pay shorts, so the book's P&L from funding is
    the negative of its weighted exposure. These are already-effective positions
    entering the dated return; applying another lag charges the wrong book.
    """
    f = funding.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    return -(weights.fillna(0.0) * f).sum(axis=1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--start", default=str(B.BRACKET_START.date()))
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--brackets", default="B1,B2,B3")
    ap.add_argument("--arm", default=None,
                    help="signal arm; default is the best from signal_ablation.csv")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    panel_path = Path(args.panel) if args.panel else data_dir / "pit_daily_listings.parquet"
    for required in (panel_path, data_dir / "perp_universe.parquet"):
        if not required.exists():
            print(f"{required} not found")
            return 1

    funding_path = data_dir / "funding_panel.parquet"
    funding = (pd.read_parquet(funding_path) if funding_path.exists()
               else pd.DataFrame(columns=["date", "base", "funding_rate"]))
    if funding.empty:
        print("WARNING: no funding panel; run stat_arb/build_funding_panel.py. "
              "Observed funding will be zero; the unmeasured signed contribution is unknown.")

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    print(f"loading panels {start.date()}..{end.date()} ...", flush=True)
    table, assignments, close, volume, mcap, refs, index = load_inputs(
        panel_path, data_dir, start, end)
    perps = pd.read_parquet(data_dir / "perp_universe.parquet")
    short_ids = shortable_ids(table, perps)
    print(f"  {len(short_ids)} of {len(table)} universe tokens have a listed perpetual")

    best_arm = args.arm
    abl_path = root / "stat_arb" / "reporting" / "brackets" / "signal_ablation.csv"
    best_by_bracket = {}
    if best_arm is None and abl_path.exists():
        abl = pd.read_parquet(abl_path) if abl_path.suffix == ".parquet" else pd.read_csv(abl_path)
        pit = abl[abl["treatment"] == "pit"]
        for bracket, g in pit.groupby("bracket"):
            best_by_bracket[bracket] = g.loc[g["net_sharpe"].idxmax(), "arm"]
        print(f"  best Step 3 arm per bracket: {best_by_bracket}")

    selected = {best_arm or best_by_bracket.get(b, "baseline")
                for b in args.brackets.split(",")}
    death_filter = None
    if "death" in selected:
        raw = pd.read_parquet(panel_path)
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw[raw["cmc_id"].isin(set(table["cmc_id"].astype(int)))]
        features = DM.build_features(raw, table, sample_every=7)
        probs = DM.death_probabilities(features)
        death_filter = make_death_filter(probs, {
            f"{int(c)}_returns": int(c) for c in close.columns})
    rows = []
    for bracket in [b.strip() for b in args.brackets.split(",") if b.strip()]:
        ids = sorted({int(c) for c in
                      assignments.loc[assignments["bracket"] == bracket, "cmc_id"]})
        ids = [c for c in ids if c in close.columns]
        full = B.bracket_membership(assignments, index, bracket, columns=ids)
        tradeable = full.copy()
        for c in tradeable.columns:
            if int(c) not in short_ids:
                tradeable[c] = False

        arm = best_arm or best_by_bracket.get(bracket, "baseline")
        reference = BEST_REFERENCE.get(bracket, "eth")
        for label, mask in (("all names", full), ("shortable only", tradeable)):
            if mask.to_numpy().sum() == 0:
                print(f"  {bracket} / {label}: empty, skipped")
                continue
            print(f"  {bracket} / {label} / arm={arm} ...", flush=True)
            fwide, covered = funding_panel_wide(
                funding, table, index, [f"{int(c)}_returns" for c in mask.columns])
            measured_funding = (fwide if label == "shortable only" and not funding.empty else None)
            stats = run_arm(close, volume, table, refs, reference, 0, mask, index,
                            funding_rates=measured_funding, **arm_kwargs(arm, death_filter))
            if stats is None:
                print("    produced no positions, skipped")
                continue
            stats.pop("net_series", None)
            stats.update(bracket=bracket, subset=label, arm=arm, reference=reference,
                         funding_coverage=len(covered) / max(len(mask.columns), 1))
            for field in ("funding_ann", "net_sharpe_after_funding", "funding_coverage_exposure"):
                stats.setdefault(field, np.nan)
            rows.append(stats)


    if not rows:
        print("no configuration produced a result")
        return 1

    out = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("weights", "net_series_for_funding")}
                        for r in rows])
    out_dir = root / "stat_arb" / "reporting" / "brackets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "tradability_corrected.csv", index=False)

    print(f"\n=== tradability by bracket (point-in-time, band {BEST_BAND:.0%}, "
          f"rebalance {BEST_FREQ}d, net 50bps) ===")
    print("  bracket  subset           members  gross    net  after funding  breakeven"
          "   cov(col)  cov(exp)")
    for _, r in out.iterrows():
        af = ("-" if not np.isfinite(r["net_sharpe_after_funding"])
              else f"{r['net_sharpe_after_funding']:.2f}")
        cov = ("-" if not np.isfinite(r["funding_coverage"])
               else f"{100 * r['funding_coverage']:.0f}%")
        cove = ("-" if not np.isfinite(r.get("funding_coverage_exposure", np.nan))
                else f"{100 * r['funding_coverage_exposure']:.0f}%")
        print(f"  {r['bracket']:<7s}  {r['subset']:<15s} {r['avg_members']:7.0f} "
              f"{r['gross_sharpe']:6.2f} {r['net_sharpe']:6.2f} {af:>13s} "
              f"{r['breakeven_bps']:10.0f} {cov:>10s} {cove:>9s}")

    print("\n  cov(exp) is the share of total absolute position sitting on a name")
    print("  with a measured funding rate on the day it is held, which is what")
    print("  determines how much of the cost is measured rather than assumed.")
    print("  cov(col) is the share of the bracket's return COLUMNS carrying a rate")
    print("  at all, and is dragged down by tokens that passed through the bracket")
    print("  years before any venue listed a perpetual on them, including names the")
    print("  book never holds.")

    print(f"\nsaved -> {out_dir / 'tradability_corrected.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
