"""Step 5: factor diagnostics. SECONDARY to the bracket result, and labelled so.

Two questions, neither of which is the strategy question.

(a) **Is within-cluster mean reversion anything more than short-term reversal
    plus a size bet, and does the answer change by bracket?** Each bracket's
    stat-arb book is regressed on crypto factors built on the academic universe
    (CoinMarketCap rank 1-1000 with a market cap above $1M), and the alpha,
    loadings and R-squared are reported. A book that is mostly loading on
    reversal is not evidence that clustering does anything.

(b) **How much of the published crypto factor zoo is survivorship?** Every
    factor is built three times on universes differing only in what they can
    see (point-in-time, survivor-only, snapshot), and the per-factor Sharpe
    difference is tested across factors with Romano-Wolf, which holds the
    family-wise error rate rather than testing twenty-odd factors at 5% each and
    collecting the false positives.

Cited for the factor set and the survivorship treatment: Liu, Tsyvinski and Wu
(JF 2022); Ammann, Burdorf, Liebi and Stoeckl (SSRN 4287573); Borri, Liu,
Tsyvinski and Wu (arXiv 2510.14435); Fieberg, Guenther, Poddig and Zaremba
(Quantitative Finance 2023, JFQA 2025); Dobrynskaya (SSRN 3913263); Mercik,
Zaremba and Demir (IRFA 2026).
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

from stat_arb.backtest.statistics import romano_wolf_stepdown  # noqa: E402
from stat_arb.data import brackets as B  # noqa: E402
from stat_arb.data import daily_listings as D  # noqa: E402
from stat_arb.factors import characteristics as C  # noqa: E402
from stat_arb.factors import portfolios as PF  # noqa: E402
from stat_arb.run_residualization_ablation import load_inputs, run_arm  # noqa: E402
from stat_arb.run_signal_ablation import BEST_REFERENCE, arm_kwargs  # noqa: E402

ACADEMIC_RANK_HI = 1000
ACADEMIC_MIN_MCAP = 1_000_000
# Factors used as regressors in (a). The full zoo is swept in (b); this is the
# subset the question is about, so the regression stays interpretable.
BOOK_FACTORS = ("size", "mom_1w", "mom_2w", "mom_4w", "reversal_1w",
                "amihud", "volatility")


def academic_membership(panel: pd.DataFrame, index: pd.DatetimeIndex,
                        columns: list[int]) -> pd.DataFrame:
    """Rank 1-1000 with a market cap above $1M, daily, point-in-time."""
    p = panel[(panel["rank"] <= ACADEMIC_RANK_HI)
              & (pd.to_numeric(panel["market_cap_usd"], errors="coerce") > ACADEMIC_MIN_MCAP)]
    wide = (p.assign(flag=True)
             .pivot_table(index="date", columns="cmc_id", values="flag", aggfunc="last")
             .reindex(index=index, columns=[int(c) for c in columns])
             .fillna(False).astype(bool))
    return wide


def ols(y: pd.Series, X: pd.DataFrame) -> dict:
    """Least squares with an intercept, plus Newey-West style plain t-stats.

    Standard errors are the classical OLS ones. The series here are weekly and
    the regressors are traded portfolios, so the residual autocorrelation that
    would demand a HAC correction is small; the alternative would be another
    dependency for a secondary table.
    """
    joined = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(joined) < len(X.columns) + 10:
        return {}
    yv = joined["y"].to_numpy()
    Xv = np.column_stack([np.ones(len(joined)), joined[X.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ beta
    dof = len(yv) - Xv.shape[1]
    s2 = float(resid @ resid) / max(dof, 1)
    cov = s2 * np.linalg.pinv(Xv.T @ Xv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    out = {"alpha": float(beta[0]),
           "alpha_t": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
           "r2": 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan,
           "n_obs": int(len(yv))}
    for i, name in enumerate(X.columns, start=1):
        out[f"beta_{name}"] = float(beta[i])
        out[f"t_{name}"] = float(beta[i] / se[i]) if se[i] > 0 else np.nan
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--start", default=str(B.BRACKET_START.date()))
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--brackets", default="B1,B2,B3")
    ap.add_argument("--cost-bps", type=float, default=50.0)
    ap.add_argument("--skip-books", action="store_true",
                    help="only build the factor table, not the (a) regressions")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    panel_path = Path(args.panel) if args.panel else data_dir / "pit_daily_listings.parquet"
    if not panel_path.exists():
        print(f"{panel_path} not found")
        return 1

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    print(f"loading panels {start.date()}..{end.date()} ...", flush=True)
    table, assignments, close, volume, mcap, refs, index = load_inputs(
        panel_path, data_dir, start, end)

    raw = pd.read_parquet(panel_path)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[(raw["date"] >= index.min()) & (raw["date"] <= index.max())]
    raw = raw[raw["cmc_id"].isin(set(int(c) for c in close.columns))]

    ids = [int(c) for c in close.columns]
    academic = academic_membership(raw, index, ids)
    print(f"  academic universe: rank <= {ACADEMIC_RANK_HI}, mcap > ${ACADEMIC_MIN_MCAP:,}; "
          f"avg {academic.sum(axis=1).mean():.0f} members")

    market = pd.Series(np.log(refs["vw_market"] / refs["vw_market"].shift(1)),
                       index=index).fillna(0.0)
    simple = close.where(close > 0).pct_change()
    # scrub the same artifacts the return panel scrubs, so the factor table and
    # the strategy see the same data
    log_r = np.log1p(simple.where(simple > -1))
    scrubbed, n_scrubbed = D.scrub_extreme_returns(log_r)
    simple = np.expm1(scrubbed)
    print(f"  scrubbed {n_scrubbed:,} extreme daily observations from the factor panel")

    print("building characteristics ...", flush=True)
    chars = C.build_characteristics(close, volume, mcap, market)
    signs = {k: C.CHARACTERISTICS[k][1] for k in chars}

    out_dir = root / "stat_arb" / "reporting" / "brackets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- (b) the three-universe factor table -------------------------------
    dead = set(table.loc[table["delisted"], "cmc_id"].astype(int))
    coverage = close.notna().mean()
    complete = {int(c) for c in coverage[coverage >= 0.99].index}

    survivor = academic.copy()
    for c in survivor.columns:
        if int(c) in dead:
            survivor[c] = False
    snapshot = survivor.copy()
    for c in snapshot.columns:
        if int(c) not in complete:
            snapshot[c] = False

    universes = {"pit": academic, "survivor-only": survivor, "snapshot": snapshot}
    all_series, all_summary = {}, []
    for label, member in universes.items():
        print(f"  factor sweep on {label} ({member.sum(axis=1).mean():.0f} avg members) ...",
              flush=True)
        series, summary = PF.factor_returns(chars, simple, mcap, member, signs,
                                            cost_bps=args.cost_bps)
        summary["universe"] = label
        all_series[label] = series
        all_summary.append(summary)

    factor_table = pd.concat(all_summary, ignore_index=True)
    factor_table.to_csv(out_dir / "factor_table_three_universes.csv", index=False)

    wide = factor_table.pivot(index="factor", columns="universe", values="net_sharpe")
    for col in ("pit", "survivor-only", "snapshot"):
        if col not in wide.columns:
            wide[col] = np.nan
    wide["pit_minus_survivor"] = wide["pit"] - wide["survivor-only"]
    wide = wide.sort_values("pit_minus_survivor")

    print(f"\n=== (b) factor table, three universes "
          f"(weekly, quintile, value-weighted, {args.cost_bps:.0f} bps) ===")
    print("  factor            PIT   surv   snap   PIT-surv")
    for name, r in wide.iterrows():
        print(f"  {name:<15s} {r['pit']:6.2f} {r['survivor-only']:6.2f} "
              f"{r['snapshot']:6.2f} {r['pit_minus_survivor']:10.2f}")

    # Romano-Wolf across factors on the PAIRED difference series
    diffs = {}
    pit_s, surv_s = all_series.get("pit"), all_series.get("survivor-only")
    if pit_s is not None and surv_s is not None:
        for name in pit_s.columns:
            if name in surv_s.columns:
                joined = pd.concat([pit_s[name], surv_s[name]], axis=1,
                                   join="inner").dropna()
                if len(joined) > 30:
                    diffs[name] = joined.iloc[:, 0] - joined.iloc[:, 1]
    if len(diffs) >= 2:
        diff_frame = pd.DataFrame(diffs).dropna(how="all")
        rw = romano_wolf_stepdown(diff_frame, n_boot=1000, two_sided=True)
        rw.to_csv(out_dir / "factor_survivorship_romano_wolf.csv", index=False)
        n_sig = int(rw["significant"].sum())
        print(f"\n  Romano-Wolf across {len(diff_frame.columns)} factors on the paired")
        print(f"  point-in-time minus survivor-only weekly difference, FWER 5%, two-sided:")
        print(f"  {n_sig} factor(s) significant")
        for _, r in rw.head(8).iterrows():
            mark = "*" if r["significant"] else " "
            print(f"    {mark} {r['strategy']:<15s} t {r['t_stat']:7.2f}  "
                  f"adj p {r['adjusted_p']:.3f}")

    # --- (a) the bracket books on the factors ------------------------------
    if not args.skip_books:
        abl_path = out_dir / "signal_ablation.csv"
        best_arm = {}
        if abl_path.exists():
            abl = pd.read_csv(abl_path)
            pit = abl[abl["treatment"] == "pit"]
            for bracket, g in pit.groupby("bracket"):
                best_arm[bracket] = g.loc[g["net_sharpe"].idxmax(), "arm"]

        factor_weekly = all_series["pit"][[f for f in BOOK_FACTORS
                                           if f in all_series["pit"].columns]]
        rows = []
        for bracket in [b.strip() for b in args.brackets.split(",") if b.strip()]:
            bids = sorted({int(c) for c in
                           assignments.loc[assignments["bracket"] == bracket, "cmc_id"]})
            bids = [c for c in bids if c in close.columns]
            member = B.bracket_membership(assignments, index, bracket, columns=bids)
            arm = best_arm.get(bracket, "baseline")
            print(f"\n  running {bracket} book (arm {arm}) for the factor regression ...",
                  flush=True)
            stats = run_arm(close, volume, table, refs,
                            BEST_REFERENCE.get(bracket, "eth"), 0, member, index,
                            **arm_kwargs(arm, None))
            if stats is None:
                continue
            daily = stats["net_series_for_funding"]
            weekly = (1.0 + daily.fillna(0.0)).resample("W-MON").prod() - 1.0
            res = ols(weekly, factor_weekly)
            if res:
                res.update(bracket=bracket, arm=arm)
                rows.append(res)

        if rows:
            books = pd.DataFrame(rows)
            books.to_csv(out_dir / "book_factor_regressions.csv", index=False)
            print(f"\n=== (a) each bracket's book on academic-universe factors "
                  f"(weekly, PIT) ===")
            print("  bracket  alpha    t   R2   " +
                  "  ".join(f"{f[:9]:>9s}" for f in factor_weekly.columns))
            for _, r in books.iterrows():
                loadings = "  ".join(f"{r.get(f'beta_{f}', np.nan):9.2f}"
                                     for f in factor_weekly.columns)
                print(f"  {r['bracket']:<7s} {r['alpha']:+.4f} {r['alpha_t']:5.2f} "
                      f"{r['r2']:5.2f}   {loadings}")

    print(f"\nsaved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
