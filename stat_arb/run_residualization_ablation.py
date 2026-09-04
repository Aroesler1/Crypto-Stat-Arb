"""Step 1: what is "the market" for each bracket?

The original report residualised against ETH. The rebuild residualised against a
PCA market mode. Those are different claims about what the common factor is, and
neither was tested against the other, so this runs the choice as an explicit
ablation with the signal held fixed.

Arms
----
``eth``        ETH-excess only, no PCA. The report's own convention.
``btc``        BTC-excess only, no PCA.
``vw``         excess over the value-weighted market of the full panel.
``eth+pca1``   ETH-excess, then the first principal component removed. This is
``eth+pca2``   what the rebuild does.
``eth+pca3``

Reported per bracket per arm
----------------------------
``var_removed``    fraction of cross-sectional variance the residualization
                   takes out, averaged over rebalances
``density``        share of possible pairs carrying an edge in the signed k-NN
                   graph, and the share of those edges that are negative
``stability``      adjusted Rand index between consecutive monthly clusterings,
                   on the assets common to both. A residualization that leaves a
                   dominant factor in place produces clusterings that reshuffle
                   every month, and an unstable partition cannot be traded
``net_sharpe``     walk-forward net Sharpe at 50 bps, the criterion that matters

B0 is excluded here: it never reaches 30 members, so it is not clustered at all
and is handled separately. Every arm runs on the point-in-time universe with the
$50k/day liquidity floor, matching the published baseline.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.data import brackets as B  # noqa: E402
from stat_arb.data import daily_listings as D  # noqa: E402
from stat_arb.data.universe import UniverseManager  # noqa: E402
from stat_arb.backtest.statistics import probabilistic_sharpe_ratio  # noqa: E402
from stat_arb.run_phase3 import annualized_sharpe, run_phase3_config  # noqa: E402

BTC_CMC_ID = 1
PERIODS_PER_YEAR = 365
BEST_BAND = 0.02
BEST_FREQ = 3
MIN_VOLUME_USD = 50_000

# (arm label, reference series, number of PCs removed on top of it)
ARMS = (
    ("eth", "eth", 0),
    ("btc", "btc", 0),
    ("vw", "vw_market", 0),
    ("eth+pca1", "eth", 1),
    ("eth+pca2", "eth", 2),
    ("eth+pca3", "eth", 3),
)

CLUSTERED_BRACKETS = ("B1", "B2", "B3")


def cluster_stability(diagnostics: list[dict]) -> float:
    """Mean adjusted Rand index between consecutive monthly clusterings.

    Compared on the assets present in both rebalances, because the universe
    reconstitutes: a token that entered or left cannot agree or disagree about
    its cluster. ARI is chance-corrected, so 0 is what independent relabelling
    would give and 1 is an identical partition.
    """
    scores = []
    for prev, cur in zip(diagnostics, diagnostics[1:]):
        common = sorted(set(prev["assets"]) & set(cur["assets"]))
        if len(common) < 10:
            continue
        pi = {a: i for i, a in enumerate(prev["assets"])}
        ci = {a: i for i, a in enumerate(cur["assets"])}
        scores.append(adjusted_rand_score(
            [prev["labels"][pi[a]] for a in common],
            [cur["labels"][ci[a]] for a in common]))
    return float(np.mean(scores)) if scores else np.nan


def load_inputs(panel_path: Path, data_dir: Path, start: pd.Timestamp,
                end: pd.Timestamp):
    """Close/volume/mcap panels, the reference series, and bracket membership."""
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[(panel["date"] >= start - pd.Timedelta(days=400)) & (panel["date"] <= end)]
    index = pd.DatetimeIndex(sorted(panel["date"].unique()))

    table = pd.read_parquet(data_dir / "bracket_universe.parquet")
    assignments = pd.read_parquet(data_dir / "bracket_assignments.parquet")
    keep_ids = set(table["cmc_id"].astype(int))
    kept = panel[panel["cmc_id"].isin(keep_ids)]

    close = D.wide(kept, "price_usd", index)
    volume = D.wide(kept, "volume_24h_usd", index)
    mcap = D.wide(kept, "market_cap_usd", index)

    refs = pd.DataFrame({
        "eth": D.wide(panel[panel["cmc_id"] == B.ETH_CMC_ID], "price_usd", index)
                .get(B.ETH_CMC_ID, pd.Series(np.nan, index=index)),
        "btc": D.wide(panel[panel["cmc_id"] == BTC_CMC_ID], "price_usd", index)
                .get(BTC_CMC_ID, pd.Series(np.nan, index=index)),
        "vw_market": D.value_weighted_market(close, mcap),
    })
    return table, assignments, close, volume, mcap, refs, index


def run_arm(close, volume, table, refs, reference, n_pca, member_mask, index,
            clusterer=None):
    """One ablation arm on one bracket. Returns stats plus the diagnostics.

    `clusterer` defaults to the published SPONGE k=3 path. The clustering-method
    sweep passes its own, which is why this takes one rather than hard-coding
    the method: the residualization ablation and the method comparison then run
    through exactly the same universe construction, liquidity filter and
    backtest, and differ only in the thing each is varying.
    """
    returns, _ = D.excess_log_returns(close, refs[reference], table)
    cols = [c for c in member_mask.columns if c in returns.columns]
    if not cols:
        return None
    returns = returns[cols]
    returns.columns = [f"{int(c)}_returns" for c in cols]

    prices = close[cols].copy()
    prices.columns = [str(int(c)) for c in cols]
    vols = volume[cols].copy()
    vols.columns = [str(int(c)) for c in cols]
    eth_data = pd.DataFrame({"close": refs["eth"], "volume": np.nan}, index=index)

    univ = UniverseManager(mcap_percentile_low=0.0, mcap_percentile_high=1.0,
                           min_volume_usd=MIN_VOLUME_USD, min_history_days=60,
                           volume_in_usd=True)
    liquid = univ.get_universe_membership(prices, vols, eth_data, returns)
    band = member_mask[cols].copy()
    band.columns = [str(int(c)) for c in cols]
    mask = liquid & band.reindex(index=liquid.index, columns=liquid.columns,
                                 fill_value=False)
    if mask.to_numpy().sum() == 0:
        return None

    diagnostics: list[dict] = []
    result = run_phase3_config(returns, mask, weight_band=BEST_BAND,
                               trade_frequency_days=BEST_FREQ,
                               n_pca_components=n_pca, diagnostics=diagnostics,
                               clusterer=clusterer)
    if result is None:
        return None

    d = pd.DataFrame([{k: v for k, v in x.items() if k not in ("labels", "assets")}
                      for x in diagnostics])
    return {
        "avg_members": float(mask.sum(axis=1).mean()),
        "var_removed": float(d["variance_removed"].mean()) if len(d) else np.nan,
        "density": float(d["graph_density"].mean()) if len(d) else np.nan,
        "neg_edge_share": float(d["negative_edge_share"].mean()) if len(d) else np.nan,
        "stability_ari": cluster_stability(diagnostics),
        "gross_sharpe": annualized_sharpe(result["gross_returns"]),
        "net_sharpe": annualized_sharpe(result["net_50"]),
        "breakeven_bps": result["breakeven"],
        "turnover": float(result["turnover"].mean()),
        "n_rebalances": len(diagnostics),
        # PSR on the net series: the probability the true Sharpe clears zero,
        # given this sample's length, skew and kurtosis. Reported on net rather
        # than gross because a gross PSR flatters a book nobody can trade.
        "psr": float(probabilistic_sharpe_ratio(result["net_50"])),
        # Kept so the tradability step can charge funding against the positions
        # actually held. Callers that only want the summary pop these.
        "net_series": result["net_50"],
        "net_series_for_funding": result["net_50"],
        "weights": result["weights"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--start", default=str(B.BRACKET_START.date()))
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--brackets", default=",".join(CLUSTERED_BRACKETS))
    ap.add_argument("--treatment", default="pit",
                    choices=("pit", "survivor-only", "snapshot"))
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    panel_path = Path(args.panel) if args.panel else data_dir / "pit_daily_listings.parquet"
    if not panel_path.exists():
        print(f"{panel_path} not found; run stat_arb/build_daily_listings.py first")
        return 1

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    print(f"loading panels {start.date()}..{end.date()} ...", flush=True)
    table, assignments, close, volume, mcap, refs, index = load_inputs(
        panel_path, data_dir, start, end)
    print(f"  {len(close.columns)} tokens, {len(index)} days, "
          f"references: {list(refs.columns)}")

    rows = []
    for bracket in [b.strip() for b in args.brackets.split(",") if b.strip()]:
        ids = sorted({int(c) for c in
                      assignments.loc[assignments["bracket"] == bracket, "cmc_id"]})
        ids = [c for c in ids if c in close.columns]
        member = B.bracket_membership(assignments, index, bracket, columns=ids)
        if args.treatment != "pit":
            dead = set(table.loc[table["delisted"], "cmc_id"].astype(int))
            for c in member.columns:
                if int(c) in dead:
                    member[c] = False
        for label, reference, n_pca in ARMS:
            print(f"  {bracket} / {label} ...", flush=True)
            stats = run_arm(close, volume, table, refs, reference, n_pca, member, index)
            if stats is None:
                print("    produced no positions, skipped")
                continue
            for _k in ("net_series", "net_series_for_funding", "weights"):
                stats.pop(_k, None)
            stats.update(bracket=bracket, arm=label, reference=reference,
                         n_pca=n_pca, treatment=args.treatment)
            rows.append(stats)

    if not rows:
        print("no arm produced a result")
        return 1

    out = pd.DataFrame(rows)
    out_dir = root / "stat_arb" / "reporting" / "brackets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"residualization_ablation_{args.treatment}.csv", index=False)

    print(f"\n=== residualization ablation ({args.treatment}, "
          f"band {BEST_BAND:.0%}, rebalance {BEST_FREQ}d, net 50bps) ===")
    print("  bracket  arm        members  var_rm  density  neg%  stability  gross    net  breakeven")
    for bracket in out["bracket"].unique():
        sub = out[out["bracket"] == bracket]
        for _, r in sub.iterrows():
            print(f"  {r['bracket']:<7s}  {r['arm']:<9s}  {r['avg_members']:7.0f}  "
                  f"{r['var_removed']:6.1%}  {r['density']:7.1%}  "
                  f"{100 * r['neg_edge_share']:4.0f}  {r['stability_ari']:9.3f}  "
                  f"{r['gross_sharpe']:5.2f}  {r['net_sharpe']:5.2f}  "
                  f"{r['breakeven_bps']:7.0f}")
        best = sub.loc[sub["net_sharpe"].idxmax()]
        print(f"    best net Sharpe for {bracket}: {best['arm']} "
              f"({best['net_sharpe']:.2f})")

    print(f"\nsaved -> {out_dir / f'residualization_ablation_{args.treatment}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
