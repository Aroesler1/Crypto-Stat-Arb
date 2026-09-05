"""Step 3: better signals inside the same cluster structure.

Each arm changes exactly one thing about how the deviation from a cluster is
measured or sized, holding the universe, the residualization, the clustering and
the execution controls fixed. Reported per bracket per survivorship treatment
with net Sharpe at 50 bps, PSR, DSR, turnover and breakeven cost.

Arms
----
``baseline``    ClusterDeviationStrategy, the published signal, as the control
``ou``          Avellaneda-Lee s-score with the half-life filter on
``ou_nofilter`` the same without the half-life filter, to price what it costs
``beta``        regress on the cluster composite, trade the regression residual
``ewma``        EWMA z-scoring plus half-life-proportional position size
``momentum``    within-cluster reversion in trailing-winner clusters only
``death``       baseline with a walk-forward death classifier gating the long leg

The death filter is the one that should matter most on a point-in-time
universe. A mean-reversion book buys losers, and on this panel a meaningful
share of the losers are about to be delisted. If death is forecastable even
weakly, declining to buy the worst-scoring names is the cheapest available fix.
DSR treats the seven arms as the multiple-testing pool.
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

from stat_arb.backtest.statistics import (  # noqa: E402
    deflated_sharpe_ratio, probabilistic_sharpe_ratio,
)
from stat_arb.data import brackets as B  # noqa: E402
from stat_arb.data import death_model as DM  # noqa: E402
from stat_arb.signals.cluster_deviation import ClusterDeviationStrategy  # noqa: E402
from stat_arb.signals.ou_score import (  # noqa: E402
    BetaAdjustedDeviation, ClusterMomentumOverlay, OUScoreStrategy,
    ewma_zscore, half_life_position_scale,
)
from stat_arb.run_phase3 import annualized_sharpe  # noqa: E402
from stat_arb.run_residualization_ablation import (  # noqa: E402
    BEST_BAND, BEST_FREQ, load_inputs, run_arm,
)

# The reference each bracket residualizes against, from the Step 1 ablation.
BEST_REFERENCE = {"B0": "btc", "B1": "eth", "B2": "vw_market", "B3": "eth"}

# Share of the cross-section the death filter refuses to go long, each day.
#
# An absolute probability cutoff does not work here. The classifier is fitted
# with class_weight="balanced" on a label whose base rate is about 1.3%, so its
# scores centre on 0.50 by construction: the median out-of-sample probability is
# 0.497 and only 0.41% of them ever exceed 0.80. A 0.80 cutoff gated so few
# names that the filtered book was identical to the unfiltered one in every
# bracket, which reads as "death is not forecastable" when it actually means
# "the gate never fired". Ranking within each day and refusing the worst decile
# is scale-free and does what the arm is supposed to test.
DEATH_WORST_FRACTION = 0.10


class EWMAClusterDeviation(ClusterDeviationStrategy):
    """Cluster deviation with EWMA standardisation and half-life sizing.

    Subclasses the control rather than reimplementing it, so the only difference
    is the standardisation and the position scale.
    """

    def __init__(self, lookback=5, zscore_window=20, halflife=10.0, **kw):
        super().__init__(lookback=lookback, zscore_window=zscore_window, **kw)
        self.halflife = halflife

    def compute_signals(self, returns, cluster_labels, tokens, lag=1):
        token_to_cluster = {t: cluster_labels[i] for i, t in enumerate(tokens)}
        col_token = {c: c.replace("_returns", "") for c in returns.columns}
        dev = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        for cluster in np.unique(cluster_labels):
            cols = [c for c in returns.columns
                    if token_to_cluster.get(col_token[c], None) == cluster]
            if len(cols) < 2:
                continue
            dev[cols] = returns[cols].sub(returns[cols].mean(axis=1), axis=0)
        cumulative = dev.rolling(self.lookback, min_periods=2).sum()
        return ewma_zscore(cumulative, halflife=self.halflife).shift(lag)


def make_death_filter(death_probs: pd.DataFrame, cols_to_id: dict,
                      worst_fraction: float = DEATH_WORST_FRACTION):
    """Zero the LONG leg of the `worst_fraction` most death-prone names.

    Ranked within each rebalance date rather than against an absolute
    probability, for the reason given at `DEATH_WORST_FRACTION`.

    Longs only. The short leg is left alone deliberately: a token about to die
    is a fine thing to be short, and gating both legs would confuse "avoid the
    losers" with "trade less".
    """
    if death_probs.empty:
        return None
    wide = death_probs.pivot_table(index="date", columns="cmc_id",
                                   values="death_prob", aggfunc="last")

    def filt(weights, assets, dates):
        idx = wide.index[wide.index <= dates[0]]
        if len(idx) == 0:
            return weights
        latest = pd.to_numeric(wide.loc[idx[-1]], errors="coerce").dropna()
        held = {c: cols_to_id[c] for c in weights.columns
                if cols_to_id.get(c) in latest.index}
        if len(held) < 10:
            return weights
        scores = latest[list(held.values())]
        cutoff = scores.quantile(1.0 - float(worst_fraction))
        risky = [c for c, cid in held.items() if float(latest[cid]) >= cutoff]
        if not risky:
            return weights
        out = weights.copy()
        out[risky] = out[risky].where(out[risky] < 0, 0.0)   # keep shorts, drop longs
        return out

    return filt


def arm_kwargs(name: str, death_filter=None) -> dict:
    if name == "baseline":
        return {}
    if name == "ou":
        return {"strategy_factory": lambda H, L: OUScoreStrategy(
            window=60, max_half_life=3.0)}
    if name == "ou_nofilter":
        return {"strategy_factory": lambda H, L: OUScoreStrategy(
            window=60, max_half_life=None)}
    if name == "beta":
        return {"strategy_factory": lambda H, L: BetaAdjustedDeviation(
            beta_window=60, zscore_window=L)}
    if name == "ewma":
        return {"strategy_factory": lambda H, L: EWMAClusterDeviation(
            lookback=H, zscore_window=L, halflife=10.0)}
    if name == "momentum":
        overlay = ClusterMomentumOverlay(momentum_window=28, top_frac=0.5)
        return {"cluster_selector": overlay.clusters_to_trade}
    if name == "death":
        return {"weight_filter": death_filter}
    raise ValueError(name)


ARMS = ("baseline", "ou", "ou_nofilter", "beta", "ewma", "momentum", "death")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--start", default=str(B.BRACKET_START.date()))
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--brackets", default="B1,B2,B3")
    ap.add_argument("--treatments", default="pit,survivor-only")
    ap.add_argument("--arms", default=",".join(ARMS))
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
    dead = set(table.loc[table["delisted"], "cmc_id"].astype(int))

    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]
    death_filter = None
    if "death" in arms:
        print("fitting the death classifier walk-forward ...", flush=True)
        raw = pd.read_parquet(panel_path)
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw[raw["cmc_id"].isin(set(table["cmc_id"].astype(int)))]
        features = DM.build_features(raw, table, sample_every=7)
        metrics, coefs = DM.walk_forward_auc(features)
        scored = metrics[~metrics["skipped"]]
        print(metrics.to_string(index=False))
        if len(scored):
            print(f"  mean out-of-sample AUC {scored['auc'].mean():.3f} "
                  f"over {len(scored)} years")
        print("  standardised coefficients by year:")
        print(coefs.to_string(index=False))
        out_dir = root / "stat_arb" / "reporting" / "brackets"
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out_dir / "death_classifier_auc.csv", index=False)
        coefs.to_csv(out_dir / "death_classifier_coefficients.csv", index=False)
        probs = DM.death_probabilities(features)
        cols_to_id = {f"{int(c)}_returns": int(c) for c in close.columns}
        death_filter = make_death_filter(probs, cols_to_id)

    rows, series = [], {}
    for bracket in [b.strip() for b in args.brackets.split(",") if b.strip()]:
        ids = sorted({int(c) for c in
                      assignments.loc[assignments["bracket"] == bracket, "cmc_id"]})
        ids = [c for c in ids if c in close.columns]
        base = B.bracket_membership(assignments, index, bracket, columns=ids)
        reference = BEST_REFERENCE.get(bracket, "eth")
        for treatment in [t.strip() for t in args.treatments.split(",") if t.strip()]:
            member = base.copy()
            if treatment != "pit":
                for c in member.columns:
                    if int(c) in dead:
                        member[c] = False
            for arm in arms:
                print(f"  {bracket} / {treatment} / {arm} ...", flush=True)
                try:
                    stats = run_arm(close, volume, table, refs, reference, 0,
                                    member, index, **arm_kwargs(arm, death_filter))
                except Exception as exc:
                    print(f"    failed: {type(exc).__name__}: {exc}")
                    continue
                if stats is None:
                    print("    produced no positions, skipped")
                    continue
                for _k in ("net_series_for_funding", "weights"):
                    stats.pop(_k, None)
                stats.update(bracket=bracket, treatment=treatment, arm=arm,
                             reference=reference)
                rows.append(stats)

    if not rows:
        print("no arm produced a result")
        return 1

    # DSR treats the arms tried within one (bracket, treatment) cell as the
    # multiple-testing pool, which is what they are: seven shots at the same
    # universe. Computed after the fact because it needs the whole family.
    for r in rows:
        family = [x for x in rows if x["bracket"] == r["bracket"]
                  and x["treatment"] == r["treatment"]]
        # DSR wants PER-PERIOD Sharpes, and net_sharpe is annualised. Passing
        # the annualised figures inflates the cross-trial variance by 365 and
        # drives the deflated Sharpe to exactly zero for every arm, which is
        # what the first run reported.
        trial_sharpes = [float(x["net_sharpe"]) / np.sqrt(365.0) for x in family]
        try:
            r["dsr"] = float(deflated_sharpe_ratio(
                r.pop("net_series"), n_trials=len(family),
                trial_sharpes=trial_sharpes)["dsr"])
        except Exception:
            r.pop("net_series", None)
            r["dsr"] = np.nan
    for r in rows:
        r.pop("net_series", None)

    out = pd.DataFrame(rows)
    out_dir = root / "stat_arb" / "reporting" / "brackets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "signal_ablation.csv", index=False)

    print(f"\n=== signal extensions (band {BEST_BAND:.0%}, rebalance {BEST_FREQ}d, "
          f"net 50bps; reference per bracket from Step 1) ===")
    for bracket in out["bracket"].unique():
        for treatment in out["treatment"].unique():
            sub = out[(out["bracket"] == bracket) & (out["treatment"] == treatment)]
            if sub.empty:
                continue
            print(f"\n  {bracket} / {treatment}  (reference {sub['reference'].iloc[0]}, "
                  f"avg members {sub['avg_members'].iloc[0]:.0f})")
            print("  arm            gross    net    PSR    DSR  turnover  breakeven")
            for _, r in sub.sort_values("net_sharpe", ascending=False).iterrows():
                print(f"  {r['arm']:<13s} {r['gross_sharpe']:6.2f} {r['net_sharpe']:6.2f} "
                      f"{r.get('psr', np.nan):6.3f} {r.get('dsr', np.nan):6.3f} "
                      f"{r['turnover']:9.3f} {r['breakeven_bps']:10.0f}")

    print(f"\nsaved -> {out_dir / 'signal_ablation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
