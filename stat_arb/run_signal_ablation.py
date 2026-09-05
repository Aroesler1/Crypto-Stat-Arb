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
    ewma_zscore, half_life_position_scale, s_scores,
)
from stat_arb.run_phase3 import annualized_sharpe  # noqa: E402

PERIODS_PER_YEAR = 365

ARMS = ("baseline", "ou", "ou_nofilter", "beta", "ewma", "ewma_sized",
        "momentum", "death")

# The EWMA half-life was fixed at 10 days before any of these results were seen,
# and the sweep below is reported as ROBUSTNESS, not as a selection. The 10-day
# row is the pre-set default; 5 and 20 exist so a reader can see whether the
# result depends on the choice.
EWMA_DEFAULT_HALFLIFE = 10.0
EWMA_HALFLIFE_SWEEP = (5.0, 10.0, 20.0)
ROBUSTNESS_BRACKET = "B3"
N_BEST_DAYS = 10

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
    """Cluster deviation standardised with an EWMA instead of a rolling window.

    This arm changes the STANDARDISATION and nothing else. It does not size by
    half-life; `EWMASizedClusterDeviation` below does that, and the two run as
    separate arms so the table can say which half of the brief's item (c) pays.

    A 20-day rolling standard deviation steps discontinuously every time a large
    move enters or leaves the window, and on a micro-cap panel that step is
    noise injected into the denominator of every signal sharing it. An EWMA
    decays instead of dropping.
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


class EWMASizedClusterDeviation(EWMAClusterDeviation):
    """EWMA standardisation PLUS position size proportional to signal half-life.

    Completes the brief's item (c): a signal expected to revert in a day is
    worth more per unit of z-score than one expected to take a week, because the
    book only holds it for the rebalance interval. Half-lives come from the same
    OU fit the s-score arm uses, so the two arms are measuring the same
    quantity.
    """

    def __init__(self, lookback=5, zscore_window=20, halflife=10.0,
                 ou_window=60, size_target=3.0, **kw):
        super().__init__(lookback=lookback, zscore_window=zscore_window,
                         halflife=halflife, **kw)
        self.ou_window = ou_window
        self.size_target = size_target
        self.half_lives_ = None

    def compute_signals(self, returns, cluster_labels, tokens, lag=1):
        signals = super().compute_signals(returns, cluster_labels, tokens, lag=lag)
        token_to_cluster = {t: cluster_labels[i] for i, t in enumerate(tokens)}
        col_token = {c: c.replace("_returns", "") for c in returns.columns}
        dev = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        for cluster in np.unique(cluster_labels):
            cols = [c for c in returns.columns
                    if token_to_cluster.get(col_token[c], None) == cluster]
            if len(cols) < 2:
                continue
            dev[cols] = returns[cols].sub(returns[cols].mean(axis=1), axis=0)
        # max_half_life=None: this arm SIZES by the half-life rather than
        # filtering on it, so nothing is dropped for reverting slowly.
        _, half_lives = s_scores(dev.dropna(axis=1, how="all"),
                                 self.ou_window, max_half_life=None)
        self.half_lives_ = half_lives.reindex(columns=returns.columns).shift(lag)
        return signals

    def generate_target_weights(self, signals, cluster_labels, tokens,
                                returns=None, clusters_to_trade=None):
        weights = super().generate_target_weights(signals, cluster_labels, tokens,
                                                  returns, clusters_to_trade)
        if self.half_lives_ is None:
            return weights
        scale = half_life_position_scale(
            self.half_lives_.reindex(index=weights.index, columns=weights.columns),
            target=self.size_target)
        return weights * scale.fillna(0.25)


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


def sharpe_by_year(net: pd.Series) -> dict[int, float]:
    """Annualised net Sharpe within each calendar year."""
    r = pd.to_numeric(net, errors="coerce").dropna()
    out = {}
    for year, g in r.groupby(r.index.year):
        if len(g) < 30 or g.std(ddof=1) == 0:
            continue
        out[int(year)] = float(g.mean() / g.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    return out


def best_days_stats(net: pd.Series, k: int = N_BEST_DAYS) -> dict:
    """How much of the result rides on a handful of days.

    A book whose Sharpe collapses once its best `k` days are removed is not a
    strategy, it is a few lucky sessions. Reported as the Sharpe excluding those
    days and their share of total return, both of which have to be read together:
    a large share with an unchanged Sharpe means the good days were big but the
    rest still worked.
    """
    r = pd.to_numeric(net, errors="coerce").dropna()
    if len(r) < k + 30:
        return {"sharpe_ex_best": np.nan, "best_days_share": np.nan}
    best = r.nlargest(k).index
    rest = r.drop(best)
    total = float(r.sum())
    return {
        "sharpe_ex_best": (float(rest.mean() / rest.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
                           if rest.std(ddof=1) > 0 else np.nan),
        "best_days_share": (float(r.loc[best].sum() / total) if total != 0 else np.nan),
    }


def robustness_row(net: pd.Series, turnover: float, **meta) -> dict:
    row = dict(meta)
    row["net_sharpe"] = (float(net.mean() / net.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
                         if net.std(ddof=1) > 0 else np.nan)
    row["turnover"] = float(turnover)
    row["n_days"] = int(net.notna().sum())
    row.update(best_days_stats(net))
    for year, value in sharpe_by_year(net).items():
        row[f"sharpe_{year}"] = value
    return row


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
            lookback=H, zscore_window=L, halflife=EWMA_DEFAULT_HALFLIFE)}
    if name == "ewma_sized":
        return {"strategy_factory": lambda H, L: EWMASizedClusterDeviation(
            lookback=H, zscore_window=L, halflife=EWMA_DEFAULT_HALFLIFE)}
    if name.startswith("ewma_hl"):
        hl = float(name.split("ewma_hl")[1])
        return {"strategy_factory": lambda H, L, _hl=hl: EWMAClusterDeviation(
            lookback=H, zscore_window=L, halflife=_hl)}
    if name == "momentum":
        overlay = ClusterMomentumOverlay(momentum_window=28, top_frac=0.5)
        return {"cluster_selector": overlay.clusters_to_trade}
    if name == "death":
        return {"weight_filter": death_filter}
    raise ValueError(name)




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
    ap.add_argument("--no-halflife-sweep", action="store_true")
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

    rows, series, robustness = [], {}, []
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
                if bracket == ROBUSTNESS_BRACKET:
                    robustness.append(robustness_row(
                        stats["net_series_for_funding"].dropna(), stats["turnover"],
                        bracket=bracket, treatment=treatment, arm=arm,
                        halflife=(EWMA_DEFAULT_HALFLIFE
                                  if arm in ("ewma", "ewma_sized") else np.nan),
                        role="arm"))
                for _k in ("net_series_for_funding", "weights"):
                    stats.pop(_k, None)
                stats.update(bracket=bracket, treatment=treatment, arm=arm,
                             reference=reference)
                rows.append(stats)

    # Half-life sensitivity, reported as ROBUSTNESS rather than selection: the
    # 10-day value was fixed before any of these results were seen, and 5 and 20
    # are here so a reader can see whether the finding depends on the choice.
    if ROBUSTNESS_BRACKET in args.brackets and not args.no_halflife_sweep:
        bids = sorted({int(c) for c in
                       assignments.loc[assignments["bracket"] == ROBUSTNESS_BRACKET,
                                       "cmc_id"]})
        bids = [c for c in bids if c in close.columns]
        base = B.bracket_membership(assignments, index, ROBUSTNESS_BRACKET, columns=bids)
        reference = BEST_REFERENCE.get(ROBUSTNESS_BRACKET, "eth")
        for treatment in [t.strip() for t in args.treatments.split(",") if t.strip()]:
            member = base.copy()
            if treatment != "pit":
                for c in member.columns:
                    if int(c) in dead:
                        member[c] = False
            for hl in EWMA_HALFLIFE_SWEEP:
                if hl == EWMA_DEFAULT_HALFLIFE:
                    continue        # already run as the `ewma` arm
                print(f"  {ROBUSTNESS_BRACKET} / {treatment} / ewma halflife={hl:g} ...",
                      flush=True)
                stats = run_arm(close, volume, table, refs, reference, 0, member, index,
                                **arm_kwargs(f"ewma_hl{hl:g}"))
                if stats is None:
                    continue
                robustness.append(robustness_row(
                    stats["net_series_for_funding"].dropna(), stats["turnover"],
                    bracket=ROBUSTNESS_BRACKET, treatment=treatment, arm="ewma",
                    halflife=hl, role="halflife_sweep"))

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

    if robustness:
        rob = pd.DataFrame(robustness)
        rob.loc[(rob["role"] == "halflife_sweep")
                | ((rob["arm"] == "ewma") & (rob["role"] == "arm")), "is_default"] = \
            rob["halflife"] == EWMA_DEFAULT_HALFLIFE
        year_cols = sorted(c for c in rob.columns if c.startswith("sharpe_")
                           and c != "sharpe_ex_best")
        front = ["bracket", "treatment", "arm", "role", "halflife", "is_default",
                 "net_sharpe", "sharpe_ex_best", "best_days_share", "turnover", "n_days"]
        rob = rob[[c for c in front if c in rob.columns] + year_cols]
        rob.to_csv(out_dir / "ewma_robustness.csv", index=False)

        print(f"\n=== {ROBUSTNESS_BRACKET} robustness "
              f"(net Sharpe, and what it survives) ===")
        print("  treatment       arm          hl    net   ex-best  best-10 share  turnover")
        for _, r in rob.iterrows():
            hl = "-" if not np.isfinite(r["halflife"]) else f"{r['halflife']:.0f}"
            mark = " *" if r.get("is_default") is True else "  "
            print(f"  {r['treatment']:<14s} {r['arm']:<11s} {hl:>3s}{mark} "
                  f"{r['net_sharpe']:6.2f} {r['sharpe_ex_best']:8.2f} "
                  f"{100 * r['best_days_share']:12.1f}% {r['turnover']:9.3f}")
        print(f"  * pre-set default half-life ({EWMA_DEFAULT_HALFLIFE:g} days); the other")
        print("    rows are robustness, not selection")
        print(f"  saved -> {out_dir / 'ewma_robustness.csv'}")

    print(f"\nsaved -> {out_dir / 'signal_ablation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
