"""Step 2: every signed-clustering method, on every bracket, same signal.

The report compared three clusterers on one universe with k fixed. This compares
ten on three brackets and two survivorship treatments, with k selected inside
each walk-forward window, and scores them on the criteria the report used plus
the one that decides anything.

Methods
-------
Incumbent   SPONGE, SPONGEsym, BNC, SignedSpectral
Regularized RegSignedSpectral, RegSPONGE               (JMLR 2021)
Power mean  PowerMean p=1, p=0, p=-10                  (ICML 2019)
Baselines   Hierarchical, PCA-kmeans, Pivot

The last three are the honesty check. Hierarchical clustering on a correlation
distance and k-means on eigenvector loadings know nothing about signs, and Pivot
is non-spectral. If the signed spectral machinery cannot beat them, it is not
what is producing the result.

Selecting k
-----------
Never on the full sample. Inside each training window the number of clusters is
chosen by signflip parallel analysis (Hong and Cape, arXiv:2509.05722) on that
window's own graph, so k is a fitted quantity like any other and a method that
needs a different k in 2018 than in 2024 is allowed to have one. Pivot ignores
the choice entirely and infers its own count, which is why its selected k is
reported separately.

Criteria
--------
``eigengap``, ``calinski_harabasz``, ``davies_bouldin``  the report's criteria,
computed on each window's embedding
``stability``   adjusted Rand index between consecutive monthly clusterings
``net_sharpe``  walk-forward net Sharpe at 50 bps, the criterion that matters

The first three score a partition that has already been produced, so they will
rank three clusters against four in a graph with no clusters at all. They are
reported because the report reported them, and read against the Sharpe rather
than instead of it.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.clustering.baselines import (  # noqa: E402
    PCALoadingKMeans, PivotCorrelationClustering, SignedHierarchicalClustering,
)
from stat_arb.clustering.bnc import BNCClustering  # noqa: E402
from stat_arb.clustering.k_selection import signflip_parallel_analysis  # noqa: E402
from stat_arb.clustering.regularized import (  # noqa: E402
    PowerMeanLaplacianClustering, RegularizedSPONGEClustering,
    RegularizedSignedSpectralClustering,
)
from stat_arb.clustering.signed_spectral import SignedSpectralClustering  # noqa: E402
from stat_arb.clustering.sponge import SPONGEClustering  # noqa: E402
from stat_arb.data import brackets as B  # noqa: E402
from stat_arb.run_phase3 import annualized_sharpe, run_phase3_config  # noqa: E402
from stat_arb.run_residualization_ablation import (  # noqa: E402
    BEST_BAND, BEST_FREQ, cluster_stability, load_inputs, run_arm,
)

K_MIN, K_MAX = 2, 8


def _make(cls, **kw):
    def build(k, seed=42):
        return cls(n_clusters=k, random_state=seed, **kw)
    return build


METHODS = {
    "SPONGE": _make(SPONGEClustering),
    "SPONGEsym": _make(SPONGEClustering),          # dispatched via symmetric=True
    "BNC": _make(BNCClustering),
    "SignedSpectral": _make(SignedSpectralClustering),
    "RegSignedSpectral": _make(RegularizedSignedSpectralClustering),
    "RegSPONGE": _make(RegularizedSPONGEClustering),
    "PowerMean p=1": _make(PowerMeanLaplacianClustering, p=1.0),
    "PowerMean p=0": _make(PowerMeanLaplacianClustering, p=0.0),
    "PowerMean p=-10": _make(PowerMeanLaplacianClustering, p=-10.0),
    "Hierarchical": lambda k, seed=42: SignedHierarchicalClustering(n_clusters=k),
    "PCA-kmeans": _make(PCALoadingKMeans),
    "Pivot": lambda k, seed=42: PivotCorrelationClustering(
        n_clusters=k, random_state=seed),
}

# Methods that infer their own cluster count and ignore the selected k.
K_FREE = {"Pivot"}


class RecordingClusterer:
    """Adapts a clustering class to the `run_phase3_config` clusterer contract.

    Selects k inside the window, runs the method, and records the report's
    quality criteria for that window. One instance per (method, bracket, arm)
    run; the records it accumulates are read afterwards.
    """

    def __init__(self, name: str, k_replicates: int = 24, seed: int = 42):
        self.name = name
        self.build = METHODS[name]
        self.k_replicates = k_replicates
        self.seed = seed
        self.records: list[dict] = []

    def __call__(self, adjacency, n_clusters):
        a = adjacency.to_numpy() if isinstance(adjacency, pd.DataFrame) else np.asarray(adjacency)

        # k chosen on this window's graph alone
        k, _, eigenvalues = signflip_parallel_analysis(
            a, n_replicates=self.k_replicates, max_k=K_MAX, random_state=self.seed)
        k = int(np.clip(k, K_MIN, K_MAX))

        model = self.build(k, self.seed)
        if self.name == "SPONGEsym":
            labels = model.fit_predict(adjacency, k, symmetric=True)
        else:
            labels = model.fit_predict(adjacency, k)
        labels = np.asarray(labels)

        embedding = getattr(model, "embedding_", None)
        if embedding is None or np.asarray(embedding).ndim != 2:
            embedding = a                       # non-spectral methods: score on the graph
        embedding = np.nan_to_num(np.asarray(embedding, dtype=float))
        # Standardise before scoring. Calinski-Harabasz is a ratio of between to
        # within scatter and is unbounded as within-cluster scatter goes to
        # zero, which is exactly what a power-mean embedding with near-degenerate
        # eigenvectors produces: unscaled, it reported 2.7e31 and made the
        # column unreadable. Standardising per dimension puts every method's
        # embedding on the same scale, which is the only way the criteria mean
        # anything side by side.
        scale = embedding.std(axis=0)
        embedding = (embedding - embedding.mean(axis=0)) / np.where(scale < 1e-12, 1.0, scale)

        rec = {"k_selected": int(len(np.unique(labels))), "k_requested": k,
               "eigengap": np.nan, "calinski_harabasz": np.nan,
               "davies_bouldin": np.nan, "degenerate_embedding": False}
        ev = np.sort(np.asarray(getattr(model, "eigenvalues_", eigenvalues), dtype=float))
        if len(ev) > k:
            gaps = np.diff(ev[:K_MAX + 1])
            rec["eigengap"] = float(np.nanmax(gaps)) if len(gaps) else np.nan
        if len(np.unique(labels)) > 1 and len(labels) > len(np.unique(labels)):
            # Within-cluster scatter, to detect a degenerate embedding. The
            # power-mean operator collapses every member of a cluster onto the
            # same point, which makes Calinski-Harabasz infinite by construction
            # rather than large: it reported 7.8e31 here. That is the criterion
            # being undefined, not the partition being good, so it is recorded
            # as missing and counted rather than printed as a number.
            within = np.mean([embedding[labels == c].var(axis=0).sum()
                              for c in np.unique(labels)])
            rec["degenerate_embedding"] = bool(within < 1e-10)
            try:
                if within >= 1e-10:
                    rec["calinski_harabasz"] = float(calinski_harabasz_score(embedding, labels))
                rec["davies_bouldin"] = float(davies_bouldin_score(embedding, labels))
            except ValueError:
                pass
        self.records.append(rec)
        return labels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--start", default=str(B.BRACKET_START.date()))
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--brackets", default="B1,B2,B3")
    ap.add_argument("--treatments", default="pit,survivor-only")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--reference", default="eth",
                    help="residualization arm held fixed across methods")
    ap.add_argument("--n-pca", type=int, default=0)
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
    print(f"  {len(close.columns)} tokens, {len(index)} days, "
          f"reference {args.reference}, {args.n_pca} PCs removed")

    methods = [m.strip() for m in args.methods.split(",") if m.strip() in METHODS]
    rows = []
    for bracket in [b.strip() for b in args.brackets.split(",") if b.strip()]:
        ids = sorted({int(c) for c in
                      assignments.loc[assignments["bracket"] == bracket, "cmc_id"]})
        ids = [c for c in ids if c in close.columns]
        base = B.bracket_membership(assignments, index, bracket, columns=ids)
        for treatment in [t.strip() for t in args.treatments.split(",") if t.strip()]:
            member = base.copy()
            if treatment != "pit":
                for c in member.columns:
                    if int(c) in dead:
                        member[c] = False
            for name in methods:
                print(f"  {bracket} / {treatment} / {name} ...", flush=True)
                rec = RecordingClusterer(name)
                stats = run_arm(close, volume, table, refs, args.reference,
                                args.n_pca, member, index, clusterer=rec)
                if stats is None:
                    print("    produced no positions, skipped")
                    continue
                for _k in ("net_series", "net_series_for_funding", "weights"):
                    stats.pop(_k, None)
                d = pd.DataFrame(rec.records)
                stats.update(
                    bracket=bracket, treatment=treatment, method=name,
                    k_selected=float(d["k_selected"].mean()) if len(d) else np.nan,
                    k_min=int(d["k_selected"].min()) if len(d) else 0,
                    k_max=int(d["k_selected"].max()) if len(d) else 0,
                    eigengap=float(d["eigengap"].mean()) if len(d) else np.nan,
                    calinski_harabasz=float(d["calinski_harabasz"].mean()) if len(d) else np.nan,
                    davies_bouldin=float(d["davies_bouldin"].mean()) if len(d) else np.nan,
                    degenerate_windows=int(d["degenerate_embedding"].sum()) if len(d) else 0,
                    n_windows=len(d),
                )
                rows.append(stats)

    if not rows:
        print("no configuration produced a result")
        return 1

    out = pd.DataFrame(rows)
    out_dir = root / "stat_arb" / "reporting" / "brackets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "clustering_sweep.csv", index=False)

    print(f"\n=== clustering methods by bracket "
          f"(reference {args.reference}, band {BEST_BAND:.0%}, "
          f"rebalance {BEST_FREQ}d, net 50bps, k walk-forward) ===")
    for bracket in out["bracket"].unique():
        for treatment in out["treatment"].unique():
            sub = out[(out["bracket"] == bracket) & (out["treatment"] == treatment)]
            if sub.empty:
                continue
            print(f"\n  {bracket} / {treatment}   (avg members "
                  f"{sub['avg_members'].iloc[0]:.0f})")
            print("  method             k   eigengap      CH     DB  stability  gross    net")
            for _, r in sub.sort_values("net_sharpe", ascending=False).iterrows():
                ch = ("degen" if r.get("degenerate_windows", 0) == r.get("n_windows", 0)
                      and r.get("n_windows", 0) > 0
                      else ("-" if not np.isfinite(r["calinski_harabasz"])
                            else f"{r['calinski_harabasz']:.1f}"))
                print(f"  {r['method']:<17s} {r['k_selected']:4.1f} "
                      f"{r['eigengap']:9.3f} {ch:>7s} "
                      f"{r['davies_bouldin']:6.2f} {r['stability_ari']:9.3f}  "
                      f"{r['gross_sharpe']:5.2f}  {r['net_sharpe']:5.2f}")

    print(f"\nsaved -> {out_dir / 'clustering_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
