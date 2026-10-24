"""Controls that keep each comparison a one-variable change.

Regression tests for the 2026-10 review:
- the EWMA arms entered at |z| > 2.0 while the baseline entered at 1.5, so the
  "standardisation only" comparison changed two things
- phases 1 and 2 read universe membership on the rebalance date itself
- non-spectral clusterers reported the k-selection spectrum's gap as their own
- downstream runners picked arms from the retired log-score ablation table
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.data.universe import membership_before  # noqa: E402
from stat_arb.run_phase3 import BASELINE_ENTRY_THRESHOLD, default_strategy  # noqa: E402
from stat_arb.run_signal_ablation import arm_kwargs  # noqa: E402


def _clustered(n_days=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D")
    tokens = [f"T{i}" for i in range(8)]
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    returns = pd.DataFrame(rng.normal(0, 0.03, (n_days, 8)), index=idx,
                           columns=[f"{t}_returns" for t in tokens])
    return returns, labels, tokens


def test_ewma_arms_enter_at_the_baseline_threshold():
    baseline = default_strategy(5, 20)
    assert baseline.entry_threshold == BASELINE_ENTRY_THRESHOLD
    for arm in ("ewma", "ewma_sized", "ewma_hl5"):
        strat = arm_kwargs(arm)["strategy_factory"](5, 20)
        assert strat.entry_threshold == baseline.entry_threshold, arm


def test_ewma_arm_differs_from_the_baseline_only_in_standardisation():
    """Same cumulative deviation; only the z-scoring of it changes."""
    from stat_arb.signals.ou_score import ewma_zscore

    returns, labels, tokens = _clustered()
    baseline = default_strategy(5, 20)
    composites = baseline.compute_cluster_composites(returns, labels, tokens)
    deviations = baseline.compute_deviations(returns, composites, labels, tokens)
    cumulative = deviations.rolling(window=5, min_periods=1).sum()
    expected = ewma_zscore(cumulative, halflife=10.0).shift(1)

    ewma = arm_kwargs("ewma")["strategy_factory"](5, 20)
    got = ewma.compute_signals(returns, labels, tokens, lag=1)
    pd.testing.assert_frame_equal(got, expected, check_names=False)


def test_membership_before_never_reads_the_rebalance_date():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    mask = pd.DataFrame({"A": [True, False, False], "B": [False, True, True]}, index=idx)
    row = membership_before(mask, idx[1])
    assert row.to_dict() == {"A": True, "B": False}     # the 1 Jan row, not 2 Jan
    assert not membership_before(mask, idx[0]).any()    # nothing earlier: empty


def test_phase_runners_use_prior_day_membership():
    from stat_arb import run_phase1, run_phase2, run_phase3

    for module in (run_phase1, run_phase2, run_phase3):
        source = inspect.getsource(module)
        assert "membership_before(universe_mask, rebalance_date)" in source, module
        assert "universe_mask.loc[rebalance_date]" not in source, module


def test_non_spectral_methods_report_no_eigengap():
    from stat_arb.run_clustering_sweep import RecordingClusterer

    rng = np.random.default_rng(0)
    truth = np.repeat([0, 1, 2], 15)
    same = truth[:, None] == truth[None, :]
    a = np.where(same, rng.uniform(0.6, 0.9, (45, 45)), rng.uniform(-0.9, -0.6, (45, 45)))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    adj = pd.DataFrame(a)
    for name in ("Hierarchical", "PCA-kmeans", "Pivot"):
        rec = RecordingClusterer(name)
        rec(adj, 3)
        assert np.isnan(rec.records[-1]["eigengap"]), name
    rec = RecordingClusterer("SPONGE")
    rec(adj, 3)
    assert np.isfinite(rec.records[-1]["eigengap"])


def test_downstream_arm_choice_uses_the_capital_ablation(tmp_path):
    """Selection reads the corrected table and never picks an insolvent arm."""
    from stat_arb.run_tradability import best_arms_from_ablation

    pd.DataFrame({"bracket": ["B1", "B1", "B3", "B3"], "treatment": "pit",
                  "arm": ["baseline", "ewma", "baseline", "ewma"],
                  "net_sharpe": [0.1, 0.4, -0.2, np.nan],
                  "status": ["ok", "ok", "ok", "insolvent"]}
                 ).to_csv(tmp_path / "signal_ablation_corrected.csv", index=False)
    # the retired log-score table would have picked differently
    pd.DataFrame({"bracket": ["B1", "B3"], "treatment": "pit", "arm": ["death", "ewma"],
                  "net_sharpe": [0.5, 1.5]}).to_csv(tmp_path / "signal_ablation.csv",
                                                     index=False)
    assert best_arms_from_ablation(tmp_path) == {"B1": "ewma", "B3": "baseline"}
