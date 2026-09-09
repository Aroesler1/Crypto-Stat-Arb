"""Independent cash-flow cases for the capital-P&L engine."""

import numpy as np
import pandas as pd
import pytest

from stat_arb.backtest.holdings import replay_holdings, simple_from_excess_log
from stat_arb.run_phase3 import run_phase3_config
from stat_arb.run_signal_ablation import arm_kwargs


def frame(values):
    return pd.DataFrame(values, index=pd.date_range("2020-01-01", periods=len(values)), columns=["a"])


def test_short_collapse_is_bounded_by_simple_cash_flow():
    result = replay_holdings(frame([-1.]), frame([-.99]), cost_bps=0, max_turnover_per_day=2)
    assert result.daily.net_return.iloc[0] == pytest.approx(.99)
    assert result.daily.equity.iloc[0] == pytest.approx(1.99)


def test_unchanged_quantities_drift_without_hidden_rebalance():
    result = replay_holdings(frame([.5, .5]), frame([.1, .1]), cost_bps=0,
                             trade_frequency_days=2, max_turnover_per_day=2)
    assert result.weights.iloc[1, 0] == pytest.approx(.55 / 1.05)
    assert result.daily.equity.iloc[-1] == pytest.approx(.5 + .5 * 1.1**2)
    assert result.daily.turnover.iloc[1] == 0


def test_daily_target_maintenance_pays_for_actual_drift_trade():
    result = replay_holdings(frame([.5, .5]), frame([.1, .1]), cost_bps=0,
                             max_turnover_per_day=2)
    assert result.daily.turnover.iloc[1] == pytest.approx(.55 / 1.05 - .5)


def test_fees_and_funding_reconcile_to_capital():
    result = replay_holdings(frame([.5]), frame([.1]), cost_bps=100,
                             max_turnover_per_day=2, funding_rates=frame([.02]))
    assert result.daily.equity.iloc[0] == pytest.approx(1 + .05 - .005 - .01)
    assert result.daily.net_return.iloc[0] == pytest.approx(.035)


def test_missing_held_return_requires_an_explicit_proxy_policy():
    with pytest.raises(ValueError, match="missing held"):
        replay_holdings(frame([1.]), frame([np.nan]), max_turnover_per_day=2)
    result = replay_holdings(frame([1.]), frame([np.nan]), max_turnover_per_day=2,
                             missing_returns="stale_mark")
    assert result.daily.missing_return_exposure.iloc[0] == 1


def test_excess_log_is_converted_back_to_asset_simple_return():
    excess = frame([np.nan, np.log(1.2) - np.log(1.1)])
    reference = pd.Series([100, 110], index=excess.index)
    assert simple_from_excess_log(excess, reference).iloc[1, 0] == pytest.approx(.2)


def test_insolvency_is_not_silently_clipped():
    with pytest.raises(ValueError, match="insolvent"):
        replay_holdings(frame([-1.]), frame([2.]), max_turnover_per_day=2)


@pytest.mark.parametrize("members", [0, 29])
def test_no_eligible_universe_or_subthreshold_universe_never_clusters(members):
    dates = pd.date_range("2020-01-01", periods=400)
    returns = pd.DataFrame(np.random.default_rng(0).normal(0, .01, (400, 35)),
                           index=dates, columns=[f"{i}_returns" for i in range(35)])
    mask = pd.DataFrame(False, index=dates, columns=[str(i) for i in range(35)])
    mask.iloc[:, :members] = True
    def forbidden(*args):
        raise AssertionError("an ineligible universe reached clustering")
    assert run_phase3_config(returns, mask, clusterer=forbidden,
                              asset_returns=np.expm1(returns)) is None


def test_death_arm_cannot_silently_drop_its_filter():
    with pytest.raises(ValueError, match="death filter"):
        arm_kwargs("death", None)


def test_log_signal_requires_separate_return_input():
    with pytest.raises(ValueError, match="raw simple"):
        run_phase3_config(frame([0.]), pd.DataFrame())


def test_failed_empty_fold_is_an_error_not_no_positions(monkeypatch):
    from stat_arb.run_phase3 import WalkForwardBacktest
    monkeypatch.setattr(WalkForwardBacktest, "run_backtest",
                        lambda *a, **k: (pd.DataFrame(), [{"status": "error"}]))
    with pytest.raises(ValueError, match="fold failed"):
        run_phase3_config(frame([0.]), pd.DataFrame(), asset_returns=frame([0.]))


@pytest.mark.parametrize("value", [np.nan, np.inf, -1])
def test_invalid_cost_refuses_before_trading(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replay_holdings(frame([.5]), frame([.1]), cost_bps=value)


def test_audit_rejects_relabelled_insolvency(tmp_path):
    import shutil
    from pathlib import Path
    from stat_arb.run_accounting_audit import verify
    root = Path(__file__).resolve().parents[1] / "stat_arb/reporting/brackets"
    for filename in ("accounting_summary.csv", "accounting_daily.csv"):
        shutil.copyfile(root / filename, tmp_path / filename)
    path = tmp_path / "accounting_summary.csv"
    table = pd.read_csv(path)
    table.loc[table.status == "insolvent", "terminal_equity"] = 0
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="terminal loss"):
        verify(tmp_path)
