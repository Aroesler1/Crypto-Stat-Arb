"""Frozen baseline/EWMA replication with capital accounting, no signal selection.

These are retrospective repairs on an already studied sample. Missing held
returns use the existing stale-mark proxy and their exposure is reported; none
of these spot-return books establishes executable perpetual performance.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stat_arb.data import brackets as B
from stat_arb.run_residualization_ablation import load_inputs, run_arm
from stat_arb.run_signal_ablation import arm_kwargs
from stat_arb.backtest.holdings import InsolvencyError


def verify(directory):
    summary = pd.read_csv(directory / "accounting_summary.csv")
    daily = pd.read_csv(directory / "accounting_daily.csv", parse_dates=["date"])
    expected = {(bracket, arm) for bracket in ("B1", "B2", "B3") for arm in ("baseline", "ewma")}
    if len(summary) != 6 or set(zip(summary.bracket, summary.arm)) != expected:
        raise ValueError("Expected the fixed three brackets by two signals")
    if daily.duplicated(["bracket", "arm", "date"]).any() or not np.isfinite(daily.net_return).all():
        raise ValueError("Daily returns must be unique and finite")
    if not set(zip(daily.bracket, daily.arm)).issubset(expected):
        raise ValueError("Unexpected daily comparison arm")
    for row in summary.itertuples():
        path = daily.loc[(daily.bracket == row.bracket) & (daily.arm == row.arm)]
        if not path.date.is_monotonic_increasing:
            raise ValueError("Daily dates must be ordered")
        values = path.net_return
        if row.status == "no_positions":
            if len(values):
                raise ValueError("No-position row has return observations")
            continue
        if row.status == "insolvent":
            if (len(values) != row.n_days or not len(values) or values.iloc[-1] > -1
                    or (values.iloc[:-1] <= -1).any()
                    or str(path.date.iloc[-1].date()) != row.failure_date
                    or not np.isclose((1 + values).prod(), row.terminal_equity, atol=1e-10)):
                raise ValueError("Insolvency row needs the observed terminal loss")
            continue
        if row.status != "stale_mark_spot_proxy" or (values <= -1).any():
            raise ValueError("Unexpected status or unreported insolvency")
        measured = values.mean() / (values.std() + 1e-8 / np.sqrt(365)) * np.sqrt(365)
        if len(values) != row.n_days or not np.isclose(measured, row.net_sharpe, atol=1e-10):
            raise ValueError("Capital-return summary does not reproduce")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    out = ROOT / "stat_arb/reporting/brackets"
    if args.check:
        verify(out)
        print("Verified six fixed accounting rows from committed daily portfolio returns")
        return
    table, assignments, close, volume, _, refs, index = load_inputs(
        args.data_dir / "pit_daily_listings.parquet", args.data_dir,
        B.BRACKET_START, pd.Timestamp("2025-06-30"))
    legacy = pd.read_csv(out / "signal_ablation.csv")
    rows, paths = [], []
    for bracket in ("B1", "B2", "B3"):
        ids = sorted(set(assignments.loc[assignments.bracket == bracket, "cmc_id"]).intersection(close.columns))
        member = B.bracket_membership(assignments, index, bracket, columns=ids)
        for arm in ("baseline", "ewma"):
            print(f"Frozen replication: {bracket} / {arm}", flush=True)
            failure = None
            try:
                stats = run_arm(close, volume, table, refs, "eth", 0, member, index, **arm_kwargs(arm))
            except InsolvencyError as exc:
                failure, stats = exc, None
            old = legacy[(legacy.bracket == bracket) & (legacy.arm == arm) & (legacy.treatment == "pit")]
            row = {"bracket": bracket, "arm": arm,
                   "historical_log_score_sharpe": float(old.net_sharpe.iloc[0]) if len(old) == 1 else np.nan,
                   "status": "no_positions" if stats is None else "stale_mark_spot_proxy",
                   "n_days": 0, "net_sharpe": np.nan, "missing_return_exposure_share": np.nan}
            if stats is not None:
                net = stats["net_series"]
                paths.append(pd.DataFrame({"date": net.index, "bracket": bracket,
                                           "arm": arm, "net_return": net.to_numpy()}))
                row.update(n_days=len(net), start=str(net.index.min().date()),
                           end=str(net.index.max().date()), net_sharpe=stats["net_sharpe"],
                           gross_sharpe=stats["gross_sharpe"], turnover=stats["turnover"],
                           missing_return_exposure_share=stats["missing_return_exposure_share"])
            elif failure is not None:
                net = failure.daily.net_return
                paths.append(pd.DataFrame({"date": net.index, "bracket": bracket,
                                           "arm": arm, "net_return": net.to_numpy()}))
                row.update(status="insolvent", n_days=len(net), failure_date=str(failure.date.date()),
                           terminal_equity=failure.equity, failure_cost_bps=failure.cost_bps)
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "accounting_summary.csv", index=False)
    daily = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame(columns=["date", "bracket", "arm", "net_return"])
    daily.to_csv(out / "accounting_daily.csv", index=False)
    verify(out)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
