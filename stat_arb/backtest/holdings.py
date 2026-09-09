"""Self-financing spot-return proxy with explicit short dollar positions.

Signals may use log returns; this ledger accepts raw simple asset returns only.
Positions are dollar holdings, so unchanged quantities drift between trades.
Cash finances buys and receives sales. No interest, borrow availability, margin
liquidation or intraday funding marks are inferred from daily spot data.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HoldingsResult:
    daily: pd.DataFrame
    weights: pd.DataFrame


class InsolvencyError(ValueError):
    """A failed book is a measured outcome, never a return to clip or restart."""

    def __init__(self, date, equity, cost_bps, daily):
        super().__init__(f"account insolvent on {date}")
        self.date, self.equity, self.cost_bps, self.daily = date, equity, cost_bps, daily


def simple_from_excess_log(excess: pd.DataFrame, reference_close: pd.Series):
    """Undo the reference subtraction before exponentiating the asset return."""
    ref = reference_close.reindex(excess.index)
    log_reference = np.log(ref / ref.shift(1))
    return np.expm1(excess.add(log_reference, axis=0))


def replay_holdings(targets, asset_returns, *, cost_bps=50.0, weight_band=0.0,
                    trade_frequency_days=1, max_turnover_per_day=0.15,
                    funding_rates=None, missing_returns="raise"):
    """Replay targets known before each return, starting with one dollar cash.

    Target fractions and the turnover cap use equity before that day's trade.
    Fees leave cash immediately; simple returns then mark held asset quantities.
    Funding uses those same effective exposures, approximating intraday notional
    by its opening daily value. Missing funding stays an explicitly unmeasured
    component. Missing held returns raise unless the caller explicitly requests
    a stale mark (zero return), whose exposure is reported every day.
    """
    if trade_frequency_days < 1 or int(trade_frequency_days) != trade_frequency_days:
        raise ValueError("trade_frequency_days must be a positive integer")
    parameters = np.array([cost_bps, weight_band, max_turnover_per_day], dtype=float)
    if not np.isfinite(parameters).all() or (parameters < 0).any():
        raise ValueError("cost, band and turnover cap must be finite and nonnegative")
    if missing_returns not in ("raise", "stale_mark"):
        raise ValueError("unknown missing-return convention")
    if not targets.index.is_unique or not targets.index.is_monotonic_increasing:
        raise ValueError("targets require unique, increasing dates")
    if not targets.columns.is_unique:
        raise ValueError("targets require unique assets")
    t = targets.to_numpy(dtype=float)
    if not np.isfinite(t).all():
        raise ValueError("targets must be finite")
    r = asset_returns.reindex(index=targets.index, columns=targets.columns).to_numpy(float)
    if np.isinf(r).any() or np.any(r[np.isfinite(r)] < -1):
        raise ValueError("simple asset returns must be finite or missing, and >= -1")
    f = (funding_rates.reindex(index=targets.index, columns=targets.columns).to_numpy(float)
         if funding_rates is not None else np.full_like(r, np.nan))
    if np.isinf(f).any():
        raise ValueError("funding rates cannot be infinite")
    holdings = np.zeros(len(targets.columns))
    cash = 1.0
    records, effective = [], []
    for i, date in enumerate(targets.index):
        equity = cash + holdings.sum()
        if equity <= 0 or not np.isfinite(equity):
            raise ValueError(f"account insolvent before {date}")
        prior = holdings / equity
        desired = prior.copy()
        if i % trade_frequency_days == 0:
            desired = np.where(np.abs(t[i] - prior) < weight_band, prior, t[i])
            delta = desired - prior
            turnover = np.abs(delta).sum()
            if turnover > max_turnover_per_day:
                desired = prior + delta * max_turnover_per_day / turnover
        trade = desired * equity - holdings
        turnover = np.abs(trade).sum() / equity
        fee = turnover * cost_bps / 10000
        cash -= trade.sum() + equity * fee
        holdings += trade
        weights = holdings / equity
        missing = ~np.isfinite(r[i]) & (np.abs(weights) > 1e-12)
        if missing.any() and missing_returns == "raise":
            raise ValueError(f"missing held-asset return on {date}")
        marked = np.nan_to_num(r[i], nan=0.0)
        gross = float(weights @ marked)
        funding = -float(weights @ np.nan_to_num(f[i], nan=0.0))
        cash += equity * funding
        holdings *= 1 + marked
        end_equity = cash + holdings.sum()
        effective.append(weights.copy())
        records.append(dict(equity=end_equity, net_return=end_equity / equity - 1,
                            gross_return=gross, turnover=turnover, trading_cost=fee,
                            funding_return=funding, gross_exposure=np.abs(weights).sum(),
                            missing_return_exposure=np.abs(weights[missing]).sum(),
                            funding_covered_exposure=np.abs(weights[np.isfinite(f[i])]).sum()))
        if end_equity <= 0 or not np.isfinite(end_equity):
            partial = pd.DataFrame(records, index=targets.index[:i + 1])
            raise InsolvencyError(date, end_equity, cost_bps, partial)
    return HoldingsResult(pd.DataFrame(records, index=targets.index),
                          pd.DataFrame(effective, index=targets.index, columns=targets.columns))
