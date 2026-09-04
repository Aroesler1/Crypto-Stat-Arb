"""Weekly quintile long-short factor portfolios, value-weighted, cost-aware.

The construction the crypto factor literature uses, and the one this repo's
survivorship comparison needs: at each weekly rebalance, rank the eligible
universe on a characteristic, form value-weighted quintile portfolios, and hold
the top-minus-bottom spread for the following week.

Choices that change the numbers, stated rather than buried:

* **Weekly, not daily.** A daily-rebalanced portfolio of micro-caps harvests the
  rebalancing bonus on assets quoted at 1e-07 and reports four-figure annualised
  returns nobody could trade. The same problem the buy-and-hold measurement in
  `run_pit_robustness` had to avoid.
* **Value-weighted, not equal-weighted.** Equal weighting on a survivorship-free
  micro-cap panel puts most of the portfolio in names with no volume. Ammann,
  Burdorf, Liebi and Stoeckl measure the equal-weighted survivorship bias at
  62.19% against 0.93% value-weighted, which is the size of the artifact being
  avoided.
* **Characteristics are lagged one day** before ranking, so a portfolio formed on
  Monday uses information through Sunday.
* **Costs at 50 bps per side on turnover**, matching the rest of the repo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_QUANTILES = 5
PERIODS_PER_YEAR = 52


def _rebalance_dates(index: pd.DatetimeIndex, freq: str = "W-MON") -> pd.DatetimeIndex:
    return pd.DatetimeIndex([d for d in pd.date_range(index.min(), index.max(), freq=freq)
                             if d in set(index)])


def quintile_long_short(
    characteristic: pd.DataFrame,
    forward_returns: pd.DataFrame,
    mcap: pd.DataFrame,
    membership: pd.DataFrame,
    higher_is_long: bool = True,
    n_quantiles: int = N_QUANTILES,
    cost_bps: float = 50.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Weekly value-weighted top-minus-bottom spread.

    Returns ``(gross, net, turnover)`` as weekly series. `forward_returns` are
    simple (not log) returns, so a week's portfolio return is the weighted mean
    of its members' compounded returns over that week.
    """
    index = forward_returns.index
    dates = _rebalance_dates(index)
    if len(dates) < 3:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    char = characteristic.shift(1)          # formed on yesterday's information
    prev_weights = pd.Series(dtype=float)
    gross, net, turn, out_dates = [], [], [], []

    for i in range(len(dates) - 1):
        start, stop = dates[i], dates[i + 1]
        if start not in char.index:
            continue
        eligible = membership.loc[start] if start in membership.index else None
        if eligible is None:
            continue
        cols = [c for c in char.columns if bool(eligible.get(c, False))]
        if len(cols) < n_quantiles * 2:
            continue

        scores = pd.to_numeric(char.loc[start, cols], errors="coerce").dropna()
        caps = pd.to_numeric(mcap.loc[start, scores.index], errors="coerce")
        scores = scores[caps.notna() & (caps > 0)]
        if len(scores) < n_quantiles * 2:
            continue

        ranks = scores.rank(method="first")
        edges = np.linspace(0, len(scores), n_quantiles + 1)
        bucket = np.searchsorted(edges[1:-1], ranks.to_numpy(), side="right")
        top = scores.index[bucket == n_quantiles - 1]
        bottom = scores.index[bucket == 0]
        long_leg, short_leg = (top, bottom) if higher_is_long else (bottom, top)

        window = forward_returns.loc[(forward_returns.index > start)
                                     & (forward_returns.index <= stop)]
        if window.empty:
            continue

        def leg_return(names):
            if len(names) == 0:
                return 0.0, pd.Series(dtype=float)
            w = pd.to_numeric(mcap.loc[start, names], errors="coerce").clip(lower=0.0)
            if w.sum() <= 0:
                w = pd.Series(1.0, index=names)
            w = w / w.sum()
            compounded = (1.0 + window[names].fillna(0.0)).prod() - 1.0
            return float((compounded * w).sum()), w

        r_long, w_long = leg_return(long_leg)
        r_short, w_short = leg_return(short_leg)
        weights = pd.concat([w_long, -w_short])
        aligned = weights.reindex(prev_weights.index.union(weights.index)).fillna(0.0)
        prior = prev_weights.reindex(aligned.index).fillna(0.0)
        t = float((aligned - prior).abs().sum())
        prev_weights = weights

        g = r_long - r_short
        gross.append(g)
        net.append(g - t * cost_bps / 10_000.0)
        turn.append(t)
        out_dates.append(stop)

    idx = pd.DatetimeIndex(out_dates)
    return (pd.Series(gross, index=idx), pd.Series(net, index=idx),
            pd.Series(turn, index=idx))


def annualised_sharpe(weekly: pd.Series) -> float:
    r = pd.to_numeric(weekly, errors="coerce").dropna()
    if len(r) < 8 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def factor_returns(
    characteristics: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    mcap: pd.DataFrame,
    membership: pd.DataFrame,
    signs: dict[str, bool],
    cost_bps: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every factor's weekly net series, and a summary row per factor."""
    net_series, rows = {}, []
    for name, panel in characteristics.items():
        gross, net, turn = quintile_long_short(
            panel, forward_returns, mcap, membership,
            higher_is_long=signs.get(name, True), cost_bps=cost_bps)
        if net.empty:
            continue
        net_series[name] = net
        rows.append({
            "factor": name,
            "n_weeks": int(len(net)),
            "gross_sharpe": annualised_sharpe(gross),
            "net_sharpe": annualised_sharpe(net),
            "mean_weekly": float(net.mean()),
            "turnover": float(turn.mean()),
        })
    return pd.DataFrame(net_series), pd.DataFrame(rows)
