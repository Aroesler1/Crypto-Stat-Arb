"""Cross-sectional characteristics for the crypto factor table.

Each function takes daily panels (close, volume in USD, market cap) and returns a
panel of the characteristic, keyed the same way. Every one is computed from
information available on the day it is dated: the sorts that use them are formed
on the characteristic and held over the FOLLOWING week, so the lag lives in
`portfolios.quintile_long_short` rather than being applied twice here.

The list follows the crypto asset-pricing literature this repo cites: Liu, Tsyvinski
and Wu (JF 2022) for the size, momentum and volume factors; Fieberg, Guenther,
Poddig and Zaremba (Quantitative Finance 2023, and JFQA 2025) for the wider
characteristic set and the survivorship treatment; Dobrynskaya
(SSRN 3913263) for downside beta; Mercik, Zaremba and Demir (IRFA 2026) for the
trend and turning-point measures; Borri, Liu, Tsyvinski and Wu (arXiv 2510.14435)
for carry.

Two conventions worth stating because they change the numbers:

* Momentum is a cumulative log return over the formation window with no skip
  period. Equities skip the most recent month to avoid the short-term reversal;
  crypto reverses over days rather than months, so a one-week reversal factor is
  reported separately instead and the momentum windows are left unskipped.
* Amihud illiquidity is ``mean(|return| / dollar volume)`` over the window, so a
  high value is illiquid. Days with zero volume are dropped from the mean rather
  than treated as infinitely illiquid, which would let a single untraded day
  dominate a token's entire score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEK = 7


def _log_returns(close: pd.DataFrame) -> pd.DataFrame:
    p = close.where(close > 0)
    return np.log(p / p.shift(1))


def momentum(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Cumulative log return over `window` days."""
    return _log_returns(close).rolling(window, min_periods=max(2, window // 2)).sum()


def reversal(close: pd.DataFrame, window: int = WEEK) -> pd.DataFrame:
    """Negated short-horizon return: high means it fell, and is the buy side."""
    return -momentum(close, window)


def size(mcap: pd.DataFrame) -> pd.DataFrame:
    """Log market cap. Negated at sort time so 'small' is the long side."""
    return np.log(mcap.where(mcap > 0))


def dollar_volume(volume: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    return np.log(volume.where(volume > 0)).rolling(window, min_periods=5).mean()


def turnover(volume: pd.DataFrame, mcap: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    t = volume.where(volume > 0) / mcap.where(mcap > 0)
    return t.rolling(window, min_periods=5).mean()


def amihud(close: pd.DataFrame, volume: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Illiquidity: mean of |return| / dollar volume. High is illiquid."""
    r = _log_returns(close).abs()
    v = volume.where(volume > 0)
    ratio = (r / v).replace([np.inf, -np.inf], np.nan)
    return ratio.rolling(window, min_periods=5).mean() * 1e9


def volatility(close: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    return _log_returns(close).rolling(window, min_periods=5).std()


def idiosyncratic_volatility(close: pd.DataFrame, market: pd.Series,
                             window: int = 30) -> pd.DataFrame:
    """Residual volatility after a rolling market regression.

    Computed from the rolling beta rather than a full per-token regression loop:
    resid = r - beta * market, and its rolling standard deviation. Same estimator
    the beta factor uses, so the two are consistent with each other.
    """
    r = _log_returns(close)
    m = pd.to_numeric(market, errors="coerce").reindex(r.index)
    var = m.rolling(window, min_periods=5).var()
    beta = r.rolling(window, min_periods=5).cov(m).div(var.replace(0.0, np.nan), axis=0)
    resid = r.sub(beta.mul(m, axis=0))
    return resid.rolling(window, min_periods=5).std()


def market_beta(close: pd.DataFrame, market: pd.Series, window: int = 90) -> pd.DataFrame:
    r = _log_returns(close)
    m = pd.to_numeric(market, errors="coerce").reindex(r.index)
    var = m.rolling(window, min_periods=20).var()
    return r.rolling(window, min_periods=20).cov(m).div(var.replace(0.0, np.nan), axis=0)


def downside_beta(close: pd.DataFrame, market: pd.Series, window: int = 90) -> pd.DataFrame:
    """Beta estimated on days the market fell (Dobrynskaya, SSRN 3913263).

    Down days are selected by the market's own sign, so the same days are used
    for every token and the cross-section stays comparable.
    """
    r = _log_returns(close)
    m = pd.to_numeric(market, errors="coerce").reindex(r.index)
    down = m < 0
    rd = r.where(down)
    md = m.where(down)
    var = md.rolling(window, min_periods=20).var()
    return rd.rolling(window, min_periods=20).cov(md).div(var.replace(0.0, np.nan), axis=0)


def max_return(close: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Largest single-day return in the window (the lottery characteristic)."""
    return _log_returns(close).rolling(window, min_periods=5).max()


def skewness(close: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    return _log_returns(close).rolling(window, min_periods=20).skew()


def high_52w(close: pd.DataFrame, window: int = 365) -> pd.DataFrame:
    """Closeness to the trailing high: price divided by its rolling maximum."""
    p = close.where(close > 0)
    return p / p.rolling(window, min_periods=60).max()


def acceleration(close: pd.DataFrame, short: int = 28, long: int = 84) -> pd.DataFrame:
    """Change in momentum: recent return minus the earlier return of equal length."""
    r = _log_returns(close)
    recent = r.rolling(short, min_periods=short // 2).sum()
    earlier = r.rolling(long, min_periods=long // 2).sum() - recent
    return recent - earlier * (short / max(long - short, 1))


def trend(close: pd.DataFrame, short: int = 28, long: int = 180) -> pd.DataFrame:
    """Moving-average trend: log ratio of a fast to a slow moving average."""
    p = close.where(close > 0)
    fast = p.rolling(short, min_periods=short // 2).mean()
    slow = p.rolling(long, min_periods=long // 2).mean()
    return np.log(fast / slow.replace(0.0, np.nan))


def turning_points(close: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """Fraction of days in the window where the sign of the return flipped.

    A high value is a choppy, directionless series; a low value is a persistent
    one. Mercik, Zaremba and Demir (IRFA 2026) use a measure of this shape.
    """
    r = _log_returns(close)
    flip = (np.sign(r) != np.sign(r.shift(1))) & r.notna() & r.shift(1).notna()
    return flip.rolling(window, min_periods=20).mean()


# name -> (builder, higher_is_long_side)
#
# `higher_is_long_side` says which tail the long leg sits in, so every factor is
# signed the way the literature reports it: small minus big, loser minus winner
# for reversal, illiquid minus liquid, and so on.
CHARACTERISTICS: dict[str, tuple] = {
    "size":            (lambda c, v, m, mk: size(m), False),
    "mom_1w":          (lambda c, v, m, mk: momentum(c, 7), True),
    "mom_2w":          (lambda c, v, m, mk: momentum(c, 14), True),
    "mom_4w":          (lambda c, v, m, mk: momentum(c, 28), True),
    "mom_8w":          (lambda c, v, m, mk: momentum(c, 56), True),
    "mom_12w":         (lambda c, v, m, mk: momentum(c, 84), True),
    "mom_26w":         (lambda c, v, m, mk: momentum(c, 182), True),
    "mom_52w":         (lambda c, v, m, mk: momentum(c, 364), True),
    "reversal_1w":     (lambda c, v, m, mk: reversal(c, 7), True),
    "lt_reversal":     (lambda c, v, m, mk: -momentum(c, 364), True),
    "high_52w":        (lambda c, v, m, mk: high_52w(c), True),
    "dollar_volume":   (lambda c, v, m, mk: dollar_volume(v), False),
    "turnover":        (lambda c, v, m, mk: turnover(v, m), False),
    "amihud":          (lambda c, v, m, mk: amihud(c, v), True),
    "volatility":      (lambda c, v, m, mk: volatility(c), False),
    "ivol":            (lambda c, v, m, mk: idiosyncratic_volatility(c, mk), False),
    "max_return":      (lambda c, v, m, mk: max_return(c), False),
    "skew":            (lambda c, v, m, mk: skewness(c), False),
    "beta":            (lambda c, v, m, mk: market_beta(c, mk), False),
    "downside_beta":   (lambda c, v, m, mk: downside_beta(c, mk), False),
    "acceleration":    (lambda c, v, m, mk: acceleration(c), True),
    "trend":           (lambda c, v, m, mk: trend(c), True),
    "turning_points":  (lambda c, v, m, mk: turning_points(c), False),
}


def build_characteristics(close: pd.DataFrame, volume: pd.DataFrame,
                          mcap: pd.DataFrame, market: pd.Series,
                          names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Every characteristic panel, keyed by name."""
    out = {}
    for name in (names or list(CHARACTERISTICS)):
        builder, _ = CHARACTERISTICS[name]
        out[name] = builder(close, volume, mcap, market)
    return out
