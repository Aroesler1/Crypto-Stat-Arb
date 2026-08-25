"""
Backtest statistics: Probabilistic and Deflated Sharpe Ratio.

Implements Bailey & Lopez de Prado:
- "The Sharpe Ratio Efficient Frontier" (2012) - PSR
- "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
  Overfitting and Non-Normality" (2014) - DSR

All Sharpe ratios inside this module are PER-PERIOD (e.g. daily),
not annualized. Helpers are provided to convert.
"""
import numpy as np
import pandas as pd
from typing import Optional, Sequence
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def per_period_sharpe(returns: pd.Series) -> float:
    """Per-period (non-annualized) Sharpe ratio."""
    r = pd.to_numeric(returns, errors='coerce').dropna()
    if len(r) < 2:
        return 0.0
    std = r.std(ddof=1)
    if std <= 0:
        return 0.0
    return float(r.mean() / std)


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 365) -> float:
    """Annualized Sharpe ratio (crypto default: 365 periods/year)."""
    return per_period_sharpe(returns) * np.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    sr_benchmark: float = 0.0,
) -> float:
    """
    PSR: probability that the true (per-period) Sharpe exceeds `sr_benchmark`,
    accounting for sample length, skewness, and kurtosis of returns.

    Parameters
    ----------
    returns : pd.Series
        Per-period returns.
    sr_benchmark : float
        Benchmark PER-PERIOD Sharpe ratio (0.0 tests "better than noise").

    Returns
    -------
    float
        PSR in [0, 1].
    """
    r = pd.to_numeric(returns, errors='coerce').dropna()
    n = len(r)
    if n < 3:
        return 0.5

    sr = per_period_sharpe(r)
    skew = float(r.skew())
    # pandas kurt() is excess kurtosis; PSR formula uses raw kurtosis
    kurt = float(r.kurt()) + 3.0

    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        # Extreme higher moments; PSR undefined, be conservative
        return 0.0

    z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(
    n_trials: int,
    var_trial_sr: float,
    mean_trial_sr: float = 0.0,
) -> float:
    """
    Expected maximum PER-PERIOD Sharpe across `n_trials` independent trials
    whose true Sharpe is zero (the "False Strategy Theorem" benchmark).

    E[max SR] ~= mean + sqrt(var) * ((1-gamma) * z_(1-1/N) + gamma * z_(1-1/(N*e)))
    """
    n = max(int(n_trials), 1)
    if n == 1 or var_trial_sr <= 0:
        return mean_trial_sr
    sd = np.sqrt(var_trial_sr)
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(mean_trial_sr + sd * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpes: Optional[Sequence[float]] = None,
    var_trial_sr: Optional[float] = None,
) -> dict:
    """
    DSR: PSR evaluated against the expected max Sharpe under multiple testing.

    Parameters
    ----------
    returns : pd.Series
        Per-period returns of the SELECTED strategy.
    n_trials : int
        Number of strategy configurations tried before selecting this one.
    trial_sharpes : sequence of float, optional
        PER-PERIOD Sharpe ratios of all trials (e.g. every config in a sweep).
        Used to estimate the cross-trial Sharpe variance. Preferred input.
    var_trial_sr : float, optional
        Direct estimate of cross-trial Sharpe variance; overrides
        `trial_sharpes` if given.

    Returns
    -------
    dict with keys:
        dsr            : probability the strategy outperforms the expected
                         max of `n_trials` zero-skill strategies
        sr_benchmark   : expected max per-period Sharpe used as benchmark
        sr_observed    : observed per-period Sharpe
        n_trials       : trials used
    """
    if var_trial_sr is None:
        if trial_sharpes is not None and len(trial_sharpes) >= 2:
            var_trial_sr = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1))
        else:
            # Fallback: sampling variance of a zero-Sharpe estimator over this
            # sample length. Understates config-search dispersion; prefer
            # passing trial_sharpes.
            n_obs = max(len(pd.to_numeric(returns, errors='coerce').dropna()), 2)
            var_trial_sr = 1.0 / (n_obs - 1)

    sr_star = expected_max_sharpe(n_trials=n_trials, var_trial_sr=var_trial_sr)
    dsr = probabilistic_sharpe_ratio(returns, sr_benchmark=sr_star)
    return {
        'dsr': dsr,
        'sr_benchmark': sr_star,
        'sr_observed': per_period_sharpe(returns),
        'n_trials': int(n_trials),
    }
