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


# ---------------------------------------------------------------------------
# Romano-Wolf stepwise multiple testing
#
# Ported from ~/Desktop/Quant_Projects/VOO_Backtest/statistics_mt.py, with a
# two-sided option added: the factor table tests whether a point-in-time Sharpe
# differs from its survivor counterpart, and that difference can legitimately go
# either way, so a one-sided test would only ever find bias in the direction it
# was pointed at.
# ---------------------------------------------------------------------------


def _circular_block_indices(n_obs, block_size, rng):
    """Circular block bootstrap indices.

    Blocks rather than individual observations, because strategy returns are
    serially dependent and an i.i.d. resample would understate the variance of
    the maximum statistic, which is precisely the quantity the procedure needs
    to get right.
    """
    n_blocks = int(np.ceil(n_obs / block_size))
    starts = rng.integers(0, n_obs, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_size)[None, :]).ravel() % n_obs
    return idx[:n_obs]


def romano_wolf_stepdown(returns, alpha=0.05, n_boot=1000, block_size=None,
                         seed=0, two_sided=False):
    """Romano-Wolf (2005) stepwise multiple test on a family of strategies.

    The Deflated Sharpe Ratio asks whether ONE selected configuration beats the
    expected best of N noise strategies. This asks a different and complementary
    question: across the whole family, WHICH members have a mean significantly
    different from zero, while controlling the family-wise error rate at
    `alpha`?

    Controlling FWER matters because a sweep is a family. Testing twelve
    configurations at 5% each gives roughly a 46% chance of at least one false
    positive; the stepdown procedure holds the probability of ANY false
    rejection at 5% instead, and unlike a Bonferroni correction it gains power
    by using the observed dependence between the members rather than assuming
    the worst case.

    Procedure: bootstrap the joint distribution of the centred t-statistics with
    a circular block bootstrap, take the (1-alpha) quantile of the MAXIMUM over
    the members not yet rejected, reject anything exceeding it, and repeat until
    no further rejections occur.

    `two_sided=True` runs the procedure on |t|, which is what the survivorship
    factor table needs: a point-in-time minus survivor Sharpe difference is
    interesting in either direction.

    Returns one row per member with its t-statistic, the critical value it
    faced, the step at which it was rejected (0 if never), and an adjusted
    p-value.
    """
    frame = returns.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    frame = frame.dropna(axis=1, how="all").fillna(0.0)
    n_obs, n_strat = frame.shape
    if n_obs < 30 or n_strat < 2:
        raise ValueError("need at least 30 observations and 2 strategies")

    if block_size is None:
        # Politis-White style rule of thumb; any O(n^(1/3)) choice is defensible
        block_size = max(2, int(round(n_obs ** (1.0 / 3.0))))

    values = frame.to_numpy(dtype=float)
    means = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1) / np.sqrt(n_obs)
    scale = np.where(scale <= 0, np.nan, scale)
    t_stats = means / scale

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, n_strat))
    for b in range(n_boot):
        idx = _circular_block_indices(n_obs, block_size, rng)
        sample = values[idx]
        # centred on the observed means: the bootstrap approximates the NULL
        b_scale = sample.std(axis=0, ddof=1) / np.sqrt(n_obs)
        b_scale = np.where(b_scale <= 0, np.nan, b_scale)
        boot[b] = (sample.mean(axis=0) - means) / b_scale

    stat = np.abs(t_stats) if two_sided else t_stats
    boot_stat = np.abs(boot) if two_sided else boot

    columns = list(frame.columns)
    rejected_step = {c: 0 for c in columns}
    critical = {c: np.nan for c in columns}
    adjusted_p = {c: np.nan for c in columns}

    remaining = list(range(n_strat))
    step = 0
    while remaining:
        step += 1
        with np.errstate(invalid="ignore"):
            max_null = np.nanmax(boot_stat[:, remaining], axis=1)
        crit = float(np.nanquantile(max_null, 1.0 - alpha))

        newly = [k for k in remaining if np.isfinite(stat[k]) and stat[k] > crit]
        for k in remaining:
            critical[columns[k]] = crit
            # adjusted p: mass of the max-null at or above this statistic
            adjusted_p[columns[k]] = float(np.nanmean(max_null >= stat[k]))
        if not newly:
            break
        for k in newly:
            rejected_step[columns[k]] = step
        remaining = [k for k in remaining if k not in newly]

    return pd.DataFrame({
        "strategy": columns,
        "mean": means,
        "t_stat": t_stats,
        "critical_value": [critical[c] for c in columns],
        "adjusted_p": [adjusted_p[c] for c in columns],
        "rejected_at_step": [rejected_step[c] for c in columns],
        "significant": [rejected_step[c] > 0 for c in columns],
    }).sort_values("t_stat", ascending=False, key=abs if two_sided else None
                   ).reset_index(drop=True)
