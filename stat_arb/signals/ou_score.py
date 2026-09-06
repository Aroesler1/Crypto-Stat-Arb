"""Step 3 signal extensions, all inside the same cluster structure.

Four variations on "how far is this token from its cluster, and how much of that
gap is worth trading". Each is an ablation against `ClusterDeviationStrategy`,
which stays the control.

a. `OUScoreStrategy`          Avellaneda and Lee's s-score
b. `BetaAdjustedDeviation`    trade the regression residual, not the raw gap
c. EWMA volatility scaling    `ewma_zscore`, `half_life_position_scale`
d. `ClusterMomentumOverlay`   reversion inside winning clusters, momentum across

The OU s-score
--------------
Avellaneda and Lee, "Statistical arbitrage in the US equities market"
(Quantitative Finance 10(7), 2010), model the cumulative residual of an asset as
an Ornstein-Uhlenbeck process and trade its standardised distance from the
long-run mean.

For each token, over a rolling window of length `window`, form the cumulative
residual ``X_t = sum_{s<=t} r_s`` and fit an AR(1)::

    X_{t+1} = a + b X_t + zeta_{t+1}

which is the discrete-time OU. The implied parameters are::

    kappa     = -log(b)                  mean-reversion speed, per day
    m         = a / (1 - b)              long-run mean of X
    sigma_eq  = sqrt(var(zeta) / (1 - b^2))   equilibrium standard deviation
    half_life = log(2) / kappa

and the signal is the standardised deviation::

    s = (X_t - m) / sigma_eq

A token is bought when ``s`` is sufficiently negative and sold when sufficiently
positive, which is the same direction as the z-score book but with a
mean-reversion speed estimated from the data rather than assumed.

Two of the paper's conditions are load-bearing and are implemented, not skipped:

1. ``b`` must lie in (0, 1). ``b >= 1`` is not mean-reverting and ``b <= 0`` is
   alternation, not reversion. Either way the OU fit is meaningless and the
   token is dropped for that date rather than traded on a nonsense s-score.
2. The implied half-life must be shorter than the holding horizon. A token whose
   residual takes three months to revert cannot be traded on a three-day
   rebalance, however large its s-score: the position would be closed long
   before the reversion arrived. Avellaneda and Lee impose the analogous
   condition as a floor on kappa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Avellaneda and Lee's entry and exit levels, in s-score units. Kept as the
# defaults so the implementation is the paper's rather than a tuned variant.
DEFAULT_ENTRY = 1.25
DEFAULT_EXIT = 0.50


def ou_parameters(x: np.ndarray) -> tuple[float, float, float]:
    """AR(1) fit of one cumulative-residual window: (kappa, m, sigma_eq).

    Returns NaNs when the fit is not a mean-reverting OU. See the module
    docstring for why that is a drop rather than a clamp.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan, np.nan, np.nan

    lhs, rhs = x[1:], x[:-1]
    var = float(np.var(rhs))
    if var <= 0:
        return np.nan, np.nan, np.nan

    b = float(np.cov(rhs, lhs, ddof=0)[0, 1] / var)
    if not (0.0 < b < 1.0):
        return np.nan, np.nan, np.nan

    a = float(np.mean(lhs) - b * np.mean(rhs))
    resid = lhs - (a + b * rhs)
    var_zeta = float(np.var(resid, ddof=1)) if len(resid) > 1 else 0.0
    if var_zeta <= 0:
        return np.nan, np.nan, np.nan

    kappa = -np.log(b)
    m = a / (1.0 - b)
    sigma_eq = np.sqrt(var_zeta / (1.0 - b * b))
    if not np.isfinite(sigma_eq) or sigma_eq <= 0:
        return np.nan, np.nan, np.nan
    return float(kappa), float(m), float(sigma_eq)


def s_scores(residuals: pd.DataFrame, window: int = 60,
             max_half_life: float | None = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling OU s-scores, and the implied half-life in days.

    `residuals` are residual returns (rows dates, columns tokens). The cumulative
    residual is rebuilt inside each rolling window, so the level is always
    measured against that window's own mean rather than an origin fixed at the
    start of the sample.

    Tokens whose implied half-life exceeds `max_half_life` days are set to NaN:
    the reversion is slower than the holding horizon, so the signal cannot be
    harvested. Pass None to keep them and measure what that costs.
    """
    r = residuals.astype(float)
    idx, cols = r.index, r.columns
    s_out = np.full((len(idx), len(cols)), np.nan)
    hl_out = np.full((len(idx), len(cols)), np.nan)
    values = r.to_numpy()

    for j in range(len(cols)):
        col = values[:, j]
        for i in range(window, len(idx)):
            w = col[i - window + 1: i + 1]
            if np.isnan(w).sum() > window * 0.2:
                continue
            w = np.nan_to_num(w, nan=0.0)
            x = np.cumsum(w)
            kappa, m, sigma_eq = ou_parameters(x)
            if not np.isfinite(kappa):
                continue
            half_life = float(np.log(2.0) / kappa)
            hl_out[i, j] = half_life
            if max_half_life is not None and half_life > max_half_life:
                continue
            s_out[i, j] = (x[-1] - m) / sigma_eq

    return (pd.DataFrame(s_out, index=idx, columns=cols),
            pd.DataFrame(hl_out, index=idx, columns=cols))


class OUScoreStrategy:
    """Within-cluster mean reversion on the Avellaneda-Lee s-score.

    Same portfolio shape as `ClusterDeviationStrategy`: long the bottom of the
    cluster, short the top, inverse-volatility weighted, cluster- and
    dollar-neutral. Only the deviation measure changes, which is the point of
    running it as an ablation.
    """

    def __init__(self, window: int = 60, entry_threshold: float = DEFAULT_ENTRY,
                 exit_threshold: float = DEFAULT_EXIT,
                 max_half_life: float | None = 3.0, vol_lookback: int = 20):
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_half_life = max_half_life
        self.vol_lookback = vol_lookback
        self.half_lives_ = None

    def compute_signals(self, returns: pd.DataFrame, cluster_labels: np.ndarray,
                        tokens: list, lag: int = 1) -> pd.DataFrame:
        """s-scores measured against the token's own cluster composite.

        The residual traded is the token's return minus its cluster's equal-
        weighted composite, so the s-score is a within-cluster statement and the
        book stays cluster-neutral by construction.
        """
        token_to_cluster = {t: cluster_labels[i] for i, t in enumerate(tokens)}
        col_token = {c: c.replace("_returns", "") for c in returns.columns}

        dev = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        for cluster in np.unique(cluster_labels):
            cols = [c for c in returns.columns
                    if token_to_cluster.get(col_token[c], None) == cluster]
            if len(cols) < 2:
                continue
            composite = returns[cols].mean(axis=1)
            dev[cols] = returns[cols].sub(composite, axis=0)

        s, hl = s_scores(dev.dropna(axis=1, how="all"), self.window, self.max_half_life)
        self.half_lives_ = hl
        # Lag so a signal is only ever formed from information already public.
        return s.shift(lag).reindex(columns=returns.columns)

    def generate_target_weights(self, signals: pd.DataFrame,
                                cluster_labels: np.ndarray, tokens: list,
                                returns: pd.DataFrame | None = None,
                                clusters_to_trade: list | None = None) -> pd.DataFrame:
        """Long negative s-scores, short positive, neutral within each cluster."""
        token_to_cluster = {t: cluster_labels[i] for i, t in enumerate(tokens)}
        col_token = {c: c.replace("_returns", "") for c in signals.columns}
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        inv_vol = None
        if returns is not None:
            vol = returns.rolling(self.vol_lookback, min_periods=5).std().shift(1)
            inv_vol = (1.0 / vol.replace(0.0, np.nan)).reindex(columns=signals.columns)

        clusters = (np.unique(cluster_labels) if clusters_to_trade is None
                    else np.array(clusters_to_trade))
        for cluster in clusters:
            cols = [c for c in signals.columns
                    if token_to_cluster.get(col_token[c], None) == cluster]
            if len(cols) < 2:
                continue
            s = signals[cols]
            # entry beyond the threshold, and hold nothing inside the exit band
            raw = (-s).where(s.abs() >= self.entry_threshold, 0.0)
            if inv_vol is not None:
                raw = raw * inv_vol[cols].fillna(0.0)
            # dollar-neutral within the cluster
            raw = raw.sub(raw.mean(axis=1), axis=0)
            gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
            weights[cols] = raw.div(gross, axis=0).fillna(0.0)

        return weights.fillna(0.0)


class BetaAdjustedDeviation:
    """Trade the regression residual against the cluster composite, not the gap.

    `ClusterDeviationStrategy` trades ``r_i - composite``, which implicitly
    assumes every member has unit exposure to its cluster. That is wrong in a
    cross-section that mixes a token moving 1.5x its cluster with one moving
    0.5x: the raw difference is then dominated by the beta mismatch rather than
    by any idiosyncratic deviation, and the book systematically shorts
    high-beta names in a cluster rally.

    So regress each token's return on its cluster composite over a rolling
    window and trade the residual::

        r_i = alpha_i + beta_i * composite_c + e_i

    Betas are estimated on data strictly before the signal date.
    """

    def __init__(self, beta_window: int = 60, zscore_window: int = 20,
                 entry_threshold: float = 1.5, min_periods: int = 20):
        self.beta_window = beta_window
        self.zscore_window = zscore_window
        self.entry_threshold = entry_threshold
        self.min_periods = min_periods
        self.betas_ = None

    def compute_signals(self, returns: pd.DataFrame, cluster_labels: np.ndarray,
                        tokens: list, lag: int = 1) -> pd.DataFrame:
        token_to_cluster = {t: cluster_labels[i] for i, t in enumerate(tokens)}
        col_token = {c: c.replace("_returns", "") for c in returns.columns}

        resid = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        betas = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

        for cluster in np.unique(cluster_labels):
            cols = [c for c in returns.columns
                    if token_to_cluster.get(col_token[c], None) == cluster]
            if len(cols) < 2:
                continue
            composite = returns[cols].mean(axis=1)
            var = composite.rolling(self.beta_window, min_periods=self.min_periods).var()
            for c in cols:
                cov = returns[c].rolling(self.beta_window,
                                         min_periods=self.min_periods).cov(composite)
                # shift so today's beta uses only data up to yesterday
                beta = (cov / var.replace(0.0, np.nan)).shift(1)
                betas[c] = beta
                resid[c] = returns[c] - beta * composite

        z = ((resid.rolling(self.zscore_window, min_periods=5).sum())
             .pipe(lambda s: (s - s.rolling(self.zscore_window, min_periods=5).mean())
                   / s.rolling(self.zscore_window, min_periods=5).std().replace(0.0, np.nan)))
        self.betas_ = betas
        return z.shift(lag)

    def generate_target_weights(self, signals, cluster_labels, tokens,
                                returns=None, clusters_to_trade=None):
        return OUScoreStrategy(entry_threshold=self.entry_threshold).generate_target_weights(
            signals, cluster_labels, tokens, returns, clusters_to_trade)


def ewma_zscore(x: pd.DataFrame, halflife: float = 10.0,
                min_periods: int = 5) -> pd.DataFrame:
    """Z-score against an exponentially weighted mean and standard deviation.

    A rolling window weights a 60-day-old observation exactly as much as
    yesterday's and then drops it entirely; in a market whose volatility moves as
    fast as this one, that puts a step change in the denominator every time a
    large day leaves the window. EWMA decays instead.
    """
    mean = x.ewm(halflife=halflife, min_periods=min_periods).mean()
    std = x.ewm(halflife=halflife, min_periods=min_periods).std()
    return (x - mean) / std.replace(0.0, np.nan)


def half_life_position_scale(half_lives: pd.DataFrame, target: float = 3.0,
                             floor: float = 0.25, cap: float = 1.0) -> pd.DataFrame:
    """Position multiplier that shrinks as the estimated half-life lengthens.

    A signal expected to revert in one day is worth more per unit of z-score than
    one expected to take a week, because the book only holds it for the
    rebalance interval. Scaling by ``target / half_life`` sizes toward the
    reversion actually reachable, bounded so a spuriously tiny half-life cannot
    produce an unbounded position.
    """
    scale = (float(target) / half_lives.replace(0.0, np.nan)).clip(lower=floor, upper=cap)
    return scale.fillna(floor)


class ClusterMomentumOverlay:
    """Reversion only inside winning clusters, plus a between-cluster momentum leg.

    Two ideas, kept separable so the table can say which one pays.

    Within: rank clusters by trailing composite return over `momentum_window`
    and run within-cluster reversion only in the top `top_frac`. The premise is
    that reversion is a liquidity-provision trade and pays better where the
    cluster itself is being bought.

    Between: a cross-sectional momentum leg on the cluster composites, long the
    top clusters and short the bottom. Moskowitz and Grinblatt, "Do industries
    explain momentum?" (Journal of Finance 54(4), 1999) find industry momentum
    subsumes much of individual stock momentum; clusters here play the part of
    industries, so if that carries over, the between-cluster leg should carry
    the momentum and the within-cluster leg should be pure reversion.
    """

    def __init__(self, momentum_window: int = 28, top_frac: float = 0.5,
                 between_weight: float = 0.0):
        self.momentum_window = momentum_window
        self.top_frac = top_frac
        self.between_weight = between_weight

    def cluster_composites(self, returns: pd.DataFrame, cluster_labels: np.ndarray,
                           tokens: list) -> pd.DataFrame:
        token_to_cluster = {t: cluster_labels[i] for i, t in enumerate(tokens)}
        col_token = {c: c.replace("_returns", "") for c in returns.columns}
        out = {}
        for cluster in np.unique(cluster_labels):
            cols = [c for c in returns.columns
                    if token_to_cluster.get(col_token[c], None) == cluster]
            if cols:
                out[int(cluster)] = returns[cols].mean(axis=1)
        return pd.DataFrame(out, index=returns.index)

    def clusters_to_trade(self, returns: pd.DataFrame, cluster_labels: np.ndarray,
                          tokens: list, as_of: pd.Timestamp) -> list:
        """The top clusters by trailing composite return, as of `as_of`.

        Uses data strictly before `as_of`, so the ranking that selects which
        clusters to trade cannot see the returns it will be judged on.
        """
        comp = self.cluster_composites(returns, cluster_labels, tokens)
        hist = comp.loc[comp.index < pd.Timestamp(as_of)]
        if hist.empty:
            return list(np.unique(cluster_labels))
        trailing = hist.tail(self.momentum_window).sum()
        n_keep = max(1, int(round(len(trailing) * self.top_frac)))
        return [int(c) for c in trailing.sort_values(ascending=False).head(n_keep).index]

    def between_cluster_weights(self, returns: pd.DataFrame, cluster_labels: np.ndarray,
                                tokens: list) -> pd.DataFrame:
        """Long the top-ranked cluster composites, short the bottom, equal-weighted
        across each cluster's members."""
        comp = self.cluster_composites(returns, cluster_labels, tokens)
        trailing = comp.rolling(self.momentum_window, min_periods=5).sum().shift(1)
        rank = trailing.rank(axis=1, pct=True)
        leg = (rank - 0.5) * 2.0                      # +1 best, -1 worst
        # Percentile ranks of k clusters are not centred on 0.5 for small k
        # (three clusters rank 1/3, 2/3, 1), so the raw leg carries a net long.
        # Demean across clusters to make the momentum leg dollar-neutral.
        leg = leg.sub(leg.mean(axis=1), axis=0)

        token_to_cluster = {t: cluster_labels[i] for i, t in enumerate(tokens)}
        col_token = {c: c.replace("_returns", "") for c in returns.columns}
        weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        for cluster in leg.columns:
            cols = [c for c in returns.columns
                    if token_to_cluster.get(col_token[c], None) == cluster]
            if not cols:
                continue
            weights[cols] = pd.concat([leg[cluster] / len(cols)] * len(cols), axis=1).values
        gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
        return weights.div(gross, axis=0).fillna(0.0)
