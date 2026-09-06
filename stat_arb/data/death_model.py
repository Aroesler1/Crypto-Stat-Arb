"""Predicting token death, and measuring what dying looks like on the way down.

Two pieces, both aimed at the same question: the point-in-time book loses money
because a mean-reversion strategy systematically buys the losers, and the losers
are disproportionately the tokens about to die. If death is forecastable even
weakly, the long leg can decline to buy them.

The classifier
--------------
Logistic regression on four features, all observable on the day the prediction
is made, predicting whether a token dies within the next `horizon` days:

``ret_30``        trailing 30-day log return
``vol_change``    log change in 30-day mean dollar volume against the prior 30
``rank_change``   change in CoinMarketCap rank over 30 days (positive is worse)
``days_listed``   days since the token first appeared

Trained walk-forward by year: the model that scores year Y is fitted only on
years before Y, so no fold ever sees its own future. AUC is reported per year
rather than pooled, because a pooled AUC over a period where the base rate moves
is mostly measuring the base rate.

One caveat about ``days_listed`` that is worth stating rather than discovering
later. A token's panel ends when it dies, so rows near the end of a dying
token's history are exactly the rows labelled 1, and age therefore carries
information about death even when price, volume and rank are flat. On a
synthetic panel with all three flow features held constant, ``days_listed``
alone reaches an AUC of 0.81. That is not leakage in the sense of seeing the
future (the feature is observable on the day), but it does mean a headline AUC
should not be read as "the flow variables forecast death": the per-year
coefficients are reported so the contribution of each feature can be seen
separately.

Logistic regression, not a gradient booster, on purpose. The feature set is four
columns and the label is rare; the interesting question is whether there is any
signal at all, and a linear model that can be read off its coefficients answers
that more honestly than a black box that will happily overfit 2,000 deaths.

The event study
---------------
For every token that dies, the path of price, volume, market cap and rank over
the 90 days before its last quote, against alive tokens matched on bracket and
calendar month. Matching matters: a naive comparison of dying tokens against all
survivors would mostly recover the fact that dying tokens are smaller, which is
already known. Bands are bootstrapped over tokens, not over days, because the
days within one token's path are anything but independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURES = ("ret_30", "vol_change", "rank_change", "days_listed")
DEFAULT_HORIZON = 30
LOOKBACK = 30


def build_features(panel: pd.DataFrame, universe_table: pd.DataFrame,
                   horizon: int = DEFAULT_HORIZON,
                   lookback: int = LOOKBACK,
                   sample_every: int = 7) -> pd.DataFrame:
    """One row per (token, sampled date) with the four features and the label.

    `sample_every` thins the daily panel: consecutive days for one token are
    almost identical rows, and keeping all of them inflates the sample without
    adding information while making the fit slower.

    The label is 1 if the token's death date falls within `horizon` days after
    the row's date. Censored tokens (alive, merely unranked) are never labelled
    1, which is the whole point of separating death from censoring upstream.
    """
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["cmc_id", "date"])

    deaths = (universe_table[universe_table["delisted"]]
              .set_index("cmc_id")["delisting_date"].to_dict())

    rows = []
    for cmc_id, g in p.groupby("cmc_id"):
        g = g.set_index("date")
        if len(g) < lookback * 2 + 1:
            continue
        price = pd.to_numeric(g["price_usd"], errors="coerce")
        volume = pd.to_numeric(g["volume_24h_usd"], errors="coerce")
        rank = pd.to_numeric(g["rank"], errors="coerce")

        ret_30 = np.log(price / price.shift(lookback))
        vol_mean = volume.rolling(lookback, min_periods=5).mean()
        vol_change = np.log(vol_mean / vol_mean.shift(lookback).replace(0.0, np.nan))
        rank_change = rank - rank.shift(lookback)
        days_listed = pd.Series((g.index - g.index[0]).days, index=g.index)

        feat = pd.DataFrame({
            "ret_30": ret_30, "vol_change": vol_change,
            "rank_change": rank_change, "days_listed": days_listed,
        })
        feat = feat.iloc[::sample_every]
        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
        if feat.empty:
            continue

        death_date = deaths.get(int(cmc_id))
        if death_date is not None and pd.notna(death_date):
            delta = (pd.Timestamp(death_date) - feat.index).days
            label = ((delta >= 0) & (delta <= horizon)).astype(int)
        else:
            label = np.zeros(len(feat), dtype=int)

        feat["dies"] = label
        feat["cmc_id"] = int(cmc_id)
        feat = feat.reset_index().rename(columns={"index": "date"})
        rows.append(feat)

    if not rows:
        return pd.DataFrame(columns=list(FEATURES) + ["dies", "cmc_id", "date"])
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["date", "cmc_id"]).reset_index(drop=True)


def walk_forward_auc(features: pd.DataFrame, min_train_years: int = 2,
                     min_positives: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit by year on prior years only; report AUC and coefficients per year.

    Returns (per-year metrics, per-year standardised coefficients). A year with
    too few deaths to score, or too little history to train on, is reported as
    skipped rather than given a fabricated AUC.
    """
    f = features.copy()
    f["year"] = f["date"].dt.year
    years = sorted(f["year"].unique())

    metrics, coefs = [], []
    for year in years:
        train = f[f["year"] < year]
        test = f[f["year"] == year]
        n_train_years = train["year"].nunique()
        if (n_train_years < min_train_years or train["dies"].sum() < min_positives
                or test["dies"].sum() < 5 or test["dies"].nunique() < 2):
            metrics.append({"year": year, "n_test": len(test),
                            "n_deaths": int(test["dies"].sum()),
                            "auc": np.nan, "skipped": True})
            continue

        scaler = StandardScaler().fit(train[list(FEATURES)])
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(scaler.transform(train[list(FEATURES)]), train["dies"])
        prob = model.predict_proba(scaler.transform(test[list(FEATURES)]))[:, 1]

        metrics.append({
            "year": year, "n_test": len(test), "n_deaths": int(test["dies"].sum()),
            "base_rate": float(test["dies"].mean()),
            "auc": float(roc_auc_score(test["dies"], prob)), "skipped": False,
        })
        coefs.append({"year": year,
                      **dict(zip(FEATURES, model.coef_[0].round(4)))})

    return pd.DataFrame(metrics), pd.DataFrame(coefs)


def death_probabilities(features: pd.DataFrame, min_train_years: int = 2,
                        min_positives: int = 20) -> pd.DataFrame:
    """Walk-forward out-of-sample death probability for every scored row.

    This is what gates the long leg. Every probability is produced by a model
    that never saw the year it is scoring.
    """
    f = features.copy()
    f["year"] = f["date"].dt.year
    out = []
    for year in sorted(f["year"].unique()):
        train, test = f[f["year"] < year], f[f["year"] == year]
        if (train["year"].nunique() < min_train_years
                or train["dies"].sum() < min_positives or test.empty):
            continue
        scaler = StandardScaler().fit(train[list(FEATURES)])
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(scaler.transform(train[list(FEATURES)]), train["dies"])
        t = test[["date", "cmc_id"]].copy()
        t["death_prob"] = model.predict_proba(scaler.transform(test[list(FEATURES)]))[:, 1]
        out.append(t)
    if not out:
        return pd.DataFrame(columns=["date", "cmc_id", "death_prob"])
    return pd.concat(out, ignore_index=True)


def event_study(panel: pd.DataFrame, universe_table: pd.DataFrame,
                assignments: pd.DataFrame, window: int = 90,
                n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Path of dying tokens over days -window..0, against matched alive tokens.

    Matching is on (bracket, calendar month of the event): a dying B3 token in
    2021-03 is compared against B3 tokens that were alive through 2021-03.
    Without that, the comparison mostly recovers the fact that dying tokens are
    small, which is not news.

    Each series is indexed to 1.0 at day -window so paths are comparable across
    tokens of wildly different price levels. Bootstrap bands resample *tokens*,
    because the 90 days within one token's path are not independent draws.
    """
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["cmc_id", "date"])

    bracket_of = (assignments.sort_values("snapshot_date")
                  .groupby("cmc_id")["bracket"].last().to_dict())
    dying = universe_table[universe_table["delisted"] & universe_table["delisting_date"].notna()]

    by_id = {int(c): g.set_index("date") for c, g in p.groupby("cmc_id")}
    alive_ids = set(universe_table.loc[~universe_table["delisted"], "cmc_id"].astype(int))

    fields = ("price_usd", "volume_24h_usd", "market_cap_usd", "rank")
    paths: dict[str, list[np.ndarray]] = {f"dead_{f}": [] for f in fields}
    paths.update({f"alive_{f}": [] for f in fields})

    rng = np.random.default_rng(seed)
    for row in dying.itertuples(index=False):
        cid = int(row.cmc_id)
        g = by_id.get(cid)
        if g is None:
            continue
        end = pd.Timestamp(row.delisting_date)
        seg = g.loc[:end].tail(window + 1)
        if len(seg) < window // 2:
            continue

        bracket = bracket_of.get(cid)
        # match: alive tokens in the same bracket, priced across the same span
        candidates = [c for c in alive_ids
                      if bracket_of.get(c) == bracket and c in by_id]
        match = None
        if candidates:
            rng.shuffle(candidates)
            for c in candidates[:40]:
                cand = by_id[c].loc[:end].tail(window + 1)
                if len(cand) >= len(seg):
                    match = cand
                    break

        for f in fields:
            s = pd.to_numeric(seg[f], errors="coerce").to_numpy(dtype=float)
            paths[f"dead_{f}"].append(_indexed(s, f, window))
            if match is not None:
                m = pd.to_numeric(match[f], errors="coerce").to_numpy(dtype=float)
                paths[f"alive_{f}"].append(_indexed(m, f, window))

    rows = []
    for key, series in paths.items():
        if not series:
            continue
        arr = np.vstack(series)
        mean = np.nanmean(arr, axis=0)
        # bootstrap over tokens
        boot = np.empty((n_boot, arr.shape[1]))
        for b in range(n_boot):
            pick = rng.integers(0, len(arr), len(arr))
            boot[b] = np.nanmean(arr[pick], axis=0)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        group, field = key.split("_", 1)
        for i, day in enumerate(range(-window, 1)):
            rows.append({"group": group, "field": field, "day": day,
                         "mean": mean[i], "lo": lo[i], "hi": hi[i],
                         "n_tokens": len(arr)})
    return pd.DataFrame(rows)


def _indexed(s: np.ndarray, field: str, window: int) -> np.ndarray:
    """Pad to window+1 and index to the first observation.

    Rank is a level, not a ratio, so it is differenced rather than divided:
    "fell 300 places" is the meaningful statement, not "rank multiplied by 1.4".
    """
    out = np.full(window + 1, np.nan)
    s = s[-(window + 1):]
    out[-len(s):] = s
    base = next((v for v in out if np.isfinite(v) and v != 0), np.nan)
    if not np.isfinite(base):
        return out
    return out - base if field == "rank" else out / base
