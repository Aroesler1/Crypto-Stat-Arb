"""Tests for the death classifier and the pre-delisting event study.

The classifier gates the long leg, so the tests that matter are the ones that
would catch it cheating: a label that leaks the future, a fold trained on its own
year, or a censored token counted as a death.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.data import death_model as M  # noqa: E402

START = pd.Timestamp("2018-01-01")


def _panel(cmc_id, n_days, start=START, price=10.0, volume=1e6, rank=500,
           decay=0.0):
    dates = pd.date_range(start, periods=n_days, freq="D")
    fall = np.exp(-decay * np.arange(n_days))
    return pd.DataFrame({
        "date": dates,
        "cmc_id": int(cmc_id),
        "symbol": f"T{cmc_id}",
        "name": f"Token {cmc_id}",
        "slug": f"token-{cmc_id}",
        "rank": rank + np.arange(n_days) * (10 * decay),
        "price_usd": price * fall,
        "volume_24h_usd": volume * fall,
        "market_cap_usd": price * fall * 1e6,
    })


def _table(rows):
    """rows: [(cmc_id, delisted, delisting_date), ...]"""
    return pd.DataFrame([
        {"cmc_id": int(c), "delisted": bool(d), "censored": False,
         "delisting_date": pd.Timestamp(dt) if dt else pd.NaT,
         "symbol": f"T{c}", "last_date": pd.Timestamp(dt) if dt else pd.NaT}
        for c, d, dt in rows])


# --- feature construction --------------------------------------------------

def test_label_marks_only_the_horizon_before_death():
    panel = _panel(10, 400)
    death = START + pd.Timedelta(days=399)
    feat = M.build_features(panel, _table([(10, True, death)]), horizon=30,
                            sample_every=1)
    delta = (death - feat["date"]).dt.days
    assert (feat.loc[feat["dies"] == 1, "date"] >= death - pd.Timedelta(days=30)).all()
    assert (feat.loc[delta > 30, "dies"] == 0).all()
    assert feat["dies"].sum() > 0


def test_a_censored_token_is_never_labelled_a_death():
    """The reason death and censoring are separated upstream."""
    panel = _panel(10, 400)
    table = _table([(10, False, None)])
    feat = M.build_features(panel, table, sample_every=1)
    assert feat["dies"].sum() == 0


def test_a_surviving_token_is_all_zeros():
    feat = M.build_features(_panel(10, 400), _table([(10, False, None)]))
    assert len(feat) > 0 and feat["dies"].sum() == 0


def test_features_are_finite_and_named_as_declared():
    feat = M.build_features(_panel(10, 400), _table([(10, False, None)]))
    assert set(M.FEATURES).issubset(feat.columns)
    assert np.isfinite(feat[list(M.FEATURES)].to_numpy()).all()


def test_short_histories_are_dropped_rather_than_padded():
    feat = M.build_features(_panel(10, 20), _table([(10, False, None)]))
    assert feat.empty


def test_days_listed_counts_from_the_first_observation():
    feat = M.build_features(_panel(10, 400), _table([(10, False, None)]),
                            sample_every=1)
    assert feat["days_listed"].min() >= 0
    assert feat["days_listed"].is_monotonic_increasing


def test_a_decaying_token_shows_negative_return_and_volume_change():
    feat = M.build_features(_panel(10, 400, decay=0.01),
                            _table([(10, True, START + pd.Timedelta(days=399))]),
                            sample_every=1)
    assert feat["ret_30"].median() < 0
    assert feat["vol_change"].median() < 0
    assert feat["rank_change"].median() > 0        # rank worsens


# --- walk-forward discipline -----------------------------------------------

def _multi_year_sample(n_tokens=60, seed=0):
    """Dying tokens decay; survivors do not. A model should find that."""
    rng = np.random.default_rng(seed)
    panels, rows = [], []
    for i in range(n_tokens):
        year = 2016 + (i % 6)
        start = pd.Timestamp(f"{year}-01-01")
        dies = i % 3 == 0
        lifetime = int(rng.integers(220, 400))
        panels.append(_panel(i, lifetime, start=start,
                             decay=0.012 if dies else 0.0,
                             rank=300 + rng.integers(0, 200)))
        rows.append((i, dies,
                     start + pd.Timedelta(days=lifetime - 1) if dies else None))
    return pd.concat(panels, ignore_index=True), _table(rows)


def test_walk_forward_never_trains_on_the_year_it_scores():
    panel, table = _multi_year_sample()
    feat = M.build_features(panel, table, sample_every=7)
    metrics, coefs = M.walk_forward_auc(feat, min_train_years=1, min_positives=5)
    scored = metrics[~metrics["skipped"]]
    assert len(scored) > 0
    # the earliest year can never be scored: nothing precedes it to train on
    assert metrics["year"].min() not in set(scored["year"])


def test_a_learnable_signal_gives_an_auc_above_chance():
    panel, table = _multi_year_sample()
    feat = M.build_features(panel, table, sample_every=7)
    metrics, _ = M.walk_forward_auc(feat, min_train_years=1, min_positives=5)
    scored = metrics[~metrics["skipped"]]
    assert scored["auc"].mean() > 0.6


def test_years_without_enough_deaths_are_skipped_not_faked():
    panel, table = _multi_year_sample()
    feat = M.build_features(panel, table, sample_every=7)
    metrics, _ = M.walk_forward_auc(feat, min_train_years=99, min_positives=5)
    assert metrics["skipped"].all()
    assert metrics["auc"].isna().all()


def test_probabilities_are_out_of_sample_and_bounded():
    panel, table = _multi_year_sample()
    feat = M.build_features(panel, table, sample_every=7)
    probs = M.death_probabilities(feat, min_train_years=1, min_positives=5)
    assert len(probs) > 0
    assert probs["death_prob"].between(0, 1).all()
    assert probs["date"].dt.year.min() > feat["date"].dt.year.min()


def test_permuted_labels_score_at_chance():
    """The sound null: shuffle the labels and the AUC must collapse to chance.

    A fully synthetic "no signal" panel does not work as a null here, because
    `days_listed` is informative whenever panels end at death: rows near the end
    of a dying token's history are exactly the ones labelled 1, so age picks up
    real structure even when price, volume and rank are flat. That is a property
    of the feature (documented in `death_model`), not a defect, so the null has
    to break the label-feature link directly rather than assume it away.
    """
    panel, table = _multi_year_sample(n_tokens=90, seed=5)
    feat = M.build_features(panel, table, sample_every=7)
    real, _ = M.walk_forward_auc(feat, min_train_years=1, min_positives=5)

    shuffled = feat.copy()
    rng = np.random.default_rng(0)
    shuffled["dies"] = rng.permutation(shuffled["dies"].to_numpy())
    null, _ = M.walk_forward_auc(shuffled, min_train_years=1, min_positives=5)

    null_auc = null.loc[~null["skipped"], "auc"].mean()
    real_auc = real.loc[~real["skipped"], "auc"].mean()
    assert abs(null_auc - 0.5) < 0.15
    assert real_auc > null_auc + 0.15


# --- event study -----------------------------------------------------------

def _assignments(ids, bracket="B3"):
    return pd.DataFrame([{"snapshot_date": START, "cmc_id": int(c),
                          "bracket": bracket} for c in ids])


def test_event_study_indexes_paths_to_the_window_start():
    panel, table = _multi_year_sample(n_tokens=12)
    ids = sorted(panel["cmc_id"].unique())
    out = M.event_study(panel, table, _assignments(ids), window=60, n_boot=50)
    assert not out.empty
    first = out[(out["group"] == "dead") & (out["field"] == "price_usd")
                & (out["day"] == -60)]
    assert first["mean"].iloc[0] == pytest.approx(1.0, abs=1e-9)


def test_event_study_shows_dying_tokens_falling():
    panel, table = _multi_year_sample(n_tokens=24)
    ids = sorted(panel["cmc_id"].unique())
    out = M.event_study(panel, table, _assignments(ids), window=60, n_boot=50)
    dead = out[(out["group"] == "dead") & (out["field"] == "price_usd")]
    assert dead.sort_values("day")["mean"].iloc[-1] < 1.0


def test_event_study_bands_bracket_the_mean():
    panel, table = _multi_year_sample(n_tokens=24)
    ids = sorted(panel["cmc_id"].unique())
    out = M.event_study(panel, table, _assignments(ids), window=60, n_boot=100)
    d = out.dropna(subset=["mean", "lo", "hi"])
    assert (d["lo"] <= d["mean"] + 1e-9).all()
    assert (d["hi"] >= d["mean"] - 1e-9).all()


def test_rank_is_differenced_not_ratioed():
    """A rank path is a level: "fell 300 places", not "multiplied by 1.4"."""
    panel, table = _multi_year_sample(n_tokens=12)
    ids = sorted(panel["cmc_id"].unique())
    out = M.event_study(panel, table, _assignments(ids), window=60, n_boot=50)
    rank0 = out[(out["field"] == "rank") & (out["day"] == -60)
                & (out["group"] == "dead")]["mean"].iloc[0]
    assert rank0 == pytest.approx(0.0, abs=1e-9)
