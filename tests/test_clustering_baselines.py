"""Tests for the non-spectral clusterers.

These exist so a signed spectral method has something honest to beat. What is
pinned is that each one recovers planted block structure, respects the shared
interface, and does not fall over on the degenerate inputs a walk-forward loop
will eventually hand it (one asset, more clusters than nodes, an all-zero graph).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.clustering.baselines import (  # noqa: E402
    PCALoadingKMeans, PivotCorrelationClustering, SignedHierarchicalClustering,
)
from stat_arb.clustering.sponge import SPONGEClustering  # noqa: E402

METHODS = (SignedHierarchicalClustering, PCALoadingKMeans, PivotCorrelationClustering)


def planted(n_per=15, k=3, noise=0.05, seed=0):
    """A signed graph with k blocks: positive within, negative between."""
    rng = np.random.default_rng(seed)
    n = n_per * k
    truth = np.repeat(np.arange(k), n_per)
    same = truth[:, None] == truth[None, :]
    a = np.where(same, rng.uniform(0.6, 0.95, (n, n)), rng.uniform(-0.95, -0.6, (n, n)))
    a = a + rng.normal(0, noise, (n, n))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    return pd.DataFrame(a), truth


def agreement(labels, truth):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(truth, labels)


@pytest.mark.parametrize("cls", METHODS)
def test_each_method_recovers_planted_blocks(cls):
    adj, truth = planted()
    labels = cls(n_clusters=3).fit_predict(adj, 3)
    assert agreement(labels, truth) > 0.9


@pytest.mark.parametrize("cls", METHODS)
def test_interface_matches_sponge(cls):
    adj, _ = planted()
    m = cls(n_clusters=3)
    out = m.fit(adj, 3)
    assert out is m                       # fit returns self, as SPONGE does
    assert m.labels_ is not None
    assert len(m.labels_) == len(adj)
    assert np.array_equal(m.fit_predict(adj, 3), m.labels_)


@pytest.mark.parametrize("cls", METHODS)
def test_accepts_a_numpy_array_as_well_as_a_frame(cls):
    adj, truth = planted()
    a = cls(n_clusters=3).fit_predict(adj, 3)
    b = cls(n_clusters=3).fit_predict(adj.to_numpy(), 3)
    assert agreement(a, truth) == pytest.approx(agreement(b, truth))


@pytest.mark.parametrize("cls", METHODS)
def test_labels_are_zero_based_and_contiguous(cls):
    adj, _ = planted()
    labels = cls(n_clusters=3).fit_predict(adj, 3)
    assert set(np.unique(labels)) == set(range(len(np.unique(labels))))


@pytest.mark.parametrize("cls", (SignedHierarchicalClustering, PCALoadingKMeans))
def test_more_clusters_requested_than_nodes_is_survivable(cls):
    adj = pd.DataFrame(np.zeros((2, 2)))
    labels = cls(n_clusters=5).fit_predict(adj, 5)
    assert len(labels) == 2


@pytest.mark.parametrize("cls", METHODS)
def test_an_all_zero_graph_does_not_raise(cls):
    adj = pd.DataFrame(np.zeros((12, 12)))
    labels = cls(n_clusters=3).fit_predict(adj, 3)
    assert len(labels) == 12


@pytest.mark.parametrize("cls", METHODS)
def test_nan_entries_are_treated_as_no_edge(cls):
    adj, truth = planted()
    dirty = adj.copy()
    dirty.iloc[0, 5] = np.nan
    dirty.iloc[5, 0] = np.nan
    labels = cls(n_clusters=3).fit_predict(dirty, 3)
    assert agreement(labels, truth) > 0.8


# --- Pivot specifics -------------------------------------------------------

def test_pivot_infers_its_own_cluster_count():
    """The point of running Pivot: it is not told k."""
    adj, truth = planted(k=4)
    m = PivotCorrelationClustering(n_clusters=3, n_restarts=20).fit(adj)
    assert m.n_clusters_found_ == 4        # finds 4 despite being constructed with 3
    assert agreement(m.labels_, truth) > 0.9


def test_pivot_can_be_forced_to_a_requested_k():
    adj, _ = planted(k=4)
    m = PivotCorrelationClustering(n_clusters=2, n_restarts=20, match_k=True).fit(adj, 2)
    assert len(np.unique(m.labels_)) == 2


def test_pivot_is_deterministic_for_a_fixed_seed():
    adj, _ = planted()
    a = PivotCorrelationClustering(random_state=7).fit_predict(adj)
    b = PivotCorrelationClustering(random_state=7).fit_predict(adj)
    assert np.array_equal(a, b)


def test_pivot_disagreement_cost_prefers_the_true_partition():
    adj, truth = planted()
    a = adj.to_numpy()
    scrambled = np.random.default_rng(0).permutation(truth)
    cost = PivotCorrelationClustering._disagreements
    assert cost(a, truth) < cost(a, scrambled)


def test_pivot_puts_every_node_in_exactly_one_cluster():
    adj, _ = planted()
    labels = PivotCorrelationClustering().fit_predict(adj)
    assert (labels >= 0).all()
    assert len(labels) == len(adj)


# --- the baselines must actually be comparable to SPONGE -------------------

def test_sponge_and_the_baselines_agree_on_clean_planted_structure():
    """On easy data every method should agree; divergence on real data is then
    a statement about the data, not about one method being broken."""
    adj, truth = planted(noise=0.02)
    sponge = SPONGEClustering(n_clusters=3, random_state=42).fit_predict(adj)
    assert agreement(sponge, truth) > 0.9
    for cls in METHODS:
        assert agreement(cls(n_clusters=3).fit_predict(adj, 3), truth) > 0.9
