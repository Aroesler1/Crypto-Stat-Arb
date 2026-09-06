"""Tests for the regularized and power-mean signed clusterers.

These are the methods `docs/signed_clustering_2026.md` recommends adding to the
comparison. What is pinned is that each recovers planted signed blocks, that the
regularization and the power-mean parameter actually do what they claim (rather
than being inert arguments that quietly reduce to the incumbent), and that they
survive the degenerate graphs a walk-forward loop will hand them.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.clustering.k_selection import signflip_parallel_analysis  # noqa: E402
from stat_arb.clustering.regularized import (  # noqa: E402
    PowerMeanLaplacianClustering, RegularizedSPONGEClustering,
    RegularizedSignedSpectralClustering, default_gamma, power_mean,
)

METHODS = (RegularizedSignedSpectralClustering, RegularizedSPONGEClustering,
           PowerMeanLaplacianClustering)


def planted(n_per=20, k=3, seed=0, noise=0.0, sparsity=0.0):
    rng = np.random.default_rng(seed)
    n = n_per * k
    truth = np.repeat(np.arange(k), n_per)
    same = truth[:, None] == truth[None, :]
    a = np.where(same, rng.uniform(0.6, 0.95, (n, n)), rng.uniform(-0.95, -0.6, (n, n)))
    if noise:
        a = a + rng.normal(0, noise, (n, n))
    if sparsity:
        a = a * (rng.random((n, n)) > sparsity)
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    return pd.DataFrame(a), truth


@pytest.mark.parametrize("cls", METHODS)
@pytest.mark.parametrize("k", [2, 3, 4])
def test_each_method_recovers_planted_blocks(cls, k):
    adj, truth = planted(k=k)
    assert adjusted_rand_score(truth, cls(n_clusters=k).fit_predict(adj, k)) > 0.95


@pytest.mark.parametrize("cls", METHODS)
def test_interface_matches_sponge(cls):
    adj, _ = planted()
    m = cls(n_clusters=3)
    assert m.fit(adj, 3) is m
    assert len(m.labels_) == len(adj)
    assert m.embedding_ is not None
    assert np.array_equal(m.fit_predict(adj, 3), m.labels_)


@pytest.mark.parametrize("cls", METHODS)
def test_accepts_numpy_and_dataframe_alike(cls):
    adj, truth = planted()
    a = adjusted_rand_score(truth, cls(n_clusters=3).fit_predict(adj, 3))
    b = adjusted_rand_score(truth, cls(n_clusters=3).fit_predict(adj.to_numpy(), 3))
    assert a == pytest.approx(b)


@pytest.mark.parametrize("cls", METHODS)
def test_more_clusters_than_nodes_is_survivable(cls):
    adj = pd.DataFrame(np.zeros((3, 3)))
    assert len(cls(n_clusters=5).fit_predict(adj, 5)) == 3


@pytest.mark.parametrize("cls", METHODS)
def test_a_layer_with_no_edges_does_not_divide_by_zero(cls):
    """A residualized graph with no negative edges is the common real case."""
    adj, _ = planted()
    positive_only = pd.DataFrame(np.clip(adj.to_numpy(), 0, None))
    labels = cls(n_clusters=3).fit_predict(positive_only, 3)
    assert np.isfinite(labels).all() and len(labels) == len(adj)


@pytest.mark.parametrize("cls", METHODS)
def test_isolated_node_is_survivable(cls):
    adj, _ = planted()
    a = adj.to_numpy().copy()
    a[0, :] = 0.0
    a[:, 0] = 0.0
    assert len(cls(n_clusters=3).fit_predict(pd.DataFrame(a), 3)) == len(a)


# --- the regularizer ------------------------------------------------------

def test_default_gamma_grows_with_graph_density():
    """gamma = (p_hat (n-1))^{7/8}: a denser graph gets a larger regularizer."""
    dense, _ = planted(sparsity=0.0)
    sparse, _ = planted(sparsity=0.9)
    assert default_gamma(dense) > default_gamma(sparse)
    assert default_gamma(sparse) >= 1.0


def test_default_gamma_is_floored_at_one():
    assert default_gamma(pd.DataFrame(np.zeros((30, 30)))) >= 1.0


def test_regularization_is_not_an_inert_argument():
    """gamma must change the operator.

    Checked on the spectrum rather than on the labels: planted blocks are strong
    enough that both settings recover them, which is the method behaving well
    and says nothing about whether the parameter is wired in.
    """
    adj, _ = planted(noise=0.9, sparsity=0.85, seed=3)
    small = RegularizedSignedSpectralClustering(3, gamma=1e-6).fit(adj, 3)
    large = RegularizedSignedSpectralClustering(3, gamma=1e4).fit(adj, 3)
    assert not np.allclose(small.eigenvalues_, large.eigenvalues_)
    assert RegularizedSignedSpectralClustering(3).fit(adj, 3).gamma_ > 0


def test_heavier_regularization_pulls_the_spectrum_toward_one():
    """D_gamma^{-1/2} A D_gamma^{-1/2} vanishes as gamma grows, so L_gamma -> I."""
    adj, _ = planted()
    heavy = RegularizedSignedSpectralClustering(3, gamma=1e6).fit(adj, 3)
    assert np.allclose(heavy.eigenvalues_, 1.0, atol=1e-3)


def test_regularized_sponge_records_the_gamma_it_used():
    adj, _ = planted()
    m = RegularizedSPONGEClustering(3).fit(adj, 3)
    assert m.gamma_ == pytest.approx(default_gamma(adj))


def test_asymmetric_gamma_split_adds_the_rank_one_term():
    """The even split makes (gamma_plus - gamma_minus)/n vanish; a skew one does not."""
    adj, _ = planted(noise=0.9, sparsity=0.85, seed=4)
    even = RegularizedSignedSpectralClustering(3, gamma=10.0).fit(adj, 3)
    skew = RegularizedSignedSpectralClustering(
        3, gamma=10.0, gamma_plus=10.0, gamma_minus=0.0).fit(adj, 3)
    assert not np.allclose(even.eigenvalues_, skew.eigenvalues_)


# --- the power mean -------------------------------------------------------

def test_power_mean_reduces_to_the_arithmetic_mean_at_p_one():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 8)); a = a @ a.T + 8 * np.eye(8)
    b = rng.normal(size=(8, 8)); b = b @ b.T + 8 * np.eye(8)
    assert np.allclose(power_mean(a, b, 1.0), (a + b) / 2.0, atol=1e-8)


def test_power_mean_of_a_matrix_with_itself_is_that_matrix():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(8, 8)); a = a @ a.T + 8 * np.eye(8)
    for p in (1.0, -1.0, 0.0, -5.0):
        assert np.allclose(power_mean(a, a, p), a, atol=1e-6), p


def test_power_mean_at_p_zero_matches_the_geometric_mean_when_commuting():
    """Diagonal matrices commute, so the log-Euclidean limit is exactly sqrt(ab)."""
    a = np.diag([1.0, 4.0, 9.0])
    b = np.diag([4.0, 1.0, 16.0])
    assert np.allclose(power_mean(a, b, 0.0), np.diag([2.0, 2.0, 12.0]), atol=1e-8)


def test_negative_p_is_dominated_by_the_smaller_argument():
    """The property that makes p a real dial: p -> -inf approaches the minimum."""
    a = np.diag([1.0, 10.0])
    b = np.diag([10.0, 10.0])
    assert power_mean(a, b, -20.0)[0, 0] < power_mean(a, b, 1.0)[0, 0]
    assert power_mean(a, b, -20.0)[0, 0] == pytest.approx(1.0, abs=0.15)


def test_p_actually_changes_the_operator():
    adj, _ = planted(noise=1.1, sparsity=0.8, seed=7)
    a = PowerMeanLaplacianClustering(3, p=1.0).fit(adj, 3)
    b = PowerMeanLaplacianClustering(3, p=-10.0).fit(adj, 3)
    assert not np.allclose(a.eigenvalues_, b.eigenvalues_)


@pytest.mark.parametrize("p", [1.0, 0.0, -1.0, -10.0])
def test_power_mean_clustering_is_finite_for_every_p(p):
    adj, truth = planted()
    labels = PowerMeanLaplacianClustering(3, p=p).fit_predict(adj, 3)
    assert np.isfinite(labels).all()
    assert adjusted_rand_score(truth, labels) > 0.9


# --- signflip parallel analysis -------------------------------------------

@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_signflip_recovers_the_planted_cluster_count(k):
    adj, _ = planted(k=k)
    est, _, _ = signflip_parallel_analysis(adj, n_replicates=30)
    assert est == k


def test_signflip_finds_no_structure_in_noise():
    """The property the other criteria lack: it can say there is nothing here."""
    rng = np.random.default_rng(0)
    n = 60
    a = rng.normal(0, 1, (n, n))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    est, threshold, eigenvalues = signflip_parallel_analysis(a, n_replicates=30)
    assert int((eigenvalues > threshold).sum()) == 0
    assert est == 2      # clipped, because the caller still has to cluster


def test_signflip_is_deterministic_for_a_fixed_seed():
    adj, _ = planted()
    a = signflip_parallel_analysis(adj, n_replicates=20, random_state=3)
    b = signflip_parallel_analysis(adj, n_replicates=20, random_state=3)
    assert a[0] == b[0] and a[1] == pytest.approx(b[1])


def test_signflip_respects_max_k():
    adj, _ = planted(k=6, n_per=12)
    est, _, _ = signflip_parallel_analysis(adj, n_replicates=20, max_k=4)
    assert est <= 4


def test_signflip_handles_a_tiny_graph():
    est, _, _ = signflip_parallel_analysis(np.zeros((2, 2)), n_replicates=5)
    assert est == 2
