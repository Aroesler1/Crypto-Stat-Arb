"""Regression tests for the signed spectral clusterers.

Both of these were wrong, and both were wrong in ways that a test on planted
structure catches immediately:

* `BNCClustering` solved ``eigh(A+ - A-, D_tot)`` and took the smallest
  eigenvalues. Balance Normalized Cut (Chiang, Whang, Dhillon, CIKM 2012) is
  ``eigh(D+ - A+ + A-, D+ + D-)``. Dropping the ``D+`` term shifts the spectrum,
  so the old form selected from the opposite end of it and scored an adjusted
  Rand index of -0.02 on blocks the corrected form recovers exactly.

* `SPONGEClustering.fit_symmetric` whitened the same unnormalized pencil `fit`
  uses, which is a similarity transform of the same eigenproblem. It returned
  identical labels and identical eigenvalues, so the "SPONGEsym" column of the
  published sweep was plain SPONGE run twice.

`run_phase2.py` uses both, so these tests exist to stop either regressing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.clustering.bnc import BNCClustering  # noqa: E402
from stat_arb.clustering.sponge import SPONGEClustering  # noqa: E402


def planted(n_per=20, k=3, seed=0, scale=None):
    rng = np.random.default_rng(seed)
    n = n_per * k
    truth = np.repeat(np.arange(k), n_per)
    same = truth[:, None] == truth[None, :]
    a = np.where(same, rng.uniform(0.6, 0.95, (n, n)), rng.uniform(-0.95, -0.6, (n, n)))
    if scale is not None:
        a = a * np.outer(scale, scale)
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    return pd.DataFrame(a), truth


# --- BNC -------------------------------------------------------------------

@pytest.mark.parametrize("k", [2, 3, 4])
def test_bnc_recovers_planted_signed_blocks(k):
    """The test the old implementation failed at -0.02."""
    adj, truth = planted(k=k)
    labels = BNCClustering(n_clusters=k).fit(adj).labels_
    assert adjusted_rand_score(truth, labels) > 0.95


def test_bnc_beats_random_labelling_by_a_wide_margin():
    adj, truth = planted()
    labels = BNCClustering(n_clusters=3).fit(adj).labels_
    rng = np.random.default_rng(0)
    random_ari = adjusted_rand_score(truth, rng.permutation(truth))
    assert adjusted_rand_score(truth, labels) > random_ari + 0.5


def test_bnc_uses_the_positive_laplacian_not_the_bare_adjacency():
    """Pins the operator: L+ + A-, generalised against the total degree.

    Recomputing the intended eigenproblem directly must reproduce the class's
    own spectrum, which the bare-adjacency form did not.
    """
    from scipy.linalg import eigh
    adj, _ = planted()
    a = adj.to_numpy()
    ap, am = np.clip(a, 0, None), np.clip(-a, 0, None)
    dp, dm = np.diag(ap.sum(1)), np.diag(am.sum(1))
    expected = np.sort(eigh(dp - ap + am, dp + dm + 1e-6 * np.eye(len(a)),
                            eigvals_only=True))
    model = BNCClustering(n_clusters=3).fit(adj)
    assert np.allclose(np.sort(model.eigenvalues_), expected, atol=1e-8)


def test_bnc_survives_an_isolated_node():
    adj, _ = planted()
    a = adj.to_numpy().copy()
    a[0, :] = 0.0
    a[:, 0] = 0.0
    labels = BNCClustering(n_clusters=3).fit(pd.DataFrame(a)).labels_
    assert len(labels) == len(a)
    assert np.isfinite(labels).all()


# --- SPONGEsym -------------------------------------------------------------

@pytest.mark.parametrize("k", [2, 3, 4])
def test_spongesym_recovers_planted_blocks(k):
    adj, truth = planted(k=k)
    labels = SPONGEClustering(n_clusters=k, random_state=42).fit_predict(
        adj, symmetric=True)
    assert adjusted_rand_score(truth, labels) > 0.95


def test_spongesym_is_a_different_operator_from_plain_sponge():
    """The bug: it used to be a similarity transform of the same pencil, so the
    two spectra were identical to machine precision."""
    rng = np.random.default_rng(1)
    n_per, k = 25, 3
    n = n_per * k
    truth = np.repeat(np.arange(k), n_per)
    same = truth[:, None] == truth[None, :]
    a = np.where(same, rng.normal(0.25, 0.5, (n, n)), rng.normal(-0.25, 0.5, (n, n)))
    scale = np.ones(n)
    scale[:6] = 6.0                     # hubs, so degree normalisation bites
    a = a * np.outer(scale, scale)
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    adj = pd.DataFrame(a)

    m = SPONGEClustering(n_clusters=3, random_state=42)
    m.fit(adj)
    plain = np.sort(m.eigenvalues_)[:6].copy()
    m.fit_symmetric(adj)
    sym = np.sort(m.eigenvalues_)[:6].copy()
    assert not np.allclose(plain, sym)


def test_spongesym_normalises_by_degree():
    """Pins the operator: identity regularisers on normalized Laplacians."""
    from scipy.linalg import eigh
    adj, _ = planted()
    a = adj.to_numpy()
    ap, am = np.clip(a, 0, None), np.clip(-a, 0, None)
    isp = 1.0 / np.sqrt(np.maximum(ap.sum(1), 1e-12))
    ism = 1.0 / np.sqrt(np.maximum(am.sum(1), 1e-12))
    eye = np.eye(len(a))
    lp = eye - (isp[:, None] * ap * isp[None, :])
    lm = eye - (ism[:, None] * am * ism[None, :])
    expected = np.sort(eigh(lp + eye, lm + eye + 1e-6 * eye, eigvals_only=True))
    model = SPONGEClustering(n_clusters=3, tau_plus=1.0, tau_minus=1.0)
    model.fit_symmetric(adj)
    assert np.allclose(np.sort(model.eigenvalues_), expected, atol=1e-8)


def test_spongesym_survives_a_layer_with_no_edges():
    """A graph with no negative edges must not divide by a zero degree."""
    adj, _ = planted()
    a = np.clip(adj.to_numpy(), 0, None)      # positive edges only
    labels = SPONGEClustering(n_clusters=3, random_state=42).fit_predict(
        pd.DataFrame(a), symmetric=True)
    assert len(labels) == len(a)
    assert np.isfinite(labels).all()


def test_plain_sponge_still_recovers_planted_blocks():
    """The fix must not have disturbed the method the published results use."""
    adj, truth = planted()
    labels = SPONGEClustering(n_clusters=3, random_state=42).fit_predict(adj)
    assert adjusted_rand_score(truth, labels) > 0.95
