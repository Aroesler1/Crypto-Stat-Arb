"""Regularized and power-mean signed clusterers, from the 2019-2021 literature.

Three methods that `docs/signed_clustering_2026.md` recommends implementing,
all pure numpy/scipy and all behind the same interface as `sponge.py`:

``RegularizedSignedSpectralClustering``  regularized signed Laplacian
``RegularizedSPONGEClustering``          regularized SPONGEsym pencil
``PowerMeanLaplacianClustering``         signed power mean Laplacian

Why regularization is the obvious thing to try here
---------------------------------------------------
This repo clusters a k=10 nearest-neighbour graph. That is sparse by
construction: a bracket with 400 members has each node joined to 10 others, so
the degree distribution is narrow but the graph is nowhere near dense, and the
negative layer is sparser still (measured on the real panel, only 1% to 5% of
edges are negative without a PCA step, rising to 43-79% with one). Sparse signed
graphs are exactly the regime where the unregularized normalized Laplacian is
known to be unstable: a low-degree node's ``D^{-1/2}`` blows up its row, and the
leading eigenvectors end up localised on a handful of near-isolated vertices
rather than describing the partition.

Cucuringu, Singh, Sulem and Tyagi (JMLR 22(264), 2021) fix that by adding a
constant to every degree before normalising, which shrinks the influence of
low-degree nodes without touching the well-connected ones. Their rule for the
regularizer is ``gamma = (p_hat (n-1))^{7/8}`` where ``p_hat`` is the observed
edge density, i.e. roughly the average degree raised to 7/8.

The power mean Laplacian
------------------------
Mercado, Tudisco and Hein (ICML 2019) take a different route. Rather than one
operator on the combined signed graph, they treat the positive and negative
edges as two layers and merge their Laplacians with a scalar power mean::

    L_p = M_p(L+_sym, Q-_sym),   M_p(A, B) = ((A^p + B^p) / 2)^{1/p}

where ``L+_sym = I - (D+)^{-1/2} A+ (D+)^{-1/2}`` is the normalized Laplacian of
the positive layer and ``Q-_sym = I + (D-)^{-1/2} A- (D-)^{-1/2}`` is the
normalized *signless* Laplacian of the negative layer. Both are positive
semi-definite and both are small on a good partition, so the smallest
eigenvectors of any mean of them describe one.

``p`` is a real dial rather than a nuisance parameter, and it is the reason this
method is worth the trouble. The power mean is dominated by its smallest
argument as ``p`` goes negative, so ``p = -10`` demands a cluster look good in
*both* layers, while ``p = 1`` is the plain arithmetic average and lets a strong
positive layer carry a weak negative one. On a graph whose negative layer is
thin and noisy, that is precisely the trade-off worth sweeping, and it must be
swept on the training fold only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.cluster import KMeans


def _as_array(adjacency) -> np.ndarray:
    a = adjacency.to_numpy() if isinstance(adjacency, pd.DataFrame) else np.asarray(adjacency)
    a = np.array(a, dtype=float, copy=True)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = (a + a.T) / 2.0
    np.fill_diagonal(a, 0.0)
    return a


def default_gamma(adjacency) -> float:
    """The JMLR 2021 regularizer: ``(p_hat (n - 1))^{7/8}``.

    ``p_hat`` is the fraction of possible edges present, so ``p_hat (n-1)`` is
    the average degree and the rule is roughly "average degree to the 7/8".
    Returns at least 1.0: a graph so sparse the rule gives less than that is the
    case regularization matters most for.
    """
    a = _as_array(adjacency)
    n = len(a)
    if n < 2:
        return 1.0
    p_hat = float((a != 0).sum()) / (n * (n - 1))
    return max(1.0, float((p_hat * (n - 1)) ** (7.0 / 8.0)))


def _embed(vectors: np.ndarray, n_eigen: int, row_normalize: bool = True) -> np.ndarray:
    u = vectors[:, :n_eigen]
    if row_normalize:
        norms = np.linalg.norm(u, axis=1, keepdims=True)
        u = u / np.where(norms < 1e-8, 1.0, norms)
    return u


def _kmeans(embedding: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embedding)


def _sym_normalized(a_layer: np.ndarray, gamma: float, signless: bool = False) -> np.ndarray:
    """``I -/+ D_gamma^{-1/2} A D_gamma^{-1/2}`` with a regularized degree."""
    d = a_layer.sum(axis=1) + float(gamma)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    norm = inv_sqrt[:, None] * a_layer * inv_sqrt[None, :]
    eye = np.eye(len(a_layer))
    return eye + norm if signless else eye - norm


class RegularizedSignedSpectralClustering:
    """Regularized signed Laplacian (Cucuringu, Singh, Sulem, Tyagi, JMLR 2021).

    Builds ``L_gamma = I - D_gamma^{-1/2} A_gamma D_gamma^{-1/2}`` on the signed
    adjacency, where ``D_gamma = D + gamma I`` uses the total degree
    ``D = D+ + D-`` and ``A_gamma = A + ((gamma_plus - gamma_minus)/n) 11^T``.

    The default splits the regularizer evenly, ``gamma_plus = gamma_minus =
    gamma/2``, which makes the rank-one term vanish and leaves a pure degree
    regularization. Both halves are exposed because an asymmetric split is what
    the paper uses to bias toward one layer.
    """

    def __init__(self, n_clusters: int = 3, gamma: float | None = None,
                 gamma_plus: float | None = None, gamma_minus: float | None = None,
                 n_eigen: int | None = None, random_state: int = 42):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.gamma_plus = gamma_plus
        self.gamma_minus = gamma_minus
        self.n_eigen = n_eigen
        self.random_state = random_state
        self.labels_ = None
        self.eigenvalues_ = None
        self.embedding_ = None
        self.gamma_ = None

    def fit(self, adjacency, n_clusters: int | None = None):
        if n_clusters is not None:
            self.n_clusters = n_clusters
        a = _as_array(adjacency)
        n = len(a)
        if n <= self.n_clusters:
            self.labels_ = np.arange(n)
            return self

        gamma = default_gamma(a) if self.gamma is None else float(self.gamma)
        gp = gamma / 2.0 if self.gamma_plus is None else float(self.gamma_plus)
        gm = gamma / 2.0 if self.gamma_minus is None else float(self.gamma_minus)
        self.gamma_ = gamma

        # Regularized adjacency: the rank-one term is zero under the even split
        a_gamma = a + ((gp - gm) / n) * np.ones((n, n))
        # Total degree uses |A|, so a node joined only by negative edges is not
        # treated as isolated
        d_gamma = np.abs(a).sum(axis=1) + gamma
        inv_sqrt = 1.0 / np.sqrt(np.maximum(d_gamma, 1e-12))
        l_gamma = np.eye(n) - (inv_sqrt[:, None] * a_gamma * inv_sqrt[None, :])

        self.eigenvalues_, vectors = eigh(l_gamma)
        # k-1 eigenvectors, matching the reference implementation: the trivial
        # eigenvector carries no partition information
        n_eigen = self.n_eigen or max(1, self.n_clusters - 1)
        self.embedding_ = _embed(vectors, n_eigen)
        self.labels_ = _kmeans(self.embedding_, self.n_clusters, self.random_state)
        return self

    def fit_predict(self, adjacency, n_clusters: int | None = None) -> np.ndarray:
        return self.fit(adjacency, n_clusters).labels_


class RegularizedSPONGEClustering:
    """Regularized SPONGEsym (Cucuringu, Singh, Sulem, Tyagi, JMLR 2021).

    The SPONGEsym pencil built on regularized, symmetrically normalized signed
    Laplacians::

        (L+_sym,gamma+ + tau_minus I) v = lambda (L-_sym,gamma- + tau_plus I) v

    solved for the smallest eigenvalues. This is the published successor to the
    incumbent `SPONGEClustering`, and differs from it only in that each layer's
    degree is regularized before normalising, which is what stops a node with
    one negative edge dominating the negative layer's spectrum.
    """

    def __init__(self, n_clusters: int = 3, tau_plus: float = 1.0,
                 tau_minus: float = 1.0, gamma: float | None = None,
                 n_eigen: int | None = None, random_state: int = 42):
        self.n_clusters = n_clusters
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.gamma = gamma
        self.n_eigen = n_eigen
        self.random_state = random_state
        self.labels_ = None
        self.eigenvalues_ = None
        self.embedding_ = None
        self.gamma_ = None

    def fit(self, adjacency, n_clusters: int | None = None):
        if n_clusters is not None:
            self.n_clusters = n_clusters
        a = _as_array(adjacency)
        n = len(a)
        if n <= self.n_clusters:
            self.labels_ = np.arange(n)
            return self

        gamma = default_gamma(a) if self.gamma is None else float(self.gamma)
        self.gamma_ = gamma
        a_plus = np.clip(a, 0, None)
        a_minus = np.clip(-a, 0, None)

        eye = np.eye(n)
        l_plus = _sym_normalized(a_plus, gamma)
        l_minus = _sym_normalized(a_minus, gamma)
        m1 = l_plus + self.tau_minus * eye
        m2 = l_minus + self.tau_plus * eye + 1e-6 * eye

        self.eigenvalues_, vectors = eigh(m1, m2)
        n_eigen = self.n_eigen or max(1, self.n_clusters - 1)
        self.embedding_ = _embed(vectors, n_eigen)
        self.labels_ = _kmeans(self.embedding_, self.n_clusters, self.random_state)
        return self

    def fit_predict(self, adjacency, n_clusters: int | None = None) -> np.ndarray:
        return self.fit(adjacency, n_clusters).labels_


def _matrix_power(m: np.ndarray, p: float, floor: float = 1e-8) -> np.ndarray:
    """``M^p`` for a symmetric positive semi-definite matrix, via eigendecomposition.

    Eigenvalues are floored before exponentiation. For negative ``p`` this is
    not cosmetic: a zero eigenvalue would otherwise send the power to infinity,
    and the positive layer of a sparse signed graph reliably has some.
    """
    w, v = eigh((m + m.T) / 2.0)
    w = np.maximum(w, floor)
    return (v * (w ** p)) @ v.T


def _log_euclidean_mean(a: np.ndarray, b: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """``exp((log A + log B) / 2)``, the p -> 0 limit used for the geometric mean.

    The scalar power mean tends to the geometric mean as p goes to zero. For
    matrices the log-Euclidean mean is the computable analogue and coincides
    with the matrix geometric mean whenever A and B commute; it is used here
    rather than the exact geometric mean because it needs one eigendecomposition
    per argument instead of three matrix square roots, and the two agree closely
    on the near-commuting Laplacian pairs this produces.
    """
    def _log(m):
        w, v = eigh((m + m.T) / 2.0)
        return (v * np.log(np.maximum(w, floor))) @ v.T

    s = (_log(a) + _log(b)) / 2.0
    w, v = eigh((s + s.T) / 2.0)
    return (v * np.exp(w)) @ v.T


def power_mean(a: np.ndarray, b: np.ndarray, p: float) -> np.ndarray:
    """Matrix power mean ``((A^p + B^p) / 2)^{1/p}``, with the p -> 0 limit."""
    if abs(p) < 1e-6:
        return _log_euclidean_mean(a, b)
    m = (_matrix_power(a, p) + _matrix_power(b, p)) / 2.0
    return _matrix_power(m, 1.0 / p)


class PowerMeanLaplacianClustering:
    """Signed power mean Laplacian (Mercado, Tudisco, Hein, ICML 2019).

    Clusters the smallest eigenvectors of ``L_p = M_p(L+_sym, Q-_sym)``, the
    scalar power mean of the positive layer's normalized Laplacian and the
    negative layer's normalized signless Laplacian.

    ``p`` controls how much a cluster has to satisfy both layers at once. A
    power mean is dominated by its smallest argument as ``p`` goes negative, so
    ``p = -10`` will only call something a cluster if it looks like one in the
    positive layer *and* the negative layer, while ``p = 1`` is the arithmetic
    average and lets a strong positive layer carry a thin negative one. Select
    it on the training fold; it is a genuine hyperparameter, not a constant.
    """

    def __init__(self, n_clusters: int = 3, p: float = 1.0,
                 gamma: float | None = 0.0, n_eigen: int | None = None,
                 random_state: int = 42):
        self.n_clusters = n_clusters
        self.p = p
        self.gamma = gamma
        self.n_eigen = n_eigen
        self.random_state = random_state
        self.labels_ = None
        self.eigenvalues_ = None
        self.embedding_ = None

    def fit(self, adjacency, n_clusters: int | None = None):
        if n_clusters is not None:
            self.n_clusters = n_clusters
        a = _as_array(adjacency)
        n = len(a)
        if n <= self.n_clusters:
            self.labels_ = np.arange(n)
            return self

        gamma = default_gamma(a) if self.gamma is None else float(self.gamma)
        a_plus = np.clip(a, 0, None)
        a_minus = np.clip(-a, 0, None)

        # positive layer: Laplacian. negative layer: SIGNLESS Laplacian, which is
        # what makes both operators small on a good partition
        l_plus = _sym_normalized(a_plus, gamma, signless=False)
        q_minus = _sym_normalized(a_minus, gamma, signless=True)

        l_p = power_mean(l_plus, q_minus, self.p)
        self.eigenvalues_, vectors = eigh((l_p + l_p.T) / 2.0)
        n_eigen = self.n_eigen or self.n_clusters
        self.embedding_ = _embed(vectors, n_eigen)
        self.labels_ = _kmeans(self.embedding_, self.n_clusters, self.random_state)
        return self

    def fit_predict(self, adjacency, n_clusters: int | None = None) -> np.ndarray:
        return self.fit(adjacency, n_clusters).labels_
