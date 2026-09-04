"""Non-spectral clusterers behind the same interface as `sponge.py`.

Every class here exposes ``fit(adjacency, n_clusters)`` and ``fit_predict``, sets
``labels_``, and takes a signed adjacency matrix, so the walk-forward runner can
swap them for SPONGE without touching the signal.

Two of these are deliberately dumb. A signed spectral method should beat
hierarchical clustering on a correlation distance and k-means on PCA loadings,
and if it does not then the clustering machinery is not what is producing the
result. That is worth knowing before any of it is called a contribution.

Pivot is not dumb: it is the classic 3-approximation for correlation clustering
(Ailon, Charikar, Newman, "Aggregating inconsistent information", JACM 2008),
and it is the natural non-spectral comparison because it optimises the objective
the signed graph actually encodes, agreements within clusters and disagreements
between them, rather than a spectral relaxation of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans


def _as_array(adjacency) -> np.ndarray:
    a = adjacency.to_numpy() if isinstance(adjacency, pd.DataFrame) else np.asarray(adjacency)
    a = np.array(a, dtype=float, copy=True)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = (a + a.T) / 2.0           # enforce symmetry; k-NN symmetrisation can drift
    np.fill_diagonal(a, 0.0)
    return a


class SignedHierarchicalClustering:
    """Average-linkage hierarchy on the signed correlation distance.

    Distance is ``d_ij = 1 - a_ij`` mapped onto [0, 2], the standard correlation
    distance: perfectly co-moving names sit at 0, uncorrelated at 1, perfectly
    opposed at 2. Average linkage is used rather than Ward because the input is a
    distance matrix, not a Euclidean embedding, and Ward's update rule assumes
    the latter.

    The k-NN graph is sparse, so most pairs have no edge and land at distance 1
    (treated as uncorrelated) rather than being dropped, which keeps the distance
    matrix complete and the linkage well defined.
    """

    def __init__(self, n_clusters: int = 3, method: str = "average"):
        self.n_clusters = n_clusters
        self.method = method
        self.labels_ = None

    def fit(self, adjacency, n_clusters: int | None = None):
        if n_clusters is not None:
            self.n_clusters = n_clusters
        a = _as_array(adjacency)
        n = len(a)
        if n <= self.n_clusters:
            self.labels_ = np.arange(n)
            return self

        dist = np.clip(1.0 - a, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        z = linkage(squareform(dist, checks=False), method=self.method)
        # fcluster returns 1-based labels; shift so every clusterer here is 0-based
        self.labels_ = fcluster(z, t=self.n_clusters, criterion="maxclust") - 1
        return self

    def fit_predict(self, adjacency, n_clusters: int | None = None) -> np.ndarray:
        return self.fit(adjacency, n_clusters).labels_


class PCALoadingKMeans:
    """k-means on the leading eigenvectors of the adjacency matrix.

    The other deliberately dumb baseline: it throws away the sign structure by
    embedding with a plain symmetric eigendecomposition and clustering the
    loadings in Euclidean space, which is what someone would do if they had never
    heard of signed graphs. If SPONGE cannot beat this, the signed machinery is
    not earning its place.
    """

    def __init__(self, n_clusters: int = 3, n_components: int | None = None,
                 random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.random_state = random_state
        self.labels_ = None
        self.embedding_ = None

    def fit(self, adjacency, n_clusters: int | None = None):
        if n_clusters is not None:
            self.n_clusters = n_clusters
        a = _as_array(adjacency)
        n = len(a)
        if n <= self.n_clusters:
            self.labels_ = np.arange(n)
            return self

        k = self.n_components or self.n_clusters
        k = min(k, n - 1)
        vals, vecs = np.linalg.eigh(a)
        # largest |eigenvalue| first: the dominant structure, whatever its sign
        order = np.argsort(-np.abs(vals))[:k]
        self.embedding_ = vecs[:, order]
        km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.labels_ = km.fit_predict(self.embedding_)
        return self

    def fit_predict(self, adjacency, n_clusters: int | None = None) -> np.ndarray:
        return self.fit(adjacency, n_clusters).labels_


class PivotCorrelationClustering:
    """Correlation clustering by the Pivot algorithm (Ailon, Charikar, Newman 2008).

    The objective: partition so that positive edges fall inside clusters and
    negative edges fall between them, minimising disagreements. Exact
    optimisation is NP-hard; Pivot is a randomised 3-approximation for the
    complete-graph case and is the standard non-spectral baseline.

    The algorithm is one line of intuition: pick a random un-clustered node, make
    it a pivot, pull in every un-clustered node joined to it by a positive edge,
    remove them all, repeat. It infers the number of clusters rather than taking
    it, which is a genuine difference from every spectral method here and the
    reason it is worth running: if the data supports a natural cluster count, an
    algorithm that is allowed to find one will not return three.

    ``n_clusters`` is accepted for interface compatibility and used only to
    reconcile the result to a requested count when `match_k` is set, by merging
    the smallest clusters into their most positively connected neighbour. That
    is off by default: forcing k defeats the point of running Pivot.
    """

    def __init__(self, n_clusters: int = 3, n_restarts: int = 10,
                 match_k: bool = False, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_restarts = n_restarts
        self.match_k = match_k
        self.random_state = random_state
        self.labels_ = None
        self.n_clusters_found_ = None

    @staticmethod
    def _disagreements(a: np.ndarray, labels: np.ndarray) -> float:
        """Positive weight cut between clusters plus negative weight kept within."""
        same = labels[:, None] == labels[None, :]
        pos, neg = np.clip(a, 0, None), np.clip(-a, 0, None)
        return float(pos[~same].sum() + neg[same].sum()) / 2.0

    def _one_pass(self, a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(a)
        labels = np.full(n, -1, dtype=int)
        remaining = list(rng.permutation(n))
        c = 0
        while remaining:
            pivot = remaining[0]
            members = [pivot]
            for node in remaining[1:]:
                if a[pivot, node] > 0:
                    members.append(node)
            for m in members:
                labels[m] = c
            member_set = set(members)
            remaining = [x for x in remaining if x not in member_set]
            c += 1
        return labels

    def _merge_to_k(self, a: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
        """Merge smallest clusters into their most positively connected neighbour."""
        labels = labels.copy()
        while len(np.unique(labels)) > k:
            uniq, counts = np.unique(labels, return_counts=True)
            smallest = uniq[np.argmin(counts)]
            mask = labels == smallest
            best, best_w = None, -np.inf
            for other in uniq:
                if other == smallest:
                    continue
                w = a[np.ix_(mask, labels == other)].sum()
                if w > best_w:
                    best, best_w = other, w
            labels[mask] = best if best is not None else smallest
            if best is None:
                break
        # compact to 0..k-1
        return np.unique(labels, return_inverse=True)[1]

    def fit(self, adjacency, n_clusters: int | None = None):
        if n_clusters is not None:
            self.n_clusters = n_clusters
        a = _as_array(adjacency)
        n = len(a)
        if n == 0:
            self.labels_ = np.empty(0, dtype=int)
            return self

        rng = np.random.default_rng(self.random_state)
        best, best_cost = None, np.inf
        for _ in range(max(1, self.n_restarts)):
            labels = self._one_pass(a, rng)
            cost = self._disagreements(a, labels)
            if cost < best_cost:
                best, best_cost = labels, cost

        if self.match_k and len(np.unique(best)) > self.n_clusters:
            best = self._merge_to_k(a, best, self.n_clusters)
        self.labels_ = np.unique(best, return_inverse=True)[1]
        self.n_clusters_found_ = int(len(np.unique(self.labels_)))
        self.disagreements_ = best_cost
        return self

    def fit_predict(self, adjacency, n_clusters: int | None = None) -> np.ndarray:
        return self.fit(adjacency, n_clusters).labels_
