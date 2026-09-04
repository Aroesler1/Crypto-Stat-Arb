"""
K-selection methods for clustering.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.linalg import eigh


class KSelector:
    """
    Select optimal number of clusters using multiple criteria.

    Methods:
    - Eigengap heuristic
    - Silhouette score
    - Calinski-Harabasz index
    - Davies-Bouldin index
    - Gap statistic
    """

    def __init__(
        self,
        k_range: Tuple[int, int] = (2, 10),
        random_state: int = 42,
    ):
        self.k_range = k_range
        self.random_state = random_state

    def eigengap(
        self,
        eigenvalues: np.ndarray,
        max_k: Optional[int] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Eigengap heuristic: find k with largest gap in eigenvalue spectrum.

        Returns (best_k, gaps).
        """
        if max_k is None:
            max_k = min(self.k_range[1], len(eigenvalues) - 1)

        # Compute gaps between consecutive eigenvalues
        sorted_eigs = np.sort(eigenvalues)
        gaps = np.diff(sorted_eigs[:max_k + 1])

        # Best k is index of largest gap + 1 (since gap[i] is between eig[i] and eig[i+1])
        best_k = np.argmax(gaps[:max_k]) + 1
        best_k = max(self.k_range[0], min(best_k, self.k_range[1]))

        return best_k, gaps

    def silhouette(
        self,
        embedding: np.ndarray,
        k_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, Dict[int, float]]:
        """
        Find best k by silhouette score.

        Returns (best_k, scores_dict).
        """
        if k_range is None:
            k_range = self.k_range

        scores = {}
        for k in range(k_range[0], k_range[1] + 1):
            if k >= len(embedding):
                continue
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(embedding)

            if len(np.unique(labels)) < 2:
                scores[k] = -1
            else:
                scores[k] = silhouette_score(embedding, labels)

        best_k = max(scores, key=scores.get)
        return best_k, scores

    def calinski_harabasz(
        self,
        embedding: np.ndarray,
        k_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, Dict[int, float]]:
        """
        Find best k by Calinski-Harabasz index (higher is better).

        Returns (best_k, scores_dict).
        """
        if k_range is None:
            k_range = self.k_range

        scores = {}
        for k in range(k_range[0], k_range[1] + 1):
            if k >= len(embedding):
                continue
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(embedding)

            if len(np.unique(labels)) < 2:
                scores[k] = 0
            else:
                scores[k] = calinski_harabasz_score(embedding, labels)

        best_k = max(scores, key=scores.get)
        return best_k, scores

    def davies_bouldin(
        self,
        embedding: np.ndarray,
        k_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, Dict[int, float]]:
        """
        Find best k by Davies-Bouldin index (lower is better).

        Returns (best_k, scores_dict).
        """
        if k_range is None:
            k_range = self.k_range

        scores = {}
        for k in range(k_range[0], k_range[1] + 1):
            if k >= len(embedding):
                continue
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(embedding)

            if len(np.unique(labels)) < 2:
                scores[k] = float('inf')
            else:
                scores[k] = davies_bouldin_score(embedding, labels)

        best_k = min(scores, key=scores.get)
        return best_k, scores

    def gap_statistic(
        self,
        embedding: np.ndarray,
        k_range: Optional[Tuple[int, int]] = None,
        n_refs: int = 10,
    ) -> Tuple[int, Dict[int, float]]:
        """
        Gap statistic for k-selection.

        Compares within-cluster dispersion to that of reference null distribution.
        Returns (best_k, gap_values).
        """
        if k_range is None:
            k_range = self.k_range

        def compute_wk(X, labels):
            """Compute within-cluster sum of squares."""
            wk = 0
            for c in np.unique(labels):
                cluster_points = X[labels == c]
                if len(cluster_points) > 1:
                    center = cluster_points.mean(axis=0)
                    wk += np.sum((cluster_points - center) ** 2)
            return wk

        # Compute for actual data
        wk_actual = {}
        for k in range(k_range[0], k_range[1] + 1):
            if k >= len(embedding):
                continue
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(embedding)
            wk_actual[k] = np.log(compute_wk(embedding, labels) + 1e-10)

        # Generate reference data (uniform over bounding box)
        mins = embedding.min(axis=0)
        maxs = embedding.max(axis=0)

        wk_refs = {k: [] for k in wk_actual.keys()}
        rng = np.random.RandomState(self.random_state)

        for _ in range(n_refs):
            ref_data = rng.uniform(mins, maxs, size=embedding.shape)
            for k in wk_actual.keys():
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(ref_data)
                wk_refs[k].append(np.log(compute_wk(ref_data, labels) + 1e-10))

        # Compute gap = E[log(Wk_ref)] - log(Wk)
        gaps = {}
        for k in wk_actual.keys():
            gaps[k] = np.mean(wk_refs[k]) - wk_actual[k]

        # Standard gap criterion: smallest k where gap[k] >= gap[k+1] - s[k+1]
        # Simplified: just take max gap
        best_k = max(gaps, key=gaps.get)
        return best_k, gaps

    def select_k(
        self,
        embedding: np.ndarray,
        eigenvalues: Optional[np.ndarray] = None,
        method: str = 'consensus',
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[int, Dict[str, Dict]]:
        """
        Select k using specified method or consensus.

        Methods: 'eigengap', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'gap', 'consensus'

        Returns (best_k, all_results).
        """
        results = {}

        # Run all methods
        if eigenvalues is not None:
            k_eig, gaps = self.eigengap(eigenvalues)
            results['eigengap'] = {'best_k': k_eig, 'scores': gaps}
        else:
            k_eig = None

        k_sil, sil_scores = self.silhouette(embedding)
        results['silhouette'] = {'best_k': k_sil, 'scores': sil_scores}

        k_ch, ch_scores = self.calinski_harabasz(embedding)
        results['calinski_harabasz'] = {'best_k': k_ch, 'scores': ch_scores}

        k_db, db_scores = self.davies_bouldin(embedding)
        results['davies_bouldin'] = {'best_k': k_db, 'scores': db_scores}

        # Gap statistic (can be slow, so make it optional)
        try:
            k_gap, gap_scores = self.gap_statistic(embedding, n_refs=5)
            results['gap'] = {'best_k': k_gap, 'scores': gap_scores}
        except Exception:
            k_gap = None

        if method == 'consensus':
            # Weighted voting
            if weights is None:
                weights = {
                    'eigengap': 1.0,
                    'silhouette': 2.0,
                    'calinski_harabasz': 1.0,
                    'davies_bouldin': 1.0,
                    'gap': 1.0,
                }

            votes = {}
            for m, w in weights.items():
                if m in results:
                    k = results[m]['best_k']
                    votes[k] = votes.get(k, 0) + w

            best_k = max(votes, key=votes.get) if votes else 3
        else:
            if method in results:
                best_k = results[method]['best_k']
            else:
                best_k = 3

        return best_k, results

    def compute_cluster_stability(
        self,
        labels_current: np.ndarray,
        labels_previous: np.ndarray,
    ) -> float:
        """
        Compute cluster stability using Adjusted Mutual Information.
        """
        from sklearn.metrics import adjusted_mutual_info_score
        return adjusted_mutual_info_score(labels_current, labels_previous)


def signflip_parallel_analysis(
    adjacency,
    n_replicates: int = 50,
    max_k: int = 12,
    random_state: int = 42,
) -> tuple:
    """Estimate the number of clusters by signflip parallel analysis.

    Hong and Cape, "Signflip parallel analysis" (arXiv:2509.05722). The idea is
    the same as Horn's parallel analysis for factor models, adapted to a matrix
    whose entries carry signs: build a null by randomly flipping the sign of
    every entry, which destroys any block structure while preserving the
    magnitude distribution and the sparsity pattern exactly, then keep only the
    eigenvalues of the real matrix that stand above what that null produces.

    This is worth having because every other criterion in this module scores a
    partition that has already been produced, so all of them will happily rank
    three clusters against four in a graph that has no clusters at all. Parallel
    analysis can return zero, which is the honest answer for a correlation graph
    with no block structure left in it after residualization.

    Returns ``(k, threshold, eigenvalues)``. ``k`` is clipped to ``max_k`` and
    reported as at least 2, because the caller has to cluster something, but the
    raw count above the threshold is recoverable from the returned eigenvalues.
    """
    a = adjacency.to_numpy() if isinstance(adjacency, pd.DataFrame) else np.asarray(adjacency)
    a = np.nan_to_num(np.array(a, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    a = (a + a.T) / 2.0
    np.fill_diagonal(a, 0.0)
    n = len(a)
    if n < 3:
        return 2, np.nan, np.array([])

    observed = np.sort(np.abs(np.linalg.eigvalsh(a)))[::-1]

    rng = np.random.default_rng(random_state)
    null_max = np.empty(n_replicates)
    iu = np.triu_indices(n, k=1)
    for b in range(n_replicates):
        signs = rng.choice((-1.0, 1.0), size=len(iu[0]))
        flipped = np.zeros_like(a)
        flipped[iu] = a[iu] * signs
        flipped = flipped + flipped.T
        null_max[b] = np.abs(np.linalg.eigvalsh(flipped)).max()

    threshold = float(np.max(null_max))
    k_raw = int((observed > threshold).sum())
    return int(np.clip(k_raw, 2, max_k)), threshold, observed
