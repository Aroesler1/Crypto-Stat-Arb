"""
Signed spectral clustering.
"""
import numpy as np
import pandas as pd
from typing import Optional
from scipy.linalg import eigh
from sklearn.cluster import KMeans


class SignedSpectralClustering:
    """
    Signed spectral clustering using signed normalized Laplacian.

    L_sym = I - D^{-1/2} A D^{-1/2}

    where A is the signed adjacency (A+ - A-) and D is the absolute degree matrix.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.embedding_ = None

    def fit(
        self,
        adjacency: np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> 'SignedSpectralClustering':
        """
        Fit signed spectral clustering.

        Parameters
        ----------
        adjacency : np.ndarray or pd.DataFrame
            Signed adjacency matrix (can be combined W+ - W- or raw signed correlation)
        n_clusters : int, optional
            Number of clusters

        Returns
        -------
        self
        """
        if n_clusters is not None:
            self.n_clusters = n_clusters

        if isinstance(adjacency, pd.DataFrame):
            adjacency = adjacency.values

        A = adjacency.copy()

        # Compute absolute degree
        degree_vector = np.abs(A).sum(axis=1)
        degree_vector = np.maximum(degree_vector, 1e-8)  # Avoid division by zero

        # D^{-1/2}
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_vector))

        # Signed normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
        I = np.eye(A.shape[0])
        L_sym = I - D_inv_sqrt @ A @ D_inv_sqrt

        # Eigendecompose (want smallest eigenvalues for graph cut)
        self.eigenvalues_, self.eigenvectors_ = eigh(L_sym)

        # Build embedding from first k eigenvectors (smallest eigenvalues)
        U = self.eigenvectors_[:, :self.n_clusters]

        # Normalize rows
        row_norms = np.linalg.norm(U, axis=1, keepdims=True)
        row_norms = np.where(row_norms < 1e-8, 1, row_norms)
        U_norm = U / row_norms
        self.embedding_ = U_norm

        # Cluster using k-means
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.labels_ = kmeans.fit_predict(U_norm)

        return self

    def fit_predict(
        self,
        adjacency: np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(adjacency, n_clusters)
        return self.labels_

    def compute_cluster_quality(
        self,
        adjacency: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute cluster quality metrics.

        Returns dict with:
        - intra_positive: positive edges within clusters
        - inter_negative: negative edges between clusters
        - ratio: quality ratio
        """
        if labels is None:
            labels = self.labels_

        if isinstance(adjacency, pd.DataFrame):
            adjacency = adjacency.values

        A_plus = np.clip(adjacency, 0, None)
        A_minus = np.clip(-adjacency, 0, None)

        intra_positive = 0
        inter_negative = 0
        intra_negative = 0
        inter_positive = 0

        for c in np.unique(labels):
            mask = labels == c
            # Positive edges within cluster (good)
            intra_positive += A_plus[np.ix_(mask, mask)].sum() / 2
            # Negative edges between clusters (good)
            inter_negative += A_minus[np.ix_(mask, ~mask)].sum() / 2
            # Negative edges within cluster (bad - frustration)
            intra_negative += A_minus[np.ix_(mask, mask)].sum() / 2
            # Positive edges between clusters (bad - should be in same cluster)
            inter_positive += A_plus[np.ix_(mask, ~mask)].sum() / 2

        total_good = intra_positive + inter_negative
        total_bad = intra_negative + inter_positive

        return {
            'intra_positive': intra_positive,
            'inter_negative': inter_negative,
            'intra_negative': intra_negative,
            'inter_positive': inter_positive,
            'quality_ratio': total_good / (total_bad + 1e-8),
            'frustration': total_bad / (total_good + total_bad + 1e-8),
        }

    def get_cluster_assignment_matrix(self) -> np.ndarray:
        """Return cluster assignment as indicator matrix."""
        n = len(self.labels_)
        k = self.n_clusters
        H = np.zeros((n, k))
        for i, c in enumerate(self.labels_):
            H[i, c] = 1
        return H

    def get_cluster_members(
        self,
        tokens: list,
    ) -> dict:
        """Get dictionary mapping cluster ID to list of tokens."""
        members = {}
        for c in np.unique(self.labels_):
            mask = self.labels_ == c
            members[c] = [tokens[i] for i in np.where(mask)[0]]
        return members
