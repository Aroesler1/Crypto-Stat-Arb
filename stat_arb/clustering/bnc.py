"""
BNC (Balance Normalized Cut) clustering for signed graphs.
"""
import numpy as np
import pandas as pd
from typing import Optional
from scipy.linalg import eigh
from sklearn.cluster import KMeans


class BNCClustering:
    """
    Balance Normalized Cut clustering for signed graphs.

    Solves: (A+ - A-) v = lambda D_tot v

    where D_tot = D+ + D- is the total degree matrix.
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
    ) -> 'BNCClustering':
        """
        Fit BNC clustering to signed adjacency matrix.

        Parameters
        ----------
        adjacency : np.ndarray or pd.DataFrame
            Signed adjacency matrix
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

        # Split into positive and negative parts
        A_plus = np.clip(adjacency, 0, None)
        A_minus = np.clip(-adjacency, 0, None)

        # Degree matrices
        D_plus = np.diag(A_plus.sum(axis=1))
        D_minus = np.diag(A_minus.sum(axis=1))
        D_tot = D_plus + D_minus

        # BNC matrix: L_bnc = A+ - A-
        L_bnc = A_plus - A_minus

        # Add regularization
        D_tot_reg = D_tot + 1e-6 * np.eye(len(D_tot))

        # Solve generalized eigenproblem
        self.eigenvalues_, self.eigenvectors_ = eigh(L_bnc, D_tot_reg)

        # Build embedding from first k eigenvectors
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

    def compute_bnc_objective(
        self,
        adjacency: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute BNC objective (balance normalized cut value).
        """
        if labels is None:
            labels = self.labels_

        if isinstance(adjacency, pd.DataFrame):
            adjacency = adjacency.values

        A_plus = np.clip(adjacency, 0, None)
        A_minus = np.clip(-adjacency, 0, None)

        total_cut = 0
        for c in np.unique(labels):
            mask = labels == c

            # Volume of cluster
            vol_c = (A_plus[mask, :].sum() + A_minus[mask, :].sum())

            if vol_c < 1e-8:
                continue

            # Cut: negative within + positive between
            neg_within = A_minus[np.ix_(mask, mask)].sum() / 2
            pos_between = A_plus[np.ix_(mask, ~mask)].sum()

            total_cut += (neg_within + pos_between) / vol_c

        return total_cut

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
