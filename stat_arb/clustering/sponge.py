"""
SPONGE (Signed Positive Over Negative Generalized Eigenproblem) clustering.

Reference: Cucuringu et al., "SPONGE: A generalized eigenproblem for clustering
signed networks"
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy.linalg import eigh
from sklearn.cluster import KMeans


class SPONGEClustering:
    """
    SPONGE clustering for signed graphs.

    Solves the generalized eigenproblem:
        (L+ + tau- * D-) v = lambda (L- + tau+ * D+) v

    where:
    - A+ = max(0, A), A- = max(0, -A)
    - D+, D- are degree matrices of A+, A-
    - L+ = D+ - A+, L- = D- - A-
    - tau+, tau- are regularization parameters
    """

    def __init__(
        self,
        n_clusters: int = 3,
        tau_plus: float = 1.0,
        tau_minus: float = 1.0,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.random_state = random_state
        self.labels_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.embedding_ = None

    def fit(
        self,
        adjacency: np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> 'SPONGEClustering':
        """
        Fit SPONGE clustering to signed adjacency matrix.

        Parameters
        ----------
        adjacency : np.ndarray
            Signed adjacency matrix (positive and negative edges)
        n_clusters : int, optional
            Number of clusters (overrides init parameter)

        Returns
        -------
        self
        """
        if n_clusters is not None:
            self.n_clusters = n_clusters

        # Convert DataFrame to array if needed
        if isinstance(adjacency, pd.DataFrame):
            adjacency = adjacency.values

        # Split into positive and negative parts
        A_plus = np.clip(adjacency, 0, None)
        A_minus = np.clip(-adjacency, 0, None)

        # Degree matrices
        D_plus = np.diag(A_plus.sum(axis=1))
        D_minus = np.diag(A_minus.sum(axis=1))

        # Laplacians
        L_plus = D_plus - A_plus
        L_minus = D_minus - A_minus

        # SPONGE matrices
        M1 = L_plus + self.tau_minus * D_minus
        M2 = L_minus + self.tau_plus * D_plus

        # Add small regularization to M2 for numerical stability
        M2 = M2 + 1e-6 * np.eye(len(M2))

        # Solve generalized eigenproblem
        self.eigenvalues_, self.eigenvectors_ = eigh(M1, M2)

        # Build embedding using first k eigenvectors
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

    def fit_symmetric(
        self,
        adjacency: np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> 'SPONGEClustering':
        """
        Fit the symmetric SPONGE variant, SPONGEsym.

        Cucuringu, Davies, Glielmo and Tyagi (AISTATS 2019) define SPONGEsym on
        the symmetrically NORMALIZED signed Laplacians

            L+_sym = I - (D+)^{-1/2} A+ (D+)^{-1/2}
            L-_sym = I - (D-)^{-1/2} A- (D-)^{-1/2}

        solving (L+_sym + tau- I) v = lambda (L-_sym + tau+ I) v for the
        smallest eigenvalues. Normalising by degree is the whole point of the
        variant: it stops high-degree nodes dominating the embedding, which is
        what makes it behave differently from plain SPONGE on a graph with an
        uneven degree distribution.

        This previously whitened the UNNORMALIZED SPONGE pencil, computing
        M2^{-1/2} M1 M2^{-1/2} from the same M1 and M2 that `fit` uses. That is
        a similarity transform of the same generalized eigenproblem, so it has
        the same eigenvalues and returns the same clustering: measured on
        planted blocks, the two paths gave identical labels and identical
        eigenvalues to machine precision. The published comparison's "SPONGEsym"
        column was therefore plain SPONGE run a second time, not a second
        method.
        """
        if n_clusters is not None:
            self.n_clusters = n_clusters

        if isinstance(adjacency, pd.DataFrame):
            adjacency = adjacency.values

        # Split into positive and negative parts
        A_plus = np.clip(adjacency, 0, None)
        A_minus = np.clip(-adjacency, 0, None)

        # Degree vectors. An isolated node in either layer has zero degree, so
        # the inverse square root is floored rather than allowed to divide by 0.
        d_plus = np.maximum(A_plus.sum(axis=1), 1e-12)
        d_minus = np.maximum(A_minus.sum(axis=1), 1e-12)
        inv_sqrt_plus = 1.0 / np.sqrt(d_plus)
        inv_sqrt_minus = 1.0 / np.sqrt(d_minus)

        n = len(A_plus)
        eye = np.eye(n)

        # Symmetrically normalized signed Laplacians
        L_plus_sym = eye - (inv_sqrt_plus[:, None] * A_plus * inv_sqrt_plus[None, :])
        L_minus_sym = eye - (inv_sqrt_minus[:, None] * A_minus * inv_sqrt_minus[None, :])

        # SPONGEsym pencil: the regularisers are identity, not degree matrices,
        # because the Laplacians are already degree-normalized
        M1 = L_plus_sym + self.tau_minus * eye
        M2 = L_minus_sym + self.tau_plus * eye
        M2 = M2 + 1e-6 * eye

        self.eigenvalues_, self.eigenvectors_ = eigh(M1, M2)

        # Build embedding
        U = self.eigenvectors_[:, :self.n_clusters]
        row_norms = np.linalg.norm(U, axis=1, keepdims=True)
        row_norms = np.where(row_norms < 1e-8, 1, row_norms)
        U_norm = U / row_norms
        self.embedding_ = U_norm

        # Cluster
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
        symmetric: bool = False,
    ) -> np.ndarray:
        """Fit and return cluster labels."""
        if symmetric:
            self.fit_symmetric(adjacency, n_clusters)
        else:
            self.fit(adjacency, n_clusters)
        return self.labels_

    def compute_sponge_objective(
        self,
        adjacency: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute SPONGE objective value.

        Objective: minimize negative edges within clusters + positive edges between clusters.
        """
        if labels is None:
            labels = self.labels_

        if isinstance(adjacency, pd.DataFrame):
            adjacency = adjacency.values

        A_plus = np.clip(adjacency, 0, None)
        A_minus = np.clip(-adjacency, 0, None)

        neg_within = 0
        pos_between = 0

        for c in np.unique(labels):
            mask = labels == c
            # Negative edges within cluster
            neg_within += A_minus[np.ix_(mask, mask)].sum() / 2
            # Positive edges between clusters
            pos_between += A_plus[np.ix_(mask, ~mask)].sum()

        return neg_within + pos_between

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
