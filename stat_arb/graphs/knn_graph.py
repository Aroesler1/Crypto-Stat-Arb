"""
k-NN graph construction from correlation matrices.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class KNNGraphBuilder:
    """Build k-NN graphs from correlation matrices."""

    def __init__(self, k: int = 10):
        self.k = k

    def compute_correlation_matrix(
        self,
        returns: pd.DataFrame,
        method: str = 'pearson',
    ) -> pd.DataFrame:
        """Compute correlation matrix from returns."""
        return returns.corr(method=method)

    def build_abs_correlation_knn(
        self,
        corr: pd.DataFrame,
        k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build k-NN graph using absolute correlation.

        For each node, keep edges to k most correlated neighbors
        (by absolute correlation value).
        """
        if k is None:
            k = self.k

        corr_abs = corr.abs()
        n = len(corr)

        # Initialize adjacency matrix
        adj = pd.DataFrame(0.0, index=corr.index, columns=corr.columns)

        for i, idx in enumerate(corr.index):
            # Get correlations for this node (exclude self)
            row = corr_abs.loc[idx].drop(idx)

            # Find k largest (most correlated)
            top_k = row.nlargest(k)

            # Add edges (use original signed correlation values)
            for neighbor in top_k.index:
                adj.loc[idx, neighbor] = corr.loc[idx, neighbor]

        # Symmetrize: keep edge if either direction selected it
        adj_sym = (adj + adj.T) / 2

        # Zero diagonal
        np.fill_diagonal(adj_sym.values, 0)

        return adj_sym

    def build_weighted_knn(
        self,
        corr: pd.DataFrame,
        k: Optional[int] = None,
        weight_by_rank: bool = False,
    ) -> pd.DataFrame:
        """
        Build weighted k-NN graph.

        Can optionally weight edges by rank (closer neighbors get higher weight).
        """
        if k is None:
            k = self.k

        adj = self.build_abs_correlation_knn(corr, k)

        if weight_by_rank:
            # Re-weight by rank within each row
            for idx in adj.index:
                row = adj.loc[idx]
                nonzero = row[row != 0]
                if len(nonzero) > 0:
                    ranks = nonzero.abs().rank(ascending=False)
                    weights = 1.0 / ranks
                    weights = weights / weights.sum()  # Normalize
                    adj.loc[idx, nonzero.index] = adj.loc[idx, nonzero.index].abs() * weights * np.sign(adj.loc[idx, nonzero.index])

        return adj

    def get_connected_components(
        self,
        adj: pd.DataFrame,
    ) -> Tuple[int, np.ndarray]:
        """
        Find connected components in the graph.

        Returns (n_components, labels).
        """
        # Convert to sparse matrix using absolute values for connectivity
        adj_sparse = csr_matrix(adj.abs().values)

        n_components, labels = connected_components(
            adj_sparse,
            directed=False,
            return_labels=True,
        )

        return n_components, labels

    def compute_graph_diagnostics(
        self,
        adj: pd.DataFrame,
    ) -> dict:
        """Compute graph diagnostics."""
        # Degree statistics
        degree = adj.abs().sum(axis=1)
        n_components, labels = self.get_connected_components(adj)

        # Sign balance
        positive_edges = (adj > 0).sum().sum() / 2
        negative_edges = (adj < 0).sum().sum() / 2
        total_edges = positive_edges + negative_edges

        return {
            'n_nodes': len(adj),
            'avg_degree': degree.mean(),
            'min_degree': degree.min(),
            'max_degree': degree.max(),
            'n_components': n_components,
            'positive_edges': int(positive_edges),
            'negative_edges': int(negative_edges),
            'sign_balance': positive_edges / (total_edges + 1e-8),
        }
