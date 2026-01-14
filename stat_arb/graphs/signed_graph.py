"""
Signed graph construction with separate positive/negative k-NN.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class SignedGraphBuilder:
    """
    Build signed k-NN graphs with separate handling of positive
    and negative edges.
    """

    def __init__(
        self,
        k_pos: int = 5,   # k for positive neighbors
        k_neg: int = 5,   # k for negative neighbors
    ):
        self.k_pos = k_pos
        self.k_neg = k_neg

    def build_signed_knn(
        self,
        corr: pd.DataFrame,
        k_pos: Optional[int] = None,
        k_neg: Optional[int] = None,
        symmetrize: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build signed k-NN graph with separate positive and negative edges.

        For each node:
        - Keep top k_pos most positively correlated neighbors
        - Keep top k_neg most negatively correlated neighbors

        Returns (W_positive, W_negative) weight matrices.
        """
        if k_pos is None:
            k_pos = self.k_pos
        if k_neg is None:
            k_neg = self.k_neg

        n = len(corr)
        W_pos = pd.DataFrame(0.0, index=corr.index, columns=corr.columns)
        W_neg = pd.DataFrame(0.0, index=corr.index, columns=corr.columns)

        for idx in corr.index:
            row = corr.loc[idx].drop(idx)

            # Positive neighbors: top k_pos positive correlations
            positive_corrs = row[row > 0]
            if len(positive_corrs) > 0:
                top_pos = positive_corrs.nlargest(min(k_pos, len(positive_corrs)))
                for neighbor in top_pos.index:
                    W_pos.loc[idx, neighbor] = corr.loc[idx, neighbor]

            # Negative neighbors: top k_neg negative correlations (most negative)
            negative_corrs = row[row < 0]
            if len(negative_corrs) > 0:
                top_neg = negative_corrs.nsmallest(min(k_neg, len(negative_corrs)))
                for neighbor in top_neg.index:
                    W_neg.loc[idx, neighbor] = abs(corr.loc[idx, neighbor])

        if symmetrize:
            # Symmetrize: keep edge if either direction selected it
            W_pos = (W_pos + W_pos.T) / 2
            W_neg = (W_neg + W_neg.T) / 2

        # Zero diagonal
        np.fill_diagonal(W_pos.values, 0)
        np.fill_diagonal(W_neg.values, 0)

        return W_pos, W_neg

    def build_combined_signed_adjacency(
        self,
        corr: pd.DataFrame,
        k_pos: Optional[int] = None,
        k_neg: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build combined signed adjacency matrix.

        Positive edges stored as positive values, negative as negative.
        """
        W_pos, W_neg = self.build_signed_knn(corr, k_pos, k_neg)

        # Combine: positive edges positive, negative edges negative
        W_combined = W_pos - W_neg

        return W_combined

    def compute_signed_laplacian(
        self,
        W_pos: pd.DataFrame,
        W_neg: pd.DataFrame,
        normalized: bool = True,
    ) -> np.ndarray:
        """
        Compute signed graph Laplacian.

        L = D - A where A = W_pos - W_neg
        For normalized: L_sym = I - D^{-1/2} A D^{-1/2}
        """
        A = W_pos.values - W_neg.values
        D_pos = np.diag(W_pos.values.sum(axis=1))
        D_neg = np.diag(W_neg.values.sum(axis=1))
        D = D_pos + D_neg  # Total degree from absolute values

        if normalized:
            # Normalized Laplacian
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
            L = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
        else:
            # Unnormalized Laplacian
            L = D - A

        return L

    def compute_graph_stats(
        self,
        W_pos: pd.DataFrame,
        W_neg: pd.DataFrame,
    ) -> dict:
        """Compute signed graph statistics."""
        pos_degree = W_pos.sum(axis=1)
        neg_degree = W_neg.sum(axis=1)
        total_degree = pos_degree + neg_degree

        n_pos_edges = (W_pos > 0).sum().sum() / 2
        n_neg_edges = (W_neg > 0).sum().sum() / 2

        return {
            'n_nodes': len(W_pos),
            'n_positive_edges': int(n_pos_edges),
            'n_negative_edges': int(n_neg_edges),
            'avg_positive_degree': pos_degree.mean(),
            'avg_negative_degree': neg_degree.mean(),
            'avg_total_degree': total_degree.mean(),
            'frustration_ratio': n_neg_edges / (n_pos_edges + n_neg_edges + 1e-8),
        }
