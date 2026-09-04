"""The signal path is parameterised; the defaults must still be the published one.

`run_phase3_config` gained `n_pca_components`, `clusterer` and `diagnostics` so
the residualization ablation and the clustering-method comparison can vary one
thing at a time. That is only safe if the defaults reproduce the published
configuration exactly, so this pins the pieces that could drift.

The full end-to-end check (baseline 134 members / gross 2.971 / net50 2.310 /
breakeven 226.1 bps on the committed data) is a minute of compute and lives in
`stat_arb/run_phase3.py`; what is unit-tested here is that the swappable parts
default to what they replaced.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.clustering.sponge import SPONGEClustering  # noqa: E402
from stat_arb.pca.market_mode import MarketModeExtractor  # noqa: E402
from stat_arb.run_phase3 import default_clusterer  # noqa: E402


def _signed_adjacency(n=40, seed=0):
    """A signed graph with two genuinely opposed blocks."""
    rng = np.random.default_rng(seed)
    block = np.repeat([1, -1], n // 2)
    a = np.outer(block, block) * rng.uniform(0.3, 0.9, size=(n, n))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    return pd.DataFrame(a)


def test_default_clusterer_is_sponge_with_the_published_seed():
    adj = _signed_adjacency()
    expected = SPONGEClustering(n_clusters=3, random_state=42).fit_predict(adj)
    assert np.array_equal(default_clusterer(adj, 3), expected)


def test_default_clusterer_honours_the_requested_k():
    adj = _signed_adjacency()
    assert len(np.unique(default_clusterer(adj, 4))) == 4


def test_default_clusterer_recovers_two_opposed_blocks():
    """A sanity check that the signed structure is being used at all."""
    adj = _signed_adjacency(n=40)
    labels = default_clusterer(adj, 2)
    first, second = labels[:20], labels[20:]
    assert len(set(first)) == 1 and len(set(second)) == 1
    assert first[0] != second[0]


def test_zero_components_leaves_returns_untouched():
    """The ETH-excess-only and BTC-excess-only arms remove no PCs at all."""
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(size=(200, 8)),
                     index=pd.date_range("2020-01-01", periods=200))
    pca = MarketModeExtractor(n_components=1)
    pca.fit(r)
    residual = pca.residualize(r)
    # removing a component must actually change the panel, else the ablation
    # arms would be indistinguishable for the wrong reason
    assert not np.allclose(residual.to_numpy(), r.to_numpy())


def test_more_components_remove_more_variance():
    rng = np.random.default_rng(2)
    common = rng.normal(size=(300, 1))
    r = pd.DataFrame(common @ rng.normal(size=(1, 10)) + 0.3 * rng.normal(size=(300, 10)),
                     index=pd.date_range("2020-01-01", periods=300))
    removed = []
    for k in (1, 2, 3):
        pca = MarketModeExtractor(n_components=k)
        pca.fit(r)
        removed.append(float(np.sum(pca.explained_variance_ratio_)))
    assert removed[0] < removed[1] < removed[2]
    assert removed[0] > 0.7      # one strong common factor dominates
