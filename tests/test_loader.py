"""Tests for data alignment.

The one thing pinned here is column ORDER. `get_aligned_data` used to derive the
panel's columns by iterating a Python set, so the order changed with the
per-process string hash seed. That reorders the k-NN graph and the k-means
embedding, which flips cluster labels and the noisy-cluster pick: baseline gross
Sharpe moved over roughly 2.96-3.23 across identical runs of identical code, and
the checked-in results could not be reproduced from a fresh process.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stat_arb.data.loader import DataLoader  # noqa: E402

# Deliberately not in sorted order, and chosen so that set iteration order is
# unlikely to coincide with sorted order by luck.
TOKENS = ["ZRX", "AAVE", "MKR", "BAT", "SNX", "COMP", "YFI", "UNI"]


def _fixture_dir(tmp_path: Path) -> Path:
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    rng = np.random.default_rng(0)

    raw = pd.DataFrame(index=idx)
    for i, tok in enumerate(TOKENS):
        raw[f"{tok}_close"] = 10.0 + i + rng.normal(0, 0.1, len(idx)).cumsum()
        raw[f"{tok}_volume"] = 1e6 * (i + 1)
    raw.index.name = "date"
    raw.to_csv(tmp_path / "all_tokens_24mo_daily.csv")

    excess = pd.DataFrame(
        {f"{tok}_returns": rng.normal(0, 0.01, len(idx)) for tok in TOKENS}, index=idx)
    excess.index.name = "date"
    excess.to_csv(tmp_path / "excess_log_returns.csv")

    eth = pd.DataFrame({"close": 2000.0, "volume": 8e9}, index=idx)
    eth.index.name = "date"
    eth.to_csv(tmp_path / "eth_ohlcv.csv")
    return tmp_path


def test_aligned_columns_are_sorted_not_set_ordered(tmp_path):
    loader = DataLoader(str(_fixture_dir(tmp_path)))
    excess, prices, volumes, _ = loader.get_aligned_data()

    assert list(prices.columns) == sorted(TOKENS)
    assert list(volumes.columns) == sorted(TOKENS)
    assert list(excess.columns) == [f"{t}_returns" for t in sorted(TOKENS)]


def test_column_order_survives_a_different_process_hash_seed(tmp_path):
    """The regression itself: same data, different PYTHONHASHSEED, same order.

    Run in subprocesses because PYTHONHASHSEED is fixed at interpreter start and
    cannot be changed from inside the running process.
    """
    data_dir = _fixture_dir(tmp_path)
    root = Path(__file__).resolve().parent.parent
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from stat_arb.data.loader import DataLoader\n"
        "_, prices, _, _ = DataLoader(%r).get_aligned_data()\n"
        "print(','.join(prices.columns))\n" % (str(root), str(data_dir))
    )

    orders = set()
    for seed in ("0", "1", "12345"):
        env = {"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env, check=True)
        orders.add(out.stdout.strip())

    assert len(orders) == 1, f"column order varies with the hash seed: {orders}"
    assert orders.pop() == ",".join(sorted(TOKENS))
