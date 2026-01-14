# Crypto Stat-Arb (Signed-Graph Clustering) — Research + Backtest

Market-neutral crypto statistical arbitrage research project and backtesting pipeline. The core approach builds a **signed k-NN correlation graph** on token returns (after **PCA market-mode removal**), clusters the graph (SPONGE / BNC / Signed Spectral), and trades **mean-reversion signals within clusters** using a **walk-forward, no-lookahead** backtest with **transaction-cost sensitivity**.

## Highlights
- Signed correlation graph construction (k-NN on absolute correlations, signed edges)
- Signed-graph clustering: **SPONGE**, **BNC**, **Signed Spectral**
- **PCA residualization** to remove market mode (PC1) before clustering/signals
- Dynamic universe selection (volume + history filters, monthly reconstitution)
- Walk-forward OOS backtesting with turnover control + transaction cost model
- Reporting outputs saved as CSV/PNG under `stat_arb/reporting/`

## Repo Structure
- `stat_arb/`: main Python package
  - `run_phase1.py`: baseline backtest (SPONGE k=3 + z-score mean reversion)
  - `run_phase2.py`: clustering sweep (SPONGE/BNC/SignedSpectral across k) + Strategy 2 test
  - `data/`: CSV loaders, cleaning utilities, universe membership logic
  - `graphs/`: k-NN correlation graph builder
  - `clustering/`: signed clustering implementations + k-selection utilities
  - `pca/`: PCA market-mode extraction/residualization
  - `signals/`: z-score strategy, cluster deviation strategy, optional pairs trading
  - `backtest/`: walk-forward runner, P&L engine, transaction cost model
  - `reporting/`: generated results + final write-up (`FINAL_REPORT.md`)
- `data/`: project datasets (prices/volumes, ETH OHLCV, excess returns, correlations)
- `pics/`: EDA/diagnostic figures
- `crypto_project.ipynb`: exploratory notebook

## Data
Expected CSVs in `data/`:
- `all_tokens_24mo_daily.csv` (prices/volumes per token)
- `eth_ohlcv.csv` (ETH OHLCV)
- `excess_log_returns.csv` (token returns vs ETH)
- `token_summary.csv`, `corr_full_cleaned.csv` (optional helpers)

The loader excludes stablecoins and wrapped assets by default.

## Quickstart
Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib statsmodels
```

Run Phase 1 (baseline):

```bash
python stat_arb/run_phase1.py
```

Outputs:
- `stat_arb/reporting/phase1/` (weights, returns, turnover, fold info, plots, summary CSVs)

Run Phase 2 (method sweep + strategy comparison):

```bash
python stat_arb/run_phase2.py
```

Outputs:
- `stat_arb/reporting/phase2/clustering_sweep_results.csv`
- `stat_arb/reporting/phase2/leaderboard.csv`

## Methodology (High Level)
1. Load and align returns/prices/volumes + ETH data.
2. Build a tradable universe mask (volume + history filters, monthly reconstitution).
3. Fit PCA on the training window; residualize returns (remove market mode).
4. Build a signed k-NN correlation graph from residualized returns.
5. Cluster the signed graph (SPONGE / BNC / Signed Spectral).
6. Generate cluster-aware mean-reversion signals (lagged to avoid lookahead).
7. Construct target weights with leverage normalization and turnover control.
8. Walk-forward backtest and apply transaction cost model; save metrics and reports.

## Results
See the full write-up and leaderboard in:
- `stat_arb/reporting/FINAL_REPORT.md`

## Notes / Disclaimer
This is a research backtest and not financial advice. Results depend heavily on data quality, slippage/execution assumptions, and the transaction cost model. Real-world performance may differ materially.

## License
Add a `LICENSE` file (MIT or Apache-2.0 are common choices for open-source).

>>>>>>> 4485fbf (initial commit)
