# Crypto Stat-Arb

This repository studies market-neutral crypto statistical arbitrage with signed-graph clustering and walk-forward backtesting. It builds a residualized correlation graph after removing the market mode, clusters the graph with signed methods such as SPONGE and BNC, and trades cluster-level mean-reversion signals under explicit turnover and transaction-cost controls.

## Repository layout

- `stat_arb/`: main research package for data loading, graph construction, clustering, signals, backtests, and reporting
- `data/`: processed market, volume, ETH, and correlation datasets used by the backtests
- `pics/`: diagnostic figures for clustering quality and exploratory analysis
- `crypto_project.ipynb`: exploratory notebook used during early research
- `archived_research/`: older exploratory artifacts retained for reference
- `Crypto_Project_Report_Pre_Backtest.pdf`: written report from the earlier research stage

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib statsmodels pytest
```

Run the test suite:

```bash
python -m pytest tests -q
```

Run the baseline SPONGE backtest:

```bash
python stat_arb/run_phase1.py
```

Run the clustering-method sweep:

```bash
python stat_arb/run_phase2.py
```

Run the cost-aware execution experiments (no-trade band x rebalance frequency):

```bash
python stat_arb/run_phase3.py
```

If you want to rerun the notebook cells that call CoinMarketCap, export your credential first:

```bash
export CMC_API_KEY=your_coinmarketcap_key
```

## Methodology

The pipeline first aligns token prices, volumes, and ETH reference data, then builds a tradable universe subject to history and liquidity filters. Returns are residualized against the market mode with PCA, transformed into a signed k-nearest-neighbor correlation graph, and clustered with SPONGE, BNC, or signed spectral methods. Signals are generated from within-cluster mean reversion, normalized to target leverage, and evaluated in a walk-forward backtest with lagging, turnover controls, and transaction-cost assumptions to limit lookahead and overstatement.

## Results

Primary outputs are written under `stat_arb/reporting/` and include fold-level returns, turnover series, clustering sweep summaries, leaderboards, and the final report. The intended use is comparative research across clustering methods rather than a production-ready live trading engine.

Reported Sharpe ratios are accompanied by the Probabilistic Sharpe Ratio and the Deflated Sharpe Ratio (Bailey and Lopez de Prado), with each sweep treated as its own multiple-testing pool. The finding comes in two honest halves. Under daily rebalancing (phases 1-2), gross Sharpe is positive across all 16 configurations but nothing survives realistic taker costs, and the best net configuration's DSR (0.09) is indistinguishable from noise. The phase-3 execution experiments then show the alpha decays over multi-day horizons: trading every third day with a 2% no-trade band keeps ~97% of the gross Sharpe at about a third of the turnover, lifting net Sharpe at 50 bps from 1.0 to 2.3, break-even cost from 75 to 225 bps, with DSR 0.955 against the full 12-cell experiment grid. The survivorship-biased universe remains the binding caveat before any live claim.

### 2026-08 revision

Results were regenerated after a signal-integrity pass. The material fixes, each covered by a regression test in `tests/`:

- inverse-volatility position sizing is now lagged one day (it previously used same-day volatility)
- cluster and dollar neutralization now apply only to selected names (they previously smeared small offsetting weights onto every token, inflating turnover)
- PCA market-mode fit no longer drops every row containing a single NaN on ragged histories
- k-NN graph construction is vectorized and compatible with pandas copy-on-write (the previous code crashed on pandas 3.x, so checked-in results could not be reproduced from a fresh environment)
- a financing-carry stress knob (`BacktestEngine.carry_bps_daily`) approximates perp funding / borrow drag on gross exposure

Headline numbers changed with these fixes (previous figures overstated gross Sharpe). See `stat_arb/reporting/FINAL_REPORT.md` for the regenerated results.

## Known limits

- Results are sensitive to crypto data quality, survivorship, and execution assumptions
- The token universe comes from a CoinMarketCap top-token snapshot, so delisted/dead tokens are absent; this survivorship bias flatters mean-reversion results and a point-in-time listing history would be needed to remove it
- Short legs are modeled as costless to hold; in practice they are perpetual futures with per-token, time-varying funding. The `carry_bps_daily` stress knob bounds this effect but is not a funding model
- The checked-in notebook and archived artifacts reflect exploratory work and are less polished than the package backtest path
- Transaction costs and liquidity in crypto can change quickly enough to invalidate static assumptions

## License

This project is distributed under the MIT License
