# Crypto Stat-Arb

Market-neutral cryptocurrency statistical arbitrage: signed-graph clustering over a market-mode-residualized correlation graph, with walk-forward backtesting, explicit transaction-cost modelling, and multiple-testing-aware validation.

## What it does

- **Removes the market mode by PCA**, then builds a signed k-nearest-neighbour correlation graph on the residuals, so clusters reflect relative rather than common movement
- **Clusters with SPONGE, BNC and signed spectral methods**, compared like-for-like across a 16-configuration sweep rather than picking one and defending it
- **Trades cluster mean reversion** under a daily turnover cap, a no-trade band, and a rebalance-frequency control
- **Validates with Probabilistic and Deflated Sharpe Ratios**, treating each sweep as its own multiple-testing pool, plus a financing-carry stress for perpetual funding

## Headline result: the alpha is an illiquidity effect

The best configuration reaches a net Sharpe of **2.30** after 50bps of taker cost. Rather than report that number, this repository locates where the edge actually lives — and the answer changes the conclusion.

The universe is a CoinMarketCap snapshot, so tokens that died are absent. Ammann, Burdorf, Liebi and Stöckl ([SSRN 4287573](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4287573), 3,904 coins, 2014–2021) measure the resulting survivorship bias at 0.93% annualised value-weighted and **62.19% equal-weighted**, and find momentum and market beta lose any positive relation to returns once delisting returns are included. This book is near-equal-weighted, the regime where that bias is largest.

So `stat_arb/run_robustness.py` raises the minimum-volume floor — applied through `UniverseManager`, which filters point-in-time at each reconstitution, so no forward-looking information enters:

| universe | avg members | gross Sharpe | net Sharpe @50bps | breakeven cost |
|---|---|---|---|---|
| ≥ $50k/day (baseline) | 134 | 2.96 | **2.30** | 226 bps |
| ≥ $1M/day | 89 | 0.74 | **0.04** | 53 bps |
| ≥ $5M/day | 57 | 1.21 | **0.66** | 110 bps |

**Raising the floor to $1M/day collapses net Sharpe from 2.30 to 0.04.** The edge is concentrated in the illiquid tail — simultaneously where delisting risk is highest, so where the missing-token bias bites hardest, and where a flat 50bps cost is least defensible. Two independent reasons to discount the headline, pointing the same way.

A second check reports the **breakeven bias**: the annualised drag that would take net Sharpe to zero. It is 34.2% at baseline and 0.6% in the liquid tier. The published 62.19% figure is a long-only upper bound rather than an estimate for a dollar-neutral book, since such a book loses both potential longs and potential shorts when a token dies — but the liquid-tier figure sitting below even the 0.93% value-weighted number is telling.

Tier results are non-monotone (0.04 at $1M, 0.66 at $5M) and should be read as noisy at 57–89 names, not as a liquidity threshold effect.

Settling this properly needs point-in-time listing history including dead tokens, which is a paid dataset and has not been purchased. The honest status: **the clustering and execution methodology is the contribution; the Sharpe is not.**

## What the repository does establish

- Signed-graph clustering (SPONGE, BNC, signed spectral) on a market-mode-residualized correlation graph, compared like-for-like across methods
- That cluster mean-reversion alpha decays over multi-day horizons: trading every third day with a 2% no-trade band retains ~97% of gross Sharpe at a third of the turnover, which is Gârleanu–Pedersen "aim in front of the target" showing up empirically
- Multiple-testing discipline throughout: Probabilistic and Deflated Sharpe Ratios, with each sweep treated as its own trial pool

Run the robustness checks:

```bash
python stat_arb/run_robustness.py
```

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
