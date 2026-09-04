# Stat-Arb Backtest: Final Report

## 2026-08 Revision (supersedes the figures below)

All results were regenerated after a signal-integrity pass. Fixes, each with a
regression test in `tests/`:

1. **Lagged inverse-vol sizing.** Position sizing previously used same-day volatility; it is now lagged one day like the signal.
2. **No weight smearing.** Cluster/dollar neutralization previously assigned small offsetting weights to every token, including names the entry rule never selected; neutralization now applies only to traded names.
3. **PCA fit on ragged panels.** The market-mode fit no longer drops every row containing a single NaN.
4. **Fresh-environment reproducibility.** k-NN graph construction crashed under pandas 3.x copy-on-write, so the previous checked-in numbers could not be reproduced from a clean clone. Fixed and vectorized.
5. **Deflated Sharpe Ratio.** The leaderboard now reports the Bailey/Lopez de Prado DSR with the sweep itself as the multiple-testing pool, plus a financing-carry stress (perp funding / borrow proxy).

### Regenerated headline results (walk-forward OOS, 2024-05-30 to 2025-05-28)

Phase 1 baseline (SPONGE k=3, z-score MR, H=5, L=60):
- Gross Sharpe **1.67**, net Sharpe @ 50 bps **-2.16**, break-even cost **21.0 bps**
- PSR 0.02, DSR 0.00, avg daily turnover 23%, max drawdown -27.7%

Phase 2 sweep (16 configurations):
- Best gross: SPONGE k=6 z-score MR, gross Sharpe **4.22**, net @ 50 bps **-1.21**, DSR 0.00
- Best net: Cluster Deviation SPONGE k=3, gross **3.10**, net @ 50 bps **0.01**, break-even **50.3 bps**, DSR **0.09**

### Honest conclusion (phases 1-2)

The clustering signal carries real gross structure (positive gross Sharpe in
all 16 configurations), but under DAILY rebalancing no configuration survives
realistic crypto taker costs plus financing carry, and the deflated Sharpe of
the best net configuration (0.09) says the selected result is
indistinguishable from the best of 16 noise strategies. The prior version of
this report overstated gross Sharpe (2.7-3.3 in the headline configs) due to
fixes 1-2.

## Phase 3: Cost-aware execution experiments (2026-08)

Phases 1-2 leave the natural question: is the alpha genuinely too small, or
is it being spent on trading costs faster than it accrues? Phase 3 sweeps two
execution controls on the best net configuration (Cluster Deviation, SPONGE
k=3): a no-trade band (ignore sub-band target changes) and a rebalance
frequency (trade every k-th day, hold in between). Full grid in
`stat_arb/reporting/phase3/execution_experiments.csv`.

| band | freq | gross SR | net SR @25 | net SR @50 | turnover/day | break-even | DSR (12-trial pool) |
|---|---|---|---|---|---|---|---|
| 2% | 3d | 2.96 | 2.63 | **2.30** | 5.3% | **225 bps** | **0.955** |
| 0 | 3d | 2.99 | 2.44 | 1.88 | 5.3% | 135 bps | 0.904 |
| 0 | 1d (baseline) | 3.06 | 2.03 | 1.00 | 15.0% | 75 bps | 0.639 |

Finding: the cluster mean-reversion alpha decays over multi-day horizons, so
harvesting it every third day with a small no-trade band keeps ~97% of the
gross Sharpe while cutting turnover 2.8x. Net Sharpe at a 50 bps taker-cost
assumption rises from 1.00 to 2.30 and the break-even cost from 75 to 225
bps, and the Deflated Sharpe Ratio against the full 12-cell grid is 0.955,
i.e. the selected cell clears the expected maximum of 12 zero-skill trials.

Caveats, stated plainly:
- the universe remains survivorship-biased (CMC snapshot), which flatters
  mean reversion; this finding needs to survive a point-in-time universe
- one ~12-month OOS window; the DSR corrects for grid selection, not for a
  short sample
- phase 3 uses a cleaner execution layer than phase 2 (the daily turnover
  cap engages from the first day, so positions ramp in); compare cells
  within the grid, not against the phase-2 table
- financing carry is still a separate stress (see phase 1), not embedded

---

## Original report (historical, pre-revision numbers)

## Executive Summary

Executed low-cap crypto stat-arb using signed k-NN correlation graph, SPONGE/BNC/signed spectral clustering, and PCA market-mode removal. Daily walk-forward OOS from May 2024 to May 2025.

### Best Configuration

**Strategy 2 (Cluster Deviation)** with SPONGE k=3:
- **Gross Sharpe: 3.27**
- **Break-even cost: 54.2 bps** (highest among all configurations)
- **Net Sharpe @ 25 bps: 1.76** | **Ann. Return: 29.2%**
- **Net Sharpe @ 50 bps: 0.24** | **Ann. Return: 4.1%** (only positive net Sharpe @ 50bps)
- **ETH Beta: ~0.02** | **PC1 Beta: ~0.00** (market-neutral)

---

## Phase 0: Audit Summary

### Existing Implementation
- Data loading with stablecoin/wrapped asset exclusion
- Universe management (mcap/volume filters, monthly reconstitution)
- k-NN graph construction from correlation matrices
- SPONGE, BNC, Signed Spectral clustering implementations
- PCA market mode extraction and residualization
- Z-score mean reversion strategy
- Cluster deviation strategy
- Walk-forward backtest framework
- Transaction cost model with sensitivity analysis

### Critical Bugs Fixed
1. Undefined `cluster_labels_array` and `assets_list` variables
2. Column name mismatch (tokens vs `_returns` suffix)
3. Missing leverage normalization (gross exposure was 4x instead of 1.5x)
4. Excessive turnover (341% daily reduced to ~23%)

---

## Phase 1: SPONGE k=3 Baseline

### Configuration
- **Clustering**: SPONGE k=3, drop noisy cluster (lowest within-cluster cohesion)
- **Signal**: z-score mean reversion (H=5, L=20)
- **Walk-forward**: 365-day train, 28-day test, 28-day refit
- **Leverage**: 1.5x gross, turnover capped at 15%/day

### Results (@ 50 bps)

| Metric | Gross | Net |
|--------|-------|-----|
| Sharpe Ratio | 2.69 | -2.33 |
| Total Return | 22.01% | -20.36% |
| Ann. Volatility | - | 8.77% |
| Max Drawdown | - | -23.92% |
| Avg Turnover | - | 23.28% |
| Cost Drag | - | 42.37% |

### Cost Sensitivity

| Cost (bps) | Sharpe | Ann. Return |
|------------|--------|-------------|
| 5 | 2.25 | 18.44% |
| 10 | 1.73 | 14.20% |
| 15 | 1.20 | 9.95% |
| 20 | 0.69 | 5.70% |
| 25 | 0.17 | 1.45% |
| **Break-even** | **26.9 bps** | - |

### H/L Parameter Sensitivity

| H | L | Gross Sharpe | Best at 50bps |
|---|---|--------------|---------------|
| 1 | 20 | 1.90 | No |
| 5 | 20 | **2.64** | No |
| 10 | 20 | 1.27 | No |
| 20 | 120 | 0.62 | -1.38 (best) |

### Risk Analysis
- **ETH Beta**: 0.0181 (near-zero)
- **PC1 Beta**: -0.0000 (market-neutral)

---

## Phase 2: Alternative Methods & Strategies

### Clustering Method Comparison

**Corrected 2026-09-04.** The BNC implementation this table was produced with
was solving `eigh(A+ - A-, D_tot)` for its smallest eigenvalues. Balance
Normalized Cut (Chiang, Whang and Dhillon, CIKM 2012) is
`eigh(D+ - A+ + A-, D+ + D-)`. Omitting the `D+` term shifts the spectrum, so
the old code selected from the opposite end of it: on planted signed blocks
that SPONGE recovers exactly (adjusted Rand index 1.000), it scored -0.02,
which is worse than assigning labels at random. Every BNC row below is
therefore a re-run, and the previously published leader (BNC k=5, gross 4.46,
net -0.18, break-even 48.3 bps) does not survive.

The same audit found `SPONGEClustering.fit_symmetric` was a similarity
transform of the pencil `fit` already solves, returning identical labels and
eigenvalues, so a "SPONGEsym" column would have been plain SPONGE run twice. It
now implements the symmetrically normalized form of Cucuringu et al.
(AISTATS 2019).

| Rank | Method | k | Gross SR | Net SR @50bps | Break-even |
|------|--------|---|----------|---------------|------------|
| 1 | BNC | 6 | 4.21 | -1.34 | 37.6 bps |
| 2 | SPONGE | 6 | 3.71 | -1.88 | 32.7 bps |
| 3 | BNC | 5 | 3.49 | -1.72 | 33.7 bps |
| 4 | BNC | 3 | 3.38 | -1.68 | 32.7 bps |
| 5 | SignedSpectral | 6 | 3.29 | -2.40 | 28.8 bps |
| 6 | SPONGE | 3 (Cluster Dev) | 3.25 | **+0.15** | **52.2 bps** |

The conclusion the sweep is used for does not change, and is if anything
sharper: of all sixteen configurations, exactly one has a positive net Sharpe
after 50 bps, and it is the same one as before, SPONGE k=3 with the Cluster
Deviation signal. Every other configuration in the sweep, the highest-gross one
included, loses money at realistic costs. High gross Sharpe in this book is a
turnover artifact, not an edge.

### Strategy 2: Cluster Deviation

**Best overall configuration for practical implementation:**

| Cost (bps) | Sharpe | Ann. Return |
|------------|--------|-------------|
| 10 | 2.67 | 44.2% |
| 20 | 2.06 | 34.2% |
| 25 | **1.76** | **29.2%** |
| 30 | 1.45 | 24.2% |
| 40 | 0.84 | 14.1% |
| 50 | **0.24** | **4.1%** |
| **Break-even** | **54.2 bps** | - |

---

## Key Findings

1. **All clustering methods produce positive gross alpha** (Sharpe 2-4+)
2. **BNC k=6 has the highest gross Sharpe (4.21)** but a net Sharpe of -1.34.
   Only one configuration in the sweep is net positive at 50 bps
3. **Strategy 2 (Cluster Deviation) is most robust to costs** with 54.2 bps break-even
4. **At realistic execution costs (25-30 bps), Strategy 2 achieves Sharpe 1.45-1.76**
5. **Strategy is fully market-neutral** (ETH beta ~0.02, PC1 beta ~0.00)

---

## Robustness Analysis

- Performance consistent across 13 walk-forward folds
- Break-even costs range from 23-54 bps depending on configuration
- Strategy 2's higher break-even provides implementation buffer
- Turnover ~23-28% daily is manageable for liquid low-cap tokens

---

## Resume-Ready Bullet

> **Executed low-cap crypto stat-arb using signed k-NN correlation graph, SPONGE/BNC/signed spectral clustering, and PCA market-mode removal; daily walk-forward OOS May 2024 - May 2025. Best configuration (Cluster Deviation strategy, SPONGE k=3) achieved 29.2% CAGR, 1.76 Sharpe, -23.9% max drawdown, and 27.5% avg daily turnover net of 25 bps/side costs with ETH/PC1 beta ~0.02/0.00.**

---

## Files Generated

### Phase 1
- `stat_arb/reporting/phase1/phase1_full_report.png`
- `stat_arb/reporting/phase1/phase1_summary.csv`
- `stat_arb/reporting/phase1/diagnostics.csv`
- `stat_arb/reporting/phase1/net_returns.csv`
- `stat_arb/reporting/phase1/weights.csv`

### Phase 2
- `stat_arb/reporting/phase2/clustering_sweep_results.csv`
- `stat_arb/reporting/phase2/leaderboard.csv`

---

## Technical Implementation

### Data
- **Period**: May 2023 - May 2025 (729 trading days)
- **Universe**: 174 tokens (excl. stablecoins, wrapped assets)
- **Avg tradable**: 134 tokens after filters

### No-Lookahead Protocol
- Signals computed using data through t-1 only
- Clustering and PCA fit on training window only
- Walk-forward with 365-day train, 28-day test

### Transaction Cost Model
- BPS per side on turnover
- Sensitivity grid: 5-200 bps
- Break-even computed via binary search

---

*Report generated: 2026-01-13*
