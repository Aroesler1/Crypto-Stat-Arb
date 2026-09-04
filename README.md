# Crypto Stat-Arb

Market-neutral cryptocurrency statistical arbitrage: signed-graph clustering over a market-mode-residualized correlation graph, with walk-forward backtesting, explicit transaction-cost modelling, and multiple-testing-aware validation.

## What it does

- **Removes the market mode by PCA**, then builds a signed k-nearest-neighbour correlation graph on the residuals, so clusters reflect relative rather than common movement
- **Clusters with SPONGE, BNC and signed spectral methods**, compared like-for-like across a 16-configuration sweep rather than picking one and defending it
- **Trades cluster mean reversion** under a daily turnover cap, a no-trade band, and a rebalance-frequency control
- **Validates with Probabilistic and Deflated Sharpe Ratios**, treating each sweep as its own multiple-testing pool, plus a financing-carry stress for perpetual funding
- **Rebuilds its own universe point-in-time from CoinMarketCap, including tokens that died**, and measures what their absence was worth

## The hypothesis, and where it survives

The pre-backtest report defined a small cap not in dollars but as a band of
Ethereum's market cap: tokens between 0.001% and 0.1% of ETH, sampled every 30
days, with every return taken in excess of ETH. The question was whether that
universe forms clusters of co-moving tokens that can be traded market-neutrally
by within-cluster mean reversion.

Defining size relative to ETH is what makes the question answerable over a
decade. Crypto market caps move two orders of magnitude between 2016 and 2025,
so a fixed dollar band is a different slice of the cross-section every year and
a rank band is a different slice whenever the number of listed tokens changes.
A ratio to ETH means the same thing in 2017 and 2025 even though the dollar
amount moves 20x.

So the question is widened from one band to the whole cap spectrum, in log
decades of `r = market cap / ETH market cap`, assigned at each month end:

| bracket | definition | avg members | months clusterable | tokens | dollar range (2019 to 2025) |
|---|---|---|---|---|---|
| B0 mega | r >= 10% | 4.6 | 0 of 114 | 25 | $1.9B+ to $28.5B+ |
| B1 large | 1% <= r < 10% | 23.9 | 20 of 114 | 204 | $191M-$1.9B to $2.8B-$28.5B |
| B2 mid | 0.1% <= r < 1% | 122.8 | 114 of 114 | 1,109 | $19M-$191M to $285M-$2.8B |
| B3 small | 0.01% <= r < 0.1% | 387.6 | 114 of 114 | 3,210 | $1.9M-$19M to $28.5M-$285M |

B3 is the upper half of the report's original 0.001% to 0.1% band. The lower
decade is dropped by decision, not for lack of data: the pull reaches rank 2000
and those tokens are present, but below 0.01% of ETH the panel is dominated by
names whose quoted price is a rounding artifact. ETH is the reference and is
never traded. Stablecoins, wrapped, staked and bridged tokens are excluded
everywhere (316 cmc_ids). Every bracket starts 2016-01 because ETH itself only
lists in 2015-08, and no ETH-relative bracket can be defined before its
reference exists.

Two structural facts fall out of the definition before any strategy is run.
**B0 never reaches 30 members**, so it cannot be clustered at all and is handled
separately by a pairs book and a cross-sectional z-score, residualized against
BTC and the value-weighted market rather than ETH. **B1 clears 30 members in
only 20 of 114 month-ends**, so for most of the sample the large-cap bracket is
too thin to cluster. That is a result about the shape of the crypto
cross-section, not a defect in the data.

### The verdict so far

Measured on the point-in-time universe with the signal held fixed
(2% no-trade band, rebalance every 3 days, net of 50 bps, $50k/day floor):

| bracket | best net Sharpe | best reference | breakeven | shortable share (2025) | verdict |
|---|---|---|---|---|---|
| B0 mega | not yet measured | BTC / value-weighted | - | 100% | shortable, too few names to cluster |
| B1 large | **+0.40** | ETH-excess, no PCA | 98 bps | 87% | the only bracket that is net positive and shortable, but clusterable in 20 of 114 months |
| B2 mid | -0.24 | value-weighted | 0 bps | 76% | clusterable and mostly shortable, no edge |
| B3 small | -0.42 | ETH-excess, no PCA | 0 bps | 48% | the original band: always clusterable, never shortable, no edge |

The expected shape holds, and it is a squeeze. Dispersion worth clustering lives
where the names cannot be shorted; the names that can be shorted are too few to
cluster. B1 is the one bracket where a positive net Sharpe and a tradable short
leg coexist, and it is thin for four fifths of the sample.

**The clustering and execution methodology is the contribution. The Sharpe is
survivorship, and the one bracket that survives is too thin to hold it.**

## What "the market" is differs by bracket, and PCA is not it

The original report residualized against ETH. The rebuild residualized against a
PCA market mode. Neither was ever tested against the other, so both are run as
an explicit ablation with the signal held fixed. Point-in-time universe, all
three clustered brackets:

| bracket | arm | var removed | graph density | negative edges | cluster stability | gross | net @50bps |
|---|---|---|---|---|---|---|---|
| B1 | **ETH-excess** | 0% | 65.4% | 1% | **0.545** | 0.81 | **+0.40** |
| B1 | BTC-excess | 0% | 68.2% | 3% | 0.527 | -0.34 | -0.86 |
| B1 | value-weighted | 0% | 66.8% | 5% | 0.511 | -0.23 | -0.74 |
| B1 | ETH + PCA1 | 28.3% | 59.6% | 79% | 0.405 | -0.07 | -0.49 |
| B1 | ETH + PCA2 | 37.3% | 57.9% | 77% | 0.280 | 0.21 | -0.27 |
| B1 | ETH + PCA3 | 43.9% | 57.4% | 75% | 0.217 | -0.34 | -0.84 |
| B2 | ETH-excess | 0% | 24.2% | 1% | 0.522 | -0.51 | -0.75 |
| B2 | **value-weighted** | 0% | 23.9% | 4% | 0.476 | -0.03 | **-0.24** |
| B2 | ETH + PCA1 | 23.1% | 20.8% | 58% | 0.410 | -0.41 | -0.59 |
| B2 | ETH + PCA3 | 31.5% | 19.7% | 61% | 0.175 | -0.19 | -0.40 |
| B3 | **ETH-excess** | 0% | 9.5% | 1% | **0.637** | -0.28 | **-0.42** |
| B3 | BTC-excess | 0% | 9.3% | 2% | 0.598 | -0.87 | -1.00 |
| B3 | ETH + PCA1 | 17.2% | 7.9% | 43% | 0.503 | -0.57 | -0.70 |
| B3 | ETH + PCA3 | 23.7% | 7.1% | 48% | 0.141 | -0.46 | -0.59 |

Cluster stability is the adjusted Rand index between consecutive monthly
clusterings, on the tokens common to both. Density is only comparable within a
bracket, because a k=10 neighbour graph over 21 nodes is necessarily denser than
one over 261. Full table: `stat_arb/reporting/brackets/`.

Three things come out of it.

**PCA creates the signed structure and then destroys the partition.** With no
PCA step only 1% to 5% of edges are negative: everything co-moves with the
market, so there is barely a signed graph to cluster. Removing one principal
component takes the negative-edge share to 43-79%. But cluster stability falls
monotonically with every component removed, in every bracket: B1 0.545 to 0.217,
B2 0.522 to 0.175, B3 0.637 to 0.141. By three components the partition is
reshuffling almost completely every month, and a cluster assignment that does
not survive to the next reconstitution cannot be traded.

**More variance removed never bought a better net Sharpe.** In all three
brackets the best arm removes zero principal components. The rebuild's choice to
residualize against a PCA market mode is worse on this panel than the original
report's plain ETH-excess.

**The best reference does differ by bracket**, which is the question the
ablation was run to answer: ETH-excess for B1 and B3, the value-weighted market
for B2. The B2 and B3 differences are between negative numbers, so they rank
least-bad rather than best.

## Survivorship is a small-cap phenomenon, and it scales down the spectrum

Running the same ablation on the survivor-only universe (the same bracket, the
same signal, minus the tokens that are dead today) turns the headline finding
into a gradient. Best net Sharpe at 50 bps in each bracket, under each
treatment:

| bracket | point-in-time | survivor-only | survivorship worth | deaths in bracket |
|---|---|---|---|---|
| B1 large | **+0.40** (ETH) | -0.07 (value-weighted) | -0.47 | 9 over 10 years |
| B2 mid | -0.24 (value-weighted) | +0.35 (BTC) | **+0.59** | 60 |
| B3 small | -0.42 (ETH) | **+0.55** (ETH+PCA3) | **+0.97** | 291 |

The further down the cap spectrum, the more of the apparent edge is dead
tokens. B3, the report's original band, goes from a tradable-looking +0.55 with
a 205 bps breakeven on survivor data to -0.42 once the tokens that died are
allowed back in. B2 shows the same sign flip at roughly two thirds the
magnitude. B1 does not show it at all, and in fact runs slightly better
point-in-time, which is what you would expect from a bracket that buried nine
tokens in ten years: there is almost no survivorship there to remove, and the
difference is noise.

**The PCA residualization's apparent benefit is itself a survivorship
artifact.** On survivor-only B3 the two best arms in the whole table are
ETH+PCA3 (+0.55) and ETH+PCA1 (+0.51), comfortably ahead of plain ETH-excess
(-0.20). On the same bracket point-in-time, every PCA arm is worse than plain
ETH-excess. Removing principal components makes the book look better only on a
universe that has had its losers deleted, which is the single most important
reason not to select a residualization on survivor data.

## Headline result: the alpha was survivorship

The previous version of this README reported a net Sharpe of 2.30 after 50bps and argued the edge was an illiquidity effect. It also said that settling the survivorship question "needs point-in-time listing history including dead tokens, which is a paid dataset and has not been purchased."

That was wrong, and it was the load-bearing claim. CoinMarketCap serves point-in-time rank snapshots and full daily OHLCV for delisted coins from its public web API, with no key and no paid plan: the endpoints the [crypto2](https://github.com/sstoeckl/crypto2) R package (Stöckl, CRAN) wraps. `stat_arb/build_pit_universe.py` pulls them in one command.

Doing it changes the answer. Same window, same rank band, same filters, same engine, three universes that differ only in which tokens they are allowed to see:

| universe | avg members | gross Sharpe | net Sharpe @50bps | breakeven cost |
|---|---|---|---|---|
| snapshot (survivors, complete history) | 224 | 2.40 | **2.02** | 314 bps |
| survivor-only (survivors, any history) | 275 | 1.73 | **1.37** | 240 bps |
| point-in-time (incl. 50 tokens that died) | 284 | 0.19 | **-0.14** | 29 bps |

*Rank band 150-500, monthly reconstitution, at least $50k/day notional, 2% no-trade band, rebalance every 3 days. `python stat_arb/run_pit_robustness.py`.*

**Net Sharpe 2.02 becomes -0.14 when the universe is allowed to contain tokens that died.** The first row reproduces this repository's own construction on rebuilt data and lands near the published 2.30, which is what makes the comparison a like-for-like. The strategy has no edge left once survivorship is removed.

The same collapse holds at every liquidity tier, so the old "the edge lives in the illiquid tail" reading does not survive either:

| tier | snapshot | survivor-only | point-in-time |
|---|---|---|---|
| at least  $50k/day | 2.02 | 1.37 | **-0.14** |
| at least  $1M/day | 1.85 | 1.84 | **0.41** |
| at least  $5M/day | 1.53 | 0.51 | **-2.03** |

The tier ordering is not the story; the column is.

### Why the bias is so much larger than a long-only bound

Measured in the strategy's own net returns, paired by date, the annualised drag from including dead tokens is **+40.9% ± 16.4%** at baseline, **+29.5% ± 14.5%** in the $1M tier and **+65.7% ± 22.0%** in the $5M tier (± 1 standard error).

The previous README argued that Ammann, Burdorf, Liebi and Stöckl's 62.19% equal-weighted figure ([SSRN 4287573](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4287573), 3,904 coins, 2014 to 2021) was a **long-only upper bound**, on the reasoning that a dollar-neutral book loses both potential longs and potential shorts when a token dies. The data says otherwise: the realised drag here sits at that bound rather than well below it.

The reasoning was wrong because it ignored what the strategy does. This is a mean-reversion book, so it systematically buys losers. A token on its way to zero is the most attractive thing a mean-reversion signal can see, and it never reverts. Removing dead tokens from the universe removes precisely the trades that would have blown up the long leg. Survivorship is not a symmetric haircut on a market-neutral book; it is a selective deletion of its worst trades.

### The universe-level bias is small; the strategy-level bias is not

Run as a monthly-reconstituted buy-and-hold portfolio (the construction the published figures actually measure) the same universe gives:

| book | equal-weighted | value-weighted |
|---|---|---|
| point-in-time | +8.24% | +4.19% |
| survivor-only | +12.79% | +6.30% |
| **survivorship bias** | **+4.56%** | **+2.11%** |
| Ammann et al. published | +62.19% | +0.93% |

At the universe level the bias is modest: **+4.56% equal-weighted**, far below the published 62.19%, because a rank 150-500 band over 24 months contains far less death (9.4% of members) than 3,904 coins over seven years. The value-weighted +2.11% is the same order as the published 0.93%.

That contrast is the point. A survivorship bias worth 4.6% a year to a buy-and-hold investor is worth 30 to 66% a year to this strategy. Quoting the universe-level number as if it bounded the strategy-level one, which is what the old README did, understates the problem by an order of magnitude.

### Two reproducibility defects found on the way, both fixed

Neither is cosmetic and both affected published numbers.

**The results were not reproducible from a fresh process.** `DataLoader.get_aligned_data` derived the panel's column order by iterating a Python `set`, whose order depends on the per-process string hash seed. That reorders the k-NN graph and the k-means embedding, which flips cluster labels and the noisy-cluster pick. Baseline gross Sharpe moved over roughly **2.96-3.23 across identical runs of identical code**, and the thin tiers moved much more (the $5M tier ranged 0.82-1.19). Fixed by sorting; pinned by a regression test that runs the loader under three different `PYTHONHASHSEED` values in subprocesses. With the fix, the committed-data table reproduces exactly:

| universe | avg members | gross Sharpe | net Sharpe @50bps | breakeven cost |
|---|---|---|---|---|
| at least  $50k/day (baseline) | 134 | 2.97 | **2.31** | 226 bps |
| at least  $1M/day | 89 | 1.07 | **0.33** | 73 bps |
| at least  $5M/day | 57 | 0.91 | **0.32** | 77 bps |

The baseline 2.31 matches the previously published 2.30. The two thin tiers do not match their published 0.04 and 0.66; those were single draws from the hash-seed distribution, which is exactly the defect.

**The liquidity tier labels were not what they said.** The `_volume` column of `data/all_tokens_24mo_daily.csv` is already denominated in USD (SNX $29M/day, ETH $8.7B/day on 2024-06-01), but `UniverseManager` filtered on `volume * price`. Every published tier floor was therefore a floor on USD × price, not on notional traded. `UniverseManager.volume_in_usd` selects the corrected filter; the default is left at the legacy behaviour so no previously published number moves silently, and `run_pit_robustness.py` reports both conventions.

### The price data behind a survivorship-free universe is dirty

Removing survivorship means admitting micro-caps whose CMC series contain redenominations and outright bad prints. vBNB prints 10.13 → 812.27 → 14.29 on flat supply; BTTOLD drops to 7.4e-07 for a single day and comes back; CAIR steps 1.0e-04 → 0.79 and stays there. A daily-rebalanced equal-weighted portfolio of this universe reports an annualised return over 3,000%, entirely the rebalancing bonus on assets quoted at 1e-07.

Circulating supply separates a redenomination (PUPS: price ÷10.4 as supply ×9.4, market cap flat) from a bad print, but not the bad prints above, which leave supply untouched. The panel therefore drops any daily move beyond `|log r| > log 5`: **328 daily observations, 0.06% of the panel**. Dropping rather than winsorising is the conservative choice, since it removes the most profitable-looking reversals from a mean-reversion book. `python stat_arb/run_pit_robustness.py` reports the count.

## The short leg mostly cannot be traded

The backtest is dollar-neutral, so it needs a short in every name it sells. Against Hyperliquid's listed perpetuals, for the committed 174-token universe:

| | count | share of universe |
|---|---|---|
| universe tokens | 174 | n/a |
| with a listed perpetual | 22 | **12.6%** |
| with funding history over the sample | 11 | **6.3%** |

**Roughly seven in eight names have no perpetual market.** That is a harder constraint than any cost assumption: it is not that shorting is expensive, it is that there is no venue. It also compounds the finding above rather than sitting beside it: the point-in-time universe is *larger* and *deeper into the tail* than the committed one, so its shortable share can only be worse.

For the 11 names that can be shorted and do have history, real funding is measured rather than approximated by the uniform `carry_bps_daily` knob:

- cross-token mean annualised funding **+5.56%** (positive means longs pay shorts, so a short position *earns* it)
- dispersion is enormous: **+49.1% (ILV) to -39.2% (BNT)**, with daily funding volatility reaching 133 bps for GAS

Funding is sourced from Hyperliquid rather than Binance for a reproducibility reason worth stating: **Binance's futures endpoints return HTTP 451 to US IP addresses**, so a US-based reader cannot reproduce a Binance-sourced funding series. Hyperliquid is an on-chain venue with an open API and no such restriction.

### That +5.56% is a regime that ended

Borri, Liu, Tsyvinski and Wu, ["Cryptocurrency as an Investable Asset Class: Coming of Age"](https://arxiv.org/abs/2510.14435) (arXiv:2510.14435), measure the Schmeling, Schrimpf and Todorov crypto-carry trade (short the perpetual, long the spot) and report an annualised Sharpe of **6.45** over 2020 to 2025, falling to **4.06** from 2024 and **turning negative in 2025**. Funding contributes a full-sample mean of roughly 8% at 0.8% volatility.

Their measurement is Bitcoin on Binance at 8-hour frequency, not a cross-section of altcoin perpetuals on Hyperliquid, so it is not a like-for-like comparison with the +5.56% above. The direction still matters: this repository's sample runs to May 2025, so the funding it measures spans exactly the period over which the carry compressed and then inverted. Treating +5.56% as a standing subsidy to the short leg extrapolates a regime that the best current evidence says has ended.

## What the repository does establish

- Signed-graph clustering (SPONGE, BNC, signed spectral) on a market-mode-residualized correlation graph, compared like-for-like across methods
- A reproducible point-in-time crypto universe including delisted tokens, built from public endpoints with no vendor licence, keyed on permanent `cmc_id` rather than reusable symbols
- That cluster mean-reversion alpha decays over multi-day horizons: trading every third day with a 2% no-trade band retains ~97% of gross Sharpe at a third of the turnover, which is Garleanu and Pedersen "aim in front of the target" showing up empirically
- Multiple-testing discipline throughout: Probabilistic and Deflated Sharpe Ratios, with each sweep treated as its own trial pool

**The clustering and execution methodology is the contribution. The Sharpe
is survivorship, and across the cap spectrum the one bracket that survives
costs (B1 large) is clusterable in only 20 of 114 months.**

## Repository layout

- `stat_arb/`: main research package for data loading, graph construction, clustering, signals, backtests, and reporting
- `data/`: processed market, volume, ETH, and correlation datasets, plus the derived point-in-time universe tables
- `pics/`: diagnostic figures for clustering quality and exploratory analysis
- `crypto_project.ipynb`: exploratory notebook used during early research
- `archived_research/`: older exploratory artifacts retained for reference
- `Crypto_Project_Report_Pre_Backtest.pdf`: written report from the earlier research stage

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib statsmodels pyarrow pytest
```

Run the test suite:

```bash
python -m pytest tests -q
```

## Reproducing the results

Build the point-in-time universe (~2,000 keyless requests, roughly 20 minutes; the raw pull is cached under `data/raw_cmc/` so a failure resumes rather than restarting):

```bash
python stat_arb/build_pit_universe.py
```

Run the point-in-time vs survivor-only comparison:

```bash
python stat_arb/run_pit_robustness.py
```

Run the original robustness checks on the committed snapshot universe:

```bash
python stat_arb/run_robustness.py
```

Run the baseline SPONGE backtest, the clustering-method sweep, and the cost-aware execution experiments:

```bash
python stat_arb/run_phase1.py
```

```bash
python stat_arb/run_phase2.py
```

```bash
python stat_arb/run_phase3.py
```

The point-in-time build needs no credential. If you want to rerun the notebook cells that call CoinMarketCap's *keyed* API, export your credential first:

```bash
export CMC_API_KEY=your_coinmarketcap_key
```

## Methodology

The pipeline first aligns token prices, volumes, and ETH reference data, then builds a tradable universe subject to history and liquidity filters. Returns are residualized against the market mode with PCA, transformed into a signed k-nearest-neighbor correlation graph, and clustered with SPONGE, BNC, or signed spectral methods. Signals are generated from within-cluster mean reversion, normalized to target leverage, and evaluated in a walk-forward backtest with lagging, turnover controls, and transaction-cost assumptions to limit lookahead and overstatement.

The point-in-time path replaces the snapshot universe at the front of that pipeline. At each monthly reconstitution, membership is the CoinMarketCap rank band as it stood on that date, so tokens that have since died are still present with the rank they actually held. Tokens whose prices stop inside the window are assigned a delisting return under an explicit, recorded rule: a documented final close on a day with non-zero volume is treated as an exit at that price, and anything else is a total loss. Because the panel is in log returns, where a -100% return is `-inf`, the total-loss case is applied as a -99% residual and the residual used is written into the universe table.

## Results

Primary outputs are written under `stat_arb/reporting/` and include fold-level returns, turnover series, clustering sweep summaries, leaderboards, and the final report. The intended use is comparative research across clustering methods rather than a production-ready live trading engine.

Reported Sharpe ratios are accompanied by the Probabilistic Sharpe Ratio and the Deflated Sharpe Ratio (Bailey and López de Prado), with each sweep treated as its own multiple-testing pool. On the committed snapshot universe the finding came in two halves: under daily rebalancing (phases 1-2) gross Sharpe is positive across all 16 configurations but nothing survives realistic taker costs, and the phase-3 execution experiments then show the alpha decays over multi-day horizons, lifting net Sharpe at 50bps from 1.0 to 2.3. The point-in-time rebuild supersedes that headline: on a universe that contains the tokens which died, the same configuration returns a net Sharpe of -0.14.

### 2026-08 revision

Results were regenerated after a signal-integrity pass. The material fixes, each covered by a regression test in `tests/`:

- inverse-volatility position sizing is now lagged one day (it previously used same-day volatility)
- cluster and dollar neutralization now apply only to selected names (they previously smeared small offsetting weights onto every token, inflating turnover)
- PCA market-mode fit no longer drops every row containing a single NaN on ragged histories
- k-NN graph construction is vectorized and compatible with pandas copy-on-write (the previous code crashed on pandas 3.x, so checked-in results could not be reproduced from a fresh environment)
- a financing-carry stress knob (`BacktestEngine.carry_bps_daily`) approximates perp funding / borrow drag on gross exposure

### 2026-09 revision

- point-in-time universe including delisted tokens, built from CoinMarketCap's public web API (`stat_arb/data/cmc_pit.py`, `stat_arb/data/pit_universe.py`, `stat_arb/build_pit_universe.py`)
- explicit, recorded delisting-return rule, with regression tests for each branch and for the ordering that keeps the scrubber from deleting the delisting shock
- column order pinned in `DataLoader.get_aligned_data`, which is what made the results reproducible from a fresh process
- `UniverseManager.volume_in_usd` for the corrected liquidity filter
- headline rewritten around the point-in-time result

## Known limits

- **The strategy has no measured edge on a survivorship-free universe.** Net Sharpe at 50bps is -0.14 at baseline. Everything above 0 in the published figures is selection
- The point-in-time universe spans 2023-05 to 2025-05 and CMC ranks 150-500, so it measures survivorship over a 24-month window in a mid-cap band, where only 9.4% of members died. A longer window or a deeper rank band would contain far more death and is the obvious next test
- Point-in-time rank snapshots come from an undocumented public endpoint. It has no stability guarantee, and the derived tables are committed precisely so results remain reproducible if it changes
- A -100% return is `-inf` in log space, so total-loss delistings are applied as -99%. `pit_universe.TOTAL_LOSS_RESIDUAL` is the knob and the value used is recorded per token
- Micro-cap price series carry redenominations and bad prints; 328 daily observations are dropped as artifacts. The threshold is a judgement call, and a mean-reversion book is exactly the strategy most sensitive to it
- Short legs are modeled as costless to hold; in practice they are perpetual futures with per-token, time-varying funding, and roughly seven in eight universe names have no perpetual market at all
- The checked-in notebook and archived artifacts reflect exploratory work and are less polished than the package backtest path
- Transaction costs and liquidity in crypto can change quickly enough to invalidate static assumptions

## License

This project is distributed under the MIT License
