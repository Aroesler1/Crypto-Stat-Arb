# Crypto Stat-Arb

Market-neutral cryptocurrency statistical arbitrage: signed-graph clustering over a market-mode-residualized correlation graph, with walk-forward backtesting, explicit transaction-cost modelling, and multiple-testing-aware validation.

## What it does

- **Removes the market mode by PCA**, then builds a signed k-nearest-neighbour correlation graph on the residuals, so clusters reflect relative rather than common movement
- **Clusters with twelve signed-graph methods**, from SPONGE and Balance Normalized Cut through the regularized and power-mean Laplacians of the 2019-2021 literature to three deliberately naive baselines, compared like-for-like with the number of clusters chosen inside each walk-forward window
- **Trades cluster mean reversion** under a daily turnover cap, a no-trade band, and a rebalance-frequency control
- **Validates with Probabilistic and Deflated Sharpe Ratios**, treating each sweep as its own multiple-testing pool, plus a financing-carry stress for perpetual funding
- **Rebuilds its own universe point-in-time from CoinMarketCap, including tokens that died**, and measures what their absence was worth
- **Cuts that universe into four ETH-relative market-cap brackets** and asks where along the cap spectrum within-cluster mean reversion exists, and where it can actually be traded

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

| bracket | best signal | net @50bps | tradeable, after funding | verdict |
|---|---|---|---|---|
| B0 mega | not clustered | - | - | never reaches 30 members; a pairs book, not a cluster book |
| B1 large | death filter | +0.52 | **-0.05** | shortable, but only 18 names survive the cut, below the clustering floor |
| B2 mid | ewma | +0.44 | **-0.22** | clusterable and half shortable; the edge does not survive the restriction |
| B3 small | ewma_sized | +1.51 | **+1.44** | the original band, and the only one that survives every restriction |

The answer is not the one the perpetual-coverage table suggested: **the original
small-cap band is the one that works, and it works because it is big enough to
survive being cut down.** B3 loses two thirds of its members to the shortability
restriction and still has 87, comfortably above the 30 needed to cluster. B1 has
the best shortability in the panel and only 18 names left after the cut, below
the floor. Size of cross-section, not availability of a short, is what binds.

**The clustering and execution methodology is the contribution. The Sharpe was
survivorship, and what recovers it is not better clustering but a better
standardisation: an EWMA z-score in place of a 20-day rolling one is worth 1.9
Sharpe on the small-cap bracket, point-in-time, after costs. Funding on that
bracket is measured on only 13% of exposure, so +1.44 after funding is a lower
bound on cost rather than a settled number.**

## Clustering methods compared, and a dumb baseline that wins

Twelve methods, three brackets, two survivorship treatments, k selected inside
each walk-forward window by signflip parallel analysis, everything else held
fixed. Net Sharpe at 50 bps on the point-in-time universe:

| method | B1 | B2 | B3 | note |
|---|---|---|---|---|
| **PCA-kmeans** | -0.82 | -0.49 | **+0.82** | baseline, ignores signs entirely |
| SPONGEsym | **+0.35** | -0.47 | -0.73 | corrected in this work |
| PowerMean p=1 | +0.33 | -0.56 | -0.61 | Mercado et al. ICML 2019 |
| BNC | +0.23 | -0.13 | -0.61 | corrected in this work |
| PowerMean p=0 | +0.15 | -0.66 | -1.16 | |
| SPONGE | -0.10 | -0.23 | -0.48 | the incumbent |
| SignedSpectral | -0.11 | -0.44 | -0.40 | |
| RegSPONGE | -0.34 | *+0.34* | -1.01 | *degenerate, see below* |
| RegSignedSpectral | -0.34 | *+0.08* | -0.97 | *degenerate, see below* |
| Pivot | -0.37 | -1.17 | -0.68 | infers k: 2.3, 14.8, 49.1 |
| Hierarchical | -0.46 | -1.10 | -1.24 | baseline, no signs |
| PowerMean p=-10 | -0.66 | -0.62 | -1.43 | |

Full table with eigengap, Calinski-Harabasz, Davies-Bouldin, cluster stability
and both survivorship treatments:
`stat_arb/reporting/brackets/clustering_sweep.csv`.

**The two apparent winners on B2 are not clustering at all.** RegSPONGE and
RegSignedSpectral select k = 1.25 on average there, so in most windows they find
a single cluster, the noisy-cluster drop removes it, and the book holds nothing.
Their turnover is 0.0135 against roughly 0.048 for every other method, a factor
of 3.6, and on B1 it is 0.0032 against 0.048, a factor of 15. A nearly flat
series with a small positive drift reports a positive Sharpe. This is not
evidence that regularization helps; it is a book that mostly declines to trade,
and it is marked as such rather than ranked first.

**The dumb baseline wins B3 outright, and it is not degenerate.** k-means on the
leading eigenvectors, which throws the sign structure away entirely, is the only
method with a positive net Sharpe on B3 point-in-time: +0.82 against -0.40 for
the best signed method. Its turnover is 0.047, in line with everything else, and
its cluster stability of 0.700 is the highest in the bracket. It trades as much
as the others and does better.

That is the honesty check firing exactly as it was built to. The signed spectral
machinery is not what produces the result on the bracket where the original
hypothesis lives. On B1, where the signed methods do win, the margin is
SPONGEsym at +0.35 over PCA-kmeans at -0.82, so the machinery earns its place
there and only there.

**The two corrected clusterers are first and third on B1.** SPONGEsym and BNC,
both of which were measuring something other than what they claimed before this
work, are the best and third-best methods on the one bracket that supports a
tradable book. Neither would have appeared in the ranking at all if the
implementations had been trusted rather than tested.

Two further observations. Pivot, the only method not told k, infers 2.3 clusters
on B1, 14.8 on B2 and 49.1 on B3, so the cross-section fragments as it gets
larger rather than resolving into a stable handful of groups. And the report's
own criteria disagree with the criterion that matters: on B3 point-in-time,
RegSignedSpectral has the best Calinski-Harabasz score in the bracket (854
against PCA-kmeans's 147) and the second-worst net Sharpe. A partition-quality
score rewards a tidy partition, not a tradable one.

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

## Signal extensions: the standardisation was throwing the signal away

Eight arms per bracket, each changing one thing about how the deviation from a
cluster is measured or sized, everything else held fixed. Net Sharpe at 50 bps
on the point-in-time universe, with the Deflated Sharpe Ratio treating the arms
in a cell as the multiple-testing pool:

| arm | B1 | B2 | B3 | B3 DSR |
|---|---|---|---|---|
| **ewma_sized** | -0.20 | +0.36 | **+1.51** | **0.590** |
| **ewma** | -0.03 | **+0.44** | +1.45 | 0.527 |
| **death filter** | **+0.52** | -0.20 | -0.29 | 0.000 |
| baseline (published signal) | +0.40 | -0.24 | -0.42 | 0.000 |
| ou (s-score, half-life filter) | -0.49 | +0.15 | -0.31 | 0.000 |
| momentum overlay | -0.07 | -0.11 | -0.23 | 0.000 |
| ou_nofilter | -0.80 | -0.36 | -0.80 | 0.000 |
| beta-adjusted | -1.04 | -0.96 | -1.14 | 0.000 |

Full table with PSR, turnover and breakeven, and the survivor-only half:
`stat_arb/reporting/brackets/signal_ablation.csv`.

**EWMA standardisation is the single largest effect in this project.** On B3
point-in-time it takes the book from -0.42 to +1.45, a breakeven above 500 bps,
and a Deflated Sharpe of 0.527 against the eight-arm pool. Adding half-life
position sizing on top (`ewma_sized`) takes it to +1.51, though that second
piece helps only on B3 and costs 0.4 and 0.1 Sharpe on B1 and B2 respectively.
Nothing else in this repository's history has produced a positive net Sharpe on
a survivorship-free small-cap universe.

Every claim below is backed by
`stat_arb/reporting/brackets/ewma_robustness.csv`, which reports per arm and per
survivorship treatment on B3: net Sharpe by calendar year, net Sharpe excluding
the ten best days, the share of total return those days carry, and turnover.

* **Not concentration.** Excluding its ten best days the Sharpe falls from 1.45
  to 1.22, and those days carry 18.6% of total return. The comparison that
  matters is against the control: the baseline falls from -0.42 to -0.66 over
  the same exclusion. Both lose about 0.23 Sharpe, so EWMA is not
  disproportionately dependent on a handful of sessions; it is simply higher
  everywhere. (For a losing book the "share" is negative, because the best days
  are offsetting a negative total. Read it against the Sharpe column, not alone.)
* **Positive in all ten years**, 3.62 in 2016 down to 0.44 in 2023 and back to
  2.02 in 2025. The baseline is negative in six of the same ten.
* **Not survivorship.** +1.45 point-in-time against +1.40 survivor-only.
  Everything else on B3 shows a large gap between the two; a signal that does
  not depend on which losers were deleted is the one worth having.
* **Not a flat book.** Turnover 0.041 against the baseline's 0.053.
* **Not the half-life.** The 10-day half-life was fixed before any of these
  results were seen. Reported as robustness rather than selection: 5 days gives
  +1.65, 10 gives +1.45, 20 gives +1.09, all strongly positive, and the pre-set
  default is not the best of the three.

The mechanism is unglamorous. The published signal standardises the cluster
deviation with a 20-day rolling mean and standard deviation. On a micro-cap
panel that denominator steps discontinuously every time a large move enters or
leaves the window, and that step is noise injected into every signal sharing it.
An exponentially weighted mean and standard deviation decay instead of dropping,
and on the bracket with the most volatile members it is worth 1.9 Sharpe. The
clustering, the residualization and the execution controls are identical.

**The death filter is the only arm that helps B1**, taking it from +0.40 to
+0.52 with a breakeven of 111 bps. It gates the long leg of the decile of names
the walk-forward classifier scores most likely to be delisted within 30 days.
The classifier is weak, mean out-of-sample AUC 0.565 over nine years (range
0.514 to 0.595), and its usable signal is not the feature we were worried about:
the standardised coefficients are dominated by rank deterioration (+0.21 to
+0.30) and falling volume (-0.13 to -0.20), while days-since-listing contributes
little and changes sign across years.

Two arms are worth reporting for going the wrong way. The **beta-adjusted
deviation is the worst arm in every bracket**, which says the raw gap to the
cluster composite is a better trade than the regression residual: estimating a
beta per token per window on this data adds more noise than the mismatch it
corrects. And the **OU s-score is better with its half-life filter than without
it in every cell**, which is the Avellaneda-Lee condition earning its place:
dropping tokens whose implied reversion is slower than the holding horizon is
worth 0.3 to 0.5 Sharpe.

## The tradability verdict, and it is not the expected shape

Restricting each bracket to names with a listed perpetual on Binance,
Hyperliquid, dYdX v4 or Deribit, then charging measured funding to the positions
actually held. Each bracket runs its own best Step 3 arm:

| bracket | arm | all names | shortable only | after funding | members kept | funding covered (exposure) |
|---|---|---|---|---|---|---|
| B1 large | death | +0.40 | -0.15 | **-0.05** | 18 of 21 (86%) | 72% |
| B2 mid | ewma | +0.44 | -0.27 | **-0.22** | 57 of 99 (58%) | 67% |
| B3 small | ewma_sized | +1.51 | +1.45 | **+1.44** | 87 of 261 (33%) | 13% |

**This is the opposite of the expected shape.** The prediction was that large
caps would be shortable but short of dispersion, and small caps would have the
dispersion but no shorts. What happens instead is that B3 keeps only a third of
its names and holds its result, while B1 and B2 keep most of theirs and are
destroyed by the restriction.

The reason is the clustering floor rather than anything about dispersion. B3
retains 87 shortable members, comfortably above the 30 needed to cluster, so the
book is intact. B1 falls to 18, below the floor, and B2's 57 come from a bracket
whose signal was marginal to begin with. Shortability does not bind on B3
because B3 is large enough to lose two thirds of itself and still be a
cross-section.

Funding is close to irrelevant at these sizes: it costs B3 0.01 Sharpe and
helps B1, whose short leg earns more funding than it pays.

**Funding coverage is reported two ways** because the naive one misleads.
`cov(exposure)` is the share of the book's total absolute position sitting on a
name with a measured rate on the day it is held; `cov(col)` is the share of the
bracket's return columns carrying a rate at all, which is dragged down by tokens
that passed through the bracket years before any venue listed a perpetual on
them, including names the book never holds. B1 and B2 clear two thirds of
exposure covered. **B3 does not: at 13% of exposure, its after-funding figure is
a lower bound on the cost, not a full accounting.** The book concentrates its
positions in exactly the names and years that no venue quoted, which is what
being a small-cap book over 2016-2025 means. Binance's archive begins in 2020,
and Hyperliquid, the only other venue whose history was pulled, lists just 6
bases Binance does not already carry, so this gap is not closable from the
venues a US IP can reach.

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

**The clustering and execution methodology is the contribution. The Sharpe was
survivorship, and the bracket that survives every restriction is B3, the
original small-cap band, traded with an EWMA-standardised signal.** B1, the
bracket that looked most promising on shortability, clears the 30-member
clustering floor in only 20 of 114 months and falls to 18 names once cut to
those with a listed perpetual, which is below the floor. What recovers the
result is not better clustering but a better standardisation.

## Factor diagnostics (secondary)

Labelled secondary because it is not the strategy question. Two things worth
knowing about the book that survived, neither of which changes the verdict.

### (a) Is within-cluster mean reversion just reversal plus a size bet?

Each bracket's best point-in-time book, regressed weekly on factors built on the
academic universe (CoinMarketCap rank 1-1000 with market cap above $1M):

| bracket | arm | weekly alpha | t | R2 | only significant loading |
|---|---|---|---|---|---|
| B1 | death | +0.20% | 0.88 | 0.02 | none |
| B2 | ewma | +0.14% | 0.60 | 0.02 | none |
| B3 | ewma | **+2.16%** | **2.17** | 0.02 | size, t = -2.70 |

**No.** The B3 book carries a significant weekly alpha of 2.16% against these
factors, its short-term reversal loading is insignificant (t = 0.56), and the
factors jointly explain 2% of its variance. The one loading that is significant
is a negative tilt on size, meaning the book leans toward the larger names
inside the small-cap bracket, which is the opposite of the "it is just a
small-cap bet" reading.

Two caveats on how far that goes. An R-squared of 0.02 is low enough that it
partly reflects these factors being weak on this universe rather than the book
being exotic; most of them have negative net Sharpes in the table below. And
B1's and B2's alphas are not distinguishable from zero, which is what their
near-zero Sharpes already said.

### (b) How much of the crypto factor zoo is survivorship?

Twenty-three factors, three universes differing only in what they can see,
weekly quintile spreads, value-weighted, net of 50 bps, 2016-2025. The largest
survivorship differences:

| factor | point-in-time | survivor-only | snapshot | PIT minus survivor | t |
|---|---|---|---|---|---|
| size | -0.76 | +0.17 | +0.40 | **-0.93** | -3.01 |
| turning_points | -1.58 | -1.12 | -0.77 | -0.45 | |
| reversal_1w | -1.82 | -1.51 | -0.86 | -0.31 | |
| high_52w | +0.32 | -0.02 | -0.37 | +0.34 | 2.14 |
| mom_2w | +0.66 | +0.60 | +0.50 | +0.07 | |

Full table: `stat_arb/reporting/brackets/factor_table_three_universes.csv`.

**Romano-Wolf across all 23 factors, two-sided, family-wise error rate 5%:
nothing is significant.** Size has the largest difference by far and a
t-statistic of -3.01, which would clear any single-factor threshold, but its
adjusted p-value is 0.107 once the other 22 tests are accounted for. Reporting
it as a finding would be exactly the error the procedure exists to prevent.

That is a real result, and it lines up with the rest of this repository rather
than contradicting it. Survivorship at the **universe** level is small (the
existing measurement puts it at 4.56% equal-weighted, 2.11% value-weighted).
Survivorship at the **factor** level is, on this construction, not detectable at
all. Survivorship at the **strategy** level is worth +0.97 Sharpe on B3. The
reason is the same each time: a value-weighted quintile spread does not
systematically overweight the tokens that are about to die, and a
mean-reversion book does, because dying tokens are exactly what a
mean-reversion signal wants to buy.

The direction of the size result is worth noting even though it fails the
correction: the size factor is the one that goes long the small names, which are
the ones that die, so it is where a survivorship effect should show up first.

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

The pipeline first aligns token prices, volumes, and ETH reference data, then builds a tradable universe subject to history and liquidity filters. Returns are residualized against a chosen reference (ETH, BTC, the value-weighted market, or a PCA market mode, selected per bracket by the Step 1 ablation), transformed into a signed k-nearest-neighbor correlation graph, and clustered by any of twelve methods with the cluster count chosen inside each walk-forward window. Signals are generated from within-cluster mean reversion, normalized to target leverage, and evaluated in a walk-forward backtest with lagging, turnover controls, and transaction-cost assumptions to limit lookahead and overstatement.

The point-in-time path replaces the snapshot universe at the front of that pipeline. At each monthly reconstitution, membership is the CoinMarketCap rank band as it stood on that date, so tokens that have since died are still present with the rank they actually held. Tokens whose prices stop inside the window are assigned a delisting return under an explicit, recorded rule: a documented final close on a day with non-zero volume is treated as an exit at that price, and anything else is a total loss. Because the panel is in log returns, where a -100% return is `-inf`, the total-loss case is applied as a -99% residual and the residual used is written into the universe table.

## Results

Primary outputs are written under `stat_arb/reporting/` and include fold-level returns, turnover series, clustering sweep summaries, leaderboards, and the final report. The intended use is comparative research across clustering methods rather than a production-ready live trading engine.

Reported Sharpe ratios are accompanied by the Probabilistic Sharpe Ratio and the Deflated Sharpe Ratio (Bailey and López de Prado), with each sweep treated as its own multiple-testing pool. On the committed snapshot universe the finding came in two halves: under daily rebalancing (phases 1-2) gross Sharpe is positive across all 16 configurations but nothing survives realistic taker costs, and the phase-3 execution experiments then show the alpha decays over multi-day horizons, lifting net Sharpe at 50bps from 1.0 to 2.3. The point-in-time rebuild supersedes that headline: on a universe that contains the tokens which died, the same configuration returns a net Sharpe of -0.14. The bracket work supersedes it again. Cut into ETH-relative market-cap brackets over 2016-2025, the original small-cap band (B3) returns +1.45 net of 50 bps once the signal is standardised with an EWMA rather than a 20-day rolling window, and +1.48 after restricting to names with a listed perpetual and charging measured funding. The clustering is not what changed.

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
- **ETH-relative market-cap brackets** (`stat_arb/data/brackets.py`,
  `stat_arb/run_bracket_panel.py`): the panel is rebuilt from 3,653 daily
  point-in-time CoinMarketCap rankings over 2015-2025 rather than per-token
  history, because that endpoint prunes coins that died three or more years ago
  and a panel built from it is close to survivor-only in its early years while
  being labelled point-in-time
- **death separated from rank censoring**, and the delisting rule's `volume > 0`
  test replaced with an explicit, reported policy: the median final-day volume
  of a dying token is $25.51, so the old test classified every death as a clean
  exit
- **twelve clustering methods** including the regularized signed Laplacian and
  regularized SPONGEsym (JMLR 2021) and the signed power mean Laplacian (ICML
  2019), with the cluster count chosen per window by signflip parallel analysis
  (`stat_arb/clustering/regularized.py`, `stat_arb/run_clustering_sweep.py`)
- **two pre-existing clusterers corrected**: `BNCClustering` solved the wrong
  eigenproblem and read the wrong end of the spectrum, and
  `SPONGEClustering.fit_symmetric` was a similarity transform of the pencil
  `fit` already solves, so "SPONGEsym" was plain SPONGE run twice
- **signal extensions** (`stat_arb/signals/ou_score.py`,
  `stat_arb/run_signal_ablation.py`): the Avellaneda-Lee s-score, a
  beta-adjusted deviation, EWMA standardisation with and without half-life
  sizing, a cluster-momentum overlay, and a walk-forward death classifier gating
  the long leg
- **measured perpetual funding across four venues**
  (`stat_arb/data/perps.py`, `stat_arb/build_funding_panel.py`), which took
  shortability coverage from 11 tokens to 945 bases and funding from 11 tokens
  to 465
- **crypto factor diagnostics** (`stat_arb/factors/`,
  `stat_arb/run_factor_diagnostics.py`): 23 characteristics, three universes,
  and Romano-Wolf across factors

## Known limits

- **The point-in-time edge lives in one bracket and one signal.** B3 with EWMA
  standardisation is +1.45 net of 50 bps; B1 and B2 are negative once restricted
  to shortable names. Everything above zero in the pre-2026-09 published figures
  is still selection
- **The EWMA half-life is one number chosen without a sweep.** It was fixed at
  10 days before any of these results were seen, and
  `stat_arb/reporting/brackets/ewma_robustness.csv` reports 5 and 20 as
  robustness rather than selection. A genuine sweep would need its own
  multiple-testing treatment, and this result deserves independent replication
  before anyone trades it
- **Funding is measured, not complete.** See the tradability table for the
  covered share by bracket, reported both as a fraction of return columns and,
  more usefully, as a fraction of the book's absolute exposure. Binance's
  archive begins in 2020 while the panel begins in 2016, and Hyperliquid, which
  is the only other venue whose history was pulled, lists just 6 bases Binance
  does not already carry. Where a rate is missing the position contributes zero,
  so the after-funding figure understates cost wherever coverage is partial
- **Micro-cap price series carry redenominations and bad prints**, and the rate
  moves by an order of magnitude across the sample. Daily observations dropped
  as artifacts, by year (`data/bracket_data_quality.csv`):

  | year | bad prints | redenominations | observations | % of panel |
  |---|---|---|---|---|
  | 2015 | 86 | 0 | 21,343 | 0.403 |
  | 2016 | 1,002 | 3 | 208,541 | 0.480 |
  | 2017 | 1,827 | 1 | 332,971 | 0.549 |
  | 2018 | 668 | 1 | 607,389 | 0.110 |
  | 2019 | 1,561 | 1 | 715,977 | 0.218 |
  | 2020 | 1,842 | 2 | 709,167 | 0.260 |
  | 2021 | 1,289 | 19 | 698,341 | 0.185 |
  | 2022 | 296 | 27 | 697,698 | 0.042 |
  | 2023 | 453 | 10 | 633,499 | 0.072 |
  | 2024 | 298 | 117 | 678,788 | 0.044 |
  | 2025 | 71 | 41 | 334,933 | 0.021 |

  The threshold is a judgement call, and a mean-reversion book is exactly the
  strategy most sensitive to it. The early years are five to ten times dirtier
  than the recent ones, which is worth remembering when reading a 2016 Sharpe
- Point-in-time rank snapshots come from an undocumented public endpoint. It has
  no stability guarantee, and the derived tables are committed precisely so
  results remain reproducible if it changes
- A -100% return is `-inf` in log space, so total-loss delistings are applied as
  -99%. `pit_universe.TOTAL_LOSS_RESIDUAL` is the knob and the value used is
  recorded per token
- **Coin Metrics community data is CC BY-NC 4.0**, which is more restrictive
  than this repository's MIT licence: attribution required, non-commercial use
  only. Any series derived from it carries that restriction rather than the
  repository's own terms. `stat_arb/data/external_factors.py` can fetch it and
  Wikipedia pageviews, but neither the network-value nor the attention factor is
  in the committed factor table; coverage is the constraint, since Coin Metrics
  carries roughly 1,000 assets against the 3,210 that pass through B3
- The deep-learning signal of Guijarro-Ordonez, Pelger and Zanotti (2021) was
  not attempted, and B0's pairs book and no-cluster cross-sectional z-score were
  not built
- The checked-in notebook and archived artifacts reflect exploratory work and are
  less polished than the package backtest path
- Transaction costs and liquidity in crypto can change quickly enough to
  invalidate static assumptions

## License

This project is distributed under the MIT License
