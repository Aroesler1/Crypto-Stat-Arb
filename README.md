# Signed-Graph Crypto Stat-Arb

I cluster crypto tokens on a signed correlation graph and backtest market-neutral
mean reversion inside each cluster, on a survivorship-free universe from 2016 to
mid-2025.

The question is whether small-cap tokens that usually move together can be traded
when one of them drifts away from its group. An honest answer needs three things:
a universe that still contains the tokens that later died, a size definition that
means the same thing in 2016 and in 2025, and accounting a real account could
reproduce. I built all three and measured how much each one changes the answer.
The short version: the strong early backtests did not survive them.

## Results

Sample: four ETH-relative market-cap brackets rebuilt at 114 month-ends from
2016-01-31 to 2025-06-30 from daily point-in-time CoinMarketCap rankings that
include dead tokens. Backtest returns run from 2016-06-30 to 2025-06-30, net of
50 bps on every dollar traded.

Capital replay of the two fixed signals in each clustered bracket, ETH-excess
returns, SPONGE with three clusters
([accounting_summary.csv](stat_arb/reporting/brackets/accounting_summary.csv)):

| bracket | signal | cash-and-holdings result at 50 bps |
|---|---|---|
| B1 large | baseline | Sharpe 0.078 |
| B1 large | EWMA | Sharpe 0.033 |
| B2 mid | baseline | Sharpe 0.539 |
| B2 mid | EWMA | Sharpe 0.077 |
| B3 small | baseline | insolvent on 2017-12-08 |
| B3 small | EWMA | insolvent on 2020-12-28 |

- **The headline small-cap result does not survive capital accounting.** Under
  the retired weighted log-return scoring, B3 with EWMA standardisation had a net
  Sharpe of 1.45 and was the project's best number
  ([signal_ablation.csv](stat_arb/reporting/brackets/signal_ablation.csv)). Held
  as cash and dollar holdings, both fixed B3 books go insolvent.
- **Some B3 books stay solvent, but not once the short leg must exist.** Of eight
  signal arms on point-in-time B3, four survive the ledger; the best, the
  Avellaneda-Lee s-score without its half-life filter, has a net Sharpe of 1.16
  ([signal_ablation_corrected.csv](stat_arb/reporting/brackets/signal_ablation_corrected.csv)).
  Restricted to the 87 names (on average) with a listed perpetual, it falls to
  0.06, and to 0.05 after observed funding
  ([tradability_corrected.csv](stat_arb/reporting/brackets/tradability_corrected.csv)).
- **Survivorship was the original edge.** On the first 2023-2025 universe
  (CoinMarketCap rank 150-500), net Sharpe falls from 2.02 with today's survivors
  to -0.14 once the 50 tokens that died are put back, under the older scoring
  ([pit_vs_survivor.csv](stat_arb/reporting/robustness/pit_vs_survivor.csv)).
- **Sign information did not help where it was supposed to.** With the number of
  clusters chosen inside each training window, every unregularized signed method
  (SPONGE, SPONGEsym, BNC, the signed Laplacian, the three power means) and Pivot
  goes insolvent on point-in-time B3. The survivors are the two regularized
  methods (0.87 and 0.85, though their embedding is degenerate in about a third
  of windows), hierarchical clustering (0.75), which ignores signs, and PCA plus
  k-means (0.27)
  ([clustering_sweep_corrected.csv](stat_arb/reporting/brackets/clustering_sweep_corrected.csv)).
- **The short leg is mostly unverifiable.** Only 29.0% of B3's perpetual-matched
  member-days have a dated, unambiguous funding print on the prior day, and none
  do before 2020
  ([summary](stat_arb/reporting/brackets/availability_summary.csv),
  [by year](stat_arb/reporting/brackets/availability_by_year.csv)).
- **The universe is wide.** B3 (0.01% to 0.1% of ETH's market cap) contains 3,210
  distinct tokens over the sample, while B1 reaches the 30-name clustering floor
  in only 20 of 114 months
  ([headline_universe_reproduction.csv](stat_arb/reporting/brackets/headline_universe_reproduction.csv)).

## How it works

1. **Point-in-time universe.** One CoinMarketCap ranking per day (top 2,000,
   2015-07-01 to 2025-06-30) from the public, keyless web API, so tokens that
   later died appear with the rank they actually had. Stablecoins and wrapped,
   staked and bridged tokens are excluded.
2. **ETH-relative brackets.** At each month-end a token's size is its market cap
   divided by ETH's: B0 at 10% or more, B1 from 1% to 10%, B2 from 0.1% to 1%, B3
   from 0.01% to 0.1%. A ratio to ETH picks out a comparable slice of the market
   in every year, which a fixed dollar band does not.
3. **Walk-forward clustering.** Every 28 days I take the previous 365 days of
   ETH-excess log returns for liquid members (at least $50k daily volume and 60
   days of history), optionally remove principal components, build a signed
   k-nearest-neighbour correlation graph (k = 10) and cluster it. The default is
   SPONGE with three clusters; the method sweep compares 12 clusterers with k
   chosen by signflip parallel analysis. A bracket with fewer than 30 eligible
   names is not traded in that 28-day block, and the least cohesive cluster is
   dropped.
4. **Signal.** Each token's 5-day cumulative return gap to its cluster average is
   standardised, either with a 20-day rolling window (baseline) or an
   exponentially weighted one with a 10-day half-life (EWMA). Both enter at
   |z| > 1.5 and use data through the previous day. I buy laggards and short
   leaders, neutral within each cluster, at 1.5x gross leverage.
5. **Execution controls.** Trade every third day, ignore weight changes under 2%,
   and cap turnover at 15% of equity per day.
6. **Accounting.** The book starts with $1 of cash. Dollar holdings drift with
   simple returns between trades, fees of 50 bps leave cash at each trade, and
   shorts are negative holdings financed through cash. If equity reaches zero the
   run stops and is reported as insolvent. A held token with a missing price is
   marked flat, and that exposure share is reported.
7. **Checks.** Probabilistic and Deflated Sharpe ratios, point-in-time against
   survivor-only reruns, and a dated audit of which shorts had a funding record.

## Limitations

- The backtest trades spot-price proxies. It has no historical perpetual listing
  dates, margin rules or liquidation engine. The capital replay charges no funding,
  and the funded experiments observe it for only part of the book, so none of
  these numbers is an executable return.
- Many references, clustering methods and signals were tried on the same sample.
  Each sweep has its own multiple-testing correction, but nothing corrects for the
  whole research history, so none of the positive rows is a discovery.
- B1 only trades in the months it has 30 eligible names, so its replay is mostly
  flat (average daily turnover 0.16% to 0.23%).
- Micro-cap price series contain bad prints. Daily moves larger than a factor of
  five are dropped, and a mean-reversion book is sensitive to that threshold.

## Repository layout

| path | contents |
|---|---|
| `stat_arb/` | the package; `backtest/holdings.py` is the cash-and-holdings ledger |
| `stat_arb/run_*.py`, `build_*.py` | runners for each experiment and data build |
| `stat_arb/reporting/brackets/` | result tables for the bracket study and the capital replay |
| `data/` | committed derived data: bracket tables, funding panel, the 2023-2025 universe |
| `tests/` | offline unit and regression tests |
| `docs/` | research log, audits and literature notes |
| `crypto_project.ipynb`, `Crypto_Project_Report_Pre_Backtest.pdf` | May 2025 exploratory work, before any backtest |
| `archived_research/` | early exploratory files kept for reference |

## Running it

The tests and checks need no API key or vendor data:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
python stat_arb/run_accounting_audit.py --check
python stat_arb/run_availability_audit.py --check
```

The `--check` commands verify the committed result tables in a few seconds.
`run_pit_robustness.py` runs offline. `run_phase1.py` to `run_phase3.py` and
`run_robustness.py` need the raw 2023-2025 CoinMarketCap panel, which is vendor
data and not distributed: set `STAT_ARB_RAW_DATA_DIR` to the folder holding it
([DATA.md](DATA.md) explains how to rebuild it). The bracket experiments need
the daily ranking panel, which takes 3,653 keyless requests (about 100 minutes):

```bash
python stat_arb/build_daily_listings.py
python stat_arb/run_signal_ablation.py --brackets B1,B2,B3
python stat_arb/run_accounting_audit.py
```

Bracket runners take `--panel` to read the panel from elsewhere and write
`*_corrected.csv` files next to the historical log-score tables. Only the
notebooks' data-pull cells need API keys (see `.env.example`).

## More detail

- [docs/audit_history.md](docs/audit_history.md): the full research log, every
  historical table, and each dated correction
- [docs/accounting_audit.md](docs/accounting_audit.md): why the log-return scoring
  was replaced and what the ledger does
- [docs/availability_review_2026.md](docs/availability_review_2026.md): the dated
  shortability audit
- [docs/signed_clustering_2026.md](docs/signed_clustering_2026.md): signed-graph
  clustering literature review
- [DATA.md](DATA.md): data sources, what is committed, and licences
- [stat_arb/reporting/FINAL_REPORT.md](stat_arb/reporting/FINAL_REPORT.md): the
  first backtest on the 2023-2025 survivor snapshot (historical)
- [Crypto_Project_Report_Pre_Backtest.pdf](Crypto_Project_Report_Pre_Backtest.pdf):
  the May 2025 pre-backtest report on 174 tokens; several of its citations are
  wrong, and the corrections are in the
  [errata](docs/audit_history.md#errata-for-the-pre-backtest-report)

## License

The code is MIT licensed. Data keeps its source-specific terms, including
Coin Metrics CC BY-NC 4.0 for any series derived from it; see [DATA.md](DATA.md).
