# Handoff

## Current goal — DONE

Replace the repository's survivorship caveat with a measurement. The README used
to say fixing it "needs a paid dataset which has not been purchased". That was
wrong: CoinMarketCap's public web API serves point-in-time rank snapshots and
full daily OHLCV for delisted coins with no key and no paid plan.

## Headline finding

Same window, same rank band, same filters, same engine; three universes that
differ only in what they are allowed to see (band 2%, rebalance 3d, net 50bps,
>=$50k/day notional):

| universe | avg members | gross SR | net SR @50bps | breakeven |
|---|---|---|---|---|
| snapshot (survivors, complete history) | 224 | 2.40 | 2.02 | 314 bps |
| survivor-only (survivors, any history) | 275 | 1.73 | 1.37 | 240 bps |
| point-in-time (incl. 50 dead tokens) | 284 | 0.19 | -0.14 | 29 bps |

The `snapshot` row reproduces the repo's own construction on rebuilt data and
lands near the published 2.30, which is what makes it a like-for-like. The edge
is selection. Collapse holds at every tier ($1M: 1.85 / 1.84 / 0.41;
$5M: 1.53 / 0.51 / -2.03).

Strategy-level survivorship drag, paired by date, +/- 1 s.e.:
+40.9% +/- 16.4% (baseline), +29.5% +/- 14.5% ($1M), +65.7% +/- 22.0% ($5M).

Universe-level monthly buy-and-hold (the construction the published figures
measure): PIT EW +8.24% / VW +4.19%; survivor-only EW +12.79% / VW +6.30%;
survivorship bias **EW +4.56%, VW +2.11%** against Ammann et al.'s EW 62.19% /
VW 0.93%. The universe-level bias is small; the strategy-level bias is 7-14x
larger because a mean-reversion book systematically buys the losers that die.

## Verified state

- `python -m pytest tests -q` -> **50 passed** (was 44; 26 new across
  `tests/test_pit_universe.py` and `tests/test_loader.py`).
- `stat_arb/run_pit_robustness.py` verified **bit-for-bit identical across two
  runs** after the hash-seed fix below.
- PIT universe built: 989 `cmc_id`s in rank band [150, 500] over
  2023-05-29..2025-05-29; 933 after stablecoin/wrapped exclusion; 88 delisted
  (9.4%); delisting rules alive 904 / total_loss 25 / last_price 4;
  560 of 933 have >=99% coverage.

## Two reproducibility defects found and fixed

1. **Results were not reproducible from a fresh process.**
   `DataLoader.get_aligned_data` derived column order by iterating a Python
   `set`, so order followed the per-process string hash seed. That reorders the
   k-NN graph and k-means embedding, flipping cluster labels and the
   noisy-cluster pick: baseline gross Sharpe ranged **2.96-3.23 across identical
   runs**, the $5M tier 0.82-1.19. Fixed with `sorted()`; pinned by
   `tests/test_loader.py`, which runs the loader under three `PYTHONHASHSEED`
   values in subprocesses. Committed-data table now reproduces exactly:
   baseline 134 / 2.971 / 2.310 / 226.1 bps (published was 2.96 / 2.30 / 226).
   The thin tiers do NOT match their published 0.04 and 0.66 -- those were
   single draws.
2. **Liquidity tier labels were wrong.** `<TOK>_volume` in
   `all_tokens_24mo_daily.csv` is already USD, but `UniverseManager` filtered on
   `volumes * prices`. Added `UniverseManager.volume_in_usd`; **default left at
   legacy** so no published number moves silently. `run_pit_robustness.py`
   reports both conventions.

## Data quality

A survivorship-free micro-cap universe has dirty prices: vBNB 10.13 -> 812.27 ->
14.29 on flat supply, BTTOLD 7.4e-07 for one day, CAIR 1.0e-04 -> 0.79 and
stays. Daily-rebalanced EW on the raw panel reports >3000% annualised. Supply
separates true redenominations (PUPS: price /10.4, supply x9.4, mcap flat) but
not the bad prints, so the panel drops `|log r| > log 5`: **328 observations,
0.06% of the panel**. Dropping (not winsorising) is conservative for a
mean-reversion book. Threshold is a judgement call and is the most obvious thing
to sensitivity-test next.

## Files

New: `stat_arb/data/cmc_pit.py` (fetch layer), `stat_arb/data/pit_universe.py`
(delisting rule, panels, membership), `stat_arb/build_pit_universe.py` (one
command build), `stat_arb/run_pit_robustness.py` (three-book comparison),
`tests/test_pit_universe.py`, `tests/test_loader.py`.
Modified: `stat_arb/data/loader.py`, `stat_arb/data/universe.py`, `README.md`,
`DATA.md`, `.gitignore`, `.github/workflows/tests.yml` (adds pyarrow).
Committed data: `data/universe_pit.parquet`, `universe_pit_ohlcv.parquet`
(28.5 MB), `universe_pit_ranks.parquet`. Raw pull cached in `data/raw_cmc/`
(44 MB, gitignored, resumable).

## Decisions

- Pure Python against CMC endpoints, not the `crypto2` R package. R 4.5.1 is
  installed but `crypto2` is not, and either route needed an install; Python
  keeps the repo single-language. crypto2 cited as provenance. User-approved
  2026-09-03.
- `pyarrow` 25.0.1 added (parquet). User-approved 2026-09-03. Added to CI.
- Keyed on `cmc_id` everywhere. CMC reuses tickers; BTM, ERC20 and BTCU in this
  sample are dead tokens whose symbols now belong to something else.
- Total-loss delisting applied as -99%, not -100%: the panel is log returns and
  `log(0)` is `-inf`. `pit_universe.TOTAL_LOSS_RESIDUAL`, recorded per token.
- Endpoint quirks handled in `cmc_pit.py`: history **ignores `timeStart`** and
  returns the last 400 bars before `timeEnd` (page backwards); the map endpoint
  returns first/last historical dates only for `listing_status=active`.

## Next actions

1. Sensitivity-test `MAX_ABS_LOG_RETURN` (log 5) and `TOTAL_LOSS_RESIDUAL`
   (0.01). Both are judgement calls sitting under the headline.
2. Extend the window past 24 months and the rank band past 500. Only 9.4% of
   this band died; a deeper band over a longer window is where the published
   62.19% comes from and would test whether the collapse deepens.
3. Decide whether `volume_in_usd=True` should become the default. That moves
   every published phase-1/2/3 number and `stat_arb/reporting/FINAL_REPORT.md`,
   which still carries the pre-fix figures.
4. Re-run shortability against the point-in-time universe. The current 12.6%
   figure is for the committed 174-token universe; the PIT universe is larger
   and deeper into the tail, so it can only be worse.
