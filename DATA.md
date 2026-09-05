# Data provenance

**Primary source:** public cryptocurrency market data (daily OHLCV, volumes, and an ETH reference series) collected from public APIs.

This repository has **no licensed-vendor dependency**. Everything here can be redistributed, reused, and retained without restriction, which is why the input datasets are committed directly.

## What is committed

- All input datasets under `data/`: token OHLCV, excess log returns, correlation matrices, ETH reference series, the token summary, and the derived point-in-time universe tables below
- Source code, tests, and all reported results

### The point-in-time universe tables

Three derived parquet files, all keyed on `cmc_id`. They are the output of `stat_arb/build_pit_universe.py`, which pulls CoinMarketCap's public web API and needs no key.

| file | grain | contents |
|---|---|---|
| `universe_pit.parquet` | one row per `cmc_id` | `symbol`, `name`, `slug`, `first_date`, `last_date`, `delisted`, `cmc_untracked`, `prices_stop_in_window`, `delisting_rule`, `delisting_date`, `total_loss_residual`, `n_obs`, `best_rank`, `months_in_band` |
| `universe_pit_ohlcv.parquet` | one row per `(cmc_id, date)` | daily `open`, `high`, `low`, `close`, `volume_usd`, `market_cap_usd` |
| `universe_pit_ranks.parquet` | one row per `(snapshot_date, cmc_id)` | point-in-time `rank`, `price_usd`, `volume_24h_usd`, `market_cap_usd` |

The universe table and the OHLCV panel are split rather than kept in one file because a single flat table would repeat every token's metadata across ~760 daily rows. Join them on `cmc_id`.

**`cmc_id` is the key everywhere, never `symbol`.** CoinMarketCap reuses tickers across projects, and this sample contains dead tokens (BTM, ERC20, BTCU) whose symbols now belong to something else. A symbol-keyed join splices the corpse and the successor into one price series and reports the join as a return. A regression test in `tests/test_pit_universe.py` pins this.

## What is not committed

The raw CoinMarketCap pull, cached under `data/raw_cmc/` and gitignored: roughly 2,000 JSON responses turned into one parquet per `cmc_id`. It is a cache, not a source of truth; the derived tables above are what results depend on, and they are committed so the repository stays reproducible even if the endpoint changes. Delete `data/raw_cmc/` to force a fresh pull, or leave it in place so an interrupted build resumes.

## Endpoints

All keyless, all unauthenticated, all reachable from a US IP. These are the endpoints the [crypto2](https://github.com/sstoeckl/crypto2) R package (Stöckl, CRAN) wraps; `stat_arb/data/cmc_pit.py` talks to them directly so reproducing the universe needs only Python.

| purpose | endpoint |
|---|---|
| dead-coin identification | `/data-api/v1/cryptocurrency/map?listing_status={active,inactive,untracked}` |
| point-in-time rankings | `/data-api/v3/cryptocurrency/listings/historical?date=YYYY-MM-DD` |
| daily OHLCV | `/data-api/v3.1/cryptocurrency/historical?id=<cmc_id>&convertId=2781` |

Two behaviours are load-bearing and handled in `cmc_pit.py` rather than left to callers:

- the history endpoint **ignores `timeStart`** and returns the last 400 daily bars ending at `timeEnd`, so long histories must be paged *backwards*
- the map endpoint returns `first_historical_data` / `last_historical_data` only for `listing_status=active`, so a dead coin's listing life has to be derived from its history pull

CoinMarketCap marks a coin dead in two ways: `inactive` (an explicit delisting) and `untracked` (CMC stopped publishing it). In this sample every death is `untracked`. Both are invisible to a present-day snapshot, so both count as survivorship.

## Reproducing

```bash
python stat_arb/build_pit_universe.py
```

```bash
python stat_arb/run_pit_robustness.py
```

```bash
python stat_arb/run_phase1.py
```

```bash
python stat_arb/run_phase3.py
```

```bash
python stat_arb/run_robustness.py
```

`build_pit_universe.py` makes roughly 2,000 requests and takes about 20 minutes on a cold cache. Everything else runs offline from committed data.

## The bracket panel (added for the ETH-relative bracket work)

### Daily point-in-time listings

`stat_arb/build_daily_listings.py` pulls one CoinMarketCap ranking per calendar
day at depth 2,000 over 2015-07-01 to 2025-06-30: 3,653 requests, about 100
minutes at `--workers 2 --sleep 0.3`, 398 MB of raw day files.

This replaced the per-token history endpoint as the price source, and the reason
is a data defect worth recording. CoinMarketCap prunes per-token daily history
for coins that died roughly three or more years ago. Measured on a stratified
sample of 120 tokens by the year they left the brackets: 83% of 2016 deaths, 75%
of 2020 deaths and 0% of 2023 deaths return no daily history at all. A panel
built from that endpoint over 2016-2025 would be close to survivor-only in its
early years while being labelled point-in-time. The `listings/historical`
endpoint has no such gap: it accepts an arbitrary date and returns the ranking as
it stood, including coins the history endpoint has dropped.

Not committed: `data/pit_daily_listings.parquet` (169 MB, 5.9M rows) and
`data/raw_cmc/listings_d2000/`. Both are gitignored and rebuildable from the
command above. Committed instead: the derived bracket tables (48 MB), which are
what every reported number is computed from.

Return panels are stored float32. Daily log returns need nothing like float64
and the round-trip error is 2.4e-07, which halved the panels from 56.6 MB to
33.3 MB.

### Perpetual venues and funding

Venue reachability from a US IP, measured 2026-09-04:

| venue | endpoint | status | perpetual bases |
|---|---|---|---|
| Binance API | `fapi.binance.com` | **HTTP 451** | not reachable |
| Binance archive | `data.binance.vision` | HTTP 200 | 864 |
| Hyperliquid | `api.hyperliquid.xyz` | HTTP 200 | 233 |
| dYdX v4 | `indexer.dydx.trade` | HTTP 200 | 296 |
| Deribit | `www.deribit.com/api/v2` | HTTP 200 | 38 |

Binance's REST API is geo-blocked but its static archive is not, and the archive
carries the full 8-hourly funding history. That is the difference between
funding for 11 tokens and funding for 533. Only history is reachable this way,
never live state, which is all a backtest needs.

`stat_arb/build_funding_panel.py` pulls about 22,000 (contract, month) pairs,
restricted to months where the token was actually a bracket member. Raw files
cached under `data/raw_cmc/binance_funding/` (gitignored); the derived
`data/funding_panel.parquet` is committed.

### On-chain activity and attention

| source | what | licence | committed |
|---|---|---|---|
| Coin Metrics community data (`github.com/coinmetrics/data`) | active addresses, transaction counts, 1,000 assets | **CC BY-NC 4.0** | derived series only |
| Wikimedia REST pageviews API | daily English-Wikipedia pageviews | CC0 1.0 | derived series only |

**The Coin Metrics licence is more restrictive than this repository's MIT
licence.** CC BY-NC 4.0 requires attribution and permits non-commercial use
only. Any series derived from it carries that restriction; it is not relicensed
by passing through this repo. Anyone using this repository commercially must
rebuild those series from a source they are licensed for, or drop the factors
that depend on them.

Google Trends is **dropped by decision**, not by oversight. The only Python
access is an unofficial scrape of an endpoint Google does not document or
support, it rate-limits aggressively, and its values are renormalised per query
so two pulls are not comparable to each other. That is not a reproducible
source. Attention is measured with Wikipedia pageviews instead.

Coverage is the binding constraint on both. Coin Metrics covers a few hundred
assets against the 3,210 tokens that pass through B3, and Wikipedia has an
article for fewer still. Factor rows that cannot be computed for enough of the
cross-section stay in the table marked "no data" with the reason, rather than
being computed on whatever subset happens to have coverage and reported as if
it were the whole universe.

## Licence and retention

No vendor licence applies.

The survivorship limitation this file used to describe as binding, "the universe is a CoinMarketCap snapshot and excludes dead tokens", is now measured rather than assumed. It is worth a net Sharpe of 2.02 against -0.14. See the README, which leads with it.
