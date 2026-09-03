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

**`cmc_id` is the key everywhere, never `symbol`.** CoinMarketCap reuses tickers across projects, and this sample contains dead tokens — BTM, ERC20, BTCU — whose symbols now belong to something else. A symbol-keyed join splices the corpse and the successor into one price series and reports the join as a return. A regression test in `tests/test_pit_universe.py` pins this.

## What is not committed

The raw CoinMarketCap pull, cached under `data/raw_cmc/` and gitignored: roughly 2,000 JSON responses turned into one parquet per `cmc_id`. It is a cache, not a source of truth — the derived tables above are what results depend on, and they are committed so the repository stays reproducible even if the endpoint changes. Delete `data/raw_cmc/` to force a fresh pull, or leave it in place so an interrupted build resumes.

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

## Licence and retention

No vendor licence applies.

The survivorship limitation this file used to describe as binding — "the universe is a CoinMarketCap snapshot and excludes dead tokens" — is now measured rather than assumed. It is worth a net Sharpe of 2.02 against −0.14. See the README, which leads with it.
