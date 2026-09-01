# Data provenance

**Primary source:** public cryptocurrency market data (daily OHLCV, volumes, and an ETH reference series) collected from public APIs.

This repository has **no licensed-vendor dependency**. Everything here can be redistributed, reused, and retained without restriction, which is why the input datasets are committed directly.

## What is committed

- All input datasets under `data/` (~62 MB): token OHLCV, excess log returns, correlation matrices, ETH reference series, and the token summary
- Source code, tests, and all reported results

## What is not committed

Nothing. This repository is fully self-contained and reproducible from a clone.

## Reproducing

```bash
python stat_arb/run_phase1.py
python stat_arb/run_phase3.py
python stat_arb/run_robustness.py
```

## Licence and retention

No vendor licence applies. **The binding data limitation here is not licensing but survivorship:** the universe is a CoinMarketCap snapshot and excludes dead tokens. See the README, which leads with what that does to the results.
