# Historical instrument support audit, 2026-09-06

## Scope and protocol

Four ETH-relative spot brackets, 3,469 calendar days from 2016-01-01 to
2025-06-30. The point-in-time membership panels precede strategy-specific
liquidity and history filters. This diagnostic was specified before computing
its tables, after inspecting the existing static venue and funding files.
It is a retrospective data-quality hypothesis, not a new investment test:
do the static venue matches have dated, unambiguous prior-day funding support?

Use the original perpetual-base normalization. Exclude normalized symbols
shared by multiple permanent ids anywhere in the committed metadata snapshot.
Retain missing identities as unknown. Require a finite funding observation
on the preceding calendar day, with no forward fill or future lookup. Zero
and negative rates count. Multiple observations of a base/day count once.
This conservative identity screen uses a later metadata snapshot; it is not a
point-in-time listing master, and the archive has no publication-time field.

## Result and limitations

B3 has 363,090 statically matched member-days, but only 105,193 (28.9716%)
pass the dated identity screen. Average membership falls from 104.67 static
matches to 30.32 supported matches. At least 30 supported names exist on
1,164 of 3,469 days, with no support in 2016-2019. There are also 1,946 spot
member-days without an id in the committed metadata. Unsupported observations
remain unknown, not proof that trading was impossible at every venue.

The support hypothesis fails. The earlier +1.44 Sharpe remains the result of
a static venue subset with partially observed funding. The 28.97% member-day
coverage is a different denominator from its 13% exposure-weighted coverage.
Signed missing funding set to zero gives neither a cost nor a Sharpe bound.
No returns were rerun, no improved specification was selected, and existing
multiple-testing results were not replaced.

`python stat_arb/run_availability_audit.py --check` reproduces the four opening
universe rows from 456 committed month-end/bracket records, then verifies:

- `availability_summary.csv`: four bracket summaries;
- `availability_by_year.csv`: 40 bracket/year rows, with only 181 days in 2025;
- `headline_universe_reproduction.csv`: four independently reaggregated rows.

All three files are in `stat_arb/reporting/brackets/`. Inputs are existing
membership, assignment, count, universe, perpetual and funding parquet files.
There is no new source, pull, raw-data export or WRDS dependency.

## Primary-source checks through 2026-09-06

- [Zeng, Yang, Han and He (2026), Point-in-Time Audit Before Alpha](https://arxiv.org/html/2608.25348v1):
  Binance BTCUSDT USD-M public archives, five-minute decisions. A strict
  all-stream requirement failed at 304.57 continuous days; a disclosed revision
  made open interest optional and retained 727 complete UTC days, split
  436/145/146 with purge and embargo. The authors report successful finite
  template checks, reduced false passes under null controls and no profitable
  historical-holdout factor. The revised sample is a different experiment,
  not confirmation of the original requirement. This supports auditing data
  admission before optimization. One BTC contract cannot validate this
  repository's cross-section of altcoins or reconstruct their listing dates.
- [Borri, Liu, Tsyvinski and Wu (2026), Coming of Age, version 4](https://arxiv.org/html/2510.14435v4):
  the carry calculation uses Bitcoin spot and perpetual prices plus funding
  from Binance at eight-hour frequency, 2020-08-01 to 2025-05-31. Its reported
  annualized carry Sharpe is 6.45 overall, 4.06 from 2024 and negative in the
  partial 2025 sample. This is evidence of variation in that sample, not proof
  that every altcoin funding regime permanently ended. The same paper's broad
  weekly coin panel is a separate study, not the carry denominator.
- [Binance USD-M market-data documentation](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data):
  API methodology, no empirical sample. Funding history supplies a rate and
  settlement timestamp. It does not supply a permanent CoinMarketCap identity,
  prove publication time for an archived daily aggregate, or establish all
  venues' historical contract availability. Prior-calendar-day support is
  therefore explicitly weaker than execution eligibility.

These targeted papers motivate the selected diagnostic. They do not establish
that the best next action is another signal search. A fresh replication needs
dated contract identities and settlement-aligned position/funding accounting
under a protocol fixed before evaluating returns.
