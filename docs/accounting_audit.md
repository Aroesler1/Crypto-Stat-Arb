# Capital accounting and execution scope

The original backtest multiplied portfolio weights by ETH-relative log returns. A short in an asset losing 99% receives a log score +4.605 rather than simple-price P&L +.99. This is a measured arithmetic failure, not a statistical dispute. The corrected ledger explicitly tracks cash, dollar holdings, actual trades, fees, funding and equity. Fixed quantities drift between scheduled trades. Targets use equity before trading; fees leave cash immediately. No interest, borrow, margin or intraday funding marks are inferred. Funding uses daily opening effective exposure as a proxy; missing rates remain unmeasured.

The default rejects missing held returns. Frozen historical replications explicitly retain the stale-mark policy and publish missing-return exposure shares. Those are diagnostic spot-return proxies, not investable perpetual strategies. The new ledger raises a structured insolvency result, with its terminal daily return preserved; it never clips losses or restarts wealth.

## Confirmed integration repairs

- Empty eligible universes stay empty; the enforced floor is 30 names. Universe observations are taken strictly before the first earned return.
- A failed walk-forward fold cannot silently drop part of the evaluation calendar.
- The death arm refuses to run without its filter; tradability constructs and passes the filter when requested.
- Funding helpers consume already-effective weights, removing the extra lag. A funded ledger lets funding affect future equity and trading.
- The old linear break-even estimate is not emitted for a dynamic holdings ledger: fees change future equity and trades.
- Updated research entry points use separate corrected filenames and require raw simple capital returns. Historical tables remain available, but their strategy performance interpretations are withdrawn.

## Fixed measured replication

Three historical brackets by two pre-existing signals were chosen before the correction run, without optimizing the new results. `run_accounting_audit.py` rebuilds them from the existing cached panel; --check uses only committed portfolio-level CSVs. The B3 baseline fails on 2017-12-08 and B3 EWMA on 2020-03-14 at 50 bps. These failures concern this ledger and spot-price proxy, not a reconstruction of actual exchange liquidation. The B1/B2 surviving results and missing exposure are listed side by side with historical scores. No new multiple-testing-adjusted discovery is claimed.

The existing eight-arm DSR is not the denominator for the cumulative reference, method and parameter search. Static exchange matches do not establish historical contract dates or borrow availability. The availability review remains applicable. Licence wording was not changed without the separately requested approval.

## Literature-led next experiments

[SPONGE, AISTATS 2019](https://proceedings.mlr.press/v89/cucuringu19a.html) motivates using both edge signs. An independently frozen comparison should hold exposure, turnover and data admission constant when testing whether signed graphs improve net convergence performance. It does not follow from graph quality alone.

The [2026 point-in-time BTC archive audit](https://arxiv.org/html/2608.25348v1) motivates data admission before alpha search. That single-contract experiment cannot validate an altcoin instrument master. First establish dated identities and settlement-aligned funding; then assess a fixed strategy.

A prospective cluster-stability filter and a prospective exposure/liquidation policy are reasonable hypotheses after these failures. Selecting either to rescue the same known sample would be another research trial. No improved trading claim is made here.
