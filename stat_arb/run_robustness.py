"""
Robustness checks for the survivorship caveat.

The token universe is a CoinMarketCap snapshot, so dead and delisted tokens are
absent. Ammann, Burdorf, Liebi and Stoeckl ("Survivorship and Delisting Bias in
Cryptocurrency Markets", 3,904 coins, 2014-2021) measure the resulting bias at
0.93% annualised for value-weighted and 62.19% for equal-weighted buy-and-hold
portfolios, and find that momentum and market beta lose any positive relation to
returns once delisting returns are included.

Buying point-in-time listing history would settle this directly. Absent that,
two checks bound the problem without new data:

1. BREAKEVEN BIAS. How large an annualised return drag would erase the result?
   Since Sharpe is mean/vol, the drag that takes net Sharpe to zero is exactly
   the strategy's annualised net return. Reporting it converts "results may be
   biased" into a falsifiable number a reader can compare against the published
   magnitudes.

   One caveat has to travel with that comparison: the 62.19% figure is for a
   LONG-ONLY equal-weighted buy-and-hold book, where survivorship removes the
   losers outright. This strategy is dollar- and cluster-neutral, so a dead
   token missing from the universe removes both potential longs and potential
   shorts. The long-only figure is therefore an upper bound on plausible drag,
   not an estimate of it.

2. LIQUIDITY TIER. Delisting risk concentrates in the smallest, least traded
   tokens. Re-running with a materially higher minimum-volume floor tests
   whether the effect survives where survivorship matters least. The floor is
   applied through UniverseManager, which filters point-in-time at each
   reconstitution, so this does not smuggle in forward-looking information the
   way a market-cap snapshot would.

Usage:
    python stat_arb/run_robustness.py
"""
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from stat_arb.data.loader import DataLoader
from stat_arb.data.universe import UniverseManager
from stat_arb.backtest.statistics import deflated_sharpe_ratio, per_period_sharpe
from stat_arb.run_phase3 import run_phase3_config

PERIODS_PER_YEAR = 365

# Best net configuration from the phase-3 grid.
BEST_BAND = 0.02
BEST_FREQ = 3

# Published survivorship magnitudes (Ammann/Burdorf/Liebi/Stoeckl).
PUBLISHED_EW_BIAS = 0.6219
PUBLISHED_VW_BIAS = 0.0093

# Baseline phase-3 floor, and a tier where delisting risk is far lower.
VOLUME_TIERS = {
    "baseline (>=$50k/day)": 50_000,
    "liquid (>=$1M/day)": 1_000_000,
    "very liquid (>=$5M/day)": 5_000_000,
}


def annualised_stats(returns: pd.Series) -> dict:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "n": len(r)}
    ann_return = float(r.mean()) * PERIODS_PER_YEAR
    ann_vol = float(r.std(ddof=1)) * np.sqrt(PERIODS_PER_YEAR)
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol > 0 else np.nan,
        "n": len(r),
    }


def run_tier(excess_returns, prices, volumes, eth_data, min_volume_usd: float) -> dict | None:
    univ = UniverseManager(
        mcap_percentile_low=0.0,
        mcap_percentile_high=1.0,
        min_volume_usd=min_volume_usd,
        min_history_days=60,
    )
    mask = univ.get_universe_membership(prices, volumes, eth_data, excess_returns)
    members = int(mask.sum(axis=1).mean()) if not mask.empty else 0

    result = run_phase3_config(
        excess_returns, mask, weight_band=BEST_BAND, trade_frequency_days=BEST_FREQ
    )
    if result is None:
        return None

    stats = annualised_stats(result["net_50"])
    stats["avg_members"] = members
    stats["gross_sharpe"] = annualised_stats(result["gross_returns"])["sharpe"]
    stats["avg_turnover"] = float(result["turnover"].mean())
    stats["breakeven_cost_bps"] = result["breakeven"]
    # the annualised drag that would take net Sharpe to zero
    stats["breakeven_bias"] = stats["ann_return"]
    stats["net_50"] = result["net_50"]
    return stats


def main() -> int:
    data_dir = Path(__file__).parent.parent / "data"
    loader = DataLoader(str(data_dir))
    excess_returns, prices, volumes, eth_data = loader.get_aligned_data()

    rows = []
    series = {}
    for label, floor in VOLUME_TIERS.items():
        print(f"running tier: {label} ...", flush=True)
        stats = run_tier(excess_returns, prices, volumes, eth_data, floor)
        if stats is None:
            print(f"  {label}: produced no positions, skipped")
            continue
        series[label] = stats.pop("net_50")
        stats["tier"] = label
        rows.append(stats)

    if not rows:
        print("no tier produced a result")
        return 1

    table = pd.DataFrame(rows).set_index("tier")

    # DSR across the tiers actually evaluated here
    trial_srs = [per_period_sharpe(s) for s in series.values()]
    table["dsr_net_50"] = [
        deflated_sharpe_ratio(series[t], n_trials=len(trial_srs), trial_sharpes=trial_srs)["dsr"]
        for t in table.index
    ]

    out_dir = Path(__file__).parent / "reporting" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "survivorship_robustness.csv")

    show = table[
        ["avg_members", "gross_sharpe", "sharpe", "ann_return", "ann_vol",
         "avg_turnover", "breakeven_cost_bps", "breakeven_bias", "dsr_net_50"]
    ]
    print("\nSURVIVORSHIP ROBUSTNESS (config: band "
          f"{BEST_BAND:.0%}, rebalance {BEST_FREQ}d, net of 50bps)")
    print(show.to_string(float_format=lambda v: f"{v:0.3f}"))

    print("\nBreakeven bias = annualised drag that would take net Sharpe to zero.")
    print(f"Published buy-and-hold survivorship bias: "
          f"{PUBLISHED_EW_BIAS:.1%} equal-weighted, {PUBLISHED_VW_BIAS:.2%} value-weighted.")
    print("The equal-weighted figure is a LONG-ONLY upper bound; a dollar-neutral")
    print("book loses both potential longs and potential shorts when a token dies,")
    print("so the realised drag here is expected to fall well below it.")
    for tier, row in table.iterrows():
        verdict = "SURVIVES" if row["breakeven_bias"] > PUBLISHED_VW_BIAS else "does not survive"
        print(f"  {tier:<26s} breakeven bias {row['breakeven_bias']:6.1%}  "
              f"vs VW {PUBLISHED_VW_BIAS:.2%} -> {verdict}")
    print(f"\nsaved -> {out_dir / 'survivorship_robustness.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
