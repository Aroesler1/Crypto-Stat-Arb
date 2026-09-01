"""Perpetual funding costs and shortability for the token universe.

A market-neutral book pays to hold its short leg. This repository previously
approximated that with `BacktestEngine.carry_bps_daily`, a uniform stress knob,
because per-token funding was not wired in. This fetches the real thing.

Source is Hyperliquid's public API rather than Binance. That is not a
preference: Binance's futures endpoints return HTTP 451 to US IP addresses, as
do several other centralised venues, so a US-based researcher cannot reproduce a
Binance-sourced funding series at all. Hyperliquid is an on-chain perpetual
venue with an open API and no such restriction, which makes the result
reproducible by anyone reading this.

The fetch turned up something more consequential than the cost itself, so it is
measured explicitly: **most of this universe has no perpetual market**. A
dollar-neutral strategy cannot take the short side of a token nobody lists a
perp for, and shortability is therefore a hard constraint on implementability
rather than a cost adjustment.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

API = "https://api.hyperliquid.xyz/info"
# Hyperliquid funds hourly and returns at most 500 rows per request.
_MAX_ROWS = 500
_HOURS_MS = 3_600_000
FUNDING_INTERVALS_PER_DAY = 24


def _post(payload: dict, timeout: int = 25) -> object:
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def list_perp_symbols() -> set[str]:
    """Uppercased names of every perpetual currently listed."""
    meta = _post({"type": "meta"})
    return {a["name"].upper() for a in meta["universe"]}


@dataclass
class ShortabilityReport:
    universe: list[str]
    shortable: list[str]
    missing: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return len(self.shortable) / len(self.universe) if self.universe else float("nan")


def assess_shortability(tokens: list[str]) -> ShortabilityReport:
    """Which universe tokens have a perpetual market at all."""
    listed = list_perp_symbols()
    upper = [str(t).upper() for t in tokens]
    shortable = [t for t in upper if t in listed]
    missing = [t for t in upper if t not in listed]
    return ShortabilityReport(universe=upper, shortable=shortable, missing=missing)


def fetch_funding(
    coin: str,
    start_ms: int,
    end_ms: int,
    pause: float = 0.12,
    max_pages: int = 60,
) -> pd.DataFrame:
    """Hourly funding rates for one coin over [start_ms, end_ms].

    Paginates forward from `start_ms`; the endpoint caps each response, so the
    cursor advances to just after the last row returned. `max_pages` bounds the
    walk so a stalled cursor cannot loop forever.
    """
    rows: list[dict] = []
    cursor = int(start_ms)
    for _ in range(max_pages):
        try:
            page = _post({"type": "fundingHistory", "coin": coin, "startTime": cursor})
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            break
        if not page:
            break
        rows.extend(page)
        last = int(page[-1]["time"])
        if last >= end_ms or len(page) < _MAX_ROWS:
            break
        cursor = last + 1
        time.sleep(pause)

    if not rows:
        return pd.DataFrame(columns=["time", "funding_rate"])
    frame = pd.DataFrame(rows)
    frame["time"] = pd.to_datetime(frame["time"], unit="ms")
    frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    frame = frame[(frame["time"] >= pd.Timestamp(start_ms, unit="ms"))
                  & (frame["time"] <= pd.Timestamp(end_ms, unit="ms"))]
    return frame[["time", "funding_rate"]].dropna().reset_index(drop=True)


def daily_funding_panel(
    tokens: list[str],
    start: str,
    end: str,
    pause: float = 0.12,
) -> pd.DataFrame:
    """Date x token panel of daily funding cost, summed from hourly rates.

    A positive rate means longs pay shorts, so a SHORT position EARNS it. The
    sign convention is left as published; the backtest applies it against the
    signed position rather than assuming a direction here.
    """
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)

    series: dict[str, pd.Series] = {}
    for token in tokens:
        frame = fetch_funding(token, start_ms, end_ms, pause=pause)
        if frame.empty:
            continue
        daily = frame.set_index("time")["funding_rate"].resample("1D").sum()
        series[token.upper()] = daily
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def summarise(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-token annualised funding, in the units a cost model wants."""
    if panel.empty:
        return pd.DataFrame()
    daily_mean = panel.mean(axis=0)
    return pd.DataFrame({
        "days": panel.notna().sum(axis=0),
        "mean_daily_rate": daily_mean,
        "annualised_pct": daily_mean * 365 * 100.0,
        "mean_daily_bps": daily_mean * 1e4,
        "vol_daily_bps": panel.std(axis=0) * 1e4,
    }).sort_values("annualised_pct", ascending=False)
