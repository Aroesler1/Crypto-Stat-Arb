"""Which tokens have a listed perpetual, across the venues a US IP can reach.

The tradability verdict in this project turns on one fact: a market-neutral book
must short, and a token with no perpetual market cannot be shorted at all. So the
short-leg question is not "what does funding cost" but "does a market exist",
and only then what it costs.

Venue reachability, measured 2026-09-04 from a US IP
---------------------------------------------------
``hyperliquid``  api.hyperliquid.xyz, HTTP 200. On-chain, no geo block.
``dydx``         indexer.dydx.trade/v4, HTTP 200, 296 markets.
``deribit``      www.deribit.com/api/v2, HTTP 200, 40 perpetuals of 118 futures.
``binance``      fapi.binance.com returns **HTTP 451** to a US IP, as the repo
                 already documented. But ``data.binance.vision`` does **not**:
                 the static archive returns HTTP 200, its S3 listing enumerates
                 952 perpetual symbols, and a monthly funding file unzips to real
                 8-hourly rates. Binance is the deepest perpetual venue by a
                 wide margin, so being able to read its history without the API
                 materially changes what the shortability constraint looks like.
                 Only history is reachable this way, never live state, which is
                 all a backtest needs.

Symbol matching, and its limit
------------------------------
Venues quote a base asset, CoinMarketCap keys on ``cmc_id``. The join has to go
through the ticker, and CMC reuses tickers across projects, so a match is
evidence a perpetual exists for *a* token with that ticker, not proof it is this
one. The bias runs one way, toward overstating shortability, which makes any
finding that the universe cannot be shorted conservative. Matches are reported
with the venue that produced them so a reader can audit the join.

Quote suffixes and contract multipliers are stripped before matching:
``1000BONKUSDT`` (Binance), ``kPEPE`` (Hyperliquid), ``BTC-USD`` (dYdX) and
``AAVE_USDC-PERPETUAL`` (Deribit) all normalise to their base asset.
"""

from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import pandas as pd

HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"
DYDX_MARKETS = "https://indexer.dydx.trade/v4/perpetualMarkets"
DERIBIT_INSTRUMENTS = (
    "https://www.deribit.com/api/v2/public/get_instruments"
    "?currency=any&kind=future&expired=false"
)
BINANCE_S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BINANCE_VISION = "https://data.binance.vision"
BINANCE_FUNDING_PREFIX = "data/futures/um/monthly/fundingRate/"

VENUES = ("binance", "hyperliquid", "dydx", "deribit")

# Quote assets a venue may append to the base. Longest first so USDT is stripped
# before USD and the base is not left holding a stray "T".
_QUOTES = ("USDT", "USDC", "BUSD", "USDD", "TUSD", "USD", "PERP")
# Contract multipliers a venue prepends when a token is too cheap to quote.
_MULTIPLIER = re.compile(r"^(?:1000000|100000|10000|1000|100|10|k|K|m|M)(?=[A-Z0-9]{2,})")


def normalize_base(symbol: str) -> str:
    """Reduce a venue's contract symbol to the base asset ticker."""
    s = (symbol or "").strip().upper()
    if not s:
        return ""
    # Deribit: AAVE_USDC-PERPETUAL ; dYdX: BTC-USD ; Binance: 1000BONKUSDT
    s = s.split("-")[0].split("_")[0]
    for q in _QUOTES:
        if s.endswith(q) and len(s) > len(q):
            s = s[: -len(q)]
            break
    s = _MULTIPLIER.sub("", s)
    return s


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _post(url: str, payload: dict, timeout: int = 30) -> object:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def hyperliquid_perp_symbols() -> set[str]:
    meta = _post(HYPERLIQUID_API, {"type": "meta"})
    return {normalize_base(a["name"]) for a in meta["universe"]}


def dydx_perp_symbols() -> set[str]:
    markets = (json.loads(_get(DYDX_MARKETS)) or {}).get("markets", {})
    return {normalize_base(m) for m in markets}


def deribit_perp_symbols() -> set[str]:
    result = (json.loads(_get(DERIBIT_INSTRUMENTS)) or {}).get("result", [])
    return {normalize_base(i["instrument_name"]) for i in result
            if i.get("settlement_period") == "perpetual"}


def binance_symbol_map() -> dict[str, str]:
    """Base asset -> Binance contract symbol, from the static archive's listing.

    The funding files are addressed by contract symbol, not base asset, and the
    two differ whenever Binance applies a multiplier: BONK's funding lives under
    ``1000BONKUSDT``. Where several contracts share a base (a USDT and a USDC
    quote, say) the USDT one wins, because it is the deeper market and has the
    longer history.
    """
    out: dict[str, str] = {}
    for raw in _binance_raw_symbols():
        base = normalize_base(raw)
        if not base:
            continue
        current = out.get(base)
        if current is None or (raw.endswith("USDT") and not current.endswith("USDT")):
            out[base] = raw
    return out


def _binance_raw_symbols() -> set[str]:
    """Raw contract symbols under the funding-rate prefix."""
    out: set[str] = set()
    marker = ""
    while True:
        url = f"{BINANCE_S3}?delimiter=/&prefix={BINANCE_FUNDING_PREFIX}"
        if marker:
            url += f"&marker={urllib.parse.quote(marker)}"
        body = _get(url).decode("utf-8", "replace")
        prefixes = re.findall(
            rf"<Prefix>{re.escape(BINANCE_FUNDING_PREFIX)}([^<]+)/</Prefix>", body)
        if not prefixes:
            break
        out.update(prefixes)
        if "<IsTruncated>true</IsTruncated>" not in body:
            break
        marker = BINANCE_FUNDING_PREFIX + prefixes[-1] + "/"
    return {s.strip().upper() for s in out if s.strip()}


def binance_perp_symbols() -> set[str]:
    """Base assets with a Binance USD-M perpetual.

    Uses the archive rather than ``fapi``, which returns HTTP 451 to a US IP.
    """
    return {b for b in (normalize_base(s) for s in _binance_raw_symbols()) if b}


_FETCHERS = {
    "binance": binance_perp_symbols,
    "hyperliquid": hyperliquid_perp_symbols,
    "dydx": dydx_perp_symbols,
    "deribit": deribit_perp_symbols,
}


def perp_universe(venues: tuple[str, ...] = VENUES) -> pd.DataFrame:
    """One row per (venue, base asset). A venue that cannot be reached is
    reported as unreachable rather than silently contributing nothing."""
    rows = []
    for v in venues:
        try:
            bases = _FETCHERS[v]()
            status = "ok"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError, OSError) as exc:
            bases, status = set(), f"unreachable: {type(exc).__name__}"
            print(f"  perp venue {v}: {status}")
        for b in sorted(bases):
            rows.append({"venue": v, "base": b, "status": status})
        if status == "ok":
            print(f"  perp venue {v}: {len(bases)} perpetual bases")
    return pd.DataFrame(rows, columns=["venue", "base", "status"])


def shortable(symbols, perps: pd.DataFrame) -> pd.DataFrame:
    """For each ticker, which venues list a perpetual on it."""
    by_venue = {v: set(g["base"]) for v, g in perps.groupby("venue")}
    rows = []
    for sym in symbols:
        s = normalize_base(sym)
        hit = sorted(v for v, bases in by_venue.items() if s in bases)
        rows.append({"symbol": sym, "base": s, "n_venues": len(hit),
                     "venues": ",".join(hit), "shortable": bool(hit)})
    return pd.DataFrame(rows)


def binance_funding_monthly(symbol: str, month: str) -> pd.DataFrame:
    """One month of 8-hourly funding from the static archive. `month` is YYYY-MM.

    Returns empty on a 404, which is the normal signal that the contract did not
    exist yet, rather than an error worth stopping a pull for.
    """
    url = (f"{BINANCE_VISION}/{BINANCE_FUNDING_PREFIX}{symbol}/"
           f"{symbol}-fundingRate-{month}.zip")
    try:
        blob = _get(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return pd.DataFrame(columns=["time", "funding_rate", "interval_hours"])
        raise
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as fh:
            df = pd.read_csv(fh)
    df = df.rename(columns={"calc_time": "time", "last_funding_rate": "funding_rate",
                            "funding_interval_hours": "interval_hours"})
    df = df[pd.to_numeric(df["time"], errors="coerce").notna()]
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    return df.dropna(subset=["funding_rate"]).reset_index(drop=True)


def binance_funding_daily(symbol: str, start: str, end: str) -> pd.Series:
    """Daily funding cost for one Binance perpetual, summed from 8-hourly rates.

    Positive means longs pay shorts, so a short leg *earns* positive funding.
    Sign conventions are left to the caller; this returns the raw daily sum.
    """
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    frames = [binance_funding_monthly(symbol, str(m)) for m in months]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.Series(dtype=float, name=symbol)
    df = pd.concat(frames, ignore_index=True)
    daily = df.set_index("time")["funding_rate"].resample("D").sum()
    return daily.rename(symbol)
