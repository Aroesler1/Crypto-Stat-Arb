"""On-chain and attention data for factors the price panel cannot produce.

Two factors in the crypto asset-pricing literature need data CoinMarketCap's
listing endpoint does not carry.

**Value / network activity.** Cong, Karolyi, Tang and Zhao construct a value
factor from network usage: active addresses and transaction counts against
market cap, the crypto analogue of a book-to-market ratio. Source here is Coin
Metrics' community data (github.com/coinmetrics/data), which publishes one CSV
per asset with ``AdrActCnt`` and ``TxCnt`` among many other columns.

**Attention.** Wikipedia pageviews through the official Wikimedia REST API.
Google Trends is deliberately not used: the only Python access is an unofficial
scrape of an endpoint Google does not document or support, it rate-limits
aggressively, and its values are renormalised per query so two pulls are not
comparable. That is not a reproducible source, so attention is measured with
pageviews and the limitation is stated rather than papered over.

Licensing, which constrains what may be committed
-------------------------------------------------
Coin Metrics community data is **CC BY-NC 4.0**: attribution required, and
**non-commercial use only**. That is more restrictive than this repository's MIT
licence, so the derived series carry the Coin Metrics restriction with them and
the distinction is recorded in DATA.md rather than being silently flattened into
the repo's own terms. Wikimedia pageviews are CC0.

Coverage is the binding constraint on both. Coin Metrics community data covers a
few hundred assets, heavily weighted to the ones that matter, and Wikipedia has
an article for a small minority of the 3,210 tokens that pass through B3. Any
factor row that cannot be computed for enough of the cross-section stays in the
table marked "no data" with the reason.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

COINMETRICS_RAW = "https://raw.githubusercontent.com/coinmetrics/data/master/csv"
COINMETRICS_API = "https://api.github.com/repos/coinmetrics/data/contents/csv"
WIKIMEDIA = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
             "en.wikipedia/all-access/user/{article}/daily/{start}/{end}")

COINMETRICS_LICENCE = "CC BY-NC 4.0 (attribution, non-commercial)"
WIKIMEDIA_LICENCE = "CC0 1.0"

# Columns worth keeping: the two the value factor needs, plus the price Coin
# Metrics itself used, which is a useful cross-check on the CMC panel.
COINMETRICS_COLUMNS = ("time", "AdrActCnt", "TxCnt", "CapMrktCurUSD", "PriceUSD")

_HEADERS = {"User-Agent": "crypto-stat-arb-research/1.0 (academic use)"}


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def coinmetrics_assets() -> list[str]:
    """Asset tickers with a community-data CSV."""
    payload = json.loads(_get(COINMETRICS_API))
    return sorted(
        entry["name"][:-4].lower()
        for entry in payload
        if isinstance(entry, dict) and entry.get("name", "").endswith(".csv")
    )


def fetch_coinmetrics(asset: str) -> pd.DataFrame:
    """Daily network activity for one asset. Empty frame if it is not covered."""
    try:
        blob = _get(f"{COINMETRICS_RAW}/{asset.lower()}.csv")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return pd.DataFrame(columns=list(COINMETRICS_COLUMNS))
        raise
    df = pd.read_csv(io.BytesIO(blob), low_memory=False)
    keep = [c for c in COINMETRICS_COLUMNS if c in df.columns]
    df = df[keep].rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "asset", asset.lower())
    return df.dropna(subset=["date"]).reset_index(drop=True)


def fetch_pageviews(article: str, start: str, end: str) -> pd.DataFrame:
    """Daily English-Wikipedia pageviews for one article.

    `article` is the exact page title with underscores, as Wikipedia stores it.
    A 404 means no such article, which for most of this universe is the normal
    answer rather than an error.
    """
    url = WIKIMEDIA.format(
        article=urllib.parse.quote(article.replace(" ", "_"), safe=""),
        start=pd.Timestamp(start).strftime("%Y%m%d"),
        end=pd.Timestamp(end).strftime("%Y%m%d"))
    try:
        payload = json.loads(_get(url, timeout=30))
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 400):
            return pd.DataFrame(columns=["date", "article", "views"])
        raise
    items = payload.get("items") or []
    if not items:
        return pd.DataFrame(columns=["date", "article", "views"])
    return pd.DataFrame({
        "date": [pd.Timestamp(i["timestamp"][:8]) for i in items],
        "article": article,
        "views": [int(i.get("views", 0)) for i in items],
    })


def network_value_ratio(activity: pd.DataFrame, mcap: pd.DataFrame,
                        column: str = "AdrActCnt") -> pd.DataFrame:
    """Network activity divided by market cap: the crypto book-to-market analogue.

    High means a lot of on-chain use per dollar of valuation, which is the value
    side of the sort. Returns a panel keyed like `mcap`, so it drops straight
    into the characteristic machinery.
    """
    wide = activity.pivot_table(index="date", columns="asset", values=column,
                                aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    out = pd.DataFrame(index=mcap.index, columns=mcap.columns, dtype=float)
    for col in mcap.columns:
        key = str(col).lower()
        if key in wide.columns:
            series = wide[key].reindex(mcap.index).ffill(limit=7)
            out[col] = series / mcap[col].where(mcap[col] > 0)
    return out
