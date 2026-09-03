"""Point-in-time universe data from CoinMarketCap's public web API.

Why this module exists
----------------------
``data/all_tokens_24mo_daily.csv`` is a single CMC snapshot: 319 tokens whose
CMC rank sat between roughly 150 and 500 when the snapshot was taken, at the end
of the sample. Every token in it was therefore alive *and still ranked* on the
snapshot date. Tokens that were in that band during the sample and then died are
simply absent. That is survivorship bias in its purest form, and this book is
near-equal-weighted, the regime where Ammann, Burdorf, Liebi and Stoeckl
(SSRN 4287573) measure the bias at 62.19% annualised.

The README used to say that fixing this needed a paid dataset. That was wrong.
CMC serves point-in-time rank snapshots and full daily OHLCV for delisted coins
from its public web API with no key and no paid plan. These are the endpoints
the ``crypto2`` R package wraps (Stoeckl, CRAN; github.com/sstoeckl/crypto2);
this module talks to them directly so reproducing the universe needs only
Python.

Keying
------
Everything is keyed on ``cmc_id``, never on symbol. CMC reuses tickers across
projects, and this sample contains dead tokens (BTM, ERC20, BTCU) whose symbols
now belong to something else entirely. A symbol-keyed panel splices two
different assets into one price series and calls the join a return.

Endpoints, all keyless
----------------------
``map``      ``/data-api/v1/cryptocurrency/map?listing_status={active,inactive,untracked}``
``listings`` ``/data-api/v3/cryptocurrency/listings/historical?date=YYYY-MM-DD``
``history``  ``/data-api/v3.1/cryptocurrency/historical?id=<cmc_id>&convertId=2781``

Two API quirks are load-bearing and are handled here rather than by callers:

1. ``history`` **ignores ``timeStart``**. It returns the last 400 daily bars
   ending at ``timeEnd``, so long histories are paged *backwards* on ``timeEnd``.
2. ``map`` only returns ``first_historical_data`` / ``last_historical_data`` for
   ``listing_status=active``. Dead coins carry no date fields, so their listing
   life has to be derived from the history pull itself.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

BASE = "https://api.coinmarketcap.com"
# CMC's internal id for USD as a conversion target.
CONVERT_USD_ID = 2781
# The history endpoint caps a response at 400 daily bars regardless of the
# requested span; paging backwards on timeEnd is the only way to go deeper.
MAX_HISTORY_ROWS = 400
MAX_LISTING_ROWS = 1000
# CMC marks a coin dead in one of two ways. "inactive" is an explicit delisting;
# "untracked" means CMC stopped publishing it, which in this sample is the far
# more common death. Both are absent from any present-day snapshot, so both are
# survivorship for our purposes.
DEAD_STATUSES = ("inactive", "untracked")

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _UA, "Accept": "application/json"}


def _get(path: str, params: dict, timeout: int = 45, retries: int = 4,
         pause: float = 0.25) -> dict:
    """GET a CMC web-API path with bounded exponential backoff.

    The endpoint is unauthenticated and politeness-limited rather than quota
    limited, so a short fixed pause plus backoff on transient failures is
    enough; anything still failing after `retries` is raised rather than
    silently returning an empty frame.
    """
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read())
            time.sleep(pause)
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last = exc
            time.sleep(pause * (2 ** attempt))
    raise RuntimeError(f"CMC request failed after {retries} attempts: {url}") from last


def fetch_map(listing_status: str = "active", page_size: int = 5000) -> pd.DataFrame:
    """Every coin CMC knows about with the given listing status.

    Columns: cmc_id, symbol, name, slug, is_active, listing_status, and
    first_historical_data / last_historical_data where CMC supplies them (it
    does so only for active coins).
    """
    rows: list[dict] = []
    start = 1
    while True:
        payload = _get(
            "/data-api/v1/cryptocurrency/map",
            {"listing_status": listing_status, "start": start, "limit": page_size},
        )
        page = payload.get("data") or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size

    if not rows:
        return pd.DataFrame(
            columns=["cmc_id", "symbol", "name", "slug", "is_active", "listing_status"]
        )

    df = pd.DataFrame(rows).rename(columns={"id": "cmc_id"})
    df["listing_status"] = listing_status
    for col in ("first_historical_data", "last_historical_data"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
    keep = [
        c
        for c in (
            "cmc_id", "symbol", "name", "slug", "is_active", "listing_status",
            "rank", "first_historical_data", "last_historical_data",
        )
        if c in df.columns
    ]
    return df[keep].drop_duplicates(subset="cmc_id").reset_index(drop=True)


def fetch_listing_snapshot(date: str | pd.Timestamp, depth: int = 500) -> pd.DataFrame:
    """Point-in-time CMC ranking on `date`, deepest rank `depth`.

    This is the endpoint that makes the whole exercise possible: it reports the
    ranking *as it stood on that date*, so coins that have since died are still
    in it, with the rank they actually held.
    """
    day = pd.Timestamp(date).strftime("%Y-%m-%d")
    rows: list[dict] = []
    start = 1
    while start <= depth:
        limit = min(MAX_LISTING_ROWS, depth - start + 1)
        payload = _get(
            "/data-api/v3/cryptocurrency/listings/historical",
            {"convertId": CONVERT_USD_ID, "date": day, "limit": limit, "start": start},
        )
        page = payload.get("data") or []
        if not page:
            break
        rows.extend(page)
        if len(page) < limit:
            break
        start += limit

    if not rows:
        return pd.DataFrame(
            columns=["snapshot_date", "cmc_id", "symbol", "name", "slug", "rank",
                     "price_usd", "volume_24h_usd", "market_cap_usd"]
        )

    out = []
    for rec in rows:
        quote = (rec.get("quotes") or [{}])[0]
        out.append({
            "snapshot_date": pd.Timestamp(day),
            "cmc_id": rec["id"],
            "symbol": rec.get("symbol"),
            "name": rec.get("name"),
            "slug": rec.get("slug"),
            "rank": rec.get("cmcRank"),
            "price_usd": quote.get("price"),
            "volume_24h_usd": quote.get("volume24h"),
            "market_cap_usd": quote.get("marketCap"),
        })
    return pd.DataFrame(out)


def _unix(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp())


def fetch_history(cmc_id: int, start: str | pd.Timestamp,
                  end: str | pd.Timestamp) -> pd.DataFrame:
    """Daily OHLCV in USD for one `cmc_id` over [start, end].

    Pages backwards because the endpoint ignores ``timeStart`` and always
    returns the 400 bars ending at ``timeEnd``. Dead coins return their real
    history and then simply stop, which is the signal the delisting rule keys
    off; no synthetic rows are invented here.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    frames: list[pd.DataFrame] = []
    cursor = end_ts
    while cursor >= start_ts:
        payload = _get(
            "/data-api/v3.1/cryptocurrency/historical",
            {
                "id": int(cmc_id),
                "convertId": CONVERT_USD_ID,
                "timeStart": _unix(start_ts),
                "timeEnd": _unix(cursor),
                "interval": "daily",
            },
        )
        quotes = ((payload.get("data") or {}).get("quotes")) or []
        if not quotes:
            break

        page = pd.DataFrame([
            {
                "date": pd.Timestamp(q["timeOpen"][:10]),
                "open": q["quote"].get("open"),
                "high": q["quote"].get("high"),
                "low": q["quote"].get("low"),
                "close": q["quote"].get("close"),
                "volume_usd": q["quote"].get("volume"),
                "market_cap_usd": q["quote"].get("marketCap"),
            }
            for q in quotes
        ])
        frames.append(page)

        earliest = page["date"].min()
        if earliest <= start_ts or len(quotes) < MAX_HISTORY_ROWS:
            break
        # step one day past the earliest bar we already hold
        cursor = earliest - pd.Timedelta(days=1)

    if not frames:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume_usd", "market_cap_usd"]
        )

    hist = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return hist[(hist["date"] >= start_ts) & (hist["date"] <= end_ts)].reset_index(drop=True)
