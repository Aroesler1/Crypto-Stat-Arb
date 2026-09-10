"""Dated funding support for static venue matches, without asserting tradability.

A rate on the previous calendar day is historical archive evidence. It is not
a listing master, a publication timestamp, a borrow quote, or an execution price.
Missing observations stay unknown and reused normalized tickers are excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.data.perps import normalize_base


def availability_masks(membership: pd.DataFrame, universe: pd.DataFrame,
                       perps: pd.DataFrame, funding: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return static, ambiguous and prior-day-supported membership masks.

    Membership is the committed point-in-time spot bracket, before strategy
    liquidity/history filters. Venue bases use the original runner's ticker
    normalization so the static comparison measures the published assumption.
    """
    index = pd.DatetimeIndex(membership.index)
    if (index.hasnans or index.has_duplicates or not index.is_monotonic_increasing
            or index.tz is not None or not index.equals(index.normalize())):
        raise ValueError("membership requires unique ascending naive daily dates")
    if membership.columns.has_duplicates or membership.isna().any().any():
        raise ValueError("membership must have unique columns and no missing cells")
    if not membership.isin([True, False]).all().all():
        raise ValueError("membership must be boolean")
    if universe.cmc_id.duplicated().any():
        raise ValueError("duplicate cmc_id in universe")
    symbols = universe.set_index("cmc_id").symbol
    bases = symbols.map(lambda value: normalize_base(value) if isinstance(value, str) else "")
    ids = [int(col) for col in membership.columns]
    unknown_columns = np.array([cmc_id not in bases.index for cmc_id in ids])
    column_bases = bases.reindex(ids).fillna("").to_numpy()
    counts = bases[bases.ne("")].value_counts()
    ambiguous_columns = np.array([counts.get(base, 0) > 1 for base in column_bases])
    listed = set(perps.base.dropna())
    static_columns = np.array([bool(base) and base in listed for base in column_bases])
    static = membership.astype(bool) & static_columns
    ambiguous = static & ambiguous_columns

    observations = funding.loc[np.isfinite(pd.to_numeric(funding.funding_rate, errors="coerce")),
                               ["date", "base"]].copy()
    observations["date"] = pd.to_datetime(observations["date"])
    dates = pd.DatetimeIndex(observations.date)
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("funding requires naive daily dates")
    # Presence, not funding's sign, defines coverage. Multiple venues may report
    # the same base/day; one or more finite rates count as one observed day.
    observations = observations.drop_duplicates(["date", "base"])
    if observations.empty:
        supported = pd.DataFrame(False, index=index, columns=membership.columns)
    else:
        present = observations.assign(present=True).pivot(index="date", columns="base", values="present")
        lagged = present.reindex(index=index - pd.Timedelta(days=1), columns=column_bases).eq(True)
        supported = pd.DataFrame(lagged.to_numpy(), index=index, columns=membership.columns)
    supported &= static & ~ambiguous_columns
    return {"static": static, "ambiguous": ambiguous, "supported": supported,
            "unknown_identity": membership.astype(bool) & unknown_columns}


def summarize_availability(membership: pd.DataFrame, masks: dict[str, pd.DataFrame],
                           bracket: str) -> pd.DataFrame:
    """Summarize dated support by year and overall, without return estimation."""
    for mask in masks.values():
        if not mask.index.equals(membership.index) or not mask.columns.equals(membership.columns):
            raise ValueError("availability masks must align exactly to membership")
    rows = []
    periods = [("all", np.ones(len(membership), dtype=bool))]
    periods += [(str(year), membership.index.year == year) for year in sorted(set(membership.index.year))]
    for period, selector in periods:
        member = membership.loc[selector]
        if member.empty:
            continue
        static = masks["static"].loc[selector]
        supported = masks["supported"].loc[selector]
        static_days = int(static.to_numpy().sum())
        support_days = int(supported.to_numpy().sum())
        rows.append({
            "bracket": bracket, "period": period,
            "sample_start": str(member.index.min().date()),
            "sample_end": str(member.index.max().date()), "calendar_days": len(member),
            "spot_member_days": int(member.to_numpy().sum()),
            "unknown_identity_member_days": int(masks["unknown_identity"].loc[selector].to_numpy().sum()),
            "static_venue_member_days": static_days,
            "ambiguous_static_member_days": int(masks["ambiguous"].loc[selector].to_numpy().sum()),
            "prior_day_supported_member_days": support_days,
            "supported_share_of_static": support_days / static_days if static_days else np.nan,
            "avg_static_members": float(static.sum(axis=1).mean()),
            "avg_supported_members": float(supported.sum(axis=1).mean()),
            "static_days_ge_30": int(static.sum(axis=1).ge(30).sum()),
            "supported_days_ge_30": int(supported.sum(axis=1).ge(30).sum()),
        })
    return pd.DataFrame(rows)
