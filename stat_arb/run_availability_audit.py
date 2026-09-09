"""Reproduce bracket counts and audit dated venue support from committed data only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stat_arb.data.availability import availability_masks, summarize_availability
from stat_arb.data.brackets import BRACKET_ORDER, BRACKET_START, MIN_MEMBERS_FOR_CLUSTERING


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify outputs without writing")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    data = root / "data"
    reports = root / "stat_arb" / "reporting" / "brackets"
    universe = pd.read_parquet(data / "bracket_universe.parquet")
    funding = pd.read_parquet(data / "funding_panel.parquet")
    perps = pd.read_parquet(data / "perp_universe.parquet")
    counts = pd.read_parquet(data / "bracket_member_counts.parquet")
    assignments = pd.read_parquet(data / "bracket_assignments.parquet")
    end = pd.to_datetime(assignments.snapshot_date).max()
    tables, headlines = [], []
    for bracket in BRACKET_ORDER:
        member = pd.read_parquet(data / f"bracket_membership_{bracket}.parquet")
        # Each committed file carries pit, survivor-only and snapshot treatments.
        member = member.filter(regex=r"^pit\|").rename(columns=lambda col: col.split("|", 1)[1])
        member = member.loc[(member.index >= BRACKET_START) & (member.index <= end)]
        masks = availability_masks(member, universe, perps, funding)
        tables.append(summarize_availability(member, masks, bracket))
        cell = counts[counts.bracket.eq(bracket)]
        observed = assignments[assignments.bracket.eq(bracket)].groupby("snapshot_date").cmc_id.nunique()
        expected = cell.set_index("snapshot_date").n_members
        pd.testing.assert_series_equal(observed.reindex(expected.index, fill_value=0), expected,
                                       check_names=False, check_dtype=False)
        assert cell.clusterable.eq(cell.n_members.ge(MIN_MEMBERS_FOR_CLUSTERING)).all()
        headlines.append({
            "bracket": bracket, "first_month_end": str(cell.snapshot_date.min().date()),
            "last_month_end": str(cell.snapshot_date.max().date()), "month_ends": len(cell),
            "avg_members": float(cell.n_members.mean()), "clusterable_month_ends": int(cell.clusterable.sum()),
            "unique_tokens": int(assignments.loc[assignments.bracket.eq(bracket), "cmc_id"].nunique()),
        })
    full = pd.concat(tables, ignore_index=True)
    outputs = {
        "availability_summary.csv": full[full.period.eq("all")].reset_index(drop=True),
        "availability_by_year.csv": full[full.period.ne("all")].reset_index(drop=True),
        "headline_universe_reproduction.csv": pd.DataFrame(headlines),
    }
    for name, table in outputs.items():
        path = reports / name
        if args.check:
            actual = pd.read_csv(path, dtype={"period": str})
            pd.testing.assert_frame_equal(actual, table, check_dtype=False, atol=1e-12, rtol=1e-12)
        else:
            table.to_csv(path, index=False)
    print(f"Verified bracket headline on {len(counts)} committed month-end/bracket rows.")
    print(outputs["availability_summary.csv"].to_string(index=False))
    print("Missing funding is unknown. Dated archive support is not a tradability claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
