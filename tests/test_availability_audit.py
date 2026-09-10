"""The archive cannot lend future listings or recycled tickers to a past book."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stat_arb.data.availability import availability_masks, summarize_availability


def _inputs():
    member = pd.DataFrame(True, index=pd.date_range("2020-01-01", periods=5), columns=["1", "2"])
    universe = pd.DataFrame({"cmc_id": [1, 2], "symbol": ["AAA", "BBB"]})
    perps = pd.DataFrame({"base": ["AAA", "BBB"]})
    funding = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-04"]),
                            "base": ["AAA", "AAA"], "funding_rate": [0.0, -0.01]})
    return member, universe, perps, funding


def test_zero_and_negative_rates_are_observed_only_the_following_day():
    member, *rest = _inputs()
    masks = availability_masks(member, *rest)
    assert masks["supported"]["1"].tolist() == [False, False, True, False, True]
    assert not masks["supported"]["2"].any()
    assert masks["static"].all().all()


def test_future_rows_do_not_change_past_availability():
    member, universe, perps, funding = _inputs()
    before = availability_masks(member, universe, perps, funding)
    extra = pd.DataFrame({"date": [pd.Timestamp("2021-01-01")], "base": ["BBB"], "funding_rate": [0.1]})
    after = availability_masks(member, universe, perps, pd.concat([funding, extra]))
    pd.testing.assert_frame_equal(before["supported"], after["supported"])


def test_calendar_day_lag_does_not_skip_missing_days():
    member, universe, perps, funding = _inputs()
    member = member.drop(pd.Timestamp("2020-01-03"))
    masks = availability_masks(member, universe, perps, funding)
    assert not masks["supported"].loc["2020-01-04", "1"]


def test_reused_ticker_is_unknown_for_both_ids():
    member, universe, perps, funding = _inputs()
    universe.loc[1, "symbol"] = "AAA"
    masks = availability_masks(member, universe, perps, funding)
    assert masks["ambiguous"].all().all()
    assert not masks["supported"].any().any()


def test_multiple_venues_count_once_and_missing_rates_do_not_count():
    member, universe, perps, funding = _inputs()
    funding = pd.concat([funding, funding, pd.DataFrame({
        "date": [pd.Timestamp("2020-01-03")], "base": ["AAA"], "funding_rate": [np.nan]})])
    masks = availability_masks(member, universe, perps, funding)
    assert masks["supported"].to_numpy().sum() == 2


def test_summary_counts_member_days_without_claiming_position_coverage():
    member, *rest = _inputs()
    masks = availability_masks(member, *rest)
    result = summarize_availability(member, masks, "B3")
    row = result[result.period.eq("all")].iloc[0]
    assert row.spot_member_days == row.static_venue_member_days == 10
    assert row.prior_day_supported_member_days == 2
    assert row.supported_share_of_static == pytest.approx(0.2)
    assert row.supported_days_ge_30 == 0


def test_empty_archive_and_zero_membership_keep_unknown_share():
    member, universe, perps, funding = _inputs()
    member[:] = False
    masks = availability_masks(member, universe, perps, funding.iloc[:0])
    result = summarize_availability(member, masks, "B0")
    assert result.supported_share_of_static.isna().all()
    assert result.prior_day_supported_member_days.eq(0).all()


def test_unknown_identity_is_counted_and_never_mapped():
    member, universe, perps, funding = _inputs()
    member = member.rename(columns={"1": "99"})
    masks = availability_masks(member, universe, perps, funding)
    assert masks["unknown_identity"]["99"].all()
    assert not masks["supported"]["99"].any()
    assert not masks["static"]["99"].any()


@pytest.mark.parametrize("defect", ["duplicate_date", "missing", "duplicate_id", "intraday", "missing_funding_date"])
def test_invalid_inputs_fail(defect):
    member, universe, perps, funding = _inputs()
    if defect == "duplicate_date":
        member = pd.concat([member, member.iloc[[0]]])
    elif defect == "missing":
        member = member.astype(object)
        member.iloc[0, 0] = None
    elif defect == "duplicate_id":
        universe.loc[1, "cmc_id"] = 1
    elif defect == "intraday":
        funding.loc[0, "date"] += pd.Timedelta(hours=1)
    else:
        funding.loc[0, "date"] = pd.NaT
    with pytest.raises(ValueError):
        availability_masks(member, universe, perps, funding)
