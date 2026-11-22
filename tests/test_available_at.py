"""AVAILABLE_AT: availability follows received_at, never block_time."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from early_trace.evidence.ledger import load_ledger_json
from early_trace.evidence.synthetic import make_synthetic_event
from early_trace.temporal.availability import (
    AvailabilityStatus,
    ExclusionReason,
    InvalidCutoffError,
    check_availability,
)
from early_trace.temporal.snapshot import make_asof_snapshot

UTC = timezone.utc
CUTOFF = datetime(2026, 3, 1, 10, 5, tzinfo=UTC)
BLOCK = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)


def event(received_at, block_time=BLOCK, event_id="e1"):
    return make_synthetic_event(event_id, block_time=block_time, received_at=received_at)


def test_block_before_cutoff_but_received_after_is_excluded():
    late = event(datetime(2026, 3, 1, 10, 8, tzinfo=UTC))
    result = check_availability(late, CUTOFF)
    assert result.status is AvailabilityStatus.NOT_YET_AVAILABLE
    assert result.reason is ExclusionReason.RECEIVED_AFTER_CUTOFF

    snapshot = make_asof_snapshot(CUTOFF, [late])
    assert snapshot.events == ()
    assert [x.reason for x in snapshot.exclusions] == [ExclusionReason.RECEIVED_AFTER_CUTOFF]


def test_received_before_cutoff_is_available():
    result = check_availability(event(datetime(2026, 3, 1, 10, 1, tzinfo=UTC)), CUTOFF)
    assert result.is_available
    assert result.received_at_utc == datetime(2026, 3, 1, 10, 1, tzinfo=UTC)


def test_received_exactly_at_cutoff_is_available_and_one_microsecond_later_is_not():
    assert check_availability(event(CUTOFF), CUTOFF).is_available
    assert not check_availability(event(CUTOFF + timedelta(microseconds=1)), CUTOFF).is_available


def test_missing_received_at_is_unknown_never_available():
    result = check_availability(event(None), CUTOFF)
    assert result.status is AvailabilityStatus.UNKNOWN
    assert result.reason is ExclusionReason.RECEIVED_AT_MISSING
    assert not result.is_available
    assert make_asof_snapshot(CUTOFF, [event(None)]).events == ()


@pytest.mark.parametrize(
    ("received_at", "reason"),
    [
        ("2026-03-01T10:01:00", ExclusionReason.NAIVE_RECEIVED_AT),
        (datetime(2026, 3, 1, 10, 1), ExclusionReason.NAIVE_RECEIVED_AT),
        ("2026-03-01", ExclusionReason.NAIVE_RECEIVED_AT),
        ("not-a-timestamp", ExclusionReason.INVALID_RECEIVED_AT),
        ("", ExclusionReason.INVALID_RECEIVED_AT),
        (1772359260, ExclusionReason.INVALID_RECEIVED_AT),
        (1772359260.0, ExclusionReason.INVALID_RECEIVED_AT),
        (True, ExclusionReason.INVALID_RECEIVED_AT),
    ],
)
def test_invalid_received_at_is_invalid_not_utc_by_assumption(received_at, reason):
    result = check_availability(event(received_at), CUTOFF)
    assert result.status is AvailabilityStatus.INVALID
    assert result.reason is reason


@pytest.mark.parametrize(
    ("block_time", "reason"),
    [
        ("2026-03-01T10:00:00", ExclusionReason.NAIVE_BLOCK_TIME),
        ("yesterday-ish", ExclusionReason.INVALID_BLOCK_TIME),
        (1772359200, ExclusionReason.INVALID_BLOCK_TIME),
    ],
)
def test_invalid_block_time_is_invalid(block_time, reason):
    result = check_availability(event("2026-03-01T10:01:00+00:00", block_time=block_time), CUTOFF)
    assert result.status is AvailabilityStatus.INVALID
    assert result.reason is reason


def test_received_before_block_time_is_inconsistent():
    result = check_availability(event("2026-03-01T09:59:00+00:00"), CUTOFF)
    assert result.reason is ExclusionReason.RECEIVED_BEFORE_BLOCK_TIME


def test_mixed_timezones_are_normalized_to_utc():
    # 15:03+05:00 == 10:03Z (available); 06:00-05:00 == 11:00Z (after cutoff)
    available = check_availability(
        event("2026-03-01T15:03:00+05:00", block_time="2026-03-01T15:00:00+05:00"), CUTOFF
    )
    assert available.is_available
    assert available.received_at_utc == datetime(2026, 3, 1, 10, 3, tzinfo=UTC)
    assert available.received_at_utc.utcoffset() == timedelta(0)

    later = check_availability(
        event("2026-03-01T06:00:00-05:00", block_time="2026-03-01T05:00:00-05:00"), CUTOFF
    )
    assert later.reason is ExclusionReason.RECEIVED_AFTER_CUTOFF


def test_cutoff_may_use_any_offset_but_must_be_aware():
    same_instant = "2026-03-01T05:05:00-05:00"
    assert check_availability(event(CUTOFF), same_instant).is_available
    for bad in ("2026-03-01T10:05:00", datetime(2026, 3, 1, 10, 5), "garbage", None, 1772359500):
        with pytest.raises(InvalidCutoffError):
            check_availability(event(CUTOFF), bad)
        with pytest.raises(InvalidCutoffError):
            make_asof_snapshot(bad, [event(CUTOFF)])


def test_basic_fixture_admits_and_excludes_the_expected_events():
    path = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic" / "ledger_basic.json"
    snapshot = make_asof_snapshot(CUTOFF, load_ledger_json(path, require_synthetic=True))
    assert set(snapshot.event_ids) == {
        "evt-001",
        "evt-003-boundary",
        "evt-004-offset",
        "evt-005-states",
        "evt-006-external",
    }
    assert {x.event.event_id: x.reason for x in snapshot.exclusions} == {
        "evt-002-late": ExclusionReason.RECEIVED_AFTER_CUTOFF,
        "evt-007-future": ExclusionReason.RECEIVED_AFTER_CUTOFF,
    }
