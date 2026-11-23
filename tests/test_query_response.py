"""Task 03 (ET24 §6.6): query response lifecycle relative to the snapshot."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from early_trace.evidence.events import ObservedValue
from early_trace.evidence.synthetic import make_synthetic_event
from early_trace.temporal.availability import TimestampError
from early_trace.temporal.query_response import (
    QueryResponse,
    ResponseStatus,
    ResponseTimelineError,
    ValidationOutcome,
    classify_response,
    incorporable,
)
from early_trace.temporal.snapshot import make_asof_snapshot

T0 = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)


def t(minute: int) -> datetime:
    return T0 + timedelta(minutes=minute)


def response(
    *, received=3, end=5, deadline=None, validation=ValidationOutcome.VALID, new=2, sent=1
):
    return QueryResponse.create(
        "q-1",
        t_query_sent=t(sent),
        t_query_received=None if received is None else t(received),
        t_feature_update_end=None if end is None else t(end),
        deadline=None if deadline is None else t(deadline),
        validation=validation,
        new_object_count=new if validation is ValidationOutcome.VALID else None,
    )


def test_not_received_response():
    r = response(received=None, end=None, validation=ValidationOutcome.PENDING)
    assert classify_response(r, t(60)) is ResponseStatus.NOT_RECEIVED
    assert not incorporable(r, t(60))


def test_received_but_not_yet_validated():
    r = response(end=None, validation=ValidationOutcome.PENDING)
    assert classify_response(r, t(4)) is ResponseStatus.RECEIVED_PENDING_VALIDATION
    assert not incorporable(r, t(4))


def test_the_same_response_changes_status_only_as_time_passes():
    r = response(received=3, end=5)
    assert classify_response(r, t(2)) is ResponseStatus.NOT_RECEIVED
    assert classify_response(r, t(3)) is ResponseStatus.RECEIVED_PENDING_VALIDATION
    assert classify_response(r, t(4)) is ResponseStatus.RECEIVED_PENDING_VALIDATION
    assert classify_response(r, t(5)) is ResponseStatus.VALIDATED
    assert incorporable(r, t(5)) and not incorporable(r, t(4))


def test_validated_after_the_snapshot_is_unknown_to_the_snapshot():
    snapshot_at = t(4)  # received (3) but validated only at 5
    assert (
        classify_response(response(received=3, end=5), snapshot_at)
        is ResponseStatus.RECEIVED_PENDING_VALIDATION
    )


def test_invalid_response_is_never_incorporable():
    r = response(validation=ValidationOutcome.INVALID)
    assert classify_response(r, t(10)) is ResponseStatus.RECEIVED_INVALID
    assert not incorporable(r, t(10))


def test_valid_response_without_new_objects():
    r = response(new=0)
    assert classify_response(r, t(10)) is ResponseStatus.VALIDATED_NO_NEW_OBJECTS
    assert incorporable(r, t(10))


def test_late_response_is_not_usable_even_if_valid():
    r = response(received=8, end=9, deadline=6)
    assert classify_response(r, t(20)) is ResponseStatus.RECEIVED_AFTER_DEADLINE
    assert not incorporable(r, t(20))
    assert classify_response(r, t(7)) is ResponseStatus.NOT_RECEIVED  # not yet known at 7


def test_on_time_response_with_a_deadline_is_usable():
    r = response(received=3, end=5, deadline=6)
    assert classify_response(r, t(10)) is ResponseStatus.VALIDATED


def test_a_late_response_cannot_change_the_initial_snapshot():
    snapshot_at = t(5)
    known = make_synthetic_event(
        "known", block_time=t(0), received_at=t(2), fields={"v": ObservedValue.observed(1)}
    )
    late_batch = [  # what a late response delivers: received after the snapshot
        make_synthetic_event(
            f"late-{i}",
            block_time=t(0),
            received_at=t(8),
            log_index=i + 1,
            fields={"v": ObservedValue.observed(5)},
        )
        for i in range(3)
    ]
    before = make_asof_snapshot(snapshot_at, [known])
    after = make_asof_snapshot(snapshot_at, [known, *late_batch])
    assert before.snapshot_hash == after.snapshot_hash
    assert before.objects == after.objects
    assert len(after.exclusions) == 3
    assert make_asof_snapshot(t(10), [known, *late_batch]).snapshot_hash != before.snapshot_hash


@pytest.mark.parametrize(
    "build",
    [
        lambda: response(received=0, sent=1),  # received before sent
        lambda: response(received=None, end=5, validation=ValidationOutcome.INVALID),
        lambda: response(received=3, end=2),  # processing ends before reception
        lambda: response(end=5, validation=ValidationOutcome.PENDING),
        lambda: response(end=None, validation=ValidationOutcome.VALID),
        lambda: response(new=-1),
    ],
)
def test_impossible_timelines_are_rejected(build):
    with pytest.raises(ResponseTimelineError):
        build()


def test_naive_response_timestamps_are_rejected_not_assumed_utc():
    with pytest.raises(ResponseTimelineError):
        QueryResponse.create(
            "q",
            t_query_sent="2026-03-01T10:00:00",
            t_query_received=None,
            t_feature_update_end=None,
            deadline=None,
            validation=ValidationOutcome.PENDING,
            new_object_count=None,
        )
    with pytest.raises(TimestampError):
        classify_response(response(), "2026-03-01T10:05:00")
