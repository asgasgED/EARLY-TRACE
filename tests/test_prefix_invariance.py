"""PREFIX_INVARIANCE: appending future events never changes the past."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from early_trace.evidence.events import ObservedValue
from early_trace.evidence.synthetic import make_synthetic_event
from early_trace.temporal.snapshot import make_asof_snapshot

UTC = timezone.utc
T0 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)


def minutes(n: int) -> datetime:
    return T0 + timedelta(minutes=n)


def history():
    return [
        make_synthetic_event(
            f"past-{i}",
            block_time=minutes(i),
            received_at=minutes(i + 1),
            log_index=i,
            fields={"transfer_count": ObservedValue.observed(i)},
        )
        for i in range(5)
    ]


def future(cutoff):
    return [
        # on-chain time before the cutoff, received after it
        make_synthetic_event(
            "fut-late-receipt",
            block_time=cutoff - timedelta(minutes=3),
            received_at=cutoff + timedelta(minutes=3),
            log_index=20,
        ),
        # everything after the cutoff
        make_synthetic_event(
            "fut-later",
            block_time=cutoff + timedelta(minutes=5),
            received_at=cutoff + timedelta(minutes=6),
            log_index=21,
        ),
        # later version of an already-known event_id
        make_synthetic_event(
            "past-2",
            block_time=minutes(2),
            received_at=cutoff + timedelta(minutes=9),
            log_index=2,
            source_version="v2",
            fields={"transfer_count": ObservedValue.observed(999)},
        ),
        # malformed future records
        make_synthetic_event(
            "fut-naive",
            block_time=cutoff.replace(tzinfo=None),
            received_at=(cutoff + timedelta(minutes=1)).replace(tzinfo=None),
            log_index=22,
        ),
        make_synthetic_event(
            "fut-unknown", block_time=cutoff + timedelta(minutes=2), received_at=None, log_index=23
        ),
    ]


def test_future_events_do_not_change_included_events_provenance_or_hash():
    cutoff = minutes(10)
    ledger_a = history()
    ledger_b = history() + future(cutoff)

    snap_a = make_asof_snapshot(cutoff, ledger_a)
    snap_b = make_asof_snapshot(cutoff, ledger_b)

    assert snap_a.events == snap_b.events
    assert snap_a.provenance == snap_b.provenance
    assert [e.content_hash for e in snap_a.events] == [e.content_hash for e in snap_b.events]
    assert snap_a.snapshot_hash == snap_b.snapshot_hash
    assert snap_a.conflicting_event_ids == snap_b.conflicting_event_ids == ()
    assert len(snap_b.exclusions) == len(future(cutoff))


@pytest.mark.parametrize("cutoff_minutes", [1, 2, 3, 4, 5, 6, 10])
def test_every_prefix_of_the_ledger_is_stable(cutoff_minutes):
    cutoff = minutes(cutoff_minutes)
    prefix = [e for e in history() if e.received_at <= cutoff]
    full = history() + future(cutoff)
    assert (
        make_asof_snapshot(cutoff, prefix).snapshot_hash
        == make_asof_snapshot(cutoff, full).snapshot_hash
    )


def test_snapshot_changes_only_when_the_cutoff_reveals_more_evidence():
    ledger = history()
    early = make_asof_snapshot(minutes(2), ledger)
    late = make_asof_snapshot(minutes(10), ledger)
    assert set(early.event_ids) < set(late.event_ids)
    assert early.snapshot_hash != late.snapshot_hash
    assert early.events == late.events[: len(early.events)]
