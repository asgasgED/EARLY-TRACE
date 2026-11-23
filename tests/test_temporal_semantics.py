"""Task 03: block time, reception time and cutoff are independent clocks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from early_trace.evidence.events import SourceType
from early_trace.evidence.synthetic import make_synthetic_event
from early_trace.temporal.availability import (
    AnomalyKind,
    AnomalyStatus,
    AvailabilityStatus,
    ExclusionReason,
    SourceClockRule,
    TemporalPolicy,
    check_availability,
)
from early_trace.temporal.snapshot import make_asof_snapshot

UTC = timezone.utc
CUTOFF = datetime(2026, 3, 1, 10, 30, tzinfo=UTC)
BLOCK = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)


def make(event_id, *, received_at, block_time=BLOCK, kind=SourceType.ONCHAIN_LOG, **kwargs):
    return make_synthetic_event(
        event_id, block_time=block_time, received_at=received_at, source_type=kind, **kwargs
    )


def early(seconds: float, **kwargs):
    return make("early", received_at=BLOCK - timedelta(seconds=seconds), **kwargs)


def test_received_before_block_time_is_admitted_with_a_recorded_anomaly():
    snapshot = make_asof_snapshot(CUTOFF, [early(5)])
    (admitted,) = snapshot.events
    (anomaly,) = admitted.anomalies
    assert anomaly.kind is AnomalyKind.RECEIVED_BEFORE_BLOCK_TIME
    assert anomaly.status is AnomalyStatus.PENDING_RESOLUTION
    assert anomaly.early_by_seconds == 5.0
    assert "no source clock rule" in anomaly.policy
    assert snapshot.anomalies == ((admitted.observation_hash, anomaly),)
    assert snapshot.to_dict()["anomalies"][0]["status"] == "pending_resolution"


def test_both_original_timestamps_are_preserved_in_the_provenance():
    snapshot = make_asof_snapshot(CUTOFF, [early(5)])
    record = snapshot.events[0].provenance
    assert record.block_time_utc == BLOCK
    assert record.received_at_utc == BLOCK - timedelta(seconds=5)


def test_block_time_after_cutoff_does_not_exclude_an_observation_received_before_it():
    # availability follows received_at only; t_block is an independent clock
    event = make(
        "e", received_at=CUTOFF - timedelta(minutes=1), block_time=CUTOFF + timedelta(minutes=5)
    )
    result = check_availability(event, CUTOFF)
    assert result.is_available and result.anomalies[0].early_by_seconds == 360.0


def test_ordinary_events_carry_no_anomaly():
    snapshot = make_asof_snapshot(CUTOFF, [make("ok", received_at=BLOCK + timedelta(seconds=30))])
    assert snapshot.events[0].anomalies == () and snapshot.anomalies == ()


def test_an_explicit_source_rule_can_prove_the_record_impossible():
    policy = TemporalPolicy((SourceClockRule("synthetic-provider-a", timedelta(seconds=2)),))
    result = check_availability(early(5), CUTOFF, policy)
    assert result.status is AvailabilityStatus.INVALID
    assert result.reason is ExclusionReason.RECEIVED_BEFORE_BLOCK_TIME
    assert "declared tolerance of 2.0s" in result.detail

    snapshot = make_asof_snapshot(CUTOFF, [early(5)], policy)
    assert snapshot.events == ()
    assert snapshot.exclusions[0].reason is ExclusionReason.RECEIVED_BEFORE_BLOCK_TIME


def test_within_the_declared_tolerance_the_anomaly_is_flagged_not_excluded():
    policy = TemporalPolicy((SourceClockRule("synthetic-provider-a", timedelta(seconds=10)),))
    (admitted,) = make_asof_snapshot(CUTOFF, [early(5)], policy).events
    assert admitted.anomalies[0].status is AnomalyStatus.WITHIN_DECLARED_TOLERANCE


def test_a_rule_for_another_source_does_not_apply():
    policy = TemporalPolicy((SourceClockRule("some-other-provider", timedelta(0)),))
    (admitted,) = make_asof_snapshot(CUTOFF, [early(5)], policy).events
    assert admitted.anomalies[0].status is AnomalyStatus.PENDING_RESOLUTION


def test_clock_check_applies_to_on_chain_kinds_only():
    external = make_synthetic_event(
        "ext",
        block_time=BLOCK,
        received_at=BLOCK - timedelta(seconds=5),
        source_type=SourceType.EXTERNAL_RECORD,
    )
    result = check_availability(
        external, CUTOFF, TemporalPolicy((SourceClockRule("synthetic-provider-a", timedelta(0)),))
    )
    assert result.is_available and result.anomalies == ()


@pytest.mark.parametrize(
    "kind", [SourceType.ONCHAIN_RECEIPT, SourceType.ONCHAIN_TRACE, SourceType.ONCHAIN_TRANSACTION]
)
def test_the_anomaly_is_detected_for_every_on_chain_kind(kind):
    assert check_availability(early(1, kind=kind), CUTOFF).anomalies


def test_clock_rules_reject_nonsense():
    with pytest.raises(ValueError):
        SourceClockRule("", timedelta(0))
    with pytest.raises(ValueError):
        SourceClockRule("p", timedelta(seconds=-1))


def test_policy_changes_only_admission_and_the_result_is_deterministic():
    policy = TemporalPolicy((SourceClockRule("synthetic-provider-a", timedelta(seconds=2)),))
    ledger = [early(5), make("ok", received_at=BLOCK + timedelta(seconds=1), log_index=1)]
    assert make_asof_snapshot(CUTOFF, ledger, policy) == make_asof_snapshot(CUTOFF, ledger, policy)
    assert (
        make_asof_snapshot(CUTOFF, ledger).snapshot_hash
        != make_asof_snapshot(CUTOFF, ledger, policy).snapshot_hash
    )
