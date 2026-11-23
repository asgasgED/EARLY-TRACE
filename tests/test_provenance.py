"""PROVENANCE: admitted events carry the provenance their source type requires."""

from __future__ import annotations

from pathlib import Path

import pytest

from early_trace.evidence.events import SourceType
from early_trace.evidence.ledger import load_ledger_json
from early_trace.evidence.provenance import REQUIRED_PROVENANCE
from early_trace.evidence.synthetic import make_synthetic_event
from early_trace.infrastructure.hashing import sha256_hex
from early_trace.temporal.availability import ExclusionReason
from early_trace.temporal.snapshot import make_asof_snapshot

CUTOFF = "2026-03-01T10:05:00+00:00"
ADVERSARIAL = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "adversarial"
    / "ledger_adversarial_timestamps.json"
)
BLOCK = "2026-03-01T10:00:00+00:00"
RECEIVED = "2026-03-01T10:01:00+00:00"


def make(event_id, source_type=SourceType.ONCHAIN_LOG, **kwargs):
    return make_synthetic_event(
        event_id,
        block_time=kwargs.pop("block_time", BLOCK),
        received_at=kwargs.pop("received_at", RECEIVED),
        source_type=source_type,
        **kwargs,
    )


@pytest.mark.parametrize("source_type", list(SourceType))
def test_admitted_events_carry_required_provenance(source_type):
    block_time = None if source_type is SourceType.EXTERNAL_RECORD else BLOCK
    snapshot = make_asof_snapshot(CUTOFF, [make("e1", source_type, block_time=block_time)])
    assert len(snapshot.events) == 1
    record = snapshot.events[0].provenance
    for name in REQUIRED_PROVENANCE[source_type]:
        assert getattr(record, name if name != "block_time" else "block_time_utc") is not None
    assert record.source and record.event_id and record.received_at_utc


@pytest.mark.parametrize(
    ("source_type", "omitted"),
    [
        (SourceType.ONCHAIN_LOG, "tx_hash"),
        (SourceType.ONCHAIN_LOG, "block_hash"),
        (SourceType.ONCHAIN_LOG, "block_number"),
        (SourceType.ONCHAIN_TRANSACTION, "tx_hash"),
        (SourceType.ONCHAIN_BLOCK, "block_hash"),
    ],
)
def test_missing_required_provenance_is_excluded_with_reason(source_type, omitted):
    snapshot = make_asof_snapshot(CUTOFF, [make("e1", source_type, omit=[omitted])])
    assert snapshot.events == ()
    (exclusion,) = snapshot.exclusions
    assert exclusion.reason is ExclusionReason.INCOMPLETE_PROVENANCE
    assert omitted in exclusion.detail


def test_onchain_event_without_block_time_lacks_provenance():
    snapshot = make_asof_snapshot(CUTOFF, [make("e1", block_time=None)])
    assert snapshot.exclusions[0].reason is ExclusionReason.INCOMPLETE_PROVENANCE


def test_identifiers_are_preserved_through_the_snapshot():
    event = make(
        "e1",
        entity_id="entity-x",
        source="provider-z",
        source_version="v9",
        block_number=777,
        log_index=5,
    )
    record = make_asof_snapshot(CUTOFF, [event]).events[0].provenance
    assert (record.event_id, record.entity_id, record.source, record.source_version) == (
        "e1",
        "entity-x",
        "provider-z",
        "v9",
    )
    assert (record.block_number, record.block_hash, record.tx_hash, record.log_index) == (
        777,
        event.block_hash,
        event.tx_hash,
        5,
    )
    assert record.observation_hash == sha256_hex(
        event.observation_dict(record.block_time_utc, record.received_at_utc)
    )


def test_no_event_is_lost_silently():
    ledger = load_ledger_json(ADVERSARIAL, require_synthetic=True)
    snapshot = make_asof_snapshot(CUTOFF, ledger)
    assert len(snapshot.events) + len(snapshot.exclusions) == len(ledger)
    accounted = [e.event for e in snapshot.events] + [x.event for x in snapshot.exclusions]
    assert sorted(map(repr, accounted)) == sorted(map(repr, ledger))


def test_adversarial_fixture_exclusion_reasons():
    ledger = load_ledger_json(ADVERSARIAL, require_synthetic=True)
    snapshot = make_asof_snapshot(CUTOFF, ledger)
    assert snapshot.exclusion_counts() == {
        "duplicate_record": 1,
        "incomplete_provenance": 1,
        "invalid_block_time": 1,
        "invalid_received_at": 2,
        "naive_block_time": 1,
        "naive_received_at": 2,
        "received_after_cutoff": 1,
        "received_at_missing": 1,
    }
    # received-before-block is admitted (anomaly), a log without log_index is admitted with
    # an unresolved identity: neither is a provenance failure any more.
    assert set(snapshot.event_ids) == {
        "adv-ok",
        "adv-dup",
        "adv-conflict",
        "adv-received-before-block",
        "adv-missing-log-index",
    }


def test_exact_duplicate_is_counted_once_but_stays_traceable():
    twin = make("e1")
    snapshot = make_asof_snapshot(CUTOFF, [twin, make("e1")])
    assert len(snapshot.events) == 1
    (exclusion,) = snapshot.exclusions
    assert exclusion.reason is ExclusionReason.DUPLICATE_RECORD
    assert exclusion.event.event_id == "e1"


def test_same_event_id_with_different_content_is_flagged_not_merged():
    first = make("e1", source_version="v1")
    second = make("e1", source_version="v2")
    snapshot = make_asof_snapshot(CUTOFF, [first, second])
    assert len(snapshot.events) == 2
    assert snapshot.conflicting_event_ids == ("e1",)
    assert snapshot.exclusions == ()
