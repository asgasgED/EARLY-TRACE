"""Task 02: canonical identity, observations and consolidation of evidence objects."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from early_trace.evidence.canonical import (
    ConflictKind,
    EvidenceStanding,
    ObservationRole,
    PlacementStatus,
)
from early_trace.evidence.events import ObservedValue, SourceType
from early_trace.evidence.identity import (
    IdentityIssue,
    ObjectKey,
    UnresolvedIdentity,
    resolve_identity,
)
from early_trace.evidence.ledger import load_ledger_json
from early_trace.evidence.observation import FinalityStatus, ObservationContext
from early_trace.evidence.synthetic import make_synthetic_event
from early_trace.temporal.snapshot import make_asof_snapshot

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CUTOFF = "2026-03-01T10:30:00+00:00"
T0 = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
ST = SourceType


def at(minute: int) -> datetime:
    return T0 + timedelta(minutes=minute)


def make(event_id, kind=ST.ONCHAIN_LOG, minute=1, **kwargs):
    return make_synthetic_event(
        event_id, block_time=at(0), received_at=at(minute), source_type=kind, **kwargs
    )


def objects_by_first_event(snapshot):
    return {o.observations[0].observation.event.event_id: o for o in snapshot.objects}


@pytest.fixture(scope="module")
def identity_snapshot():
    ledger = load_ledger_json(
        FIXTURES / "synthetic" / "ledger_identity.json", require_synthetic=True
    )
    return make_asof_snapshot(CUTOFF, ledger)


@pytest.fixture(scope="module")
def adversarial_snapshot():
    path = FIXTURES / "adversarial" / "ledger_identity_adversarial.json"
    return make_asof_snapshot(CUTOFF, load_ledger_json(path, require_synthetic=True))


# ---------------------------------------------------------------- identity of one object


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (ST.ONCHAIN_BLOCK, ("chain_id", "block_hash")),
        (ST.ONCHAIN_TRANSACTION, ("chain_id", "tx_hash")),
        (ST.ONCHAIN_RECEIPT, ("chain_id", "block_hash", "tx_hash")),
        (ST.ONCHAIN_LOG, ("chain_id", "block_hash", "tx_hash", "log_index")),
        (ST.ONCHAIN_TRACE, ("chain_id", "block_hash", "tx_hash", "trace_address")),
    ],
)
def test_identity_uses_exactly_the_fields_of_its_kind(kind, expected):
    key = resolve_identity(make("e1", kind))
    assert isinstance(key, ObjectKey)
    assert tuple(name for name, _ in key.components) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"source": "another-provider"},
        {"source_version": "v9"},
        {"context": ObservationContext(method="eth_getLogs", query_id="q-1")},
        {"context": ObservationContext(decoder_version="dec-2", parser_version="p-3")},
        {"context": ObservationContext(finality=FinalityStatus.FINALIZED)},
        {"minute": 20},
    ],
)
def test_identity_is_independent_of_provider_method_decoder_and_reception_time(overrides):
    base = resolve_identity(make("e1", tx_label="t"))
    other = resolve_identity(make("e2", tx_label="t", **overrides))
    assert base == other and base.key_id == other.key_id


@pytest.mark.parametrize(
    "overrides",
    [{"chain_id": 2}, {"block_label": "other"}, {"tx_label": "other"}, {"log_index": 1}],
)
def test_any_identity_component_change_yields_a_different_object(overrides):
    base = resolve_identity(make("e1", tx_label="t"))
    other = resolve_identity(make("e1", **{"tx_label": "t", **overrides}))
    assert base.key_id != other.key_id


def test_kind_is_part_of_the_identity():
    receipt = resolve_identity(make("e1", ST.ONCHAIN_RECEIPT, tx_label="t"))
    log = resolve_identity(make("e1", ST.ONCHAIN_LOG, tx_label="t"))
    assert receipt.key_id != log.key_id


def test_transaction_identity_ignores_the_including_block():
    x = resolve_identity(make("e1", ST.ONCHAIN_TRANSACTION, tx_label="t", block_label="x"))
    y = resolve_identity(make("e2", ST.ONCHAIN_TRANSACTION, tx_label="t", block_label="y"))
    assert x.key_id == y.key_id


def test_trace_position_distinguishes_traces_and_the_root_call_is_a_valid_position():
    root = resolve_identity(make("e1", ST.ONCHAIN_TRACE, tx_label="t", trace_address=()))
    child = resolve_identity(make("e2", ST.ONCHAIN_TRACE, tx_label="t", trace_address=(0,)))
    assert isinstance(root, ObjectKey) and isinstance(child, ObjectKey)
    assert root.key_id != child.key_id


@pytest.mark.parametrize(
    ("kind", "omit", "missing"),
    [
        (ST.ONCHAIN_RECEIPT, "chain_id", "chain_id"),
        (ST.ONCHAIN_LOG, "log_index", "log_index"),
        (ST.ONCHAIN_TRACE, "trace_address", "trace_address"),
        (ST.ONCHAIN_LOG, "block_hash", "block_hash"),
    ],
)
def test_missing_identity_fields_leave_the_identity_unresolved_with_a_reason(kind, omit, missing):
    identity = resolve_identity(make("e1", kind, omit=[omit]))
    assert isinstance(identity, UnresolvedIdentity)
    assert identity.issue is IdentityIssue.MISSING_FIELDS
    assert identity.missing_fields == (missing,)
    assert missing in identity.detail
    assert not hasattr(identity, "key_id")  # no key is ever invented


def test_external_records_have_no_canonical_identity_defined():
    event = make_synthetic_event(
        "e1", block_time=None, received_at=at(1), source_type=ST.EXTERNAL_RECORD
    )
    identity = resolve_identity(event)
    assert isinstance(identity, UnresolvedIdentity)
    assert identity.issue is IdentityIssue.UNDEFINED_FOR_SOURCE_TYPE


def test_key_id_is_a_stable_sha256_of_the_canonical_components():
    key = resolve_identity(make("e1", tx_label="t"))
    assert key.key_id == resolve_identity(make("e1", tx_label="t")).key_id
    assert len(key.key_id) == 64 and int(key.key_id, 16) >= 0


def test_four_hashes_are_four_different_concepts():
    a = make(
        "r-a",
        ST.ONCHAIN_RECEIPT,
        tx_label="t",
        source="prov-a",
        fields={"gas": ObservedValue.observed(1)},
    )
    b = make(
        "r-b",
        ST.ONCHAIN_RECEIPT,
        minute=2,
        tx_label="t",
        source="prov-b",
        fields={"gas": ObservedValue.observed(1)},
    )
    snap_a = make_asof_snapshot(CUTOFF, [a])
    snap_ab = make_asof_snapshot(CUTOFF, [a, b])
    pa, pb = snap_ab.events[0].provenance, snap_ab.events[1].provenance
    assert pa.object_key_id == pb.object_key_id  # same object
    assert pa.payload_hash == pb.payload_hash  # same content
    assert pa.observation_hash != pb.observation_hash  # different observation
    assert snap_a.snapshot_hash != snap_ab.snapshot_hash  # different snapshot
    assert len({pa.object_key_id, pa.payload_hash, pa.observation_hash, snap_ab.snapshot_hash}) == 4


# ---------------------------------------------------------------- consolidation scenarios


def test_one_object_observed_by_several_paths_is_one_canonical_object(identity_snapshot):
    s = identity_snapshot
    assert len(s.events) == 14 and s.exclusions == ()  # every observation is preserved
    assert len(s.objects) == 9
    by_event = objects_by_first_event(s)

    s1 = by_event["s1-receipt-a"]  # Q_RECEIPT vs Q_BLOCK_RECEIPTS
    assert [link.role for link in s1.observations] == [
        ObservationRole.FIRST,
        ObservationRole.DUPLICATE,
    ]
    assert {link.observation.event.context.method for link in s1.observations} == {
        "eth_getTransactionReceipt",
        "eth_getBlockReceipts",
    }
    assert s1.providers == ("prov-a", "prov-b")
    assert by_event["s2-log-via-receipt"].observations[1].role is ObservationRole.DUPLICATE
    assert (
        by_event["s4-log-d1"].observations[1].role is ObservationRole.DUPLICATE
    )  # decoder differs


def test_different_logs_of_one_transaction_are_different_objects(identity_snapshot):
    by_event = objects_by_first_event(identity_snapshot)
    assert by_event["s6-log-0"].key_id != by_event["s6-log-1"].key_id
    assert all(len(by_event[e].observations) == 1 for e in ("s6-log-0", "s6-log-1"))


def test_first_availability_is_the_earliest_reception(identity_snapshot):
    s5 = objects_by_first_event(identity_snapshot)["s5-log-early"]
    assert s5.first_received_at == datetime(2026, 3, 1, 10, 9, tzinfo=timezone.utc)
    assert [link.observation.event.event_id for link in s5.observations] == [
        "s5-log-early",
        "s5-log-late",
    ]


def test_independent_confirmation_requires_audited_distinct_groups(identity_snapshot):
    s3 = objects_by_first_event(identity_snapshot)["s3-log-g1"]
    assert s3.observations[1].role is ObservationRole.INDEPENDENT_CONFIRMATION


@pytest.mark.parametrize(
    ("first_group", "second_group"),
    [(None, None), ("g1", None), (None, "g2"), ("g1", "g1")],
)
def test_independence_is_never_presumed(first_group, second_group):
    def ctx(group):
        return ObservationContext(infrastructure_group=group)

    a = make("a", minute=1, tx_label="t", source="independent-looking-a", context=ctx(first_group))
    b = make("b", minute=2, tx_label="t", source="independent-looking-b", context=ctx(second_group))
    (obj,) = make_asof_snapshot(CUTOFF, [a, b]).objects
    assert obj.observations[1].role is ObservationRole.DUPLICATE


def test_a_reorganized_transaction_keeps_both_placements_and_marks_the_orphan(identity_snapshot):
    s = identity_snapshot
    by_event = objects_by_first_event(s)
    x, y = by_event["s7-receipt-x"], by_event["s7-receipt-y"]
    assert x.key_id != y.key_id  # different blocks are different objects
    assert (x.standing, x.placement) == (
        EvidenceStanding.ORPHANED,
        PlacementStatus.DIVERGENT_NON_CANONICAL,
    )
    assert (y.standing, y.placement) == (
        EvidenceStanding.USABLE,
        PlacementStatus.DIVERGENT_CANONICAL,
    )
    (divergence,) = s.placement_divergences
    assert divergence.resolved and len(divergence.block_hashes) == 2
    assert divergence.canonical_block_hash == y.observations[0].observation.event.block_hash


def test_adversarial_identity_cases_are_classified_without_resolving_anything(adversarial_snapshot):
    by_event = objects_by_first_event(adversarial_snapshot)
    for event_id in ("a1-no-chain", "a2-log-no-index", "a3-trace-no-position", "a4-external"):
        obj = by_event[event_id]
        assert obj.standing is EvidenceStanding.UNRESOLVED_IDENTITY
        assert obj.observations[0].role is ObservationRole.UNRESOLVED
        assert obj.key_id is None
    kinds = {
        "a5-content-1": ConflictKind.CONTENT,
        "a6-interp-1": ConflictKind.INTERPRETATION,
        "a7-tx-block-x": ConflictKind.PLACEMENT,
        "a9-finality-reorg": ConflictKind.FINALITY,
    }
    for event_id, kind in kinds.items():
        obj = by_event[event_id]
        assert [c.kind for c in obj.conflicts] == [kind]
        assert obj.standing is EvidenceStanding.CONFLICTED
        assert {link.role for link in obj.observations} == {ObservationRole.CONFLICTING}
    assert len(by_event["a5-content-1"].observations) == 2  # nothing discarded


def test_same_transaction_in_two_blocks_without_finality_is_unresolved(adversarial_snapshot):
    by_event = objects_by_first_event(adversarial_snapshot)
    for event_id in ("a8-rcpt-x", "a8-rcpt-y"):
        assert by_event[event_id].standing is EvidenceStanding.PLACEMENT_UNRESOLVED
    unresolved = [d for d in adversarial_snapshot.placement_divergences if not d.resolved]
    assert len(unresolved) == 2  # a7 (transaction) and a8 (receipts)


@pytest.mark.parametrize(
    ("finality_x", "finality_y", "standing_x", "standing_y"),
    [
        (
            FinalityStatus.REORGED,
            FinalityStatus.UNKNOWN,
            EvidenceStanding.ORPHANED,
            EvidenceStanding.USABLE,
        ),
        (
            FinalityStatus.UNKNOWN,
            FinalityStatus.FINALIZED,
            EvidenceStanding.SUPERSEDED,
            EvidenceStanding.USABLE,
        ),
        (
            FinalityStatus.FINALIZED,
            FinalityStatus.FINALIZED,
            EvidenceStanding.PLACEMENT_UNRESOLVED,
            EvidenceStanding.PLACEMENT_UNRESOLVED,
        ),
        (
            FinalityStatus.REORGED,
            FinalityStatus.REORGED,
            EvidenceStanding.ORPHANED,
            EvidenceStanding.ORPHANED,
        ),
    ],
)
def test_placement_resolution_uses_finality_only(finality_x, finality_y, standing_x, standing_y):
    x = make(
        "x",
        ST.ONCHAIN_RECEIPT,
        tx_label="t",
        block_label="x",
        context=ObservationContext(finality=finality_x),
    )
    y = make(
        "y",
        ST.ONCHAIN_RECEIPT,
        tx_label="t",
        block_label="y",
        context=ObservationContext(finality=finality_y),
    )
    snapshot = make_asof_snapshot(CUTOFF, [x, y])
    by_event = objects_by_first_event(snapshot)
    assert by_event["x"].standing is standing_x
    assert by_event["y"].standing is standing_y


def test_finality_history_is_ordered_and_the_latest_known_state_wins():
    states = [
        (1, FinalityStatus.UNFINALIZED),
        (2, FinalityStatus.UNKNOWN),
        (3, FinalityStatus.FINALIZED),
    ]
    ledger = [
        make(
            f"o{m}",
            ST.ONCHAIN_RECEIPT,
            minute=m,
            tx_label="t",
            source=f"p{m}",
            context=ObservationContext(finality=f),
        )
        for m, f in states
    ]
    (obj,) = make_asof_snapshot(CUTOFF, ledger).objects
    assert [p.finality for p in obj.finality_history] == [
        FinalityStatus.UNFINALIZED,
        FinalityStatus.FINALIZED,
    ]
    assert obj.latest_finality is FinalityStatus.FINALIZED


def test_one_object_related_to_two_entities_is_two_links_of_one_object():
    a = make("a", tx_label="t", entity_id="entity-a", fields={"v": ObservedValue.observed(1)})
    b = make("b", tx_label="t", entity_id="entity-b", fields={"v": ObservedValue.observed(9)})
    objs = make_asof_snapshot(CUTOFF, [a, b]).objects
    assert len(objs) == 2 and len({o.key_id for o in objs}) == 1
    assert all(
        o.standing is EvidenceStanding.USABLE for o in objs
    )  # entity-relative payloads, no conflict


def test_unresolved_observations_are_never_merged():
    a = make("a", omit=["log_index"], tx_label="t", source="p1")
    b = make("b", omit=["log_index"], tx_label="t", source="p2")
    objs = make_asof_snapshot(CUTOFF, [a, b]).objects
    assert len(objs) == 2 and all(o.key_id is None for o in objs)


def test_ledger_order_does_not_change_the_consolidation(identity_snapshot):
    ledger = list(
        load_ledger_json(FIXTURES / "synthetic" / "ledger_identity.json", require_synthetic=True)
    )
    rng = random.Random(7)
    for _ in range(10):
        rng.shuffle(ledger)
        shuffled = make_asof_snapshot(CUTOFF, ledger)
        assert shuffled.to_dict() == identity_snapshot.to_dict()


def test_future_observations_of_a_known_object_do_not_change_the_past():
    first = make("a", minute=1, tx_label="t", source="p1", fields={"v": ObservedValue.observed(1)})
    later = make(
        "b", minute=20, tx_label="t", source="p2", fields={"v": ObservedValue.observed(999)}
    )
    early_cutoff = at(10)
    alone = make_asof_snapshot(early_cutoff, [first])
    with_future = make_asof_snapshot(early_cutoff, [first, later])
    assert alone.snapshot_hash == with_future.snapshot_hash
    assert alone.objects == with_future.objects  # the conflicting future value never shows
    full = make_asof_snapshot(at(30), [first, later])
    assert full.objects[0].standing is EvidenceStanding.CONFLICTED


def test_canonical_key_is_pinned_across_runs_and_processes():
    # regression pin: changing the identity contract changes every canonical key
    key = resolve_identity(make("e1", tx_label="golden", log_index=3))
    assert key.key_id == "ef83e20325a125f7467b6ebfaef3e68689a8a14000aaecc4b577f037a61b941b"


def test_an_empty_response_is_preserved_as_evidence_but_is_not_negative_evidence():
    from early_trace.features.aggregation import aggregate_feature
    from early_trace.schemas.aggregation import AggregationOp, AggregationSpec, ResultState
    from early_trace.schemas.feature_schema import (
        AbsenceSpec,
        AvailabilityClock,
        FeatureDType,
        FeatureSpec,
        MissingnessRule,
        SchemaVersion,
    )

    mask = AbsenceSpec.mask_only()
    spec = FeatureSpec(
        name="transfer_total",
        dtype=FeatureDType.INT,
        description="Total transferred value.",
        source_type=ST.ONCHAIN_LOG,
        source_field="v",
        availability_clock=AvailabilityClock.RECEIVED_AT,
        missingness=MissingnessRule(mask, mask, mask, mask, mask),
        aggregation=AggregationSpec(AggregationOp.SUM, 1),
        feature_version=1,
        introduced_in=SchemaVersion.parse("1.0.0"),
    )
    event = make("empty", fields={"v": ObservedValue.empty_response()})
    snapshot = make_asof_snapshot(CUTOFF, [event])
    assert len(snapshot.events) == 1 and len(snapshot.objects) == 1  # kept, not dropped
    result = aggregate_feature(spec, snapshot, "synthetic-entity-1")
    assert (result.state, result.value) == (ResultState.EMPTY_RESPONSE, None)  # not zero
