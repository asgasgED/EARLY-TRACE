"""Task 04: aggregation contract (spec validation) and its reference semantics."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from early_trace.evidence.events import ObservedValue as OV
from early_trace.evidence.events import SourceType as ST
from early_trace.evidence.ledger import load_ledger_json
from early_trace.evidence.observation import ObservationContext
from early_trace.evidence.synthetic import make_synthetic_event
from early_trace.features.aggregation import (
    AggregationIssue,
    ExclusionKind,
    aggregate_feature,
    declared_fill,
)
from early_trace.schemas.aggregation import (
    AGGREGATION_CONTRACT,
    AggregationOp,
    AggregationSpec,
    AggregationSpecError,
    OrderingClock,
    ResultState,
    WindowSpec,
)
from early_trace.schemas.feature_schema import (
    AbsenceSpec,
    AvailabilityClock,
    FeatureDType,
    FeatureSchema,
    FeatureSpec,
    InvalidFeatureSpecError,
    MissingnessRule,
    SchemaVersion,
)
from early_trace.temporal.snapshot import make_asof_snapshot

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
T0 = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
CUTOFF = T0 + timedelta(minutes=30)
ENTITY = "synthetic-entity-1"
MASK = AbsenceSpec.mask_only()
R = ResultState
Op = AggregationOp


def at(minute: float) -> datetime:
    return T0 + timedelta(minutes=minute)


def fs(
    op,
    *,
    dtype=FeatureDType.INT,
    kind=ST.ONCHAIN_LOG,
    order=None,
    window=None,
    rule=1,
    missingness=None,
    field="v",
    version=1,
) -> FeatureSpec:
    return FeatureSpec(
        name="feature_x",
        dtype=dtype,
        description="Synthetic aggregated feature.",
        source_type=kind,
        source_field=field,
        availability_clock=AvailabilityClock.RECEIVED_AT,
        missingness=missingness or MissingnessRule(MASK, MASK, MASK, MASK, MASK),
        aggregation=AggregationSpec(op, rule, order, window),
        feature_version=version,
        introduced_in=SchemaVersion.parse("1.0.0"),
    )


def log(
    eid,
    value,
    *,
    minute=1,
    block=0,
    idx=0,
    tx=None,
    entity=ENTITY,
    source="prov-a",
    ctx=None,
    kind=ST.ONCHAIN_LOG,
    field="v",
    omit=(),
):
    fields = (
        {} if value is None else {field: value if isinstance(value, OV) else OV.observed(value)}
    )
    return make_synthetic_event(
        eid,
        block_time=at(block),
        received_at=at(minute),
        source_type=kind,
        log_index=idx,
        tx_label=tx or eid,
        entity_id=entity,
        source=source,
        context=ctx,
        fields=fields,
        omit=omit,
    )


def run(spec, events, cutoff=CUTOFF, entity=ENTITY):
    return aggregate_feature(spec, make_asof_snapshot(cutoff, events), entity)


# ------------------------------------------------------------------ spec validation


@pytest.mark.parametrize(
    "build",
    [
        lambda: AggregationSpec(Op.LATEST, 1),  # order clock missing
        lambda: AggregationSpec(Op.EARLIEST, 1),
        lambda: AggregationSpec(Op.SUM, 1, OrderingClock.BLOCK_TIME),  # order clock not used
        lambda: AggregationSpec(Op.SUM, 0),
        lambda: AggregationSpec(Op.SUM, True),  # type: ignore[arg-type]
        lambda: AggregationSpec("sum", 1),  # type: ignore[arg-type]
        lambda: WindowSpec(OrderingClock.RECEIVED_AT, 0),
        lambda: WindowSpec("received_at", 60),  # type: ignore[arg-type]
        lambda: WindowSpec(OrderingClock.RECEIVED_AT, True),  # type: ignore[arg-type]
    ],
)
def test_invalid_aggregation_declarations_are_rejected(build):
    with pytest.raises(AggregationSpecError):
        build()


@pytest.mark.parametrize(
    ("op", "dtype", "ok"),
    [
        (Op.SUM, FeatureDType.INT, True),
        (Op.SUM, FeatureDType.FLOAT, True),
        (Op.SUM, FeatureDType.BOOL, False),
        (Op.SUM, FeatureDType.CATEGORY, False),
        (Op.MEAN, FeatureDType.FLOAT, True),
        (Op.MEAN, FeatureDType.INT, False),
        (Op.COUNT_DISTINCT, FeatureDType.INT, True),
        (Op.COUNT_DISTINCT, FeatureDType.FLOAT, False),
        (Op.MIN, FeatureDType.CATEGORY, False),
        (Op.SINGLE, FeatureDType.CATEGORY, True),
    ],
)
def test_operation_and_dtype_must_be_compatible(op, dtype, ok):
    order = OrderingClock.RECEIVED_AT if op in (Op.LATEST, Op.EARLIEST) else None
    if ok:
        assert fs(op, dtype=dtype, order=order).dtype is dtype
    else:
        with pytest.raises(InvalidFeatureSpecError):
            fs(op, dtype=dtype, order=order)


def test_aggregation_serializes_with_its_contract_and_round_trips():
    spec = fs(
        Op.LATEST,
        order=OrderingClock.BLOCK_TIME,
        window=WindowSpec(OrderingClock.RECEIVED_AT, 3600),
    )
    raw = spec.to_dict()["aggregation"]
    assert raw["contract"] == AGGREGATION_CONTRACT
    assert FeatureSpec.from_dict(spec.to_dict()) == spec


def test_an_implicit_or_foreign_aggregation_is_rejected_when_loading():
    raw = fs(Op.SUM).to_dict()
    del raw["aggregation"]
    with pytest.raises(InvalidFeatureSpecError, match="aggregation"):
        FeatureSpec.from_dict(raw)
    raw = fs(Op.SUM).to_dict()
    raw["aggregation"]["contract"] = "aggregation/99"
    with pytest.raises(InvalidFeatureSpecError):
        FeatureSpec.from_dict(raw)
    raw = fs(Op.SUM).to_dict()
    del raw["aggregation"]["window"]
    with pytest.raises(InvalidFeatureSpecError):
        FeatureSpec.from_dict(raw)


@pytest.mark.parametrize(
    "variant",
    [
        dict(op=Op.MAX),
        dict(op=Op.SUM, rule=2),
        dict(op=Op.SUM, window=WindowSpec(OrderingClock.RECEIVED_AT, 60)),
        dict(op=Op.SUM, window=WindowSpec(OrderingClock.BLOCK_TIME, 60)),
        dict(op=Op.LATEST, order=OrderingClock.BLOCK_TIME),
    ],
)
def test_every_aggregation_field_changes_the_schema_hash(variant):
    def schema(spec):
        return FeatureSchema("s", SchemaVersion.parse("1.0.0"), (spec,))

    assert schema(fs(**variant)).schema_hash != schema(fs(Op.SUM)).schema_hash
    assert (
        schema(fs(**variant)).schema_hash
        != schema(fs(Op.LATEST, order=OrderingClock.RECEIVED_AT)).schema_hash
    )


# ------------------------------------------------------------------ values and operations


def test_single_value():
    r = run(fs(Op.SINGLE), [log("a", 7)])
    assert (r.state, r.value) == (R.OBSERVED, 7)


def test_sum_over_distinct_objects():
    r = run(fs(Op.SUM), [log("a", 3, idx=0), log("b", 4, idx=1)])
    assert (r.state, r.value) == (R.OBSERVED, 7)


def test_the_same_receipt_through_two_providers_is_aggregated_once():
    spec = fs(Op.SUM, kind=ST.ONCHAIN_RECEIPT)
    ledger = [
        log(
            "r-a",
            21000,
            tx="t",
            source="prov-a",
            kind=ST.ONCHAIN_RECEIPT,
            ctx=ObservationContext(method="eth_getTransactionReceipt"),
        ),
        log(
            "r-b",
            21000,
            minute=2,
            tx="t",
            source="prov-b",
            kind=ST.ONCHAIN_RECEIPT,
            ctx=ObservationContext(method="eth_getBlockReceipts"),
        ),
    ]
    r = run(spec, ledger)
    assert (r.state, r.value) == (R.OBSERVED, 21000)
    assert len(r.provenance.object_key_ids) == 1
    assert len(r.provenance.observation_hashes) == 2
    assert len(r.provenance.duplicate_observation_hashes) == 1
    assert r.provenance.sources == ("prov-a", "prov-b")


def test_different_decoder_versions_of_one_object_count_once():
    ledger = [
        log("a", 9, tx="t", ctx=ObservationContext(decoder_version="dec-1")),
        log("b", 9, minute=2, tx="t", ctx=ObservationContext(decoder_version="dec-2")),
    ]
    assert run(fs(Op.SUM), ledger).value == 9


def test_independent_confirmations_do_not_multiply_the_value():
    ledger = [
        log("a", 5, tx="t", source="p1", ctx=ObservationContext(infrastructure_group="g1")),
        log(
            "b", 5, minute=2, tx="t", source="p2", ctx=ObservationContext(infrastructure_group="g2")
        ),
    ]
    r = run(fs(Op.SUM), ledger)
    assert r.value == 5 and len(r.provenance.confirmation_observation_hashes) == 1


def _two_orderings():
    older_by_block_received_late = log("a", 1, block=0, minute=9, idx=0)
    newer_by_block_received_early = log("b", 2, block=5, minute=6, idx=1)
    return [older_by_block_received_late, newer_by_block_received_early]


@pytest.mark.parametrize(
    ("op", "clock", "expected"),
    [
        (Op.LATEST, OrderingClock.BLOCK_TIME, 2),
        (Op.LATEST, OrderingClock.RECEIVED_AT, 1),
        (Op.EARLIEST, OrderingClock.BLOCK_TIME, 1),
        (Op.EARLIEST, OrderingClock.RECEIVED_AT, 2),
    ],
)
def test_latest_and_earliest_follow_the_declared_clock(op, clock, expected):
    assert run(fs(op, order=clock), _two_orderings()).value == expected


def test_ties_are_fine_when_values_agree_and_a_conflict_when_they_do_not():
    spec = fs(Op.LATEST, order=OrderingClock.BLOCK_TIME)
    agree = run(spec, [log("a", 4, idx=0), log("b", 4, idx=1)])
    assert (agree.state, agree.value) == (R.OBSERVED, 4)
    clash = run(spec, [log("a", 4, idx=0), log("b", 5, idx=1)])
    assert (clash.state, clash.issue) == (R.CONFLICT, AggregationIssue.TIE_WITH_INCOMPATIBLE_VALUES)
    assert clash.value is None


def test_latest_ignores_the_value_of_older_objects_but_not_of_the_selected_one():
    spec = fs(Op.LATEST, order=OrderingClock.BLOCK_TIME)
    older_unknown = [log("a", OV.unknown(), block=0, idx=0), log("b", 8, block=5, idx=1)]
    assert run(spec, older_unknown).value == 8
    latest_unknown = [log("a", 8, block=0, idx=0), log("b", OV.unknown(), block=5, idx=1)]
    assert run(spec, latest_unknown).state is R.UNKNOWN


def test_count_distinct_counts_canonical_objects():
    ledger = [
        log("a", 1, idx=0),
        log("b", 1, idx=1),
        log("c", 1, idx=2),
        log("c-again", 1, minute=2, idx=2, tx="c", source="prov-b"),  # same object as "c"
    ]
    r = run(fs(Op.COUNT_DISTINCT), ledger)
    assert (r.state, r.value) == (R.OBSERVED, 3)
    assert len(r.provenance.observation_hashes) == 4


def test_min_max_and_mean():
    ledger = [
        log("a", 2.0, idx=0),
        log("b", 3.0, idx=1),
        log("c", 4.0, idx=2),
        log("d", 1.0, idx=3),
    ]
    assert run(fs(Op.MIN, dtype=FeatureDType.FLOAT), ledger).value == 1.0
    assert run(fs(Op.MAX, dtype=FeatureDType.FLOAT), ledger).value == 4.0
    mean = run(fs(Op.MEAN, dtype=FeatureDType.FLOAT), ledger)
    assert (mean.value, type(mean.value)) == (2.5, float)


def test_float_sum_is_exact_and_independent_of_ledger_order():
    values = [0.1, 0.2, 0.3, 0.4, 1e-9]
    ledger = [log(f"e{i}", v, idx=i) for i, v in enumerate(values)]
    spec = fs(Op.SUM, dtype=FeatureDType.FLOAT)
    reference = run(spec, ledger)
    for seed in range(5):
        shuffled = ledger[:]
        random.Random(seed).shuffle(shuffled)
        again = run(spec, shuffled)
        assert again.value == reference.value and again.result_hash == reference.result_hash


# ------------------------------------------------------------------ absence states


def test_an_observed_zero_is_a_value_not_an_absence():
    r = run(fs(Op.SUM), [log("a", 0, idx=0), log("b", 0, idx=1)])
    assert (r.state, r.value) == (R.OBSERVED, 0) and r.value is not None
    assert run(fs(Op.SINGLE), [log("a", 0)]).state is R.OBSERVED


@pytest.mark.parametrize(
    ("value", "state"),
    [
        (OV.null(), R.NULL),
        (OV.unknown(), R.UNKNOWN),
        (OV.empty_response(), R.EMPTY_RESPONSE),
        (OV.not_indexed(), R.NOT_INDEXED),
    ],
)
@pytest.mark.parametrize("op", [Op.SUM, Op.MAX, Op.COUNT_DISTINCT, Op.SINGLE])
def test_absence_states_propagate_and_never_become_numbers(op, value, state):
    r = run(fs(op), [log("a", value)])
    assert (r.state, r.value) == (state, None)


def test_an_absent_participant_makes_the_aggregate_absent_not_partial():
    r = run(fs(Op.SUM), [log("a", 5, idx=0), log("b", OV.not_indexed(), idx=1)])
    assert (r.state, r.value) == (R.NOT_INDEXED, None)
    mixed = run(fs(Op.SUM), [log("a", OV.unknown(), idx=0), log("b", OV.null(), idx=1)])
    assert mixed.state is R.UNKNOWN


def test_no_observation_is_not_received_and_never_zero():
    for op in (Op.SUM, Op.COUNT_DISTINCT, Op.MAX, Op.SINGLE):
        r = run(fs(op), [])
        assert (r.state, r.value) == (R.NOT_RECEIVED, None)
    other_entity = run(fs(Op.SUM), [log("a", 5, entity="someone-else")])
    assert other_entity.state is R.NOT_RECEIVED
    other_kind = run(fs(Op.SUM, kind=ST.ONCHAIN_RECEIPT), [log("a", 5)])
    assert other_kind.state is R.NOT_RECEIVED


def test_objects_without_the_field_are_excluded_and_reported():
    r = run(fs(Op.SUM), [log("a", 1, idx=0), log("b", 2, idx=1, field="other")])
    assert r.value == 1
    assert [kind for _, kind in r.provenance.excluded] == [ExclusionKind.FIELD_NOT_CARRIED]


def test_declared_fills_are_coherent_with_the_missingness_rule_and_keep_the_state():
    rule = MissingnessRule(MASK, AbsenceSpec.fill(0), MASK, AbsenceSpec.fill(-1), MASK)
    spec = fs(Op.SUM, missingness=rule)
    empty = run(spec, [log("a", OV.empty_response())])
    assert (empty.state, empty.value, declared_fill(spec, empty)) == (R.EMPTY_RESPONSE, None, 0)
    null = run(spec, [log("a", OV.null())])
    assert (null.state, declared_fill(spec, null)) == (R.NULL, -1)
    for value in (OV.unknown(), OV.not_indexed()):
        assert declared_fill(spec, run(spec, [log("a", value)])) is None
    assert declared_fill(spec, run(spec, [])) is None  # not received is never filled
    assert declared_fill(spec, run(spec, [log("a", 5)])) is None


# ------------------------------------------------------------------ conflicts and identity


def test_conflicting_values_for_one_object_make_the_result_a_conflict():
    ledger = [log("a", 1, tx="t", source="p1"), log("b", 2, minute=2, tx="t", source="p2")]
    r = run(fs(Op.SUM), ledger)
    assert (r.state, r.value, r.issue) == (R.CONFLICT, None, AggregationIssue.OBJECT_CONFLICT)
    assert r.provenance.conflicts and r.provenance.conflicts[0].startswith("content:")


def test_single_requires_distinct_objects_to_agree():
    agree = run(fs(Op.SINGLE), [log("a", 3, idx=0), log("b", 3, idx=1)])
    assert (agree.state, agree.value) == (R.OBSERVED, 3)
    clash = run(fs(Op.SINGLE), [log("a", 3, idx=0), log("b", 4, idx=1)])
    assert (clash.state, clash.issue) == (R.CONFLICT, AggregationIssue.INCOMPATIBLE_VALUES)


def test_one_and_true_are_not_the_same_value():
    spec = fs(Op.SINGLE, dtype=FeatureDType.CATEGORY)
    r = run(spec, [log("a", 1, idx=0), log("b", True, idx=1)])
    assert r.state is R.CONFLICT


def test_unresolved_identity_blocks_duplicate_sensitive_operations_only():
    ledger = [log("a", 5, idx=0), log("b", 7, omit=["log_index"], tx="b")]
    for op in (Op.SUM, Op.COUNT_DISTINCT, Op.SINGLE):
        r = run(fs(op), ledger)
        assert (r.state, r.issue) == (R.AMBIGUOUS, AggregationIssue.UNRESOLVED_IDENTITY)
    assert run(fs(Op.MAX), ledger).value == 7  # duplicates cannot change a maximum
    assert run(fs(Op.MAX), ledger).provenance.unresolved_observation_hashes


def test_orphaned_objects_are_excluded_and_reported():
    ledger = load_ledger_json(
        FIXTURES / "synthetic" / "ledger_identity.json", require_synthetic=True
    )
    snapshot = make_asof_snapshot("2026-03-01T10:30:00+00:00", ledger)
    spec = fs(Op.SUM, kind=ST.ONCHAIN_RECEIPT, field="gas_used")
    r = aggregate_feature(spec, snapshot, ENTITY)
    # receipts: s1 (21000, one object) + s7 canonical (30000); the reorged one is excluded
    assert (r.state, r.value) == (R.OBSERVED, 51000)
    assert [kind for _, kind in r.provenance.excluded] == [ExclusionKind.ORPHANED]


def test_unresolved_placement_is_a_conflict():
    ledger = load_ledger_json(
        FIXTURES / "adversarial" / "ledger_identity_adversarial.json", require_synthetic=True
    )
    snapshot = make_asof_snapshot("2026-03-01T10:30:00+00:00", ledger)
    r = aggregate_feature(fs(Op.SUM, kind=ST.ONCHAIN_RECEIPT, field="gas_used"), snapshot, ENTITY)
    assert (r.state, r.value) == (R.CONFLICT, None)


@pytest.mark.parametrize(
    ("spec", "value"),
    [
        (fs(Op.SUM), "text"),
        (fs(Op.SUM), 1.5),  # INT feature fed with a float
        (fs(Op.SUM, dtype=FeatureDType.FLOAT), True),
        (fs(Op.SINGLE, dtype=FeatureDType.BOOL), 1),
        (fs(Op.SINGLE, dtype=FeatureDType.CATEGORY), 3),
    ],
)
def test_values_of_the_wrong_type_give_an_explicit_invalid_input(spec, value):
    r = run(spec, [log("a", value)])
    assert (r.state, r.issue, r.value) == (R.INVALID_INPUT, AggregationIssue.TYPE_MISMATCH, None)


# ------------------------------------------------------------------ time: cutoff and windows


def test_evidence_available_after_the_cutoff_never_enters_the_aggregate():
    ledger = [log("a", 1, minute=5, idx=0), log("late", 100, minute=45, idx=1)]
    r = run(fs(Op.SUM), ledger)
    assert r.value == 1 and r.provenance.cutoff == CUTOFF
    assert run(fs(Op.SUM), ledger, cutoff=at(60)).value == 101


def test_window_bounds_are_open_below_and_closed_above():
    window = WindowSpec(OrderingClock.RECEIVED_AT, 1800)  # (10:00, 10:30]
    ledger = [
        log("low", 1, minute=0, idx=0),
        log("in", 10, minute=1, idx=1),
        log("edge", 100, minute=30, idx=2),
    ]
    r = run(fs(Op.SUM, window=window), ledger)
    assert r.value == 110
    assert [kind for _, kind in r.provenance.excluded] == [ExclusionKind.OUT_OF_WINDOW]


def test_window_on_block_time_and_objects_without_that_clock_are_ambiguous():
    window = WindowSpec(OrderingClock.BLOCK_TIME, 600)  # (10:20, 10:30]
    ledger = [log("old", 1, block=5, minute=6, idx=0), log("new", 2, block=25, minute=26, idx=1)]
    assert run(fs(Op.SUM, window=window), ledger).value == 2
    external = make_synthetic_event(
        "ext",
        block_time=None,
        received_at=at(2),
        source_type=ST.EXTERNAL_RECORD,
        fields={"v": OV.observed(1)},
    )
    r = run(fs(Op.MAX, kind=ST.EXTERNAL_RECORD, window=window), [external])
    assert (r.state, r.issue) == (R.AMBIGUOUS, AggregationIssue.MISSING_ORDER_TIMESTAMP)


def test_ordering_by_an_absent_clock_is_ambiguous_not_guessed():
    spec = fs(Op.LATEST, kind=ST.EXTERNAL_RECORD, order=OrderingClock.BLOCK_TIME)
    external = make_synthetic_event(
        "ext",
        block_time=None,
        received_at=at(2),
        source_type=ST.EXTERNAL_RECORD,
        fields={"v": OV.observed(1)},
    )
    assert run(spec, [external]).issue is AggregationIssue.MISSING_ORDER_TIMESTAMP


# ------------------------------------------------------------------ provenance and determinism


def test_provenance_traces_the_value_back_to_its_evidence():
    spec = fs(Op.SUM, version=3, rule=2)
    ledger = [log("a", 1, minute=2, idx=0, source="p1"), log("b", 2, minute=9, idx=1, source="p2")]
    snapshot = make_asof_snapshot(CUTOFF, ledger)
    p = aggregate_feature(spec, snapshot, ENTITY).provenance
    assert (p.feature, p.feature_version, p.op, p.rule_version) == ("feature_x", 3, Op.SUM, 2)
    assert p.aggregation_contract == AGGREGATION_CONTRACT
    assert (p.entity_id, p.snapshot_hash, p.cutoff) == (ENTITY, snapshot.snapshot_hash, CUTOFF)
    assert p.sources == ("p1", "p2")
    assert list(p.object_key_ids) == sorted(p.object_key_ids) and len(p.object_key_ids) == 2
    assert p.observation_hashes == tuple(sorted(e.observation_hash for e in snapshot.events))
    assert p.available_at == at(9)  # when the last contributing object became available
    assert p.result_state is R.OBSERVED


def test_results_are_deterministic_and_the_ledger_is_untouched():
    ledger = [log(f"e{i}", i, idx=i, minute=1 + i % 3) for i in range(8)]
    snapshot_before = make_asof_snapshot(CUTOFF, ledger)
    reference = aggregate_feature(fs(Op.SUM), snapshot_before, ENTITY)
    rng = random.Random(3)
    for _ in range(6):
        shuffled = ledger[:]
        rng.shuffle(shuffled)
        again = aggregate_feature(fs(Op.SUM), make_asof_snapshot(CUTOFF, shuffled), ENTITY)
        assert again.result_hash == reference.result_hash
    assert aggregate_feature(fs(Op.SUM), snapshot_before, ENTITY) == reference


def test_changing_the_rule_version_changes_the_result_hash_but_not_the_value():
    ledger = [log("a", 3)]
    first = run(fs(Op.SUM, rule=1), ledger)
    second = run(fs(Op.SUM, rule=2), ledger)
    assert first.value == second.value and first.result_hash != second.result_hash
