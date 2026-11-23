"""Task 01: the order and meaning of schema features are immutable across versions."""

from __future__ import annotations

from dataclasses import replace

import pytest

from early_trace.evidence.events import SourceType
from early_trace.schemas.aggregation import AggregationOp, AggregationSpec, OrderingClock
from early_trace.schemas.feature_schema import (
    AbsenceSpec,
    AvailabilityClock,
    EvolutionIssueKind,
    FeatureDType,
    FeatureSchema,
    FeatureSchemaError,
    FeatureSpec,
    MissingnessRule,
    SchemaEvolutionError,
    SchemaVersion,
    check_evolution,
    validate_evolution,
)

V1 = SchemaVersion.parse("1.0.0")
V2 = SchemaVersion.parse("2.0.0")
V3 = SchemaVersion.parse("3.0.0")
MASK = AbsenceSpec.mask_only()
RULE = MissingnessRule(MASK, MASK, MASK, MASK, MASK)
KINDS = EvolutionIssueKind


def feat(name: str, version: SchemaVersion = V1, **overrides) -> FeatureSpec:
    base = dict(
        name=name,
        dtype=FeatureDType.INT,
        description=f"Synthetic feature {name}.",
        source_type=SourceType.ONCHAIN_LOG,
        source_field="transfer_value",
        availability_clock=AvailabilityClock.RECEIVED_AT,
        missingness=RULE,
        aggregation=AggregationSpec(AggregationOp.SUM, 1),
        feature_version=1,
        introduced_in=version,
    )
    base.update(overrides)
    return FeatureSpec(**base)


def schema(*features: FeatureSpec, version: SchemaVersion = V1) -> FeatureSchema:
    return FeatureSchema(name="early_trace_features", version=version, features=features)


def issues(
    previous: FeatureSchema, new: FeatureSchema
) -> list[tuple[EvolutionIssueKind, str | None]]:
    return [(i.kind, i.feature) for i in check_evolution(previous, new)]


V1_SCHEMA = schema(feat("address_age"), feat("transaction_count"), feat("incoming_volume"))


def test_appending_a_feature_at_the_end_is_valid():
    new = schema(*V1_SCHEMA.features, feat("outgoing_volume", V2), version=V2)
    validate_evolution(V1_SCHEMA, new)
    assert check_evolution(V1_SCHEMA, new) == ()


def test_inserting_a_feature_in_the_middle_is_rejected_and_located():
    new = schema(
        V1_SCHEMA.get("address_age"),
        feat("outgoing_volume", V2),
        V1_SCHEMA.get("transaction_count"),
        V1_SCHEMA.get("incoming_volume"),
        version=V2,
    )
    assert issues(V1_SCHEMA, new) == [(KINDS.FEATURE_INSERTED_NOT_AT_END, "outgoing_volume")]
    (issue,) = check_evolution(V1_SCHEMA, new)
    assert "position 1" in issue.detail and "transaction_count" in issue.detail


def test_reordering_existing_features_is_rejected_and_names_every_displaced_feature():
    # the example of the task: positions 1 and 2 are swapped while a new feature is appended
    new = schema(
        V1_SCHEMA.get("address_age"),
        V1_SCHEMA.get("incoming_volume"),
        V1_SCHEMA.get("transaction_count"),
        feat("outgoing_volume", V2),
        version=V2,
    )
    found = issues(V1_SCHEMA, new)
    assert (KINDS.FEATURE_MOVED, "transaction_count") in found
    assert (KINDS.FEATURE_MOVED, "incoming_volume") in found
    assert len(found) == 2
    details = {i.feature: i.detail for i in check_evolution(V1_SCHEMA, new)}
    assert "from position 1 to 2" in details["transaction_count"]
    assert "from position 2 to 1" in details["incoming_volume"]


def test_removing_a_feature_is_rejected():
    new = schema(V1_SCHEMA.get("address_age"), V1_SCHEMA.get("incoming_volume"), version=V2)
    assert issues(V1_SCHEMA, new) == [(KINDS.FEATURE_REMOVED, "transaction_count")]


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": FeatureDType.FLOAT},
        {"source_type": SourceType.ONCHAIN_RECEIPT},
        {"source_field": "other_field"},
        {"description": "Now means something else."},
        {"unit": "wei"},
        {"missingness": MissingnessRule(MASK, AbsenceSpec.fill(0), MASK, MASK, MASK)},
    ],
)
def test_semantic_change_without_feature_version_bump_is_rejected(change):
    changed = replace(V1_SCHEMA.get("transaction_count"), **change)
    new = schema(
        V1_SCHEMA.get("address_age"), changed, V1_SCHEMA.get("incoming_volume"), version=V2
    )
    (issue,) = check_evolution(V1_SCHEMA, new)
    assert issue.kind is KINDS.CHANGED_WITHOUT_VERSION_BUMP
    assert next(iter(change)) in issue.detail


def test_semantic_change_with_feature_version_bump_is_valid():
    changed = replace(
        V1_SCHEMA.get("transaction_count"), dtype=FeatureDType.FLOAT, feature_version=2
    )
    new = schema(
        V1_SCHEMA.get("address_age"), changed, V1_SCHEMA.get("incoming_volume"), version=V2
    )
    validate_evolution(V1_SCHEMA, new)


def test_aggregation_change_needs_both_feature_and_rule_version_bumps():
    old = schema(feat("total_value"))
    swapped = AggregationSpec(AggregationOp.MAX, 1)

    silent = schema(replace(old.get("total_value"), aggregation=swapped), version=V2)
    assert {k for k, _ in issues(old, silent)} == {
        KINDS.CHANGED_WITHOUT_VERSION_BUMP,
        KINDS.AGGREGATION_CHANGED_WITHOUT_RULE_BUMP,
    }

    only_feature = replace(old.get("total_value"), aggregation=swapped, feature_version=2)
    assert issues(old, schema(only_feature, version=V2)) == [
        (KINDS.AGGREGATION_CHANGED_WITHOUT_RULE_BUMP, "total_value")
    ]

    proper = replace(
        old.get("total_value"), aggregation=AggregationSpec(AggregationOp.MAX, 2), feature_version=2
    )
    validate_evolution(old, schema(proper, version=V2))


def test_window_or_order_clock_change_counts_as_an_aggregation_change():
    old = schema(
        feat(
            "latest_value",
            aggregation=AggregationSpec(AggregationOp.LATEST, 1, OrderingClock.BLOCK_TIME),
        )
    )
    flipped = AggregationSpec(AggregationOp.LATEST, 1, OrderingClock.RECEIVED_AT)
    new = schema(
        replace(old.get("latest_value"), aggregation=flipped, feature_version=2), version=V2
    )
    assert issues(old, new) == [(KINDS.AGGREGATION_CHANGED_WITHOUT_RULE_BUMP, "latest_value")]


def test_identical_schemas_are_reported_as_identical():
    assert issues(V1_SCHEMA, V1_SCHEMA) == [(KINDS.IDENTICAL_SCHEMA, None)]
    with pytest.raises(SchemaEvolutionError):
        validate_evolution(V1_SCHEMA, V1_SCHEMA)


def test_version_must_increase():
    new = schema(*V1_SCHEMA.features, feat("outgoing_volume", V1), version=V1)
    assert (KINDS.VERSION_NOT_INCREASED, None) in issues(V1_SCHEMA, new)


def test_new_feature_must_declare_an_introducing_version_after_the_previous_one():
    backdated = schema(*V1_SCHEMA.features, feat("outgoing_volume", V1), version=V2)
    assert (KINDS.NEW_FEATURE_WRONG_INTRODUCED_IN, "outgoing_volume") in issues(
        V1_SCHEMA, backdated
    )
    # a feature cannot be introduced by a version later than the schema that contains it
    with pytest.raises(FeatureSchemaError):
        schema(*V1_SCHEMA.features, feat("outgoing_volume", V3), version=V2)


def test_existing_feature_cannot_change_its_introduced_in():
    moved = replace(V1_SCHEMA.get("address_age"), introduced_in=V2, feature_version=2)
    new = schema(
        moved, V1_SCHEMA.get("transaction_count"), V1_SCHEMA.get("incoming_volume"), version=V2
    )
    assert (KINDS.INTRODUCED_IN_CHANGED, "address_age") in issues(V1_SCHEMA, new)


def test_schema_name_change_is_rejected():
    other = FeatureSchema(name="other_schema", version=V2, features=V1_SCHEMA.features)
    assert issues(V1_SCHEMA, other) == [(KINDS.SCHEMA_NAME_CHANGED, None)]


def test_all_problems_are_reported_together_and_deterministically():
    new = schema(
        V1_SCHEMA.get("incoming_volume"),
        replace(V1_SCHEMA.get("address_age"), dtype=FeatureDType.FLOAT),
        feat("outgoing_volume", V1),
        version=V1,
    )
    first = check_evolution(V1_SCHEMA, new)
    kinds = {i.kind for i in first}
    assert {
        KINDS.VERSION_NOT_INCREASED,
        KINDS.FEATURE_REMOVED,
        KINDS.CHANGED_WITHOUT_VERSION_BUMP,
        KINDS.FEATURE_MOVED,
    } <= kinds
    assert check_evolution(V1_SCHEMA, new) == first
    with pytest.raises(SchemaEvolutionError) as excinfo:
        validate_evolution(V1_SCHEMA, new)
    assert excinfo.value.issues == first


def test_feature_order_is_the_column_order_and_part_of_the_hash():
    swapped = schema(*reversed(V1_SCHEMA.features))
    assert V1_SCHEMA.schema_hash != swapped.schema_hash
    assert [V1_SCHEMA.column_index(n) for n in V1_SCHEMA.feature_names] == [0, 1, 2]
    with pytest.raises(KeyError):
        V1_SCHEMA.column_index("nope")


def test_serialization_roundtrip_preserves_order_and_hash():
    restored = FeatureSchema.from_dict(V1_SCHEMA.to_dict())
    assert restored.feature_names == V1_SCHEMA.feature_names
    assert restored.schema_hash == V1_SCHEMA.schema_hash
    assert restored == V1_SCHEMA


def test_chained_evolution_keeps_every_original_position():
    v2 = schema(*V1_SCHEMA.features, feat("outgoing_volume", V2), version=V2)
    v3 = schema(*v2.features, feat("counterparties", V3), version=V3)
    validate_evolution(V1_SCHEMA, v2)
    validate_evolution(v2, v3)
    validate_evolution(V1_SCHEMA, v3)
    assert v3.feature_names[:3] == V1_SCHEMA.feature_names


def test_several_new_features_can_be_appended_in_one_version():
    new = schema(
        *V1_SCHEMA.features,
        feat("outgoing_volume", V2),
        feat("counterparties", V2),
        feat("burstiness", V2),
        version=V2,
    )
    validate_evolution(V1_SCHEMA, new)
    assert new.feature_names[3:] == ("outgoing_volume", "counterparties", "burstiness")


def test_one_misplaced_feature_among_several_new_ones_is_the_only_one_reported():
    new = schema(
        V1_SCHEMA.get("address_age"),
        feat("outgoing_volume", V2),
        V1_SCHEMA.get("transaction_count"),
        V1_SCHEMA.get("incoming_volume"),
        feat("counterparties", V2),
        version=V2,
    )
    assert issues(V1_SCHEMA, new) == [(KINDS.FEATURE_INSERTED_NOT_AT_END, "outgoing_volume")]
