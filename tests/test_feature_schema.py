"""FeatureSchema validation and append-only evolution."""

from __future__ import annotations

from dataclasses import replace

import pytest

from early_trace.evidence.events import SourceType, ValueState
from early_trace.schemas.feature_schema import (
    AbsenceSpec,
    AvailabilityClock,
    DuplicateFeatureError,
    FeatureDType,
    FeatureSchema,
    FeatureSchemaError,
    FeatureSpec,
    InvalidFeatureSpecError,
    InvalidSchemaVersionError,
    MissingnessRule,
    SchemaEvolutionError,
    SchemaVersion,
    validate_evolution,
)

V100 = SchemaVersion.parse("1.0.0")
V110 = SchemaVersion.parse("1.1.0")


def rule(empty: AbsenceSpec | None = None) -> MissingnessRule:
    return MissingnessRule(
        unknown=AbsenceSpec.mask_only(),
        empty_response=empty or AbsenceSpec.mask_only(),
        not_indexed=AbsenceSpec.mask_only(),
    )


def spec(name: str = "transfer_count", **overrides) -> FeatureSpec:
    base = dict(
        name=name,
        dtype=FeatureDType.INT,
        description="Number of incoming transfers observed for the entity.",
        source_type=SourceType.ONCHAIN_LOG,
        source_field="transfer_count",
        availability_clock=AvailabilityClock.RECEIVED_AT,
        missingness=rule(),
        feature_version=1,
        introduced_in=V100,
        unit="transfers",
    )
    base.update(overrides)
    return FeatureSpec(**base)


def schema(*features: FeatureSpec, version: SchemaVersion = V100) -> FeatureSchema:
    return FeatureSchema(name="early_trace_features", version=version, features=features)


def test_valid_schema_roundtrips_with_stable_hash():
    original = schema(
        spec(), spec("is_contract", dtype=FeatureDType.BOOL, unit=None, source_field="is_contract")
    )
    restored = FeatureSchema.from_dict(original.to_dict())
    assert restored == original
    assert restored.schema_hash == original.schema_hash
    assert original.feature_names == ("transfer_count", "is_contract")


def test_schema_hash_changes_when_a_definition_changes():
    assert (
        schema(spec()).schema_hash
        != schema(spec(description="A different meaning entirely.")).schema_hash
    )


def test_duplicate_names_rejected():
    with pytest.raises(DuplicateFeatureError):
        schema(spec(), spec())


@pytest.mark.parametrize("dtype", ["decimal", "", None, 3])
def test_unsupported_dtype_rejected(dtype):
    raw = spec().to_dict()
    raw["dtype"] = dtype
    with pytest.raises(InvalidFeatureSpecError):
        FeatureSpec.from_dict(raw)


@pytest.mark.parametrize(
    "key", ["description", "source_type", "availability_clock", "missingness", "feature_version"]
)
def test_incomplete_definition_rejected(key):
    raw = spec().to_dict()
    del raw[key]
    with pytest.raises(InvalidFeatureSpecError):
        FeatureSpec.from_dict(raw)


def test_unexpected_keys_rejected():
    raw = spec().to_dict()
    raw["importance"] = 0.9
    with pytest.raises(InvalidFeatureSpecError):
        FeatureSpec.from_dict(raw)


@pytest.mark.parametrize(
    "overrides", [{"description": "  "}, {"name": "BadName"}, {"unit": " "}, {"feature_version": 0}]
)
def test_invalid_definitions_rejected(overrides):
    with pytest.raises(InvalidFeatureSpecError):
        spec(**overrides)


@pytest.mark.parametrize("text", ["1.0", "0.0.0", "01.0.0", "1.0.0-beta", "-1.0.0", "", "v1.0.0"])
def test_invalid_versions_rejected(text):
    with pytest.raises(InvalidSchemaVersionError):
        SchemaVersion.parse(text)


def test_feature_cannot_be_introduced_after_schema_version():
    with pytest.raises(FeatureSchemaError):
        schema(spec(introduced_in=V110), version=V100)


def test_block_time_is_not_an_availability_clock():
    raw = spec().to_dict()
    raw["availability_clock"] = "block_time"
    with pytest.raises(InvalidFeatureSpecError):
        FeatureSpec.from_dict(raw)


def test_unknown_and_not_indexed_can_never_be_filled():
    for name in ("unknown", "not_indexed"):
        kwargs = {
            "unknown": AbsenceSpec.mask_only(),
            "empty_response": AbsenceSpec.mask_only(),
            "not_indexed": AbsenceSpec.mask_only(),
        }
        kwargs[name] = AbsenceSpec.fill(0)
        with pytest.raises(InvalidFeatureSpecError):
            MissingnessRule(**kwargs)


def test_ambiguous_absence_declarations_rejected():
    with pytest.raises(InvalidFeatureSpecError):
        AbsenceSpec(treatment="fill_declared", fill_value=0)  # type: ignore[arg-type]
    with pytest.raises(InvalidFeatureSpecError):
        AbsenceSpec.fill(None)  # type: ignore[arg-type]
    with pytest.raises(InvalidFeatureSpecError):
        AbsenceSpec(AbsenceSpec.mask_only().treatment, 0)


def test_empty_response_fill_must_match_dtype():
    assert (
        spec(missingness=rule(AbsenceSpec.fill(0)))
        .missingness.spec_for(ValueState.EMPTY_RESPONSE)
        .fill_value
        == 0
    )
    with pytest.raises(InvalidFeatureSpecError):
        spec(missingness=rule(AbsenceSpec.fill("zero")))
    with pytest.raises(InvalidFeatureSpecError):
        spec(missingness=rule(AbsenceSpec.fill(True)))  # bool is not an int here


def test_evolution_append_only_is_accepted():
    old = schema(spec())
    new = schema(
        spec(),
        spec(
            "is_contract",
            dtype=FeatureDType.BOOL,
            unit=None,
            source_field="is_contract",
            introduced_in=V110,
        ),
        version=V110,
    )
    validate_evolution(old, new)


def test_evolution_rejects_silent_redefinition():
    old = schema(spec())
    changed = spec(description="Now counts outgoing transfers instead.")
    with pytest.raises(SchemaEvolutionError):
        validate_evolution(old, schema(changed, version=V110))


def test_evolution_allows_explicit_redefinition_with_feature_version_bump():
    old = schema(spec())
    changed = spec(description="Now counts outgoing transfers instead.", feature_version=2)
    validate_evolution(old, schema(changed, version=V110))


def test_evolution_rejects_removal_and_non_increasing_version():
    old = schema(
        spec(), spec("is_contract", dtype=FeatureDType.BOOL, unit=None, source_field="is_contract")
    )
    with pytest.raises(SchemaEvolutionError):
        validate_evolution(old, schema(spec(), version=V110))
    with pytest.raises(SchemaEvolutionError):
        validate_evolution(old, replace(old))


def test_evolution_rejects_new_feature_backdated_to_old_version():
    old = schema(spec())
    backdated = spec(
        "is_contract",
        dtype=FeatureDType.BOOL,
        unit=None,
        source_field="is_contract",
        introduced_in=V100,
    )
    with pytest.raises(SchemaEvolutionError):
        validate_evolution(old, schema(spec(), backdated, version=V110))
