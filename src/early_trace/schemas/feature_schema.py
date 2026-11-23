"""Typed, versioned feature schema.

A schema documents and validates features; it does not decide which features
help detection. Rules enforced here:

* Every feature declares its evidence source, availability clock, aggregation rule and
  how each absence state (null / unknown / empty response / not indexed / not received)
  is treated. Nothing is defaulted, so neither absence nor aggregation can be implicit.
* ``unknown``, ``not_indexed`` and ``not_received`` can only be masked, never filled.
* Availability is decided by ``received_at`` only; ``block_time`` is not an
  available clock.
* Evolution is strictly append-only (see ``check_evolution``): existing features keep
  their position and meaning, new features go at the end, and any semantic change
  (including the aggregation rule) needs an explicit version bump. Because the order of
  ``features`` is the column order of the future matrix, it is part of the contract.

Serialization ``feature-schema/2`` adds ``aggregation`` to each feature and the ``null_value``
and ``not_received`` rules; documents lacking them are rejected, never defaulted.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from early_trace.evidence.events import Scalar, SourceType, ValueState
from early_trace.infrastructure.hashing import sha256_hex
from early_trace.schemas.aggregation import (
    AggregationOp,
    AggregationSpec,
    AggregationSpecError,
    ResultState,
)

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class FeatureSchemaError(ValueError):
    """Base class for schema validation errors."""


class InvalidFeatureSpecError(FeatureSchemaError):
    """A feature definition is incomplete or inconsistent."""


class DuplicateFeatureError(FeatureSchemaError):
    """Two features share the same name."""


class InvalidSchemaVersionError(FeatureSchemaError):
    """A schema version is malformed."""


class EvolutionIssueKind(str, Enum):
    SCHEMA_NAME_CHANGED = "schema_name_changed"
    IDENTICAL_SCHEMA = "identical_schema"
    VERSION_NOT_INCREASED = "version_not_increased"
    FEATURE_REMOVED = "feature_removed"
    FEATURE_MOVED = "feature_moved"
    FEATURE_INSERTED_NOT_AT_END = "feature_inserted_not_at_end"
    CHANGED_WITHOUT_VERSION_BUMP = "changed_without_version_bump"
    AGGREGATION_CHANGED_WITHOUT_RULE_BUMP = "aggregation_changed_without_rule_bump"
    INTRODUCED_IN_CHANGED = "introduced_in_changed"
    NEW_FEATURE_WRONG_INTRODUCED_IN = "new_feature_wrong_introduced_in"


@dataclass(frozen=True, slots=True)
class EvolutionIssue:
    """One precise reason why a schema change is not a valid evolution."""

    kind: EvolutionIssueKind
    feature: str | None
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind.value}] {self.detail}"


class SchemaEvolutionError(FeatureSchemaError):
    """A new schema silently changes, reorders or drops what an older one defined."""

    def __init__(self, issues: tuple[EvolutionIssue, ...]) -> None:
        self.issues = issues
        super().__init__("invalid schema evolution: " + "; ".join(str(i) for i in issues))


class FeatureDType(str, Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CATEGORY = "category"


class AvailabilityClock(str, Enum):
    """Clock that decides when a feature's evidence becomes usable."""

    RECEIVED_AT = "received_at"


class AbsenceTreatment(str, Enum):
    MASK_ONLY = "mask_only"
    FILL_DECLARED = "fill_declared"


@dataclass(frozen=True, slots=True)
class SchemaVersion:
    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        parts = (self.major, self.minor, self.patch)
        if any(isinstance(p, bool) or not isinstance(p, int) or p < 0 for p in parts):
            raise InvalidSchemaVersionError(f"version parts must be non-negative integers: {parts}")
        if parts == (0, 0, 0):
            raise InvalidSchemaVersionError("version 0.0.0 is not allowed")

    @classmethod
    def parse(cls, text: str) -> SchemaVersion:
        match = _VERSION_RE.fullmatch(text) if isinstance(text, str) else None
        if match is None:
            raise InvalidSchemaVersionError(f"invalid schema version: {text!r}")
        return cls(*(int(part) for part in match.groups()))

    def _key(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def __lt__(self, other: SchemaVersion) -> bool:
        return self._key() < other._key()

    def __le__(self, other: SchemaVersion) -> bool:
        return self._key() <= other._key()

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def _value_matches(dtype: FeatureDType, value: object) -> bool:
    if dtype is FeatureDType.BOOL:
        return isinstance(value, bool)
    if dtype is FeatureDType.INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if dtype is FeatureDType.FLOAT:
        return isinstance(value, float) and math.isfinite(value)
    return isinstance(value, str)


@dataclass(frozen=True, slots=True)
class AbsenceSpec:
    treatment: AbsenceTreatment
    fill_value: Scalar | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.treatment, AbsenceTreatment):
            raise InvalidFeatureSpecError(f"invalid absence treatment: {self.treatment!r}")
        if self.treatment is AbsenceTreatment.MASK_ONLY and self.fill_value is not None:
            raise InvalidFeatureSpecError("mask_only absence cannot declare a fill value")
        if self.treatment is AbsenceTreatment.FILL_DECLARED and self.fill_value is None:
            raise InvalidFeatureSpecError("fill_declared absence requires a fill value")

    @classmethod
    def mask_only(cls) -> AbsenceSpec:
        return cls(AbsenceTreatment.MASK_ONLY)

    @classmethod
    def fill(cls, value: Scalar) -> AbsenceSpec:
        return cls(AbsenceTreatment.FILL_DECLARED, value)

    def to_dict(self) -> dict[str, Any]:
        return {"treatment": self.treatment.value, "fill_value": self.fill_value}


_MASK_ONLY_STATES = ("unknown", "not_indexed", "not_received")
_VALUE_STATE_RULE = {
    ValueState.NULL: "null_value",
    ValueState.UNKNOWN: "unknown",
    ValueState.EMPTY_RESPONSE: "empty_response",
    ValueState.NOT_INDEXED: "not_indexed",
}
_RESULT_STATE_RULE = {
    ResultState.NULL: "null_value",
    ResultState.UNKNOWN: "unknown",
    ResultState.EMPTY_RESPONSE: "empty_response",
    ResultState.NOT_INDEXED: "not_indexed",
    ResultState.NOT_RECEIVED: "not_received",
}


@dataclass(frozen=True, slots=True)
class MissingnessRule:
    """Explicit treatment of every absence state; all five must be declared.

    ``null_value`` (the source returned null) and ``empty_response`` are defined technical
    states and may declare a fill; ``unknown``, ``not_indexed`` and ``not_received`` never can.
    Conflicts and ambiguities are not absences: they are always masked and never filled.
    """

    unknown: AbsenceSpec
    empty_response: AbsenceSpec
    not_indexed: AbsenceSpec
    null_value: AbsenceSpec
    not_received: AbsenceSpec

    def __post_init__(self) -> None:
        for name in ("unknown", "empty_response", "not_indexed", "null_value", "not_received"):
            if not isinstance(getattr(self, name), AbsenceSpec):
                raise InvalidFeatureSpecError(f"missingness.{name} must be an AbsenceSpec")
        for name in _MASK_ONLY_STATES:
            if getattr(self, name).treatment is not AbsenceTreatment.MASK_ONLY:
                raise InvalidFeatureSpecError(f"'{name}' may only be masked, never filled")

    def spec_for(self, state: ValueState) -> AbsenceSpec:
        if state is ValueState.OBSERVED:
            raise ValueError("an observed value has no absence rule")
        return cast(AbsenceSpec, getattr(self, _VALUE_STATE_RULE[state]))

    def treatment_for(self, state: ResultState) -> AbsenceSpec | None:
        """Rule for an aggregated absence state; ``None`` for states that are never filled."""
        name = _RESULT_STATE_RULE.get(state)
        return None if name is None else cast(AbsenceSpec, getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "unknown": self.unknown.to_dict(),
            "empty_response": self.empty_response.to_dict(),
            "not_indexed": self.not_indexed.to_dict(),
            "null_value": self.null_value.to_dict(),
            "not_received": self.not_received.to_dict(),
        }


_OP_DTYPES: dict[AggregationOp, frozenset[FeatureDType]] = {
    AggregationOp.SINGLE: frozenset(FeatureDType),
    AggregationOp.LATEST: frozenset(FeatureDType),
    AggregationOp.EARLIEST: frozenset(FeatureDType),
    AggregationOp.COUNT_DISTINCT: frozenset({FeatureDType.INT}),
    AggregationOp.SUM: frozenset({FeatureDType.INT, FeatureDType.FLOAT}),
    AggregationOp.MIN: frozenset({FeatureDType.INT, FeatureDType.FLOAT}),
    AggregationOp.MAX: frozenset({FeatureDType.INT, FeatureDType.FLOAT}),
    AggregationOp.MEAN: frozenset({FeatureDType.FLOAT}),
}


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Definition of one feature. ``source_field`` names the ledger event field it reads."""

    name: str
    dtype: FeatureDType
    description: str
    source_type: SourceType
    source_field: str
    availability_clock: AvailabilityClock
    missingness: MissingnessRule
    aggregation: AggregationSpec
    feature_version: int
    introduced_in: SchemaVersion
    unit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise InvalidFeatureSpecError(f"invalid feature name: {self.name!r}")
        if not isinstance(self.source_field, str) or not _NAME_RE.fullmatch(self.source_field):
            raise InvalidFeatureSpecError(f"invalid source_field: {self.source_field!r}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise InvalidFeatureSpecError(f"feature '{self.name}' needs a description")
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit.strip()):
            raise InvalidFeatureSpecError(f"feature '{self.name}' has an empty unit")
        for attr, kind in (
            ("dtype", FeatureDType),
            ("source_type", SourceType),
            ("availability_clock", AvailabilityClock),
            ("missingness", MissingnessRule),
            ("aggregation", AggregationSpec),
            ("introduced_in", SchemaVersion),
        ):
            if not isinstance(getattr(self, attr), kind):
                raise InvalidFeatureSpecError(
                    f"feature '{self.name}': {attr} must be {kind.__name__}"
                )
        if (
            isinstance(self.feature_version, bool)
            or not isinstance(self.feature_version, int)
            or self.feature_version < 1
        ):
            raise InvalidFeatureSpecError(f"feature '{self.name}': feature_version must be >= 1")
        if self.dtype not in _OP_DTYPES[self.aggregation.op]:
            raise InvalidFeatureSpecError(
                f"feature '{self.name}': aggregation '{self.aggregation.op.value}' "
                f"cannot produce dtype {self.dtype.value}"
            )
        for state in (ValueState.EMPTY_RESPONSE, ValueState.NULL):
            spec = self.missingness.spec_for(state)
            if spec.fill_value is not None and not _value_matches(self.dtype, spec.fill_value):
                raise InvalidFeatureSpecError(
                    f"feature '{self.name}': fill value {spec.fill_value!r} does not match "
                    f"dtype {self.dtype.value}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype.value,
            "description": self.description,
            "unit": self.unit,
            "source_type": self.source_type.value,
            "source_field": self.source_field,
            "availability_clock": self.availability_clock.value,
            "missingness": self.missingness.to_dict(),
            "aggregation": self.aggregation.to_dict(),
            "feature_version": self.feature_version,
            "introduced_in": str(self.introduced_in),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FeatureSpec:
        required = {
            "name", "dtype", "description", "source_type", "source_field",
            "availability_clock", "missingness", "aggregation", "feature_version",
            "introduced_in",
        }  # fmt: skip
        _check_keys(raw, required, {"unit"}, "feature")
        miss = raw["missingness"]
        _check_keys(
            miss,
            {"unknown", "empty_response", "not_indexed", "null_value", "not_received"},
            set(),
            "missingness",
        )
        try:
            aggregation = AggregationSpec.from_dict(raw["aggregation"])
        except AggregationSpecError as exc:
            raise InvalidFeatureSpecError(f"feature '{raw['name']}': {exc}") from exc
        return cls(
            name=raw["name"],
            dtype=_enum(FeatureDType, raw["dtype"], "dtype"),
            description=raw["description"],
            source_type=_enum(SourceType, raw["source_type"], "source_type"),
            source_field=raw["source_field"],
            availability_clock=_enum(
                AvailabilityClock, raw["availability_clock"], "availability_clock"
            ),
            missingness=MissingnessRule(
                unknown=_absence_from_dict(miss["unknown"]),
                empty_response=_absence_from_dict(miss["empty_response"]),
                not_indexed=_absence_from_dict(miss["not_indexed"]),
                null_value=_absence_from_dict(miss["null_value"]),
                not_received=_absence_from_dict(miss["not_received"]),
            ),
            aggregation=aggregation,
            feature_version=raw["feature_version"],
            introduced_in=SchemaVersion.parse(raw["introduced_in"]),
            unit=raw.get("unit"),
        )


def _check_keys(raw: object, required: set[str], optional: set[str], context: str) -> None:
    if not isinstance(raw, Mapping):
        raise InvalidFeatureSpecError(f"{context} must be a mapping")
    missing = required - set(raw)
    extra = set(raw) - required - optional
    if missing or extra:
        raise InvalidFeatureSpecError(
            f"{context}: missing keys {sorted(missing)}, unexpected keys {sorted(extra)}"
        )


def _enum(kind: type[Enum], raw: object, field: str) -> Any:
    try:
        return kind(raw)
    except ValueError as exc:
        raise InvalidFeatureSpecError(f"unsupported {field}: {raw!r}") from exc


def _absence_from_dict(raw: Mapping[str, Any]) -> AbsenceSpec:
    _check_keys(raw, {"treatment", "fill_value"}, set(), "absence spec")
    return AbsenceSpec(_enum(AbsenceTreatment, raw["treatment"], "treatment"), raw["fill_value"])


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Ordered, versioned collection of features; order defines column order of X."""

    name: str
    version: SchemaVersion
    features: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise FeatureSchemaError(f"invalid schema name: {self.name!r}")
        if not isinstance(self.version, SchemaVersion):
            raise InvalidSchemaVersionError("version must be a SchemaVersion")
        features = tuple(self.features)
        if not features:
            raise FeatureSchemaError("a schema needs at least one feature")
        if not all(isinstance(f, FeatureSpec) for f in features):
            raise FeatureSchemaError("features must be FeatureSpec instances")
        names = [f.name for f in features]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise DuplicateFeatureError(f"duplicate feature names: {duplicates}")
        for feature in features:
            if self.version < feature.introduced_in:
                raise FeatureSchemaError(
                    f"feature '{feature.name}' introduced in {feature.introduced_in}, "
                    f"after schema version {self.version}"
                )
        object.__setattr__(self, "features", features)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    def column_index(self, name: str) -> int:
        """Position of ``name``; this is its column in the future feature matrix."""
        try:
            return self.feature_names.index(name)
        except ValueError:
            raise KeyError(name) from None

    def get(self, name: str) -> FeatureSpec:
        for feature in self.features:
            if feature.name == name:
                return feature
        raise KeyError(name)

    @property
    def schema_hash(self) -> str:
        return sha256_hex(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": str(self.version),
            "features": [f.to_dict() for f in self.features],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FeatureSchema:
        _check_keys(raw, {"name", "version", "features"}, set(), "schema")
        if isinstance(raw["features"], (str, bytes)) or not isinstance(raw["features"], Iterable):
            raise FeatureSchemaError("features must be a list")
        return cls(
            name=raw["name"],
            version=SchemaVersion.parse(raw["version"]),
            features=tuple(FeatureSpec.from_dict(item) for item in raw["features"]),
        )


_DIFF_FIELDS = (
    "dtype",
    "description",
    "unit",
    "source_type",
    "source_field",
    "availability_clock",
    "missingness",
    "aggregation",
    "feature_version",
)


def _changed_fields(old: FeatureSpec, new: FeatureSpec) -> tuple[str, ...]:
    before, after = old.to_dict(), new.to_dict()
    return tuple(name for name in _DIFF_FIELDS if before[name] != after[name])


def _order_issues(previous: FeatureSchema, new: FeatureSchema) -> list[EvolutionIssue]:
    old_names = previous.feature_names
    new_names = new.feature_names
    issues: list[EvolutionIssue] = []

    survivors_old = [n for n in old_names if n in new_names]
    survivors_new = [n for n in new_names if n in old_names]
    if survivors_old != survivors_new:
        for name in survivors_old:
            if survivors_old.index(name) != survivors_new.index(name):
                issues.append(
                    EvolutionIssue(
                        EvolutionIssueKind.FEATURE_MOVED,
                        name,
                        f"'{name}' moved from position {old_names.index(name)} "
                        f"to {new_names.index(name)}; existing features keep their position",
                    )
                )

    last_old = max((new_names.index(n) for n in survivors_new), default=-1)
    for position, name in enumerate(new_names):
        if name not in old_names and position < last_old:
            displaced = next(n for n in new_names[position + 1 :] if n in old_names)
            issues.append(
                EvolutionIssue(
                    EvolutionIssueKind.FEATURE_INSERTED_NOT_AT_END,
                    name,
                    f"new feature '{name}' was inserted at position {position}, displacing "
                    f"existing feature '{displaced}'; new features may only be appended",
                )
            )
    return issues


def _definition_issues(old: FeatureSpec, new: FeatureSpec) -> list[EvolutionIssue]:
    if new.introduced_in != old.introduced_in:
        return [
            EvolutionIssue(
                EvolutionIssueKind.INTRODUCED_IN_CHANGED,
                old.name,
                f"'{old.name}': introduced_in changed from {old.introduced_in} "
                f"to {new.introduced_in}",
            )
        ]
    changed = _changed_fields(old, new)
    if not changed or changed == ("feature_version",):
        return []
    issues: list[EvolutionIssue] = []
    if new.feature_version <= old.feature_version:
        issues.append(
            EvolutionIssue(
                EvolutionIssueKind.CHANGED_WITHOUT_VERSION_BUMP,
                old.name,
                f"'{old.name}' changed ({', '.join(changed)}) without increasing feature_version "
                f"(still {new.feature_version})",
            )
        )
    if "aggregation" in changed and new.aggregation.rule_version <= old.aggregation.rule_version:
        issues.append(
            EvolutionIssue(
                EvolutionIssueKind.AGGREGATION_CHANGED_WITHOUT_RULE_BUMP,
                old.name,
                f"'{old.name}': aggregation changed without increasing rule_version "
                f"(still {new.aggregation.rule_version})",
            )
        )
    return issues


def check_evolution(previous: FeatureSchema, new: FeatureSchema) -> tuple[EvolutionIssue, ...]:
    """Every reason why ``new`` is not a valid append-only evolution of ``previous``."""
    if previous.name != new.name:
        return (
            EvolutionIssue(
                EvolutionIssueKind.SCHEMA_NAME_CHANGED,
                None,
                f"schema name changed: {previous.name!r} -> {new.name!r}",
            ),
        )
    if previous == new:
        return (
            EvolutionIssue(
                EvolutionIssueKind.IDENTICAL_SCHEMA,
                None,
                f"schemas are identical (version {new.version}); nothing to evolve",
            ),
        )

    issues: list[EvolutionIssue] = []
    if not previous.version < new.version:
        issues.append(
            EvolutionIssue(
                EvolutionIssueKind.VERSION_NOT_INCREASED,
                None,
                f"version must increase: {previous.version} -> {new.version}",
            )
        )

    current = {f.name: f for f in new.features}
    for old in previous.features:
        updated = current.get(old.name)
        if updated is None:
            issues.append(
                EvolutionIssue(
                    EvolutionIssueKind.FEATURE_REMOVED,
                    old.name,
                    f"feature '{old.name}' (position {previous.feature_names.index(old.name)}) "
                    "was removed",
                )
            )
        else:
            issues.extend(_definition_issues(old, updated))

    issues.extend(_order_issues(previous, new))

    previous_names = set(previous.feature_names)
    for feature in new.features:
        if feature.name in previous_names:
            continue
        if not previous.version < feature.introduced_in <= new.version:
            issues.append(
                EvolutionIssue(
                    EvolutionIssueKind.NEW_FEATURE_WRONG_INTRODUCED_IN,
                    feature.name,
                    f"new feature '{feature.name}' declares introduced_in {feature.introduced_in}"
                    f" but must be introduced after {previous.version} and by {new.version}",
                )
            )
    return tuple(issues)


def validate_evolution(previous: FeatureSchema, new: FeatureSchema) -> None:
    """Raise ``SchemaEvolutionError`` (with every issue in ``.issues``) unless ``new`` is valid.

    Valid means: same schema name, strictly greater version, no feature removed, existing
    features at their original positions, new features only appended with an introducing
    version in ``(previous.version, new.version]``, and any change to an existing feature
    carrying a higher ``feature_version`` (and a higher aggregation ``rule_version`` when
    the aggregation changed).
    """
    issues = check_evolution(previous, new)
    if issues:
        raise SchemaEvolutionError(issues)
