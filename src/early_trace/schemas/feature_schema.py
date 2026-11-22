"""Typed, versioned feature schema.

A schema documents and validates features; it does not decide which features
help detection. Rules enforced here:

* Every feature declares its evidence source, availability clock and how each
  absence state (unknown / empty response / not indexed) is treated. Nothing is
  defaulted, so absence can never be ambiguous.
* ``unknown`` and ``not_indexed`` can only be masked, never filled with a value.
* Availability is decided by ``received_at`` only; ``block_time`` is not an
  available clock.
* Evolution is append-only: an existing feature keeps its meaning unless its
  ``feature_version`` is explicitly bumped (see ``validate_evolution``).
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


class SchemaEvolutionError(FeatureSchemaError):
    """A new schema silently changes or drops what an older one defined."""


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


@dataclass(frozen=True, slots=True)
class MissingnessRule:
    """Explicit treatment of every absence state; all three must be declared."""

    unknown: AbsenceSpec
    empty_response: AbsenceSpec
    not_indexed: AbsenceSpec

    def __post_init__(self) -> None:
        for name in ("unknown", "empty_response", "not_indexed"):
            if not isinstance(getattr(self, name), AbsenceSpec):
                raise InvalidFeatureSpecError(f"missingness.{name} must be an AbsenceSpec")
        for name in ("unknown", "not_indexed"):
            if getattr(self, name).treatment is not AbsenceTreatment.MASK_ONLY:
                raise InvalidFeatureSpecError(f"'{name}' may only be masked, never filled")

    def spec_for(self, state: ValueState) -> AbsenceSpec:
        if state is ValueState.OBSERVED:
            raise ValueError("an observed value has no absence rule")
        return cast(AbsenceSpec, getattr(self, state.value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "unknown": self.unknown.to_dict(),
            "empty_response": self.empty_response.to_dict(),
            "not_indexed": self.not_indexed.to_dict(),
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
        for state in (ValueState.EMPTY_RESPONSE,):
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
            "feature_version": self.feature_version,
            "introduced_in": str(self.introduced_in),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FeatureSpec:
        required = {
            "name", "dtype", "description", "source_type", "source_field",
            "availability_clock", "missingness", "feature_version", "introduced_in",
        }  # fmt: skip
        _check_keys(raw, required, {"unit"}, "feature")
        miss = raw["missingness"]
        _check_keys(miss, {"unknown", "empty_response", "not_indexed"}, set(), "missingness")
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
            ),
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


def validate_evolution(previous: FeatureSchema, new: FeatureSchema) -> None:
    """Raise ``SchemaEvolutionError`` unless ``new`` extends ``previous`` safely.

    Requires a strictly greater version, no removed features, no change to a
    feature's definition without a higher ``feature_version``, and new features
    introduced after the previous version.
    """
    if previous.name != new.name:
        raise SchemaEvolutionError(f"schema name changed: {previous.name!r} -> {new.name!r}")
    if not previous.version < new.version:
        raise SchemaEvolutionError(f"version must increase: {previous.version} -> {new.version}")

    current = {f.name: f for f in new.features}
    for old in previous.features:
        updated = current.get(old.name)
        if updated is None:
            raise SchemaEvolutionError(f"feature '{old.name}' was removed")
        if updated == old:
            continue
        if updated.introduced_in != old.introduced_in:
            raise SchemaEvolutionError(f"feature '{old.name}': introduced_in cannot change")
        if updated.feature_version <= old.feature_version:
            raise SchemaEvolutionError(
                f"feature '{old.name}' changed without increasing feature_version"
            )

    previous_names = {f.name for f in previous.features}
    for feature in new.features:
        if feature.name not in previous_names and not previous.version < feature.introduced_in:
            raise SchemaEvolutionError(
                f"new feature '{feature.name}' must be introduced after {previous.version}"
            )
