"""Reference semantics of the aggregation contract (``schemas.aggregation``).

``aggregate_feature`` builds the value of *one* feature for *one* entity from an as-of
snapshot. It is the executable definition of the contract, not the Feature Builder: it
builds no matrix and no mask. Taking the snapshot (never raw ledger events) makes it
impossible to aggregate evidence that was not available at the cutoff.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from early_trace.evidence.canonical import CanonicalEvidence, EvidenceStanding, ObservationRole
from early_trace.evidence.events import ObservedValue, Scalar, ValueState
from early_trace.infrastructure.hashing import sha256_hex
from early_trace.schemas.aggregation import (
    AggregationOp,
    OrderingClock,
    ResultState,
)
from early_trace.schemas.feature_schema import AbsenceTreatment, FeatureDType, FeatureSpec
from early_trace.temporal.snapshot import AsOfSnapshot


class AggregationIssue(str, Enum):
    """Why a result is ``CONFLICT``, ``AMBIGUOUS`` or ``INVALID_INPUT``."""

    OBJECT_CONFLICT = "object_conflict"
    PLACEMENT_UNRESOLVED = "placement_unresolved"
    INCOMPATIBLE_VALUES = "incompatible_values"
    TIE_WITH_INCOMPATIBLE_VALUES = "tie_with_incompatible_values"
    UNRESOLVED_IDENTITY = "unresolved_identity"
    MISSING_ORDER_TIMESTAMP = "missing_order_timestamp"
    TYPE_MISMATCH = "type_mismatch"


class ExclusionKind(str, Enum):
    ORPHANED = "orphaned"
    SUPERSEDED = "superseded"
    OUT_OF_WINDOW = "out_of_window"
    FIELD_NOT_CARRIED = "field_not_carried"


@dataclass(frozen=True, slots=True)
class AggregationProvenance:
    """Everything needed to trace how a value was obtained (ET24 §7.2)."""

    feature: str
    feature_version: int
    op: AggregationOp
    rule_version: int
    aggregation_contract: str
    entity_id: str
    snapshot_hash: str
    cutoff: datetime
    object_key_ids: tuple[str, ...]
    unresolved_observation_hashes: tuple[str, ...]
    observation_hashes: tuple[str, ...]
    duplicate_observation_hashes: tuple[str, ...]
    confirmation_observation_hashes: tuple[str, ...]
    sources: tuple[str, ...]
    excluded: tuple[tuple[str, ExclusionKind], ...]
    conflicts: tuple[str, ...]
    available_at: datetime | None
    result_state: ResultState

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "feature_version": self.feature_version,
            "op": self.op.value,
            "rule_version": self.rule_version,
            "aggregation_contract": self.aggregation_contract,
            "entity_id": self.entity_id,
            "snapshot_hash": self.snapshot_hash,
            "cutoff": self.cutoff.isoformat(timespec="microseconds"),
            "object_key_ids": list(self.object_key_ids),
            "unresolved_observation_hashes": list(self.unresolved_observation_hashes),
            "observation_hashes": list(self.observation_hashes),
            "duplicate_observation_hashes": list(self.duplicate_observation_hashes),
            "confirmation_observation_hashes": list(self.confirmation_observation_hashes),
            "sources": list(self.sources),
            "excluded": [[ref, kind.value] for ref, kind in self.excluded],
            "conflicts": list(self.conflicts),
            "available_at": None
            if self.available_at is None
            else self.available_at.isoformat(timespec="microseconds"),
            "result_state": self.result_state.value,
        }


@dataclass(frozen=True, slots=True)
class AggregationResult:
    feature: str
    state: ResultState
    value: Scalar | None
    issue: AggregationIssue | None
    provenance: AggregationProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "state": self.state.value,
            "value": self.value,
            "value_type": None if self.value is None else type(self.value).__name__,
            "issue": None if self.issue is None else self.issue.value,
            "provenance": self.provenance.to_dict(),
        }

    @property
    def result_hash(self) -> str:
        return sha256_hex(self.to_dict())


def declared_fill(spec: FeatureSpec, result: AggregationResult) -> Scalar | None:
    """Fill the schema declares for an absence state, if any (the mask keeps the state)."""
    rule = spec.missingness.treatment_for(result.state)
    if rule is not None and rule.treatment is AbsenceTreatment.FILL_DECLARED:
        return rule.fill_value
    return None


def _ref(obj: CanonicalEvidence) -> str:
    return obj.key_id or f"unresolved:{obj.observations[0].observation.observation_hash}"


def _instant(obj: CanonicalEvidence, clock: OrderingClock) -> datetime | None:
    return obj.first_received_at if clock is OrderingClock.RECEIVED_AT else obj.block_time_utc


def _signature(value: ObservedValue) -> tuple[str, str, Scalar | None]:
    return (value.state.value, type(value.value).__name__, value.value)


def _numeric(value: Scalar | None) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _type_ok(dtype: FeatureDType, value: Scalar | None) -> bool:
    if dtype is FeatureDType.INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if dtype is FeatureDType.FLOAT:
        return _numeric(value)
    if dtype is FeatureDType.BOOL:
        return isinstance(value, bool)
    return isinstance(value, str)


class _Collector:
    """Accumulates provenance while the operation runs."""

    def __init__(self, spec: FeatureSpec, snapshot: AsOfSnapshot, entity_id: str) -> None:
        self.spec = spec
        self.snapshot = snapshot
        self.entity_id = entity_id
        self.excluded: list[tuple[str, ExclusionKind]] = []
        self.used: list[CanonicalEvidence] = []

    def result(
        self, state: ResultState, value: Scalar | None = None, issue: AggregationIssue | None = None
    ) -> AggregationResult:
        used = sorted(self.used, key=lambda o: o.group_id)
        links = [link for obj in used for link in obj.observations]
        agg = self.spec.aggregation
        conflicts = sorted(f"{c.kind.value}:{_ref(obj)}" for obj in used for c in obj.conflicts)
        provenance = AggregationProvenance(
            feature=self.spec.name,
            feature_version=self.spec.feature_version,
            op=agg.op,
            rule_version=agg.rule_version,
            aggregation_contract=agg.to_dict()["contract"],
            entity_id=self.entity_id,
            snapshot_hash=self.snapshot.snapshot_hash,
            cutoff=self.snapshot.cutoff,
            object_key_ids=tuple(k for k in (o.key_id for o in used) if k is not None),
            unresolved_observation_hashes=tuple(
                sorted(
                    link.observation.observation_hash
                    for o in used
                    if o.key_id is None
                    for link in o.observations
                )
            ),
            observation_hashes=tuple(sorted(link.observation.observation_hash for link in links)),
            duplicate_observation_hashes=tuple(
                sorted(
                    link.observation.observation_hash
                    for link in links
                    if link.role is ObservationRole.DUPLICATE
                )
            ),
            confirmation_observation_hashes=tuple(
                sorted(
                    link.observation.observation_hash
                    for link in links
                    if link.role is ObservationRole.INDEPENDENT_CONFIRMATION
                )
            ),
            sources=tuple(sorted({s for o in used for s in o.providers})),
            excluded=tuple(sorted(self.excluded, key=lambda e: (e[0], e[1].value))),
            conflicts=tuple(conflicts),
            available_at=max((o.first_received_at for o in used), default=None),
            result_state=state,
        )
        return AggregationResult(self.spec.name, state, value, issue, provenance)


def _absence_state(values: Sequence[ObservedValue]) -> ResultState | None:
    """Result state when any participating value is absent; one shared state or ``UNKNOWN``."""
    absent = {v.state for v in values if v.state is not ValueState.OBSERVED}
    if not absent:
        return None
    return ResultState(next(iter(absent)).value) if len(absent) == 1 else ResultState.UNKNOWN


def _select(
    spec: FeatureSpec, entries: list[tuple[CanonicalEvidence, ObservedValue]], c: _Collector
) -> AggregationResult:
    """``single`` / ``latest`` / ``earliest``: pick the object(s) deciding the value."""
    agg = spec.aggregation
    chosen = entries
    tie = False
    if agg.op is not AggregationOp.SINGLE:
        assert agg.order_clock is not None
        stamped: list[tuple[datetime, tuple[CanonicalEvidence, ObservedValue]]] = []
        for obj, value in entries:
            instant = _instant(obj, agg.order_clock)
            if instant is None:
                c.used = [o for o, _ in entries]
                return c.result(
                    ResultState.AMBIGUOUS, None, AggregationIssue.MISSING_ORDER_TIMESTAMP
                )
            stamped.append((instant, (obj, value)))
        pick = max if agg.op is AggregationOp.LATEST else min
        edge = pick(instant for instant, _ in stamped)
        chosen = [entry for instant, entry in stamped if instant == edge]
        tie = len(chosen) > 1

    c.used = [o for o, _ in chosen]
    if len({_signature(v) for _, v in chosen}) > 1:
        issue = (
            AggregationIssue.TIE_WITH_INCOMPATIBLE_VALUES
            if tie
            else AggregationIssue.INCOMPATIBLE_VALUES
        )
        return c.result(ResultState.CONFLICT, None, issue)

    value = chosen[0][1]
    if value.state is not ValueState.OBSERVED:
        return c.result(ResultState(value.state.value))
    if not _type_ok(spec.dtype, value.value):
        return c.result(ResultState.INVALID_INPUT, None, AggregationIssue.TYPE_MISMATCH)
    return c.result(ResultState.OBSERVED, value.value)


def _reduce(
    spec: FeatureSpec, entries: list[tuple[CanonicalEvidence, ObservedValue]], c: _Collector
) -> AggregationResult:
    """``count_distinct`` / ``sum`` / ``min`` / ``max`` / ``mean`` over all participants."""
    op = spec.aggregation.op
    c.used = [o for o, _ in entries]
    absence = _absence_state([v for _, v in entries])
    if absence is not None:
        return c.result(absence)
    if op is AggregationOp.COUNT_DISTINCT:
        return c.result(ResultState.OBSERVED, len(entries))

    values = [v.value for _, v in entries]
    if not all(_numeric(x) for x in values) or (
        spec.dtype is FeatureDType.INT and not all(isinstance(x, int) for x in values)
    ):
        return c.result(ResultState.INVALID_INPUT, None, AggregationIssue.TYPE_MISMATCH)
    numbers: list[int | float] = [x for x in values if isinstance(x, (int, float))]
    result: int | float
    if op is AggregationOp.MIN:
        result = min(numbers)
    elif op is AggregationOp.MAX:
        result = max(numbers)
    elif spec.dtype is FeatureDType.INT:
        result = sum(int(x) for x in numbers)
    else:
        total = math.fsum(float(x) for x in numbers)
        result = total / len(numbers) if op is AggregationOp.MEAN else total
    if spec.dtype is FeatureDType.FLOAT:
        result = float(result)
    return c.result(ResultState.OBSERVED, result)


def aggregate_feature(
    spec: FeatureSpec, snapshot: AsOfSnapshot, entity_id: str
) -> AggregationResult:
    """Aggregate ``spec`` for ``entity_id`` from the objects admitted in ``snapshot``."""
    agg = spec.aggregation
    c = _Collector(spec, snapshot, entity_id)

    scope: list[CanonicalEvidence] = []
    for obj in snapshot.objects:
        if obj.entity_id != entity_id or obj.kind is not spec.source_type:
            continue
        if obj.standing is EvidenceStanding.ORPHANED:
            c.excluded.append((_ref(obj), ExclusionKind.ORPHANED))
        elif obj.standing is EvidenceStanding.SUPERSEDED:
            c.excluded.append((_ref(obj), ExclusionKind.SUPERSEDED))
        else:
            scope.append(obj)

    if agg.window is not None:
        low = snapshot.cutoff - timedelta(seconds=agg.window.length_seconds)
        kept: list[CanonicalEvidence] = []
        for obj in scope:
            instant = _instant(obj, agg.window.clock)
            if instant is None:
                c.used = scope
                return c.result(
                    ResultState.AMBIGUOUS, None, AggregationIssue.MISSING_ORDER_TIMESTAMP
                )
            if low < instant <= snapshot.cutoff:
                kept.append(obj)
            else:
                c.excluded.append((_ref(obj), ExclusionKind.OUT_OF_WINDOW))
        scope = kept

    entries: list[tuple[CanonicalEvidence, ObservedValue]] = []
    for obj in scope:
        value = obj.fields.get(spec.source_field)
        if value is None:
            c.excluded.append((_ref(obj), ExclusionKind.FIELD_NOT_CARRIED))
        else:
            entries.append((obj, value))

    if not entries:
        return c.result(ResultState.NOT_RECEIVED)

    c.used = [o for o, _ in entries]
    for obj, _ in entries:
        if obj.standing is EvidenceStanding.CONFLICTED:
            return c.result(ResultState.CONFLICT, None, AggregationIssue.OBJECT_CONFLICT)
        if obj.standing is EvidenceStanding.PLACEMENT_UNRESOLVED:
            return c.result(ResultState.CONFLICT, None, AggregationIssue.PLACEMENT_UNRESOLVED)
    if agg.duplicate_sensitive and any(
        o.standing is EvidenceStanding.UNRESOLVED_IDENTITY for o, _ in entries
    ):
        return c.result(ResultState.AMBIGUOUS, None, AggregationIssue.UNRESOLVED_IDENTITY)

    if agg.op in (AggregationOp.SINGLE, AggregationOp.LATEST, AggregationOp.EARLIEST):
        return _select(spec, entries, c)
    return _reduce(spec, entries, c)
