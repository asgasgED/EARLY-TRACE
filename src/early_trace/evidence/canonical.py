"""Consolidation of admitted observations into canonical evidence objects.

Many observations can describe one object. Consolidation groups them by canonical
identity (and by affected entity, since an object may relate to several entities)
without discarding any observation, and classifies how each one relates to the object:

* ``FIRST``: the earliest observation (the object's first availability).
* ``DUPLICATE``: same object and payload through another provider, method or time.
* ``INDEPENDENT_CONFIRMATION``: like a duplicate, but both observations declare audited,
  different infrastructure groups. Independence is never inferred from provider names.
* ``CONFLICTING``: the group holds an incompatible payload, placement or finality state;
  no observation is preferred, so nothing is resolved arbitrarily.
* ``UNRESOLVED``: the identity could not be established; such observations are never merged.

Reorganizations are represented, not reconciled: the same transaction under different
blocks is reported as a placement divergence and each placement keeps its own finality.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum

from early_trace.evidence.events import LedgerEvent, ObservedValue, SourceType
from early_trace.evidence.identity import Identity, ObjectKey
from early_trace.evidence.observation import FinalityStatus
from early_trace.evidence.provenance import ProvenanceRecord
from early_trace.temporal.availability import TemporalAnomaly


@dataclass(frozen=True, slots=True)
class AdmittedEvent:
    """An observation that is part of F0, with its provenance, identity and anomalies."""

    event: LedgerEvent
    provenance: ProvenanceRecord
    identity: Identity
    anomalies: tuple[TemporalAnomaly, ...] = ()

    @property
    def observation_hash(self) -> str:
        return self.provenance.observation_hash

    @property
    def payload_hash(self) -> str:
        return self.provenance.payload_hash


class ObservationRole(str, Enum):
    FIRST = "first"
    DUPLICATE = "duplicate"
    INDEPENDENT_CONFIRMATION = "independent_confirmation"
    CONFLICTING = "conflicting"
    UNRESOLVED = "unresolved"


class ConflictKind(str, Enum):
    CONTENT = "content"
    INTERPRETATION = "interpretation"
    PLACEMENT = "placement"
    FINALITY = "finality"


class PlacementStatus(str, Enum):
    SINGLE = "single"
    DIVERGENT_CANONICAL = "divergent_canonical"
    DIVERGENT_NON_CANONICAL = "divergent_non_canonical"
    DIVERGENT_UNRESOLVED = "divergent_unresolved"


class EvidenceStanding(str, Enum):
    """Whether an object may feed an aggregation; derived, never chosen by the caller."""

    USABLE = "usable"
    UNRESOLVED_IDENTITY = "unresolved_identity"
    ORPHANED = "orphaned"
    SUPERSEDED = "superseded"
    CONFLICTED = "conflicted"
    PLACEMENT_UNRESOLVED = "placement_unresolved"


@dataclass(frozen=True, slots=True)
class ObservationLink:
    observation: AdmittedEvent
    role: ObservationRole


@dataclass(frozen=True, slots=True)
class EvidenceConflict:
    kind: ConflictKind
    observation_hashes: tuple[str, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class FinalityPoint:
    received_at: datetime
    finality: FinalityStatus


@dataclass(frozen=True, slots=True)
class PlacementDivergence:
    """One transaction observed under different blocks (a reorganization candidate)."""

    chain_id: int
    tx_hash: str
    block_hashes: tuple[str, ...]
    canonical_block_hash: str | None
    resolved: bool


@dataclass(frozen=True, slots=True)
class CanonicalEvidence:
    """One evidence object as seen for one entity, with every observation preserved."""

    identity: Identity
    entity_id: str
    observations: tuple[ObservationLink, ...]
    conflicts: tuple[EvidenceConflict, ...]
    finality_history: tuple[FinalityPoint, ...]
    latest_finality: FinalityStatus
    placement: PlacementStatus = PlacementStatus.SINGLE
    standing: EvidenceStanding = EvidenceStanding.USABLE

    @property
    def kind(self) -> SourceType:
        return self.identity.kind

    @property
    def key_id(self) -> str | None:
        return self.identity.key_id if isinstance(self.identity, ObjectKey) else None

    @property
    def group_id(self) -> tuple[str, str]:
        head = self.key_id or f"unresolved:{self.observations[0].observation.observation_hash}"
        return (head, self.entity_id)

    @property
    def first_received_at(self) -> datetime:
        return self.observations[0].observation.provenance.received_at_utc

    @property
    def block_time_utc(self) -> datetime | None:
        return self.observations[0].observation.provenance.block_time_utc

    @property
    def payload_hashes(self) -> tuple[str, ...]:
        return tuple(sorted({link.observation.payload_hash for link in self.observations}))

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(sorted({link.observation.event.source for link in self.observations}))

    @property
    def fields(self) -> Mapping[str, ObservedValue]:
        """Decoded fields; meaningful only when the object has no payload conflict."""
        return dict(self.observations[0].observation.event.fields)


@dataclass(frozen=True, slots=True)
class ConsolidatedEvidence:
    objects: tuple[CanonicalEvidence, ...]
    placement_divergences: tuple[PlacementDivergence, ...]


def _ordered(items: Iterable[AdmittedEvent]) -> list[AdmittedEvent]:
    return sorted(items, key=lambda a: (a.provenance.received_at_utc, a.observation_hash))


def _resolve_finality(points: list[FinalityPoint]) -> FinalityStatus:
    """Latest known finality; contradictory states at the same instant stay unknown."""
    if not points:
        return FinalityStatus.UNKNOWN
    last = points[-1].received_at
    states = {p.finality for p in points if p.received_at == last}
    return next(iter(states)) if len(states) == 1 else FinalityStatus.UNKNOWN


def _finality_points(observations: list[AdmittedEvent]) -> list[FinalityPoint]:
    return [
        FinalityPoint(a.provenance.received_at_utc, a.event.context.finality)
        for a in observations
        if a.event.context.finality is not FinalityStatus.UNKNOWN
    ]


def _group_conflicts(kind: SourceType, observations: list[AdmittedEvent]) -> list[EvidenceConflict]:
    hashes = tuple(sorted(a.observation_hash for a in observations))
    conflicts: list[EvidenceConflict] = []

    by_interpretation: dict[tuple[str | None, ...], set[str]] = defaultdict(set)
    for a in observations:
        by_interpretation[a.event.context.interpretation_key].add(a.payload_hash)
    if len({a.payload_hash for a in observations}) > 1:
        same_context = any(len(payloads) > 1 for payloads in by_interpretation.values())
        conflicts.append(
            EvidenceConflict(
                ConflictKind.CONTENT if same_context else ConflictKind.INTERPRETATION,
                hashes,
                "incompatible payloads under the same interpretation"
                if same_context
                else "payloads differ across decoder/parser/interpretation versions",
            )
        )

    if kind is SourceType.ONCHAIN_TRANSACTION:
        placements = {
            (a.event.block_number, a.event.block_hash, a.provenance.block_time_utc)
            for a in observations
        }
        if len(placements) > 1:
            conflicts.append(
                EvidenceConflict(
                    ConflictKind.PLACEMENT,
                    hashes,
                    "one transaction observed under different blocks",
                )
            )

    by_instant: dict[datetime, set[FinalityStatus]] = defaultdict(set)
    for p in _finality_points(observations):
        by_instant[p.received_at].add(p.finality)
    if any(len(states) > 1 for states in by_instant.values()):
        conflicts.append(
            EvidenceConflict(
                ConflictKind.FINALITY, hashes, "contradictory finality states at the same instant"
            )
        )
    return conflicts


def _roles(observations: list[AdmittedEvent], conflicted: bool) -> list[ObservationRole]:
    if conflicted:
        return [ObservationRole.CONFLICTING] * len(observations)
    first_group = observations[0].event.context.infrastructure_group
    roles = [ObservationRole.FIRST]
    for a in observations[1:]:
        group = a.event.context.infrastructure_group
        independent = first_group is not None and group is not None and group != first_group
        roles.append(
            ObservationRole.INDEPENDENT_CONFIRMATION if independent else ObservationRole.DUPLICATE
        )
    return roles


def _build_group(
    identity: Identity, entity_id: str, members: list[AdmittedEvent]
) -> CanonicalEvidence:
    observations = _ordered(members)
    unresolved = not isinstance(identity, ObjectKey)
    conflicts = [] if unresolved else _group_conflicts(identity.kind, observations)
    points = _finality_points(observations)
    if unresolved:
        roles = [ObservationRole.UNRESOLVED] * len(observations)
    else:
        roles = _roles(observations, bool(conflicts))
    return CanonicalEvidence(
        identity=identity,
        entity_id=entity_id,
        observations=tuple(ObservationLink(a, r) for a, r in zip(observations, roles, strict=True)),
        conflicts=tuple(conflicts),
        finality_history=tuple(points),
        latest_finality=_resolve_finality(points),
    )


def _placement_finality(observations: list[AdmittedEvent]) -> FinalityStatus:
    return _resolve_finality(_finality_points(_ordered(observations)))


def _analyze_placements(
    admitted: Iterable[AdmittedEvent],
) -> dict[tuple[int, str], PlacementDivergence]:
    by_tx: dict[tuple[int, str], dict[str, list[AdmittedEvent]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for a in admitted:
        e = a.event
        if e.chain_id is not None and e.tx_hash is not None and e.block_hash is not None:
            by_tx[(e.chain_id, e.tx_hash)][e.block_hash].append(a)

    result: dict[tuple[int, str], PlacementDivergence] = {}
    for (chain_id, tx_hash), blocks in by_tx.items():
        if len(blocks) < 2:
            continue
        finality = {b: _placement_finality(obs) for b, obs in blocks.items()}
        finalized = sorted(b for b, f in finality.items() if f is FinalityStatus.FINALIZED)
        alive = sorted(b for b, f in finality.items() if f is not FinalityStatus.REORGED)
        canonical: str | None = None
        if len(finalized) == 1:
            canonical = finalized[0]
        elif not finalized and len(alive) == 1:
            canonical = alive[0]
        result[(chain_id, tx_hash)] = PlacementDivergence(
            chain_id, tx_hash, tuple(sorted(blocks)), canonical, canonical is not None
        )
    return result


def _single_block(obj: CanonicalEvidence) -> str | None:
    blocks = {link.observation.event.block_hash for link in obj.observations}
    return next(iter(blocks)) if len(blocks) == 1 else None


def _placement_of(
    obj: CanonicalEvidence, divergences: dict[tuple[int, str], PlacementDivergence]
) -> PlacementStatus:
    first = obj.observations[0].observation.event
    if first.chain_id is None or first.tx_hash is None:
        return PlacementStatus.SINGLE
    divergence = divergences.get((first.chain_id, first.tx_hash))
    if divergence is None:
        return PlacementStatus.SINGLE
    block = _single_block(obj)
    if block is None or not divergence.resolved:
        return PlacementStatus.DIVERGENT_UNRESOLVED
    if block == divergence.canonical_block_hash:
        return PlacementStatus.DIVERGENT_CANONICAL
    return PlacementStatus.DIVERGENT_NON_CANONICAL


def _standing(obj: CanonicalEvidence) -> EvidenceStanding:
    if not isinstance(obj.identity, ObjectKey):
        return EvidenceStanding.UNRESOLVED_IDENTITY
    if obj.latest_finality is FinalityStatus.REORGED:
        return EvidenceStanding.ORPHANED
    if obj.placement is PlacementStatus.DIVERGENT_NON_CANONICAL:
        return EvidenceStanding.SUPERSEDED
    if obj.conflicts:
        return EvidenceStanding.CONFLICTED
    if obj.placement is PlacementStatus.DIVERGENT_UNRESOLVED:
        return EvidenceStanding.PLACEMENT_UNRESOLVED
    return EvidenceStanding.USABLE


def consolidate(admitted: Iterable[AdmittedEvent]) -> ConsolidatedEvidence:
    """Group observations into canonical evidence; order and result are deterministic."""
    items = list(admitted)
    groups: dict[tuple[str, str], tuple[Identity, list[AdmittedEvent]]] = {}
    for a in items:
        head = (
            a.identity.key_id
            if isinstance(a.identity, ObjectKey)
            else f"unresolved:{a.observation_hash}"
        )
        slot = groups.setdefault((head, a.event.entity_id), (a.identity, []))
        slot[1].append(a)

    divergences = _analyze_placements(items)
    objects: list[CanonicalEvidence] = []
    for (_, entity_id), (identity, members) in sorted(groups.items(), key=lambda kv: kv[0]):
        obj = _build_group(identity, entity_id, members)
        obj = replace(obj, placement=_placement_of(obj, divergences))
        objects.append(replace(obj, standing=_standing(obj)))

    ordered = tuple(divergences[k] for k in sorted(divergences))
    return ConsolidatedEvidence(tuple(objects), ordered)
