"""As-of snapshot (F0): the evidence admissible for a decision taken at ``cutoff``.

``make_asof_snapshot`` is pure: it never mutates the ledger, and identical
inputs always yield the same snapshot and the same ``snapshot_hash``.

Four hashes, never reused for different concepts:
* ``ObjectKey.key_id``: canonical identity of the evidence object (provider-independent).
* ``payload_hash``: provider-independent content of one (object, entity) link.
* ``observation_hash``: everything about one observation (provider, method, versions,
  reception time, finality, payload).
* ``snapshot_hash``: cutoff plus the ordered observation hashes (contract
  ``asof-snapshot/2``).

Record policy (conservative, no heuristics):
* A repeated ledger record with an identical ``observation_hash`` is the same record
  ingested twice; the repeat is excluded as ``DUPLICATE_RECORD`` and stays traceable.
* Different observations of the same object (other provider, method, decoder or time)
  are *all* kept; ``objects`` groups them into canonical evidence so they are not
  counted as new evidence (see ``early_trace.evidence.canonical``).
* Records sharing an ``event_id`` but differing in content are not merged: both are kept
  and the id is listed in ``conflicting_event_ids`` (a ledger-level id collision).
* Consolidation only ever sees observations already admitted by availability, so future
  records cannot influence the past. Temporal anomalies are recorded, not discarded.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from early_trace.evidence.canonical import (
    AdmittedEvent,
    CanonicalEvidence,
    PlacementDivergence,
    consolidate,
)
from early_trace.evidence.events import LedgerContractError, LedgerEvent
from early_trace.evidence.identity import ObjectKey, resolve_identity
from early_trace.evidence.provenance import ProvenanceRecord, build_provenance, missing_provenance
from early_trace.infrastructure.hashing import sha256_hex
from early_trace.temporal.availability import (
    ExclusionReason,
    TemporalAnomaly,
    TemporalPolicy,
    check_availability,
    normalize_cutoff,
)

__all__ = ["F0", "AdmittedEvent", "AsOfSnapshot", "ExcludedEvent", "make_asof_snapshot"]

SNAPSHOT_CONTRACT = "asof-snapshot/2"
PAYLOAD_CONTRACT = "evidence-payload/1"


@dataclass(frozen=True, slots=True)
class ExcludedEvent:
    """An event left out of F0 with an explicit, typed reason."""

    event: LedgerEvent
    reason: ExclusionReason
    detail: str
    record_hash: str


@dataclass(frozen=True, slots=True)
class AsOfSnapshot:
    cutoff: datetime
    events: tuple[AdmittedEvent, ...]
    exclusions: tuple[ExcludedEvent, ...]
    conflicting_event_ids: tuple[str, ...]
    objects: tuple[CanonicalEvidence, ...]
    placement_divergences: tuple[PlacementDivergence, ...]
    snapshot_hash: str
    contract: str = SNAPSHOT_CONTRACT

    @property
    def provenance(self) -> tuple[ProvenanceRecord, ...]:
        return tuple(item.provenance for item in self.events)

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(item.event.event_id for item in self.events)

    @property
    def anomalies(self) -> tuple[tuple[str, TemporalAnomaly], ...]:
        """``(observation_hash, anomaly)`` for every admitted observation with anomalies."""
        return tuple((a.observation_hash, x) for a in self.events for x in a.anomalies)

    def exclusion_counts(self) -> dict[str, int]:
        counts = Counter(item.reason.value for item in self.exclusions)
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "cutoff": self.cutoff.isoformat(timespec="microseconds"),
            "snapshot_hash": self.snapshot_hash,
            "admitted_count": len(self.events),
            "excluded_count": len(self.exclusions),
            "exclusion_counts": self.exclusion_counts(),
            "conflicting_event_ids": list(self.conflicting_event_ids),
            "object_count": len(self.objects),
            "anomalies": [
                {
                    "observation_hash": observation_hash,
                    "kind": anomaly.kind.value,
                    "status": anomaly.status.value,
                    "early_by_seconds": anomaly.early_by_seconds,
                }
                for observation_hash, anomaly in self.anomalies
            ],
            "objects": [
                {
                    "key_id": obj.key_id,
                    "kind": obj.kind.value,
                    "entity_id": obj.entity_id,
                    "standing": obj.standing.value,
                    "placement": obj.placement.value,
                    "latest_finality": obj.latest_finality.value,
                    "conflicts": [
                        {"kind": c.kind.value, "detail": c.detail} for c in obj.conflicts
                    ],
                    "observations": [
                        {
                            "observation_hash": link.observation.observation_hash,
                            "role": link.role.value,
                            "source": link.observation.event.source,
                        }
                        for link in obj.observations
                    ],
                }
                for obj in self.objects
            ],
            "placement_divergences": [
                {
                    "chain_id": d.chain_id,
                    "tx_hash": d.tx_hash,
                    "block_hashes": list(d.block_hashes),
                    "canonical_block_hash": d.canonical_block_hash,
                    "resolved": d.resolved,
                }
                for d in self.placement_divergences
            ],
            "admitted": [item.provenance.to_dict() for item in self.events],
            "exclusions": [
                {
                    "event_id": item.event.event_id,
                    "reason": item.reason.value,
                    "detail": item.detail,
                    "record_hash": item.record_hash,
                }
                for item in self.exclusions
            ],
        }


F0 = AsOfSnapshot


def _exclude(event: LedgerEvent, reason: ExclusionReason, detail: str) -> ExcludedEvent:
    return ExcludedEvent(event, reason, detail, sha256_hex(event.raw_dict()))


def _admission_key(item: AdmittedEvent) -> tuple[datetime, str, str]:
    return (item.provenance.received_at_utc, item.event.event_id, item.observation_hash)


def make_asof_snapshot(
    cutoff: object, ledger: Iterable[LedgerEvent], policy: TemporalPolicy | None = None
) -> AsOfSnapshot:
    """Build F0 for a decision at ``cutoff`` from ``ledger``.

    ``policy`` carries explicit, source-specific temporal rules; without it no clock
    anomaly causes an exclusion.

    Raises ``InvalidCutoffError`` for an unusable cutoff and
    ``LedgerContractError`` if the ledger holds anything but ``LedgerEvent``.
    """
    cutoff_utc = normalize_cutoff(cutoff)
    exclusions: list[ExcludedEvent] = []
    candidates: list[AdmittedEvent] = []

    for event in tuple(ledger):
        if not isinstance(event, LedgerEvent):
            raise LedgerContractError(f"ledger item is not a LedgerEvent: {type(event).__name__}")

        result = check_availability(event, cutoff_utc, policy)
        if not result.is_available:
            assert result.reason is not None
            exclusions.append(_exclude(event, result.reason, result.detail))
            continue

        missing = missing_provenance(event)
        if missing:
            exclusions.append(
                _exclude(
                    event,
                    ExclusionReason.INCOMPLETE_PROVENANCE,
                    f"missing required provenance: {', '.join(missing)}",
                )
            )
            continue

        assert result.received_at_utc is not None
        identity = resolve_identity(event)
        payload_hash = sha256_hex(
            {
                "payload": PAYLOAD_CONTRACT,
                "object": identity.key_id if isinstance(identity, ObjectKey) else None,
                **event.payload_dict(result.block_time_utc),
            }
        )
        observation_hash = sha256_hex(
            event.observation_dict(result.block_time_utc, result.received_at_utc)
        )
        provenance = build_provenance(
            event,
            result.block_time_utc,
            result.received_at_utc,
            identity=identity,
            payload_hash=payload_hash,
            observation_hash=observation_hash,
        )
        candidates.append(AdmittedEvent(event, provenance, identity, result.anomalies))

    admitted: list[AdmittedEvent] = []
    seen_hashes: set[str] = set()
    for item in sorted(candidates, key=_admission_key):
        if item.observation_hash in seen_hashes:
            exclusions.append(
                _exclude(
                    item.event,
                    ExclusionReason.DUPLICATE_RECORD,
                    f"identical observation already admitted: {item.observation_hash}",
                )
            )
            continue
        seen_hashes.add(item.observation_hash)
        admitted.append(item)

    versions = Counter(item.event.event_id for item in admitted)
    conflicting = tuple(sorted(event_id for event_id, n in versions.items() if n > 1))

    consolidated = consolidate(admitted)
    exclusions.sort(key=lambda item: (item.event.event_id, item.reason.value, item.record_hash))
    snapshot_hash = sha256_hex(
        {
            "contract": SNAPSHOT_CONTRACT,
            "cutoff": cutoff_utc.isoformat(timespec="microseconds"),
            "observations": [item.observation_hash for item in admitted],
        }
    )
    return AsOfSnapshot(
        cutoff=cutoff_utc,
        events=tuple(admitted),
        exclusions=tuple(exclusions),
        conflicting_event_ids=conflicting,
        objects=consolidated.objects,
        placement_divergences=consolidated.placement_divergences,
        snapshot_hash=snapshot_hash,
    )
