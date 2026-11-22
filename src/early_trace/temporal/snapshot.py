"""As-of snapshot (F0): the evidence admissible for a decision taken at ``cutoff``.

``make_asof_snapshot`` is pure: it never mutates the ledger, and identical
inputs always yield the same snapshot and the same ``snapshot_hash``.

Identity policy (conservative, no heuristics):
* Records whose normalized content is byte-identical (same ``event_id`` and
  identical canonical content) are one piece of evidence; the repeat is
  excluded as ``DUPLICATE_RECORD`` and stays traceable.
* Records sharing an ``event_id`` but differing in content are *not* merged or
  resolved: both are kept and the id is listed in ``conflicting_event_ids``.
* Deduplication only ever sees events already admitted by availability, so
  future records cannot influence the past.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from early_trace.evidence.events import LedgerContractError, LedgerEvent
from early_trace.evidence.provenance import ProvenanceRecord, build_provenance, missing_provenance
from early_trace.infrastructure.hashing import sha256_hex
from early_trace.temporal.availability import (
    ExclusionReason,
    check_availability,
    normalize_cutoff,
)

SNAPSHOT_CONTRACT = "asof-snapshot/1"


@dataclass(frozen=True, slots=True)
class AdmittedEvent:
    """An event that is part of F0, with its verifiable provenance."""

    event: LedgerEvent
    provenance: ProvenanceRecord

    @property
    def content_hash(self) -> str:
        return self.provenance.content_hash


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
    snapshot_hash: str
    contract: str = SNAPSHOT_CONTRACT

    @property
    def provenance(self) -> tuple[ProvenanceRecord, ...]:
        return tuple(item.provenance for item in self.events)

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(item.event.event_id for item in self.events)

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
    return (item.provenance.received_at_utc, item.event.event_id, item.content_hash)


def make_asof_snapshot(cutoff: object, ledger: Iterable[LedgerEvent]) -> AsOfSnapshot:
    """Build F0 for a decision at ``cutoff`` from ``ledger``.

    Raises ``InvalidCutoffError`` for an unusable cutoff and
    ``LedgerContractError`` if the ledger holds anything but ``LedgerEvent``.
    """
    cutoff_utc = normalize_cutoff(cutoff)
    exclusions: list[ExcludedEvent] = []
    candidates: list[AdmittedEvent] = []

    for event in tuple(ledger):
        if not isinstance(event, LedgerEvent):
            raise LedgerContractError(f"ledger item is not a LedgerEvent: {type(event).__name__}")

        result = check_availability(event, cutoff_utc)
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
        content_hash = sha256_hex(
            event.canonical_dict(result.block_time_utc, result.received_at_utc)
        )
        provenance = build_provenance(
            event, result.block_time_utc, result.received_at_utc, content_hash
        )
        candidates.append(AdmittedEvent(event, provenance))

    admitted: list[AdmittedEvent] = []
    seen_hashes: set[str] = set()
    for item in sorted(candidates, key=_admission_key):
        if item.content_hash in seen_hashes:
            exclusions.append(
                _exclude(
                    item.event,
                    ExclusionReason.DUPLICATE_RECORD,
                    f"identical content already admitted: {item.content_hash}",
                )
            )
            continue
        seen_hashes.add(item.content_hash)
        admitted.append(item)

    versions = Counter(item.event.event_id for item in admitted)
    conflicting = tuple(sorted(event_id for event_id, n in versions.items() if n > 1))

    exclusions.sort(key=lambda item: (item.event.event_id, item.reason.value, item.record_hash))
    snapshot_hash = sha256_hex(
        {
            "contract": SNAPSHOT_CONTRACT,
            "cutoff": cutoff_utc.isoformat(timespec="microseconds"),
            "events": [item.content_hash for item in admitted],
        }
    )
    return AsOfSnapshot(
        cutoff=cutoff_utc,
        events=tuple(admitted),
        exclusions=tuple(exclusions),
        conflicting_event_ids=conflicting,
        snapshot_hash=snapshot_hash,
    )
