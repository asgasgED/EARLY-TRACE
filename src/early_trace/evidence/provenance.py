"""Provenance requirements per source type and verifiable provenance records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

from early_trace.evidence.events import LedgerEvent, SourceType

_BLOCK_FIELDS = ("block_number", "block_hash", "block_time")

# Fields required beyond the universal ones (event_id, entity_id, source,
# received_at), which the ledger contract and availability check already enforce.
REQUIRED_PROVENANCE: Mapping[SourceType, tuple[str, ...]] = MappingProxyType(
    {
        SourceType.ONCHAIN_BLOCK: _BLOCK_FIELDS,
        SourceType.ONCHAIN_TRANSACTION: (*_BLOCK_FIELDS, "tx_hash"),
        SourceType.ONCHAIN_LOG: (*_BLOCK_FIELDS, "tx_hash", "log_index"),
        SourceType.EXTERNAL_RECORD: (),
    }
)


def missing_provenance(event: LedgerEvent) -> tuple[str, ...]:
    """Names of provenance fields required by the event's source type but absent."""
    return tuple(
        name for name in REQUIRED_PROVENANCE[event.source_type] if getattr(event, name) is None
    )


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Verifiable link between an admitted event and its originating evidence."""

    event_id: str
    entity_id: str
    source_type: SourceType
    source: str
    source_version: str | None
    block_number: int | None
    block_hash: str | None
    tx_hash: str | None
    log_index: int | None
    block_time_utc: datetime | None
    received_at_utc: datetime
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "entity_id": self.entity_id,
            "source_type": self.source_type.value,
            "source": self.source,
            "source_version": self.source_version,
            "block_number": self.block_number,
            "block_hash": self.block_hash,
            "tx_hash": self.tx_hash,
            "log_index": self.log_index,
            "block_time": None
            if self.block_time_utc is None
            else self.block_time_utc.isoformat(timespec="microseconds"),
            "received_at": self.received_at_utc.isoformat(timespec="microseconds"),
            "content_hash": self.content_hash,
        }


def build_provenance(
    event: LedgerEvent,
    block_time_utc: datetime | None,
    received_at_utc: datetime,
    content_hash: str,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        event_id=event.event_id,
        entity_id=event.entity_id,
        source_type=event.source_type,
        source=event.source,
        source_version=event.source_version,
        block_number=event.block_number,
        block_hash=event.block_hash,
        tx_hash=event.tx_hash,
        log_index=event.log_index,
        block_time_utc=block_time_utc,
        received_at_utc=received_at_utc,
        content_hash=content_hash,
    )
