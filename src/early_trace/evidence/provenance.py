"""Provenance requirements per source type and verifiable provenance records.

Provenance describes the *observation* (where the evidence came from and when). Identity
fields that only matter for canonical deduplication (``chain_id``, ``log_index``,
``trace_address``) are not provenance requirements: their absence leaves the object's
identity unresolved (see ``identity.py``) instead of excluding the observation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

from early_trace.evidence.events import LedgerEvent, SourceType
from early_trace.evidence.identity import Identity, ObjectKey, UnresolvedIdentity
from early_trace.evidence.observation import ObservationContext

_BLOCK_FIELDS = ("block_number", "block_hash", "block_time")

# Fields required beyond the universal ones (event_id, entity_id, source,
# received_at), which the ledger contract and availability check already enforce.
REQUIRED_PROVENANCE: Mapping[SourceType, tuple[str, ...]] = MappingProxyType(
    {
        SourceType.ONCHAIN_BLOCK: _BLOCK_FIELDS,
        SourceType.ONCHAIN_TRANSACTION: (*_BLOCK_FIELDS, "tx_hash"),
        SourceType.ONCHAIN_RECEIPT: (*_BLOCK_FIELDS, "tx_hash"),
        SourceType.ONCHAIN_LOG: (*_BLOCK_FIELDS, "tx_hash"),
        SourceType.ONCHAIN_TRACE: (*_BLOCK_FIELDS, "tx_hash"),
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
    chain_id: int | None
    trace_address: tuple[int, ...] | None
    context: ObservationContext
    object_key_id: str | None
    identity_issue: str | None
    payload_hash: str
    observation_hash: str

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
            "chain_id": self.chain_id,
            "trace_address": None if self.trace_address is None else list(self.trace_address),
            "context": self.context.to_dict(),
            "object_key_id": self.object_key_id,
            "identity_issue": self.identity_issue,
            "payload_hash": self.payload_hash,
            "observation_hash": self.observation_hash,
        }


def build_provenance(
    event: LedgerEvent,
    block_time_utc: datetime | None,
    received_at_utc: datetime,
    *,
    identity: Identity,
    payload_hash: str,
    observation_hash: str,
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
        chain_id=event.chain_id,
        trace_address=event.trace_address,
        context=event.context,
        object_key_id=identity.key_id if isinstance(identity, ObjectKey) else None,
        identity_issue=identity.detail if isinstance(identity, UnresolvedIdentity) else None,
        payload_hash=payload_hash,
        observation_hash=observation_hash,
    )
