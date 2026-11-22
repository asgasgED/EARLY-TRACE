"""Synthetic event factory for tests, fixtures and demos. Never real data."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from early_trace.evidence.events import FieldsInput, LedgerEvent, SourceType, TimestampInput

SYNTHETIC_SOURCE = "synthetic-provider-a"
SYNTHETIC_ENTITY = "synthetic-entity-1"


def synthetic_hash(label: str) -> str:
    """Deterministic 32-byte hex identifier derived from a label."""
    return "0x" + hashlib.sha256(f"early-trace-synthetic:{label}".encode()).hexdigest()


def make_synthetic_event(
    event_id: str,
    *,
    block_time: TimestampInput,
    received_at: TimestampInput,
    source_type: SourceType = SourceType.ONCHAIN_LOG,
    entity_id: str = SYNTHETIC_ENTITY,
    source: str = SYNTHETIC_SOURCE,
    source_version: str | None = "v1",
    block_number: int = 100,
    log_index: int = 0,
    fields: FieldsInput = (),
    omit: Iterable[str] = (),
) -> LedgerEvent:
    """Create an event with full provenance for ``source_type``; ``omit`` drops fields.

    Timestamps are mandatory keyword arguments: this factory never invents time.
    """
    values: dict[str, object] = {
        "block_number": block_number,
        "block_hash": synthetic_hash(f"block:{block_number}"),
        "tx_hash": synthetic_hash(f"tx:{event_id}"),
        "log_index": log_index,
    }
    if source_type is SourceType.EXTERNAL_RECORD:
        values.update(block_number=None, block_hash=None, tx_hash=None, log_index=None)
    elif source_type is SourceType.ONCHAIN_BLOCK:
        values.update(tx_hash=None, log_index=None)
    elif source_type is SourceType.ONCHAIN_TRANSACTION:
        values.update(log_index=None)
    for name in omit:
        if name not in values:
            raise ValueError(f"cannot omit unknown field: {name!r}")
        values[name] = None
    return LedgerEvent(
        event_id=event_id,
        entity_id=entity_id,
        source_type=source_type,
        source=source,
        block_time=block_time,
        received_at=received_at,
        source_version=source_version,
        fields=fields,
        **values,  # type: ignore[arg-type]
    )
