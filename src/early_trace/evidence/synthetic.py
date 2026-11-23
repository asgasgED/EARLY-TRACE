"""Synthetic event factory for tests, fixtures and demos. Never real data."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from early_trace.evidence.events import FieldsInput, LedgerEvent, SourceType, TimestampInput
from early_trace.evidence.observation import ObservationContext

SYNTHETIC_SOURCE = "synthetic-provider-a"
SYNTHETIC_ENTITY = "synthetic-entity-1"
SYNTHETIC_CHAIN_ID = 1


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
    chain_id: int | None = SYNTHETIC_CHAIN_ID,
    trace_address: tuple[int, ...] = (0,),
    block_label: str | None = None,
    tx_label: str | None = None,
    context: ObservationContext | None = None,
    fields: FieldsInput = (),
    omit: Iterable[str] = (),
) -> LedgerEvent:
    """Create an event with full provenance and identity for ``source_type``.

    ``block_label``/``tx_label`` choose which synthetic block/transaction hash is used
    (default: derived from ``block_number`` / ``event_id``), so several records can
    observe the same object or the same transaction under different blocks. ``omit``
    drops the named identity/provenance fields. Timestamps are mandatory keyword
    arguments: this factory never invents time.
    """
    values: dict[str, object] = {
        "block_number": block_number,
        "block_hash": synthetic_hash(f"block:{block_label or block_number}"),
        "tx_hash": synthetic_hash(f"tx:{tx_label or event_id}"),
        "log_index": log_index,
        "chain_id": chain_id,
        "trace_address": trace_address,
    }
    drop = {
        SourceType.EXTERNAL_RECORD: ("block_number", "block_hash", "tx_hash", "log_index"),
        SourceType.ONCHAIN_BLOCK: ("tx_hash", "log_index"),
        SourceType.ONCHAIN_TRANSACTION: ("log_index",),
        SourceType.ONCHAIN_RECEIPT: ("log_index",),
        SourceType.ONCHAIN_LOG: (),
        SourceType.ONCHAIN_TRACE: ("log_index",),
    }[source_type]
    for name in drop:
        values[name] = None
    if source_type is not SourceType.ONCHAIN_TRACE:
        values["trace_address"] = None
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
        context=context or ObservationContext(),
        fields=fields,
        **values,  # type: ignore[arg-type]
    )
