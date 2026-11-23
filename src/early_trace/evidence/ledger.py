"""JSON ledger loading.

File layout::

    {"fixture": {"id": "...", "synthetic": true, "notice": "..."},
     "events": [{...record...}, ...]}

Both timestamp keys are mandatory in every record (``null`` means absent) so a
missing key can never be mistaken for a deliberate absence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from early_trace.evidence.events import LedgerContractError, LedgerEvent, ObservedValue, SourceType
from early_trace.evidence.observation import FinalityStatus, ObservationContext

_REQUIRED_KEYS = {"event_id", "entity_id", "source_type", "source", "block_time", "received_at"}
_OPTIONAL_KEYS = {
    "block_number",
    "block_hash",
    "tx_hash",
    "log_index",
    "source_version",
    "fields",
    "chain_id",
    "trace_address",
    "context",
}
_CONTEXT_KEYS = {
    "method",
    "query_id",
    "response_hash",
    "parser_version",
    "decoder_version",
    "provider_state",
    "interpretation_version",
    "infrastructure_group",
    "finality",
}


def context_from_record(raw: Mapping[str, Any]) -> ObservationContext:
    """Build an ``ObservationContext``; unknown keys and unknown finality values are rejected."""
    if not isinstance(raw, Mapping) or set(raw) - _CONTEXT_KEYS:
        raise LedgerContractError(f"invalid context record: {raw!r}")
    values = dict(raw)
    if "finality" in values:
        try:
            values["finality"] = FinalityStatus(values["finality"])
        except ValueError as exc:
            raise LedgerContractError(f"unknown finality: {values['finality']!r}") from exc
    return ObservationContext(**values)


def event_from_record(record: Mapping[str, Any]) -> LedgerEvent:
    """Build a ``LedgerEvent`` from a plain mapping, rejecting unknown or missing keys."""
    if not isinstance(record, Mapping):
        raise LedgerContractError(f"ledger record must be an object, got {type(record).__name__}")
    missing = _REQUIRED_KEYS - set(record)
    unknown = set(record) - _REQUIRED_KEYS - _OPTIONAL_KEYS
    if missing or unknown:
        raise LedgerContractError(
            f"record {record.get('event_id')!r}: missing keys {sorted(missing)}, "
            f"unknown keys {sorted(unknown)}"
        )
    try:
        source_type = SourceType(record["source_type"])
    except ValueError as exc:
        raise LedgerContractError(f"unknown source_type: {record['source_type']!r}") from exc

    raw_fields = record.get("fields", {})
    if not isinstance(raw_fields, Mapping):
        raise LedgerContractError("fields must be an object")
    fields = {name: ObservedValue.from_dict(value) for name, value in raw_fields.items()}

    return LedgerEvent(
        event_id=record["event_id"],
        entity_id=record["entity_id"],
        source_type=source_type,
        source=record["source"],
        block_time=record["block_time"],
        received_at=record["received_at"],
        block_number=record.get("block_number"),
        block_hash=record.get("block_hash"),
        tx_hash=record.get("tx_hash"),
        log_index=record.get("log_index"),
        source_version=record.get("source_version"),
        fields=fields,
        chain_id=record.get("chain_id"),
        trace_address=record.get("trace_address"),
        context=context_from_record(record.get("context", {})),
    )


def load_ledger_json(path: Path | str, *, require_synthetic: bool) -> tuple[LedgerEvent, ...]:
    """Load a ledger file. With ``require_synthetic`` the file must declare itself synthetic."""
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping) or set(document) != {"fixture", "events"}:
        raise LedgerContractError("ledger file must contain exactly 'fixture' and 'events'")
    meta = document["fixture"]
    if require_synthetic and not (isinstance(meta, Mapping) and meta.get("synthetic") is True):
        raise LedgerContractError("ledger file is not declared synthetic")
    if not isinstance(document["events"], list):
        raise LedgerContractError("'events' must be a list")
    return tuple(event_from_record(record) for record in document["events"])
