"""Ledger event contract.

Structural violations raise ``LedgerContractError`` at construction. Temporal
problems (missing, naive or malformed timestamps) are deliberately *not*
rejected here: they are classified by ``early_trace.temporal.availability`` so
that they surface as explicit exclusions instead of crashes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, TypeAlias, cast

TimestampInput: TypeAlias = datetime | str | None
Scalar: TypeAlias = bool | int | float | str

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_HASH_RE = re.compile(r"^0x[0-9a-f]{64}$")


class LedgerContractError(ValueError):
    """A ledger record violates the structural contract."""


class SourceType(str, Enum):
    """Kind of source an event comes from; determines the required provenance."""

    ONCHAIN_BLOCK = "onchain_block"
    ONCHAIN_TRANSACTION = "onchain_transaction"
    ONCHAIN_LOG = "onchain_log"
    EXTERNAL_RECORD = "external_record"


class ValueState(str, Enum):
    """Epistemic state of a single observed field. Never collapsed into zero."""

    OBSERVED = "observed"
    UNKNOWN = "unknown"
    EMPTY_RESPONSE = "empty_response"
    NOT_INDEXED = "not_indexed"


@dataclass(frozen=True, slots=True)
class ObservedValue:
    """A field value together with its state.

    ``OBSERVED`` requires a finite scalar (``0`` is a real observation). Every
    other state must carry no value.
    """

    state: ValueState
    value: Scalar | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, ValueState):
            raise LedgerContractError(f"invalid value state: {self.state!r}")
        if self.state is ValueState.OBSERVED:
            if not isinstance(self.value, (bool, int, float, str)):
                raise LedgerContractError("observed value must be bool, int, float or str")
            if isinstance(self.value, float) and not math.isfinite(self.value):
                raise LedgerContractError("observed float must be finite")
        elif self.value is not None:
            raise LedgerContractError(f"state '{self.state.value}' cannot carry a value")

    @classmethod
    def observed(cls, value: Scalar) -> ObservedValue:
        return cls(ValueState.OBSERVED, value)

    @classmethod
    def unknown(cls) -> ObservedValue:
        return cls(ValueState.UNKNOWN)

    @classmethod
    def empty_response(cls) -> ObservedValue:
        return cls(ValueState.EMPTY_RESPONSE)

    @classmethod
    def not_indexed(cls) -> ObservedValue:
        return cls(ValueState.NOT_INDEXED)

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "value": self.value}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ObservedValue:
        extra = set(raw) - {"state", "value"}
        if extra or "state" not in raw:
            raise LedgerContractError(f"invalid observed-value record: {dict(raw)!r}")
        try:
            state = ValueState(raw["state"])
        except ValueError as exc:
            raise LedgerContractError(f"unknown value state: {raw['state']!r}") from exc
        return cls(state, raw.get("value"))


FieldsInput: TypeAlias = Mapping[str, ObservedValue] | Iterable[tuple[str, ObservedValue]]


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LedgerContractError(f"{name} must be a non-empty string")


def _check_optional_text(name: str, value: object) -> None:
    if value is not None:
        _require_text(name, value)


def _check_optional_nonneg_int(name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerContractError(f"{name} must be a non-negative integer or None")


def _check_optional_hash(name: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise LedgerContractError(f"{name} must be a lowercase 0x-prefixed 32-byte hex string")


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat(timespec="microseconds")


def _raw_timestamp(value: object) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return {"str": value}
    if isinstance(value, datetime):
        return {"datetime": value.isoformat(timespec="microseconds")}
    return {"unsupported": type(value).__name__, "repr": repr(value)}


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    """One immutable ledger record.

    ``block_time`` is when the event occurred on-chain; ``received_at`` is when
    the system or source first knew about it. Both are mandatory arguments
    (``None`` meaning "absent") so no temporal default can be applied silently.
    ``fields`` is normalized to a tuple of ``(name, ObservedValue)`` sorted by name.
    """

    event_id: str
    entity_id: str
    source_type: SourceType
    source: str
    block_time: TimestampInput
    received_at: TimestampInput
    block_number: int | None = None
    block_hash: str | None = None
    tx_hash: str | None = None
    log_index: int | None = None
    source_version: str | None = None
    fields: FieldsInput = ()

    def __post_init__(self) -> None:
        _require_text("event_id", self.event_id)
        _require_text("entity_id", self.entity_id)
        _require_text("source", self.source)
        if not isinstance(self.source_type, SourceType):
            raise LedgerContractError(f"invalid source_type: {self.source_type!r}")
        _check_optional_text("source_version", self.source_version)
        _check_optional_nonneg_int("block_number", self.block_number)
        _check_optional_nonneg_int("log_index", self.log_index)
        _check_optional_hash("block_hash", self.block_hash)
        _check_optional_hash("tx_hash", self.tx_hash)
        object.__setattr__(self, "fields", self._normalize_fields(self.fields))

    @staticmethod
    def _normalize_fields(raw: FieldsInput) -> tuple[tuple[str, ObservedValue], ...]:
        items = list(raw.items()) if isinstance(raw, Mapping) else list(raw)
        seen: set[str] = set()
        for pair in items:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise LedgerContractError(f"invalid field entry: {pair!r}")
            name, value = pair
            if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
                raise LedgerContractError(f"invalid field name: {name!r}")
            if not isinstance(value, ObservedValue):
                raise LedgerContractError(f"field '{name}' must be an ObservedValue")
            if name in seen:
                raise LedgerContractError(f"duplicate field name: {name!r}")
            seen.add(name)
        return tuple(sorted(items, key=lambda pair: pair[0]))

    def _items(self) -> tuple[tuple[str, ObservedValue], ...]:
        return cast("tuple[tuple[str, ObservedValue], ...]", self.fields)

    def _base_dict(self) -> dict[str, Any]:
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
            "fields": {name: value.to_dict() for name, value in self._items()},
        }

    def canonical_dict(
        self, block_time_utc: datetime | None, received_at_utc: datetime | None
    ) -> dict[str, Any]:
        """Content identity of an *admitted* event, with timestamps normalized to UTC."""
        return {
            **self._base_dict(),
            "block_time": _iso(block_time_utc),
            "received_at": _iso(received_at_utc),
        }

    def raw_dict(self) -> dict[str, Any]:
        """Record identity as supplied, used to order and trace excluded events."""
        return {
            **self._base_dict(),
            "block_time": _raw_timestamp(self.block_time),
            "received_at": _raw_timestamp(self.received_at),
        }
