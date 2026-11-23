"""Aggregation contract: how one feature value is built from many observations.

The rule is part of a feature's semantic definition (``FeatureSpec.aggregation``); the
future Feature Builder never chooses an operation implicitly. Everything not configurable
per feature is fixed by ``AGGREGATION_CONTRACT`` and therefore versioned and hashed with
the schema:

* Aggregation runs over *canonical objects* of the feature's ``source_type`` linked to one
  entity, never over raw observations. The same receipt received through two providers or
  two RPC methods is one object, so it cannot be counted twice. Independent confirmations
  stay in the provenance; counting independent sources would be a different feature.
* Duplicates are detected by canonical object identity (``evidence.identity``).
* Objects whose latest finality is ``reorged`` (or superseded by a finalized placement)
  are excluded and reported; objects with conflicting payload or unresolved placement make
  the result ``CONFLICT``; nothing is resolved by picking a value.
* A field in an absence state (null, unknown, empty response, not indexed) is never turned
  into a number: the absence state propagates to the result.
* No observation at all is ``NOT_RECEIVED``, never zero. A zero exists only as an observed
  zero, or from a declared fill of ``empty_response`` / ``null`` in the missingness rule.
* There is no implicit window: without ``window`` every object admitted in F0 counts.
* Ties on the ordering clock between incompatible values are a conflict; the canonical key
  only orders provenance, it never decides which value wins.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

AGGREGATION_CONTRACT = "aggregation/1"


class AggregationSpecError(ValueError):
    """An aggregation declaration is incomplete or inconsistent."""


class AggregationOp(str, Enum):
    SINGLE = "single"
    LATEST = "latest"
    EARLIEST = "earliest"
    COUNT_DISTINCT = "count_distinct"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"


class OrderingClock(str, Enum):
    """Clock used to order or window observations. It is never assumed: always declared."""

    BLOCK_TIME = "block_time"
    RECEIVED_AT = "received_at"


class ResultState(str, Enum):
    """State of an aggregated value; absence states mirror ``ValueState`` plus ``NOT_RECEIVED``."""

    OBSERVED = "observed"
    NULL = "null"
    UNKNOWN = "unknown"
    EMPTY_RESPONSE = "empty_response"
    NOT_INDEXED = "not_indexed"
    NOT_RECEIVED = "not_received"
    CONFLICT = "conflict"
    AMBIGUOUS = "ambiguous"
    INVALID_INPUT = "invalid_input"


# Operations whose result changes if the same object is counted twice.
DUPLICATE_SENSITIVE = frozenset(
    {
        AggregationOp.SINGLE,
        AggregationOp.COUNT_DISTINCT,
        AggregationOp.SUM,
        AggregationOp.MEAN,
    }
)
NEEDS_ORDER_CLOCK = frozenset({AggregationOp.LATEST, AggregationOp.EARLIEST})


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """Objects whose ``clock`` instant lies in ``(cutoff - length, cutoff]``."""

    clock: OrderingClock
    length_seconds: int

    def __post_init__(self) -> None:
        if not isinstance(self.clock, OrderingClock):
            raise AggregationSpecError(f"invalid window clock: {self.clock!r}")
        if (
            isinstance(self.length_seconds, bool)
            or not isinstance(self.length_seconds, int)
            or self.length_seconds < 1
        ):
            raise AggregationSpecError("window length_seconds must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {"clock": self.clock.value, "length_seconds": self.length_seconds}


@dataclass(frozen=True, slots=True)
class AggregationSpec:
    """Declared aggregation of one feature; ``rule_version`` versions the rule itself."""

    op: AggregationOp
    rule_version: int
    order_clock: OrderingClock | None = None
    window: WindowSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.op, AggregationOp):
            raise AggregationSpecError(f"invalid aggregation op: {self.op!r}")
        if (
            isinstance(self.rule_version, bool)
            or not isinstance(self.rule_version, int)
            or self.rule_version < 1
        ):
            raise AggregationSpecError("rule_version must be an integer >= 1")
        if self.op in NEEDS_ORDER_CLOCK and self.order_clock is None:
            raise AggregationSpecError(f"'{self.op.value}' must declare an order_clock")
        if self.op not in NEEDS_ORDER_CLOCK and self.order_clock is not None:
            raise AggregationSpecError(f"'{self.op.value}' does not use an order_clock")
        if self.order_clock is not None and not isinstance(self.order_clock, OrderingClock):
            raise AggregationSpecError(f"invalid order_clock: {self.order_clock!r}")
        if self.window is not None and not isinstance(self.window, WindowSpec):
            raise AggregationSpecError("window must be a WindowSpec")

    @property
    def duplicate_sensitive(self) -> bool:
        return self.op in DUPLICATE_SENSITIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": AGGREGATION_CONTRACT,
            "op": self.op.value,
            "rule_version": self.rule_version,
            "order_clock": None if self.order_clock is None else self.order_clock.value,
            "window": None if self.window is None else self.window.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AggregationSpec:
        required = {"contract", "op", "rule_version", "order_clock", "window"}
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise AggregationSpecError(
                f"aggregation must declare exactly {sorted(required)}; "
                "an implicit aggregation is not allowed"
            )
        if raw["contract"] != AGGREGATION_CONTRACT:
            raise AggregationSpecError(f"unsupported aggregation contract: {raw['contract']!r}")
        window = raw["window"]
        try:
            return cls(
                op=AggregationOp(raw["op"]),
                rule_version=raw["rule_version"],
                order_clock=None
                if raw["order_clock"] is None
                else OrderingClock(raw["order_clock"]),
                window=None
                if window is None
                else WindowSpec(OrderingClock(window["clock"]), window["length_seconds"]),
            )
        except (ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, AggregationSpecError):
                raise
            raise AggregationSpecError(f"invalid aggregation declaration: {exc}") from exc
