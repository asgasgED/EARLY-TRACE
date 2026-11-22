"""Availability validator: could the system know this event at the cutoff?

Contract
--------
* ``block_time`` is when the event happened on-chain. It never decides
  availability by itself.
* ``received_at`` is when the system/source first knew the event. It is the
  only clock that decides availability: an event is available iff
  ``received_at <= cutoff`` (inclusive, per ET24 §5.4 "hasta t_snapshot").
* Accepted timestamps are timezone-aware ``datetime`` objects or ISO-8601
  strings carrying an explicit UTC offset. All are normalized to UTC.
* Naive timestamps are invalid; UTC is never assumed. Integers (epoch seconds
  vs. milliseconds is ambiguous) and any other type are invalid; loaders must
  convert them explicitly.
* A missing ``received_at`` is ``UNKNOWN`` availability and is never treated as
  available.
* ``received_at < block_time`` is inconsistent and therefore ``INVALID``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from early_trace.evidence.events import LedgerEvent


class TimestampError(ValueError):
    """A timestamp cannot be given clear UTC semantics."""


class NaiveTimestampError(TimestampError):
    """The timestamp carries no timezone information."""


class InvalidTimestampError(TimestampError):
    """The timestamp is malformed or of an unsupported type."""


class InvalidCutoffError(TimestampError):
    """The decision cutoff cannot be used as a temporal boundary."""


class AvailabilityStatus(str, Enum):
    AVAILABLE = "available"
    NOT_YET_AVAILABLE = "not_yet_available"
    UNKNOWN = "unknown"
    INVALID = "invalid"


class ExclusionReason(str, Enum):
    """Why an event is not part of F0 (availability and snapshot-level reasons)."""

    RECEIVED_AFTER_CUTOFF = "received_after_cutoff"
    RECEIVED_AT_MISSING = "received_at_missing"
    INVALID_RECEIVED_AT = "invalid_received_at"
    NAIVE_RECEIVED_AT = "naive_received_at"
    INVALID_BLOCK_TIME = "invalid_block_time"
    NAIVE_BLOCK_TIME = "naive_block_time"
    RECEIVED_BEFORE_BLOCK_TIME = "received_before_block_time"
    INCOMPLETE_PROVENANCE = "incomplete_provenance"
    DUPLICATE_RECORD = "duplicate_record"


@dataclass(frozen=True, slots=True)
class AvailabilityResult:
    status: AvailabilityStatus
    reason: ExclusionReason | None
    block_time_utc: datetime | None
    received_at_utc: datetime | None
    detail: str = ""

    @property
    def is_available(self) -> bool:
        return self.status is AvailabilityStatus.AVAILABLE


def to_utc(value: object) -> datetime:
    """Normalize a timestamp to an aware UTC ``datetime`` or raise ``TimestampError``."""
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise InvalidTimestampError(f"not an ISO-8601 timestamp: {value!r}") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise InvalidTimestampError(f"unsupported timestamp type: {type(value).__name__}")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise NaiveTimestampError(f"timestamp has no timezone: {value!r}")
    try:
        return parsed.astimezone(timezone.utc)
    except OverflowError as exc:
        raise InvalidTimestampError(f"timestamp out of range: {value!r}") from exc


def normalize_cutoff(cutoff: object) -> datetime:
    """Return the cutoff as aware UTC or raise ``InvalidCutoffError``."""
    try:
        return to_utc(cutoff)
    except TimestampError as exc:
        raise InvalidCutoffError(f"invalid cutoff: {exc}") from exc


def _parse_field(
    value: object,
    field: str,
    naive_reason: ExclusionReason,
    invalid_reason: ExclusionReason,
) -> datetime | AvailabilityResult:
    try:
        return to_utc(value)
    except NaiveTimestampError as exc:
        reason = naive_reason
        detail = f"{field}: {exc}"
    except TimestampError as exc:
        reason = invalid_reason
        detail = f"{field}: {exc}"
    return AvailabilityResult(AvailabilityStatus.INVALID, reason, None, None, detail)


def check_availability(event: LedgerEvent, cutoff: object) -> AvailabilityResult:
    """Decide whether ``event`` could be known at ``cutoff``.

    Raises ``InvalidCutoffError`` if the cutoff itself is unusable; every
    problem with the event is reported through the returned result.
    """
    cutoff_utc = normalize_cutoff(cutoff)

    if event.received_at is None:
        return AvailabilityResult(
            AvailabilityStatus.UNKNOWN,
            ExclusionReason.RECEIVED_AT_MISSING,
            None,
            None,
            "received_at is absent; availability cannot be established",
        )

    received = _parse_field(
        event.received_at,
        "received_at",
        ExclusionReason.NAIVE_RECEIVED_AT,
        ExclusionReason.INVALID_RECEIVED_AT,
    )
    if isinstance(received, AvailabilityResult):
        return received

    block_time: datetime | None = None
    if event.block_time is not None:
        parsed = _parse_field(
            event.block_time,
            "block_time",
            ExclusionReason.NAIVE_BLOCK_TIME,
            ExclusionReason.INVALID_BLOCK_TIME,
        )
        if isinstance(parsed, AvailabilityResult):
            return parsed
        block_time = parsed

    if block_time is not None and received < block_time:
        return AvailabilityResult(
            AvailabilityStatus.INVALID,
            ExclusionReason.RECEIVED_BEFORE_BLOCK_TIME,
            block_time,
            received,
            f"received_at {received.isoformat()} precedes block_time {block_time.isoformat()}",
        )

    if received > cutoff_utc:
        return AvailabilityResult(
            AvailabilityStatus.NOT_YET_AVAILABLE,
            ExclusionReason.RECEIVED_AFTER_CUTOFF,
            block_time,
            received,
            f"received_at {received.isoformat()} is after cutoff {cutoff_utc.isoformat()}",
        )

    return AvailabilityResult(AvailabilityStatus.AVAILABLE, None, block_time, received)
