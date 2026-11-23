"""Lifecycle of a query response (ET24 §5.4, §6.1): what the system knew, and when.

A response is described by clocks of the system itself (not of the chain):

* ``t_query_sent``: the query left the system.
* ``t_query_received``: the response arrived; ``None`` while it has not.
* ``t_feature_update_end``: validation, canonicalization and update finished; ``None``
  while processing is pending. It is the instant the validation *outcome* became known.
* ``deadline``: latest reception time usable by the decision, when one exists.

``classify_response`` answers "what was the status at instant ``as_of``?", so the same record
yields different statuses over time and nothing is read from the future. A response only
reaches the post-query state when it was received on time *and* validated; it never alters
``F0``, which is built from the ledger at its own cutoff (late observations carry a
``received_at`` after that cutoff and are therefore excluded there).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from early_trace.evidence.events import TimestampInput
from early_trace.temporal.availability import TimestampError, to_utc


class ResponseTimelineError(ValueError):
    """The timestamps or states of a query response are mutually impossible."""


class ValidationOutcome(str, Enum):
    PENDING = "pending"
    INVALID = "invalid"
    VALID = "valid"


class ResponseStatus(str, Enum):
    NOT_RECEIVED = "not_received"
    RECEIVED_PENDING_VALIDATION = "received_pending_validation"
    RECEIVED_INVALID = "received_invalid"
    RECEIVED_AFTER_DEADLINE = "received_after_deadline"
    VALIDATED = "validated"
    VALIDATED_NO_NEW_OBJECTS = "validated_no_new_objects"


def _utc(value: TimestampInput, name: str) -> datetime | None:
    if value is None:
        return None
    try:
        return to_utc(value)
    except TimestampError as exc:
        raise ResponseTimelineError(f"{name}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class QueryResponse:
    """Timeline of one query; timestamps are normalized to UTC and must be aware."""

    query_id: str
    t_query_sent: datetime
    t_query_received: datetime | None
    t_feature_update_end: datetime | None
    deadline: datetime | None
    validation: ValidationOutcome
    new_object_count: int | None

    @classmethod
    def create(
        cls,
        query_id: str,
        *,
        t_query_sent: TimestampInput,
        t_query_received: TimestampInput,
        t_feature_update_end: TimestampInput,
        deadline: TimestampInput,
        validation: ValidationOutcome,
        new_object_count: int | None,
    ) -> QueryResponse:
        sent = _utc(t_query_sent, "t_query_sent")
        if sent is None:
            raise ResponseTimelineError("t_query_sent is required")
        return cls(
            query_id,
            sent,
            _utc(t_query_received, "t_query_received"),
            _utc(t_feature_update_end, "t_feature_update_end"),
            _utc(deadline, "deadline"),
            validation,
            new_object_count,
        )

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ResponseTimelineError("query_id must be non-empty")
        received, end = self.t_query_received, self.t_feature_update_end
        if received is not None and received < self.t_query_sent:
            raise ResponseTimelineError("t_query_received precedes t_query_sent")
        if end is not None and (received is None or end < received):
            raise ResponseTimelineError("t_feature_update_end requires a prior t_query_received")
        pending = self.validation is ValidationOutcome.PENDING
        if pending and end is not None:
            raise ResponseTimelineError("a pending validation cannot have t_feature_update_end")
        if not pending and end is None:
            raise ResponseTimelineError("a decided validation needs t_feature_update_end")
        valid = self.validation is ValidationOutcome.VALID
        if valid != (self.new_object_count is not None) or (
            self.new_object_count is not None and self.new_object_count < 0
        ):
            raise ResponseTimelineError("new_object_count is required (>= 0) only when valid")


def classify_response(response: QueryResponse, as_of: object) -> ResponseStatus:
    """Status of ``response`` as known at ``as_of`` (inclusive); later facts are ignored."""
    instant = to_utc(as_of)
    received = response.t_query_received
    if received is None or received > instant:
        return ResponseStatus.NOT_RECEIVED
    if response.deadline is not None and received > response.deadline:
        return ResponseStatus.RECEIVED_AFTER_DEADLINE
    end = response.t_feature_update_end
    if end is None or end > instant:
        return ResponseStatus.RECEIVED_PENDING_VALIDATION
    if response.validation is ValidationOutcome.INVALID:
        return ResponseStatus.RECEIVED_INVALID
    if response.new_object_count == 0:
        return ResponseStatus.VALIDATED_NO_NEW_OBJECTS
    return ResponseStatus.VALIDATED


def incorporable(response: QueryResponse, as_of: object) -> bool:
    """Whether the response may enter the post-query state at ``as_of``."""
    return classify_response(response, as_of) in {
        ResponseStatus.VALIDATED,
        ResponseStatus.VALIDATED_NO_NEW_OBJECTS,
    }
