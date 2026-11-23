"""Observation context: how, when and through whom an evidence object was obtained.

An observation is *not* an evidence object. The same object (receipt, log, ...)
can be observed many times through different providers, methods or decoders;
this context keeps each of those paths auditable without making any of them
part of the object's identity (ET24 §6.2-6.3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from early_trace.evidence.errors import LedgerContractError

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FinalityStatus(str, Enum):
    """Finality of the block holding the object, as stated by the observation."""

    UNKNOWN = "unknown"
    UNFINALIZED = "unfinalized"
    FINALIZED = "finalized"
    REORGED = "reorged"


_TEXT_FIELDS = (
    "method",
    "query_id",
    "parser_version",
    "decoder_version",
    "provider_state",
    "interpretation_version",
    "infrastructure_group",
)


@dataclass(frozen=True, slots=True)
class ObservationContext:
    """Audit trail of one observation; every field is optional and ``None`` means not stated.

    ``infrastructure_group`` is evidence of audited independence (an identifier of the
    infrastructure/data-source group). It must stay ``None`` unless an audit established
    it: independence is never inferred from provider names.
    """

    method: str | None = None
    query_id: str | None = None
    response_hash: str | None = None
    parser_version: str | None = None
    decoder_version: str | None = None
    provider_state: str | None = None
    interpretation_version: str | None = None
    infrastructure_group: str | None = None
    finality: FinalityStatus = FinalityStatus.UNKNOWN

    def __post_init__(self) -> None:
        for name in _TEXT_FIELDS:
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise LedgerContractError(f"context.{name} must be a non-empty string or None")
        if self.response_hash is not None and not (
            isinstance(self.response_hash, str) and _SHA256_RE.fullmatch(self.response_hash)
        ):
            raise LedgerContractError("context.response_hash must be 64 lowercase hex characters")
        if not isinstance(self.finality, FinalityStatus):
            raise LedgerContractError(f"invalid finality: {self.finality!r}")

    @property
    def interpretation_key(self) -> tuple[str | None, str | None, str | None]:
        """Versions that decide how raw content was interpreted."""
        return (self.decoder_version, self.parser_version, self.interpretation_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "query_id": self.query_id,
            "response_hash": self.response_hash,
            "parser_version": self.parser_version,
            "decoder_version": self.decoder_version,
            "provider_state": self.provider_state,
            "interpretation_version": self.interpretation_version,
            "infrastructure_group": self.infrastructure_group,
            "finality": self.finality.value,
        }
