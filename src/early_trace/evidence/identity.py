"""Canonical identity of evidence objects (ET24 §6.2-6.3).

The identity answers *which object* is observed, independently of the provider,
RPC method, decoder version or reception time through which it was obtained.
Those belong to the observation (see ``observation.py``), never to the key.

Identity is resolved only from fields the event actually carries. When a
required field is absent the identity is *unresolved*, with an explicit reason:
no key is ever invented, and unresolved objects are never merged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias

from early_trace.evidence.events import LedgerEvent, SourceType
from early_trace.infrastructure.hashing import sha256_hex

IDENTITY_CONTRACT = "evidence-object/1"

Component: TypeAlias = int | str | tuple[int, ...]

# Identity fields per evidence kind. A transaction is identified by its hash alone: the
# block that includes it is an attribute (placement) that a reorganization can change.
# External records have no canonical key defined by ET24 and stay unresolved.
IDENTITY_FIELDS: Mapping[SourceType, tuple[str, ...]] = MappingProxyType(
    {
        SourceType.ONCHAIN_BLOCK: ("chain_id", "block_hash"),
        SourceType.ONCHAIN_TRANSACTION: ("chain_id", "tx_hash"),
        SourceType.ONCHAIN_RECEIPT: ("chain_id", "block_hash", "tx_hash"),
        SourceType.ONCHAIN_LOG: ("chain_id", "block_hash", "tx_hash", "log_index"),
        SourceType.ONCHAIN_TRACE: ("chain_id", "block_hash", "tx_hash", "trace_address"),
    }
)


class IdentityIssue(str, Enum):
    MISSING_FIELDS = "missing_fields"
    UNDEFINED_FOR_SOURCE_TYPE = "undefined_for_source_type"


@dataclass(frozen=True, slots=True)
class ObjectKey:
    """Resolved canonical identity of one evidence object."""

    kind: SourceType
    components: tuple[tuple[str, Component], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": IDENTITY_CONTRACT,
            "kind": self.kind.value,
            "components": {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in self.components
            },
        }

    @property
    def key_id(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True, slots=True)
class UnresolvedIdentity:
    """The object cannot be identified reliably; nothing is guessed."""

    kind: SourceType
    issue: IdentityIssue
    missing_fields: tuple[str, ...] = ()

    @property
    def detail(self) -> str:
        if self.issue is IdentityIssue.MISSING_FIELDS:
            return (
                f"missing identity fields for {self.kind.value}: {', '.join(self.missing_fields)}"
            )
        return f"no canonical identity is defined for {self.kind.value}"


Identity: TypeAlias = ObjectKey | UnresolvedIdentity


def resolve_identity(event: LedgerEvent) -> Identity:
    """Return the canonical key of the object ``event`` observes, or why it is unresolved."""
    required = IDENTITY_FIELDS.get(event.source_type)
    if required is None:
        return UnresolvedIdentity(event.source_type, IdentityIssue.UNDEFINED_FOR_SOURCE_TYPE)
    missing = tuple(name for name in required if getattr(event, name) is None)
    if missing:
        return UnresolvedIdentity(event.source_type, IdentityIssue.MISSING_FIELDS, missing)
    return ObjectKey(event.source_type, tuple((name, getattr(event, name)) for name in required))
