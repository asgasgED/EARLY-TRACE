"""Canonical serialization and content hashing (SHA-256)."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(payload: Any) -> bytes:
    """Serialize JSON-compatible data deterministically (sorted keys, no NaN, ASCII)."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_hex(payload: Any) -> str:
    """Return the SHA-256 hex digest of the canonical serialization of ``payload``."""
    return hashlib.sha256(canonical_json(payload)).hexdigest()
