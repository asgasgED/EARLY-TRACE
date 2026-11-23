"""Shared exception for structural ledger contract violations."""

from __future__ import annotations


class LedgerContractError(ValueError):
    """A ledger record violates the structural contract."""
