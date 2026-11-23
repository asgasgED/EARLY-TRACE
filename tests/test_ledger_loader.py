"""Ledger loader: strict structure, explicit synthetic marking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from early_trace.evidence.events import LedgerContractError
from early_trace.evidence.ledger import event_from_record, load_ledger_json

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def record(**overrides):
    base = {
        "event_id": "e1",
        "entity_id": "x",
        "source_type": "external_record",
        "source": "s",
        "block_time": None,
        "received_at": "2026-03-01T10:01:00+00:00",
    }
    base.update(overrides)
    return base


def test_fixtures_declare_themselves_synthetic():
    for path in (
        FIXTURES / "synthetic" / "ledger_basic.json",
        FIXTURES / "adversarial" / "ledger_adversarial_timestamps.json",
    ):
        with open(path, encoding="utf-8") as handle:
            meta = json.load(handle)["fixture"]
        assert meta["synthetic"] is True and "SYNTHETIC" in meta["notice"]


def test_non_synthetic_file_is_refused(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"fixture": {"id": "x", "synthetic": False}, "events": []}))
    with pytest.raises(LedgerContractError):
        load_ledger_json(path, require_synthetic=True)


@pytest.mark.parametrize(
    "bad",
    [
        {"unexpected": 1},
        {"source_type": "mystery"},
        {"block_hash": "0xABC"},
        {"log_index": -1},
        {"fields": {"Bad-Name": {"state": "observed", "value": 1}}},
        {"fields": {"ok": {"state": "unknown", "value": 0}}},
    ],
)
def test_invalid_records_are_rejected(bad):
    with pytest.raises(LedgerContractError):
        event_from_record(record(**bad))


@pytest.mark.parametrize("key", ["block_time", "received_at"])
def test_timestamp_keys_must_be_explicit(key):
    raw = record()
    del raw[key]
    with pytest.raises(LedgerContractError):
        event_from_record(raw)


def test_identity_and_context_fields_load_from_records():
    event = event_from_record(
        record(
            source_type="onchain_trace",
            chain_id=1,
            trace_address=[0, 2],
            context={
                "method": "trace_transaction",
                "finality": "finalized",
                "decoder_version": "d1",
            },
            fields={"x": {"state": "null", "value": None}},
        )
    )
    assert (event.chain_id, event.trace_address) == (1, (0, 2))
    assert event.context.method == "trace_transaction"
    assert event.context.finality.value == "finalized"
    assert dict(event.fields)["x"].state.value == "null"


def test_an_empty_trace_address_is_the_root_call_and_differs_from_absent():
    root = event_from_record(record(source_type="onchain_trace", trace_address=[]))
    absent = event_from_record(record(source_type="onchain_trace"))
    assert root.trace_address == () and absent.trace_address is None


@pytest.mark.parametrize(
    "bad",
    [
        {"chain_id": 0},
        {"chain_id": "1"},
        {"chain_id": True},
        {"trace_address": [-1]},
        {"trace_address": "0.1"},
        {"trace_address": [True]},
        {"context": {"unexpected": "x"}},
        {"context": {"finality": "maybe"}},
        {"context": {"response_hash": "0xABC"}},
        {"context": {"method": ""}},
        {"context": "not-an-object"},
    ],
)
def test_invalid_identity_and_context_records_are_rejected(bad):
    with pytest.raises(LedgerContractError):
        event_from_record(record(**bad))
