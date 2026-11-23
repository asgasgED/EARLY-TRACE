"""MISSINGNESS: observed zero, unknown, empty response and not-indexed stay distinct."""

from __future__ import annotations

from pathlib import Path

import pytest

from early_trace.evidence.events import LedgerContractError, ObservedValue, ValueState
from early_trace.evidence.ledger import load_ledger_json
from early_trace.evidence.synthetic import make_synthetic_event
from early_trace.infrastructure.hashing import sha256_hex
from early_trace.temporal.snapshot import make_asof_snapshot

CUTOFF = "2026-03-01T10:05:00+00:00"
BASIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic" / "ledger_basic.json"
STATES = {
    "zero_observed": ObservedValue.observed(0),
    "null_field": ObservedValue.null(),
    "unknown_value": ObservedValue.unknown(),
    "empty_reply": ObservedValue.empty_response(),
    "unindexed": ObservedValue.not_indexed(),
}


def test_five_states_are_pairwise_distinct():
    values = list(STATES.values())
    assert len({v.state for v in values}) == 5
    assert len({sha256_hex(v.to_dict()) for v in values}) == 5
    assert ObservedValue.observed(0).value == 0
    for state in (
        ValueState.NULL,
        ValueState.UNKNOWN,
        ValueState.EMPTY_RESPONSE,
        ValueState.NOT_INDEXED,
    ):
        assert ObservedValue(state).value is None


def test_zero_false_and_float_zero_do_not_collapse_in_content_hashes():
    hashes = {sha256_hex(ObservedValue.observed(v).to_dict()) for v in (0, False, 0.0, "0")}
    assert len(hashes) == 4


def test_states_survive_the_snapshot_unchanged():
    event = make_synthetic_event(
        "e1",
        block_time="2026-03-01T10:00:00+00:00",
        received_at="2026-03-01T10:01:00+00:00",
        fields=STATES,
    )
    admitted = make_asof_snapshot(CUTOFF, [event]).events[0].event
    assert dict(admitted.fields) == STATES


def test_events_differing_only_in_absence_state_have_different_content_hashes():
    def build(value):
        return make_synthetic_event(
            "e1",
            block_time="2026-03-01T10:00:00+00:00",
            received_at="2026-03-01T10:01:00+00:00",
            fields={"transfer_count": value},
        )

    hashes = {make_asof_snapshot(CUTOFF, [build(v)]).snapshot_hash for v in STATES.values()}
    assert len(hashes) == 5


def test_fixture_with_all_states_is_preserved():
    ledger = load_ledger_json(BASIC, require_synthetic=True)
    snapshot = make_asof_snapshot(CUTOFF, ledger)
    states = next(i.event for i in snapshot.events if i.event.event_id == "evt-005-states")
    assert {name: v.state for name, v in states.fields} == {
        "empty_reply": ValueState.EMPTY_RESPONSE,
        "null_field": ValueState.NULL,
        "unindexed": ValueState.NOT_INDEXED,
        "unknown_value": ValueState.UNKNOWN,
        "zero_observed": ValueState.OBSERVED,
    }


@pytest.mark.parametrize(
    "build",
    [
        lambda: ObservedValue(ValueState.OBSERVED, None),
        lambda: ObservedValue(ValueState.UNKNOWN, 0),
        lambda: ObservedValue(ValueState.EMPTY_RESPONSE, ""),
        lambda: ObservedValue(ValueState.NOT_INDEXED, -1),
        lambda: ObservedValue(ValueState.NULL, 0),
        lambda: ObservedValue.observed(float("nan")),
        lambda: ObservedValue.observed(float("inf")),
        lambda: ObservedValue("observed", 1),  # type: ignore[arg-type]
    ],
)
def test_contradictory_states_are_rejected(build):
    with pytest.raises(LedgerContractError):
        build()
