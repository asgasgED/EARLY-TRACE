"""DETERMINISM: same input and config give the same snapshot, order and hash."""

from __future__ import annotations

import copy
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from early_trace.evidence.ledger import load_ledger_json
from early_trace.infrastructure.hashing import sha256_hex
from early_trace.temporal.snapshot import make_asof_snapshot

CUTOFF = "2026-03-01T10:05:00+00:00"
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FIXTURES = ROOT / "fixtures"
BASIC = str(FIXTURES / "synthetic" / "ledger_basic.json")
ADVERSARIAL = str(FIXTURES / "adversarial" / "ledger_adversarial_timestamps.json")
BASIC_GOLDEN_HASH = "e159ea866c6177bc39cf0287fe3f96027689e9abf5247ad9b4d22492e321f0ad"
ADVERSARIAL_GOLDEN_HASH = "c8f7f46590576defb9dd5d8f5a560b0ae657f634059c14e565afacfd0d127bda"


@pytest.mark.parametrize("path", [BASIC, ADVERSARIAL])
def test_repeated_runs_are_identical(path):
    ledger = load_ledger_json(path, require_synthetic=True)
    runs = [make_asof_snapshot(CUTOFF, ledger) for _ in range(5)]
    assert all(run == runs[0] for run in runs)
    assert len({run.snapshot_hash for run in runs}) == 1
    assert len({sha256_hex(run.to_dict()) for run in runs}) == 1


@pytest.mark.parametrize("path", [BASIC, ADVERSARIAL])
def test_ledger_order_does_not_matter(path):
    ledger = list(load_ledger_json(path, require_synthetic=True))
    reference = make_asof_snapshot(CUTOFF, ledger)
    rng = random.Random(24)
    for _ in range(10):
        shuffled = ledger[:]
        rng.shuffle(shuffled)
        snapshot = make_asof_snapshot(CUTOFF, shuffled)
        assert snapshot.event_ids == reference.event_ids
        assert snapshot.snapshot_hash == reference.snapshot_hash
        assert snapshot.to_dict() == reference.to_dict()


@pytest.mark.parametrize("path", [BASIC, ADVERSARIAL])
def test_input_ledger_is_not_modified(path):
    ledger = list(load_ledger_json(path, require_synthetic=True))
    before = copy.deepcopy(ledger)
    fingerprint = sha256_hex([event.raw_dict() for event in ledger])
    make_asof_snapshot(CUTOFF, ledger)
    assert ledger == before
    assert sha256_hex([event.raw_dict() for event in ledger]) == fingerprint


def test_equivalent_utc_offsets_produce_the_same_hash():
    ledger = load_ledger_json(BASIC, require_synthetic=True)
    same_instant = "2026-03-01T15:05:00+05:00"
    assert (
        make_asof_snapshot(CUTOFF, ledger).snapshot_hash
        == make_asof_snapshot(same_instant, ledger).snapshot_hash
    )


@pytest.mark.parametrize(
    ("path", "golden"), [(BASIC, BASIC_GOLDEN_HASH), (ADVERSARIAL, ADVERSARIAL_GOLDEN_HASH)]
)
def test_hash_matches_pinned_golden_value(path, golden):
    ledger = load_ledger_json(path, require_synthetic=True)
    assert make_asof_snapshot(CUTOFF, ledger).snapshot_hash == golden


def test_hash_does_not_depend_on_python_hash_randomization():
    code = (
        "from early_trace.evidence.ledger import load_ledger_json;"
        "from early_trace.temporal.snapshot import make_asof_snapshot;"
        f"l=load_ledger_json({BASIC!r}, require_synthetic=True);"
        f"print(make_asof_snapshot({CUTOFF!r}, l).snapshot_hash)"
    )
    outputs = set()
    for seed in ("1", "2", "random"):
        pythonpath = os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH")]))
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": pythonpath}
        done = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        outputs.add(done.stdout.strip())
    assert outputs == {BASIC_GOLDEN_HASH}
