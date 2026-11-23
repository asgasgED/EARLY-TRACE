"""End-to-end: ledger -> as-of snapshot -> canonical objects -> aggregation -> schema evolution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from early_trace.evidence.canonical import EvidenceStanding
from early_trace.evidence.events import ObservedValue as OV
from early_trace.evidence.events import SourceType as ST
from early_trace.evidence.ledger import event_from_record, load_ledger_json
from early_trace.evidence.synthetic import make_synthetic_event
from early_trace.features.aggregation import aggregate_feature
from early_trace.schemas.aggregation import (
    AggregationOp,
    AggregationSpec,
    OrderingClock,
    ResultState,
)
from early_trace.schemas.feature_schema import (
    AbsenceSpec,
    AvailabilityClock,
    FeatureDType,
    FeatureSchema,
    FeatureSpec,
    MissingnessRule,
    SchemaVersion,
    validate_evolution,
)
from early_trace.temporal.snapshot import make_asof_snapshot

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
CUTOFF = "2026-03-01T10:30:00+00:00"
ENTITY = "synthetic-entity-1"
MASK = AbsenceSpec.mask_only()
V1, V2 = SchemaVersion.parse("1.0.0"), SchemaVersion.parse("2.0.0")


def feature(name, kind, field, op, dtype=FeatureDType.INT, version=V1, order=None):
    return FeatureSpec(
        name=name,
        dtype=dtype,
        description=f"Synthetic {name}.",
        source_type=kind,
        source_field=field,
        availability_clock=AvailabilityClock.RECEIVED_AT,
        missingness=MissingnessRule(MASK, MASK, MASK, MASK, MASK),
        aggregation=AggregationSpec(op, 1, order),
        feature_version=1,
        introduced_in=version,
    )


SCHEMA_V1 = FeatureSchema(
    "integration",
    V1,
    (
        feature("receipt_gas_total", ST.ONCHAIN_RECEIPT, "gas_used", AggregationOp.SUM),
        feature("transfer_value_total", ST.ONCHAIN_LOG, "transfer_value", AggregationOp.SUM),
        feature(
            "latest_transfer",
            ST.ONCHAIN_LOG,
            "transfer_value",
            AggregationOp.LATEST,
            order=OrderingClock.RECEIVED_AT,
        ),
    ),
)


def snapshot():
    ledger = load_ledger_json(
        FIXTURES / "synthetic" / "ledger_identity.json", require_synthetic=True
    )
    return make_asof_snapshot(CUTOFF, ledger)


def test_identity_fixture_end_to_end_values():
    snap = snapshot()
    values = {f.name: aggregate_feature(f, snap, ENTITY) for f in SCHEMA_V1.features}
    # receipts: s1 once (21000) + s7 canonical (30000); the reorged placement is excluded
    assert values["receipt_gas_total"].value == 51000
    # logs: s2=5, s3=7 (confirmed twice), s4=9 (two decoders), s5=3, s6=1+2 -> counted once each
    assert values["transfer_value_total"].value == 5 + 7 + 9 + 3 + 1 + 2
    # s6-0 (value 1) and s6-1 (value 2) became available at the same instant: a tie between
    # incompatible values is a conflict, never a pick by canonical key
    assert values["latest_transfer"].state is ResultState.CONFLICT
    assert values["latest_transfer"].issue.value == "tie_with_incompatible_values"
    assert all(r.provenance.snapshot_hash == snap.snapshot_hash for r in values.values())


def test_a_new_schema_version_keeps_earlier_columns_and_their_values():
    snap = snapshot()
    extra = feature(
        "transfer_count", ST.ONCHAIN_LOG, "transfer_value", AggregationOp.COUNT_DISTINCT, version=V2
    )
    schema_v2 = FeatureSchema("integration", V2, (*SCHEMA_V1.features, extra))
    validate_evolution(SCHEMA_V1, schema_v2)
    for index, name in enumerate(SCHEMA_V1.feature_names):
        assert schema_v2.column_index(name) == index
        before = aggregate_feature(SCHEMA_V1.get(name), snap, ENTITY)
        after = aggregate_feature(schema_v2.get(name), snap, ENTITY)
        assert before.result_hash == after.result_hash
    assert aggregate_feature(extra, snap, ENTITY).value == 6  # distinct log objects of the entity


def test_future_evidence_changes_no_earlier_aggregate():
    base = list(
        load_ledger_json(FIXTURES / "synthetic" / "ledger_identity.json", require_synthetic=True)
    )
    cutoff = datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc)
    future = [
        make_synthetic_event(
            f"future-{i}",
            block_time=cutoff - timedelta(minutes=1),
            received_at=cutoff + timedelta(minutes=1 + i),
            log_index=50 + i,
            fields={"transfer_value": OV.observed(1000)},
        )
        for i in range(3)
    ]
    spec = SCHEMA_V1.get("transfer_value_total")
    assert (
        aggregate_feature(spec, make_asof_snapshot(cutoff, base), ENTITY).result_hash
        == aggregate_feature(spec, make_asof_snapshot(cutoff, base + future), ENTITY).result_hash
    )


def test_legacy_records_without_chain_id_load_but_their_identity_stays_unresolved():
    legacy = event_from_record(
        {
            "event_id": "legacy-1",
            "entity_id": ENTITY,
            "source_type": "onchain_log",
            "source": "old-provider",
            "block_time": "2026-03-01T10:00:00+00:00",
            "received_at": "2026-03-01T10:01:00+00:00",
            "block_number": 100,
            "block_hash": "0x" + "ab" * 32,
            "tx_hash": "0x" + "cd" * 32,
            "log_index": 0,
            "fields": {"transfer_value": {"state": "observed", "value": 4}},
        }
    )
    assert legacy.chain_id is None  # never inferred from a default chain
    snap = make_asof_snapshot(CUTOFF, [legacy])
    assert len(snap.events) == 1 and snap.exclusions == ()
    (obj,) = snap.objects
    assert obj.standing is EvidenceStanding.UNRESOLVED_IDENTITY
    result = aggregate_feature(SCHEMA_V1.get("transfer_value_total"), snap, ENTITY)
    assert result.state is ResultState.AMBIGUOUS  # a sum cannot trust an unidentified object


def test_snapshot_report_exposes_objects_roles_and_placement():
    report = snapshot().to_dict()
    assert report["contract"] == "asof-snapshot/2"
    assert report["admitted_count"] == 14 and report["object_count"] == 9
    roles = {o["role"] for obj in report["objects"] for o in obj["observations"]}
    assert roles == {"first", "duplicate", "independent_confirmation"}
    assert len(report["placement_divergences"]) == 1
    json.dumps(report, sort_keys=True)  # fully serializable


def test_cli_reports_objects_and_matches_the_library_hash(tmp_path):
    out = tmp_path / "report.json"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")])
        ),
    }
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "asof_snapshot.py"),
            "--ledger",
            str(FIXTURES / "synthetic" / "ledger_identity.json"),
            "--cutoff",
            CUTOFF,
            "--output",
            str(out),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["snapshot_hash"] == snapshot().snapshot_hash
    assert report["snapshot_hash"] in done.stdout
