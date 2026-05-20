"""End-to-end integration scenarios for the dedup pipeline."""

import json
import sqlite3

import pytest

from nzgd.dedup.data_types import CPT_TABLE_CONFIG, SPT_TABLE_CONFIG
from nzgd.dedup.executor import apply_merge_plan
from nzgd.dedup.pass1_hash import generate_hash_merge_plan
from nzgd.dedup.schema import apply_dedup_schema
from tests.dedup.conftest import (
    add_cpt_record,
    add_cpt_report,
    add_bh_record,
    add_spt_report,
)


def _start_run(conn: sqlite3.Connection) -> int:
    apply_dedup_schema(conn)
    cur = conn.execute(
        "INSERT INTO dedup_run (started_at, source_db_path, script_version, config_snapshot_json) "
        "VALUES (?, ?, ?, ?)",
        ("2026-05-20T00:00:00Z", ":memory:", "test", "{}"),
    )
    return cur.lastrowid


def test_exact_duplicate_pair_is_merged(fresh_db: sqlite3.Connection) -> None:
    trace = [(0.10, 1.00, 0.010, 0.000),
             (0.20, 1.10, 0.011, 0.000),
             (0.30, 1.20, 0.012, 0.000)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0,
                   investigation_name="Site A", investigation_date="2024-01-15")
    add_cpt_record(fresh_db, nzgd_id=2, lat=-41.0, lon=174.0,
                   investigation_name=None, investigation_date="2024-02-01")
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace)
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=2, trace=trace)

    run_id = _start_run(fresh_db)
    plan = generate_hash_merge_plan(fresh_db, CPT_TABLE_CONFIG)
    n_clusters, n_records = apply_merge_plan(fresh_db, plan, run_id, CPT_TABLE_CONFIG)

    assert n_clusters == 1 and n_records == 1
    # nzgd 1 has more non-null metadata (investigation_name set), so it wins.
    merged_into = fresh_db.execute(
        "SELECT merged_into_nzgd_id FROM nzgdrecord WHERE nzgd_id = 2"
    ).fetchone()[0]
    assert merged_into == 1
    assert fresh_db.execute(
        "SELECT merged_into_nzgd_id FROM nzgdrecord WHERE nzgd_id = 1"
    ).fetchone()[0] is None
    # cpt_id 20 deleted; cpt_id 10 still there
    remaining = [r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport")]
    assert remaining == [10]
    # cptmeasurements for cpt_id 20 deleted
    assert fresh_db.execute("SELECT COUNT(*) FROM cptmeasurements WHERE cpt_id = 20").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM cptmeasurements WHERE cpt_id = 10").fetchone()[0] == 3
    # audit row exists with hash pass
    audit = fresh_db.execute(
        "SELECT cluster_id, canonical_nzgd_id, merged_nzgd_id, match_pass FROM dedup_audit"
    ).fetchall()
    assert audit == [(1, 1, 2, "hash")]
