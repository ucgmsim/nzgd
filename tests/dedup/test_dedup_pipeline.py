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


def _run_both_passes(conn: sqlite3.Connection, cfg, thresholds) -> tuple[int, int]:
    run_id = _start_run(conn)
    from nzgd.dedup.pass2_fuzzy import generate_fuzzy_merge_plan
    hash_plan = generate_hash_merge_plan(conn, cfg)
    c1, r1 = apply_merge_plan(conn, hash_plan, run_id, cfg)
    fuzzy_plan = generate_fuzzy_merge_plan(conn, cfg, thresholds)
    c2, r2 = apply_merge_plan(conn, fuzzy_plan, run_id, cfg)
    return c1 + c2, r1 + r2


_DEFAULT_THRESHOLDS = {
    "spatial_radius_m": 50,
    "date_window_days": 90,
    "name_similarity_min": 80,
    "trace_score_max": 0.05,
    "trace_resample_step_m": 0.05,
}


def test_slight_perturbation_pair_is_merged_by_fuzzy(fresh_db: sqlite3.Connection) -> None:
    trace_a = [(d, 1.0 + 0.1 * d, 0.01 + 0.001 * d, 0.0) for d in [0.1, 0.2, 0.3, 0.4]]
    trace_b = [(d, 1.001 + 0.1 * d, 0.0101 + 0.001 * d, 0.0) for d in [0.1, 0.2, 0.3, 0.4]]
    add_cpt_record(fresh_db, 1, -41.0, 174.0, "Wellington Site A", "2024-01-01")
    add_cpt_record(fresh_db, 2, -41.0, 174.0001, "Wellington Site A", "2024-01-05")
    add_cpt_report(fresh_db, 10, 1, trace_a)
    add_cpt_report(fresh_db, 20, 2, trace_b)
    total_c, total_r = _run_both_passes(fresh_db, CPT_TABLE_CONFIG, _DEFAULT_THRESHOLDS)
    assert (total_c, total_r) == (1, 1)
    audit = fresh_db.execute("SELECT match_pass FROM dedup_audit").fetchall()
    assert audit == [("fuzzy",)]


def test_nearby_but_distinct_sites_are_not_merged(fresh_db: sqlite3.Connection) -> None:
    # Same lat/lon (within radius) but very different traces and names → no merge
    trace_a = [(d, 1.0, 0.01, 0.0) for d in [0.1, 0.2, 0.3]]
    trace_b = [(d, 50.0, 0.5, 0.0) for d in [0.1, 0.2, 0.3]]  # qc 50x larger
    add_cpt_record(fresh_db, 1, -41.0, 174.0, "Site Alpha", "2024-01-01")
    add_cpt_record(fresh_db, 2, -41.0, 174.0001, "Site Beta", "2024-06-01")
    add_cpt_report(fresh_db, 10, 1, trace_a)
    add_cpt_report(fresh_db, 20, 2, trace_b)
    total_c, _ = _run_both_passes(fresh_db, CPT_TABLE_CONFIG, _DEFAULT_THRESHOLDS)
    assert total_c == 0
    merged = fresh_db.execute("SELECT COUNT(*) FROM nzgdrecord WHERE merged_into_nzgd_id IS NOT NULL").fetchone()[0]
    assert merged == 0


def test_3way_transitive_cluster_via_hash(fresh_db: sqlite3.Connection) -> None:
    # nzgd 1 has reports r10, r11. nzgd 2 has r20 (matches r10). nzgd 3 has r30 (matches r11).
    trace_x = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    trace_y = [(0.1, 2.0, 0.02, 0.0), (0.2, 2.1, 0.021, 0.0)]
    add_cpt_record(fresh_db, 1, -41.0, 174.0, "Has both", "2024-01-01")
    add_cpt_record(fresh_db, 2, -41.0, 174.0, None, None)
    add_cpt_record(fresh_db, 3, -41.0, 174.0, None, None)
    add_cpt_report(fresh_db, 10, 1, trace_x)
    add_cpt_report(fresh_db, 11, 1, trace_y)
    add_cpt_report(fresh_db, 20, 2, trace_x)
    add_cpt_report(fresh_db, 30, 3, trace_y)
    run_id = _start_run(fresh_db)
    plan = generate_hash_merge_plan(fresh_db, CPT_TABLE_CONFIG)
    apply_merge_plan(fresh_db, plan, run_id, CPT_TABLE_CONFIG)
    audit = fresh_db.execute(
        "SELECT cluster_id, canonical_nzgd_id, merged_nzgd_id FROM dedup_audit ORDER BY merged_nzgd_id"
    ).fetchall()
    # Two rows, both sharing cluster_id, both pointing to canonical 1
    assert len(audit) == 2
    assert audit[0][0] == audit[1][0]  # same cluster_id
    assert audit[0][1] == 1 and audit[1][1] == 1
    assert {audit[0][2], audit[1][2]} == {2, 3}


def test_partial_overlap_reparents_unique_reports(fresh_db: sqlite3.Connection) -> None:
    # nzgd 1: r10 (shared), r11 (unique). nzgd 2: r20 (shared with r10), r21 (unique).
    shared = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    unique_1 = [(0.1, 5.0, 0.05, 0.0), (0.2, 5.1, 0.051, 0.0), (0.3, 5.2, 0.052, 0.0)]  # 3 rows
    unique_2 = [(0.1, 7.0, 0.07, 0.0)]  # 1 row
    add_cpt_record(fresh_db, 1)
    add_cpt_record(fresh_db, 2)
    add_cpt_report(fresh_db, 10, 1, shared)
    add_cpt_report(fresh_db, 11, 1, unique_1)
    add_cpt_report(fresh_db, 20, 2, shared)
    add_cpt_report(fresh_db, 21, 2, unique_2)
    run_id = _start_run(fresh_db)
    plan = generate_hash_merge_plan(fresh_db, CPT_TABLE_CONFIG)
    apply_merge_plan(fresh_db, plan, run_id, CPT_TABLE_CONFIG)
    # nzgd 1 has more unique data (3 rows) → canonical; nzgd 2 merged into 1.
    assert fresh_db.execute("SELECT merged_into_nzgd_id FROM nzgdrecord WHERE nzgd_id = 2").fetchone()[0] == 1
    # r20 deleted; r21 re-parented to nzgd 1.
    reports = dict(fresh_db.execute("SELECT cpt_id, nzgd_id FROM cptreport").fetchall())
    assert reports == {10: 1, 11: 1, 21: 1}
    # cptmeasurements for r21 still present (re-parent doesn't touch measurements)
    assert fresh_db.execute("SELECT COUNT(*) FROM cptmeasurements WHERE cpt_id = 21").fetchone()[0] == 1


def test_metadata_conflict_picks_one_and_records_conflict(fresh_db: sqlite3.Connection) -> None:
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    # nzgd 1: NULL investigation_name, but set investigation_date so its non-null
    # metadata count matches the other two records (each has investigation_name).
    # All three then tie on tiebreaker 1; smallest nzgd_id picks nzgd 1 as canonical.
    add_cpt_record(fresh_db, 1, investigation_name=None, investigation_date="2024-01-01")
    add_cpt_record(fresh_db, 2, investigation_name="Foo", investigation_date=None)
    add_cpt_record(fresh_db, 3, investigation_name="Bar", investigation_date=None)
    add_cpt_report(fresh_db, 10, 1, trace)
    add_cpt_report(fresh_db, 20, 2, trace)
    add_cpt_report(fresh_db, 30, 3, trace)
    run_id = _start_run(fresh_db)
    plan = generate_hash_merge_plan(fresh_db, CPT_TABLE_CONFIG)
    apply_merge_plan(fresh_db, plan, run_id, CPT_TABLE_CONFIG)
    # Canonical's investigation_name is now non-NULL (one of Foo/Bar).
    name = fresh_db.execute("SELECT original_investigation_name FROM nzgdrecord WHERE nzgd_id = 1").fetchone()[0]
    assert name in {"Foo", "Bar"}
    # Conflict recorded in at least one audit row.
    rows = fresh_db.execute(
        "SELECT metadata_copied_json, metadata_conflicts_json FROM dedup_audit"
    ).fetchall()
    found_conflict = False
    for copied_json, conflict_json in rows:
        if conflict_json and "original_investigation_name" in conflict_json:
            found_conflict = True
            data = json.loads(conflict_json)["original_investigation_name"]
            values = {entry["value"] for entry in data}
            assert {"Foo", "Bar"}.issubset(values)
    assert found_conflict


def test_predicate_accepts_when_date_or_name_is_null(fresh_db: sqlite3.Connection) -> None:
    trace_a = [(d, 1.0 + 0.1 * d, 0.01 + 0.001 * d, 0.0) for d in [0.1, 0.2, 0.3, 0.4]]
    trace_b = [(d, 1.001 + 0.1 * d, 0.0101 + 0.001 * d, 0.0) for d in [0.1, 0.2, 0.3, 0.4]]
    add_cpt_record(fresh_db, 1, -41.0, 174.0, None, None)  # both sentinels
    add_cpt_record(fresh_db, 2, -41.0, 174.0001, None, None)
    add_cpt_report(fresh_db, 10, 1, trace_a)
    add_cpt_report(fresh_db, 20, 2, trace_b)
    total_c, total_r = _run_both_passes(fresh_db, CPT_TABLE_CONFIG, _DEFAULT_THRESHOLDS)
    assert (total_c, total_r) == (1, 1)


def test_spt_cluster_cascades_dependent_table_deletes(fresh_db: sqlite3.Connection) -> None:
    trace = [(1.0, 5, 5, 5), (2.0, 7, 7, 7), (3.0, 10, 10, 10)]
    add_bh_record(fresh_db, 1)
    add_bh_record(fresh_db, 2)
    add_spt_report(fresh_db, 100, 1, trace)
    add_spt_report(fresh_db, 200, 2, trace)
    # Add dependents on the duplicate (spt_id 200) so we can verify cascade
    fresh_db.execute("INSERT INTO soilmeasurements (soil_measurement_id, spt_id, top_depth_m, bottom_depth_m) VALUES (1000, 200, 0.0, 1.0)")
    fresh_db.execute("INSERT INTO soilmeasurements (soil_measurement_id, spt_id, top_depth_m, bottom_depth_m) VALUES (1001, 200, 1.0, 2.0)")
    fresh_db.execute("INSERT INTO densitymeasurements (density_measurement_id, spt_id, top_depth_m, bottom_depth_m, density_keyword) VALUES (2000, 200, 0.0, 1.0, 'loose')")
    fresh_db.execute("INSERT INTO soilmeasurementsoiltype (soil_measurement_id, soil_type_id) VALUES (1000, 1), (1000, 2), (1001, 3)")
    # And one on the canonical to confirm we don't touch it
    fresh_db.execute("INSERT INTO soilmeasurements (soil_measurement_id, spt_id, top_depth_m, bottom_depth_m) VALUES (1002, 100, 0.0, 1.0)")
    fresh_db.execute("INSERT INTO soilmeasurementsoiltype (soil_measurement_id, soil_type_id) VALUES (1002, 1)")

    run_id = _start_run(fresh_db)
    plan = generate_hash_merge_plan(fresh_db, SPT_TABLE_CONFIG)
    apply_merge_plan(fresh_db, plan, run_id, SPT_TABLE_CONFIG)

    # spt_id 200 and all its dependents are gone
    assert fresh_db.execute("SELECT COUNT(*) FROM sptreport WHERE spt_id = 200").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM sptmeasurements WHERE spt_id = 200").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM soilmeasurements WHERE spt_id = 200").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM densitymeasurements WHERE spt_id = 200").fetchone()[0] == 0
    assert fresh_db.execute(
        "SELECT COUNT(*) FROM soilmeasurementsoiltype WHERE soil_measurement_id IN (1000, 1001)"
    ).fetchone()[0] == 0
    # Canonical untouched
    assert fresh_db.execute("SELECT COUNT(*) FROM soilmeasurements WHERE spt_id = 100").fetchone()[0] == 1
    assert fresh_db.execute(
        "SELECT COUNT(*) FROM soilmeasurementsoiltype WHERE soil_measurement_id = 1002"
    ).fetchone()[0] == 1
    # Merge recorded with record_type = 'BH'
    assert fresh_db.execute("SELECT record_type FROM dedup_audit").fetchall() == [("BH",)]
