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


def test_spt_with_string_blowcount_values_hashes_without_error(fresh_db: sqlite3.Connection) -> None:
    # Regression: SQLite stores ISPT_REP as TEXT even though the schema declares INTEGER.
    # _encode_value must handle strings without raising ValueError.
    add_bh_record(fresh_db, 1)
    add_bh_record(fresh_db, 2)
    for spt_id in (100, 200):
        fresh_db.execute(
            "INSERT INTO sptreport (spt_id, nzgd_id, source_file) VALUES (?,?,?)",
            (spt_id, spt_id // 100, "synthetic.ags"),
        )
        fresh_db.executemany(
            "INSERT INTO sptmeasurements (spt_id, depth_m, ISPT_MAIN, ISPT_NVAL, ISPT_REP) "
            "VALUES (?,?,?,?,?)",
            [
                (spt_id, 1.5, None, 17, "1/4//5/4/4/4,"),
                (spt_id, 3.0, 12,   12, "3/6/6,"),
                (spt_id, 4.5, 8,    8,  "2/4/4,"),
            ],
        )

    run_id = _start_run(fresh_db)
    plan = generate_hash_merge_plan(fresh_db, SPT_TABLE_CONFIG)
    apply_merge_plan(fresh_db, plan, run_id, SPT_TABLE_CONFIG)

    audit = fresh_db.execute(
        "SELECT match_pass, record_type FROM dedup_audit"
    ).fetchall()
    assert len(audit) == 1
    assert audit[0] == ("hash", "BH")


def test_spt_fuzzy_pass_handles_string_blowcount_without_crash(fresh_db: sqlite3.Connection) -> None:
    """Real-data SPT records have string ISPT_REP values; fuzzy pass must not crash on them."""
    from nzgd.dedup.pass2_fuzzy import generate_fuzzy_merge_plan
    add_bh_record(fresh_db, 1, lat=-41.0, lon=174.0, investigation_name="Site Alpha", investigation_date="2024-01-01")
    add_bh_record(fresh_db, 2, lat=-41.0, lon=174.0001, investigation_name="Site Beta", investigation_date="2024-06-01")
    fresh_db.execute("INSERT INTO sptreport (spt_id, nzgd_id) VALUES (100, 1)")
    fresh_db.execute("INSERT INTO sptreport (spt_id, nzgd_id) VALUES (200, 2)")
    # ISPT_REP holds a string in both records; values DIFFER so the pair is not a real duplicate.
    fresh_db.executemany(
        "INSERT INTO sptmeasurements (spt_id, depth_m, ISPT_MAIN, ISPT_NVAL, ISPT_REP) VALUES (?,?,?,?,?)",
        [
            (100, 1.5, 5, 17, "1/4//5/4/4/4,"),
            (100, 3.0, 7, 19, "3/3//4/5/4/6,"),
            (100, 4.5, 9, 25, "5/6//6/7/8/8,"),
            (200, 1.5, 50, 100, "9/9//9/9/9/9,"),  # very different
            (200, 3.0, 60, 110, "8/8//8/8/8/8,"),
            (200, 4.5, 70, 120, "7/7//7/7/7/7,"),
        ],
    )

    _start_run(fresh_db)
    # Should not raise. Returns plan (likely empty since traces differ and strings become NaN).
    plan = generate_fuzzy_merge_plan(fresh_db, SPT_TABLE_CONFIG, _DEFAULT_THRESHOLDS)
    # No merges expected: the records are distinct and their string blow-counts become NaN, so trace_score is NaN, predicate rejects.
    assert plan == []


# === Pass 0 (within-record consolidation) scenarios ===


def _run_pass0(conn: sqlite3.Connection, cfg) -> tuple[int, int]:
    from nzgd.dedup.pass0_within_record import (
        apply_within_record_consolidation_plan,
        generate_within_record_consolidation_plan,
    )
    run_id = _start_run(conn)
    thresholds = {"trace_score_max": 0.05, "trace_resample_step_m": 0.05}
    plan = generate_within_record_consolidation_plan(conn, cfg, thresholds)
    return apply_within_record_consolidation_plan(conn, plan, run_id, cfg)


def test_pass0_typical_multi_sheet_collapse(fresh_db: sqlite3.Connection) -> None:
    """5 cptreport rows from 2 stems; both stems have a data-bearing row with the
    same trace → stems link via hash → all 5 rows collapse into 1 surviving row."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0), (0.3, 1.2, 0.012, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=8920, lat=-43.54, lon=172.66)
    add_cpt_report(fresh_db, cpt_id=11533, nzgd_id=8920, trace=[],    source_file="CPT_8920_AGS01.ags_sheet_0")
    add_cpt_report(fresh_db, cpt_id=11534, nzgd_id=8920, trace=trace, source_file="CPT_8920_AGS01.ags_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11535, nzgd_id=8920, trace=[],    source_file="CPT_8920_Raw01.xlsx_sheet_0")
    add_cpt_report(fresh_db, cpt_id=11536, nzgd_id=8920, trace=trace, source_file="CPT_8920_Raw01.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11537, nzgd_id=8920, trace=[],    source_file="CPT_8920_Raw01.xlsx_sheet_Header")

    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (1, 4)
    # Canonical: smallest cpt_id with has_data=True → 11534
    remaining = [r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport ORDER BY cpt_id")]
    assert remaining == [11534]
    audit = fresh_db.execute(
        "SELECT canonical_nzgd_id, merged_nzgd_id, match_pass, report_pairs_json FROM dedup_audit"
    ).fetchall()
    assert len(audit) == 1
    assert audit[0][0] == 8920 and audit[0][1] == 8920 and audit[0][2] == "within_record"
    absorbed = json.loads(audit[0][3])
    assert len(absorbed) == 4
    assert {a["absorbed_report_id"] for a in absorbed} == {11533, 11535, 11536, 11537}


def test_pass0_sentinel_aware_enrichment(fresh_db: sqlite3.Connection) -> None:
    """Canonical has gwl=0 (sentinel); absorbed has gwl=5.2 (real). Canonical gets updated."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_Header")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m = 0 WHERE cpt_id = 10")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m = 5.2 WHERE cpt_id = 11")

    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (1, 1)
    gwl = fresh_db.execute("SELECT extracted_gwl_m FROM cptreport WHERE cpt_id = 10").fetchone()[0]
    assert gwl == 5.2
    copied = fresh_db.execute("SELECT metadata_copied_json FROM dedup_audit").fetchone()[0]
    assert copied is not None
    assert "extracted_gwl_m" in json.loads(copied)


def test_pass0_plausibility_conflict_records_all(fresh_db: sqlite3.Connection) -> None:
    """Canonical NULL; two absorbed with conflicting in-range values; smallest-id wins, both logged."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_H1")
    add_cpt_report(fresh_db, cpt_id=12, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_H2")
    fresh_db.execute("UPDATE cptreport SET tip_net_area_ratio = 0.80 WHERE cpt_id = 11")
    fresh_db.execute("UPDATE cptreport SET tip_net_area_ratio = 0.92 WHERE cpt_id = 12")

    _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    chosen = fresh_db.execute("SELECT tip_net_area_ratio FROM cptreport WHERE cpt_id = 10").fetchone()[0]
    assert chosen == 0.80  # smallest absorbed id wins
    audit = fresh_db.execute("SELECT metadata_copied_json, metadata_conflicts_json FROM dedup_audit").fetchone()
    conflicts = json.loads(audit[1])
    assert "tip_net_area_ratio" in conflicts
    values = {e["value"] for e in conflicts["tip_net_area_ratio"]}
    assert values == {0.80, 0.92}


def test_pass0_multi_cpt_per_nzgd_stays_separate(fresh_db: sqlite3.Connection) -> None:
    """Two stems with non-matching data-bearing traces → 2 surviving rows."""
    trace_a = [(d, 1.0, 0.01, 0.0) for d in [0.1, 0.2, 0.3]]
    trace_b = [(d, 50.0, 0.5, 0.0) for d in [0.1, 0.2, 0.3]]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace_a, source_file="X.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=1, trace=trace_b, source_file="Y.xlsx_sheet_Data")
    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert n_clusters == 0
    remaining = sorted(r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport"))
    assert remaining == [10, 20]


def test_pass0_all_no_data_cluster(fresh_db: sqlite3.Connection) -> None:
    """3 rows in one stem, all has_cpt_data=0 → 1 surviving row (smallest id)."""
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=[], source_file="A.xlsx_sheet_0")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=[], source_file="A.xlsx_sheet_H1")
    add_cpt_report(fresh_db, cpt_id=12, nzgd_id=1, trace=[], source_file="A.xlsx_sheet_H2")
    fresh_db.execute("UPDATE cptreport SET has_cpt_data = 0")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m = 3.0 WHERE cpt_id = 12")
    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (1, 2)
    remaining = [r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport")]
    assert remaining == [10]
    gwl = fresh_db.execute("SELECT extracted_gwl_m FROM cptreport WHERE cpt_id = 10").fetchone()[0]
    assert gwl == 3.0


def test_pass0_stem_only_attachment(fresh_db: sqlite3.Connection) -> None:
    """Stem A: 1 data + 2 no-data; stem B: 1 no-data → stem A cluster collapses, stem B singleton survives."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_H1")
    add_cpt_report(fresh_db, cpt_id=12, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_H2")
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=1, trace=[],    source_file="B.ags_sheet_0")
    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (1, 2)
    remaining = sorted(r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport"))
    assert remaining == [10, 20]


def test_pass0_fuzzy_within_record_match(fresh_db: sqlite3.Connection) -> None:
    """Two data-bearing rows from different stems with slightly perturbed traces consolidate via fuzzy."""
    trace_a = [(d, 1.0 + 0.1 * d, 0.01 + 0.001 * d, 0.0) for d in [0.1, 0.2, 0.3, 0.4]]
    trace_b = [(d, 1.001 + 0.1 * d, 0.0101 + 0.001 * d, 0.0) for d in [0.1, 0.2, 0.3, 0.4]]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace_a, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=1, trace=trace_b, source_file="B.ags_sheet_0")
    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (1, 1)
    remaining = [r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport")]
    assert remaining == [10]


def test_pass0_then_pass1_cross_record(fresh_db: sqlite3.Connection) -> None:
    """Two nzgd_ids each with within-record duplicates, and they are also cross-record duplicates."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0, investigation_name="Site A")
    add_cpt_record(fresh_db, nzgd_id=2, lat=-41.0, lon=174.0001, investigation_name="Site A")
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_H")
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=2, trace=trace, source_file="B.ags_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=21, nzgd_id=2, trace=[],    source_file="B.ags_sheet_H")

    run_id = _start_run(fresh_db)
    thresholds = {"trace_score_max": 0.05, "trace_resample_step_m": 0.05}
    from nzgd.dedup.pass0_within_record import (
        apply_within_record_consolidation_plan,
        generate_within_record_consolidation_plan,
    )
    plan0 = generate_within_record_consolidation_plan(fresh_db, CPT_TABLE_CONFIG, thresholds)
    apply_within_record_consolidation_plan(fresh_db, plan0, run_id, CPT_TABLE_CONFIG)
    plan1 = generate_hash_merge_plan(fresh_db, CPT_TABLE_CONFIG)
    apply_merge_plan(fresh_db, plan1, run_id, CPT_TABLE_CONFIG)

    merged = fresh_db.execute("SELECT merged_into_nzgd_id FROM nzgdrecord WHERE nzgd_id = 2").fetchone()[0]
    assert merged == 1
    remaining = sorted(r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport"))
    assert remaining == [10]
    audit_passes = sorted(r[0] for r in fresh_db.execute("SELECT match_pass FROM dedup_audit"))
    assert "within_record" in audit_passes and "hash" in audit_passes


def test_pass0_schema_migration_drops_legacy_column(fresh_db: sqlite3.Connection, tmp_path) -> None:
    """A DB carrying the legacy column gets it dropped by apply_dedup_schema."""
    import sqlite3 as _sqlite3
    from nzgd.dedup.schema import apply_dedup_schema

    db_path = tmp_path / "legacy.db"
    conn = _sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
        CREATE TABLE nzgdrecord (nzgd_id INTEGER PRIMARY KEY);
        CREATE TABLE cptreport (
            cpt_id INTEGER PRIMARY KEY, nzgd_id INTEGER, source_file TEXT NOT NULL,
            has_cpt_data INTEGER NOT NULL DEFAULT 1,
            cpt_data_duplicate_of_cpt_id INTEGER
        );
        INSERT INTO cptreport (cpt_id, nzgd_id, source_file, cpt_data_duplicate_of_cpt_id)
            VALUES (1, 100, 'x.xlsx_sheet_0', 99);
    """)
    apply_dedup_schema(conn)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(cptreport)")]
    assert "cpt_data_duplicate_of_cpt_id" not in cols
    n = conn.execute("SELECT COUNT(*) FROM cptreport").fetchone()[0]
    assert n == 1
    conn.close()


def test_pass0_spt_consolidation_cascade(fresh_db: sqlite3.Connection) -> None:
    """SPT within-record consolidation: deletion cascades through soilmeasurements, etc."""
    trace = [(1.0, 5, 5, 5), (2.0, 7, 7, 7), (3.0, 10, 10, 10)]
    add_bh_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_spt_report(fresh_db, spt_id=100, nzgd_id=1, trace=trace, source_file="X.ags_sheet_0")
    add_spt_report(fresh_db, spt_id=101, nzgd_id=1, trace=[],    source_file="X.ags_sheet_H")
    fresh_db.execute("INSERT INTO soilmeasurements (soil_measurement_id, spt_id, top_depth_m, bottom_depth_m) VALUES (5000, 101, 0, 1)")
    fresh_db.execute("INSERT INTO densitymeasurements (density_measurement_id, spt_id, top_depth_m, bottom_depth_m, density_keyword) VALUES (6000, 101, 0, 1, 'loose')")
    fresh_db.execute("INSERT INTO soilmeasurementsoiltype (soil_measurement_id, soil_type_id) VALUES (5000, 1)")

    n_clusters, n_records = _run_pass0(fresh_db, SPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (1, 1)
    assert fresh_db.execute("SELECT COUNT(*) FROM sptreport WHERE spt_id = 101").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM soilmeasurements WHERE spt_id = 101").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM densitymeasurements WHERE spt_id = 101").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM soilmeasurementsoiltype WHERE soil_measurement_id = 5000").fetchone()[0] == 0


def test_pass0_single_file_multi_cpt_split(fresh_db: sqlite3.Connection) -> None:
    """One stem with 2 data-bearing rows with non-matching traces + 2 no-data rows → 2 sub-clusters."""
    trace_a = [(d, 1.0, 0.01, 0.0) for d in [0.1, 0.2, 0.3]]
    trace_b = [(d, 50.0, 0.5, 0.0) for d in [0.1, 0.2, 0.3]]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace_a, source_file="X.xlsx_sheet_DataA")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=trace_b, source_file="X.xlsx_sheet_DataB")
    add_cpt_report(fresh_db, cpt_id=12, nzgd_id=1, trace=[],      source_file="X.xlsx_sheet_H1")
    add_cpt_report(fresh_db, cpt_id=13, nzgd_id=1, trace=[],      source_file="X.xlsx_sheet_H2")
    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert n_clusters == 1
    assert n_records == 2
    remaining = sorted(r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport"))
    assert remaining == [10, 11]


def test_cross_record_plausibility_aware_enrichment(fresh_db: sqlite3.Connection) -> None:
    """Cross-record hash match: canonical's latitude out-of-range → replaced by absorbed's plausible value."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-1.0, lon=174.0, investigation_name="Site A")
    add_cpt_record(fresh_db, nzgd_id=2, lat=-41.0, lon=174.0, investigation_name=None)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=2, trace=trace, source_file="B.xlsx_sheet_Data")
    run_id = _start_run(fresh_db)
    plan = generate_hash_merge_plan(fresh_db, CPT_TABLE_CONFIG)
    apply_merge_plan(fresh_db, plan, run_id, CPT_TABLE_CONFIG)
    lat = fresh_db.execute("SELECT latitude FROM nzgdrecord WHERE merged_into_nzgd_id IS NULL").fetchone()[0]
    assert lat == -41.0
    audit = fresh_db.execute("SELECT metadata_copied_json FROM dedup_audit").fetchone()[0]
    assert audit is not None and "latitude" in json.loads(audit)


def test_pass0_sentinel_preserved_when_no_useful_alternative(fresh_db: sqlite3.Connection) -> None:
    """gwl=0 on canonical AND on absorbed; no useful value anywhere → canonical's 0 stays (not NULL'd)."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_H")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m = 0 WHERE cpt_id IN (10, 11)")
    _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    gwl = fresh_db.execute("SELECT extracted_gwl_m FROM cptreport WHERE cpt_id = 10").fetchone()[0]
    assert gwl == 0
    copied = fresh_db.execute("SELECT metadata_copied_json FROM dedup_audit").fetchone()[0]
    if copied:
        assert "extracted_gwl_m" not in json.loads(copied)


def test_pass0_spt_with_all_nan_depths_does_not_crash(fresh_db: sqlite3.Connection) -> None:
    """Real-data SPT depths can be non-numeric (coerced to NaN); fuzzy path must not crash."""
    from nzgd.dedup.pass0_within_record import (
        apply_within_record_consolidation_plan,
        generate_within_record_consolidation_plan,
    )
    add_bh_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_spt_report(fresh_db, spt_id=100, nzgd_id=1, trace=[], source_file="X.ags_sheet_0")
    add_spt_report(fresh_db, spt_id=101, nzgd_id=1, trace=[], source_file="X.ags_sheet_H")
    # Insert NaN-depth rows directly (bypass the trace helper that requires float tuples)
    fresh_db.execute(
        "INSERT INTO sptmeasurements (spt_id, depth_m, ISPT_MAIN, ISPT_NVAL, ISPT_REP) "
        "VALUES (?, ?, ?, ?, ?)", (100, "garbage", 5, 5, "1/2/3,")
    )
    fresh_db.execute(
        "INSERT INTO sptmeasurements (spt_id, depth_m, ISPT_MAIN, ISPT_NVAL, ISPT_REP) "
        "VALUES (?, ?, ?, ?, ?)", (101, "garbage", 7, 7, "2/3/4,")
    )
    run_id = _start_run(fresh_db)
    thresholds = {"trace_score_max": 0.05, "trace_resample_step_m": 0.05}
    # Must not raise
    plan = generate_within_record_consolidation_plan(fresh_db, SPT_TABLE_CONFIG, thresholds)
    apply_within_record_consolidation_plan(fresh_db, plan, run_id, SPT_TABLE_CONFIG)


def test_pass0_singleton_independent_stem_not_absorbed_into_matching_pair(fresh_db: sqlite3.Connection) -> None:
    """Regression: idx=1 singleton stem must not collide with 1-based component label 1.

    Three stems for one nzgd_id, in insertion order: A.xlsx (data, trace T at idx=0),
    B.ags (no data, independent, idx=1), C.xlsx (data, same trace T, idx=2). The
    cross-stem edge (0, 2) creates component {0:1, 2:1}. A buggy fallback that uses
    idx as the key for singletons would map idx=1 to label 1, incorrectly absorbing
    B.ags's no-data report.
    """
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace, source_file="A.xlsx_sheet_Data")
    # The middle stem (B.ags) is an INDEPENDENT no-data row
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=1, trace=[], source_file="B.ags_sheet_0")
    add_cpt_report(fresh_db, cpt_id=30, nzgd_id=1, trace=trace, source_file="C.xlsx_sheet_Data")
    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    # Cluster 1: stems A and C link via matching data; canonical = 10 (smallest data-bearing id), absorbs 30. 1 record absorbed.
    # Cluster 2: stem B is a singleton with 1 row → no consolidation needed (only triggers when len(cluster) > 1).
    assert (n_clusters, n_records) == (1, 1)
    remaining = sorted(r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport"))
    assert remaining == [10, 20]  # B.ags's row (20) survives
