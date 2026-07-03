"""Integration tests for the constant-column quality filter."""

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nzgd.dedup.data_types import CPT_TABLE_CONFIG
from nzgd.dedup.quality_filter import (
    apply_quality_filter,
    delete_emptied_records,
    find_constant_column_reports,
)
from nzgd.dedup.reports import (
    write_quality_filter_report,
    write_quality_reject_record_report,
)
from nzgd.dedup.schema import apply_dedup_schema
from nzgd.scripts.db.deduplicate import app
from tests.dedup.conftest import _make_fresh_db, add_cpt_record, add_cpt_report

_COLUMNS = ["depth_m", "qc_MPa", "fs_MPa", "u2_MPa"]


def _populate_scenarios(conn: sqlite3.Connection) -> None:
    add_cpt_record(conn, nzgd_id=1)
    # R1 (10): everything varies -> kept
    add_cpt_report(conn, 10, 1, [(0.1, 1.0, 0.010, 0.05),
                                 (0.2, 1.1, 0.011, 0.06),
                                 (0.3, 1.2, 0.012, 0.07)])
    # R2 (11): constant qc -> discard
    add_cpt_report(conn, 11, 1, [(0.1, 2.0, 0.010, 0.05),
                                 (0.2, 2.0, 0.011, 0.06),
                                 (0.3, 2.0, 0.012, 0.07)])
    # R3 (12): constant u2 = 0 with good qc/fs -> discard (validates u2 inclusion)
    add_cpt_report(conn, 12, 1, [(0.1, 1.0, 0.010, 0.0),
                                 (0.2, 1.1, 0.011, 0.0),
                                 (0.3, 1.2, 0.012, 0.0)])
    # R4 (13): constant fs -> discard
    add_cpt_report(conn, 13, 1, [(0.1, 1.0, 0.02, 0.05),
                                 (0.2, 1.1, 0.02, 0.06),
                                 (0.3, 1.2, 0.02, 0.07)])
    # R5 (14): constant qc but only 2 rows (< min_non_null_rows) -> kept
    add_cpt_report(conn, 14, 1, [(0.1, 3.0, 0.02, 0.05),
                                 (0.2, 3.0, 0.03, 0.06)])


def test_find_constant_column_reports(fresh_db: sqlite3.Connection) -> None:
    _populate_scenarios(fresh_db)
    entries = find_constant_column_reports(fresh_db, CPT_TABLE_CONFIG, _COLUMNS, 3)
    by_id = {e.report_id: e for e in entries}
    assert set(by_id) == {11, 12, 13}
    assert by_id[11].constant_columns == {"qc_MPa": 2.0}
    assert by_id[12].constant_columns == {"u2_MPa": 0.0}
    assert by_id[13].constant_columns == {"fs_MPa": 0.02}
    assert by_id[11].nzgd_id == 1
    assert by_id[11].n_rows == 3
    assert by_id[11].reason == "constant_column"


def test_invalid_column_raises(fresh_db: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        find_constant_column_reports(fresh_db, CPT_TABLE_CONFIG, ["not_a_column"], 3)


def _start_run(conn: sqlite3.Connection) -> int:
    apply_dedup_schema(conn)
    cur = conn.execute(
        "INSERT INTO dedup_run (started_at, source_db_path, script_version, config_snapshot_json) "
        "VALUES (?, ?, ?, ?)",
        ("2026-07-03T00:00:00Z", ":memory:", "test", "{}"),
    )
    return cur.lastrowid


def test_apply_quality_filter_discards_and_audits(fresh_db: sqlite3.Connection, tmp_path: Path) -> None:
    _populate_scenarios(fresh_db)
    run_id = _start_run(fresh_db)
    failures: list[dict] = []
    entries = find_constant_column_reports(fresh_db, CPT_TABLE_CONFIG, _COLUMNS, 3)
    n = apply_quality_filter(fresh_db, entries, run_id, CPT_TABLE_CONFIG, failures=failures)

    assert n == 3
    assert failures == []
    remaining = sorted(r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport"))
    assert remaining == [10, 14]  # R1 and R5 kept; R2/R3/R4 discarded
    gone = fresh_db.execute(
        "SELECT COUNT(*) FROM cptmeasurements WHERE cpt_id IN (11, 12, 13)"
    ).fetchone()[0]
    assert gone == 0
    kept = fresh_db.execute(
        "SELECT COUNT(*) FROM cptmeasurements WHERE cpt_id = 10"
    ).fetchone()[0]
    assert kept == 3

    rej = fresh_db.execute(
        "SELECT report_id, reason, constant_columns_json, n_rows "
        "FROM quality_reject ORDER BY report_id"
    ).fetchall()
    assert [r[0] for r in rej] == [11, 12, 13]
    assert all(r[1] == "constant_column" for r in rej)
    assert json.loads(rej[1][2]) == {"u2_MPa": 0.0}

    out = tmp_path / "qf.csv"
    write_quality_filter_report(fresh_db, run_id, out)
    text = out.read_text()
    assert "report_id" in text and "constant_columns" in text
    assert "u2_MPa" in text and "constant_column" in text


def test_delete_emptied_records(fresh_db: sqlite3.Connection, tmp_path: Path) -> None:
    # RecA (nzgd 1): single report, constant u2 -> record emptied -> deleted
    add_cpt_record(fresh_db, nzgd_id=1)
    add_cpt_report(fresh_db, 10, 1, [(0.1, 1.0, 0.010, 0.0),
                                     (0.2, 1.1, 0.011, 0.0),
                                     (0.3, 1.2, 0.012, 0.0)])
    # RecB (nzgd 2): constant report + a good report -> record kept
    add_cpt_record(fresh_db, nzgd_id=2)
    add_cpt_report(fresh_db, 20, 2, [(0.1, 1.0, 0.010, 0.0),
                                     (0.2, 1.1, 0.011, 0.0),
                                     (0.3, 1.2, 0.012, 0.0)])
    add_cpt_report(fresh_db, 21, 2, [(0.1, 1.0, 0.010, 0.05),
                                     (0.2, 1.1, 0.011, 0.06),
                                     (0.3, 1.2, 0.012, 0.07)])
    # RecC (nzgd 3): single constant report BUT a merge tombstone -> NOT deleted
    add_cpt_record(fresh_db, nzgd_id=3)
    add_cpt_report(fresh_db, 30, 3, [(0.1, 1.0, 0.010, 0.0),
                                     (0.2, 1.1, 0.011, 0.0),
                                     (0.3, 1.2, 0.012, 0.0)])

    run_id = _start_run(fresh_db)
    # Make RecC a merge tombstone (redirect to RecB); the guard must protect it.
    fresh_db.execute("UPDATE nzgdrecord SET merged_into_nzgd_id = 2 WHERE nzgd_id = 3")

    failures: list[dict] = []
    entries = find_constant_column_reports(fresh_db, CPT_TABLE_CONFIG, _COLUMNS, 3)
    apply_quality_filter(fresh_db, entries, run_id, CPT_TABLE_CONFIG, failures=failures)
    n_emptied = delete_emptied_records(fresh_db, run_id, CPT_TABLE_CONFIG, failures=failures)

    assert n_emptied == 1
    assert failures == []
    # nzgd 1 deleted; nzgd 2 (good report) and nzgd 3 (tombstone) survive.
    assert sorted(r[0] for r in fresh_db.execute("SELECT nzgd_id FROM nzgdrecord")) == [2, 3]
    # RecB's good report survives.
    assert sorted(r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport")) == [21]
    # audit: exactly one emptied-record row, for nzgd 1.
    qrr = fresh_db.execute(
        "SELECT nzgd_id, reason, n_reports_discarded FROM quality_reject_record"
    ).fetchall()
    assert qrr == [(1, "emptied_by_quality_filter", 1)]

    out = tmp_path / "qrr.csv"
    write_quality_reject_record_report(fresh_db, run_id, out)
    text = out.read_text()
    assert "nzgd_id" in text and "emptied_by_quality_filter" in text


def test_quality_filter_runs_in_full_pipeline(tmp_path: Path) -> None:
    src = tmp_path / "source.db"
    conn = _make_fresh_db(src)
    add_cpt_record(conn, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_record(conn, nzgd_id=2, lat=-41.0, lon=174.0)
    # normal report -> survives
    add_cpt_report(conn, 10, 1, [(0.1, 1.0, 0.010, 0.05),
                                 (0.2, 1.1, 0.011, 0.06),
                                 (0.3, 1.2, 0.012, 0.07)])
    # constant-qc report -> discarded by the filter before any dedup pass
    add_cpt_report(conn, 20, 2, [(0.1, 2.0, 0.010, 0.05),
                                 (0.2, 2.0, 0.011, 0.06),
                                 (0.3, 2.0, 0.012, 0.07)])
    conn.commit()
    conn.close()

    target = tmp_path / "deduped.db"
    result = CliRunner().invoke(
        app, ["--source", str(src), "--target", str(target), "--skip-spt"]
    )
    assert result.exit_code == 0, result.output

    out = sqlite3.connect(target)
    try:
        remaining = [r[0] for r in out.execute("SELECT cpt_id FROM cptreport ORDER BY cpt_id")]
        assert remaining == [10]
        rej = out.execute("SELECT report_id, record_type, reason FROM quality_reject").fetchall()
        assert rej == [(20, "CPT", "constant_column")]
    finally:
        out.close()
    assert (tmp_path / "quality_filter_report.csv").exists()


def test_emptied_record_deleted_in_full_pipeline(tmp_path: Path) -> None:
    src = tmp_path / "source.db"
    conn = _make_fresh_db(src)
    # normal record -> survives with its report
    add_cpt_record(conn, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(conn, 10, 1, [(0.1, 1.0, 0.010, 0.05),
                                 (0.2, 1.1, 0.011, 0.06),
                                 (0.3, 1.2, 0.012, 0.07)])
    # single-report record with a constant-u2 trace -> report discarded AND record deleted
    add_cpt_record(conn, nzgd_id=2, lat=-41.0, lon=174.0)
    add_cpt_report(conn, 20, 2, [(0.1, 1.0, 0.010, 0.0),
                                 (0.2, 1.1, 0.011, 0.0),
                                 (0.3, 1.2, 0.012, 0.0)])
    conn.commit()
    conn.close()

    target = tmp_path / "deduped.db"
    result = CliRunner().invoke(app, ["--source", str(src), "--target", str(target), "--skip-spt"])
    assert result.exit_code == 0, result.output

    out = sqlite3.connect(target)
    try:
        assert [r[0] for r in out.execute("SELECT cpt_id FROM cptreport ORDER BY cpt_id")] == [10]
        assert [r[0] for r in out.execute("SELECT nzgd_id FROM nzgdrecord ORDER BY nzgd_id")] == [1]
        assert out.execute("SELECT nzgd_id, record_type FROM quality_reject_record").fetchall() == [(2, "CPT")]
    finally:
        out.close()
    assert (tmp_path / "quality_reject_record_report.csv").exists()
