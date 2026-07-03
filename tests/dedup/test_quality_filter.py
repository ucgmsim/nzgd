"""Integration tests for the constant-column quality filter."""

import sqlite3

import pytest

from nzgd.dedup.data_types import CPT_TABLE_CONFIG
from nzgd.dedup.quality_filter import find_constant_column_reports
from tests.dedup.conftest import add_cpt_record, add_cpt_report

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
