"""Canonical selection for a dedup cluster.

Picks the canonical nzgd_id per the spec rule:
  1. Highest count of measurement rows in reports with no matched-pair counterpart.
  2. Tiebreaker: most non-null nzgdrecord columns.
  3. Tiebreaker: smallest nzgd_id.
"""

import sqlite3
from collections.abc import Iterable

from nzgd.dedup.data_types import TableConfig

_NZGDRECORD_METADATA_COLUMNS = (
    "type_id", "latitude", "longitude",
    "model_vs30_foster_2019_m_per_s", "model_vs30_stddev_foster_2019_ln",
    "model_gwl_westerhoff_2018_m", "model_gwl_nlm_2025_m", "model_gwl_nlm_2025_stddev_m",
    "original_investigation_name", "record_created_on", "record_last_modified_on",
    "region_id", "district_id", "city_id", "suburb_id",
)


def _matched_report_ids_for_nzgd(
    nzgd_id: int, matched_pairs: Iterable[tuple[int, int, int]]
) -> set[int]:
    """Return set of report ids in this nzgd_id that appear in any matched pair.

    matched_pairs is iterable of (nzgd_id_a, report_id_a, nzgd_id_b_or_report_id_b...)
    For our use, a pair is represented as (nzgd_a, report_a, nzgd_b, report_b).
    """
    out: set[int] = set()
    for nzgd_a, report_a, nzgd_b, report_b in matched_pairs:
        if nzgd_a == nzgd_id:
            out.add(report_a)
        if nzgd_b == nzgd_id:
            out.add(report_b)
    return out


def _unique_measurement_row_count(
    conn: sqlite3.Connection,
    nzgd_id: int,
    matched_report_ids: set[int],
    table_cfg: TableConfig,
) -> int:
    """Count measurement rows for `nzgd_id` in reports NOT in `matched_report_ids`."""
    cur = conn.cursor()
    if matched_report_ids:
        placeholders = ",".join("?" * len(matched_report_ids))
        query = (
            f"SELECT COUNT(*) FROM {table_cfg.measurement_table} m "
            f"JOIN {table_cfg.report_table} r ON r.{table_cfg.report_id_column} = m.{table_cfg.report_id_column} "
            f"WHERE r.nzgd_id = ? "
            f"AND r.{table_cfg.report_id_column} NOT IN ({placeholders})"
        )
        cur.execute(query, (nzgd_id, *matched_report_ids))
    else:
        query = (
            f"SELECT COUNT(*) FROM {table_cfg.measurement_table} m "
            f"JOIN {table_cfg.report_table} r ON r.{table_cfg.report_id_column} = m.{table_cfg.report_id_column} "
            f"WHERE r.nzgd_id = ?"
        )
        cur.execute(query, (nzgd_id,))
    return cur.fetchone()[0]


def _non_null_metadata_count(conn: sqlite3.Connection, nzgd_id: int) -> int:
    cur = conn.cursor()
    cols_sql = ", ".join(_NZGDRECORD_METADATA_COLUMNS)
    cur.execute(f"SELECT {cols_sql} FROM nzgdrecord WHERE nzgd_id = ?", (nzgd_id,))
    row = cur.fetchone()
    if row is None:
        return 0
    return sum(1 for v in row if v is not None)


def select_canonical(
    conn: sqlite3.Connection,
    cluster_nzgd_ids: Iterable[int],
    matched_pairs: Iterable[tuple[int, int, int, int]],
    table_cfg: TableConfig,
    completeness: dict[int, float] | None = None,
) -> int:
    """Pick the canonical nzgd_id from a cluster of nzgd_ids per the spec rule.

    Parameters
    ----------
    conn
        Open SQLite connection to the target DB.
    cluster_nzgd_ids
        nzgd_ids in the cluster.
    matched_pairs
        Iterable of (nzgd_id_a, report_id_a, nzgd_id_b, report_id_b) — the matched
        report pairs across nzgd_ids in this cluster (as identified by the pass).
    table_cfg
        Per-record-type table configuration.
    completeness
        Optional `{nzgd_id: depth-coverage}` map. When provided, coverage is the
        primary sort key so the most-complete trace survives; ties fall through
        to the original rule (most unique measurement rows, then most non-null
        metadata, then smallest nzgd_id). When None, the ranking is unchanged.

    Returns
    -------
    int
        The selected canonical nzgd_id.
    """
    pairs = list(matched_pairs)
    nzgd_ids = list(cluster_nzgd_ids)
    scored = []
    for nz in nzgd_ids:
        matched_ids = _matched_report_ids_for_nzgd(nz, pairs)
        unique_rows = _unique_measurement_row_count(conn, nz, matched_ids, table_cfg)
        meta_count = _non_null_metadata_count(conn, nz)
        cov = completeness.get(nz, 0.0) if completeness is not None else 0.0
        # Sort key: maximise coverage, then unique_rows, then meta_count; minimise nzgd_id
        scored.append((-cov, -unique_rows, -meta_count, nz))
    scored.sort()
    return scored[0][3]
