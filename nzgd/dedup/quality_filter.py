"""Constant-column data-quality filter: discard CPT reports with a flat channel.

A physically real CPT varies with depth in every channel. A measurement column
that holds a single repeated value is a broken extraction or an unmeasured-channel
placeholder, so the whole report is discarded. This runs before the dedup passes
so a flat report cannot be chosen as canonical or pollute fuzzy matching.
"""

import sqlite3

from nzgd.dedup.data_types import QualityRejectEntry, TableConfig


def find_constant_column_reports(
    conn: sqlite3.Connection,
    table_cfg: TableConfig,
    columns: list[str],
    min_non_null_rows: int,
) -> list[QualityRejectEntry]:
    """Return reports that have a constant value in at least one of `columns`.

    A column is constant when it has ``>= min_non_null_rows`` non-null values that
    are all equal (``COUNT(DISTINCT col) = 1``). An all-NULL column is never
    constant. The DB is not modified.

    Parameters
    ----------
    conn
        Connection to the (deduped-target) DB.
    table_cfg
        Table configuration for the record type (only CPT is used in practice).
    columns
        Measurement columns to test; each must be in
        ``table_cfg.measurement_value_columns``.
    min_non_null_rows
        Minimum non-null values a column must have before it can be judged
        constant.

    Returns
    -------
    list[QualityRejectEntry]
        One entry per offending report; ``constant_columns`` maps each offending
        column to its constant value.
    """
    valid = set(table_cfg.measurement_value_columns)
    invalid = [c for c in columns if c not in valid]
    if invalid:
        raise ValueError(
            f"quality_filter columns {invalid} are not measurement columns of "
            f"{table_cfg.record_type}: {sorted(valid)}"
        )
    if not columns:
        return []

    report_id_col = table_cfg.report_id_column
    select_terms = [f"m.{report_id_col}", "r.nzgd_id", "COUNT(*) AS n_rows"]
    for i, col in enumerate(columns):
        select_terms.append(f"COUNT({col}) AS nn{i}")
        select_terms.append(f"COUNT(DISTINCT {col}) AS d{i}")
        select_terms.append(f"MIN({col}) AS v{i}")
    having_terms = [
        f"(COUNT({col}) >= ? AND COUNT(DISTINCT {col}) = 1)" for col in columns
    ]
    sql = (
        f"SELECT {', '.join(select_terms)} "
        f"FROM {table_cfg.measurement_table} m "
        f"JOIN {table_cfg.report_table} r ON r.{report_id_col} = m.{report_id_col} "
        f"GROUP BY m.{report_id_col}, r.nzgd_id "
        f"HAVING {' OR '.join(having_terms)}"
    )

    entries: list[QualityRejectEntry] = []
    for row in conn.execute(sql, [min_non_null_rows] * len(columns)).fetchall():
        report_id, nzgd_id, n_rows = row[0], row[1], row[2]
        constant_columns: dict[str, float] = {}
        for i, col in enumerate(columns):
            nn, distinct, value = row[3 + i * 3], row[4 + i * 3], row[5 + i * 3]
            if nn >= min_non_null_rows and distinct == 1:
                constant_columns[col] = value
        entries.append(
            QualityRejectEntry(
                record_type=table_cfg.record_type,
                nzgd_id=nzgd_id,
                report_id=report_id,
                reason="constant_column",
                constant_columns=constant_columns,
                n_rows=n_rows,
            )
        )
    return entries
