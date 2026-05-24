"""DDL for dedup-specific schema additions on the target (deduped) DB."""

import sqlite3


_ADD_MERGED_INTO_COLUMN = (
    "ALTER TABLE nzgdrecord "
    "ADD COLUMN merged_into_nzgd_id INTEGER REFERENCES nzgdrecord(nzgd_id)"
)

_INDEX_MERGED_INTO = (
    "CREATE INDEX IF NOT EXISTS idx_nzgdrecord_merged_into "
    "ON nzgdrecord(merged_into_nzgd_id)"
)

_CREATE_DEDUP_RUN = """
CREATE TABLE IF NOT EXISTS dedup_run (
    run_id               INTEGER PRIMARY KEY,
    started_at           TEXT NOT NULL,
    finished_at          TEXT,
    source_db_path       TEXT NOT NULL,
    script_version       TEXT NOT NULL,
    config_snapshot_json TEXT NOT NULL,
    n_clusters_merged    INTEGER,
    n_records_merged     INTEGER
)
"""

_CREATE_DEDUP_AUDIT = """
CREATE TABLE IF NOT EXISTS dedup_audit (
    audit_id                INTEGER PRIMARY KEY,
    run_id                  INTEGER NOT NULL REFERENCES dedup_run(run_id),
    cluster_id              INTEGER NOT NULL,
    canonical_nzgd_id       INTEGER NOT NULL,
    merged_nzgd_id          INTEGER NOT NULL,
    record_type             TEXT NOT NULL CHECK(record_type IN ('CPT', 'BH')),
    match_pass              TEXT NOT NULL CHECK(match_pass IN ('hash', 'fuzzy')),
    report_pairs_json       TEXT NOT NULL,
    metadata_copied_json    TEXT,
    metadata_conflicts_json TEXT,
    merged_at               TEXT NOT NULL
)
"""

_INDEX_AUDIT_CANONICAL = "CREATE INDEX IF NOT EXISTS idx_dedup_audit_canonical ON dedup_audit(canonical_nzgd_id)"
_INDEX_AUDIT_MERGED    = "CREATE INDEX IF NOT EXISTS idx_dedup_audit_merged    ON dedup_audit(merged_nzgd_id)"
_INDEX_AUDIT_CLUSTER   = "CREATE INDEX IF NOT EXISTS idx_dedup_audit_cluster   ON dedup_audit(cluster_id)"

_DROP_CPT_DATA_DUPLICATE_COLUMN = (
    "ALTER TABLE cptreport DROP COLUMN cpt_data_duplicate_of_cpt_id"
)

_WIDENED_DEDUP_AUDIT_DDL = """
CREATE TABLE dedup_audit_new (
    audit_id                INTEGER PRIMARY KEY,
    run_id                  INTEGER NOT NULL REFERENCES dedup_run(run_id),
    cluster_id              INTEGER NOT NULL,
    canonical_nzgd_id       INTEGER NOT NULL,
    merged_nzgd_id          INTEGER NOT NULL,
    record_type             TEXT NOT NULL CHECK(record_type IN ('CPT', 'BH')),
    match_pass              TEXT NOT NULL CHECK(match_pass IN ('hash', 'fuzzy', 'within_record')),
    report_pairs_json       TEXT NOT NULL,
    metadata_copied_json    TEXT,
    metadata_conflicts_json TEXT,
    merged_at               TEXT NOT NULL
)
"""


def _audit_check_includes_within_record(conn: sqlite3.Connection) -> bool:
    """True if the current `dedup_audit.match_pass` CHECK already accepts 'within_record'."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='dedup_audit'"
    ).fetchone()
    if row is None:
        return False
    return "'within_record'" in row[0]


def _cptreport_has_duplicate_column(conn: sqlite3.Connection) -> bool:
    """True if cptreport currently has the legacy cpt_data_duplicate_of_cpt_id column."""
    cur = conn.execute("PRAGMA table_info(cptreport)")
    return any(r[1] == "cpt_data_duplicate_of_cpt_id" for r in cur.fetchall())


def _migrate_widen_audit_check(conn: sqlite3.Connection) -> None:
    """Recreate dedup_audit with the widened CHECK constraint. Idempotent via caller's check."""
    cur = conn.cursor()
    cur.execute(_WIDENED_DEDUP_AUDIT_DDL)
    cur.execute("INSERT INTO dedup_audit_new SELECT * FROM dedup_audit")
    cur.execute("DROP TABLE dedup_audit")
    cur.execute("ALTER TABLE dedup_audit_new RENAME TO dedup_audit")
    cur.execute(_INDEX_AUDIT_CANONICAL)
    cur.execute(_INDEX_AUDIT_MERGED)
    cur.execute(_INDEX_AUDIT_CLUSTER)


def apply_dedup_schema(conn: sqlite3.Connection) -> None:
    """Apply dedup-specific schema additions to a deduped target DB.

    Adds `nzgdrecord.merged_into_nzgd_id`, creates `dedup_run` and
    `dedup_audit` tables, and creates supporting indexes. Idempotent: if a
    second invocation runs against an already-migrated DB, the ALTER TABLE
    will fail with "duplicate column"; the function catches that case and
    proceeds. All other DDL is `IF NOT EXISTS`.

    Foreign-key enforcement requires `PRAGMA foreign_keys = ON` on the
    connection; callers should issue that before invoking this function.
    """
    cur = conn.cursor()
    try:
        cur.execute(_ADD_MERGED_INTO_COLUMN)
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise
    cur.execute(_INDEX_MERGED_INTO)
    cur.execute(_CREATE_DEDUP_RUN)
    cur.execute(_CREATE_DEDUP_AUDIT)
    cur.execute(_INDEX_AUDIT_CANONICAL)
    cur.execute(_INDEX_AUDIT_MERGED)
    cur.execute(_INDEX_AUDIT_CLUSTER)

    # Migrate: widen dedup_audit.match_pass CHECK constraint if not already done
    if not _audit_check_includes_within_record(conn):
        _migrate_widen_audit_check(conn)

    # Migrate: drop legacy cpt_data_duplicate_of_cpt_id column if still present
    if _cptreport_has_duplicate_column(conn):
        try:
            cur.execute(_DROP_CPT_DATA_DUPLICATE_COLUMN)
        except sqlite3.OperationalError as e:
            if "no such column" not in str(e).lower():
                raise
    conn.commit()
