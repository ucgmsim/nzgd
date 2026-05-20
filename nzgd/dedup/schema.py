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
    conn.commit()
