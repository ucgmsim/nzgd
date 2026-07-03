"""CSV writers for dedup script outputs."""

import csv
import json
import sqlite3
from pathlib import Path


def write_dedup_report(conn: sqlite3.Connection, run_id: int, path: Path) -> None:
    """Flatten dedup_audit rows for a given run into a CSV."""
    cur = conn.cursor()
    cur.execute(
        "SELECT cluster_id, canonical_nzgd_id, merged_nzgd_id, record_type, "
        "match_pass, report_pairs_json, metadata_copied_json, metadata_conflicts_json, merged_at "
        "FROM dedup_audit WHERE run_id = ? "
        "ORDER BY record_type, cluster_id, merged_nzgd_id",
        (run_id,),
    )
    rows = cur.fetchall()
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster_id", "canonical_nzgd_id", "merged_nzgd_id", "record_type",
            "match_pass", "matched_pair_count", "metadata_copied_fields",
            "metadata_conflict_fields", "merged_at",
        ])
        for r in rows:
            (cluster, canon, merged, rtype, mpass, pairs_json,
             copied_json, conflicts_json, merged_at) = r
            pair_count = len(json.loads(pairs_json))
            copied_fields = ",".join(json.loads(copied_json).keys()) if copied_json else ""
            conflict_fields = ",".join(json.loads(conflicts_json).keys()) if conflicts_json else ""
            writer.writerow([cluster, canon, merged, rtype, mpass, pair_count,
                             copied_fields, conflict_fields, merged_at])


def write_calibration_report(
    positive_rows: list[dict],
    negative_rows: list[dict],
    path: Path,
) -> None:
    """Write feature distributions for positive and negative fuzzy-pass examples."""
    fieldnames = ["group", "spatial_m", "date_days", "name_sim",
                  "max_depth_diff_m", "trace_score"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in positive_rows:
            writer.writerow({"group": "positive", **{k: r.get(k) for k in fieldnames if k != "group"}})
        for r in negative_rows:
            writer.writerow({"group": "negative", **{k: r.get(k) for k in fieldnames if k != "group"}})


def write_failures_report(failures: list[dict], path: Path) -> None:
    """Write rolled-back-cluster failures to CSV with full JSON detail."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "canonical_nzgd_id", "merged_nzgd_ids",
                         "record_type", "error"])
        for fail in failures:
            writer.writerow([
                fail.get("cluster_id"),
                fail.get("canonical_nzgd_id"),
                json.dumps(fail.get("merged_nzgd_ids")),
                fail.get("record_type"),
                fail.get("error"),
            ])


def write_supplemental_consolidation_report(
    conn: sqlite3.Connection, run_id: int, path: Path
) -> None:
    """Flatten supplemental_consolidation audit rows for a run into a CSV."""
    cur = conn.cursor()
    cur.execute(
        "SELECT canonical_nzgd_id, metadata_copied_json, metadata_conflicts_json, merged_at "
        "FROM dedup_audit WHERE run_id = ? AND match_pass = 'supplemental_consolidation' "
        "ORDER BY canonical_nzgd_id",
        (run_id,),
    )
    rows = cur.fetchall()
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nzgd_id", "filled_fields", "conflict_fields", "conflict_detail", "consolidated_at"])
        for nzgd_id, copied_json, conflicts_json, merged_at in rows:
            filled = ",".join(json.loads(copied_json).keys()) if copied_json else ""
            conflicts = json.loads(conflicts_json) if conflicts_json else {}
            writer.writerow([
                nzgd_id, filled, ",".join(conflicts.keys()),
                json.dumps(conflicts) if conflicts else "", merged_at,
            ])


def write_quality_filter_report(conn: sqlite3.Connection, run_id: int, path: Path) -> None:
    """Flatten quality_reject rows for a given run into a CSV."""
    cur = conn.cursor()
    cur.execute(
        "SELECT record_type, nzgd_id, report_id, reason, constant_columns_json, n_rows, rejected_at "
        "FROM quality_reject WHERE run_id = ? "
        "ORDER BY record_type, nzgd_id, report_id",
        (run_id,),
    )
    rows = cur.fetchall()
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "record_type", "nzgd_id", "report_id", "reason",
            "constant_columns", "n_rows", "rejected_at",
        ])
        for record_type, nzgd_id, report_id, reason, cc_json, n_rows, rejected_at in rows:
            cols = ",".join(json.loads(cc_json).keys()) if cc_json else ""
            writer.writerow([record_type, nzgd_id, report_id, reason, cols, n_rows, rejected_at])


def write_quality_reject_record_report(conn: sqlite3.Connection, run_id: int, path: Path) -> None:
    """Flatten quality_reject_record rows for a given run into a CSV."""
    cur = conn.cursor()
    cur.execute(
        "SELECT record_type, nzgd_id, reason, n_reports_discarded, deleted_at "
        "FROM quality_reject_record WHERE run_id = ? "
        "ORDER BY record_type, nzgd_id",
        (run_id,),
    )
    rows = cur.fetchall()
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record_type", "nzgd_id", "reason", "n_reports_discarded", "deleted_at"])
        writer.writerows(rows)
