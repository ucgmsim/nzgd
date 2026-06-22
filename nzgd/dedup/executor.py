"""Apply a list of MergePlanEntry to a deduped target DB."""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterable

from tqdm import tqdm

from nzgd.dedup.data_types import MergePlanEntry, TableConfig
from nzgd.dedup.plausibility import is_useful_value


_NZGDRECORD_METADATA_COLUMNS = (
    "type_id", "latitude", "longitude",
    "model_vs30_foster_2019_m_per_s", "model_vs30_stddev_foster_2019_ln",
    "model_gwl_westerhoff_2018_m", "model_gwl_nlm_2025_m", "model_gwl_nlm_2025_stddev_m",
    "original_investigation_name", "record_created_on", "record_last_modified_on",
    "region_id", "district_id", "city_id", "suburb_id",
)


def _enrich_canonical_metadata(
    conn: sqlite3.Connection, canonical_nzgd_id: int, merged_nzgd_ids: list[int]
) -> tuple[dict, dict]:
    """Merge non-null metadata from merged records into canonical.

    Returns `(copied, conflicts)`. `copied` is `{column: {"value": v, "source_nzgd_id": id}}`.
    `conflicts` is `{column: [{"value": v, "source_nzgd_id": id}, ...]}`.
    """
    cur = conn.cursor()
    cols_sql = ", ".join(_NZGDRECORD_METADATA_COLUMNS)
    cur.execute(f"SELECT {cols_sql} FROM nzgdrecord WHERE nzgd_id = ?", (canonical_nzgd_id,))
    canon_row = cur.fetchone()
    canon_vals = dict(zip(_NZGDRECORD_METADATA_COLUMNS, canon_row))

    # Build {column: [(value, source_nzgd_id), ...]} of non-null values from merged records
    merged_vals: dict[str, list[tuple]] = {c: [] for c in _NZGDRECORD_METADATA_COLUMNS}
    for nz in merged_nzgd_ids:
        cur.execute(f"SELECT {cols_sql} FROM nzgdrecord WHERE nzgd_id = ?", (nz,))
        row = cur.fetchone()
        if row is None:
            continue
        for col, val in zip(_NZGDRECORD_METADATA_COLUMNS, row):
            if val is not None:
                merged_vals[col].append((val, nz))

    copied: dict[str, dict] = {}
    conflicts: dict[str, list[dict]] = {}
    updates: dict[str, object] = {}
    for col in _NZGDRECORD_METADATA_COLUMNS:
        if is_useful_value(canon_vals[col], "nzgdrecord", col):
            continue  # canonical's useful value wins
        useful_candidates = [
            (v, nz) for v, nz in merged_vals[col]
            if is_useful_value(v, "nzgdrecord", col)
        ]
        if not useful_candidates:
            continue  # nothing useful to copy; leave canonical's value (NULL or sentinel) alone
        distinct = {v for v, _ in useful_candidates}
        if len(distinct) == 1:
            chosen = useful_candidates[0]
            updates[col] = chosen[0]
            copied[col] = {"value": chosen[0], "source_nzgd_id": chosen[1]}
        else:
            chosen = min(useful_candidates, key=lambda c: c[1])
            updates[col] = chosen[0]
            copied[col] = {"value": chosen[0], "source_nzgd_id": chosen[1]}
            conflicts[col] = [
                {"value": v, "source_nzgd_id": nz} for v, nz in useful_candidates
            ]

    if updates:
        set_clause = ", ".join(f"{c} = ?" for c in updates)
        cur.execute(
            f"UPDATE nzgdrecord SET {set_clause} WHERE nzgd_id = ?",
            (*updates.values(), canonical_nzgd_id),
        )
    return copied, conflicts


def delete_report(
    conn: sqlite3.Connection, report_id: int, table_cfg: TableConfig
) -> None:
    """Delete a report row and all its dependent rows in the spec-mandated order."""
    cur = conn.cursor()
    # Special-case SPT: soilmeasurementsoiltype joins soilmeasurements via soil_measurement_id.
    if table_cfg.record_type == "BH":
        cur.execute(
            "DELETE FROM soilmeasurementsoiltype "
            "WHERE soil_measurement_id IN (SELECT soil_measurement_id FROM soilmeasurements WHERE spt_id = ?)",
            (report_id,),
        )
    for tbl, fk in table_cfg.dependent_tables:
        cur.execute(f"DELETE FROM {tbl} WHERE {fk} = ?", (report_id,))
    cur.execute(
        f"DELETE FROM {table_cfg.report_table} WHERE {table_cfg.report_id_column} = ?",
        (report_id,),
    )


def _reparent_report(
    conn: sqlite3.Connection, report_id: int, new_nzgd_id: int, table_cfg: TableConfig
) -> None:
    cur = conn.cursor()
    cur.execute(
        f"UPDATE {table_cfg.report_table} SET nzgd_id = ? "
        f"WHERE {table_cfg.report_id_column} = ?",
        (new_nzgd_id, report_id),
    )
    if table_cfg.record_type == "CPT":
        cur.execute(
            "UPDATE cptvs30estimates SET nzgd_id = ? WHERE cpt_id = ?",
            (new_nzgd_id, report_id),
        )


def apply_merge_plan(
    conn: sqlite3.Connection,
    plan: Iterable[MergePlanEntry],
    run_id: int,
    table_cfg: TableConfig,
    failures: list[dict] | None = None,
) -> tuple[int, int]:
    """Apply a merge plan and write audit rows. Returns (n_clusters, n_records).

    Each cluster is applied in its own SAVEPOINT so a single bad cluster only
    rolls back itself. Failures are appended to `failures` as dicts if provided.
    """
    plan = list(plan)
    if not plan:
        return 0, 0
    # Group entries by cluster_id so we apply per-cluster transactions.
    by_cluster: dict[int, list[MergePlanEntry]] = {}
    for e in plan:
        by_cluster.setdefault(e.cluster_id, []).append(e)

    n_clusters_ok = 0
    n_records_merged_ok = 0
    cur = conn.cursor()
    for cluster_id, entries in tqdm(by_cluster.items(), desc=f"merging {table_cfg.record_type}"):
        canonical = entries[0].canonical_nzgd_id
        merged_ids = [e.merged_nzgd_id for e in entries]
        savepoint = f"cluster_{cluster_id}"
        cur.execute(f"SAVEPOINT {savepoint}")
        try:
            copied, conflicts = _enrich_canonical_metadata(conn, canonical, merged_ids)
            for entry in entries:
                for pair in entry.matched_pairs:
                    delete_report(conn, pair.merged_report_id, table_cfg)
                for rid in entry.unique_merged_report_ids:
                    _reparent_report(conn, rid, canonical, table_cfg)
                cur.execute(
                    "UPDATE nzgdrecord SET merged_into_nzgd_id = ? WHERE nzgd_id = ?",
                    (canonical, entry.merged_nzgd_id),
                )
                pairs_json = json.dumps([
                    {
                        "canonical_report_id": p.canonical_report_id,
                        "merged_report_id": p.merged_report_id,
                        "metrics": p.metrics,
                    }
                    for p in entry.matched_pairs
                ])
                cur.execute(
                    "INSERT INTO dedup_audit (run_id, cluster_id, canonical_nzgd_id, merged_nzgd_id, "
                    "record_type, match_pass, report_pairs_json, metadata_copied_json, "
                    "metadata_conflicts_json, merged_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        entry.cluster_id,
                        entry.canonical_nzgd_id,
                        entry.merged_nzgd_id,
                        entry.record_type,
                        entry.match_pass,
                        pairs_json,
                        json.dumps(copied) if copied else None,
                        json.dumps(conflicts) if conflicts else None,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                n_records_merged_ok += 1
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            n_clusters_ok += 1
        except Exception as exc:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            if failures is not None:
                failures.append({
                    "cluster_id": cluster_id,
                    "canonical_nzgd_id": canonical,
                    "merged_nzgd_ids": merged_ids,
                    "record_type": table_cfg.record_type,
                    "error": repr(exc),
                })
    conn.commit()
    return n_clusters_ok, n_records_merged_ok
