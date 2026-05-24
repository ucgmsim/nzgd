"""Pass 0: within-record consolidation of cptreport/sptreport rows.

For each nzgd_id, cluster the report rows by source-file stem + trace identity,
then collapse each cluster to a single canonical row with metadata merged from
absorbed rows via the plausibility-aware enrichment rule.
"""

import importlib
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from typing import Iterable

import numpy as np
from tqdm import tqdm

from nzgd import constants
from nzgd.dedup import executor
from nzgd.dedup.canonical_selectors import (
    CanonicalSelector,
    ClusterRow,
    default_within_record_canonical,
)
from nzgd.dedup.cluster import connected_components_from_edges
from nzgd.dedup.data_types import TableConfig
from nzgd.dedup.fingerprint import compute_trace_hash
from nzgd.dedup.plausibility import is_useful_value
from nzgd.dedup.trace_compare import best_trace_score, load_traces


_CPTREPORT_METADATA_COLUMNS = (
    "max_depth_m", "min_depth_m", "extracted_gwl_m", "gwl_method_id",
    "tip_net_area_ratio", "predrill_depth_m", "termination_reason_id",
    "did_explicit_unit_conversion", "did_inferred_unit_conversion",
    "source_file",
)

_SPTREPORT_METADATA_COLUMNS = (
    "efficiency", "extracted_gwl_m", "borehole_diameter", "casing_diameter",
    "source_file",
)


def _metadata_columns_for(table_cfg: TableConfig) -> tuple[str, ...]:
    return _CPTREPORT_METADATA_COLUMNS if table_cfg.record_type == "CPT" else _SPTREPORT_METADATA_COLUMNS


@dataclass(frozen=True)
class _AbsorbedReport:
    """One absorbed row's contribution within a cluster (for the audit ledger)."""

    absorbed_report_id: int
    absorbed_source_file: str
    trace_match: str  # 'hash', 'fuzzy', or 'stem_only'


@dataclass(frozen=True)
class WithinRecordConsolidation:
    """One within-record cluster consolidation action."""

    cluster_id: int
    nzgd_id: int
    canonical_report_id: int
    absorbed_reports: list[_AbsorbedReport]
    record_type: str  # 'CPT' or 'BH'


def _resolve_default_selector() -> CanonicalSelector:
    """Resolve the default canonical-selector callable from config (dotted path)."""
    cfg = constants.DEDUP_CONFIG.get("within_record", {})
    dotted = cfg.get(
        "canonical_selector",
        "nzgd.dedup.canonical_selectors.default_within_record_canonical",
    )
    module_path, _, func_name = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _get_stem(source_file: str | None) -> str:
    """Return the source-file stem (substring before '_sheet_')."""
    if not source_file:
        return ""
    idx = source_file.find("_sheet_")
    if idx == -1:
        return source_file
    return source_file[:idx]


def _trace_hash_for(traces: dict[int, np.ndarray], report_id: int) -> bytes | None:
    """Compute a fingerprint of a loaded trace ndarray. Returns None if no rows.

    Converts ndarray rows to Python float tuples so the cross-record-grade
    `compute_trace_hash` (which has the NaN/None-sentinel + string-handling
    fixes) is the single source of truth for hash equality across passes.
    """
    arr = traces.get(report_id)
    if arr is None or arr.shape[0] == 0:
        return None
    rows = [tuple(float(v) for v in row) for row in arr]
    return compute_trace_hash(rows)


def _cluster_within_stem(
    stem_report_ids: list[int],
    has_data_by_id: dict[int, bool],
    traces: dict[int, np.ndarray],
    trace_score_max: float,
    trace_resample_step_m: float,
) -> list[list[int]]:
    """Split a single stem's rows into sub-clusters when data-bearing rows don't match.

    Data-bearing rows whose hashes match (or fuzzy trace_score below threshold)
    join the same sub-cluster. Non-matching data-bearing rows form separate
    sub-clusters. No-data rows attach to the sub-cluster with the smallest
    cpt_id (deterministic heuristic).
    """
    data_bearing = sorted(rid for rid in stem_report_ids if has_data_by_id.get(rid))
    no_data = sorted(rid for rid in stem_report_ids if not has_data_by_id.get(rid))

    if len(data_bearing) <= 1:
        # No conflict possible — whole stem is one sub-cluster
        return [sorted(stem_report_ids)]

    # Cluster data-bearing rows by trace identity (hash first, fuzzy fallback)
    hashes = {rid: _trace_hash_for(traces, rid) for rid in data_bearing}
    parent = {rid: rid for rid in data_bearing}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a, b in combinations(data_bearing, 2):
        ha, hb = hashes[a], hashes[b]
        if ha is not None and ha == hb:
            union(a, b)
            continue
        ta, tb = traces.get(a), traces.get(b)
        if ta is None or tb is None:
            continue
        score, _ = best_trace_score({a: ta}, {b: tb}, trace_resample_step_m)
        if not math.isnan(score) and score < trace_score_max:
            union(a, b)

    sub_clusters: dict[int, list[int]] = defaultdict(list)
    for rid in data_bearing:
        sub_clusters[find(rid)].append(rid)

    # If only one sub-cluster, no split needed
    if len(sub_clusters) == 1:
        return [sorted(stem_report_ids)]

    # Multiple sub-clusters: attach no-data rows to the smallest-cpt_id sub-cluster
    sub_cluster_list = sorted(sub_clusters.values(), key=lambda lst: min(lst))
    sub_cluster_list[0].extend(no_data)
    return [sorted(c) for c in sub_cluster_list]


def _build_clusters_for_nzgd(
    conn: sqlite3.Connection,
    nzgd_id: int,
    table_cfg: TableConfig,
    thresholds: dict,
) -> tuple[list[list[int]], dict[int, str], dict[int, int]]:
    """Return (clusters, source_file_by_id, measurement_count_by_id) for an nzgd_id."""
    cur = conn.cursor()
    cur.execute(
        f"SELECT r.{table_cfg.report_id_column}, r.source_file, "
        f"  (SELECT COUNT(*) FROM {table_cfg.measurement_table} m "
        f"   WHERE m.{table_cfg.report_id_column} = r.{table_cfg.report_id_column}) "
        f"FROM {table_cfg.report_table} r WHERE r.nzgd_id = ? "
        f"ORDER BY r.{table_cfg.report_id_column}",
        (nzgd_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return [], {}, {}

    source_file_by_id = {r[0]: r[1] for r in rows}
    measurement_count_by_id = {r[0]: r[2] for r in rows}
    has_data_by_id = {r[0]: r[2] > 0 for r in rows}

    # Group by stem
    by_stem: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_stem[_get_stem(r[1])].append(r[0])

    # Load all traces for this nzgd_id once
    traces = load_traces(conn, nzgd_id, table_cfg)

    # Within-stem clustering (split when stems contain non-matching data-bearing rows)
    stem_nodes: list[list[int]] = []
    for stem_ids in by_stem.values():
        sub_clusters = _cluster_within_stem(
            stem_ids, has_data_by_id, traces,
            thresholds["trace_score_max"], thresholds["trace_resample_step_m"],
        )
        stem_nodes.extend(sub_clusters)

    # Cross-stem edges: any data-bearing row in node A matches any data-bearing row in node B
    edges: list[tuple[int, int]] = []
    for i, j in combinations(range(len(stem_nodes)), 2):
        data_a = [r for r in stem_nodes[i] if has_data_by_id.get(r)]
        data_b = [r for r in stem_nodes[j] if has_data_by_id.get(r)]
        if not data_a or not data_b:
            continue
        hashes_a = {r: _trace_hash_for(traces, r) for r in data_a}
        hashes_b = {r: _trace_hash_for(traces, r) for r in data_b}
        linked = False
        for ra in data_a:
            for rb in data_b:
                if hashes_a[ra] is not None and hashes_a[ra] == hashes_b[rb]:
                    linked = True
                    break
                ta, tb = traces.get(ra), traces.get(rb)
                if ta is None or tb is None:
                    continue
                score, _ = best_trace_score({ra: ta}, {rb: tb}, thresholds["trace_resample_step_m"])
                if not math.isnan(score) and score < thresholds["trace_score_max"]:
                    linked = True
                    break
            if linked:
                break
        if linked:
            edges.append((i, j))

    # Connected components on node indices
    if edges:
        node_to_component = connected_components_from_edges(edges)
    else:
        node_to_component = {}

    components: dict[int, list[int]] = defaultdict(list)
    for idx, node in enumerate(stem_nodes):
        comp = node_to_component.get(idx, idx)  # singletons keyed by their own index
        components[comp].extend(node)

    return [sorted(rep_ids) for rep_ids in components.values()], source_file_by_id, measurement_count_by_id


def _classify_match(
    canonical_id: int,
    absorbed_id: int,
    has_data_by_id: dict[int, bool],
    traces: dict[int, np.ndarray],
    trace_score_max: float,
    trace_resample_step_m: float,
) -> str:
    """Return 'hash', 'fuzzy', or 'stem_only' for an absorbed row's relationship to canonical."""
    if not has_data_by_id.get(canonical_id) or not has_data_by_id.get(absorbed_id):
        return "stem_only"
    ha = _trace_hash_for(traces, canonical_id)
    hb = _trace_hash_for(traces, absorbed_id)
    if ha is not None and ha == hb:
        return "hash"
    ta, tb = traces.get(canonical_id), traces.get(absorbed_id)
    if ta is None or tb is None:
        return "stem_only"
    score, _ = best_trace_score({canonical_id: ta}, {absorbed_id: tb}, trace_resample_step_m)
    if not math.isnan(score) and score < trace_score_max:
        return "fuzzy"
    return "stem_only"


def _metadata_non_null_count(conn: sqlite3.Connection, report_id: int, table_cfg: TableConfig) -> int:
    cols = _metadata_columns_for(table_cfg)
    cur = conn.execute(
        f"SELECT {', '.join(cols)} FROM {table_cfg.report_table} "
        f"WHERE {table_cfg.report_id_column} = ?",
        (report_id,),
    )
    row = cur.fetchone()
    if row is None:
        return 0
    return sum(1 for v in row if v is not None)


def generate_within_record_consolidation_plan(
    conn: sqlite3.Connection,
    table_cfg: TableConfig,
    thresholds: dict,
    canonical_selector: CanonicalSelector | None = None,
) -> list[WithinRecordConsolidation]:
    """Build a list of consolidation actions, one per cluster needing collapse."""
    if canonical_selector is None:
        canonical_selector = _resolve_default_selector()

    cur = conn.cursor()
    type_id = 1 if table_cfg.record_type == "CPT" else 2
    cur.execute(
        f"SELECT DISTINCT n.nzgd_id FROM nzgdrecord n "
        f"JOIN {table_cfg.report_table} r ON r.nzgd_id = n.nzgd_id "
        f"WHERE n.merged_into_nzgd_id IS NULL AND n.type_id = ? "
        f"ORDER BY n.nzgd_id",
        (type_id,),
    )
    nzgd_ids = [r[0] for r in cur.fetchall()]

    plans: list[WithinRecordConsolidation] = []
    next_cluster_id = 1
    for nzgd_id in tqdm(nzgd_ids, desc=f"within-record {table_cfg.record_type}"):
        clusters, source_file_by_id, measurement_count_by_id = _build_clusters_for_nzgd(
            conn, nzgd_id, table_cfg, thresholds,
        )
        for cluster_report_ids in clusters:
            if len(cluster_report_ids) <= 1:
                continue
            cluster_rows = [
                ClusterRow(
                    report_id=rid,
                    has_data=measurement_count_by_id[rid] > 0,
                    measurement_row_count=measurement_count_by_id[rid],
                    metadata_non_null_count=_metadata_non_null_count(conn, rid, table_cfg),
                )
                for rid in cluster_report_ids
            ]
            canonical_id = canonical_selector(cluster_rows, table_cfg)
            traces = load_traces(conn, nzgd_id, table_cfg)
            has_data_by_id = {rid: measurement_count_by_id[rid] > 0 for rid in cluster_report_ids}
            absorbed = []
            for rid in cluster_report_ids:
                if rid == canonical_id:
                    continue
                absorbed.append(_AbsorbedReport(
                    absorbed_report_id=rid,
                    absorbed_source_file=source_file_by_id.get(rid) or "",
                    trace_match=_classify_match(
                        canonical_id, rid, has_data_by_id, traces,
                        thresholds["trace_score_max"], thresholds["trace_resample_step_m"],
                    ),
                ))
            plans.append(WithinRecordConsolidation(
                cluster_id=next_cluster_id,
                nzgd_id=nzgd_id,
                canonical_report_id=canonical_id,
                absorbed_reports=absorbed,
                record_type=table_cfg.record_type,
            ))
            next_cluster_id += 1

    return plans


def apply_within_record_consolidation_plan(
    conn: sqlite3.Connection,
    plan: Iterable[WithinRecordConsolidation],
    run_id: int,
    table_cfg: TableConfig,
    failures: list[dict] | None = None,
) -> tuple[int, int]:
    """Apply within-record consolidation. Returns (n_clusters, n_records_absorbed)."""
    plan_list = list(plan)
    if not plan_list:
        return 0, 0

    metadata_cols = _metadata_columns_for(table_cfg)
    n_clusters_ok = 0
    n_records_absorbed = 0
    cur = conn.cursor()

    for consolidation in tqdm(plan_list, desc=f"consolidating {table_cfg.record_type}"):
        savepoint = f"within_cluster_{consolidation.cluster_id}"
        cur.execute(f"SAVEPOINT {savepoint}")
        try:
            # Read all rows in the cluster
            all_ids = [consolidation.canonical_report_id] + [
                a.absorbed_report_id for a in consolidation.absorbed_reports
            ]
            placeholders = ",".join("?" * len(all_ids))
            cur.execute(
                f"SELECT {table_cfg.report_id_column}, {', '.join(metadata_cols)} "
                f"FROM {table_cfg.report_table} "
                f"WHERE {table_cfg.report_id_column} IN ({placeholders})",
                all_ids,
            )
            rows_by_id: dict[int, dict] = {}
            for row in cur.fetchall():
                rid = row[0]
                rows_by_id[rid] = dict(zip(metadata_cols, row[1:]))

            canon_vals = rows_by_id[consolidation.canonical_report_id]

            # Compute enrichment using the plausibility helper
            updates: dict[str, object] = {}
            copied: dict[str, dict] = {}
            conflicts: dict[str, list[dict]] = {}
            for col in metadata_cols:
                if col == "source_file":
                    # Always keep canonical's source_file
                    continue
                if is_useful_value(canon_vals[col], table_cfg.report_table, col):
                    continue
                useful_candidates = []
                for absorbed in consolidation.absorbed_reports:
                    v = rows_by_id[absorbed.absorbed_report_id][col]
                    if is_useful_value(v, table_cfg.report_table, col):
                        useful_candidates.append((v, absorbed.absorbed_report_id))
                if not useful_candidates:
                    continue
                distinct = {v for v, _ in useful_candidates}
                if len(distinct) == 1:
                    chosen = useful_candidates[0]
                    updates[col] = chosen[0]
                    copied[col] = {"value": chosen[0], "source_report_id": chosen[1]}
                else:
                    chosen = min(useful_candidates, key=lambda c: c[1])
                    updates[col] = chosen[0]
                    copied[col] = {"value": chosen[0], "source_report_id": chosen[1]}
                    conflicts[col] = [
                        {"value": v, "source_report_id": rid} for v, rid in useful_candidates
                    ]

            # UPDATE canonical with enriched metadata (if any)
            if updates:
                set_clause = ", ".join(f"{c} = ?" for c in updates)
                cur.execute(
                    f"UPDATE {table_cfg.report_table} SET {set_clause} "
                    f"WHERE {table_cfg.report_id_column} = ?",
                    (*updates.values(), consolidation.canonical_report_id),
                )

            # Delete each absorbed row + its dependents via the cross-record executor helper
            for absorbed in consolidation.absorbed_reports:
                executor.delete_report(conn, absorbed.absorbed_report_id, table_cfg)
                n_records_absorbed += 1

            # Audit row
            pairs_json = json.dumps([
                {
                    "canonical_report_id": consolidation.canonical_report_id,
                    "absorbed_report_id": a.absorbed_report_id,
                    "absorbed_source_file": a.absorbed_source_file,
                    "trace_match": a.trace_match,
                }
                for a in consolidation.absorbed_reports
            ])
            cur.execute(
                "INSERT INTO dedup_audit (run_id, cluster_id, canonical_nzgd_id, merged_nzgd_id, "
                "record_type, match_pass, report_pairs_json, metadata_copied_json, "
                "metadata_conflicts_json, merged_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    consolidation.cluster_id,
                    consolidation.nzgd_id,
                    consolidation.nzgd_id,  # within-record: canonical == merged nzgd_id
                    consolidation.record_type,
                    "within_record",
                    pairs_json,
                    json.dumps(copied) if copied else None,
                    json.dumps(conflicts) if conflicts else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            n_clusters_ok += 1
        except Exception as exc:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            if failures is not None:
                failures.append({
                    "cluster_id": consolidation.cluster_id,
                    "canonical_nzgd_id": consolidation.nzgd_id,
                    "merged_nzgd_ids": [consolidation.nzgd_id],
                    "record_type": table_cfg.record_type,
                    "error": repr(exc),
                })
    conn.commit()
    return n_clusters_ok, n_records_absorbed
