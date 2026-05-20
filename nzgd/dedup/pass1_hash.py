"""Pass 1: bit-exact match via per-trace fingerprint hashing."""

import sqlite3
from collections import defaultdict
from itertools import combinations

from tqdm import tqdm

from nzgd.dedup.cluster import connected_components_from_edges
from nzgd.dedup.data_types import (
    MergePlanEntry,
    ReportPairMatch,
    TableConfig,
)
from nzgd.dedup.fingerprint import compute_trace_hash
from nzgd.dedup.selection import select_canonical


def _compute_report_hashes(
    conn: sqlite3.Connection, table_cfg: TableConfig
) -> dict[int, bytes]:
    """Compute one fingerprint per non-merged report.

    Returns a dict `{report_id: hash}`. Only reports whose nzgd_id has
    `merged_into_nzgd_id IS NULL` are processed (so a re-run after pass 1
    doesn't re-hash already-merged data). Streams the full measurement table
    in (report_id, depth) order and filters in Python — using an `IN (?, ?, ...)`
    clause is not viable with ~144k cpt_ids (exceeds SQLite's host-parameter limit).
    """
    report_id_col = table_cfg.report_id_column
    cur = conn.cursor()
    cur.execute(
        f"SELECT r.{report_id_col} "
        f"FROM {table_cfg.report_table} r "
        f"JOIN nzgdrecord n ON n.nzgd_id = r.nzgd_id "
        f"WHERE n.merged_into_nzgd_id IS NULL"
    )
    active_report_ids: set[int] = {r[0] for r in cur.fetchall()}
    if not active_report_ids:
        return {}

    select_cols = ", ".join(table_cfg.measurement_value_columns)
    cur.execute(
        f"SELECT {report_id_col}, {select_cols} "
        f"FROM {table_cfg.measurement_table} "
        f"ORDER BY {report_id_col}, depth_m"
    )

    hashes: dict[int, bytes] = {}
    current_id: int | None = None
    current_rows: list[tuple] = []
    for row in tqdm(cur, desc=f"hashing {table_cfg.measurement_table}"):
        rid = row[0]
        if rid not in active_report_ids:
            continue
        if rid != current_id:
            if current_id is not None:
                hashes[current_id] = compute_trace_hash(current_rows)
                current_rows = []
            current_id = rid
        current_rows.append(row[1:])
    if current_id is not None:
        hashes[current_id] = compute_trace_hash(current_rows)
    return hashes


def generate_hash_merge_plan(
    conn: sqlite3.Connection, table_cfg: TableConfig
) -> list[MergePlanEntry]:
    """Produce a list of MergePlanEntry for byte-identical traces across nzgd_ids."""
    hashes = _compute_report_hashes(conn, table_cfg)
    if not hashes:
        return []

    # Build report_id → nzgd_id lookup by scanning the full report table and
    # filtering in Python (an `IN (...)` clause would exceed the host-parameter limit).
    cur = conn.cursor()
    cur.execute(
        f"SELECT {table_cfg.report_id_column}, nzgd_id FROM {table_cfg.report_table}"
    )
    report_to_nzgd = {rid: nz for rid, nz in cur.fetchall() if rid in hashes}

    # Bucket reports by hash; drop singletons; drop buckets confined to one nzgd_id.
    by_hash: dict[bytes, list[int]] = defaultdict(list)
    for rid, h in hashes.items():
        by_hash[h].append(rid)
    cross_nzgd_buckets = []
    for h, rids in by_hash.items():
        if len(rids) < 2:
            continue
        nz_set = {report_to_nzgd[r] for r in rids}
        if len(nz_set) >= 2:
            cross_nzgd_buckets.append((h, rids))

    if not cross_nzgd_buckets:
        return []

    # Build a single list of all cross-nzgd_id matched pairs, carrying the hash hex.
    all_matched_pairs: list[tuple[int, int, int, int, str]] = []
    for h, rids in cross_nzgd_buckets:
        h_hex = h.hex()
        for a, b in combinations(rids, 2):
            nz_a, nz_b = report_to_nzgd[a], report_to_nzgd[b]
            if nz_a == nz_b:
                continue
            all_matched_pairs.append((nz_a, a, nz_b, b, h_hex))

    edges = [(nz_a, nz_b) for nz_a, _, nz_b, _, _ in all_matched_pairs]

    nzgd_to_cluster = connected_components_from_edges(edges)
    clusters: dict[int, list[int]] = defaultdict(list)
    for nz, cl in nzgd_to_cluster.items():
        clusters[cl].append(nz)

    plan: list[MergePlanEntry] = []
    for cluster_id, nzgd_ids in clusters.items():
        cluster_set = set(nzgd_ids)
        cluster_pairs = [
            (nz_a, rep_a, nz_b, rep_b, h_hex)
            for nz_a, rep_a, nz_b, rep_b, h_hex in all_matched_pairs
            if nz_a in cluster_set and nz_b in cluster_set
        ]

        canonical = select_canonical(
            conn, nzgd_ids, [(nz_a, ra, nz_b, rb) for nz_a, ra, nz_b, rb, _ in cluster_pairs], table_cfg
        )
        # For each non-canonical nzgd_id, build a MergePlanEntry.
        for merged_nz in nzgd_ids:
            if merged_nz == canonical:
                continue
            # Matched pairs between (canonical, merged_nz): orient with canonical first.
            entry_matched_pairs: list[ReportPairMatch] = []
            for nz_a, rep_a, nz_b, rep_b, h_hex in cluster_pairs:
                if {nz_a, nz_b} == {canonical, merged_nz}:
                    if nz_a == canonical:
                        entry_matched_pairs.append(
                            ReportPairMatch(rep_a, rep_b, {"hash": h_hex})
                        )
                    else:
                        entry_matched_pairs.append(
                            ReportPairMatch(rep_b, rep_a, {"hash": h_hex})
                        )
            # Reports unique to merged_nz: query its report ids, subtract matched.
            cur.execute(
                f"SELECT {table_cfg.report_id_column} FROM {table_cfg.report_table} "
                f"WHERE nzgd_id = ?",
                (merged_nz,),
            )
            merged_reports = {r[0] for r in cur.fetchall()}
            matched_merged_ids = {p.merged_report_id for p in entry_matched_pairs}
            unique_ids = sorted(merged_reports - matched_merged_ids)
            plan.append(
                MergePlanEntry(
                    cluster_id=cluster_id,
                    canonical_nzgd_id=canonical,
                    merged_nzgd_id=merged_nz,
                    record_type=table_cfg.record_type,
                    match_pass="hash",
                    matched_pairs=entry_matched_pairs,
                    unique_merged_report_ids=unique_ids,
                )
            )
    return plan
