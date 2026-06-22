"""Pass 2: metadata-blocked fuzzy match over post-pass-1 survivors."""

import datetime as _dt
import math
import random
import sqlite3
from collections import defaultdict
from itertools import combinations

import numpy as np
from rapidfuzz import fuzz
from sklearn.neighbors import BallTree
from tqdm import tqdm

from nzgd.dedup.cluster import connected_components_from_edges
from nzgd.dedup.data_types import MergePlanEntry, ReportPairMatch, TableConfig
from nzgd.dedup.trace_compare import (
    best_trace_score as _best_trace_score,
    coerce_to_float as _coerce_to_float,
    load_traces as _load_traces,
    trace_score as _trace_score,
)
from nzgd.dedup.selection import select_canonical


def _load_active_records(
    conn: sqlite3.Connection, table_cfg: TableConfig
) -> list[dict]:
    """Return one dict per nzgd_id of the given record type that still has active reports."""
    type_id = 1 if table_cfg.record_type == "CPT" else 2
    cur = conn.cursor()
    cur.execute(
        "SELECT n.nzgd_id, n.latitude, n.longitude, n.record_created_on, "
        "n.original_investigation_name "
        "FROM nzgdrecord n "
        f"WHERE n.merged_into_nzgd_id IS NULL AND n.type_id = ? "
        f"AND EXISTS (SELECT 1 FROM {table_cfg.report_table} r WHERE r.nzgd_id = n.nzgd_id)",
        (type_id,),
    )
    return [
        {"nzgd_id": r[0], "lat": r[1], "lon": r[2], "date": r[3], "name": r[4]}
        for r in cur.fetchall()
    ]


def _predicate(features: dict, thresholds: dict) -> bool:
    """Spec's conjunctive match predicate. Missing date/name are 'no signal'."""
    if features["spatial_m"] >= thresholds["spatial_radius_m"]:
        return False
    if features["date_days"] is not None and features["date_days"] >= thresholds["date_window_days"]:
        return False
    if features["name_sim"] is not None and features["name_sim"] <= thresholds["name_similarity_min"]:
        return False
    if features["trace_score"] >= thresholds["trace_score_max"]:
        return False
    return True


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return float(2 * r * math.asin(math.sqrt(a)))


def _date_diff_days(a: str | None, b: str | None) -> float | None:
    if a is None or b is None:
        return None
    try:
        da = _dt.date.fromisoformat(a)
        db = _dt.date.fromisoformat(b)
    except (TypeError, ValueError):
        return None
    return float(abs((da - db).days))


def _name_sim(a: str | None, b: str | None) -> float | None:
    if a is None or b is None:
        return None
    return float(fuzz.token_set_ratio(a, b))


def _blocked_candidate_pairs(
    records: list[dict], radius_m: float
) -> list[tuple[int, int]]:
    """BallTree haversine-radius query → unique unordered nzgd_id pairs."""
    if len(records) < 2:
        return []
    rad = np.radians(np.array([[r["lat"], r["lon"]] for r in records]))
    tree = BallTree(rad, metric="haversine")
    earth_r = 6371008.8
    radius_rad = radius_m / earth_r
    pairs: set[tuple[int, int]] = set()
    for i, point in enumerate(rad):
        idxs = tree.query_radius(point.reshape(1, -1), r=radius_rad)[0]
        for j in idxs:
            if j == i:
                continue
            a, b = records[i]["nzgd_id"], records[int(j)]["nzgd_id"]
            pairs.add((min(a, b), max(a, b)))
    return list(pairs)


def generate_fuzzy_merge_plan(
    conn: sqlite3.Connection,
    table_cfg: TableConfig,
    thresholds: dict,
    calibration_collector: dict | None = None,
    random_seed: int = 0,
) -> list[MergePlanEntry]:
    """Produce MergePlanEntry list for fuzzy near-duplicates.

    Entries within a cluster are sorted by trace_score (best-match first) so the
    executor's per-step metadata-enrichment respects the spec's score-order rule.

    `calibration_collector`, if provided, is populated with two lists:
      - `positive`: feature dicts for pairs that fired the predicate.
      - `negative`: feature dicts for a random sample of pairs that did not fire.
    Caller passes both to `reports.write_calibration_report`.
    """
    records = _load_active_records(conn, table_cfg)
    by_nzgd = {r["nzgd_id"]: r for r in records}
    candidate_pairs = _blocked_candidate_pairs(records, thresholds["spatial_radius_m"])

    step = thresholds["trace_resample_step_m"]
    trace_cache: dict[int, dict[int, np.ndarray]] = {}

    def traces_for(nz: int) -> dict[int, np.ndarray]:
        if nz not in trace_cache:
            trace_cache[nz] = _load_traces(conn, nz, table_cfg)
        return trace_cache[nz]

    positive_examples: list[dict] = []
    negative_examples: list[dict] = []
    rng = random.Random(random_seed)
    negative_target = thresholds.get("random_pair_sample_size", 5000)

    edges: list[tuple[int, int]] = []
    pair_metadata: dict[tuple[int, int], dict] = {}

    for nz_a, nz_b in tqdm(candidate_pairs, desc=f"fuzzy {table_cfg.record_type}"):
        rec_a = by_nzgd[nz_a]
        rec_b = by_nzgd[nz_b]
        spatial = _haversine_m(rec_a["lat"], rec_a["lon"], rec_b["lat"], rec_b["lon"])
        date_days = _date_diff_days(rec_a["date"], rec_b["date"])
        name_sim = _name_sim(rec_a["name"], rec_b["name"])
        ta, tb = traces_for(nz_a), traces_for(nz_b)
        trace_score, best_pair = _best_trace_score(ta, tb, step)
        max_depth_diff: float | None = None
        if ta and tb:
            a_depths = [t[:, 0].max() for t in ta.values() if t.shape[0] > 0]
            b_depths = [t[:, 0].max() for t in tb.values() if t.shape[0] > 0]
            if a_depths and b_depths:
                max_depth_diff = float(abs(max(a_depths) - max(b_depths)))
        features = {
            "spatial_m": spatial,
            "date_days": date_days,
            "name_sim": name_sim,
            "max_depth_diff_m": max_depth_diff,
            "trace_score": trace_score,
        }

        predicate_matched = _predicate(features, thresholds) and best_pair is not None
        if predicate_matched:
            edges.append((nz_a, nz_b))
            key = (min(nz_a, nz_b), max(nz_a, nz_b))
            best_pair_oriented = best_pair if nz_a < nz_b else (best_pair[1], best_pair[0])
            pair_metadata[key] = {"features": features, "best_pair": best_pair_oriented}

        if calibration_collector is not None:
            if predicate_matched:
                positive_examples.append(features)
            elif len(negative_examples) < negative_target and rng.random() < 0.5:
                negative_examples.append(features)

    if calibration_collector is not None:
        calibration_collector["positive"] = positive_examples
        calibration_collector["negative"] = negative_examples

    if not edges:
        return []

    nzgd_to_cluster = connected_components_from_edges(edges)
    clusters: dict[int, list[int]] = defaultdict(list)
    for nz, cl in nzgd_to_cluster.items():
        clusters[cl].append(nz)

    cur = conn.cursor()
    plan: list[MergePlanEntry] = []
    for cluster_id, nzgd_ids in clusters.items():
        matched_pairs_for_selection: list[tuple[int, int, int, int]] = []
        per_pair_match: dict[tuple[int, int], ReportPairMatch] = {}
        for a, b in combinations(sorted(nzgd_ids), 2):
            key = (min(a, b), max(a, b))
            if key not in pair_metadata:
                continue
            ra, rb = pair_metadata[key]["best_pair"]
            matched_pairs_for_selection.append((a, ra, b, rb))
            per_pair_match[key] = ReportPairMatch(
                canonical_report_id=ra,
                merged_report_id=rb,
                metrics=dict(pair_metadata[key]["features"]),
            )

        canonical = select_canonical(conn, nzgd_ids, matched_pairs_for_selection, table_cfg)

        # Sort the merged nzgd_ids by trace_score with the canonical (best first),
        # so the executor applies metadata enrichment in score order.
        def _score_with_canonical(merged_nz: int) -> float:
            key = (min(canonical, merged_nz), max(canonical, merged_nz))
            return pair_metadata.get(key, {}).get("features", {}).get("trace_score", math.inf)

        ordered_merged = sorted(
            (nz for nz in nzgd_ids if nz != canonical),
            key=_score_with_canonical,
        )

        for merged_nz in ordered_merged:
            key = (min(canonical, merged_nz), max(canonical, merged_nz))
            matched_pairs: list[ReportPairMatch] = []
            if key in per_pair_match:
                pm = per_pair_match[key]
                if canonical < merged_nz:
                    matched_pairs.append(pm)
                else:
                    matched_pairs.append(
                        ReportPairMatch(pm.merged_report_id, pm.canonical_report_id, pm.metrics)
                    )

            cur.execute(
                f"SELECT {table_cfg.report_id_column} FROM {table_cfg.report_table} "
                f"WHERE nzgd_id = ?",
                (merged_nz,),
            )
            merged_reports = {r[0] for r in cur.fetchall()}
            matched_merged_ids = {p.merged_report_id for p in matched_pairs}
            unique_ids = sorted(merged_reports - matched_merged_ids)
            plan.append(
                MergePlanEntry(
                    cluster_id=cluster_id,
                    canonical_nzgd_id=canonical,
                    merged_nzgd_id=merged_nz,
                    record_type=table_cfg.record_type,
                    match_pass="fuzzy",
                    matched_pairs=matched_pairs,
                    unique_merged_report_ids=unique_ids,
                )
            )
    return plan
