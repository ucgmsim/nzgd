"""Post-dedup within-record supplemental-value consolidation.

After the dedup passes, one nzgd_id can have several surviving cptreport rows
(format-sibling files that landed in different trace clusters). A supplemental
value extracted from one sibling never reaches the others. This step fills each
record's surviving rows with the record's best-available value per supplemental
column — faithful to the sources: prefer the single in-range value, else the
single recorded value (e.g. a literal 0), else leave NULL. Small-spread valid
conflicts are resolved by a corroboration-first selector; large-spread (likely
artifact) conflicts are skipped and reported. Only NULL/non-useful cells are
filled; a valid value is never overridden.
"""

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone

from nzgd import constants
from nzgd.dedup.data_types import TableConfig
from nzgd.dedup.plausibility import is_useful_value

# Supplemental columns consolidated per record (CPT). gwl_method_id is filled
# from the same source row as extracted_gwl_m, not consolidated independently.
_SUPPLEMENTAL_COLUMNS = (
    "extracted_gwl_m",
    "tip_net_area_ratio",
    "predrill_depth_m",
    "termination_reason_id",
)
_GWL_METHOD_COLUMN = "gwl_method_id"


def _decimal_places(value: float) -> int:
    """Decimal places of `value` after rounding to 3 dp (guards against float noise)."""
    text = f"{round(float(value), 3):.3f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text and text.split(".")[1] else 0


def select_value(candidates: list[tuple[float, int]]) -> float:
    """Pick a consolidated value from (value, cpt_id) candidates of small spread.

    Corroboration first (the value on the most rows), then most decimal places
    (rounded to 3 dp), then smallest cpt_id. Only ever returns a value present in
    `candidates`; unbiased; deterministic.
    """
    counts = Counter(v for v, _ in candidates)
    best = max(counts.values())
    tied = [v for v in counts if counts[v] == best]
    if len(tied) == 1:
        return tied[0]

    def smallest_cpt(v: float) -> int:
        return min(cid for vv, cid in candidates if vv == v)

    return sorted(tied, key=lambda v: (-_decimal_places(v), smallest_cpt(v)))[0]


def consolidate_within_record_supplemental(
    conn: sqlite3.Connection,
    table_cfg: TableConfig,
    run_id: int,
    failures: list[dict] | None = None,
) -> tuple[int, int]:
    """Fill each nzgd_id's surviving rows with its best-available supplemental value.

    Returns (n_records_changed, n_cells_filled). Writes one dedup_audit row per
    affected nzgd_id (match_pass='supplemental_consolidation'), recording filled
    cells in metadata_copied_json and skipped conflicts in metadata_conflicts_json.
    If `failures` is given, a per-record error appends `{nzgd_id, error}` and
    continues; otherwise it re-raises.
    """
    table = table_cfg.report_table
    id_col = table_cfg.report_id_column
    cols = list(_SUPPLEMENTAL_COLUMNS) + [_GWL_METHOD_COLUMN]
    thresholds = constants.DEDUP_CONFIG["within_record_supplemental"]["small_spread_threshold"]

    cur = conn.cursor()
    cur.execute(
        f"SELECT nzgd_id, {id_col}, {', '.join(cols)} FROM {table} ORDER BY nzgd_id, {id_col}"
    )
    rows_by_nzgd: dict[int, list[dict]] = {}
    for row in cur.fetchall():
        rec = {"_id": row[1], **dict(zip(cols, row[2:]))}
        rows_by_nzgd.setdefault(row[0], []).append(rec)

    n_records_changed = 0
    n_cells_filled = 0

    for nzgd_id, rows in rows_by_nzgd.items():
        if len(rows) < 2:
            continue
        savepoint = f"supp_consol_{nzgd_id}"
        cur.execute(f"SAVEPOINT {savepoint}")
        try:
            copied: dict[str, dict] = {}
            conflicts: dict[str, dict] = {}
            record_changed = False
            for col in _SUPPLEMENTAL_COLUMNS:
                useful = [(r[col], r["_id"]) for r in rows if is_useful_value(r[col], table, col)]
                recorded = [(r[col], r["_id"]) for r in rows if r[col] is not None]

                consolidated = None
                if useful:
                    distinct = sorted({v for v, _ in useful})
                    if len(distinct) == 1:
                        consolidated = distinct[0]
                    else:
                        spread = distinct[-1] - distinct[0]
                        thr = thresholds.get(col)
                        if thr is not None and spread <= thr:
                            consolidated = select_value(useful)
                        else:
                            conflicts[col] = {"values": distinct, "spread": spread}
                            continue
                    source_id = min(cid for v, cid in useful if v == consolidated)
                elif recorded:
                    distinct = sorted({v for v, _ in recorded})
                    if len(distinct) == 1:
                        consolidated = distinct[0]
                        source_id = min(cid for v, cid in recorded if v == consolidated)
                    else:
                        conflicts[col] = {"values": distinct, "spread": None}
                        continue
                else:
                    continue  # all NULL

                gwl_method = None
                if col == "extracted_gwl_m":
                    gwl_method = next(r[_GWL_METHOD_COLUMN] for r in rows if r["_id"] == source_id)

                filled = []
                for r in rows:
                    if r[col] == consolidated or is_useful_value(r[col], table, col):
                        continue  # already set, or a valid value we must not override
                    cur.execute(
                        f"UPDATE {table} SET {col} = ? WHERE {id_col} = ?",
                        (consolidated, r["_id"]),
                    )
                    if col == "extracted_gwl_m":
                        cur.execute(
                            f"UPDATE {table} SET {_GWL_METHOD_COLUMN} = ? WHERE {id_col} = ?",
                            (gwl_method, r["_id"]),
                        )
                    r[col] = consolidated
                    filled.append(r["_id"])
                    n_cells_filled += 1
                if filled:
                    record_changed = True
                    copied[col] = {
                        "value": consolidated,
                        "source_report_id": source_id,
                        "target_report_ids": filled,
                    }

            if copied or conflicts:
                cur.execute(
                    "INSERT INTO dedup_audit (run_id, cluster_id, canonical_nzgd_id, "
                    "merged_nzgd_id, record_type, match_pass, report_pairs_json, "
                    "metadata_copied_json, metadata_conflicts_json, merged_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id, nzgd_id, nzgd_id, nzgd_id, table_cfg.record_type,
                        "supplemental_consolidation", json.dumps([]),
                        json.dumps(copied) if copied else None,
                        json.dumps(conflicts) if conflicts else None,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            if record_changed:
                n_records_changed += 1
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception as exc:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            if failures is not None:
                failures.append({"nzgd_id": nzgd_id, "error": repr(exc)})
                continue
            raise
    conn.commit()
    return n_records_changed, n_cells_filled
