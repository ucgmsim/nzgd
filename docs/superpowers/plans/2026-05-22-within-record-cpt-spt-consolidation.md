# Within-Record CPT/SPT Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Pass 0 to the dedup module that collapses redundant cptreport / sptreport rows within a single nzgd_id into one canonical row, with plausibility-aware metadata merging. Remove the existing extraction-time `remove_duplicate_extractions` and the deprecated `cpt_data_duplicate_of_cpt_id` column.

**Architecture:** A new pass in `nzgd/dedup/pass0_within_record.py` runs before the existing cross-record passes. Three new helpers — `plausibility.is_useful_value`, `canonical_selectors.default_within_record_canonical`, and a `trace_compare` module that extracts shared trace-handling helpers from `pass2_fuzzy` — are reused by the cross-record passes too. Schema migration drops one column and widens a CHECK constraint.

**Tech Stack:** Python 3.9+, SQLite (stdlib `sqlite3`), `scipy.sparse.csgraph.connected_components`, `numpy`, `tqdm`, `pytest`. Reference spec: `docs/superpowers/specs/2026-05-22-within-record-cpt-spt-consolidation-design.md`.

**Environment:** Python at `/home/arr65/venvs/dev_nzgd_venv/bin/python`. Source DB at `/home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p6p0_20260403.db`. Run all commands from repo root `/home/arr65/src/nzgd`.

---

## File Structure

**New files:**
- `nzgd/dedup/trace_compare.py` — shared trace-handling helpers (extracted from `pass2_fuzzy.py`)
- `nzgd/dedup/plausibility.py` — `is_useful_value(value, table, column)` helper
- `nzgd/dedup/canonical_selectors.py` — `ClusterRow`, `default_within_record_canonical`, `CanonicalSelector` type
- `nzgd/dedup/pass0_within_record.py` — Pass 0 implementation (clustering, plan generation, plan application)

**Modify:**
- `nzgd/dedup/pass2_fuzzy.py` — remove the helpers moved to `trace_compare.py`; import them from there
- `nzgd/dedup/executor.py` — use `is_useful_value` in `_enrich_canonical_metadata`; rename `_delete_report` → `delete_report`; update internal callers
- `nzgd/dedup/schema.py` — extend `apply_dedup_schema` with DROP COLUMN and CHECK-constraint widening
- `nzgd/scripts/db/deduplicate.py` — invoke Pass 0 before Pass 1 for each record type
- `nzgd/resources/config.yaml` — add `within_record` and `field_plausibility_ranges` blocks
- `nzgd/db/orm.py` — remove `cpt_data_duplicate_of_cpt_id` field (line 375)
- `nzgd/extract/cpt/workflow.py` — remove call to `remove_duplicate_extractions`
- `nzgd/extract/cpt/conditioning.py` — remove `remove_duplicate_extractions` function
- `nzgd/extract/cpt/output.py` — remove `removed_duplicates` field handling
- `nzgd/scripts/db/put_cpts_in_db.py` — drop keeper/duplicate branching and `cpt_data_duplicate_of_cpt_id` column from INSERT
- `tests/dedup/conftest.py` — remove `cpt_data_duplicate_of_cpt_id` from `_FULL_SCHEMA_SQL`
- `tests/dedup/test_dedup_pipeline.py` — append 13 new integration scenarios

The existing cross-record cpt_data_duplicate_of_cpt_id column is dropped from the schema; the deduped DB after Pass 0 carries no record of within-record duplicates beyond the new `dedup_audit` entries (whose `match_pass = 'within_record'` rows replace the column's role).

---

## Task 1: Refactor trace helpers into shared module

**Files:**
- Create: `nzgd/dedup/trace_compare.py`
- Modify: `nzgd/dedup/pass2_fuzzy.py`

- [ ] **Step 1: Create `trace_compare.py` with the helpers moved out of `pass2_fuzzy.py`**

Create `/home/arr65/src/nzgd/nzgd/dedup/trace_compare.py` with EXACTLY this content (identical to the current `pass2_fuzzy.py` functions but as public names):

```python
"""Trace-comparison helpers shared between the dedup passes."""

import math
import sqlite3
from collections import defaultdict

import numpy as np

from nzgd.dedup.data_types import TableConfig


def coerce_to_float(v) -> float:
    """Coerce a SQLite cell to float; non-numeric strings become NaN."""
    if v is None:
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def load_traces(
    conn: sqlite3.Connection, nzgd_id: int, table_cfg: TableConfig
) -> dict[int, np.ndarray]:
    """Return `{report_id: ndarray of shape (n_rows, len(value_columns))}` for one nzgd_id.

    Columns appear in the order given by `table_cfg.measurement_value_columns`;
    the first must be `depth_m` so column 0 of each array is depth. Non-numeric
    measurement values (e.g., SPT `ISPT_REP` blow-count strings) become NaN.
    """
    value_cols_with_m = ", ".join(f"m.{c}" for c in table_cfg.measurement_value_columns)
    cur = conn.cursor()
    cur.execute(
        f"SELECT r.{table_cfg.report_id_column}, {value_cols_with_m} "
        f"FROM {table_cfg.measurement_table} m "
        f"JOIN {table_cfg.report_table} r "
        f"ON r.{table_cfg.report_id_column} = m.{table_cfg.report_id_column} "
        f"WHERE r.nzgd_id = ? "
        f"ORDER BY r.{table_cfg.report_id_column}, m.depth_m",
        (nzgd_id,),
    )
    rows_by_report: dict[int, list[tuple]] = defaultdict(list)
    for row in cur.fetchall():
        rid = row[0]
        rows_by_report[rid].append(tuple(coerce_to_float(v) for v in row[1:]))
    return {rid: np.array(rows, dtype=float) for rid, rows in rows_by_report.items()}


def trace_score(a: np.ndarray, b: np.ndarray, step: float) -> float:
    """Aligned normalised-RMSE sum across non-depth channels.

    Returns inf if either trace has fewer than 2 points or their depth ranges
    do not overlap. Each channel's RMSE is divided by the mean absolute value
    across both traces (floored at 1e-6) before being summed across channels,
    so the score is comparable across qc/fs/u2 which have different magnitudes.
    """
    if a.shape[0] < 2 or b.shape[0] < 2:
        return math.inf
    lo = max(a[:, 0].min(), b[:, 0].min())
    hi = min(a[:, 0].max(), b[:, 0].max())
    if hi <= lo:
        return math.inf
    grid = np.arange(lo, hi + step / 2, step)
    if grid.size < 2:
        return math.inf
    total = 0.0
    for ch in range(1, a.shape[1]):
        ai = np.interp(grid, a[:, 0], a[:, ch])
        bi = np.interp(grid, b[:, 0], b[:, ch])
        denom = max((np.abs(ai).mean() + np.abs(bi).mean()) / 2.0, 1e-6)
        rmse = float(np.sqrt(np.mean((ai - bi) ** 2)))
        total += rmse / denom
    return total


def best_trace_score(
    traces_a: dict[int, np.ndarray],
    traces_b: dict[int, np.ndarray],
    step: float,
) -> tuple[float, tuple[int, int] | None]:
    """Best (lowest) trace_score over all cross-record report pairs, plus the winning pair."""
    best_score = math.inf
    best_pair: tuple[int, int] | None = None
    for ra, ta in traces_a.items():
        for rb, tb in traces_b.items():
            s = trace_score(ta, tb, step)
            if s < best_score:
                best_score = s
                best_pair = (ra, rb)
    return best_score, best_pair
```

- [ ] **Step 2: Update `pass2_fuzzy.py` to import from `trace_compare`**

Edit `/home/arr65/src/nzgd/nzgd/dedup/pass2_fuzzy.py`:

Replace the entire block of private functions (`_coerce_to_float`, `_load_traces`, `_trace_score`, `_best_trace_score`) with a single import. The block to replace starts at `def _coerce_to_float(v) -> float:` and ends at the final `return best_score, best_pair` of `_best_trace_score` (before `def _predicate`).

Replace it with:

```python
from nzgd.dedup.trace_compare import (
    best_trace_score as _best_trace_score,
    coerce_to_float as _coerce_to_float,
    load_traces as _load_traces,
    trace_score as _trace_score,
)
```

(The `as _x` aliases keep `pass2_fuzzy.py`'s internal call sites unchanged.) Place this import after the existing `from nzgd.dedup.cluster import ...` block, before `from nzgd.dedup.selection import select_canonical`.

Remove the now-unused `defaultdict` import if no other code in the file uses it. (Grep first: `grep "defaultdict" /home/arr65/src/nzgd/nzgd/dedup/pass2_fuzzy.py`. If used elsewhere, keep the import.)

- [ ] **Step 3: Run the full existing test suite to confirm no regressions**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -v
```

Expected: all 10 existing tests still pass.

- [ ] **Step 4: Commit**

```bash
cd /home/arr65/src/nzgd && git add nzgd/dedup/trace_compare.py nzgd/dedup/pass2_fuzzy.py && git commit -m "Extract shared trace helpers into nzgd/dedup/trace_compare.py"
```

---

## Task 2: Add plausibility helper module

**Files:**
- Create: `nzgd/dedup/plausibility.py`

- [ ] **Step 1: Write `plausibility.py`**

Create `/home/arr65/src/nzgd/nzgd/dedup/plausibility.py` with EXACTLY this content:

```python
"""Per-field plausibility check shared by all dedup passes."""

from typing import Any

from nzgd import constants


def is_useful_value(value: Any, table: str, column: str) -> bool:
    """Return True if `value` is non-NULL and (if a range is configured) within range.

    `table` is 'nzgdrecord', 'cptreport', or 'sptreport'. `column` is a column
    name within that table.

    A `None` value is never useful. A numeric value is checked against the
    configured plausibility range (inclusive) for `(table, column)`; if no
    range is configured, the value is treated as useful. Non-numeric values
    (text fields, dates) bypass the range check entirely — only the non-NULL
    check applies.
    """
    if value is None:
        return False
    ranges = constants.DEDUP_CONFIG.get("field_plausibility_ranges", {}).get(table, {})
    if column not in ranges:
        return True
    lo, hi = ranges[column]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    return lo <= v <= hi
```

- [ ] **Step 2: Smoke check on a synthetic config**

Verify the helper works against the project's `DEDUP_CONFIG` (currently doesn't have plausibility ranges; this confirms the empty-ranges path returns True for any non-NULL).

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
from nzgd.dedup.plausibility import is_useful_value
# No ranges configured yet → just non-NULL check applies
assert is_useful_value(None, 'cptreport', 'extracted_gwl_m') is False
assert is_useful_value(0, 'cptreport', 'extracted_gwl_m') is True   # would be False after Task 7
assert is_useful_value(5.2, 'cptreport', 'extracted_gwl_m') is True
assert is_useful_value('CPT1', 'nzgdrecord', 'original_investigation_name') is True
print('plausibility OK')
"
```

Expected: `plausibility OK`. (Note: the `extracted_gwl_m=0` case currently returns True because no range is configured yet. After Task 7 adds the config, it returns False.)

- [ ] **Step 3: Commit**

```bash
cd /home/arr65/src/nzgd && git add nzgd/dedup/plausibility.py && git commit -m "Add plausibility helper for dedup field-range checks"
```

---

## Task 3: Add canonical-selector module

**Files:**
- Create: `nzgd/dedup/canonical_selectors.py`

- [ ] **Step 1: Write `canonical_selectors.py`**

Create `/home/arr65/src/nzgd/nzgd/dedup/canonical_selectors.py` with EXACTLY this content:

```python
"""Pluggable canonical-selection rules for within-record consolidation."""

from dataclasses import dataclass
from typing import Callable, Sequence

from nzgd.dedup.data_types import TableConfig


@dataclass(frozen=True)
class ClusterRow:
    """Compact summary of one cptreport/sptreport row for selector input."""

    report_id: int                  # cpt_id or spt_id
    has_data: bool                  # has_cpt_data=1 for CPT; measurement_row_count > 0 for SPT
    measurement_row_count: int
    metadata_non_null_count: int    # non-NULL fields in cptreport/sptreport metadata


CanonicalSelector = Callable[[Sequence[ClusterRow], TableConfig], int]


def default_within_record_canonical(
    cluster_rows: Sequence[ClusterRow],
    table_cfg: TableConfig,
) -> int:
    """v1 default: prefer rows with has_data=True; tiebreaker smallest report_id."""
    candidates = [r for r in cluster_rows if r.has_data] or list(cluster_rows)
    return min(candidates, key=lambda r: r.report_id).report_id
```

- [ ] **Step 2: Smoke check**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
from nzgd.dedup.canonical_selectors import ClusterRow, default_within_record_canonical
from nzgd.dedup.data_types import CPT_TABLE_CONFIG

# data_bearing wins over no-data, even when no-data has smaller id
rows = [
    ClusterRow(report_id=10, has_data=False, measurement_row_count=0, metadata_non_null_count=5),
    ClusterRow(report_id=20, has_data=True,  measurement_row_count=100, metadata_non_null_count=3),
]
assert default_within_record_canonical(rows, CPT_TABLE_CONFIG) == 20

# Among data-bearing, smallest id wins
rows = [
    ClusterRow(report_id=30, has_data=True, measurement_row_count=100, metadata_non_null_count=3),
    ClusterRow(report_id=20, has_data=True, measurement_row_count=200, metadata_non_null_count=5),
]
assert default_within_record_canonical(rows, CPT_TABLE_CONFIG) == 20

# All no-data: smallest id
rows = [
    ClusterRow(report_id=50, has_data=False, measurement_row_count=0, metadata_non_null_count=4),
    ClusterRow(report_id=30, has_data=False, measurement_row_count=0, metadata_non_null_count=2),
]
assert default_within_record_canonical(rows, CPT_TABLE_CONFIG) == 30

print('canonical_selectors OK')
"
```

Expected: `canonical_selectors OK`.

- [ ] **Step 3: Commit**

```bash
cd /home/arr65/src/nzgd && git add nzgd/dedup/canonical_selectors.py && git commit -m "Add canonical-selector module with ClusterRow and default rule"
```

---

## Task 4: Update executor to use plausibility helper + expose `delete_report`

**Files:**
- Modify: `nzgd/dedup/executor.py`

- [ ] **Step 1: Update `_enrich_canonical_metadata` to use `is_useful_value`**

Edit `/home/arr65/src/nzgd/nzgd/dedup/executor.py`. Add this import after `from nzgd.dedup.data_types import MergePlanEntry, TableConfig`:

```python
from nzgd.dedup.plausibility import is_useful_value
```

In `_enrich_canonical_metadata`, find this loop (around the middle of the function):

```python
    for col in _NZGDRECORD_METADATA_COLUMNS:
        if canon_vals[col] is not None:
            continue  # canonical wins
        candidates = merged_vals[col]
        if not candidates:
            continue  # nothing to copy
        distinct = {c[0] for c in candidates}
        if len(distinct) == 1:
            chosen = candidates[0]
            updates[col] = chosen[0]
            copied[col] = {"value": chosen[0], "source_nzgd_id": chosen[1]}
        else:
            # Multiple distinct values; pick the one from the smallest merged nzgd_id.
            chosen = min(candidates, key=lambda c: c[1])
            updates[col] = chosen[0]
            copied[col] = {"value": chosen[0], "source_nzgd_id": chosen[1]}
            conflicts[col] = [
                {"value": v, "source_nzgd_id": nz} for v, nz in candidates
            ]
```

Replace it with:

```python
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
```

- [ ] **Step 2: Rename `_delete_report` to `delete_report` (public)**

In the same file, find `def _delete_report(` and rename to `def delete_report(`. Find the one call site inside `apply_merge_plan` (the line `_delete_report(conn, pair.merged_report_id, table_cfg)`) and update it to `delete_report(conn, pair.merged_report_id, table_cfg)`.

- [ ] **Step 3: Run all existing tests to confirm no regressions**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -v
```

Expected: all 10 existing tests still pass. (None should be affected by the plausibility change yet because no ranges are configured.)

- [ ] **Step 4: Commit**

```bash
cd /home/arr65/src/nzgd && git add nzgd/dedup/executor.py && git commit -m "Cross-record executor: use is_useful_value; expose delete_report"
```

---

## Task 5: Extend schema migration (DROP COLUMN + widen CHECK)

**Files:**
- Modify: `nzgd/dedup/schema.py`

- [ ] **Step 1: Add helpers and extend `apply_dedup_schema`**

Edit `/home/arr65/src/nzgd/nzgd/dedup/schema.py`. After the existing constants and before the `def apply_dedup_schema` definition, add:

```python
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
```

In the existing `def apply_dedup_schema(conn: sqlite3.Connection) -> None:` function, find the final `conn.commit()` line. Replace the block from `cur = conn.cursor()` down to (but not including) `conn.commit()` with:

```python
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
```

- [ ] **Step 2: Smoke check the migration on an in-memory DB that simulates the pre-migration state**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
import sqlite3
from nzgd.dedup.schema import apply_dedup_schema

conn = sqlite3.connect(':memory:')
conn.execute('PRAGMA foreign_keys = ON')
# Simulate pre-migration: nzgdrecord, cptreport with legacy column, OLD dedup_audit
conn.executescript('''
CREATE TABLE nzgdrecord (nzgd_id INTEGER PRIMARY KEY);
CREATE TABLE cptreport (cpt_id INTEGER PRIMARY KEY, nzgd_id INTEGER, source_file TEXT,
    cpt_data_duplicate_of_cpt_id INTEGER);
CREATE TABLE dedup_run (run_id INTEGER PRIMARY KEY, started_at TEXT NOT NULL,
    source_db_path TEXT NOT NULL, script_version TEXT NOT NULL, config_snapshot_json TEXT NOT NULL,
    finished_at TEXT, n_clusters_merged INTEGER, n_records_merged INTEGER);
CREATE TABLE dedup_audit (
    audit_id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES dedup_run(run_id),
    cluster_id INTEGER NOT NULL, canonical_nzgd_id INTEGER NOT NULL, merged_nzgd_id INTEGER NOT NULL,
    record_type TEXT NOT NULL CHECK(record_type IN ('CPT', 'BH')),
    match_pass TEXT NOT NULL CHECK(match_pass IN ('hash', 'fuzzy')),
    report_pairs_json TEXT NOT NULL, metadata_copied_json TEXT, metadata_conflicts_json TEXT,
    merged_at TEXT NOT NULL
);
INSERT INTO cptreport (cpt_id, nzgd_id, source_file, cpt_data_duplicate_of_cpt_id) VALUES (1, 100, 'f.xlsx_sheet_0', 99);
INSERT INTO dedup_run (run_id, started_at, source_db_path, script_version, config_snapshot_json) VALUES (1, 't', 't', 't', '{}');
INSERT INTO dedup_audit (run_id, cluster_id, canonical_nzgd_id, merged_nzgd_id, record_type, match_pass, report_pairs_json, merged_at) VALUES (1, 1, 100, 200, 'CPT', 'hash', '[]', 't');
''')
apply_dedup_schema(conn)
# Column dropped
cur = conn.execute('PRAGMA table_info(cptreport)')
cols = [r[1] for r in cur.fetchall()]
assert 'cpt_data_duplicate_of_cpt_id' not in cols, cols
print('column dropped OK; remaining:', cols)
# CHECK widened
sql = conn.execute(\"SELECT sql FROM sqlite_master WHERE name='dedup_audit'\").fetchone()[0]
assert \"'within_record'\" in sql, sql
print('CHECK widened OK')
# Existing data preserved
n = conn.execute('SELECT COUNT(*) FROM dedup_audit').fetchone()[0]
assert n == 1
print('audit data preserved:', n)
# Within_record now accepted
conn.execute(\"INSERT INTO dedup_audit (run_id, cluster_id, canonical_nzgd_id, merged_nzgd_id, record_type, match_pass, report_pairs_json, merged_at) VALUES (1, 2, 100, 100, 'CPT', 'within_record', '[]', 't')\")
print('within_record insert OK')
# Idempotent re-run
apply_dedup_schema(conn)
print('idempotent re-run OK')
"
```

Expected: each `OK` line prints. `cpt_data_duplicate_of_cpt_id` not in cols; CHECK includes within_record; data preserved; new INSERT succeeds; second `apply_dedup_schema` call doesn't raise.

- [ ] **Step 3: Run existing tests**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -v
```

Expected: all 10 tests still pass (the conftest's `_FULL_SCHEMA_SQL` still includes the legacy column; the migration drops it during each test's `apply_dedup_schema` call — still works).

- [ ] **Step 4: Commit**

```bash
cd /home/arr65/src/nzgd && git add nzgd/dedup/schema.py && git commit -m "Schema: drop cpt_data_duplicate_of_cpt_id; widen dedup_audit match_pass CHECK"
```

---

## Task 6: Implement Pass 0 (within-record consolidation)

**Files:**
- Create: `nzgd/dedup/pass0_within_record.py`

- [ ] **Step 1: Write `pass0_within_record.py`**

Create `/home/arr65/src/nzgd/nzgd/dedup/pass0_within_record.py` with EXACTLY this content:

```python
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
```

- [ ] **Step 2: Smoke check Pass 0 end-to-end against the existing test fixture**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
import sqlite3
from nzgd.dedup.schema import apply_dedup_schema
from nzgd.dedup.data_types import CPT_TABLE_CONFIG
from nzgd.dedup.pass0_within_record import (
    generate_within_record_consolidation_plan,
    apply_within_record_consolidation_plan,
)

conn = sqlite3.connect(':memory:')
conn.execute('PRAGMA foreign_keys = ON')
conn.executescript('''
CREATE TABLE nzgdrecord (nzgd_id INTEGER PRIMARY KEY,
    type_id INTEGER NOT NULL, latitude REAL, longitude REAL,
    model_vs30_foster_2019_km_per_s REAL, model_vs30_stddev_foster_2019_km_per_s REAL,
    model_gwl_westerhoff_2018_m REAL, model_gwl_nlm_2025_m REAL, model_gwl_nlm_2025_stddev_m REAL,
    original_investigation_name TEXT, investigation_date TEXT, published_date TEXT,
    region_id INTEGER DEFAULT 0, district_id INTEGER DEFAULT 0,
    city_id INTEGER DEFAULT 0, suburb_id INTEGER DEFAULT 0);
CREATE TABLE cptreport (cpt_id INTEGER PRIMARY KEY, nzgd_id INTEGER,
    max_depth_m REAL, min_depth_m REAL,
    extracted_gwl_m REAL, gwl_method_id INTEGER, tip_net_area_ratio REAL,
    predrill_depth_m REAL, termination_reason_id INTEGER, has_cpt_data INTEGER DEFAULT 1,
    did_explicit_unit_conversion INTEGER, did_inferred_unit_conversion INTEGER,
    source_file TEXT NOT NULL);
CREATE TABLE cptmeasurements (measurement_id INTEGER PRIMARY KEY, cpt_id INTEGER,
    depth_m REAL, qc_MPa REAL, fs_MPa REAL, u2_MPa REAL);
CREATE TABLE cptvs30estimates (vs30_id INTEGER PRIMARY KEY, cpt_id INTEGER, nzgd_id INTEGER);
INSERT INTO nzgdrecord (nzgd_id, type_id, latitude, longitude) VALUES (1, 1, -41.0, 174.0);
-- 2 stems, both with the same trace data → should consolidate to 1 row
INSERT INTO cptreport (cpt_id, nzgd_id, source_file, has_cpt_data) VALUES
    (10, 1, 'CPT_1_AGS01.ags_sheet_0', 1),
    (11, 1, 'CPT_1_Raw01.xlsx_sheet_Data', 1);
INSERT INTO cptmeasurements (cpt_id, depth_m, qc_MPa, fs_MPa, u2_MPa) VALUES
    (10, 0.1, 1.0, 0.01, 0.0), (10, 0.2, 1.1, 0.011, 0.0),
    (11, 0.1, 1.0, 0.01, 0.0), (11, 0.2, 1.1, 0.011, 0.0);
''')
apply_dedup_schema(conn)
conn.execute('INSERT INTO dedup_run (started_at, source_db_path, script_version, config_snapshot_json) VALUES (?,?,?,?)', ('t', ':memory:', 't', '{}'))
thresholds = {'trace_score_max': 0.05, 'trace_resample_step_m': 0.05}
plan = generate_within_record_consolidation_plan(conn, CPT_TABLE_CONFIG, thresholds)
print(f'plan length: {len(plan)}')
for p in plan:
    print(f'  cluster={p.cluster_id} canonical={p.canonical_report_id} absorbed={[a.absorbed_report_id for a in p.absorbed_reports]}')
n_c, n_r = apply_within_record_consolidation_plan(conn, plan, 1, CPT_TABLE_CONFIG)
print(f'merged: clusters={n_c} records={n_r}')
remaining = [r for r in conn.execute('SELECT cpt_id FROM cptreport')]
print(f'remaining cptreport rows: {remaining}')
audit = list(conn.execute('SELECT canonical_nzgd_id, merged_nzgd_id, match_pass FROM dedup_audit'))
print(f'audit: {audit}')
"
```

Expected output (canonical is `cpt_id=10` because it's the smallest data-bearing id):
- `plan length: 1`
- One absorbed cpt_id: 11
- `merged: clusters=1 records=1`
- `remaining cptreport rows: [(10,)]`
- `audit: [(1, 1, 'within_record')]`

- [ ] **Step 3: Commit**

```bash
cd /home/arr65/src/nzgd && git add nzgd/dedup/pass0_within_record.py && git commit -m "Add Pass 0: within-record CPT/SPT consolidation"
```

---

## Task 7: Add `within_record` and `field_plausibility_ranges` to config

**Files:**
- Modify: `nzgd/resources/config.yaml`

- [ ] **Step 1: Append config block**

Append the following to the END of `/home/arr65/src/nzgd/nzgd/resources/config.yaml`. The existing `deduplication:` block is updated in place: find the existing `deduplication:` line and add the two new sub-blocks underneath it (alongside `hash_pass`, `fuzzy_pass`, `calibration`, `output`). Specifically, locate the lines:

```yaml
deduplication:
  hash_pass:
    stream_chunk_size: 100000
```

After all the existing sub-blocks under `deduplication:` (the last one is `output:`), add the two new sub-blocks at the same indentation level (2 spaces) as `hash_pass`:

```yaml

  within_record:
    canonical_selector: nzgd.dedup.canonical_selectors.default_within_record_canonical

  field_plausibility_ranges:
    nzgdrecord:
      latitude:                                  [-47.5, -33.5]
      longitude:                                 [165.0, 180.0]
      model_vs30_foster_2019_km_per_s:           [50.0, 2000.0]
      model_vs30_stddev_foster_2019_km_per_s:    [0.0, 10.0]
      model_gwl_westerhoff_2018_m:               [0.0, 100.0]
      model_gwl_nlm_2025_m:                      [0.0, 100.0]
      model_gwl_nlm_2025_stddev_m:               [0.0, 100.0]
    cptreport:
      extracted_gwl_m:     [0.01, 50.0]
      tip_net_area_ratio:  [0.5, 1.0]
      predrill_depth_m:    [0.0, 30.0]
    sptreport:
      extracted_gwl_m:     [0.01, 50.0]
```

- [ ] **Step 2: Verify it loads**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
from nzgd import constants
from nzgd.dedup.plausibility import is_useful_value
# Config loaded
assert constants.DEDUP_CONFIG['within_record']['canonical_selector'].endswith('.default_within_record_canonical')
assert constants.DEDUP_CONFIG['field_plausibility_ranges']['cptreport']['extracted_gwl_m'] == [0.01, 50.0]
# Plausibility helper now uses ranges
assert is_useful_value(0, 'cptreport', 'extracted_gwl_m') is False  # below range
assert is_useful_value(5.2, 'cptreport', 'extracted_gwl_m') is True
assert is_useful_value(-1.0, 'nzgdrecord', 'latitude') is False  # outside NZ bounds
assert is_useful_value(-41.0, 'nzgdrecord', 'latitude') is True
print('config + plausibility OK')
"
```

Expected: `config + plausibility OK`.

- [ ] **Step 3: Commit**

```bash
cd /home/arr65/src/nzgd && git add nzgd/resources/config.yaml && git commit -m "Config: add within_record block and field_plausibility_ranges"
```

---

## Task 8: Wire Pass 0 into the CLI

**Files:**
- Modify: `nzgd/scripts/db/deduplicate.py`

- [ ] **Step 1: Add Pass 0 import and invocation**

Edit `/home/arr65/src/nzgd/nzgd/scripts/db/deduplicate.py`. Add this import alongside the existing dedup imports:

```python
from nzgd.dedup.pass0_within_record import (
    apply_within_record_consolidation_plan,
    generate_within_record_consolidation_plan,
)
```

In the `for cfg, skip in ((CPT_TABLE_CONFIG, skip_cpt), (SPT_TABLE_CONFIG, skip_spt)):` loop, find the existing line `typer.echo(f"[{cfg.record_type}] Pass 1: hash ...")`. Insert before it:

```python
        typer.echo(f"[{cfg.record_type}] Pass 0: within-record consolidation ...")
        pass0_thresholds = {
            "trace_score_max": constants.DEDUP_CONFIG["fuzzy_pass"]["trace_score_max"],
            "trace_resample_step_m": constants.DEDUP_CONFIG["fuzzy_pass"]["trace_resample_step_m"],
        }
        within_plan = generate_within_record_consolidation_plan(conn, cfg, pass0_thresholds)
        c0, r0 = apply_within_record_consolidation_plan(
            conn, within_plan, run_id, cfg, failures=all_failures,
        )
        typer.echo(f"[{cfg.record_type}] Pass 0: absorbed {r0} rows across {c0} clusters.")
```

Also update the running totals lower in the function. Find:

```python
        total_clusters += c1 + c2
        total_records += r1 + r2
```

Replace with:

```python
        total_clusters += c0 + c1 + c2
        total_records += r0 + r1 + r2
```

- [ ] **Step 2: Verify the module still imports cleanly**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -c "from nzgd.scripts.db.deduplicate import app; print('CLI module ok')"
```

Expected: `CLI module ok`.

- [ ] **Step 3: Commit**

```bash
cd /home/arr65/src/nzgd && git add nzgd/scripts/db/deduplicate.py && git commit -m "CLI: invoke Pass 0 before Pass 1 for each record type"
```

---

## Task 9: Update test fixture + add first integration test

**Files:**
- Modify: `tests/dedup/conftest.py`
- Modify: `tests/dedup/test_dedup_pipeline.py`

- [ ] **Step 1: Remove `cpt_data_duplicate_of_cpt_id` from conftest's schema SQL**

Edit `/home/arr65/src/nzgd/tests/dedup/conftest.py`. In `_FULL_SCHEMA_SQL`, find the line:

```
    cpt_data_duplicate_of_cpt_id INTEGER,
```

Delete it (and any trailing comma syntax it leaves behind — the next line is `did_explicit_unit_conversion INTEGER,` which already has a leading comma if needed; verify by reading the resulting block).

- [ ] **Step 2: Append the first Pass 0 integration test**

Append the following to `/home/arr65/src/nzgd/tests/dedup/test_dedup_pipeline.py`:

```python


# === Pass 0 (within-record consolidation) scenarios ===


def _run_pass0(conn: sqlite3.Connection, cfg) -> tuple[int, int]:
    from nzgd.dedup.pass0_within_record import (
        apply_within_record_consolidation_plan,
        generate_within_record_consolidation_plan,
    )
    run_id = _start_run(conn)
    thresholds = {"trace_score_max": 0.05, "trace_resample_step_m": 0.05}
    plan = generate_within_record_consolidation_plan(conn, cfg, thresholds)
    return apply_within_record_consolidation_plan(conn, plan, run_id, cfg)


def test_pass0_typical_multi_sheet_collapse(fresh_db: sqlite3.Connection) -> None:
    """5 cptreport rows from 2 stems, all data-bearing rows match → 1 surviving row."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0), (0.3, 1.2, 0.012, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=8920, lat=-43.54, lon=172.66)
    add_cpt_report(fresh_db, cpt_id=11533, nzgd_id=8920, trace=[],    source_file="CPT_8920_AGS01.ags_sheet_0")
    add_cpt_report(fresh_db, cpt_id=11534, nzgd_id=8920, trace=[],    source_file="CPT_8920_AGS01.ags_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11535, nzgd_id=8920, trace=[],    source_file="CPT_8920_Raw01.xlsx_sheet_0")
    add_cpt_report(fresh_db, cpt_id=11536, nzgd_id=8920, trace=trace, source_file="CPT_8920_Raw01.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11537, nzgd_id=8920, trace=[],    source_file="CPT_8920_Raw01.xlsx_sheet_Header")

    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (1, 4)
    # Only canonical remains
    remaining = [r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport ORDER BY cpt_id")]
    assert remaining == [11536]
    # Audit row
    audit = fresh_db.execute(
        "SELECT canonical_nzgd_id, merged_nzgd_id, match_pass, report_pairs_json FROM dedup_audit"
    ).fetchall()
    assert len(audit) == 1
    assert audit[0][0] == 8920 and audit[0][1] == 8920 and audit[0][2] == "within_record"
    absorbed = json.loads(audit[0][3])
    assert len(absorbed) == 4
    assert {a["absorbed_report_id"] for a in absorbed} == {11533, 11534, 11535, 11537}
```

Note: `add_cpt_report` already accepts `source_file` as a keyword argument per the existing helper in `conftest.py`.

- [ ] **Step 3: Run the new test**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_pass0_typical_multi_sheet_collapse -v
```

Expected: 1 passed.

- [ ] **Step 4: Run the full test suite to confirm no regressions**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -v
```

Expected: 11 tests pass (10 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
cd /home/arr65/src/nzgd && git add tests/dedup/conftest.py tests/dedup/test_dedup_pipeline.py && git commit -m "Tests: drop legacy column from fixture; first Pass 0 scenario"
```

---

## Task 10: Update ORM (remove legacy field)

**Files:**
- Modify: `nzgd/db/orm.py`

- [ ] **Step 1: Remove the field definition**

Edit `/home/arr65/src/nzgd/nzgd/db/orm.py`. Delete the line (currently at line 375):

```python
    cpt_data_duplicate_of_cpt_id = IntegerField(null=True)
```

- [ ] **Step 2: Verify the ORM still imports**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "from nzgd.db import orm; print('orm imports ok')"
```

Expected: `orm imports ok`.

- [ ] **Step 3: Commit**

```bash
cd /home/arr65/src/nzgd && git add nzgd/db/orm.py && git commit -m "ORM: remove cpt_data_duplicate_of_cpt_id field from cptreport"
```

---

## Task 11: Remove extraction-time within-record dedup

**Files:**
- Modify: `nzgd/extract/cpt/workflow.py`
- Modify: `nzgd/extract/cpt/conditioning.py`
- Modify: `nzgd/extract/cpt/output.py`

- [ ] **Step 1: Remove the call from `workflow.py`**

Edit `/home/arr65/src/nzgd/nzgd/extract/cpt/workflow.py`. Find the block at line ~65:

```python
    extractions_no_duplicates = conditioning.remove_duplicate_extractions(
        conditioned_extractions,
    )
    output.write_extracted_data(extractions_no_duplicates)
```

Replace with:

```python
    output.write_extracted_data(conditioned_extractions)
```

- [ ] **Step 2: Remove `remove_duplicate_extractions` from `conditioning.py`**

Edit `/home/arr65/src/nzgd/nzgd/extract/cpt/conditioning.py`. Delete the entire function definition starting at `def remove_duplicate_extractions(` (line ~123) through its closing return — the whole function body. Search for any other code in the file that still references `removed_duplicates` and remove it too. Use:

```bash
grep -n "removed_duplicates\|remove_duplicate_extractions" /home/arr65/src/nzgd/nzgd/extract/cpt/conditioning.py
```

After removal, the grep should return no output.

- [ ] **Step 3: Remove `removed_duplicates` handling from `output.py`**

Edit `/home/arr65/src/nzgd/nzgd/extract/cpt/output.py`. Find the block around line 50:

```python
            if extraction_result.removed_duplicates:
                extraction_result.removed_duplicates = "___".join(
                    extraction_result.removed_duplicates,
                )
            ...
            extraction_result.data_df.loc[
                :, "removed_duplicates"
            ] = extraction_result.removed_duplicates
```

Delete every line that references `removed_duplicates`. Verify:

```bash
grep -n "removed_duplicates" /home/arr65/src/nzgd/nzgd/extract/cpt/output.py
```

Should return no output.

- [ ] **Step 4: Verify other modules don't reference the removed function**

```bash
grep -rn "remove_duplicate_extractions\|removed_duplicates" /home/arr65/src/nzgd/nzgd/ /home/arr65/src/nzgd/tests/ 2>/dev/null
```

Expected: no output (or only matches inside docs/superpowers/ which we ignore).

- [ ] **Step 5: Run the dedup test suite (confirms our removals didn't break the dedup module)**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -v
```

Expected: all 11 tests still pass.

- [ ] **Step 6: Commit**

```bash
cd /home/arr65/src/nzgd && git add nzgd/extract/cpt/workflow.py nzgd/extract/cpt/conditioning.py nzgd/extract/cpt/output.py && git commit -m "Remove extraction-time within-record dedup (replaced by Pass 0)"
```

---

## Task 12: Simplify DB ingest script

**Files:**
- Modify: `nzgd/scripts/db/put_cpts_in_db.py`

- [ ] **Step 1: Remove the `cpt_data_duplicate_of_cpt_id` variable and its keeper/duplicate branching**

Edit `/home/arr65/src/nzgd/nzgd/scripts/db/put_cpts_in_db.py`. Make these specific changes:

1. Delete the initialization line (around line 44):

   ```python
           cpt_data_duplicate_of_cpt_id = None
   ```

2. Delete the entire `else:` branch (around lines 99-118) that begins:

   ```python
               else:
                   removed_duplicates = cpt_data_df["removed_duplicates"].unique()
                   ...
                   cpt_data_duplicate_of_cpt_id = int(...)
   ```

   This whole `else` block goes away (it was the duplicate-row branch).

3. Modify the INSERT statement (around line 121-128). Find:

   ```python
           cursor.execute(
               """
               INSERT OR REPLACE INTO cptreport (
                   cpt_id, nzgd_id, max_depth_m, min_depth_m, extracted_gwl_m, gwl_method_id,
                   tip_net_area_ratio, predrill_depth_m, termination_reason_id, has_cpt_data, cpt_data_duplicate_of_cpt_id,
                   did_explicit_unit_conversion, did_inferred_unit_conversion, source_file
               )
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               """,
   ```

   Replace with (drop the column from the column list and the `?` count from 14 to 13):

   ```python
           cursor.execute(
               """
               INSERT OR REPLACE INTO cptreport (
                   cpt_id, nzgd_id, max_depth_m, min_depth_m, extracted_gwl_m, gwl_method_id,
                   tip_net_area_ratio, predrill_depth_m, termination_reason_id, has_cpt_data,
                   did_explicit_unit_conversion, did_inferred_unit_conversion, source_file
               )
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               """,
   ```

4. In the VALUES tuple that follows the INSERT, remove the `cpt_data_duplicate_of_cpt_id,` parameter (line ~141). The tuple goes from 14 entries to 13.

- [ ] **Step 2: Verify the script still imports**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -c "from nzgd.scripts.db.put_cpts_in_db import serialize_cpt_reports; print('put_cpts ok')"
```

Expected: `put_cpts ok`.

- [ ] **Step 3: Verify no references remain anywhere**

```bash
grep -rn "cpt_data_duplicate_of_cpt_id" /home/arr65/src/nzgd/nzgd/ /home/arr65/src/nzgd/tests/ 2>/dev/null
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
cd /home/arr65/src/nzgd && git add nzgd/scripts/db/put_cpts_in_db.py && git commit -m "DB ingest: drop cpt_data_duplicate_of_cpt_id from INSERT; remove duplicate branch"
```

---

## Task 13: Add the remaining 12 integration test scenarios

**Files:**
- Modify: `tests/dedup/test_dedup_pipeline.py`

- [ ] **Step 1: Append the 12 remaining scenarios**

Append the following to the end of `/home/arr65/src/nzgd/tests/dedup/test_dedup_pipeline.py`:

```python


def test_pass0_sentinel_aware_enrichment(fresh_db: sqlite3.Connection) -> None:
    """Canonical has gwl=0 (sentinel); absorbed has gwl=5.2 (real). Canonical gets updated."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_Header")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m = 0 WHERE cpt_id = 10")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m = 5.2 WHERE cpt_id = 11")

    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (1, 1)
    gwl = fresh_db.execute("SELECT extracted_gwl_m FROM cptreport WHERE cpt_id = 10").fetchone()[0]
    assert gwl == 5.2
    copied = fresh_db.execute("SELECT metadata_copied_json FROM dedup_audit").fetchone()[0]
    assert copied is not None
    assert "extracted_gwl_m" in json.loads(copied)


def test_pass0_plausibility_conflict_records_all(fresh_db: sqlite3.Connection) -> None:
    """Canonical NULL; two absorbed with conflicting in-range values; smallest-id wins, both logged."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_H1")
    add_cpt_report(fresh_db, cpt_id=12, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_H2")
    # Canonical (10) has NULL tip_net_area_ratio (default); absorbed 11 has 0.80, absorbed 12 has 0.92
    fresh_db.execute("UPDATE cptreport SET tip_net_area_ratio = 0.80 WHERE cpt_id = 11")
    fresh_db.execute("UPDATE cptreport SET tip_net_area_ratio = 0.92 WHERE cpt_id = 12")

    _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    chosen = fresh_db.execute("SELECT tip_net_area_ratio FROM cptreport WHERE cpt_id = 10").fetchone()[0]
    assert chosen == 0.80  # smallest absorbed id wins
    audit = fresh_db.execute("SELECT metadata_copied_json, metadata_conflicts_json FROM dedup_audit").fetchone()
    conflicts = json.loads(audit[1])
    assert "tip_net_area_ratio" in conflicts
    values = {e["value"] for e in conflicts["tip_net_area_ratio"]}
    assert values == {0.80, 0.92}


def test_pass0_multi_cpt_per_nzgd_stays_separate(fresh_db: sqlite3.Connection) -> None:
    """Two stems with non-matching data-bearing traces → 2 surviving rows."""
    trace_a = [(d, 1.0, 0.01, 0.0) for d in [0.1, 0.2, 0.3]]
    trace_b = [(d, 50.0, 0.5, 0.0) for d in [0.1, 0.2, 0.3]]  # 50x qc
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace_a, source_file="X.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=1, trace=trace_b, source_file="Y.xlsx_sheet_Data")
    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    # Both reports are singletons in their own clusters → no consolidation
    assert n_clusters == 0
    remaining = sorted(r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport"))
    assert remaining == [10, 20]


def test_pass0_all_no_data_cluster(fresh_db: sqlite3.Connection) -> None:
    """3 rows in one stem, all has_cpt_data=0 → 1 surviving row (smallest id)."""
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=[], source_file="A.xlsx_sheet_0")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=[], source_file="A.xlsx_sheet_H1")
    add_cpt_report(fresh_db, cpt_id=12, nzgd_id=1, trace=[], source_file="A.xlsx_sheet_H2")
    # All three have has_cpt_data=0 (no measurements via add_cpt_report with empty trace)
    fresh_db.execute("UPDATE cptreport SET has_cpt_data = 0")
    # Sprinkle metadata across the three so we can verify merge
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m = 3.0 WHERE cpt_id = 12")
    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (1, 2)
    remaining = [r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport")]
    assert remaining == [10]
    gwl = fresh_db.execute("SELECT extracted_gwl_m FROM cptreport WHERE cpt_id = 10").fetchone()[0]
    assert gwl == 3.0


def test_pass0_stem_only_attachment(fresh_db: sqlite3.Connection) -> None:
    """Stem A: 1 data + 2 no-data; stem B: 1 no-data → stem A cluster collapses, stem B singleton survives."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_H1")
    add_cpt_report(fresh_db, cpt_id=12, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_H2")
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=1, trace=[],    source_file="B.ags_sheet_0")
    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (1, 2)  # cluster A absorbs 11, 12; B is a singleton, not touched
    remaining = sorted(r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport"))
    assert remaining == [10, 20]


def test_pass0_fuzzy_within_record_match(fresh_db: sqlite3.Connection) -> None:
    """Two data-bearing rows from different stems with slightly perturbed traces consolidate via fuzzy."""
    trace_a = [(d, 1.0 + 0.1 * d, 0.01 + 0.001 * d, 0.0) for d in [0.1, 0.2, 0.3, 0.4]]
    trace_b = [(d, 1.001 + 0.1 * d, 0.0101 + 0.001 * d, 0.0) for d in [0.1, 0.2, 0.3, 0.4]]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace_a, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=1, trace=trace_b, source_file="B.ags_sheet_0")
    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (1, 1)
    remaining = [r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport")]
    assert remaining == [10]


def test_pass0_then_pass1_cross_record(fresh_db: sqlite3.Connection) -> None:
    """Two nzgd_ids each with within-record duplicates, and they are also cross-record duplicates."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0, investigation_name="Site A")
    add_cpt_record(fresh_db, nzgd_id=2, lat=-41.0, lon=174.0001, investigation_name="Site A")
    # nzgd 1: data + header
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_H")
    # nzgd 2: data + header (same trace as nzgd 1's data)
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=2, trace=trace, source_file="B.ags_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=21, nzgd_id=2, trace=[],    source_file="B.ags_sheet_H")

    run_id = _start_run(fresh_db)
    thresholds = {"trace_score_max": 0.05, "trace_resample_step_m": 0.05}
    from nzgd.dedup.pass0_within_record import (
        apply_within_record_consolidation_plan,
        generate_within_record_consolidation_plan,
    )
    plan0 = generate_within_record_consolidation_plan(fresh_db, CPT_TABLE_CONFIG, thresholds)
    apply_within_record_consolidation_plan(fresh_db, plan0, run_id, CPT_TABLE_CONFIG)
    plan1 = generate_hash_merge_plan(fresh_db, CPT_TABLE_CONFIG)
    apply_merge_plan(fresh_db, plan1, run_id, CPT_TABLE_CONFIG)

    # nzgd 2 merged into nzgd 1; only canonical's cpt_id remains
    merged = fresh_db.execute("SELECT merged_into_nzgd_id FROM nzgdrecord WHERE nzgd_id = 2").fetchone()[0]
    assert merged == 1
    remaining = sorted(r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport"))
    assert remaining == [10]
    audit_passes = sorted(r[0] for r in fresh_db.execute("SELECT match_pass FROM dedup_audit"))
    assert "within_record" in audit_passes and "hash" in audit_passes


def test_pass0_schema_migration_drops_legacy_column(fresh_db: sqlite3.Connection, tmp_path) -> None:
    """A DB carrying the legacy column gets it dropped by apply_dedup_schema."""
    import sqlite3 as _sqlite3
    from nzgd.dedup.schema import apply_dedup_schema

    db_path = tmp_path / "legacy.db"
    conn = _sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
        CREATE TABLE nzgdrecord (nzgd_id INTEGER PRIMARY KEY);
        CREATE TABLE cptreport (
            cpt_id INTEGER PRIMARY KEY, nzgd_id INTEGER, source_file TEXT NOT NULL,
            has_cpt_data INTEGER NOT NULL DEFAULT 1,
            cpt_data_duplicate_of_cpt_id INTEGER
        );
        INSERT INTO cptreport (cpt_id, nzgd_id, source_file, cpt_data_duplicate_of_cpt_id)
            VALUES (1, 100, 'x.xlsx_sheet_0', 99);
    """)
    apply_dedup_schema(conn)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(cptreport)")]
    assert "cpt_data_duplicate_of_cpt_id" not in cols
    # Surviving cptreport data is intact
    n = conn.execute("SELECT COUNT(*) FROM cptreport").fetchone()[0]
    assert n == 1
    conn.close()


def test_pass0_spt_consolidation_cascade(fresh_db: sqlite3.Connection) -> None:
    """SPT within-record consolidation: deletion cascades through soilmeasurements, etc."""
    trace = [(1.0, 5, 5, 5), (2.0, 7, 7, 7), (3.0, 10, 10, 10)]
    add_bh_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_spt_report(fresh_db, spt_id=100, nzgd_id=1, trace=trace, source_file="X.ags_sheet_0")
    add_spt_report(fresh_db, spt_id=101, nzgd_id=1, trace=[],    source_file="X.ags_sheet_H")
    # Dependent rows on the soon-to-be-absorbed row (101)
    fresh_db.execute("INSERT INTO soilmeasurements (soil_measurement_id, spt_id, top_depth_m, bottom_depth_m) VALUES (5000, 101, 0, 1)")
    fresh_db.execute("INSERT INTO densitymeasurements (density_measurement_id, spt_id, top_depth_m, bottom_depth_m, density_keyword) VALUES (6000, 101, 0, 1, 'loose')")
    fresh_db.execute("INSERT INTO soilmeasurementsoiltype (soil_measurement_id, soil_type_id) VALUES (5000, 1)")

    n_clusters, n_records = _run_pass0(fresh_db, SPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (1, 1)
    # All dependents of spt_id 101 are gone
    assert fresh_db.execute("SELECT COUNT(*) FROM sptreport WHERE spt_id = 101").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM soilmeasurements WHERE spt_id = 101").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM densitymeasurements WHERE spt_id = 101").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM soilmeasurementsoiltype WHERE soil_measurement_id = 5000").fetchone()[0] == 0


def test_pass0_single_file_multi_cpt_split(fresh_db: sqlite3.Connection) -> None:
    """One stem with 2 data-bearing rows with non-matching traces + 2 no-data rows → 2 sub-clusters."""
    trace_a = [(d, 1.0, 0.01, 0.0) for d in [0.1, 0.2, 0.3]]
    trace_b = [(d, 50.0, 0.5, 0.0) for d in [0.1, 0.2, 0.3]]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    # All four rows share one source-file stem ('X.xlsx')
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace_a, source_file="X.xlsx_sheet_DataA")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=trace_b, source_file="X.xlsx_sheet_DataB")
    add_cpt_report(fresh_db, cpt_id=12, nzgd_id=1, trace=[],      source_file="X.xlsx_sheet_H1")
    add_cpt_report(fresh_db, cpt_id=13, nzgd_id=1, trace=[],      source_file="X.xlsx_sheet_H2")
    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    # Two sub-clusters: smallest-id sub-cluster (containing 10) gets 12, 13 (no-data); other sub-cluster is singleton {11}
    assert n_clusters == 1  # only the sub-cluster with no-data rows needs consolidation
    assert n_records == 2
    remaining = sorted(r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport"))
    assert remaining == [10, 11]


def test_cross_record_plausibility_aware_enrichment(fresh_db: sqlite3.Connection) -> None:
    """Cross-record hash match: canonical's latitude out-of-range → replaced by absorbed's plausible value."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    # Canonical has latitude outside NZ bounds
    add_cpt_record(fresh_db, nzgd_id=1, lat=-1.0, lon=174.0, investigation_name="Site A")
    # Absorbed has plausible latitude
    add_cpt_record(fresh_db, nzgd_id=2, lat=-41.0, lon=174.0, investigation_name=None)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=2, trace=trace, source_file="B.xlsx_sheet_Data")
    run_id = _start_run(fresh_db)
    plan = generate_hash_merge_plan(fresh_db, CPT_TABLE_CONFIG)
    apply_merge_plan(fresh_db, plan, run_id, CPT_TABLE_CONFIG)
    # Canonical's lat is now -41.0 (overwritten because -1.0 is out-of-range)
    lat = fresh_db.execute("SELECT latitude FROM nzgdrecord WHERE merged_into_nzgd_id IS NULL").fetchone()[0]
    assert lat == -41.0
    audit = fresh_db.execute("SELECT metadata_copied_json FROM dedup_audit").fetchone()[0]
    assert audit is not None and "latitude" in json.loads(audit)


def test_pass0_sentinel_preserved_when_no_useful_alternative(fresh_db: sqlite3.Connection) -> None:
    """gwl=0 on canonical AND on absorbed; no useful value anywhere → canonical's 0 stays (not NULL'd)."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace, source_file="A.xlsx_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=11, nzgd_id=1, trace=[],    source_file="A.xlsx_sheet_H")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m = 0 WHERE cpt_id IN (10, 11)")
    _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    gwl = fresh_db.execute("SELECT extracted_gwl_m FROM cptreport WHERE cpt_id = 10").fetchone()[0]
    assert gwl == 0  # preserved, not NULL'd
    copied = fresh_db.execute("SELECT metadata_copied_json FROM dedup_audit").fetchone()[0]
    if copied:
        assert "extracted_gwl_m" not in json.loads(copied)
```

- [ ] **Step 2: Run all tests**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -v
```

Expected: 23 tests pass (10 cross-record + 13 Pass 0 / cross-record-plausibility).

- [ ] **Step 3: Commit**

```bash
cd /home/arr65/src/nzgd && git add tests/dedup/test_dedup_pipeline.py && git commit -m "Tests: add 12 Pass 0 + plausibility scenarios"
```

---

## Task 14: Real-data validation run

**Files:** none (manual command + observation)

- [ ] **Step 1: Remove stale dedup outputs from the previous run**

```bash
rm -f /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p6p0_20260403_deduped.db \
      /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/dedup_report.csv \
      /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/cpt_calibration_report.csv \
      /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/bh_calibration_report.csv \
      /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/dedup_failures.csv
ls /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/*deduped* 2>/dev/null || echo "clean"
```

Expected: `clean`.

- [ ] **Step 2: Run the full CLI against the production DB**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -m nzgd.scripts.db.deduplicate \
    --source /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p6p0_20260403.db \
    > /tmp/dedup_run_pass0.log 2>&1
echo "EXIT_CODE=$?"
```

Expected: `EXIT_CODE=0`. Runtime ~10–15 minutes (Pass 0 adds time over the previous 7-minute run).

- [ ] **Step 3: Inspect Pass 0 cluster/record counts**

```bash
grep -E "Pass [012]:|Done" /tmp/dedup_run_pass0.log
```

Expected: four lines per record type. For CPT: Pass 0 (within-record consolidation) likely absorbs ~70k records given the 71,873 no-data rows + 3,453 dup-flagged rows. For BH: smaller.

- [ ] **Step 4: Compare cross-record merge counts to the previous deduped DB**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python <<'PY'
import sqlite3
conn = sqlite3.connect("/home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p6p0_20260403_deduped.db")
print("=== Merge counts by (record_type, match_pass) ===")
for r in conn.execute(
    "SELECT record_type, match_pass, COUNT(*) FROM dedup_audit "
    "GROUP BY record_type, match_pass ORDER BY record_type, match_pass"
):
    print(" ", r)
print("=== Plausibility-driven changes (cross-record metadata_copied_json non-empty entries) ===")
n = conn.execute(
    "SELECT COUNT(*) FROM dedup_audit "
    "WHERE match_pass IN ('hash', 'fuzzy') AND metadata_copied_json IS NOT NULL"
).fetchone()[0]
print(f"  cross-record audit rows with copied metadata: {n}")
PY
```

Compare to the previous deduped DB's stats. Any cross-record rows that gained entries in `metadata_copied_json` are due to the new plausibility rule. Spot-check 5–10 of them by querying the source DB to confirm the new outcome is correct (i.e., the canonical's original value was implausible and the absorbed record's value is plausible).

- [ ] **Step 5: Inspect a Pass 0 audit sample**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python <<'PY'
import sqlite3, json
conn = sqlite3.connect("/home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p6p0_20260403_deduped.db")
rows = conn.execute(
    "SELECT canonical_nzgd_id, report_pairs_json, metadata_copied_json "
    "FROM dedup_audit WHERE match_pass = 'within_record' "
    "ORDER BY canonical_nzgd_id LIMIT 5"
).fetchall()
for canon, pairs, copied in rows:
    print(f"nzgd_id={canon}: {len(json.loads(pairs))} absorbed; copied={copied}")
PY
```

Verify the absorbed counts and copied-metadata entries look reasonable.

- [ ] **Step 6: Document the run**

If the plausibility rule produced cross-record outcomes that differ from the previous run AND those differences are correct, no code change is needed — the new behavior is the intended improvement. If thresholds need tuning (e.g., a plausibility range is too narrow and excludes legitimate values), edit `nzgd/resources/config.yaml`, delete the deduped DB, and re-run.

- [ ] **Step 7: Commit (only if `config.yaml` thresholds were adjusted)**

If you tuned the plausibility ranges:

```bash
cd /home/arr65/src/nzgd && git add nzgd/resources/config.yaml && git commit -m "Tune plausibility ranges based on real-data validation"
```

---

## Notes

- The `dedup_run.config_snapshot_json` captures the entire `DEDUP_CONFIG` dict at run time, including the new `within_record` and `field_plausibility_ranges` blocks. Every audit row can be traced back to the exact ranges in effect for that run.
- Pass 0's within-record consolidation does NOT modify `nzgdrecord` rows. Cross-record passes still own all changes to `nzgdrecord` (canonical/merged links, metadata enrichment).
- For records that have a *single* cptreport row (no consolidation needed), Pass 0 is a no-op — no audit row written, no change.
- The CLI refuses to overwrite an existing target DB; to re-run with different config, delete the target first.
- Pre-existing `cpt_data_duplicate_of_cpt_id` values in the production DB are silently lost when the column is dropped. They're not needed: every report previously flagged as a within-record duplicate will be absorbed by Pass 0 anyway.
