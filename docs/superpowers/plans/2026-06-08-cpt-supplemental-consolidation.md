# CPT Supplemental-Value Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the ~98.5% of "missing" CPT supplemental values that are already extracted but stranded on non-canonical sibling rows, by adding a post-dedup within-record consolidation step — plus two small filter-stage corrections (GWL negative-sentinel family; predrill `Nil`→0).

**Architecture:** A new `nzgd/dedup/supplemental_consolidation.py` runs once per CPT record-type in `deduplicate.py` after Pass 2. For each `nzgd_id` with ≥2 surviving `cptreport` rows it computes a best-available value per supplemental column (prefer the single in-range value; else the single recorded value; else NULL; small-spread valid conflicts resolved by a corroboration-first selector, large-spread skipped), then fills only NULL/non-useful cells. It reuses the existing `dedup.plausibility.is_useful_value` and the `dedup_audit` table. The two filter corrections live in `filter_potential_cpt_supplemental_values.py` (+ `constants.py`/`config.yaml`).

**Tech Stack:** Python 3, SQLite (`sqlite3` stdlib), pandas/numpy (filter), pytest (tests), Typer (existing CLI). Spec: `docs/superpowers/specs/2026-06-05-cpt-supplemental-propagation-design.md`. Evidence: `docs/gwl_zero_is_placeholder.md`.

**Run tests with:** `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -v` (the project venv; the repo `.venv` lacks deps).

---

## File Structure

- **Create:** `nzgd/dedup/supplemental_consolidation.py` — the selector + the consolidation function (one responsibility: post-dedup supplemental fill).
- **Modify:** `nzgd/dedup/schema.py` — widen `dedup_audit.match_pass` CHECK to add `'supplemental_consolidation'`.
- **Modify:** `nzgd/dedup/reports.py` — add `write_supplemental_consolidation_report`.
- **Modify:** `nzgd/scripts/db/deduplicate.py` — invoke the new step after Pass 2; write its report.
- **Modify:** `nzgd/resources/config.yaml` — add `within_record_supplemental.small_spread_threshold`, `gwl_no_water_sentinels`, and an output filename.
- **Modify:** `nzgd/constants.py` — expose `GWL_NO_WATER_SENTINELS`.
- **Modify:** `nzgd/scripts/extract/cpt/filter_potential_cpt_supplemental_values.py` — GWL sentinel drop (A); predrill `Nil`→0 (B); wrap module-level run in `main()`/`__main__`.
- **Modify:** `nzgd/scripts/extract/cpt/extract_all_potential_cpt_supplemental_values.py` is unaffected; **Modify** `constants.py` `term_dict` predrill entries to emit `nil` candidates (B).
- **Modify:** `tests/dedup/test_dedup_pipeline.py` — consolidation + selector + schema-migration integration tests.
- **Create:** `tests/extract/test_supplemental_filter_corrections.py` — focused tests for the two filter corrections.
- **Create:** `nzgd/scripts/db/validate_supplemental_consolidation.py` — the real-data categorized-diff-vs-Maxim measurement.

---

## Task 1: Config + constants foundation

**Files:**
- Modify: `nzgd/resources/config.yaml` (append under `deduplication:`, and a top-level sentinel key; output block)
- Modify: `nzgd/constants.py` (add `GWL_NO_WATER_SENTINELS`)

- [ ] **Step 1: Add the config blocks.** In `nzgd/resources/config.yaml`, under the existing `deduplication:` mapping (sibling of `within_record:` / `field_plausibility_ranges:`), add the threshold block, and add a report filename under the existing `output:` block:

```yaml
  output:
    deduped_db_suffix: "_deduped"
    report_filename: "dedup_report.csv"
    calibration_report_filename: "calibration_report.csv"
    failures_filename: "dedup_failures.csv"
    supplemental_consolidation_report_filename: "supplemental_consolidation_report.csv"

  within_record_supplemental:
    small_spread_threshold:
      predrill_depth_m:   0.5
      extracted_gwl_m:    0.5
      tip_net_area_ratio: 0.05
```

And add a top-level key (sibling of `known_missing_value_placeholders:`, NOT under `deduplication:`) — the GWL-only negative "no-water" sentinels:

```yaml
gwl_no_water_sentinels:
  - -30
  - -60
  - -100
```

- [ ] **Step 2: Expose the sentinel constant.** In `nzgd/constants.py`, near the existing GWL bound constants (after `MIN_ALLOWED_GWL = ...`), add:

```python
# GWL "no-water" sentinel defaults (e.g. RAW01.txt "Waterlevel: -30/-60/-100").
# np.abs() would otherwise fabricate 30/60/100 (see docs/gwl_zero_is_placeholder.md Part 2).
GWL_NO_WATER_SENTINELS = frozenset(float(v) for v in CONFIG["gwl_no_water_sentinels"])
```

- [ ] **Step 3: Verify config loads.** Run:

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "from nzgd import constants; print(sorted(constants.GWL_NO_WATER_SENTINELS)); print(constants.DEDUP_CONFIG['within_record_supplemental']['small_spread_threshold']); print(constants.DEDUP_CONFIG['output']['supplemental_consolidation_report_filename'])"
```

Expected: `[-100.0, -60.0, -30.0]`, the threshold dict, and `supplemental_consolidation_report.csv`.

- [ ] **Step 4: Commit.**

```bash
git add nzgd/resources/config.yaml nzgd/constants.py
git commit -m "config: add supplemental-consolidation thresholds and GWL no-water sentinels"
```

---

## Task 2: Widen the `dedup_audit.match_pass` CHECK constraint

The new step records audit rows with `match_pass='supplemental_consolidation'`, which the current CHECK (`IN ('hash','fuzzy','within_record')`) rejects. Mirror the existing idempotent table-recreate migration in `schema.py`.

**Files:**
- Modify: `nzgd/dedup/schema.py`
- Test: `tests/dedup/test_dedup_pipeline.py`

- [ ] **Step 1: Write the failing test.** Append to `tests/dedup/test_dedup_pipeline.py`:

```python
def test_supplemental_consolidation_match_pass_allowed(fresh_db: sqlite3.Connection) -> None:
    """After apply_dedup_schema, dedup_audit accepts match_pass='supplemental_consolidation'."""
    run_id = _start_run(fresh_db)  # applies schema (incl. the new migration)
    fresh_db.execute(
        "INSERT INTO dedup_audit (run_id, cluster_id, canonical_nzgd_id, merged_nzgd_id, "
        "record_type, match_pass, report_pairs_json, merged_at) VALUES (?,?,?,?,?,?,?,?)",
        (run_id, 1, 5, 5, "CPT", "supplemental_consolidation", "[]", "2026-06-08T00:00:00Z"),
    )
    got = fresh_db.execute(
        "SELECT match_pass FROM dedup_audit WHERE canonical_nzgd_id = 5"
    ).fetchone()[0]
    assert got == "supplemental_consolidation"
```

- [ ] **Step 2: Run to verify it fails.**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_supplemental_consolidation_match_pass_allowed -v`
Expected: FAIL with `sqlite3.IntegrityError: CHECK constraint failed`.

- [ ] **Step 3: Implement the migration.** In `nzgd/dedup/schema.py`, add (after the existing `_migrate_widen_audit_check`):

```python
_WIDENED_DEDUP_AUDIT_DDL_SUPP = """
CREATE TABLE dedup_audit_new (
    audit_id                INTEGER PRIMARY KEY,
    run_id                  INTEGER NOT NULL REFERENCES dedup_run(run_id),
    cluster_id              INTEGER NOT NULL,
    canonical_nzgd_id       INTEGER NOT NULL,
    merged_nzgd_id          INTEGER NOT NULL,
    record_type             TEXT NOT NULL CHECK(record_type IN ('CPT', 'BH')),
    match_pass              TEXT NOT NULL CHECK(match_pass IN ('hash', 'fuzzy', 'within_record', 'supplemental_consolidation')),
    report_pairs_json       TEXT NOT NULL,
    metadata_copied_json    TEXT,
    metadata_conflicts_json TEXT,
    merged_at               TEXT NOT NULL
)
"""


def _audit_check_includes_supplemental(conn: sqlite3.Connection) -> bool:
    """True if dedup_audit.match_pass CHECK already accepts 'supplemental_consolidation'."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='dedup_audit'"
    ).fetchone()
    if row is None:
        return False
    return "'supplemental_consolidation'" in row[0]


def _migrate_widen_audit_check_supplemental(conn: sqlite3.Connection) -> None:
    """Recreate dedup_audit accepting 'supplemental_consolidation'. Idempotent via caller's check."""
    cur = conn.cursor()
    cur.execute(_WIDENED_DEDUP_AUDIT_DDL_SUPP)
    cur.execute("INSERT INTO dedup_audit_new SELECT * FROM dedup_audit")
    cur.execute("DROP TABLE dedup_audit")
    cur.execute("ALTER TABLE dedup_audit_new RENAME TO dedup_audit")
    cur.execute(_INDEX_AUDIT_CANONICAL)
    cur.execute(_INDEX_AUDIT_MERGED)
    cur.execute(_INDEX_AUDIT_CLUSTER)
```

Then, inside `apply_dedup_schema`, immediately after the existing `if not _audit_check_includes_within_record(conn): _migrate_widen_audit_check(conn)` block, add:

```python
    if not _audit_check_includes_supplemental(conn):
        _migrate_widen_audit_check_supplemental(conn)
```

- [ ] **Step 4: Run to verify it passes.**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_supplemental_consolidation_match_pass_allowed -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add nzgd/dedup/schema.py tests/dedup/test_dedup_pipeline.py
git commit -m "dedup: widen dedup_audit.match_pass CHECK for supplemental_consolidation"
```

---

## Task 3: The selector (`select_value`)

**Files:**
- Create: `nzgd/dedup/supplemental_consolidation.py`
- Test: `tests/dedup/test_dedup_pipeline.py`

- [ ] **Step 1: Write the failing tests.** Append to `tests/dedup/test_dedup_pipeline.py`:

```python
def test_select_value_mode_wins() -> None:
    from nzgd.dedup.supplemental_consolidation import select_value
    # 0.8 appears twice, 0.75 once -> mode 0.8
    assert select_value([(0.8, 10), (0.8, 11), (0.75, 12)]) == 0.8


def test_select_value_tiebreak_decimals_then_cpt_id() -> None:
    from nzgd.dedup.supplemental_consolidation import select_value
    # tie on count (1 each); 0.75 has more decimals than 0.8 -> 0.75
    assert select_value([(0.8, 10), (0.75, 11)]) == 0.75
    # tie on count AND decimals -> smallest cpt_id wins (1.2 @ id 5)
    assert select_value([(1.2, 7), (1.3, 5)]) == 1.3
```

- [ ] **Step 2: Run to verify it fails.**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -k select_value -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nzgd.dedup.supplemental_consolidation'`.

- [ ] **Step 3: Create the module with the selector.** Create `nzgd/dedup/supplemental_consolidation.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes.**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -k select_value -v`
Expected: PASS (3 assertions across 2 tests).

- [ ] **Step 5: Commit.**

```bash
git add nzgd/dedup/supplemental_consolidation.py tests/dedup/test_dedup_pipeline.py
git commit -m "dedup: add supplemental-consolidation value selector"
```

---

## Task 4: The consolidation function

**Files:**
- Modify: `nzgd/dedup/supplemental_consolidation.py`
- Test: `tests/dedup/test_dedup_pipeline.py`

- [ ] **Step 1: Add a Pass-0-style run helper to the tests.** Append to `tests/dedup/test_dedup_pipeline.py`:

```python
def _run_supp_consolidation(conn: sqlite3.Connection, cfg) -> tuple[int, int]:
    from nzgd.dedup.supplemental_consolidation import (
        consolidate_within_record_supplemental,
    )
    run_id = _start_run(conn)
    return consolidate_within_record_supplemental(conn, cfg, run_id)
```

- [ ] **Step 2: Write the failing integration tests.** Append to `tests/dedup/test_dedup_pipeline.py` (covers spec scenarios 1–8):

```python
def test_supp_cross_cluster_fill(fresh_db: sqlite3.Connection) -> None:
    """record-3 shape: .ags trace row NULL, .xls sibling has predrill+GWL -> .ags filled."""
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=3)
    add_cpt_report(fresh_db, cpt_id=13, nzgd_id=3, trace=trace, source_file="CPT_3_AGS01.ags_sheet_0")
    add_cpt_report(fresh_db, cpt_id=16, nzgd_id=3, trace=trace, source_file="CPT_3_AGS01.xls_sheet_TabulatedData")
    fresh_db.execute("UPDATE cptreport SET predrill_depth_m=0.8, extracted_gwl_m=1.3, gwl_method_id=1 WHERE cpt_id=16")

    records, cells = _run_supp_consolidation(fresh_db, CPT_TABLE_CONFIG)
    assert records == 1
    row = fresh_db.execute(
        "SELECT predrill_depth_m, extracted_gwl_m, gwl_method_id FROM cptreport WHERE cpt_id=13"
    ).fetchone()
    assert row == (0.8, 1.3, 1)


def test_supp_gwl_zero_preserved_when_only_value(fresh_db: sqlite3.Connection) -> None:
    """Lone GWL 0 (no better alternative) is preserved, not nulled, and fills the NULL sibling."""
    trace = [(0.1, 1.0, 0.01, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=5)
    add_cpt_report(fresh_db, cpt_id=1, nzgd_id=5, trace=trace, source_file="a.ags_sheet_0")
    add_cpt_report(fresh_db, cpt_id=2, nzgd_id=5, trace=trace, source_file="a.xls_sheet_Data")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m=0 WHERE cpt_id=2")

    _run_supp_consolidation(fresh_db, CPT_TABLE_CONFIG)
    vals = [r[0] for r in fresh_db.execute("SELECT extracted_gwl_m FROM cptreport ORDER BY cpt_id")]
    assert vals == [0.0, 0.0]


def test_supp_gwl_zero_overridden_by_positive(fresh_db: sqlite3.Connection) -> None:
    """A positive in-range sibling overrides a non-useful 0."""
    trace = [(0.1, 1.0, 0.01, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=6)
    add_cpt_report(fresh_db, cpt_id=1, nzgd_id=6, trace=trace, source_file="a.ags_sheet_0")
    add_cpt_report(fresh_db, cpt_id=2, nzgd_id=6, trace=trace, source_file="a.xls_sheet_Data")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m=0 WHERE cpt_id=1")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m=1.5 WHERE cpt_id=2")

    _run_supp_consolidation(fresh_db, CPT_TABLE_CONFIG)
    vals = [r[0] for r in fresh_db.execute("SELECT extracted_gwl_m FROM cptreport ORDER BY cpt_id")]
    assert vals == [1.5, 1.5]


def test_supp_small_spread_conflict_selector(fresh_db: sqlite3.Connection) -> None:
    """0.75 vs 0.80 predrill (spread 0.05): NULL canonical filled with selected 0.75; sibling keeps 0.80."""
    trace = [(0.1, 1.0, 0.01, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=7)
    add_cpt_report(fresh_db, cpt_id=1, nzgd_id=7, trace=trace, source_file="a.ags_sheet_0")
    add_cpt_report(fresh_db, cpt_id=2, nzgd_id=7, trace=trace, source_file="a.xls_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=3, nzgd_id=7, trace=trace, source_file="a.txt_sheet_0")
    fresh_db.execute("UPDATE cptreport SET predrill_depth_m=0.75 WHERE cpt_id=2")
    fresh_db.execute("UPDATE cptreport SET predrill_depth_m=0.80 WHERE cpt_id=3")

    _run_supp_consolidation(fresh_db, CPT_TABLE_CONFIG)
    vals = {r[0]: r[1] for r in fresh_db.execute("SELECT cpt_id, predrill_depth_m FROM cptreport")}
    assert vals == {1: 0.75, 2: 0.75, 3: 0.80}


def test_supp_large_spread_conflict_skipped(fresh_db: sqlite3.Connection) -> None:
    """1.5 vs 22.0 GWL (spread 20.5 > 0.5): nothing changes; canonical stays NULL; conflict logged."""
    trace = [(0.1, 1.0, 0.01, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=8)
    add_cpt_report(fresh_db, cpt_id=1, nzgd_id=8, trace=trace, source_file="a.ags_sheet_0")
    add_cpt_report(fresh_db, cpt_id=2, nzgd_id=8, trace=trace, source_file="a.xls_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=3, nzgd_id=8, trace=trace, source_file="a.txt_sheet_0")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m=1.5 WHERE cpt_id=2")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m=22.0 WHERE cpt_id=3")

    _run_supp_consolidation(fresh_db, CPT_TABLE_CONFIG)
    vals = [r[0] for r in fresh_db.execute("SELECT extracted_gwl_m FROM cptreport ORDER BY cpt_id")]
    assert vals == [None, 1.5, 22.0]
    conflicts = fresh_db.execute(
        "SELECT metadata_conflicts_json FROM dedup_audit WHERE canonical_nzgd_id=8"
    ).fetchone()[0]
    assert conflicts is not None and "extracted_gwl_m" in json.loads(conflicts)


def test_supp_idempotent(fresh_db: sqlite3.Connection) -> None:
    """Running twice produces no further changes."""
    trace = [(0.1, 1.0, 0.01, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=9)
    add_cpt_report(fresh_db, cpt_id=1, nzgd_id=9, trace=trace, source_file="a.ags_sheet_0")
    add_cpt_report(fresh_db, cpt_id=2, nzgd_id=9, trace=trace, source_file="a.xls_sheet_Data")
    fresh_db.execute("UPDATE cptreport SET predrill_depth_m=1.2 WHERE cpt_id=2")
    _run_supp_consolidation(fresh_db, CPT_TABLE_CONFIG)
    records2, cells2 = _run_supp_consolidation(fresh_db, CPT_TABLE_CONFIG)
    assert cells2 == 0
```

- [ ] **Step 3: Run to verify they fail.**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -k supp_ -v`
Expected: FAIL with `ImportError: cannot import name 'consolidate_within_record_supplemental'`.

- [ ] **Step 4: Implement `consolidate_within_record_supplemental`.** Append to `nzgd/dedup/supplemental_consolidation.py`:

```python
def consolidate_within_record_supplemental(
    conn: sqlite3.Connection,
    table_cfg: TableConfig,
    run_id: int,
) -> tuple[int, int]:
    """Fill each nzgd_id's surviving rows with its best-available supplemental value.

    Returns (n_records_changed, n_cells_filled). Writes one dedup_audit row per
    affected nzgd_id (match_pass='supplemental_consolidation'), recording filled
    cells in metadata_copied_json and skipped conflicts in metadata_conflicts_json.
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
        except Exception:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
    conn.commit()
    return n_records_changed, n_cells_filled
```

- [ ] **Step 5: Run to verify they pass.**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -k supp_ -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Run the full dedup suite (no regressions).**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/ -q`
Expected: all pass.

- [ ] **Step 7: Commit.**

```bash
git add nzgd/dedup/supplemental_consolidation.py tests/dedup/test_dedup_pipeline.py
git commit -m "dedup: add within-record supplemental consolidation step"
```

---

## Task 5: The report writer

**Files:**
- Modify: `nzgd/dedup/reports.py`
- Test: `tests/dedup/test_dedup_pipeline.py`

- [ ] **Step 1: Write the failing test.** Append to `tests/dedup/test_dedup_pipeline.py`:

```python
def test_supplemental_consolidation_report(fresh_db: sqlite3.Connection, tmp_path) -> None:
    from nzgd.dedup.reports import write_supplemental_consolidation_report
    trace = [(0.1, 1.0, 0.01, 0.0)]
    add_cpt_record(fresh_db, nzgd_id=8)
    add_cpt_report(fresh_db, cpt_id=1, nzgd_id=8, trace=trace, source_file="a.ags_sheet_0")
    add_cpt_report(fresh_db, cpt_id=2, nzgd_id=8, trace=trace, source_file="a.xls_sheet_Data")
    add_cpt_report(fresh_db, cpt_id=3, nzgd_id=8, trace=trace, source_file="a.txt_sheet_0")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m=1.5 WHERE cpt_id=2")
    fresh_db.execute("UPDATE cptreport SET extracted_gwl_m=22.0 WHERE cpt_id=3")
    run_id = _start_run(fresh_db)
    from nzgd.dedup.supplemental_consolidation import consolidate_within_record_supplemental
    consolidate_within_record_supplemental(fresh_db, CPT_TABLE_CONFIG, run_id)

    out = tmp_path / "supp.csv"
    write_supplemental_consolidation_report(fresh_db, run_id, out)
    text = out.read_text()
    assert "nzgd_id" in text and "conflict_fields" in text
    assert "extracted_gwl_m" in text  # the skipped conflict is reported
```

- [ ] **Step 2: Run to verify it fails.**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_supplemental_consolidation_report -v`
Expected: FAIL with `ImportError: cannot import name 'write_supplemental_consolidation_report'`.

- [ ] **Step 3: Implement the writer.** Append to `nzgd/dedup/reports.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes.**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_supplemental_consolidation_report -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add nzgd/dedup/reports.py tests/dedup/test_dedup_pipeline.py
git commit -m "dedup: add supplemental_consolidation_report writer"
```

---

## Task 6: Wire the step into the dedup CLI

**Files:**
- Modify: `nzgd/scripts/db/deduplicate.py`

- [ ] **Step 1: Add imports.** In `nzgd/scripts/db/deduplicate.py`, extend the dedup imports block:

```python
from nzgd.dedup.reports import (
    write_calibration_report,
    write_dedup_report,
    write_failures_report,
    write_supplemental_consolidation_report,
)
from nzgd.dedup.supplemental_consolidation import (
    consolidate_within_record_supplemental,
)
```

- [ ] **Step 2: Invoke after Pass 2.** Inside the `for cfg, skip in (...)` loop, immediately after the Pass 2 `typer.echo(f"[{cfg.record_type}] Pass 2: merged ...")` line, add:

```python
        if cfg.record_type in within_enabled:
            typer.echo(f"[{cfg.record_type}] Supplemental consolidation ...")
            supp_records, supp_cells = consolidate_within_record_supplemental(conn, cfg, run_id)
            typer.echo(
                f"[{cfg.record_type}] Supplemental consolidation: filled {supp_cells} "
                f"cells across {supp_records} records."
            )
```

(Consolidation fills cells; it is not a record/cluster merge, so it is intentionally NOT added to `total_clusters` / `total_records`.)

- [ ] **Step 3: Write the report after the loop.** After the existing `write_dedup_report(conn, run_id, report_path)` call, add:

```python
    supp_report_path = out_dir / constants.DEDUP_CONFIG["output"]["supplemental_consolidation_report_filename"]
    write_supplemental_consolidation_report(conn, run_id, supp_report_path)
    typer.echo(f"Supplemental consolidation report at {supp_report_path}.")
```

- [ ] **Step 4: Smoke-test the CLI imports.** Run:

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "import nzgd.scripts.db.deduplicate as d; print('ok', hasattr(d, 'app'))"
```

Expected: `ok True` (module imports cleanly; no syntax/import errors).

- [ ] **Step 5: Commit.**

```bash
git add nzgd/scripts/db/deduplicate.py
git commit -m "dedup: run supplemental consolidation after Pass 2 in the CLI"
```

---

## Task 7: Filter correction (A) — GWL negative "no-water" sentinels

Drop GWL candidates whose raw value is `-30`/`-60`/`-100` **before** `np.abs()` fabricates `30`/`60`/`100`. Filter-stage only (re-runnable on the existing candidate CSV; no re-extraction). First make the filter importable without side effects.

**Files:**
- Modify: `nzgd/scripts/extract/cpt/filter_potential_cpt_supplemental_values.py`
- Test: `tests/extract/test_supplemental_filter_corrections.py` (create)

- [ ] **Step 1: Make the module importable.** In `filter_potential_cpt_supplemental_values.py`, wrap the module-level run block (the `all_options_df = pd.read_csv(...)` at lines ~569 through the final `extracted_df.to_csv(...)` at ~675) inside a function and a guard. Replace the top-level statements with:

```python
def main() -> None:
    all_options_df = pd.read_csv(
        constants.SUPPLEMENTAL_VALUES_OUTPUT_DIR
        / constants.ALL_POTENTIAL_CPT_SUPPLEMENTAL_VALUES_FILENAME,
    )
    # ... (the existing body verbatim, indented one level) ...
    extracted_df.to_csv(
        constants.SUPPLEMENTAL_VALUES_OUTPUT_DIR
        / constants.CPT_SUPPLEMENTAL_VALUES_FILENAME,
        index=False,
    )


if __name__ == "__main__":
    main()
```

Run `/home/arr65/venvs/dev_nzgd_venv/bin/python -c "import nzgd.scripts.extract.cpt.filter_potential_cpt_supplemental_values as m; print(hasattr(m, 'extract_numerical_value'))"` — Expected: `True` and it must NOT read/write CSVs (no file errors printed).

- [ ] **Step 2: Write the failing test.** Create `tests/extract/__init__.py` (empty) and `tests/extract/test_supplemental_filter_corrections.py`:

```python
from nzgd.scripts.extract.cpt.filter_potential_cpt_supplemental_values import (
    extract_numerical_value,
)


def test_gwl_negative_sentinels_dropped():
    for sentinel in ("-30.00", "-60", "-100.0"):
        assert extract_numerical_value(sentinel, check_for_cm=True, is_gwl=True) is None


def test_gwl_real_negative_below_ground_kept():
    # below-ground sign convention: -1.2 m -> 1.2 m (NOT a sentinel)
    assert extract_numerical_value("-1.2", check_for_cm=True, is_gwl=True) == 1.2


def test_non_gwl_unaffected_by_sentinel_rule():
    # predrill/other: -30 still becomes 30 (no GWL sentinel handling)
    assert extract_numerical_value("-30.00", check_for_cm=True, is_gwl=False) == 30.0
```

- [ ] **Step 3: Run to verify it fails.**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/extract/test_supplemental_filter_corrections.py -v`
Expected: FAIL with `TypeError: extract_numerical_value() got an unexpected keyword argument 'is_gwl'`.

- [ ] **Step 4: Implement the sentinel drop.** In `extract_numerical_value`, add the `is_gwl` parameter and the pre-`np.abs` check:

```python
def extract_numerical_value(
    s: str,
    check_for_cm: bool = False,
    is_gwl: bool = False,
) -> float | None:
    match = re.search(constants.NUMERICAL_VALUES_REGEX, s)
    if not match:
        return None

    raw = float(match.group())
    if is_gwl and raw in constants.GWL_NO_WATER_SENTINELS:
        # "no-water" template default; np.abs would fabricate 30/60/100.
        return None
    value = np.abs(raw)

    if check_for_cm:
        end_pos = match.end()
        remaining_text = s[end_pos:].strip()
        if remaining_text.lower().startswith("cm"):
            return value / 100.0
    return value
```

Then thread `is_gwl` from `extract_numerical_quantity`: just below the existing `check_for_cm = True` block (after line ~246), add:

```python
    is_gwl = quantity_to_extract == constants.QuantityToExtract.ground_water_level
```

and pass `is_gwl=is_gwl` to BOTH `extract_numerical_value(...)` call sites (the one taking the sliced `cell_contents[...]` and the one taking `cell_contents`).

- [ ] **Step 5: Run to verify it passes.**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/extract/test_supplemental_filter_corrections.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit.**

```bash
git add nzgd/scripts/extract/cpt/filter_potential_cpt_supplemental_values.py tests/extract/
git commit -m "filter: drop GWL -30/-60/-100 no-water sentinels before np.abs"
```

---

## Task 8: Filter correction (B) — predrill `Nil` → 0

Lowest priority; only takes effect after a full re-extraction (the extract stage must emit the `Nil` candidate). Two edits: emit the candidate (extract), map it to 0 (filter).

**Files:**
- Modify: `nzgd/constants.py` (predrill `term_dict` entries)
- Modify: `nzgd/scripts/extract/cpt/filter_potential_cpt_supplemental_values.py`
- Test: `tests/extract/test_supplemental_filter_corrections.py`

- [ ] **Step 1: Write the failing test.** Append to `tests/extract/test_supplemental_filter_corrections.py`:

```python
import pandas as pd
from nzgd import constants
from nzgd.scripts.extract.cpt.filter_potential_cpt_supplemental_values import (
    extract_numerical_quantity,
)


def test_predrill_nil_becomes_zero():
    df = pd.DataFrame([
        {"nzgd_id": 1, "file_name": "a.xls", "sheet_name": "s", "likely_orientation": "columns",
         "search_term": "predrill", "search_assumption": "assuming_cell_is_a_field_name_in_need_of_a_value",
         "assumed_orientation": "columns", "field_label": "Pre-Drill:", "value": "Nil"},
    ])
    result = extract_numerical_quantity(df, constants.QuantityToExtract.predrill_depth)
    assert result.predrill_depth == 0.0
```

(Adjust the input columns if the real `extract_numerical_quantity` signature/columns differ — verify against the function's expected `possible_values_df` columns before writing the assertion.)

- [ ] **Step 2: Run to verify it fails.**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/extract/test_supplemental_filter_corrections.py -k nil -v`
Expected: FAIL (the `Nil` row is dropped by the numeric filter → no value / `None`).

- [ ] **Step 3: Map `Nil`→0 in the predrill branch.** In `extract_numerical_quantity`, in the predrill branch (the `elif quantity_to_extract == constants.QuantityToExtract.predrill_depth:` block, ~line 239), before the numeric-filter at lines ~221-226 runs, normalise `Nil`. The cleanest placement is immediately after `possible_values_df` is available and before the numeric filter; add a predrill-gated replacement:

```python
    if quantity_to_extract == constants.QuantityToExtract.predrill_depth:
        nil_mask = possible_values_df["value"].astype(str).str.strip().str.lower() == "nil"
        possible_values_df.loc[nil_mask, "value"] = "0"
```

(Place this just before the `# Filter out rows that do not include a numerical value` block so the substituted `"0"` survives the numeric filter and extracts as `0.0`.)

- [ ] **Step 4: Emit `nil` candidates from the extract stage.** In `nzgd/constants.py`, in the three predrill `term_dict` entries (`"predrill"`, `"pre-drill"`, `"predrilled"`, lines ~494-508), add `"nil"` to the two non-empty pattern lists. For each entry change `[NUMERICAL_VALUES_REGEX]` → `[NUMERICAL_VALUES_REGEX, "nil"]` in both `assuming_cell_is_standalone` and `assuming_cell_is_a_field_name_in_need_of_a_value`. Example for `"predrill"`:

```python
    "predrill": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX, "nil"],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX, "nil"],
    },
```

- [ ] **Step 5: Run to verify it passes.**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/extract/test_supplemental_filter_corrections.py -k nil -v`
Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add nzgd/constants.py nzgd/scripts/extract/cpt/filter_potential_cpt_supplemental_values.py tests/extract/test_supplemental_filter_corrections.py
git commit -m "extract/filter: treat predrill Nil as 0 (no-predrilling)"
```

---

## Task 9: Real-data validation (categorized diff vs Maxim)

A standalone measurement script — NOT a pytest. Runs consolidation on a working copy of the no-fill deduped DB and categorises every difference vs the Maxim-filled DB (per the spec's success criterion). Maxim is a diagnostic, not ground truth.

**Files:**
- Create: `nzgd/scripts/db/validate_supplemental_consolidation.py`

- [ ] **Step 1: Write the script.** Create `nzgd/scripts/db/validate_supplemental_consolidation.py`:

```python
"""Validate within-record supplemental consolidation against Maxim's backfill.

Runs consolidation on a working COPY of the no-fill deduped DB, then categorises
every per-(cpt_id, field) difference vs the Maxim-filled deduped DB:
match / preserved-0 / intended-improvement / intended-difference (Maxim 0 from
Nil -> our NULL) / genuine-residual-gap / conflict. Read-only on the inputs.
"""

import shutil
import sqlite3
from collections import Counter
from pathlib import Path

from nzgd.dedup.data_types import CPT_TABLE_CONFIG
from nzgd.dedup.schema import apply_dedup_schema
from nzgd.dedup.supplemental_consolidation import consolidate_within_record_supplemental

DATA = Path("/home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data")
NOFILL = DATA / "uc_nzgd_v0p7p0_20260528_deduped_NO_FILL_WITH_MAXIM_VALUES.db"
MAXIM = DATA / "uc_nzgd_v0p7p0_20260528_deduped.db"
WORK = DATA / "supplemental_value_analysis" / "consolidation_validation_work.db"
FIELDS = ("predrill_depth_m", "extracted_gwl_m", "tip_net_area_ratio")


def main() -> None:
    shutil.copyfile(NOFILL, WORK)
    conn = sqlite3.connect(WORK)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_dedup_schema(conn)
    cur = conn.execute(
        "INSERT INTO dedup_run (started_at, source_db_path, script_version, config_snapshot_json) "
        "VALUES ('validate', ?, 'validate', '{}')",
        (str(NOFILL),),
    )
    run_id = cur.lastrowid
    conn.commit()
    recs, cells = consolidate_within_record_supplemental(conn, CPT_TABLE_CONFIG, run_id)
    print(f"consolidation: filled {cells} cells across {recs} records")

    ours = {}
    for r in conn.execute(f"SELECT cpt_id, {', '.join(FIELDS)} FROM cptreport"):
        for i, f in enumerate(FIELDS):
            ours[(r[0], f)] = r[i + 1]
    conn.close()

    mx = sqlite3.connect(f"file:{MAXIM}?mode=ro", uri=True)
    cats: dict[str, Counter] = {f: Counter() for f in FIELDS}
    gaps = []
    for r in mx.execute(f"SELECT cpt_id, {', '.join(FIELDS)} FROM cptreport"):
        cpt_id = r[0]
        for i, f in enumerate(FIELDS):
            maxim_v, our_v = r[i + 1], ours.get((cpt_id, f))
            if maxim_v is None:
                continue
            if our_v is not None and abs(float(our_v) - float(maxim_v)) <= 0.011:
                cats[f]["match"] += 1
            elif maxim_v == 0 and our_v is None:
                cats[f]["intended_difference (Maxim 0 -> our NULL)"] += 1
            elif our_v is not None and float(our_v) > 0 and float(maxim_v) == 0:
                cats[f]["intended_improvement (Maxim 0 -> our positive)"] += 1
            elif our_v is None:
                cats[f]["genuine_residual_gap"] += 1
                if len(gaps) < 50:
                    gaps.append((cpt_id, f, maxim_v))
            else:
                cats[f]["other_difference"] += 1
    mx.close()
    for f in FIELDS:
        print(f"\n{f}:")
        for k, n in sorted(cats[f].items()):
            print(f"  {k}: {n}")
    print(f"\nSample genuine residual gaps (investigate): {gaps[:20]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it.**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m nzgd.scripts.db.validate_supplemental_consolidation`
Expected: a per-field breakdown dominated by `match`, with `intended_difference`/`intended_improvement` buckets and a SMALL `genuine_residual_gap` bucket (~49 predrill `Nil` + a handful). Eyeball the listed gaps — each should be a value living only in a format/sheet we don't read, not a logic bug.

- [ ] **Step 3: Record findings + commit the script.** Note the per-field counts in the PR/commit body. Commit:

```bash
git add nzgd/scripts/db/validate_supplemental_consolidation.py
git commit -m "db: add supplemental-consolidation vs Maxim validation script"
```

- [ ] **Step 4: Decide Maxim's fate (deferred decision, now informed).** With the `genuine_residual_gap` bucket quantified, report to the user whether the Maxim backfill can be retired or kept as a fallback. (No code change in this task — it's the decision gate.)

---

## Self-Review notes

- **Spec coverage:** consolidation step (Tasks 3-5), CLI placement after Pass 2 (Task 6), B3 + corroboration selector + per-field thresholds (Tasks 1, 3, 4), fill-only-NULL/non-useful + gwl_method coupling (Task 4), GWL `0` preserve / override / `Nil`→NULL (Task 4 tests + the extractor never emitting GWL `Nil`), negative-sentinel family `-30/-60/-100` (Task 7), predrill `Nil`→0 tail (Task 8), audit + report (Tasks 2, 5), measurement vs Maxim as categorized diff (Task 9). The `22` free-text precision class is intentionally out of scope (surfaced via the Task 5 conflict report).
- **Re-extraction dependency:** Task 7 (sentinel drop) is filter-only and re-runnable on the existing candidate CSV; Task 8 (predrill `Nil`) needs a full re-extraction to emit candidates — flagged in the task.
- **Type/name consistency:** `consolidate_within_record_supplemental(conn, table_cfg, run_id) -> (int, int)`, `select_value(list[tuple[float,int]]) -> float`, `write_supplemental_consolidation_report(conn, run_id, path)`, `extract_numerical_value(s, check_for_cm, is_gwl)` are used identically across tasks.
- **Operational note for applying to the current DB:** Tasks 1-7 corrections require regenerating the supplemental CSV (re-run the filter `main()`) and rebuilding+re-deduping the DB to fully land; the standalone consolidation (Task 9) demonstrates the recovery on the existing deduped DB without a full rebuild. The regenerate-vs-patch operational choice is the user's at execution time.
