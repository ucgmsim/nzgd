# CPT Constant-Column Quality Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pre-dedup data-quality filter that discards any CPT report whose depth/qc/fs/u2 column holds a single constant value, recording each discard in an audit table.

**Architecture:** A new `nzgd/dedup/quality_filter.py` exposes `find_constant_column_reports` (one SQL `GROUP BY … HAVING` per record type) and `apply_quality_filter` (deletes each offending report via the existing `executor.delete_report`, writes a `quality_reject` audit row, one SAVEPOINT per report). `deduplicate.py` invokes it first in its per-record-type loop, before Pass 0, gated by config and a new `--skip-quality-filter` flag. Config-driven columns; CPT only.

**Tech Stack:** Python 3.12, SQLite (`sqlite3`), typer CLI, pytest. No new dependencies.

## Global Constraints

- **Python / tests:** use `/home/arr65/venvs/dev_nzgd_venv/bin/python` for everything (the project `.venv` lacks deps). Run tests with `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest`.
- **Style:** ruff, numpy docstring convention, type hints throughout.
- **Import style (dedup package):** the `nzgd/dedup/` package imports names directly (`from nzgd.dedup.executor import delete_report`, `from nzgd.dedup.data_types import TableConfig`). Match that local convention in new dedup files. In `deduplicate.py`, match its existing `from nzgd.dedup.X import func` imports.
- **Scope:** CPT only. SPT/borehole untouched. Extraction pipeline untouched.
- **"Constant" definition:** a measurement column with **≥ `min_non_null_rows` non-null values that are all exactly equal** (`COUNT(DISTINCT col) = 1`). Exact equality, no tolerance. An all-NULL column is never constant.
- **Discard unit:** one `cptreport` (one trace), physically deleted with its `cptmeasurements` + `cptvs30estimates` via `executor.delete_report`. The source DB is never modified (dedup runs on a copy).
- **Approved config values:** columns `["depth_m", "qc_MPa", "fs_MPa", "u2_MPa"]`, `min_non_null_rows: 3`, `enabled_record_types: ["CPT"]`. Under these, a full run discards ~5,222 CPT reports (5,214 constant-u2, 8 fs, 1 qc, 0 depth).
- **Spec:** `docs/superpowers/specs/2026-07-03-cpt-constant-column-quality-filter-design.md`.

---

### Task 1: Foundations — `quality_reject` table and `QualityRejectEntry`

**Files:**
- Modify: `nzgd/dedup/schema.py` (add table DDL + indexes; call them in `apply_dedup_schema`)
- Modify: `nzgd/dedup/data_types.py` (add dataclass)

**Interfaces:**
- Consumes: existing `apply_dedup_schema(conn)`.
- Produces:
  - Table `quality_reject(reject_id, run_id, record_type, nzgd_id, report_id, reason, constant_columns_json, n_rows, rejected_at)` created idempotently by `apply_dedup_schema`.
  - `QualityRejectEntry(record_type: str, nzgd_id: int, report_id: int, reason: str, constant_columns: dict[str, float], n_rows: int)` — frozen dataclass in `nzgd/dedup/data_types.py`.

This task is pure scaffolding (DDL + a frozen dataclass) with no behavior of its own; it is exercised end-to-end by Task 3's test. Verification here is a smoke check rather than a committed test.

- [ ] **Step 1: Add the dataclass**

In `nzgd/dedup/data_types.py`, append after `MergePlanEntry`:

```python
@dataclass(frozen=True)
class QualityRejectEntry:
    """One CPT report discarded by the constant-column quality filter."""

    record_type: str
    nzgd_id: int
    report_id: int
    reason: str                          # 'constant_column'
    constant_columns: dict[str, float]   # {column_name: constant_value}
    n_rows: int
```

- [ ] **Step 2: Add the table DDL and indexes to `schema.py`**

In `nzgd/dedup/schema.py`, after the `_INDEX_AUDIT_CLUSTER` definition (near line 47), add:

```python
_CREATE_QUALITY_REJECT = """
CREATE TABLE IF NOT EXISTS quality_reject (
    reject_id             INTEGER PRIMARY KEY,
    run_id                INTEGER NOT NULL REFERENCES dedup_run(run_id),
    record_type           TEXT NOT NULL,
    nzgd_id               INTEGER NOT NULL,
    report_id             INTEGER NOT NULL,
    reason                TEXT NOT NULL,
    constant_columns_json TEXT NOT NULL,
    n_rows                INTEGER NOT NULL,
    rejected_at           TEXT NOT NULL
)
"""

_INDEX_QUALITY_REJECT_RUN = "CREATE INDEX IF NOT EXISTS idx_quality_reject_run ON quality_reject(run_id)"
_INDEX_QUALITY_REJECT_NZGD = "CREATE INDEX IF NOT EXISTS idx_quality_reject_nzgd_id ON quality_reject(nzgd_id)"
```

- [ ] **Step 3: Create the table inside `apply_dedup_schema`**

In `nzgd/dedup/schema.py`, inside `apply_dedup_schema`, immediately before the final `conn.commit()` (after the legacy-column migration block near line 175), add:

```python
    cur.execute(_CREATE_QUALITY_REJECT)
    cur.execute(_INDEX_QUALITY_REJECT_RUN)
    cur.execute(_INDEX_QUALITY_REJECT_NZGD)
```

- [ ] **Step 4: Smoke-verify schema + dataclass**

Run:
```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python - <<'PY'
import sqlite3
from nzgd.dedup.schema import apply_dedup_schema
from nzgd.dedup.data_types import QualityRejectEntry
c = sqlite3.connect(":memory:")
c.execute("PRAGMA foreign_keys = ON")
c.executescript("CREATE TABLE nzgdrecord (nzgd_id INTEGER PRIMARY KEY); "
                "CREATE TABLE cptreport (cpt_id INTEGER PRIMARY KEY, nzgd_id INTEGER, source_file TEXT NOT NULL, has_cpt_data INTEGER NOT NULL DEFAULT 1);")
apply_dedup_schema(c)
apply_dedup_schema(c)  # idempotent: must not raise
cols = [r[1] for r in c.execute("PRAGMA table_info(quality_reject)")]
print("quality_reject cols:", cols)
assert "constant_columns_json" in cols and "report_id" in cols
e = QualityRejectEntry("CPT", 1, 10, "constant_column", {"u2_MPa": 0.0}, 3)
print("dataclass ok:", e)
print("SMOKE OK")
PY
```
Expected: prints the column list, `dataclass ok`, and `SMOKE OK`, with no exception.

- [ ] **Step 5: Commit**

```bash
git add nzgd/dedup/schema.py nzgd/dedup/data_types.py
git commit -m "feat(dedup): add quality_reject table and QualityRejectEntry"
```

---

### Task 2: Detection — `find_constant_column_reports`

**Files:**
- Create: `nzgd/dedup/quality_filter.py`
- Test: `tests/dedup/test_quality_filter.py`

**Interfaces:**
- Consumes: `QualityRejectEntry`, `TableConfig`, `CPT_TABLE_CONFIG` from `nzgd.dedup.data_types`.
- Produces:
  `find_constant_column_reports(conn: sqlite3.Connection, table_cfg: TableConfig, columns: list[str], min_non_null_rows: int) -> list[QualityRejectEntry]`
  — returns one entry per report that has at least one constant column among `columns`; `constant_columns` maps each offending column to its constant value; does not mutate the DB.

- [ ] **Step 1: Write the failing test**

Create `tests/dedup/test_quality_filter.py`:

```python
"""Integration tests for the constant-column quality filter."""

import sqlite3

import pytest

from nzgd.dedup.data_types import CPT_TABLE_CONFIG
from nzgd.dedup.quality_filter import find_constant_column_reports
from tests.dedup.conftest import add_cpt_record, add_cpt_report

_COLUMNS = ["depth_m", "qc_MPa", "fs_MPa", "u2_MPa"]


def _populate_scenarios(conn: sqlite3.Connection) -> None:
    add_cpt_record(conn, nzgd_id=1)
    # R1 (10): everything varies -> kept
    add_cpt_report(conn, 10, 1, [(0.1, 1.0, 0.010, 0.05),
                                 (0.2, 1.1, 0.011, 0.06),
                                 (0.3, 1.2, 0.012, 0.07)])
    # R2 (11): constant qc -> discard
    add_cpt_report(conn, 11, 1, [(0.1, 2.0, 0.010, 0.05),
                                 (0.2, 2.0, 0.011, 0.06),
                                 (0.3, 2.0, 0.012, 0.07)])
    # R3 (12): constant u2 = 0 with good qc/fs -> discard (validates u2 inclusion)
    add_cpt_report(conn, 12, 1, [(0.1, 1.0, 0.010, 0.0),
                                 (0.2, 1.1, 0.011, 0.0),
                                 (0.3, 1.2, 0.012, 0.0)])
    # R4 (13): constant fs -> discard
    add_cpt_report(conn, 13, 1, [(0.1, 1.0, 0.02, 0.05),
                                 (0.2, 1.1, 0.02, 0.06),
                                 (0.3, 1.2, 0.02, 0.07)])
    # R5 (14): constant qc but only 2 rows (< min_non_null_rows) -> kept
    add_cpt_report(conn, 14, 1, [(0.1, 3.0, 0.02, 0.05),
                                 (0.2, 3.0, 0.03, 0.06)])


def test_find_constant_column_reports(fresh_db: sqlite3.Connection) -> None:
    _populate_scenarios(fresh_db)
    entries = find_constant_column_reports(fresh_db, CPT_TABLE_CONFIG, _COLUMNS, 3)
    by_id = {e.report_id: e for e in entries}
    assert set(by_id) == {11, 12, 13}
    assert by_id[11].constant_columns == {"qc_MPa": 2.0}
    assert by_id[12].constant_columns == {"u2_MPa": 0.0}
    assert by_id[13].constant_columns == {"fs_MPa": 0.02}
    assert by_id[11].nzgd_id == 1
    assert by_id[11].n_rows == 3
    assert by_id[11].reason == "constant_column"


def test_invalid_column_raises(fresh_db: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        find_constant_column_reports(fresh_db, CPT_TABLE_CONFIG, ["not_a_column"], 3)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_quality_filter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nzgd.dedup.quality_filter'`.

- [ ] **Step 3: Write the implementation**

Create `nzgd/dedup/quality_filter.py`:

```python
"""Constant-column data-quality filter: discard CPT reports with a flat channel.

A physically real CPT varies with depth in every channel. A measurement column
that holds a single repeated value is a broken extraction or an unmeasured-channel
placeholder, so the whole report is discarded. This runs before the dedup passes
so a flat report cannot be chosen as canonical or pollute fuzzy matching.
"""

import sqlite3

from nzgd.dedup.data_types import QualityRejectEntry, TableConfig


def find_constant_column_reports(
    conn: sqlite3.Connection,
    table_cfg: TableConfig,
    columns: list[str],
    min_non_null_rows: int,
) -> list[QualityRejectEntry]:
    """Return reports that have a constant value in at least one of `columns`.

    A column is constant when it has ``>= min_non_null_rows`` non-null values that
    are all equal (``COUNT(DISTINCT col) = 1``). An all-NULL column is never
    constant. The DB is not modified.

    Parameters
    ----------
    conn
        Connection to the (deduped-target) DB.
    table_cfg
        Table configuration for the record type (only CPT is used in practice).
    columns
        Measurement columns to test; each must be in
        ``table_cfg.measurement_value_columns``.
    min_non_null_rows
        Minimum non-null values a column must have before it can be judged
        constant.

    Returns
    -------
    list[QualityRejectEntry]
        One entry per offending report; ``constant_columns`` maps each offending
        column to its constant value.
    """
    valid = set(table_cfg.measurement_value_columns)
    invalid = [c for c in columns if c not in valid]
    if invalid:
        raise ValueError(
            f"quality_filter columns {invalid} are not measurement columns of "
            f"{table_cfg.record_type}: {sorted(valid)}"
        )
    if not columns:
        return []

    report_id_col = table_cfg.report_id_column
    select_terms = [f"m.{report_id_col}", "r.nzgd_id", "COUNT(*) AS n_rows"]
    for i, col in enumerate(columns):
        select_terms.append(f"COUNT({col}) AS nn{i}")
        select_terms.append(f"COUNT(DISTINCT {col}) AS d{i}")
        select_terms.append(f"MIN({col}) AS v{i}")
    having_terms = [
        f"(COUNT({col}) >= ? AND COUNT(DISTINCT {col}) = 1)" for col in columns
    ]
    sql = (
        f"SELECT {', '.join(select_terms)} "
        f"FROM {table_cfg.measurement_table} m "
        f"JOIN {table_cfg.report_table} r ON r.{report_id_col} = m.{report_id_col} "
        f"GROUP BY m.{report_id_col}, r.nzgd_id "
        f"HAVING {' OR '.join(having_terms)}"
    )

    entries: list[QualityRejectEntry] = []
    for row in conn.execute(sql, [min_non_null_rows] * len(columns)).fetchall():
        report_id, nzgd_id, n_rows = row[0], row[1], row[2]
        constant_columns: dict[str, float] = {}
        for i, col in enumerate(columns):
            nn, distinct, value = row[3 + i * 3], row[4 + i * 3], row[5 + i * 3]
            if nn >= min_non_null_rows and distinct == 1:
                constant_columns[col] = value
        entries.append(
            QualityRejectEntry(
                record_type=table_cfg.record_type,
                nzgd_id=nzgd_id,
                report_id=report_id,
                reason="constant_column",
                constant_columns=constant_columns,
                n_rows=n_rows,
            )
        )
    return entries
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_quality_filter.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add nzgd/dedup/quality_filter.py tests/dedup/test_quality_filter.py
git commit -m "feat(dedup): detect CPT reports with a constant measurement column"
```

---

### Task 3: Discard + report — `apply_quality_filter` and `write_quality_filter_report`

**Files:**
- Modify: `nzgd/dedup/quality_filter.py` (add `apply_quality_filter`)
- Modify: `nzgd/dedup/reports.py` (add `write_quality_filter_report`)
- Test: `tests/dedup/test_quality_filter.py` (add discard + report test)

**Interfaces:**
- Consumes: `find_constant_column_reports` (Task 2); `delete_report` from `nzgd.dedup.executor`; the `quality_reject` table (Task 1).
- Produces:
  - `apply_quality_filter(conn, entries: list[QualityRejectEntry], run_id: int, table_cfg: TableConfig, failures: list[dict] | None = None) -> int` — deletes each report and writes a `quality_reject` row, one SAVEPOINT per report; commits; returns the number discarded. On per-report failure, rolls back that report and appends a dict (keys: `cluster_id`, `canonical_nzgd_id`, `merged_nzgd_ids`, `record_type`, `error`) to `failures` for the shared failures report.
  - `write_quality_filter_report(conn, run_id: int, path: Path) -> None` — flattens `quality_reject` rows for the run into a CSV.

- [ ] **Step 1: Write the failing test**

Extend the **top** import block of `tests/dedup/test_quality_filter.py` so imports stay at the top of the file (do not append imports below the existing functions — that trips ruff E402). `json` is added here because this is where it is first used; `apply_quality_filter` is added to the existing `from nzgd.dedup.quality_filter import ...` line. The block becomes:

```python
"""Integration tests for the constant-column quality filter."""

import json
import sqlite3

import pytest

from nzgd.dedup.data_types import CPT_TABLE_CONFIG
from nzgd.dedup.quality_filter import apply_quality_filter, find_constant_column_reports
from nzgd.dedup.reports import write_quality_filter_report
from nzgd.dedup.schema import apply_dedup_schema
from tests.dedup.conftest import add_cpt_record, add_cpt_report
```

Then **append** the helper and test function to the end of the file:

```python
def _start_run(conn: sqlite3.Connection) -> int:
    apply_dedup_schema(conn)
    cur = conn.execute(
        "INSERT INTO dedup_run (started_at, source_db_path, script_version, config_snapshot_json) "
        "VALUES (?, ?, ?, ?)",
        ("2026-07-03T00:00:00Z", ":memory:", "test", "{}"),
    )
    return cur.lastrowid


def test_apply_quality_filter_discards_and_audits(fresh_db, tmp_path):
    _populate_scenarios(fresh_db)
    run_id = _start_run(fresh_db)
    failures: list[dict] = []
    entries = find_constant_column_reports(fresh_db, CPT_TABLE_CONFIG, _COLUMNS, 3)
    n = apply_quality_filter(fresh_db, entries, run_id, CPT_TABLE_CONFIG, failures=failures)

    assert n == 3
    assert failures == []
    remaining = sorted(r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport"))
    assert remaining == [10, 14]  # R1 and R5 kept; R2/R3/R4 discarded
    gone = fresh_db.execute(
        "SELECT COUNT(*) FROM cptmeasurements WHERE cpt_id IN (11, 12, 13)"
    ).fetchone()[0]
    assert gone == 0
    kept = fresh_db.execute(
        "SELECT COUNT(*) FROM cptmeasurements WHERE cpt_id = 10"
    ).fetchone()[0]
    assert kept == 3

    rej = fresh_db.execute(
        "SELECT report_id, reason, constant_columns_json, n_rows "
        "FROM quality_reject ORDER BY report_id"
    ).fetchall()
    assert [r[0] for r in rej] == [11, 12, 13]
    assert all(r[1] == "constant_column" for r in rej)
    assert json.loads(rej[1][2]) == {"u2_MPa": 0.0}

    out = tmp_path / "qf.csv"
    write_quality_filter_report(fresh_db, run_id, out)
    text = out.read_text()
    assert "report_id" in text and "constant_columns" in text
    assert "u2_MPa" in text and "constant_column" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_quality_filter.py::test_apply_quality_filter_discards_and_audits -v`
Expected: FAIL — `ImportError: cannot import name 'apply_quality_filter'` (and `write_quality_filter_report`).

- [ ] **Step 3: Implement `apply_quality_filter`**

First extend the imports at the top of `nzgd/dedup/quality_filter.py` so the block reads:

```python
import json
import sqlite3
from datetime import datetime, timezone

from nzgd.dedup.data_types import QualityRejectEntry, TableConfig
from nzgd.dedup.executor import delete_report
```

Then append the function:

```python
def apply_quality_filter(
    conn: sqlite3.Connection,
    entries: list[QualityRejectEntry],
    run_id: int,
    table_cfg: TableConfig,
    failures: list[dict] | None = None,
) -> int:
    """Delete each report in `entries` and record it in `quality_reject`.

    Each report is deleted in its own SAVEPOINT (via ``executor.delete_report``),
    so one failure only rolls back that report. Failures are appended to
    `failures` (shared with the dedup failures report). Returns the number of
    reports successfully discarded.
    """
    if not entries:
        return 0
    cur = conn.cursor()
    n_discarded = 0
    for entry in entries:
        savepoint = f"quality_reject_{entry.report_id}"
        cur.execute(f"SAVEPOINT {savepoint}")
        try:
            delete_report(conn, entry.report_id, table_cfg)
            cur.execute(
                "INSERT INTO quality_reject (run_id, record_type, nzgd_id, report_id, "
                "reason, constant_columns_json, n_rows, rejected_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    entry.record_type,
                    entry.nzgd_id,
                    entry.report_id,
                    entry.reason,
                    json.dumps(entry.constant_columns),
                    entry.n_rows,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            n_discarded += 1
        except Exception as exc:  # noqa: BLE001 — one bad report must not abort the rest
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            if failures is not None:
                # A discard is not a merge: no records are merged, so
                # merged_nzgd_ids is empty. The offending report's id is a
                # report_id (cpt_id), a different ID space from nzgd_id, so it
                # is carried in the error text rather than in an nzgd_id column.
                failures.append(
                    {
                        "cluster_id": None,
                        "canonical_nzgd_id": entry.nzgd_id,
                        "merged_nzgd_ids": [],
                        "record_type": table_cfg.record_type,
                        "error": f"quality-filter discard of report_id={entry.report_id} failed: {exc!r}",
                    }
                )
    conn.commit()
    return n_discarded
```

- [ ] **Step 4: Implement `write_quality_filter_report`**

In `nzgd/dedup/reports.py`, append (the module already imports `csv`, `json`, `sqlite3`, and `Path`):

```python
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_quality_filter.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 6: Lint the changed files (including the test file)**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m ruff check nzgd/dedup/quality_filter.py nzgd/dedup/reports.py tests/dedup/test_quality_filter.py`
Expected: `All checks passed!` — in particular no `F401` (unused import) or `E402` (import not at top) in the test file. Fix any issue and re-run.

- [ ] **Step 7: Commit**

```bash
git add nzgd/dedup/quality_filter.py nzgd/dedup/reports.py tests/dedup/test_quality_filter.py
git commit -m "feat(dedup): discard constant-column CPT reports and write audit report"
```

---

### Task 4: Config + driver wiring in `deduplicate.py`

**Files:**
- Modify: `nzgd/resources/config.yaml` (add `quality_filter` block + output filename)
- Modify: `nzgd/scripts/db/deduplicate.py` (imports, `--skip-quality-filter`, invoke first in loop, write report)
- Test: `tests/dedup/test_quality_filter.py` (end-to-end pipeline test)

**Interfaces:**
- Consumes: `find_constant_column_reports`, `apply_quality_filter` (Tasks 2–3); `write_quality_filter_report` (Task 3); `constants.DEDUP_CONFIG`.
- Produces: the `deduplicate` CLI runs the quality filter first for each enabled record type, honors `--skip-quality-filter`, and writes `quality_filter_report.csv` to the target directory.

- [ ] **Step 1: Add config**

In `nzgd/resources/config.yaml`, under `deduplication:`, add a new block (place it after the `within_record:` block, before `field_plausibility_ranges:`):

```yaml
  quality_filter:
    # Discard any report with a single constant value in one of these columns.
    enabled_record_types: ["CPT"]
    constant_columns:
      CPT: ["depth_m", "qc_MPa", "fs_MPa", "u2_MPa"]
    min_non_null_rows: 3
```

And add one line to the existing `output:` block:

```yaml
    quality_filter_report_filename: "quality_filter_report.csv"
```

- [ ] **Step 2: Write the failing end-to-end test**

Add these three imports to the **top** import block of `tests/dedup/test_quality_filter.py` (keep imports at the top — do not append them below the functions, which trips ruff E402):

```python
from typer.testing import CliRunner

from nzgd.scripts.db.deduplicate import app
from tests.dedup.conftest import _make_fresh_db
```

(`sqlite3`, `Path`, `add_cpt_record`, and `add_cpt_report` are already imported at the top from earlier tasks.) Then **append** the test function to the end of the file (note the type-annotated signature — the project's ruff `ANN` rules require it and are not test-exempt):

```python
def test_quality_filter_runs_in_full_pipeline(tmp_path: Path) -> None:
    src = tmp_path / "source.db"
    conn = _make_fresh_db(src)
    add_cpt_record(conn, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_record(conn, nzgd_id=2, lat=-41.0, lon=174.0)
    # normal report -> survives
    add_cpt_report(conn, 10, 1, [(0.1, 1.0, 0.010, 0.05),
                                 (0.2, 1.1, 0.011, 0.06),
                                 (0.3, 1.2, 0.012, 0.07)])
    # constant-qc report -> discarded by the filter before any dedup pass
    add_cpt_report(conn, 20, 2, [(0.1, 2.0, 0.010, 0.05),
                                 (0.2, 2.0, 0.011, 0.06),
                                 (0.3, 2.0, 0.012, 0.07)])
    conn.commit()
    conn.close()

    target = tmp_path / "deduped.db"
    result = CliRunner().invoke(
        app, ["--source", str(src), "--target", str(target), "--skip-spt"]
    )
    assert result.exit_code == 0, result.output

    out = sqlite3.connect(target)
    try:
        remaining = [r[0] for r in out.execute("SELECT cpt_id FROM cptreport ORDER BY cpt_id")]
        assert remaining == [10]
        rej = out.execute("SELECT report_id, record_type, reason FROM quality_reject").fetchall()
        assert rej == [(20, "CPT", "constant_column")]
    finally:
        out.close()
    assert (tmp_path / "quality_filter_report.csv").exists()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_quality_filter.py::test_quality_filter_runs_in_full_pipeline -v`
Expected: FAIL — the constant-qc report (cpt_id 20) still exists (`remaining == [10, 20]`) and/or `quality_reject` is empty, because the filter is not wired into `main` yet.

- [ ] **Step 4: Add imports to `deduplicate.py`**

In `nzgd/scripts/db/deduplicate.py`, add to the dedup imports:

```python
from nzgd.dedup.quality_filter import (
    apply_quality_filter,
    find_constant_column_reports,
)
```

and add `write_quality_filter_report` to the existing `from nzgd.dedup.reports import (...)` group.

- [ ] **Step 5: Add the `--skip-quality-filter` option**

In the `main` signature, after the `skip_spt` option, add:

```python
    skip_quality_filter: bool = typer.Option(
        False, "--skip-quality-filter", help="Skip the constant-column quality filter."
    ),
```

- [ ] **Step 6: Read config and invoke the filter first in the loop**

In `main`, after the `within_enabled = set(...)` block (near line 149), add:

```python
    qf_cfg = constants.DEDUP_CONFIG.get("quality_filter", {})
    qf_enabled = set(qf_cfg.get("enabled_record_types", []))
    qf_columns = qf_cfg.get("constant_columns", {})
    qf_min_rows = qf_cfg.get("min_non_null_rows", 3)
```

Then inside the `for cfg, skip in (...)` loop, immediately after the `if skip: ... continue` guard and before the Pass 0 block, add:

```python
        if not skip_quality_filter and cfg.record_type in qf_enabled:
            typer.echo(f"[{cfg.record_type}] Quality filter: discarding constant-column reports ...")
            qf_entries = find_constant_column_reports(
                conn, cfg, qf_columns.get(cfg.record_type, []), qf_min_rows
            )
            n_qf = apply_quality_filter(conn, qf_entries, run_id, cfg, failures=all_failures)
            typer.echo(f"[{cfg.record_type}] Quality filter: discarded {n_qf} reports.")
```

- [ ] **Step 7: Write the report after the loop**

In `main`, after the supplemental-consolidation report block (near line 209, after `typer.echo(f"Supplemental consolidation report at {supp_report_path}.")`), add:

```python
    qf_report_path = out_dir / constants.DEDUP_CONFIG["output"]["quality_filter_report_filename"]
    write_quality_filter_report(conn, run_id, qf_report_path)
    typer.echo(f"Quality filter report at {qf_report_path}.")
```

- [ ] **Step 8: Run the end-to-end test to verify it passes**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_quality_filter.py -v`
Expected: PASS (all four tests, including `test_quality_filter_runs_in_full_pipeline`).

- [ ] **Step 9: Run the full dedup test suite (no regressions)**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/ -v`
Expected: PASS — all pre-existing tests plus the new ones.

- [ ] **Step 10: Lint**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m ruff check nzgd/dedup/quality_filter.py nzgd/dedup/schema.py nzgd/dedup/reports.py nzgd/dedup/data_types.py nzgd/scripts/db/deduplicate.py tests/dedup/test_quality_filter.py`
Expected: no errors. Fix any reported issues, then re-run.

- [ ] **Step 11: Commit**

```bash
git add nzgd/resources/config.yaml nzgd/scripts/db/deduplicate.py tests/dedup/test_quality_filter.py
git commit -m "feat(dedup): wire constant-column quality filter into the pipeline"
```

---

## Optional real-data smoke (not a committed test)

After the plan is implemented, optionally confirm behaviour on real data. This
copies and dedups a real DB (minutes; needs disk for the copy), so it is not part
of the automated suite:

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -m nzgd.scripts.db.deduplicate \
    --source /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p8p1_20260625.db \
    --target /tmp/qf_smoke_deduped.db --skip-spt
```
Expected: a `[CPT] Quality filter: discarded ~5222 reports.` line, and
`quality_filter_report.csv` next to the target listing them (dominated by
`u2_MPa`). Delete `/tmp/qf_smoke_deduped.db` afterwards.

---

## Self-Review

**Spec coverage:**
- Detection (spec §5.2) → Task 2. Discard mechanics + SAVEPOINT (§5.3) → Task 3. `quality_reject` table (§5.4) → Task 1. `QualityRejectEntry` (§5.5) → Task 1. Report writer (§5.6) → Task 3. Config (§5.7) → Task 4. CLI + first-in-loop placement + `--skip-quality-filter` (§5.8) → Task 4. Testing (§8) → Tasks 2–4. All spec sections map to a task.
- Non-goals (§6): all-NULL not discarded — enforced by the `COUNT(DISTINCT)=1` definition (an all-NULL column has `COUNT(DISTINCT)=0`); no separate work needed. SPT untouched — config `enabled_record_types: ["CPT"]`. Extraction untouched — no extraction files modified.

**Placeholder scan:** No TBD/TODO; every code step contains complete code; every run step states the exact command and expected result.

**Type consistency:** `QualityRejectEntry` fields (`record_type`, `nzgd_id`, `report_id`, `reason`, `constant_columns`, `n_rows`) are produced by `find_constant_column_reports` (Task 2) and consumed by `apply_quality_filter` (Task 3) identically. `apply_quality_filter` and `find_constant_column_reports` signatures match their call sites in `deduplicate.py` (Task 4). `write_quality_filter_report(conn, run_id, path)` matches its call site. Table/column names in the DDL (Task 1), the INSERT (Task 3), and the report SELECT (Task 3) all agree.
