# CPT Emptied-Record Deletion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the constant-column quality filter discards CPT reports, delete any `nzgdrecord` it leaves with zero reports — before any merge pass, so merge tombstones are never touched.

**Architecture:** A new `delete_emptied_records` phase in `nzgd/dedup/quality_filter.py` runs immediately after `apply_quality_filter` in the driver loop. It takes the records this run's filter touched (distinct `nzgd_id`s in `quality_reject`), deletes each that now has zero reports **and** `merged_into_nzgd_id IS NULL`, records each in a new `quality_reject_record` audit table, and a CSV report is written. Config-driven (reuses the existing `quality_filter.enabled_record_types` gate), CPT-only.

**Tech Stack:** Python 3.12, SQLite (`sqlite3`), typer CLI, pytest. No new dependencies.

## Global Constraints

- **Python / tests:** use `/home/arr65/venvs/dev_nzgd_venv/bin/python` for everything. Run tests with `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest`.
- **Style:** ruff, numpy docstring convention, type hints throughout. The project enables ruff `ANN` rules and does **not** exempt tests, so **every new test function needs type-annotated params and a return type** (e.g. `def test_x(tmp_path: Path) -> None:`), and **all test imports stay at the top of the file** (appending imports below functions trips `E402`). Lint the test file too.
- **Import style (dedup package):** import names directly (`from nzgd.dedup.data_types import TableConfig`). In `deduplicate.py`, extend the existing `from nzgd.dedup.quality_filter import (...)` and `from nzgd.dedup.reports import (...)` groups.
- **Scope:** CPT only. SPT/extraction untouched. This builds on the shipped constant-column filter; do **not** modify `find_constant_column_reports` or `apply_quality_filter`.
- **Tombstone safety (correctness requirement):** a record is deleted only if its report table has zero rows for it **AND** `nzgdrecord.merged_into_nzgd_id IS NULL`. The phase runs before any merge pass, so no `merged_into` pointers exist yet; the NULL check is a mandatory belt-and-suspenders guard. A naive "delete any zero-report record" would destroy merge redirect chains (915 such tombstones exist in a real deduped DB).
- **Placement:** `delete_emptied_records` runs inside the per-record-type loop, immediately after the existing `apply_quality_filter` call, still before Pass 0.
- **Discard unit safety:** deletion uses the same per-record `SAVEPOINT` discipline as `apply_quality_filter` (release on success; rollback-then-release on failure; failures appended to the shared `all_failures` list with keys `cluster_id`, `canonical_nzgd_id`, `merged_nzgd_ids`, `record_type`, `error`).
- **Source DB never modified** (dedup runs on a copy); every deletion is auditable in `quality_reject_record`.
- **Approved impact:** ~955 CPT records deleted under the shipped filter config.
- **Spec:** `docs/superpowers/specs/2026-07-03-cpt-emptied-record-deletion-design.md`.

---

### Task 1: Foundations — `quality_reject_record` table

**Files:**
- Modify: `nzgd/dedup/schema.py`

**Interfaces:**
- Consumes: existing `apply_dedup_schema(conn)`.
- Produces: table `quality_reject_record(reject_record_id, run_id, record_type, nzgd_id, reason, n_reports_discarded, deleted_at)`, created idempotently by `apply_dedup_schema`.

Pure scaffolding (exercised end-to-end by Task 2). Verification here is a smoke check, not a pytest test.

- [ ] **Step 1: Add the DDL constants**

In `nzgd/dedup/schema.py`, immediately after the existing `_INDEX_QUALITY_REJECT_NZGD = "..."` line, add:

```python
_CREATE_QUALITY_REJECT_RECORD = """
CREATE TABLE IF NOT EXISTS quality_reject_record (
    reject_record_id     INTEGER PRIMARY KEY,
    run_id               INTEGER NOT NULL REFERENCES dedup_run(run_id),
    record_type          TEXT NOT NULL,
    nzgd_id              INTEGER NOT NULL,
    reason               TEXT NOT NULL,
    n_reports_discarded  INTEGER NOT NULL,
    deleted_at           TEXT NOT NULL
)
"""

_INDEX_QUALITY_REJECT_RECORD_RUN = "CREATE INDEX IF NOT EXISTS idx_quality_reject_record_run ON quality_reject_record(run_id)"
```

- [ ] **Step 2: Create the table inside `apply_dedup_schema`**

In `nzgd/dedup/schema.py`, in `apply_dedup_schema`, immediately after the existing line `cur.execute(_INDEX_QUALITY_REJECT_NZGD)`, add:

```python
    cur.execute(_CREATE_QUALITY_REJECT_RECORD)
    cur.execute(_INDEX_QUALITY_REJECT_RECORD_RUN)
```

- [ ] **Step 3: Smoke-verify schema**

Run:
```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python - <<'PY'
import sqlite3
from nzgd.dedup.schema import apply_dedup_schema
c = sqlite3.connect(":memory:")
c.execute("PRAGMA foreign_keys = ON")
c.executescript(
    "CREATE TABLE nzgdrecord (nzgd_id INTEGER PRIMARY KEY); "
    "CREATE TABLE cptreport (cpt_id INTEGER PRIMARY KEY, nzgd_id INTEGER, source_file TEXT NOT NULL, has_cpt_data INTEGER NOT NULL DEFAULT 1);"
)
apply_dedup_schema(c)
apply_dedup_schema(c)  # idempotent: must not raise
cols = [r[1] for r in c.execute("PRAGMA table_info(quality_reject_record)")]
print("quality_reject_record cols:", cols)
assert cols == ["reject_record_id", "run_id", "record_type", "nzgd_id", "reason", "n_reports_discarded", "deleted_at"]
print("SMOKE OK")
PY
```
Expected: prints the column list and `SMOKE OK`, no exception.

- [ ] **Step 4: Lint**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m ruff check nzgd/dedup/schema.py`
Expected: note `nzgd/dedup/schema.py` has ONE pre-existing `I001` import-order finding that predates this feature — it is out of scope; do not fix it. There must be no *new* findings introduced by this task. (If ruff reports only that single pre-existing `I001`, proceed.)

- [ ] **Step 5: Commit**

```bash
git add nzgd/dedup/schema.py
git commit -m "feat(dedup): add quality_reject_record table"
```

---

### Task 2: `delete_emptied_records` + report writer

**Files:**
- Modify: `nzgd/dedup/quality_filter.py` (add `delete_emptied_records`)
- Modify: `nzgd/dedup/reports.py` (add `write_quality_reject_record_report`)
- Test: `tests/dedup/test_quality_filter.py` (add integration test)

**Interfaces:**
- Consumes: the `quality_reject_record` table (Task 1); the existing `quality_reject` table + `find_constant_column_reports` / `apply_quality_filter`; `TableConfig` / `CPT_TABLE_CONFIG`.
- Produces:
  - `delete_emptied_records(conn, run_id: int, table_cfg: TableConfig, failures: list[dict] | None = None) -> int` — deletes `nzgdrecord` rows this run's filter emptied; returns the count deleted.
  - `write_quality_reject_record_report(conn, run_id: int, path: Path) -> None` — flattens `quality_reject_record` rows for the run into a CSV.

- [ ] **Step 1: Write the failing test**

Append this test function to the END of `tests/dedup/test_quality_filter.py` (the helper `_start_run`, `_COLUMNS`, and the conftest helpers already exist in this file; add `delete_emptied_records` to the existing `from nzgd.dedup.quality_filter import ...` line and `write_quality_reject_record_report` to the existing `from nzgd.dedup.reports import ...` line — keep both at the top of the file):

```python
def test_delete_emptied_records(fresh_db: sqlite3.Connection, tmp_path: Path) -> None:
    # RecA (nzgd 1): single report, constant u2 -> record emptied -> deleted
    add_cpt_record(fresh_db, nzgd_id=1)
    add_cpt_report(fresh_db, 10, 1, [(0.1, 1.0, 0.010, 0.0),
                                     (0.2, 1.1, 0.011, 0.0),
                                     (0.3, 1.2, 0.012, 0.0)])
    # RecB (nzgd 2): constant report + a good report -> record kept
    add_cpt_record(fresh_db, nzgd_id=2)
    add_cpt_report(fresh_db, 20, 2, [(0.1, 1.0, 0.010, 0.0),
                                     (0.2, 1.1, 0.011, 0.0),
                                     (0.3, 1.2, 0.012, 0.0)])
    add_cpt_report(fresh_db, 21, 2, [(0.1, 1.0, 0.010, 0.05),
                                     (0.2, 1.1, 0.011, 0.06),
                                     (0.3, 1.2, 0.012, 0.07)])
    # RecC (nzgd 3): single constant report BUT a merge tombstone -> NOT deleted
    add_cpt_record(fresh_db, nzgd_id=3)
    add_cpt_report(fresh_db, 30, 3, [(0.1, 1.0, 0.010, 0.0),
                                     (0.2, 1.1, 0.011, 0.0),
                                     (0.3, 1.2, 0.012, 0.0)])

    run_id = _start_run(fresh_db)
    # Make RecC a merge tombstone (redirect to RecB); the guard must protect it.
    fresh_db.execute("UPDATE nzgdrecord SET merged_into_nzgd_id = 2 WHERE nzgd_id = 3")

    failures: list[dict] = []
    entries = find_constant_column_reports(fresh_db, CPT_TABLE_CONFIG, _COLUMNS, 3)
    apply_quality_filter(fresh_db, entries, run_id, CPT_TABLE_CONFIG, failures=failures)
    n_emptied = delete_emptied_records(fresh_db, run_id, CPT_TABLE_CONFIG, failures=failures)

    assert n_emptied == 1
    assert failures == []
    # nzgd 1 deleted; nzgd 2 (good report) and nzgd 3 (tombstone) survive.
    assert sorted(r[0] for r in fresh_db.execute("SELECT nzgd_id FROM nzgdrecord")) == [2, 3]
    # RecB's good report survives.
    assert sorted(r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport")) == [21]
    # audit: exactly one emptied-record row, for nzgd 1.
    qrr = fresh_db.execute(
        "SELECT nzgd_id, reason, n_reports_discarded FROM quality_reject_record"
    ).fetchall()
    assert qrr == [(1, "emptied_by_quality_filter", 1)]

    out = tmp_path / "qrr.csv"
    write_quality_reject_record_report(fresh_db, run_id, out)
    text = out.read_text()
    assert "nzgd_id" in text and "emptied_by_quality_filter" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_quality_filter.py::test_delete_emptied_records -v`
Expected: FAIL — `ImportError: cannot import name 'delete_emptied_records'` (and `write_quality_reject_record_report`).

- [ ] **Step 3: Implement `delete_emptied_records`**

Append to `nzgd/dedup/quality_filter.py` (the module already imports `sqlite3`, `datetime`/`timezone`, and `TableConfig` — no new imports needed):

```python
def delete_emptied_records(
    conn: sqlite3.Connection,
    run_id: int,
    table_cfg: TableConfig,
    failures: list[dict] | None = None,
) -> int:
    """Delete nzgdrecord rows this run's quality filter left with zero reports.

    Candidates are the records this run's filter touched (distinct nzgd_ids in
    ``quality_reject`` for this run and record type). A record is deleted only if
    its report table now has zero rows for it AND its ``merged_into_nzgd_id`` is
    NULL — never a merge tombstone. This runs before any merge pass, so no
    ``merged_into`` pointers exist yet; the NULL check is belt-and-suspenders.

    Each deletion is its own SAVEPOINT (mirroring ``apply_quality_filter``) and is
    recorded in ``quality_reject_record``. Failures are appended to ``failures``
    (shared with the dedup failures report). Returns the number of records deleted.
    """
    cur = conn.cursor()
    candidates = [
        row[0]
        for row in cur.execute(
            "SELECT DISTINCT nzgd_id FROM quality_reject WHERE run_id = ? AND record_type = ?",
            (run_id, table_cfg.record_type),
        ).fetchall()
    ]
    n_deleted = 0
    for nzgd_id in candidates:
        remaining = cur.execute(
            f"SELECT COUNT(*) FROM {table_cfg.report_table} WHERE nzgd_id = ?",
            (nzgd_id,),
        ).fetchone()[0]
        if remaining != 0:
            continue
        row = cur.execute(
            "SELECT merged_into_nzgd_id FROM nzgdrecord WHERE nzgd_id = ?",
            (nzgd_id,),
        ).fetchone()
        if row is None or row[0] is not None:
            continue  # record missing, or a merge tombstone — never delete
        savepoint = f"quality_reject_record_{nzgd_id}"
        cur.execute(f"SAVEPOINT {savepoint}")
        try:
            n_reports = cur.execute(
                "SELECT COUNT(*) FROM quality_reject WHERE run_id = ? AND nzgd_id = ?",
                (run_id, nzgd_id),
            ).fetchone()[0]
            cur.execute("DELETE FROM nzgdrecord WHERE nzgd_id = ?", (nzgd_id,))
            cur.execute(
                "INSERT INTO quality_reject_record (run_id, record_type, nzgd_id, "
                "reason, n_reports_discarded, deleted_at) VALUES (?,?,?,?,?,?)",
                (
                    run_id,
                    table_cfg.record_type,
                    nzgd_id,
                    "emptied_by_quality_filter",
                    n_reports,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            n_deleted += 1
        except Exception as exc:  # noqa: BLE001 — one bad record must not abort the rest
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            if failures is not None:
                failures.append(
                    {
                        "cluster_id": None,
                        "canonical_nzgd_id": nzgd_id,
                        "merged_nzgd_ids": [],
                        "record_type": table_cfg.record_type,
                        "error": f"emptied-record delete of nzgd_id={nzgd_id} failed: {exc!r}",
                    }
                )
    conn.commit()
    return n_deleted
```

- [ ] **Step 4: Implement `write_quality_reject_record_report`**

In `nzgd/dedup/reports.py`, append (the module already imports `csv`, `sqlite3`, and `Path`):

```python
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_quality_filter.py::test_delete_emptied_records -v`
Expected: PASS.

- [ ] **Step 6: Run the full dedup suite (no regressions)**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/ -q`
Expected: PASS — all pre-existing tests plus the new one.

- [ ] **Step 7: Lint (including the test file)**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m ruff check nzgd/dedup/quality_filter.py nzgd/dedup/reports.py tests/dedup/test_quality_filter.py`
Expected: `All checks passed!` — no `F401`/`E402`/`ANN` findings. Fix any issue and re-run.

- [ ] **Step 8: Commit**

```bash
git add nzgd/dedup/quality_filter.py nzgd/dedup/reports.py tests/dedup/test_quality_filter.py
git commit -m "feat(dedup): delete CPT records emptied by the quality filter"
```

---

### Task 3: Config + driver wiring

**Files:**
- Modify: `nzgd/resources/config.yaml` (add output filename)
- Modify: `nzgd/scripts/db/deduplicate.py` (imports, call in loop, write report)
- Test: `tests/dedup/test_quality_filter.py` (end-to-end pipeline test)

**Interfaces:**
- Consumes: `delete_emptied_records`, `write_quality_reject_record_report` (Task 2); `constants.DEDUP_CONFIG`.
- Produces: the `deduplicate` CLI deletes emptied CPT records right after the filter and writes `quality_reject_record_report.csv`.

- [ ] **Step 1: Add config**

In `nzgd/resources/config.yaml`, in the existing `deduplication.output:` block, add one line after `quality_filter_report_filename: "quality_filter_report.csv"`:

```yaml
    quality_reject_record_report_filename: "quality_reject_record_report.csv"
```

- [ ] **Step 2: Write the failing end-to-end test**

Append this test to the END of `tests/dedup/test_quality_filter.py` (`Path`, `CliRunner`, `app`, `_make_fresh_db`, `add_cpt_record`, `add_cpt_report` are all already imported at the top of the file):

```python
def test_emptied_record_deleted_in_full_pipeline(tmp_path: Path) -> None:
    src = tmp_path / "source.db"
    conn = _make_fresh_db(src)
    # normal record -> survives with its report
    add_cpt_record(conn, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(conn, 10, 1, [(0.1, 1.0, 0.010, 0.05),
                                 (0.2, 1.1, 0.011, 0.06),
                                 (0.3, 1.2, 0.012, 0.07)])
    # single-report record with a constant-u2 trace -> report discarded AND record deleted
    add_cpt_record(conn, nzgd_id=2, lat=-41.0, lon=174.0)
    add_cpt_report(conn, 20, 2, [(0.1, 1.0, 0.010, 0.0),
                                 (0.2, 1.1, 0.011, 0.0),
                                 (0.3, 1.2, 0.012, 0.0)])
    conn.commit()
    conn.close()

    target = tmp_path / "deduped.db"
    result = CliRunner().invoke(app, ["--source", str(src), "--target", str(target), "--skip-spt"])
    assert result.exit_code == 0, result.output

    out = sqlite3.connect(target)
    try:
        assert [r[0] for r in out.execute("SELECT cpt_id FROM cptreport ORDER BY cpt_id")] == [10]
        assert [r[0] for r in out.execute("SELECT nzgd_id FROM nzgdrecord ORDER BY nzgd_id")] == [1]
        assert out.execute("SELECT nzgd_id, record_type FROM quality_reject_record").fetchall() == [(2, "CPT")]
    finally:
        out.close()
    assert (tmp_path / "quality_reject_record_report.csv").exists()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_quality_filter.py::test_emptied_record_deleted_in_full_pipeline -v`
Expected: FAIL — `nzgd_id=2` is still present in `nzgdrecord` (and/or `quality_reject_record` is empty / the report file is missing), because the driver does not yet delete emptied records.

- [ ] **Step 4: Add imports to `deduplicate.py`**

In `nzgd/scripts/db/deduplicate.py`, add `delete_emptied_records` to the existing quality_filter import group so it reads:

```python
from nzgd.dedup.quality_filter import (
    apply_quality_filter,
    delete_emptied_records,
    find_constant_column_reports,
)
```

and add `write_quality_reject_record_report` to the existing `from nzgd.dedup.reports import (...)` group (keep the group alphabetically ordered: it goes after `write_quality_filter_report`).

- [ ] **Step 5: Call `delete_emptied_records` in the loop**

In `nzgd/scripts/db/deduplicate.py`, inside the `if not skip_quality_filter and cfg.record_type in qf_enabled:` block, immediately after the existing line `typer.echo(f"[{cfg.record_type}] Quality filter: discarded {n_qf} reports.")`, add (same indentation, still inside that `if`):

```python
            n_emptied = delete_emptied_records(conn, run_id, cfg, failures=all_failures)
            typer.echo(f"[{cfg.record_type}] Quality filter: deleted {n_emptied} emptied records.")
```

- [ ] **Step 6: Write the report after the loop**

In `nzgd/scripts/db/deduplicate.py`, immediately after the existing line `typer.echo(f"Quality filter report at {qf_report_path}.")`, add:

```python
    qrr_report_path = out_dir / constants.DEDUP_CONFIG["output"]["quality_reject_record_report_filename"]
    write_quality_reject_record_report(conn, run_id, qrr_report_path)
    typer.echo(f"Quality reject record report at {qrr_report_path}.")
```

- [ ] **Step 7: Run the end-to-end test to verify it passes**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_quality_filter.py::test_emptied_record_deleted_in_full_pipeline -v`
Expected: PASS.

- [ ] **Step 8: Run the full dedup suite (no regressions)**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/ -q`
Expected: PASS.

- [ ] **Step 9: Lint**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m ruff check nzgd/scripts/db/deduplicate.py tests/dedup/test_quality_filter.py`
Expected: `All checks passed!`. Fix any issue and re-run.

- [ ] **Step 10: Commit**

```bash
git add nzgd/resources/config.yaml nzgd/scripts/db/deduplicate.py tests/dedup/test_quality_filter.py
git commit -m "feat(dedup): wire emptied-record deletion into the pipeline"
```

---

## Optional real-data smoke (not a committed test)

After implementation, optionally confirm on real data (heavy — copies + dedups a real DB):

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -m nzgd.scripts.db.deduplicate \
    --source /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p8p1_20260625.db \
    --target /tmp/qrr_smoke_deduped.db --skip-spt
```
Expected: a `[CPT] Quality filter: deleted ~955 emptied records.` line, and `quality_reject_record_report.csv` next to the target. Delete `/tmp/qrr_smoke_deduped.db` afterwards.

---

## Self-Review

**Spec coverage:**
- `quality_reject_record` table (spec §4.2) → Task 1. `delete_emptied_records` (§4.1) → Task 2. Report writer (§4.3) → Task 2. Config (§4.4) → Task 3. Driver wiring + placement (§4.5) → Task 3. Testing RecA/RecB/RecC tombstone-guard (§6) → Task 2; e2e → Task 3.
- Tombstone guard (§2, §3, §7): enforced by the `merged_into_nzgd_id IS NULL` check in `delete_emptied_records` and exercised by RecC in the Task 2 test.
- Failure-dict shape (§4.1) matches the shared `write_failures_report` consumer (same 5 keys as the existing quality-filter failure path).

**Placeholder scan:** No TBD/TODO; every code step has complete code; every run step has an exact command and expected result.

**Type consistency:** `delete_emptied_records(conn, run_id, table_cfg, failures=None) -> int` and `write_quality_reject_record_report(conn, run_id, path)` are defined in Task 2 and called with identical signatures in the Task 2 test and the Task 3 driver. The `quality_reject_record` column list is identical across the DDL (Task 1), the INSERT (Task 2), and the report SELECT (Task 2). The reason string `"emptied_by_quality_filter"` matches between the INSERT and the Task 2 assertion.
