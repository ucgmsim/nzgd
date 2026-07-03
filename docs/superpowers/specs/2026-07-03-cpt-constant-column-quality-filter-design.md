# CPT constant-column data-quality filter — design

**Date:** 2026-07-03
**Status:** Approved design, pending spec review
**Scope:** New data-quality filter in the `nzgd/dedup/` pipeline.

## 1. Goal

Discard any CPT trace (one `cptreport`) whose measurement data is **constant** in
any one of its columns. A physically real CPT varies with depth in every channel;
a column that holds a single repeated value is a broken extraction or an
unmeasured-channel placeholder, and such a report is of no analytical use.

The filter runs as the first step of the deduplication pipeline, before any dedup
pass, so bad reports cannot be chosen as canonical or pollute fuzzy matching.

## 2. Background

Two dedup systems exist; only one is live:

- **Retired:** `conditioning.remove_duplicate_extractions()` — no longer defined or
  called anywhere in the codebase. Superseded.
- **Live:** the `nzgd/dedup/` package, driven by `nzgd/scripts/db/deduplicate.py`.
  It copies the source SQLite DB → target, then runs Pass 0 (within-record
  consolidation), Pass 1 (bit-exact hash), Pass 2 (metadata-blocked fuzzy), then
  supplemental consolidation. Audit rows go to `dedup_audit`; the source DB is
  never modified.

Relevant facts about the data model (measured on
`uc_nzgd_v0p8p1_20260625.db`):

- One CPT `nzgdrecord` commonly has **many** `cptreport` rows (51,431 records →
  177,791 reports; up to 23 per record). Therefore the unit of discard is the
  **report** (one trace), not the record — a flat report is dropped while its good
  siblings survive.
- Deleting a CPT report is already implemented and proven by the merge path:
  `executor.delete_report()` deletes `cptvs30estimates` + `cptmeasurements` +
  `cptreport` in FK-safe order. It is reused unchanged.

## 3. Motivating data

Constant-column prevalence over the 83,589 CPT reports that have trace rows
(constant = a column with ≥ 3 non-null values that are all equal):

| Column | # reports constant | notes |
|--------|-------------------:|-------|
| depth  | 0     | never constant (monotonic) |
| qc     | 1     | value = 0 |
| fs     | 8     | 7 of them = 0 |
| u2     | 5,214 | 4,829 of them = 0 |
| **any of the four** | **5,222** (6.2%) | one report is constant in both qc and fs, so the total is 5,222, not 5,223 |

Additional facts that shaped the design:

- **No report has an all-NULL column** (qc, fs, and u2 each have ≥ 1 non-null
  value in every report; u2-all-null count = 0). "Unmeasured u2" is always encoded
  as a constant placeholder, never as NULL.
- No report has ≤ 2 rows, so there is no "trivially constant because tiny"
  artifact.

## 4. Decisions

- **Columns checked (config-driven):** `depth_m`, `qc_MPa`, `fs_MPa`, `u2_MPa` —
  all four. Including u2 is deliberate: constant-placeholder u2 reports (5,214) are
  of no use and are discarded even though qc/fs may be fine. depth never triggers
  today but is included for completeness and robustness against future breakage.
- **"Constant" definition:** a column with **≥ `min_non_null_rows` non-null values,
  all exactly equal** (`COUNT(DISTINCT col) = 1`). Exact equality — no tolerance
  (real placeholders are bit-exact; a tolerance would risk discarding low-variance
  real traces).
- **All-NULL is not "constant":** a column with no values does not trigger the
  filter. This is a deliberate boundary (see Non-goals). It has zero present effect
  because no all-NULL columns exist in the data.
- **Unit of discard:** the `cptreport` (one trace), physically deleted.
- **Record types:** CPT only. SPT is excluded because `ISPT_*` blow-count columns
  can legitimately be constant (e.g., refusal). Config-driven via
  `enabled_record_types`.

## 5. Design

### 5.1 Placement in the pipeline

In `deduplicate.py`, inside the existing per-record-type loop, the quality filter
runs **before Pass 0**:

```
for cfg in (CPT_TABLE_CONFIG, SPT_TABLE_CONFIG):
    quality filter   ← NEW (CPT only, gated by config + --skip-quality-filter)
    Pass 0 (within-record)
    Pass 1 (hash)
    Pass 2 (fuzzy)
    supplemental consolidation
```

Rationale for running first: `select_canonical` does not consider trace quality,
so a flat report could otherwise be picked as canonical and absorb good data, or
enter fuzzy calibration. Removing it up front prevents both.

### 5.2 Detection — `nzgd/dedup/quality_filter.py`

`find_constant_column_reports(conn, table_cfg, columns, min_non_null_rows)
-> list[QualityRejectEntry]`

- Validates `columns ⊆ table_cfg.measurement_value_columns` (guards against typos /
  SQL injection when column names are interpolated).
- Issues one `GROUP BY … HAVING` query per record type, e.g. for CPT with the four
  columns:

```sql
SELECT m.cpt_id, r.nzgd_id, COUNT(*) AS n_rows,
       COUNT(qc_MPa) AS nn_qc, COUNT(DISTINCT qc_MPa) AS d_qc, MIN(qc_MPa) AS v_qc,
       COUNT(fs_MPa) AS nn_fs, COUNT(DISTINCT fs_MPa) AS d_fs, MIN(fs_MPa) AS v_fs,
       COUNT(u2_MPa) AS nn_u2, COUNT(DISTINCT u2_MPa) AS d_u2, MIN(u2_MPa) AS v_u2,
       COUNT(depth_m) AS nn_d, COUNT(DISTINCT depth_m) AS d_d, MIN(depth_m) AS v_d
FROM cptmeasurements m
JOIN cptreport r ON r.cpt_id = m.cpt_id
GROUP BY m.cpt_id
HAVING (nn_qc >= :min AND d_qc = 1)
    OR (nn_fs >= :min AND d_fs = 1)
    OR (nn_u2 >= :min AND d_u2 = 1)
    OR (nn_d  >= :min AND d_d  = 1)
```

- For each returned row, builds a `QualityRejectEntry` with a
  `constant_columns` dict mapping each offending column to its constant value
  (`MIN(col)`, which equals the repeated value).

The `SELECT`/`HAVING` fragments are generated from the configured column list, so
adding/removing a column is a config change only.

### 5.3 Discard — `nzgd/dedup/quality_filter.py`

`apply_quality_filter(conn, entries, run_id, table_cfg, failures=None) -> int`

- For each entry, in its own `SAVEPOINT` (mirroring `executor.apply_merge_plan`):
  1. `executor.delete_report(conn, entry.report_id, table_cfg)`
  2. Insert one `quality_reject` audit row.
- On exception: `ROLLBACK TO SAVEPOINT`, append a dict to `failures` (shared with
  the dedup failures report). One bad report does not abort the rest.
- Returns the number of reports discarded.

### 5.4 New audit table — `nzgd/dedup/schema.py`

Added in `apply_dedup_schema()` (idempotent, `IF NOT EXISTS`). A dedicated table
rather than `dedup_audit`, because `dedup_audit` is built around canonical/merged
pairs (NOT-NULL `canonical_nzgd_id`, `merged_nzgd_id`, `report_pairs_json`) which
do not exist for a discard.

```sql
CREATE TABLE IF NOT EXISTS quality_reject (
    reject_id             INTEGER PRIMARY KEY,
    run_id                INTEGER NOT NULL REFERENCES dedup_run(run_id),
    record_type           TEXT NOT NULL,
    nzgd_id               INTEGER NOT NULL,
    report_id             INTEGER NOT NULL,
    reason                TEXT NOT NULL,           -- 'constant_column'
    constant_columns_json TEXT NOT NULL,           -- {"u2_MPa": 0.0}
    n_rows                INTEGER NOT NULL,
    rejected_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quality_reject_run     ON quality_reject(run_id);
CREATE INDEX IF NOT EXISTS idx_quality_reject_nzgd_id ON quality_reject(nzgd_id);
```

### 5.5 Data type — `nzgd/dedup/data_types.py`

```python
@dataclass(frozen=True)
class QualityRejectEntry:
    record_type: str
    nzgd_id: int
    report_id: int
    reason: str                          # 'constant_column'
    constant_columns: dict[str, float]   # {column_name: constant_value}
    n_rows: int
```

### 5.6 Report — `nzgd/dedup/reports.py`

`write_quality_filter_report(conn, run_id, path)` — flattens `quality_reject` rows
for the run into a CSV: `record_type, nzgd_id, report_id, reason,
constant_columns, n_rows, rejected_at`. Called once after the record-type loop,
alongside the other report writers.

### 5.7 Config — `nzgd/resources/config.yaml` (under `deduplication:`)

```yaml
  quality_filter:
    enabled_record_types: ["CPT"]
    constant_columns:
      CPT: ["depth_m", "qc_MPa", "fs_MPa", "u2_MPa"]
    min_non_null_rows: 3

  output:
    # added to the existing output block
    quality_filter_report_filename: "quality_filter_report.csv"
```

Loaded via the existing `constants.DEDUP_CONFIG` (= `CONFIG["deduplication"]`).

### 5.8 CLI — `nzgd/scripts/db/deduplicate.py`

- New `--skip-quality-filter` flag (parity with `--skip-cpt` / `--skip-spt`).
- Invoke the filter first in the loop, gated by `enabled_record_types` and the
  flag; echo a summary line (`[CPT] Quality filter: discarded N reports.`).
- Write the quality-filter report after the loop.
- Quality-filter discards are **not** merges: they are recorded in
  `quality_reject` and stdout, and are deliberately not added to
  `dedup_run.n_records_merged`. (Adding a dedicated `dedup_run` counter column is
  possible but omitted as unnecessary.)

## 6. Non-goals / explicit boundaries

- **All-NULL columns are not discarded.** The filter triggers on constant *values*
  only. No all-NULL qc/fs/u2 columns exist in the current data, so this has no
  present effect; if future extractions produce all-NULL channels that should also
  be discarded, that is a separate criterion (a "required-channel" filter), not
  this feature.
- **No tolerance / near-constant detection.** Only bit-exact constancy.
- **SPT/borehole records are untouched.**
- **Extraction pipeline is unchanged.** The filter operates on the assembled trace
  in the DB, not on per-sheet extraction output.

## 7. Files changed

| File | Change |
|------|--------|
| `nzgd/dedup/quality_filter.py` | **new** — `find_constant_column_reports`, `apply_quality_filter` |
| `nzgd/dedup/data_types.py` | add `QualityRejectEntry` |
| `nzgd/dedup/schema.py` | add `quality_reject` table + indexes in `apply_dedup_schema` |
| `nzgd/dedup/reports.py` | add `write_quality_filter_report` |
| `nzgd/scripts/db/deduplicate.py` | invoke filter first in loop; `--skip-quality-filter`; write report |
| `nzgd/resources/config.yaml` | add `quality_filter` block + output filename |
| test module | one integration test (see §8) |

## 8. Testing

Per project convention (minimize tests, prefer integration over unit), a single
integration test:

- Build a temporary on-disk SQLite DB with the real minimal schema
  (`nzgdrecord`, `cptreport`, `cptmeasurements`, `cptvs30estimates`), run
  `apply_dedup_schema`, insert a `dedup_run` row, and insert reports:
  - **R1** normal varying qc/fs/u2 → kept
  - **R2** constant qc (≥ 3 equal rows) → discarded
  - **R3** constant u2 = 0 with good qc/fs → discarded (validates the u2 decision)
  - **R4** constant fs → discarded
  - **R5** 2-row constant qc (below `min_non_null_rows`) → kept (validates the guard)
- Run `find_constant_column_reports` + `apply_quality_filter`.
- Assert: discarded report ids = {R2, R3, R4}; their `cptmeasurements` /
  `cptvs30estimates` are gone; R1 and R5 intact; `quality_reject` has three rows
  with correct `constant_columns` and values; `failures` is empty.

## 9. Risks & rollback

- **Over-discard from including u2** (~5,214 reports). This is the intended
  behaviour, confirmed with the user; it is config-reversible (drop `u2_MPa` from
  `constant_columns`).
- **Recoverability:** the source DB is never modified and every discard is listed
  in `quality_reject` with its reason, so any deletion is auditable and
  re-derivable.
- **Ordering:** running before Pass 0 changes what later passes see (fewer, cleaner
  reports). This is intended; it cannot resurrect a discarded report.
- **Records left with zero reports (new state):** because a CPT `nzgdrecord`
  can have several `cptreport` rows, discarding one usually leaves siblings
  intact. But a single-report record whose only report has a flat column ends
  up with **zero** `cptreport` rows. On `uc_nzgd_v0p8p1_20260625.db` this is
  **955 of 51,431 CPT records (1.9%)** — dominantly single-report records whose
  only trace has a placeholder u2. This is a *new* state: pre-filter, every CPT
  record has ≥1 report (0 currently have zero). Investigated for downstream
  safety: every CPT-trace consumer in the codebase is report-driven
  (`SELECT ... FROM cptreport` — the metadata summary, vs30 estimation, etc.),
  so a zero-report record simply contributes no trace rows (the correct outcome
  for a record with no usable trace) rather than breaking any join or query.
  The record itself remains in `nzgdrecord` with its location/model metadata,
  and every discarded report is listed in `quality_reject`. Decision: accept —
  it is the intended consequence of the confirmed u2 inclusion, and it is
  config-reversible.

## 10. Impact summary

Under the approved config (all four columns), a run discards **~5,222 CPT reports**
(5,214 constant-u2, 8 constant-fs, 1 constant-qc, 0 depth) out of 83,589 CPT
reports with trace data. These fall in **4,246 CPT records** (8.3% of 51,431);
of those, **955 records (1.9%)** lose *all* their reports (1,811 reports),
ending up with zero `cptreport` rows — see the zero-report note in §9.
