# CPT emptied-record deletion — design

**Date:** 2026-07-03
**Status:** Approved design, pending spec review
**Scope:** Follow-on to the constant-column quality filter (`nzgd/dedup/`).
**Parent spec:** `docs/superpowers/specs/2026-07-03-cpt-constant-column-quality-filter-design.md`

## 1. Goal

After the constant-column quality filter discards CPT reports, delete any
`nzgdrecord` that is left with **zero** reports, so a CPT record whose only
trace was a flat/placeholder column vanishes from the deduped DB entirely —
not just its trace, but the record row too. Downstream consumers read only the
deduped DB, so a record with no usable trace should not linger there as
location/metadata-only noise.

## 2. Background

The parent filter discards `cptreport` rows (and their `cptmeasurements` /
`cptvs30estimates` cascade) but never touches `nzgdrecord`. `nzgdrecord` rows
are created independently by `nzgd/scripts/db/put_nzgd_metadata.py` from the
index metadata — one per investigation, regardless of trace quality — so after
the filter runs, a single-report record whose only report was discarded is left
present with zero reports. On `uc_nzgd_v0p8p1_20260625.db` that is **955 CPT
records (1.9% of 51,431)**, holding 1,811 of the 5,222 discarded reports.

### The merge-tombstone trap (why placement matters)

A naive "delete any CPT `nzgdrecord` with zero reports" is **wrong**. In the
deduped DB, **915 CPT records already have zero `cptreport`s — and every one is
a merge tombstone**: `merged_into_nzgd_id` is set, redirecting the merged id to
its surviving canonical record. `verify.py` (`find_spt_format_orphans`) and
consumers follow `merged_into_nzgd_id` to resolve a merged id to its survivor.
Deleting those tombstones would break the redirect chains.

The clean avoidance: **delete emptied records inside the quality-filter step,
immediately after it discards reports and before any merge pass runs.** At that
moment (the filter is the first step, before Pass 0 / hash / fuzzy):
- the only zero-report records in existence are exactly the ones the filter just
  emptied;
- no `merged_into_nzgd_id` pointers exist yet;
- so no tombstone can be touched, and no fragile "is this a tombstone?" heuristic
  is needed — a `merged_into_nzgd_id IS NULL` guard is kept purely as
  belt-and-suspenders.

### Referential integrity

FKs referencing `nzgdrecord.nzgd_id` (all `ON DELETE NO ACTION`): `cptreport`,
`cptvs30estimates`, `sptreport`, and the self-reference `merged_into_nzgd_id`.
The dedup connection runs `PRAGMA foreign_keys = ON`. After the filter deletes
an emptied record's only report (+ its `cptvs30estimates` cascade), and with
`merged_into_nzgd_id` all NULL at this stage, nothing references the record, so
`DELETE FROM nzgdrecord` is FK-safe.

## 3. Decisions

- **Placement:** run right after `apply_quality_filter`, inside the same
  per-record-type loop iteration, before Pass 0.
- **What to delete:** an `nzgdrecord` whose report table (`cptreport`) now has
  zero rows for it **and** whose `merged_into_nzgd_id IS NULL`.
- **Scope:** CPT only (the filter is CPT-only). SPT untouched.
- **Audit:** a dedicated `quality_reject_record` table + a CSV report, mirroring
  `quality_reject` / `write_quality_filter_report`. Deleting whole records is
  destructive enough to warrant first-class, queryable provenance.
- **Config:** always-on whenever the CPT quality filter runs; gated by the
  existing `quality_filter.enabled_record_types`. No separate toggle (it is
  reversible via the untouched source DB, and there are downstream review gates).

## 4. Design

### 4.1 `delete_emptied_records` — `nzgd/dedup/quality_filter.py`

`delete_emptied_records(conn, run_id, table_cfg, failures=None) -> int`

- Discover candidates purely from the audit trail (decoupled from the caller's
  entry list): `SELECT DISTINCT nzgd_id FROM quality_reject WHERE run_id = ? AND
  record_type = ?`. These are exactly the records this run's filter touched.
- For each candidate `nz`, in its own `SAVEPOINT` (mirroring
  `apply_quality_filter`):
  1. Skip unless it is now empty and not a merge target/source:
     `SELECT COUNT(*) FROM {table_cfg.report_table} WHERE nzgd_id = ?` is 0
     **and** `SELECT merged_into_nzgd_id FROM nzgdrecord WHERE nzgd_id = ?` is
     NULL.
  2. `n_reports = SELECT COUNT(*) FROM quality_reject WHERE run_id = ? AND
     nzgd_id = ?` (how many reports the filter discarded for this record).
  3. `DELETE FROM nzgdrecord WHERE nzgd_id = ?`.
  4. Insert one `quality_reject_record` row.
- On per-record exception: `ROLLBACK TO` then `RELEASE` the savepoint, and append
  a failure dict to `failures` with the same 5 keys the shared
  `write_failures_report` expects — `cluster_id=None`,
  `canonical_nzgd_id=nz`, `merged_nzgd_ids=[]`, `record_type=table_cfg.record_type`,
  `error=f"emptied-record delete of nzgd_id={nz} failed: {exc!r}"`.
- `conn.commit()` at the end; return the number of records deleted.

### 4.2 `quality_reject_record` table — `nzgd/dedup/schema.py`

Added in `apply_dedup_schema` (idempotent, `IF NOT EXISTS`), alongside
`quality_reject`:

```sql
CREATE TABLE IF NOT EXISTS quality_reject_record (
    reject_record_id     INTEGER PRIMARY KEY,
    run_id               INTEGER NOT NULL REFERENCES dedup_run(run_id),
    record_type          TEXT NOT NULL,
    nzgd_id              INTEGER NOT NULL,
    reason               TEXT NOT NULL,          -- 'emptied_by_quality_filter'
    n_reports_discarded  INTEGER NOT NULL,
    deleted_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quality_reject_record_run ON quality_reject_record(run_id);
```

### 4.3 Report — `nzgd/dedup/reports.py`

`write_quality_reject_record_report(conn, run_id, path)` — flattens
`quality_reject_record` rows for the run into a CSV with header
`record_type, nzgd_id, reason, n_reports_discarded, deleted_at`.

### 4.4 Config — `nzgd/resources/config.yaml`

Add one line to the existing `deduplication.output` block:

```yaml
    quality_reject_record_report_filename: "quality_reject_record_report.csv"
```

No new behavioural config.

### 4.5 Driver — `nzgd/scripts/db/deduplicate.py`

- Import `delete_emptied_records` (from `quality_filter`) and
  `write_quality_reject_record_report` (from `reports`).
- Inside the per-record-type loop, immediately after the existing
  `apply_quality_filter` call:

  ```python
  n_emptied = delete_emptied_records(conn, run_id, cfg, failures=all_failures)
  typer.echo(f"[{cfg.record_type}] Quality filter: deleted {n_emptied} emptied records.")
  ```

- After the loop, write the report:
  `write_quality_reject_record_report(conn, run_id, out_dir / constants.DEDUP_CONFIG["output"]["quality_reject_record_report_filename"])`.

## 5. Files changed

| File | Change |
|------|--------|
| `nzgd/dedup/quality_filter.py` | add `delete_emptied_records` |
| `nzgd/dedup/schema.py` | add `quality_reject_record` table + index in `apply_dedup_schema` |
| `nzgd/dedup/reports.py` | add `write_quality_reject_record_report` |
| `nzgd/scripts/db/deduplicate.py` | call `delete_emptied_records` in loop; write report |
| `nzgd/resources/config.yaml` | add output filename |
| test module | one integration test (see §6) |

## 6. Testing

Per project convention (minimize tests, prefer integration over unit), one
integration test on a temporary real-schema SQLite DB:

- **RecA** — one record, one report, that report constant → after filter +
  `delete_emptied_records`, the report is gone **and** the `nzgdrecord` row is
  deleted; a `quality_reject_record` row exists with `n_reports_discarded = 1`.
- **RecB** — one record, two reports (one constant, one good) → the constant
  report is discarded but the record **survives** (still has the good report);
  no `quality_reject_record` row for it.
- **RecC (tombstone guard)** — one record whose only report is constant, but with
  `merged_into_nzgd_id` pre-set to another id → after the run the report is
  discarded but the `nzgdrecord` row is **NOT** deleted (guard holds), and no
  `quality_reject_record` row for it.
- Assert `failures == []`.

## 7. Non-goals / assumptions

- **Merge tombstones are never deleted** (guarded by `merged_into_nzgd_id IS
  NULL`; also structurally impossible at this pipeline stage).
- **CPT only.** Assumes record types are exclusive (a CPT-type `nzgdrecord` has
  no `sptreport`); the check inspects only the record's own report table, so no
  cross-type coupling. If the data model ever mixes types on one record, revisit.
- **Only records emptied by this run's filter** are considered (candidates come
  from `quality_reject`), not arbitrary pre-existing empty records.
- Deleting a record removes it from location/metadata-only queries too — this is
  the intended effect.

## 8. Risks & rollback

- **Destructive** (deletes `nzgdrecord` rows). Mitigated: the source DB is never
  modified, and every deletion is listed in `quality_reject_record` with its
  reason and discarded-report count, so it is auditable and re-derivable.
- **Tombstone safety:** the `merged_into_nzgd_id IS NULL` guard plus the
  before-any-merge placement together guarantee no redirect chain is broken.

## 9. Impact summary

Under the approved config, a run deletes **~955 CPT records** (the single-report
records whose only trace had a constant column), removing 1,811 already-discarded
reports' parent records from the deduped DB.
