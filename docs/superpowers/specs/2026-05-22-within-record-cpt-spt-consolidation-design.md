# Within-Record CPT/SPT Consolidation Design

Date: 2026-05-22
Status: Approved for plan-writing

## Background

The cross-record CPT/SPT deduplication feature (spec
`2026-05-20-cpt-spt-deduplication-design.md`, in production) identifies and
merges duplicate physical investigations uploaded under multiple NZGD IDs.
Below that record-level pass, each NZGD ID also typically carries multiple
`cptreport` rows — one per extracted source file or sheet (AGS / XLS / XLSX /
CSV variants of the same investigation). Today, the extraction-time function
`remove_duplicate_extractions` in `nzgd/extract/cpt/conditioning.py` drops
measurement data from sheets it identifies as duplicates of others, but the
`cptreport` rows themselves remain — five-plus rows per investigation is
common, with each row contributing a slightly different slice of per-report
metadata (`extracted_gwl_m`, `tip_net_area_ratio`, `predrill_depth_m`, etc.).

Two limitations of the current state motivate this design:

1. **No consolidation step**: there is no place in the pipeline where the
   multiple `cptreport` rows of one investigation get collapsed into a single
   row carrying the union of useful metadata. Downstream consumers see
   redundant rows and must pick a "best" one themselves.
2. **No sentinel handling on per-report metadata**: known sentinel values like
   `extracted_gwl_m = 0` (representing "no GWL recorded" in some upload
   formats) are stored as if they were real values. With the simple
   "canonical wins where non-NULL" rule we'd otherwise inherit, a sentinel
   would beat a real value during enrichment.

This design adds a within-record consolidation pass to the dedup module and
generalises the metadata-enrichment rule across all three tables
(`nzgdrecord`, `cptreport`, `sptreport`) using per-field plausibility ranges.
It also retires the existing extraction-time dedup and the now-redundant
`cpt_data_duplicate_of_cpt_id` column.

## Scope

In scope:

- A new Pass 0 (within-record consolidation) in `nzgd/dedup/`, running before
  the existing cross-record Passes 1 (hash) and 2 (fuzzy), for both CPT and
  SPT (BH) records.
- Per-field plausibility-range checks shared by Pass 0 and the existing
  cross-record `_enrich_canonical_metadata`, configured in `config.yaml`.
- Pluggable canonical-selection callable for Pass 0 (config-resolved).
- Removal of the extraction-time `remove_duplicate_extractions` and the
  `cpt_data_duplicate_of_cpt_id` column on `cptreport`.

Not in scope:

- Pluggable cross-record canonical selector (separate change to in-production
  code; one-diff extension if wanted later).
- Re-engineering the SPT extraction pipeline.
- Schema changes beyond dropping `cpt_data_duplicate_of_cpt_id` and widening
  the `dedup_audit.match_pass` CHECK constraint.

## Decisions settled during design

| Decision                              | Choice                                                                                                              | Reason                                                                                                                       |
| ---                                   | ---                                                                                                                 | ---                                                                                                                          |
| Architecture                          | Replace extraction-time dedup entirely; do all dedup post-DB as Pass 0 + existing Passes 1 & 2                       | Single code path; same audit ledger; ~5% temporary measurement-row bloat in DB is acceptable                                |
| Clustering rule                       | Source-file stem + trace identity (hash or fuzzy); stems with non-matching data-bearing rows are split into sub-stems | Handles typical multi-format upload AND the rare "single file, multiple CPTs in different sheets" case                       |
| No-data row attribution in split stem | Attach to the sub-stem with the smallest `cpt_id` (deterministic heuristic)                                          | Perfect attribution impossible without sheet-name pattern matching; smallest-id rule is reproducible and rarely material     |
| Sentinel handling                     | Per-field plausibility ranges in `config.yaml`                                                                       | More powerful than universal placeholder list (catches `gwl=0`, out-of-range values); per-field flexibility                  |
| Plausibility scope                    | Applies uniformly to `nzgdrecord`, `cptreport`, `sptreport` enrichment                                               | Existing cross-record `_enrich_canonical_metadata` benefits from the same check; shared helper module                       |
| Source-file provenance                | Canonical keeps one `source_file`; absorbed source_files recorded in `dedup_audit` JSON                              | No schema change to `cptreport`; downstream consumers see one source_file per row                                            |
| Canonical selection within a cluster  | Swappable callable via config (default: prefer `has_data=True`, tiebreaker smallest `report_id`)                     | Allows future rule changes (e.g., "most measurement rows") without code changes                                              |
| `cpt_data_duplicate_of_cpt_id` column | Drop entirely from `cptreport` schema                                                                                | Now redundant; keeping it deprecated would invite confusion                                                                  |
| Pass order                            | Pass 0 (within-record) → Pass 1 (cross-record hash) → Pass 2 (cross-record fuzzy)                                    | Cross-record sees consolidated records; intuitive "clean up own room before tidying shared space"                            |

## Architecture

A new pass `nzgd/dedup/pass0_within_record.py` runs in the existing dedup CLI
(`nzgd/scripts/db/deduplicate.py`) before the cross-record passes. For each
`nzgd_id`, it clusters that record's `cptreport`/`sptreport` rows by source-file
stem and trace identity, then collapses each cluster to a single canonical row
with metadata enriched from the absorbed rows.

A new helper module `nzgd/dedup/plausibility.py` provides
`is_useful_value(value, table, column)`, consumed by Pass 0's enrichment and
by the existing cross-record `_enrich_canonical_metadata`.

A new helper module `nzgd/dedup/canonical_selectors.py` provides default and
alternative selector callables. Pass 0's entry point resolves the configured
selector by dotted path at runtime; tests inject mocks via the entry point's
keyword argument.

The extraction-time `remove_duplicate_extractions` is removed; the DB ingest
script `put_cpts_in_db.py` is simplified to insert every (file, sheet)
extraction uniformly. Initial DB ingest temporarily carries ~5% more
measurement rows that Pass 0 deletes.

## Schema changes (deduped output DB)

### `cptreport` — drop one column

```sql
ALTER TABLE cptreport DROP COLUMN cpt_data_duplicate_of_cpt_id;
```

Idempotent via try/except (`OperationalError: no such column` is caught).
SQLite has supported `DROP COLUMN` since 3.35 (2021). The project's dev venv
sits well above that.

### `dedup_audit` — widen the `match_pass` CHECK constraint

The existing constraint `match_pass TEXT NOT NULL CHECK(match_pass IN ('hash',
'fuzzy'))` must accept a new value `'within_record'`. SQLite doesn't support
modifying CHECK constraints in place, so the migration recreates the table:

```sql
CREATE TABLE dedup_audit_new (... match_pass TEXT NOT NULL
    CHECK(match_pass IN ('hash', 'fuzzy', 'within_record')) ...);
INSERT INTO dedup_audit_new SELECT * FROM dedup_audit;
DROP TABLE dedup_audit;
ALTER TABLE dedup_audit_new RENAME TO dedup_audit;
-- recreate indexes
```

Idempotent: only runs if introspection of the current table shows the CHECK
constraint lacks `'within_record'`.

## Plausibility helper

`nzgd/dedup/plausibility.py`:

```python
"""Per-field plausibility check shared by all dedup passes."""

from typing import Any

from nzgd import constants


def is_useful_value(value: Any, table: str, column: str) -> bool:
    """Return True if `value` is non-NULL and (if a range is configured) within range.

    `table` is 'nzgdrecord', 'cptreport', or 'sptreport'.
    Non-numeric values bypass the range check; only the non-NULL check applies
    to them. This lets text and date fields share the helper with numeric fields.
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
        return True  # text/date/non-numeric fields fall through
    return lo <= v <= hi
```

## Pass 0 — within-record consolidation

### Clustering rule

For each `nzgd_id` of the relevant `type_id` (CPT or SPT) with at least one
report row:

1. Group rows by **source-file stem** — the substring of
   `cptreport.source_file` (or `sptreport.source_file`) up to but not including
   `_sheet_`. Stems uniquely identify uploads (confirmed empirically: in the
   current 144,742-row cptreport, no stem spans multiple `nzgd_id`s).
2. For each stem, identify data-bearing rows (`has_cpt_data = 1` for CPT, or
   the analogous "has at least one row in sptmeasurements" for SPT). Hash each
   data-bearing trace via `compute_trace_hash`.
3. **Split stems with non-matching data-bearing rows.** If a stem has two or
   more data-bearing rows whose hashes differ AND whose pairwise fuzzy
   `trace_score` exceeds the threshold, split the stem into sub-stems — one
   per trace-identity sub-cluster. The stem's no-data rows attach to the
   sub-cluster containing the smallest cpt_id (deterministic heuristic;
   acknowledged limitation in §"Deferred").
4. Build cross-stem edges: an edge from (sub-)stem A to (sub-)stem B exists if
   any data-bearing row in A matches (by hash or fuzzy) any data-bearing row
   in B.
5. Connected components → clusters. Each cluster surviving consolidation
   yields one canonical row.

Multi-CPT-per-`nzgd_id` (rare) is handled naturally: two stems whose
data-bearing rows don't match produce two clusters → two surviving rows.

### Fuzzy predicate within Pass 0

The cross-record fuzzy predicate's spatial / date / name signals are
degenerate within an `nzgd_id` (all rows share the same `nzgdrecord` lat/lon,
investigation_date, investigation_name). Pass 0's predicate reduces to:

```
trace_score < thresholds["trace_score_max"]
```

The `trace_score_max` and `trace_resample_step_m` thresholds are read from
the existing `deduplication.fuzzy_pass` config block (no new threshold knobs).

### Canonical selection

`nzgd/dedup/canonical_selectors.py`:

```python
"""Pluggable canonical-selection rules for within-record consolidation."""

from dataclasses import dataclass
from typing import Callable, Sequence

from nzgd.dedup.data_types import TableConfig


@dataclass(frozen=True)
class ClusterRow:
    report_id: int                  # cpt_id or spt_id
    has_data: bool                  # has_cpt_data=1 for CPT; measurement_row_count>0 for SPT
    measurement_row_count: int      # 0 if no data
    metadata_non_null_count: int    # non-null cptreport/sptreport metadata fields


CanonicalSelector = Callable[[Sequence[ClusterRow], TableConfig], int]


def default_within_record_canonical(
    cluster_rows: Sequence[ClusterRow],
    table_cfg: TableConfig,
) -> int:
    """v1 default: prefer has_data=True; tiebreaker smallest report_id."""
    candidates = [r for r in cluster_rows if r.has_data] or list(cluster_rows)
    return min(candidates, key=lambda r: r.report_id).report_id
```

Pass 0's entry point:

```python
def generate_within_record_consolidation_plan(
    conn: sqlite3.Connection,
    table_cfg: TableConfig,
    thresholds: dict,
    canonical_selector: CanonicalSelector | None = None,
) -> list[WithinRecordConsolidation]:
    """Build a list of consolidation actions, one per cluster needing collapse.

    `canonical_selector` defaults to the callable resolved from
    `constants.DEDUP_CONFIG["within_record"]["canonical_selector"]` (dotted path
    resolved via importlib). Tests pass the callable directly.
    """
    ...
```

The CLI resolves the selector once per record-type pass.

### Metadata enrichment

For each cluster (after canonical is chosen), for each per-report metadata
column on the relevant report table:

1. Read canonical's value. If `is_useful_value(value, table, column)`, keep
   it.
2. Otherwise scan the cluster's other rows in ascending `report_id` order:
   - If a useful value is found, take the first one as the new canonical
     value and log it in `metadata_copied_json`.
   - If **no** useful value is found among any of the cluster's rows
     (canonical's value isn't useful AND no other row has a useful value
     either), **leave canonical's value unchanged**. Pass 0 does not NULL
     out unscrubbed sentinels. The plausibility check is consulted only
     when picking between competing values during a merge; it is not used
     to retroactively clean up values that have no useful alternative.
3. If multiple non-canonical rows have distinct useful values, record all
   candidates in `metadata_conflicts_json`; canonical takes the value from the
   smallest `report_id`.
4. UPDATE the canonical row with the chosen values in a single statement.
   Columns where no useful candidate was found are omitted from the UPDATE.

`max_depth_m` and `min_depth_m` are derived from measurements and will agree
across same-trace rows; the enrichment rule still applies but produces no
change in practice.

**The same "leave alone if no useful alternative" rule applies to the
cross-record `_enrich_canonical_metadata` after the helper migration.**
A canonical record whose `latitude` is, say, `-1.0` (out of NZ range) keeps
that value if no merged record has a plausible latitude either. Scrubbing
implausible values across the entire DB (independent of cluster merges) is
a separate operation not in scope here.

### Per-cluster execution

Inside a SAVEPOINT per cluster (mirroring the cross-record executor pattern):

1. UPDATE canonical row with enriched metadata.
2. For each absorbed `report_id`: reuse the existing
   `executor._delete_report(conn, report_id, table_cfg)` helper, which already
   handles the SPT dependent-table cascade (`soilmeasurementsoiltype` →
   `soilmeasurements` → `densitymeasurements` → `sptvs30estimates` →
   `sptmeasurements` → `sptreport`).
3. INSERT one audit row into `dedup_audit`:

| Column                    | Value                                                                                                                              |
| ---                       | ---                                                                                                                                |
| `cluster_id`              | Per-run sequential id                                                                                                              |
| `canonical_nzgd_id`       | The cluster's `nzgd_id`                                                                                                            |
| `merged_nzgd_id`          | Same `nzgd_id` (signals within-record merge)                                                                                       |
| `record_type`             | `'CPT'` or `'BH'`                                                                                                                  |
| `match_pass`              | `'within_record'`                                                                                                                  |
| `report_pairs_json`       | `[{"canonical_report_id": X, "absorbed_report_id": Y, "absorbed_source_file": "...", "trace_match": "hash" \| "fuzzy" \| "stem_only"}, ...]` |
| `metadata_copied_json`    | `{"<column>": {"value": v, "source_report_id": Z}, ...}` (per-report fields only; `nzgdrecord` enrichment happens in cross-record passes) |
| `metadata_conflicts_json` | `{"<column>": [{"value": v, "source_report_id": Z}, ...], ...}` or NULL                                                            |
| `merged_at`               | ISO timestamp                                                                                                                      |

4. RELEASE SAVEPOINT.

On exception within a cluster: ROLLBACK TO SAVEPOINT, RELEASE SAVEPOINT,
record the failure in the `dedup_failures.csv` collector if provided.

## Updates to cross-record passes

The existing `executor._enrich_canonical_metadata` switches from
`value is not None` to `is_useful_value(value, "nzgdrecord", col)` for the
canonical-value check and the candidate-scan check. This is a behaviour
change: previously-kept implausible values (e.g., a canonical with
`latitude = -1.0` outside the NZ range) now get overwritten by plausible
candidates from merged records. The plan includes a real-data validation
re-run to surface and document any cross-record merges whose outcome differs
from the existing deduped DB.

No other cross-record code changes are required.

## Outputs

Every run produces (alongside the deduped DB):

- `dedup_report.csv` — extended to include within-record rows; same columns as
  today (within-record rows show `match_pass=within_record` and
  `canonical_nzgd_id == merged_nzgd_id`).
- `cpt_calibration_report.csv` / `bh_calibration_report.csv` — unchanged from
  cross-record (Pass 0 doesn't write calibration data).
- `dedup_failures.csv` — unchanged.

## Testing strategy

Single integration layer, same philosophy as the cross-record dedup spec:
minimise tests, no implicit library testing, real composition behaviour
verified end-to-end against a synthetic SQLite DB.

Append scenarios to the existing `tests/dedup/test_dedup_pipeline.py`:

1. **Typical multi-sheet collapse** — single nzgd_id, 5 cptreport rows from 2
   source-file stems (mirrors real nzgd 8920). All data-bearing rows share a
   hash. Verify: 1 surviving row; canonical's source_file is the keeper's;
   audit row records all 4 absorbed cpt_ids and source_files.
2. **Sentinel-aware enrichment** — cluster of 2 rows: canonical has
   `extracted_gwl_m = 0` (within the `[0.01, 50.0]` range's *invalid* zone);
   absorbed has `extracted_gwl_m = 5.2`. Verify: canonical updated to 5.2,
   change logged in `metadata_copied_json`.
3. **Plausibility conflict** — cluster of 3 rows, canonical NULL on
   `tip_net_area_ratio`, two absorbed rows with conflicting in-range values
   (0.80 vs 0.92). Verify: smaller-cpt_id value wins; both candidates logged
   in `metadata_conflicts_json`.
4. **Multi-CPT-per-nzgd_id** — single nzgd_id with two source-file stems whose
   data-bearing rows have non-matching traces. Verify: 2 surviving rows; each
   cluster consolidated independently.
5. **All-no-data cluster** — single nzgd_id with 3 rows, all `has_cpt_data=0`,
   all sharing one stem. Verify: 1 surviving row (smallest cpt_id); per-report
   metadata merged from the 3.
6. **Stem-only attachment** — single nzgd_id with 3 rows: 1 data-bearing in
   stem A, 2 no-data in stem A, 1 no-data in stem B. Verify: 2 surviving rows
   (stem-A cluster collapses to 1; stem-B singleton survives).
7. **Fuzzy within-record match** — single nzgd_id with 2 data-bearing rows
   from different stems, traces perturbed at sub-row precision. Verify: stems
   link via fuzzy; single cluster; 1 surviving row.
8. **Pass 0 → Pass 1 interaction** — two nzgd_ids each with multiple
   within-record duplicates, and the two nzgd_ids are cross-record duplicates
   of each other. Run the full CLI. Verify: Pass 0 consolidates each
   independently; Pass 1 then merges the two consolidated records; final
   deduped DB has one canonical with all unique data; audit rows reflect both
   passes.
9. **`cpt_data_duplicate_of_cpt_id` removal** — fixture loads a DB that has
   the legacy column; Pass 0's schema migration drops it. Verify: column
   absent post-migration; Pass 0 still works; existing dedup_audit rows
   migrated through the CHECK-constraint widening.
10. **SPT within-record consolidation** — analogous to scenario 1 but with
    sptreport rows. Verify: deletion cascade through `soilmeasurements`,
    `densitymeasurements`, `soilmeasurementsoiltype`.
11. **Single-file multi-CPT split** — single nzgd_id with one source-file stem
    containing 4 rows: 2 data-bearing with non-matching traces (sub-clusters A
    and B); 2 no-data. Verify: stem splits into 2 sub-clusters; both no-data
    rows attach to the sub-cluster with the smallest cpt_id; result is 2
    consolidated rows.
12. **Cross-record plausibility-aware enrichment** — two CPT records that
    hash-match. Canonical's `latitude = -1.0` (out-of-range); absorbed
    record's `latitude = -41.0` (plausible). Verify: canonical updated to
    -41.0; audit logs the change. Confirms the cross-record `_enrich`
    function uses the new plausibility helper.
13. **Sentinel preserved when no useful alternative exists** — cluster of 2
    cptreport rows: canonical has `extracted_gwl_m = 0` (out of the
    `[0.01, 50.0]` range); absorbed row also has `extracted_gwl_m = 0`.
    Verify: canonical's `extracted_gwl_m` remains 0 after Pass 0 runs (not
    changed to NULL); `metadata_copied_json` contains no entry for
    `extracted_gwl_m`. Confirms Pass 0 does not retroactively clean up
    values that have no useful alternative.

Thirteen new scenarios. Total test file goes from 10 → 23 tests. Estimated
runtime: ~6 seconds.

### Behaviour we explicitly do **not** test

(Inherited from the cross-record spec; reiterated here.)

- `blake2b` determinism, `struct.pack` bytes, `BallTree` neighbour queries,
  `scipy.sparse.csgraph.connected_components` correctness, `numpy.interp`,
  `rapidfuzz.token_set_ratio`, SQLite transactions / FK enforcement,
  `pandas.to_csv`.
- `importlib`-based dotted-path resolution (stdlib).

### Real-data validation

Manual: run the full CLI against the production DB after the rewrite. Inspect:

- Pass 0 merge counts per `record_type` (CPT vs BH).
- Cross-record merge counts compared to the existing deduped DB
  (`uc_nzgd_v0p6p0_20260403_deduped.db`). Any merges whose outcome differs are
  due to the new plausibility-aware enrichment; spot-check a sample to verify
  the new outcomes are correct.
- `metadata_copied_json` contents for a sample of Pass 0 rows to confirm the
  plausibility-range filtering behaves as intended.

## Config additions

New section in `nzgd/resources/config.yaml`:

```yaml
deduplication:
  within_record:
    canonical_selector: nzgd.dedup.canonical_selectors.default_within_record_canonical

  field_plausibility_ranges:
    nzgdrecord:
      latitude:                                  [-47.5, -33.5]      # NZ bounds
      longitude:                                 [165.0, 180.0]      # incl. Chatham Is.
      model_vs30_foster_2019_km_per_s:           [50.0, 2000.0]      # broad — catches 0 / negatives
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

Loaded into `constants.DEDUP_CONFIG` per existing pattern. Range values are
starting points — refine after the first real-data run if false-positives
appear.

## Module layout

```
nzgd/dedup/
    __init__.py                          # unchanged
    config.py                            # n/a (still inline via constants)
    data_types.py                        # unchanged
    schema.py                            # extended: DROP COLUMN + widen CHECK
    fingerprint.py                       # unchanged
    cluster.py                           # unchanged
    selection.py                         # unchanged (cross-record canonical)
    canonical_selectors.py               # NEW — within-record selector callables
    plausibility.py                      # NEW — is_useful_value(value, table, column)
    pass1_hash.py                        # unchanged
    pass2_fuzzy.py                       # unchanged
    pass0_within_record.py               # NEW — within-record consolidation
    executor.py                          # tiny edit: use is_useful_value
    reports.py                           # tiny edit: handle within-record rows

nzgd/scripts/db/
    deduplicate.py                       # extended: invoke Pass 0 before Pass 1 for both CPT and SPT

nzgd/extract/cpt/
    workflow.py                          # remove call to remove_duplicate_extractions
    conditioning.py                      # remove the function itself
    output.py                            # remove removed_duplicates string handling

nzgd/scripts/db/
    put_cpts_in_db.py                    # simplified: insert all (file, sheet) rows uniformly

tests/dedup/
    conftest.py                          # unchanged
    test_dedup_pipeline.py               # +12 scenarios
```

New runtime dependency: none. Reuses existing `scipy`, `sklearn`, `rapidfuzz`,
`numpy`, `pandas`.

## Migration / removing old code path

The plan must include explicit steps for each:

- **Schema migration** (`nzgd/dedup/schema.py`):
  - Drop `cptreport.cpt_data_duplicate_of_cpt_id`. Idempotent via try/except.
  - Widen `dedup_audit.match_pass` CHECK constraint to include
    `'within_record'`. Idempotent: check current schema first via
    `PRAGMA table_info` + `sqlite_master.sql` inspection; only recreate if
    the existing constraint omits `'within_record'`.

- **Extraction pipeline**:
  - Remove the call to `remove_duplicate_extractions` in
    `nzgd/extract/cpt/workflow.py`.
  - Remove the `remove_duplicate_extractions` function from
    `nzgd/extract/cpt/conditioning.py`.
  - Remove `removed_duplicates` field handling in
    `nzgd/extract/cpt/output.py`'s `write_extracted_data`.

- **DB ingest script** (`nzgd/scripts/db/put_cpts_in_db.py`):
  - Remove the keeper/duplicate branching; insert all (file, sheet) extractions
    uniformly with `has_cpt_data` set per whether the extraction produced
    measurement rows.
  - Stop populating `cpt_data_duplicate_of_cpt_id` (column won't exist after
    migration).

- **Grep for legacy column readers**: any analysis script or query reading
  `cpt_data_duplicate_of_cpt_id` must have the read removed or replaced. Plan
  step: grep the repo, list hits, edit accordingly.

- **Cross-record executor edit** (`nzgd/dedup/executor.py`):
  - In `_enrich_canonical_metadata`, swap the canonical-value and
    candidate-scan checks from `value is not None` to
    `is_useful_value(value, "nzgdrecord", col)`. ~5-line diff.

- **CLI** (`nzgd/scripts/db/deduplicate.py`):
  - Invoke Pass 0 for each `TableConfig` before invoking Pass 1, in the
    existing record-type loop.

- **Real-data validation re-run**:
  - Delete previously-generated `_deduped.db` and report CSVs.
  - Run the full CLI against the production DB.
  - Compare the new `dedup_report.csv` against the existing one. Surface any
    cross-record merges that differ (due to plausibility-aware enrichment).
    Spot-check the diffs to confirm the new outcomes are correct.

## Deferred — not in v1 scope

Items deliberately not implemented in v1. The design accommodates them but
adding them now would expand scope unnecessarily.

### Better no-data row attribution within split stems

When a stem splits into multiple sub-clusters (single-file-multi-CPT case),
no-data rows attach to the sub-cluster with the smallest `cpt_id`. This is
deterministic but imperfect; perfect attribution would require sheet-name
pattern matching, which is fragile across upload formats. Revisit if real-data
validation surfaces miscategorised rows.

### Cross-record canonical-selector pluggability

The pattern Pass 0 introduces (config-resolved dotted-path callable) could
extend to the cross-record `select_canonical`. Not in scope; one-diff
extension if a use case emerges.

### Additional SPT plausibility ranges

v1 ships ranges only for `sptreport.extracted_gwl_m`. Fields like
`efficiency`, `borehole_diameter`, `casing_diameter` get the non-NULL check
only. Add ranges as concrete sentinel cases emerge from real SPT data.

### Auto-tuning plausibility ranges

Ranges are hand-set in config. Could be derived empirically from the
distribution of non-NULL values across the production DB. Out of scope; manual
tuning is sufficient at current scale.
