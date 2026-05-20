# CPT/SPT Cross-Record Deduplication Design

Date: 2026-05-20
Status: Approved for plan-writing

## Background

The NZGD package extracts geotechnical investigation data from heterogeneous
source files and loads it into a SQLite database. Downstream users have reported
that the same physical investigation is sometimes uploaded under multiple NZGD
IDs, causing data duplication that contaminates analyses. This design adds a
deduplication step that identifies cross-NZGD-ID duplicates and consolidates
them into single canonical records while preserving full provenance.

The existing `cpt_data_duplicate_of_cpt_id` column tracks *within-record*
duplicates only (3,453 rows in the current dev DB, all same-`nzgd_id` pairs).
That semantics is preserved untouched. This design adds *cross-record* dedup as
a separate, complementary mechanism.

## Scope

- In scope: cross-NZGD-ID dedup of CPT records and SPT (borehole) records.
- Out of scope: dedup against external data sources; review-band /
  human-approval flow; reversibility of merges within a single deduped DB
  (reversibility comes from re-running against the always-preserved source DB).

## Decisions settled during design

| Decision                       | Choice                                                                                                                                | Reason                                                                                                                                |
| ---                            | ---                                                                                                                                   | ---                                                                                                                                   |
| Match signal                   | Two-pass: (1) hash-based exact match, (2) metadata-blocked fuzzy match                                                                | Hash pass is O(N) and finds byte-identical re-uploads with zero false positives; fuzzy pass catches re-processed re-uploads.          |
| Auto-merge conservatism        | Aggressive — auto-merge all detected duplicates from both passes                                                                      | User accepted threshold risk in exchange for completeness; audit ledger plus always-preserved source DB provide reversibility.        |
| Record-merge granularity       | Liberal — any matching report between two records triggers a record-level merge                                                       | Hash matches and well-calibrated fuzzy thresholds make false positives unlikely; user explicitly chose liberal over conservative.     |
| Canonical selection            | (1) Highest count of unique (non-shared) measurement rows; (2) most non-null `nzgdrecord` fields; (3) smallest `nzgd_id`              | "Most unique actual data" semantics from the user, with stable tiebreakers.                                                           |
| Duplicate data handling        | DELETE duplicate report rows + their dependent rows; the duplicate's `nzgdrecord` row stays with `merged_into_nzgd_id` set            | Matches existing precedent (within-record duplicates already physically lack measurements); preserves provenance via the audit table. |
| Source DB mutation             | Never — always copy source → mutate the copy                                                                                          | Safety and cheap iteration during threshold tuning. DB is only ~5 GB.                                                                 |
| Trace fingerprint precision    | Bit-exact (`struct.pack('<d', v)`), no rounding                                                                                       | Hash pass must be zero-false-positives; tolerance lives in the fuzzy pass.                                                            |
| Match-decision combination     | Single conjunctive predicate over the feature vector, configurable in `config.yaml`                                                   | User chose aggressive auto-merge — no review band, just a match/no-match decision.                                                    |
| `--dry-run` flag               | Not implemented                                                                                                                       | Copy-source-mutate-copy already provides equivalent safety; can be added later in ~20 LOC if needed.                                  |

## Architecture

A new module `nzgd/dedup/` and a runnable script
`nzgd/scripts/db/deduplicate.py`. The script reads a source SQLite DB
read-only, copies it to a target path (`<source>_deduped.db` by default), and
applies merges in-place on the copy. The source DB is never mutated. Re-runs
with different thresholds are cheap: discard the target, re-run.

The same code path handles both CPT and SPT, parameterised over
`(record_table, report_table, measurement_table, id_column, value_columns,
dependent_tables)`.

Two sequential passes:

1. **Hash pass** — exact-match via per-trace fingerprint hashing.
2. **Fuzzy pass** — metadata-blocked near-duplicate matching over the
   post-hash survivor set.

Each pass produces a `list[MergePlanEntry]` consumed by a shared merge
executor. The executor is the only component that writes to the deduped DB.

A `dedup_report.csv` is always written alongside the deduped DB. A
`calibration_report.csv` is written before the fuzzy pass applies thresholds
(see Pass 2 below).

## Schema changes (deduped output DB only)

### `nzgdrecord` — one new column

```sql
ALTER TABLE nzgdrecord
    ADD COLUMN merged_into_nzgd_id INTEGER REFERENCES nzgdrecord(nzgd_id);
CREATE INDEX idx_nzgdrecord_merged_into ON nzgdrecord(merged_into_nzgd_id);
```

`NULL` = canonical or independent. Set = this `nzgd_id` was absorbed into the
referenced canonical. A lookup by an obsolete `nzgd_id` still returns a row,
with the redirect explicit in the new column.

SQLite supports `ADD COLUMN … REFERENCES …` only when the new column's default
is `NULL`, which is the case here. Foreign-key enforcement requires the
connection to issue `PRAGMA foreign_keys = ON` after opening; the script does
this before any DDL or mutation.

### New `dedup_run` table — one row per script invocation

```sql
CREATE TABLE dedup_run (
    run_id              INTEGER PRIMARY KEY,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    source_db_path      TEXT NOT NULL,
    script_version      TEXT NOT NULL,
    config_snapshot_json TEXT NOT NULL,
    n_clusters_merged   INTEGER,
    n_records_merged    INTEGER
);
```

Records the configuration and script version that produced a given set of
merges, for reproducibility. Typically one row per deduped DB.

### New `dedup_audit` table — one row per (canonical, merged) pair

```sql
CREATE TABLE dedup_audit (
    audit_id                INTEGER PRIMARY KEY,
    run_id                  INTEGER NOT NULL REFERENCES dedup_run(run_id),
    cluster_id              INTEGER NOT NULL,
    canonical_nzgd_id       INTEGER NOT NULL,
    merged_nzgd_id          INTEGER NOT NULL,
    record_type             TEXT NOT NULL CHECK(record_type IN ('CPT', 'BH')),
    match_pass              TEXT NOT NULL CHECK(match_pass IN ('hash', 'fuzzy')),
    report_pairs_json       TEXT NOT NULL,
    metadata_copied_json    TEXT,
    metadata_conflicts_json TEXT,
    merged_at               TEXT NOT NULL
);
CREATE INDEX idx_dedup_audit_canonical ON dedup_audit(canonical_nzgd_id);
CREATE INDEX idx_dedup_audit_merged    ON dedup_audit(merged_nzgd_id);
CREATE INDEX idx_dedup_audit_cluster   ON dedup_audit(cluster_id);
```

A 3-way cluster produces 2 audit rows sharing a `cluster_id`. The CSV report is
this table flattened with JSON columns expanded.

### Existing tables — runtime mutation only, no schema change

The existing `cpt_data_duplicate_of_cpt_id` column on `cptreport` is **not
touched** — it continues to carry its current within-record semantics.

## Pass 1 — hash-based exact match

### Per-record fingerprint

For each `cpt_id`:

1. Stream rows: `SELECT depth_m, qc_MPa, fs_MPa, u2_MPa FROM cptmeasurements
   ORDER BY cpt_id, depth_m`. Index on `(cpt_id, depth_m)` is added before the
   sweep if it does not exist. Group in Python.
2. Serialise each value: `NULL` or `NaN` → 8-byte sentinel
   `b'\x00\x00\x00\x00NaN_'`; finite floats → `struct.pack('<d', v)` (IEEE 754
   raw bytes, no rounding).
3. `hashlib.blake2b(digest_size=16)` over the concatenated bytes.

Same scheme for SPT with `(depth_m, ISPT_MAIN, ISPT_NVAL, ISPT_REP)`.

### Cluster formation

- `GROUP BY hash` over the (cpt_id → hash) mapping → drop singleton buckets.
- For each multi-`nzgd_id` bucket, add the involved `nzgd_id`s as edges in a
  graph (union-find).
- Connected components → record-level clusters.

Transitive matches are handled correctly: if A↔B via report X and A↔D via
report Y, then `{A, B, D}` forms one cluster.

### Canonical selection within a cluster

For each candidate record in the cluster, define its "unique data": the
measurement rows belonging to reports whose `cpt_id` (or `spt_id`) does **not**
appear in any matched pair identified by this pass. (For Pass 1, "matched
pair" means hash-equal; for Pass 2, it means predicate-positive. Both passes
reuse the same selection function with the pass's matched-pair set as input.)

1. **Primary**: highest count of measurement rows in the candidate's unique
   data.
2. **Tiebreaker 1**: most non-null `nzgdrecord` columns.
3. **Tiebreaker 2**: smallest `nzgd_id`.

### Output

A `list[MergePlanEntry]` with `match_pass='hash'`, fed to the merge executor.

## Pass 2 — metadata-blocked fuzzy match

Operates only on records with `merged_into_nzgd_id IS NULL` after Pass 1.

### Blocking

`sklearn.neighbors.BallTree(metric='haversine')` over post-Pass-1 canonicals'
`(lat, lon)` (converted to radians). For each record, query neighbours within
`spatial_radius_m` (default 50). Separate trees for CPT and SPT.

### Per-pair feature vector

For each blocked pair:

| Feature             | Definition                                                                                                                                            |
| ---                 | ---                                                                                                                                                   |
| `spatial_m`         | Haversine distance, metres                                                                                                                            |
| `date_days`         | `abs(investigation_date_a - investigation_date_b)`; `NULL` on either side → sentinel                                                                  |
| `name_sim`          | `rapidfuzz.fuzz.token_set_ratio(name_a, name_b)`, 0–100; `NULL` on either side → sentinel                                                              |
| `max_depth_diff_m`  | `abs(max_depth_m_a - max_depth_m_b)`                                                                                                                  |
| `trace_score`       | Best cross-record `(report_id, report_id')` similarity score. For each candidate report-pair: resample both traces to a common 0.05 m depth grid over the overlapping depth range, compute per-channel normalised RMSE on `qc`, `fs`, `u2` (CPT) or N values (SPT), sum-of-RMSEs. |

`trace_score` reflects the liberal-merge rule: one matching report is
sufficient to merge the records.

### Match predicate

Single conjunctive predicate, all thresholds configurable in `config.yaml`:

```
spatial_m < spatial_radius_m
AND (date_days < date_window_days OR date is sentinel on either side)
AND (name_sim > name_similarity_min OR name is sentinel on either side)
AND trace_score < trace_score_max
```

Missing date or name is treated as "no signal"; the trace and spatial signals
must still pass on their own.

### Calibration step (precedes threshold application)

Before merging, the script writes a `calibration_report.csv` containing the
feature-vector distribution for two reference groups:

- All Pass-1 hash matches (known-positive examples).
- A random sample of blocked-but-non-hash pairs
  (`random_pair_sample_size`, default 5000) (likely-negative examples).

This gives an empirical basis for setting thresholds: pick thresholds looser
than the typical hash-match metadata but tighter than the random-pair
distribution. Default thresholds in `config.yaml` are starting points; the
first real run produces the calibration report and the user adjusts before the
next run.

### Cluster formation, canonical selection

Identical to Pass 1: union-find over the edges where the match predicate
fires, then the same canonical-selection function. The matched-pair set fed
into canonical selection is the set of cross-record `(report_id, report_id')`
pairs whose `trace_score` was the best for their pair of records and which
fired the predicate.

### Per-cluster merge ordering

Within a fuzzy cluster of >2 records, merges are applied in score order
(best-match-first). The canonical's enriched metadata is recomputed after each
merge — so a later merge sees the post-enrichment canonical.

### Output

A `list[MergePlanEntry]` with `match_pass='fuzzy'`, fed to the merge executor.

## Merge executor

Takes a `list[MergePlanEntry]` and applies each cluster atomically.

### Per cluster

1. `BEGIN TRANSACTION`.
2. **Enrich canonical's `nzgdrecord` metadata**, field-by-field:
   - Canonical's value non-NULL → keep (canonical wins).
   - Canonical's value NULL → scan merged records' values for this field:
     - All non-NULLs agree → copy that value to canonical; record it in
       `metadata_copied_json` with the source `nzgd_id`.
     - Multiple distinct non-NULL values → record all candidates in
       `metadata_conflicts_json`, leave canonical NULL.
3. **For each merged record in the cluster** (CPT case; SPT is analogous):
   - For each matched `(canonical_cpt_id, merged_cpt_id)` pair:
     `DELETE FROM cptvs30estimates WHERE cpt_id = ?` (currently empty,
     future-proofing), then `DELETE FROM cptmeasurements WHERE cpt_id = ?`,
     then `DELETE FROM cptreport WHERE cpt_id = ?`.
   - For each unique `cpt_id` (no match in canonical):
     `UPDATE cptreport SET nzgd_id = canonical_nzgd_id WHERE cpt_id = ?`.
     `UPDATE cptvs30estimates SET nzgd_id = canonical_nzgd_id WHERE cpt_id = ?`.
     Measurements stay attached via `cpt_id` and need no rewrite.
   - `UPDATE nzgdrecord SET merged_into_nzgd_id = canonical_nzgd_id
     WHERE nzgd_id = merged_nzgd_id`.
   - `INSERT INTO dedup_audit (...)`.
4. `COMMIT`. On any error: `ROLLBACK`, log the failed cluster, continue to the
   next cluster.

### SPT-specific deletion ordering

SPT reports have dependent rows in `soilmeasurements`, `densitymeasurements`,
`soilmeasurementsoiltype` (M-N via `soil_measurement_id`), and
`sptvs30estimates`. When DELETing a duplicate `spt_id`, the script deletes in
this order inside the same transaction:

1. `soilmeasurementsoiltype` rows whose `soil_measurement_id` ties to a
   `soilmeasurements` row with this `spt_id`.
2. `soilmeasurements` rows with this `spt_id`.
3. `densitymeasurements` rows with this `spt_id`.
4. `sptvs30estimates` rows with this `spt_id` (currently empty,
   future-proofing).
5. `sptmeasurements` rows with this `spt_id`.
6. `sptreport` row with this `spt_id`.

When re-parenting a unique `spt_id` (UPDATE `nzgd_id`), nothing else needs to
change — all dependent rows tie via `spt_id`, which is unchanged.

### Per-cluster atomicity

A single bad cluster (e.g., schema violation, unexpected NULL) rolls back only
that cluster's transaction, not the whole run. The failure is logged and
recorded as a row in a `dedup_failures.csv` output, with the cluster's
`MergePlanEntry` JSON-serialised so a human can investigate.

## Outputs

Every run produces, in the same directory as the deduped DB:

- The deduped DB itself (e.g., `<source>_deduped.db`).
- `dedup_report.csv` — `dedup_audit` flattened to a portable file. One row per
  merged `nzgd_id` pair, with `match_pass`, similarity metrics, and copied
  metadata fields.
- `calibration_report.csv` — feature distributions for hash-match positives
  and random-pair negatives, to support fuzzy-threshold tuning.
- `dedup_failures.csv` — any clusters whose transaction rolled back, with the
  full `MergePlanEntry` JSON.

## Testing strategy

`tests/dedup/`:

1. **Unit** — pure functions:
   - Fingerprint determinism and NaN/NULL canonicalisation.
   - Canonical selection on mocked clusters (each tiebreaker exercised).
   - Trace resampling and RMSE computation.
   - Match-predicate evaluation with sentinel-handling cases.
   - Union-find correctness.
2. **Integration** — a small synthetic SQLite DB built in `conftest.py` with
   deliberately constructed scenarios:
   - Exact duplicates (hash pass merges them).
   - Slight perturbations passing fuzzy thresholds (fuzzy pass merges).
   - Nearby-but-distinct sites (no merge).
   - 3-way transitive cluster via hash.
   - Partial-overlap cluster (some reports re-parented, some deleted).
   - Multi-field metadata conflict (canonical NULL, multiple distinct values
     from merged records — recorded in `metadata_conflicts_json`).
   - SPT cluster with `soilmeasurements`/`densitymeasurements` rows — verify
     deletion ordering.
   Run script end-to-end; assert against expected `nzgdrecord`, `dedup_audit`,
   and dependent-table state in the output DB.
3. **Real-data validation** — manual: run against a copy of the production DB,
   inspect the first N merges in `dedup_report.csv`, spot-check source files.
   Used during initial threshold calibration; not a regression test.

## Config additions

New `deduplication` section in `nzgd/resources/config.yaml`:

```yaml
deduplication:
  hash_pass:
    stream_chunk_size: 100000

  fuzzy_pass:
    spatial_radius_m: 50
    date_window_days: 90
    name_similarity_min: 80
    trace_score_max: 0.05
    trace_resample_step_m: 0.05

  calibration:
    random_pair_sample_size: 5000

  output:
    deduped_db_suffix: "_deduped"
    report_filename: "dedup_report.csv"
    calibration_report_filename: "calibration_report.csv"
    failures_filename: "dedup_failures.csv"
```

Loaded into `nzgd/constants.py` per the existing project pattern.

### New runtime dependency

- `rapidfuzz` — for `token_set_ratio`.

### Existing dependencies used

- `scikit-learn` (BallTree), `numpy`, `pandas`, `hashlib` (stdlib),
  `struct` (stdlib), `sqlite3` (stdlib), `tqdm`.

## Module layout

```
nzgd/dedup/
    __init__.py
    config.py             # loads dedup config from constants
    fingerprint.py        # hash function, value normalisation
    pass1_hash.py         # hash sweep, cluster formation
    pass2_fuzzy.py        # blocking, feature extraction, predicate, clustering
    selection.py          # canonical selection (shared by both passes)
    executor.py           # merge plan execution + audit logging
    cluster.py            # union-find utility
    reports.py            # CSV/calibration/failures report writing
    schema.py             # new column/table DDL

nzgd/scripts/db/
    deduplicate.py        # CLI entry point: --source --target [--cpt-only|--spt-only]

tests/dedup/
    conftest.py           # synthetic-DB fixture
    test_fingerprint.py
    test_selection.py
    test_pass1_hash.py
    test_pass2_fuzzy.py
    test_executor.py
    test_end_to_end.py
```

## Open considerations / future work

- **Reviewable band**: if aggressive auto-merge produces false positives in
  practice, add a middle band where pairs are emitted to a
  `pending_dedup_review.csv` instead of being auto-merged. Schema and code
  changes are localised.
- **Pipeline integration**: today the package has no integrated end-to-end
  pipeline. The script is intended to be runnable both standalone and as a
  future pipeline stage; the input/output contract is a SQLite DB path.
- **Calibration loop**: the first run produces `calibration_report.csv` but
  expects a human to update `config.yaml` and re-run. If a second iteration
  becomes routine, consider a `--auto-calibrate` mode that picks thresholds
  from the calibration distributions automatically.
- **Cross-record dedup of derived tables**: `cptvs30estimates` and
  `sptvs30estimates` are empty in the current DB but already linked into the
  delete/re-parent logic so future population is safe.
