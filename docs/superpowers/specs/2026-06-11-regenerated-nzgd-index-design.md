# Regenerated NZGD Index Design

Date: 2026-06-11
Status: Approved for plan-writing

## Background

The NZGD index file `nzgd/resources/nzgd_metadata_from_coordinates_22_august_2025.csv`
maps NZGD IDs to investigation types and carries per-record metadata. It is
load-bearing: the CPT/SPT extraction scripts use it to decide which records to
process, the DB scripts use it for record metadata and the location support
tables, and `ags_miner` uses it for `InvestigationId` lookups.

It is also frozen. It was produced once by
`assemble_nzgd_metadata_from_coordinates.py` (in the retired
`nzgd_data_extraction` repo, with a copy in `nzgd/scripts/temp/`) from inputs
snapshotted on 22 August 2025, and there is no working path to update it. The
script that looks like an updater,
`update_nzgd_metadata_for_past_and_current_nzgd_investigations.py`, is in fact
a downstream consumer and is currently broken: it reads a `.csv.gz` variant of
the index that does not exist and a retired `api_nzgd` catalog path. Its
output (`...with_nlm_gwl.csv.gz`, gitignored) feeds
`make_metadata_summary_csv.py`, which produces the per-ID metadata summary CSV
(`uc_nzgd_metadata_summary_<date>.csv`).

Meanwhile the ground truth has moved: `api_nzgd` syncs now maintain
`nzgd/resources/nzgd_catalogs_from_api/current_nzgd_investigation_catalog.csv.gz`
(git-tracked, updated per sync, preservation-aware). During the June 2026 sync
walkthrough, updating the index was a missed step — there was nothing to run.

This design replaces the frozen file with a regenerated, undated, git-tracked
index built from the catalog plus locally computed enrichments, and corrects
several inaccurate column names found along the way.

## Investigation findings the design relies on

Established empirically on 2026-06-11 (catalog as of the 2026-05-26 pull):

- The index's 197,647 IDs are a strict subset of the catalog's 197,649. The
  catalog adds IDs 230469 and 230470. No ID exists only in the index.
- 197,644 of the common rows are identical across all 26 shared columns.
  Exactly 3 differ:
  - **16**: index State (`no_longer_in_nzgd`) is stale; the record returned to
    NZGD and is Published in the catalog.
  - **229775** (was Trial pit `TP-1`, now Rotary-cored borehole) and
    **229822** (was CPT `CPT03`, now Sonic core `BH01`): true NZGD ID
    reassignments. The old rows survive only in the index — and in the
    archive at
    `/home/arr65/data/nzgd/downloads/nzgd_source_files_of_overwritten_nzgd_ids/{16,229775,229822}/`
    (saved metadata row per ID, plus old source files for 229822), and in the
    index's own git history (`20933a0`).
- Every active consumer loads the index via `constants.INDEX_FILE_PATH`
  (config key `nzgd_index_file_name`) except `put_nzgd_metadata.py`, which
  hardcodes the filename. Columns actually consumed: `nzgd_id`, `Type`,
  `TypeDisplay`, `InvestigationId`, `Latitude`/`Longitude`,
  `CreatedOn`/`LastModifiedOn`, and `region`/`district`/`city`/`suburb`.
- The index's baked `nztm_*` and `model_*` columns are unused by the active
  pipeline: `put_nzgd_metadata.py` re-samples the rasters itself at run time.
- All enrichments are locally recomputable, with no NZGD API involvement:
  LINZ district and suburb shapefiles are on disk
  (`/home/arr65/data/nzgd/resources/shapefiles/`), the Westerhoff/NLM/Foster
  GeoTIFF paths are already in `config.yaml`, and NZTM conversion is
  `qcore.coordinates.wgs_depth_to_nztm`.
- The two on-disk copies of the index (nzgd repo and `nzgd_data_extraction`
  repo) are md5-identical. `nzgd_data_extraction` is retired and may break.
- Foster GeoTIFF band evidence: band 1 ("Vs30") spans 193–770 with median
  628 — metres per second, not km/s. Band 2 ("Standard Deviation") spans
  0.17–0.85 with median 0.33 — the dimensionless natural-log sigma published
  by Foster et al. (2019), not a velocity.
- Date-field evidence (ID 16): NZGD `CreatedOn` is 2012-06-13 while the
  investigation's `EndDate` is 2011-05-12. `CreatedOn` is when the record
  entered NZGD, not when the investigation was conducted — so the DB column
  `investigation_date` (fed from `CreatedOn`) is misnamed.

## Scope

In scope:

- A build script that regenerates the index from the catalog after each NZGD
  sync, plus the shared location/raster modules it needs.
- A new git-tracked index `nzgd/resources/nzgd_index.csv.gz` and a new
  git-tracked location sidecar `nzgd/resources/nzgd_id_to_location.csv.gz`.
- Column-name corrections (Vs30 units, Vs30 sigma units, the two date
  columns) applied across the nzgd repo: ORM, DB writers, summary SQL, dedup
  column lists, config plausibility keys, tests.
- Repointing `make_metadata_summary_csv.py` at the new index and deleting the
  broken `update_nzgd_metadata_...` intermediate step.
- A one-time verification script proving no information is lost before any
  old artifact is removed.
- Workflow documentation: a "rebuild the index" stage appended to the
  `api_nzgd` sync README.

Not in scope:

- The map webapps (`nzgd_map`, `nzgd_gwl_map`): their DB schema expectations
  are already incompatible with the current DB and will be addressed in
  separate focused work.
- The retired `nzgd_data_extraction` and `codex_nzgd` checkouts (the latter is
  just a second clone of this repo). They may break.
- Renaming `model_gwl_*_m` to a "depth" vocabulary. Reviewed and kept: the
  values are water-table depth in metres below ground; the unit (m) is
  correct, and "GWL = x m bgl" is the convention used consistently across the
  codebase (including `extracted_gwl_m`).
- Adding an investigation end-date column to the DB (an enhancement, not a
  naming fix).
- Rewriting historical specs/plans under `docs/superpowers/` that mention the
  old column names; they describe past states.
- Cleaning the inherited junk `Type` values (`Investigation Type`,
  `Investigation Type 2`) present in both catalog and index; preservation
  policy says pass them through.

## Design

### Data flow

```
current_nzgd_investigation_catalog.csv.gz   (ground truth, from api_nzgd sync)
        │
        │   nzgd_id_to_location.csv.gz  (location sidecar, git-tracked)
        │   LINZ shapefiles              (local; classify new IDs only)
        │   GeoTIFFs: Westerhoff, NLM ×2, Foster   (local rasters)
        ▼
nzgd/scripts/metadata/build_nzgd_index.py   (one command, run after each sync)
        ▼
nzgd/resources/nzgd_index.csv.gz        (undated, git-tracked)
        ▼
all existing consumers via config.yaml `nzgd_index_file_name`
```

### Artifact: `nzgd/resources/nzgd_index.csv.gz`

37 columns:

- The 26 catalog columns verbatim, with `Id` renamed to `nzgd_id`: `State`,
  `InvestigationId`, `Type`, `TypeDisplay`, `Latitude`, `Longitude`,
  `Northings`, `Eastings`, `EpsgCode`, `FinalDepth`, `GroundLevel`,
  `MethodOfGroundLevel`, `MethodOfGroundLevelDisplay`,
  `HasGroundImprovementConducted`, `HasReportedIssues`, `ElevationDatum`,
  `ElevationDatumDisplay`, `Remarks`, `LocationDescription`,
  `MethodOfLocation`, `MethodOfLocationDisplay`, `ProjectId`, `EndDate`,
  `CreatedOn`, `LastModifiedOn`.
- `region`, `district`, `city`, `suburb` (from the sidecar).
- `nztm_y`, `nztm_x`.
- Five model columns under corrected names: `model_gwl_westerhoff_2018_m`,
  `model_gwl_nlm_2025_m`, `model_gwl_nlm_2025_stddev_m`,
  `model_vs30_foster_2019_m_per_s`, `model_vs30_stddev_foster_2019_ln`.

Gains over the old index: the NLM columns are included (previously they
existed only in the `with_nlm_gwl` intermediate), and DB-aligned names remove
the current summary-CSV quirk where the same quantity appears as two columns
(one from the index, one from the DB).

### Artifact: `nzgd/resources/nzgd_id_to_location.csv.gz`

Columns: `nzgd_id`, `region`, `district`, `city`, `suburb`. Git-tracked and
append-only: rows, once written, are never modified by tooling. Seeded once
from the old index's location columns, which preserves the manually assembled
classification batches (the "regions found on rch" work) as a first-class
artifact. On each build, only IDs absent from the sidecar are classified.

A sidecar (rather than carrying location forward from the previous index)
keeps the index a pure, reproducible function of
(catalog + sidecar + rasters), and gives the manual work a clear home.

### Column renames

Pure renames — the stored values are unchanged in all four cases, so dedup
date-window behavior and all numeric comparisons are unaffected.

| Old name | New name | Reason |
| --- | --- | --- |
| `model_vs30_foster_2019_km_per_s` | `model_vs30_foster_2019_m_per_s` | values are m/s |
| `model_vs30_stddev_foster_2019_km_per_s` (DB) / `model_vs30_std_foster_2019` (old index) / `model_vs30_std_foster_2019_km_per_s` (writer-internal) | `model_vs30_stddev_foster_2019_ln` | band 2 is dimensionless natural-log sigma |
| `investigation_date` | `record_created_on` | holds NZGD `CreatedOn` (record creation, not investigation date) |
| `published_date` | `record_last_modified_on` | holds NZGD `LastModifiedOn` |

Touchpoints: `nzgd/db/orm.py` (field definitions, the `Meta.indexes` tuple,
and the incorrect docstrings), `scripts/db/put_nzgd_metadata.py`,
`scripts/metadata/make_metadata_summary_csv.py` (SQL),
`nzgd/dedup/selection.py` and `nzgd/dedup/executor.py` column lists,
`config.yaml` `field_plausibility_ranges` keys, `tests/dedup/conftest.py` and
any tests naming these columns. The DB is rebuilt from scratch per version, so
there is no migration; DBs already on disk keep the old names, and the next
versioned build picks up the new schema.

### Build script: `nzgd/scripts/metadata/build_nzgd_index.py`

Shared modules, so each piece of logic exists in exactly one place:

- `nzgd/metadata/location.py` — port of the `find_region` point-in-polygon
  classification (geopandas `sjoin` against the LINZ shapefiles), the
  `replace_chars` sanitization (space, apostrophe, comma, slash →
  underscore), and `unclassified` fill for no-match or invalid coordinates
  (e.g. NZGD's lat/lon 0.0 placeholders).
- `nzgd/metadata/rasters.py` — the GeoTIFF sampling functions, moved out of
  `put_nzgd_metadata.py` (which currently duplicates ~230 lines of them).

Steps:

1. Read the catalog via `constants.NZGD_API_INVESTIGATION_CATALOG_PATH_CURRENT`
   (config plumbing that already exists, currently unused). Rename
   `Id` → `nzgd_id`. Assert IDs are unique.
2. Read the sidecar. Classify only IDs not yet in it (typically a handful per
   sync). Append, rewrite sorted by `nzgd_id`, and assert every pre-existing
   sidecar row is unchanged.
3. Compute `nztm_y`/`nztm_x` for rows with valid coordinates (vectorized
   qcore transform).
4. Sample the five model values for all rows under the corrected names;
   outside-raster samples, nodata sentinels (e.g. -32767), and invalid
   coordinates all yield NaN. (The legacy index baked nodata sentinels into
   its model columns for out-of-raster points; the rebuild fixes this.)
5. Assemble the 37 columns and write atomically (temp file + `os.replace`).

Guards — fail loudly with the violated invariant named, exit non-zero,
previous artifacts untouched:

- If a previous `nzgd_index.csv.gz` exists, the new ID set must be a superset
  of the old (no record ever disappears).
- No duplicate `nzgd_id`.
- Honest end-of-run report: totals; newly added IDs listed by name; new IDs
  classified `unclassified`; NaN model-value counts; and which existing rows'
  catalog columns changed since the previous index (with IDs when few). No
  tautological "success" banners.

Config additions: `district_shapefile_path` and `suburbs_shapefile_path`
pointing at the LINZ shapefiles. Expected runtime: a few minutes (sjoin runs
only for new IDs; raster sampling of ~198k points is vectorized).

### Consumer changes

1. `config.yaml`: `nzgd_index_file_name: "nzgd_index.csv.gz"`. Every
   config-driven consumer is `pd.read_csv(INDEX_FILE_PATH)` and pandas infers
   gzip from the extension, so the CPT/SPT extraction scripts, `ags_miner`,
   `put_cpts_in_db.py`, and `create_empty_db_and_fill_support_tables.py` need
   no code changes. Remove the stale `only_127920_...` comment line.
2. `put_nzgd_metadata.py`: load via `constants.INDEX_FILE_PATH`; delete its
   three sampling functions and take the five model columns from the index
   (same rasters, same coordinates — identical values); apply the renames.
3. `make_metadata_summary_csv.py`: read the index instead of the
   `with_nlm_gwl` intermediate. The rename map shrinks to the raw NZGD
   fields: `State`→`availability_status`,
   `InvestigationId`→`original_investigation_name`, `Latitude`→`latitude`,
   `Longitude`→`longitude`, `CreatedOn`→`record_created_on`,
   `LastModifiedOn`→`record_last_modified_on`.
4. `scripts/temp/` one-offs that read `INDEX_FILE_PATH` keep working through
   config and are not individually updated.

### Behavior deltas (all intended)

- **16** (Published SCP again; files re-downloaded during the June 2026
  Stage 3): CPT extraction gains a record.
- **230470** (SCP, downloaded): CPT extraction gains a record.
- **230469** (TMP): matches no extraction filter.
- **229775** (now RC) and **229822** (now SNC): both new TypeDisplays are in
  the borehole list, so SPT extraction will consider them; they have no
  download directories in the main tree (old files live in the
  overwritten-IDs archive), the per-ID glob returns empty, and they are
  skipped gracefully.

## Migration sequence

One-time, in order. Work branches from `feat/cpt-supplemental-consolidation`.

1. Create `feat/regenerated-nzgd-index` from
   `feat/cpt-supplemental-consolidation`.
2. Add the shared modules, build script, config keys, and tests — no behavior
   change yet.
3. Seed `nzgd_id_to_location.csv.gz` from the old index via
   `nzgd/scripts/metadata/one_time/seed_location_sidecar_from_legacy_index.py`
   (kept in git for provenance). The script reports any row whose
   coordinates fall outside the NZ bounding box yet carries a real
   classification, for case-by-case review — blanket invalidation could
   clobber legitimate manual classifications (e.g. 8753, a plausible
   latitude sign typo). One known correction is applied: 229630 (an
   upstream test record at 12.123/34.123) is seeded `unclassified` instead
   of the legacy index's erroneous Canterbury/Casebrook values.
4. Run the build, producing `nzgd_index.csv.gz`.
5. Run the verification script (below); review its report.
6. Flip the config line; apply the consumer edits and renames; run pytest.
7. Coordination note: the summary CSV can next be regenerated only from a DB
   built with the new schema. Dev DBs are versioned per build
   (`uc_nzgd_v0p8p0_20260612.db`), so the next normal DB rebuild picks this
   up; older DBs on disk keep the old column names.
8. Delete superseded artifacts (list below).
9. Documentation: append "Stage 4 — rebuild the NZGD index
   (`build_nzgd_index.py`)" to `api_nzgd/README_to_download_nzgd_updates.md`.
   The git-tracked index's commit history doubles as the record of when the
   mirror was last updated.

## Verification: `nzgd/scripts/metadata/one_time/verify_new_index_against_legacy.py`

Kept in git as the audit record. Must pass before step 8 (deletions). All
checks compare the legacy index against the freshly built one:

- Old ID set ⊆ new ID set; report additions. At design time the expected
  additions are exactly {230469, 230470}; if the catalog has been re-synced
  since, review the reported list rather than assuming.
- NaN-aware per-row comparison of the 26 raw catalog columns. Assert the set
  of differing IDs ⊆ {16, 229775, 229822} and print full old/new rows for
  each. A failure means new divergence appeared upstream: review it, archive
  the old row if it is a reassignment, then extend the known set.
- For each differing ID, assert its archive directory exists under
  `nzgd_source_files_of_overwritten_nzgd_ids/` and contains a saved metadata
  row.
- Location columns byte-identical for all old IDs except the documented
  seed-time correction (229630 → `unclassified`), proving seeding fidelity,
  including the manual batches.
- `nztm_*` values allclose to the old ones.
- Model columns reported, not asserted equal: sampling is fresh, names are
  corrected, and the NLM columns are new.
- Non-zero exit on any failure; honest final summary either way.

Independent of all checks, the legacy file remains recoverable from git
history (`20933a0`) forever.

## Removed artifacts (after verification passes)

- `nzgd/resources/nzgd_metadata_from_coordinates_22_august_2025.csv`
  (`git rm`; content preserved in history).
- `nzgd/scripts/metadata/update_nzgd_metadata_for_past_and_current_nzgd_investigations.py`
  (broken; superseded).
- `nzgd/scripts/temp/assemble_nzgd_metadata_from_coordinates.py` (superseded
  producer copy).
- On-disk gitignored intermediates and their `.gitignore` entries:
  `nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl.csv`
  and `.csv.gz`, `nzgd_id_to_region.csv`.
- Optional (repo is retired): the duplicate dated CSV in
  `nzgd_data_extraction/resources/`.

## Error handling

- Atomic writes for both the index and the sidecar.
- Guard failures name the exact invariant violated and exit non-zero; prior
  artifacts are left untouched.
- Missing inputs (catalog, shapefiles, GeoTIFFs) raise immediately, naming
  the config key to fix.
- New IDs with invalid coordinates are classified `unclassified` and listed
  in the report — never silent.
- No network access anywhere in the build or verification.

## Testing

- Unit tests (`tests/metadata/`) for the pure logic: the superset guard trips
  when an ID disappears; the append-only sidecar assertion trips on a mutated
  row; `Id`→`nzgd_id` rename; `replace_chars`; the NaN-aware comparison
  helper.
- Small synthetic fixtures — a tiny in-memory GeoTIFF and toy polygons — so
  `rasters.py` and `location.py` are tested without the 90 MB real inputs.
- Port validation for `location.py`: recompute the location of a random
  sample of already-classified IDs and report the agreement rate against the
  sidecar (high agreement expected; not asserted at 100% because shapefile
  vintages and boundary edge cases differ).
- The dedup suite keeps passing after the renames (`conftest.py` updated).
- The one-time verification script is the integration test against real data.

## Observations recorded, deliberately untouched

- The catalog and index both carry four NZGD-side test records (IDs 229615,
  229617, 229625, 229630) with placeholder values throughout: `Type` of
  `Investigation Type` / `Investigation Type 2`, `TypeDisplay` of
  `TypeDisplay`/`TypeDisplay1`, coordinates 12.123/34.123 (Sudan), and
  `EndDate` of `0001-01-01T00:00:00Z` (the .NET date default). They are
  still Published upstream (created Feb 2025, touched June 2025), so they
  are kept for mirror fidelity — a local deletion would be undone or
  flagged by the next sync. They match no extraction filter and are absent
  from the file catalog, so they are inert. A wider family of upstream test
  records with valid-looking `Type` codes also exists (e.g. 229555 "Rodd's
  Test Project", 229580, 229583, 229662–229664).
- 229775's NZGD Files endpoint returns HTTP 500 and 229822 is absent from the
  file catalog; both are upstream NZGD conditions, not local defects.
- DBs already on disk keep the old column names until their next versioned
  rebuild.
- The map webapps' schema expectations are already incompatible with the
  current DB and are out of scope here by decision.
