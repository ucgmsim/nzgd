# Initial NZGD Vs30 database population

Date: 2026-09-19. Vs30 working branch: `andrew-latest`.

## Scope

Target:
`/home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p8p2_20260709_deduped.db`

Both `cptvs30estimates` and `sptvs30estimates` were empty before this work.
Approximately 49,489 CPT reports have extracted data and 16,312 SPT reports
have measurements. The CPT table contains approximately 59.6 million samples.
There are at most 758,094 combinations for those reports before eligibility
checks: seven CPT-to-Vs or two SPT-to-Vs relations, each with two Boore relations.

Calculate separate central estimates in m/s for every supported combination.
The initial population left `vs30_stddev` NULL (filled in October 2026; see below): see [uncertainty findings](https://github.com/ucgmsim/Vs30/blob/dev_scripts_archive/dev/docs/vs_calc/vs30_uncertainty_followup.md).
No alteration to source measurements, report metadata, or lookup tables is
authorised by the population step. Model applicability to geology is not inferred
from a finite result; the alternatives remain labelled by correlation IDs.

## Input and eligibility policy

- Use the database's MPa columns directly, without reapplying file-import unit
  heuristics. Reject non-finite/non-positive required measurements, retaining a
  count of exclusions. Retain negative measured pore pressures if finite.
- Evaluate input requirements per CPT correlation: McGann can operate without
  pore pressure; Andrus Tertiary does not require sleeve friction. Reuse CPT
  normalization only across correlations requiring the same inputs.
- Reject conflicting duplicate measurements at a depth; collapse exact duplicates.
- Prefer finite extracted groundwater and net area ratio; otherwise preserve CPT
  defaults of 1 m and 0.8. SPT defaults are groundwater 2 m, energy ratio 75 percent,
  diameter 150 mm and automatic hammer. Do not divide percentage efficiency by 100.
  Do not introduce modelled groundwater or borrow ambiguous sibling-report data.
- Retain literal extracted groundwater zero, while flagging it: some source zeros
  are placeholders. Reject negative groundwater and physically invalid metadata.
- SPT N uses finite `ISPT_NVAL`, falling back to `ISPT_MAIN`. Zero N observations
  are retained in input accounting but are not supported by the Vs regressions.
- For unambiguous SPT soil logs, use the validated nearest-layer gap policy:
  nearest logged interval within explicit gaps, switching at gap midpoints;
  infer absent bottoms from the next top; extend end layers as required. Do not
  bridge conflicting overlapping soils as though they were unambiguous.
- Preserve the existing clay mapping for soil labels outside sand/silt/gravel/clay,
  and flag it. Where no usable soil log exists, a clearly flagged all-clay,
  no-layer fallback may be used; conflicting logs are logged for later review.
- Require at least two usable samples and a profile reaching 5 m for Boore 2011,
  or 10 m for Boore 2004. Use the existing integer-depth treatment. At 30 m and
  deeper, both method-labelled rows hold the same directly integrated central
  estimate, with NULL uncertainty.
- Reject non-finite or non-positive predictions. Record the precise failure or
  ineligibility reason; do not insert failed predictions as successful estimates.
- Reject corrected cone resistances that are non-positive within floating-point
  roundoff: `qt <= 8*eps*(abs(qc) + abs(u2*(1-area_ratio)))`. A QA example was
  `qc=0.04`, `u2=-0.2`, `area_ratio=0.8`, whose mathematical zero became about
  `6.94e-18` MPa and yielded near-zero Vs30. This is a numerical-domain check,
  not an arbitrary lower cutoff on the output velocity.
- Bound CPT normalization at 1,000 iterations per sample and record a failure
  if it does not converge, instead of letting a pathological record stall a run.

## Execution design

The two NZGD entry scripts delegate to a common batch runner. An explicit database
argument replaces the obsolete database configured for extraction. Workers read
the source database in SQLite read-only mode. Only the coordinator writes the run
audit database; source population is a separate, explicit publish action.

The audit contains per-report input assumptions, per-combination results or
exclusion reasons, timings, source identity, code hashes and lookup IDs. Completed
reports are checkpointed and skipped on resume. Changing source inputs or code
requires a new run directory. Resume never silently mixes numerical policies.

Parallel workers compute one report at a time, reusing safe shared calculations
and isolating mutable correlation/profile arrays. BLAS thread counts are limited
to one per worker. A single writer avoids concurrent SQLite insert contention.

Before publication, validate staged results and create a full SQLite backup.
Insert only missing natural-key combinations, in one transaction, with foreign
keys checked. Existing differing estimates must stop the import rather than be
overwritten. Preserve the audit and backup after completion.

The initial implementation is conservative about a bad Vs value anywhere in
the candidate profile, even below the eventual 30 m cutoff: that correlation
is skipped and logged. Follow-up can recover valid shallower portions under an
explicit policy. Likewise, mixed soil labels are not treated as missing logs
and silently replaced by clay; they are skipped for review.

## Commands

Use `/home/arr65/venvs/dev_nzgd_venv/bin/python` with:

```bash
source /home/arr65/venvs/dev_nzgd_venv/bin/activate
export PYTHONPATH=/home/arr65/src/Vs30/VsViewer:/home/arr65/src/nzgd
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Common CLI (replace `DATABASE` and `RUN_DIRECTORY` with explicit paths):

```bash
python -m nzgd.scripts.estimate_vs30.batch run DATABASE --run-dir RUN_DIRECTORY --kind both --workers 8 --limit 100
python -m nzgd.scripts.estimate_vs30.batch run DATABASE --run-dir RUN_DIRECTORY --kind both --workers 8
python -m nzgd.scripts.estimate_vs30.batch status --run-dir RUN_DIRECTORY
python -m nzgd.scripts.estimate_vs30.batch publish DATABASE --run-dir RUN_DIRECTORY --backup BACKUP_DATABASE
```

`--limit` limits pending reports per kind for a pilot, evenly spread over IDs.
The unrestricted command resumes the same checkpoint. The original
`estimate_vs30_from_cpt` / `estimate_vs30_from_spt` module names are retained as
entry points with the corresponding default kind. They do no work on import.

The publication command is intentionally separate from calculation. It requires
write access to the explicitly named target and backup locations. Do not run the
legacy extraction-default database path by accident.

After publication the source file identity has changed, so `run` intentionally
will not resume calculation into that old run directory. `status` and repeat
`publish` remain available; repeat publication inserts zero rows if estimates
are unchanged. Use a new run directory for subsequent model revisions. If a
publication is interrupted before its completion is recorded, preserve the
backup and inspect the target before retrying; never force past identity checks.

## Acceptance checks and handoff

- Tests of depth gates, numerical filtering, metadata defaults, input isolation,
  nearest-layer policy, checkpoint/resume, and non-overwriting publication.
- Pilot comparisons with direct VsViewer calculations and order-invariance checks.
- Counts by method; finite positive means; NULL uncertainty everywhere; no duplicate
  natural keys; target foreign keys and backup integrity verified.
- Summarise skipped reports/combinations, assumption flags and extremes. These
  are provisional correlation-based estimates, not source-data certification.
- Record actual runtime and production output counts in a separate results report.

Initial timing of three CPTs was too limited to forecast the complete job. The
full-run estimate must come from the pilot, not the initial several-day assumption.

## Run directories

- `.../dev_extracted_cpt_and_scpt_data/vs30_estimates_20260919`: preliminary
  checkpoint preserved for audit, marked invalidated and blocked from publication
  after the floating-point cancellation finding. No estimates from it were imported.
- `.../dev_extracted_cpt_and_scpt_data/vs30_estimates_20260919_v2`: replacement
  run with the roundoff guard and complete fresh provenance (policy version 2).

See `vs30_population_results_20260919.md` for final counts and verification.

## Querying the audit for later work

Open the run's `estimates.sqlite` separately from the populated NZGD database.
`records` has one row per processed report, including a JSON `assumptions` field.
`results` has one row per attempted method combination, including skipped ones.
Successful results join the destination tables on report and correlation IDs,
using the names in the source lookup tables. `metadata.manifest` records source
identity, code/resource hashes, environment versions and the exact ID mappings.

```sql
-- Why were estimates skipped? Count combinations, not investigations.
SELECT kind, reason, count(*) AS combinations
FROM results WHERE status != 'ok'
GROUP BY kind, reason ORDER BY combinations DESC;

-- Inspect SPT assumptions, including gap filling and metadata defaults.
SELECT report_id, json_extract(assumptions, '$.input_error') AS input_error,
       json_extract(assumptions, '$.gap_depth_filled_m') AS filled_gap_m,
       json_extract(assumptions, '$.soil_types') AS logged_soils,
       json_extract(assumptions, '$.soil_fallback') AS fallback
FROM records WHERE kind = 'spt';

-- Illustrative review queue; 50/2000 m/s are triage thresholds, not validity limits.
SELECT * FROM results
WHERE status = 'ok' AND (vs30 < 50 OR vs30 > 2000)
ORDER BY vs30;

-- CPTs whose first usable sample is well below the surface.
SELECT report_id, json_extract(assumptions, '$.groups') AS input_groups
FROM records WHERE kind = 'cpt'
AND json_extract(assumptions, '$.groups.qc_fs.minimum_depth_m') > 2;
```

Keep extrapolation depth, unlogged near-surface thickness and soil-log ambiguity
separate from statistical uncertainty. A finite positive estimate is not evidence
that the investigation or geological correlation choice is reliable.

## Filling `vs30_stddev` (October 2026)

The initial population left `vs30_stddev` NULL. The batch now records vs_calc's
provisional uncertainty with every estimate and publishes it with new rows.
`vs30_stddev` is the standard deviation of ln(Vs30):
`sqrt(0.2**2 + (ln(10) * Boore residual)**2)`. The Boore residual is the published
one for the profile's integer depth, and the term is 0 at 30 m, so the value is
0.20 for profiles reaching 30 m and up to 0.34 at 5 m. It doesn't cover the choice
of correlation or the input assumptions; see the
[uncertainty note](https://github.com/ucgmsim/Vs30/blob/dev_scripts_archive/dev/docs/vs_calc/vs30_uncertainty_followup.md).

To fill rows published without it, calculate a new run against the current
database, then run `fill-stddev`. That command:

- backs up the database
- matches every estimate to the run by report, correlation and Boore method
- requires identical Vs30
- sets `vs30_stddev` in one transaction

It stops without changes on any mismatch, missing match or existing value.

```bash
python -m nzgd.scripts.estimate_vs30.batch run DATABASE --run-dir RUN_DIRECTORY --kind both --workers 6
python -m nzgd.scripts.estimate_vs30.batch fill-stddev DATABASE --run-dir RUN_DIRECTORY --backup BACKUP_DATABASE
```
