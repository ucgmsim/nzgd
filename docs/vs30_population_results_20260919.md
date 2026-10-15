# Initial Vs30 population results — 2026-09-19

## Completed outcome

Populated:
`/home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p8p2_20260709_deduped.db`

| Destination | Inserted estimates | Reports with at least one estimate |
| --- | ---: | ---: |
| `cptvs30estimates` | 514,444 | 43,231 |
| `sptvs30estimates` | 24,350 | 6,657 |
| Total | 538,794 | 49,888 |

All `vs30` values are finite positive central estimates in m/s. All
`vs30_stddev` fields are **SQL NULL**, not zero. Uncertainty work remains deferred
as requested; see [the uncertainty findings and proposal](https://github.com/ucgmsim/Vs30/blob/dev_scripts_archive/dev/docs/vs_calc/vs30_uncertainty_followup.md).
Do not use missing uncertainty as evidence of precision.

The replacement production calculation took **279.36 seconds (4 min 39 s)**
using eight worker processes with one BLAS thread per process. This excludes
development, tests, the preliminary QA run, backup and publication. The full
calculation began at 10:00:46 UTC and the import committed at 10:07:07 UTC.
The original several-day computation estimate was not borne out by this run.

## Scope and coverage

All 49,489 CPT reports and 16,312 SPT reports with measurements were processed.
There were 758,094 attempted method combinations, of which 538,794 succeeded and
219,300 were excluded by input, numerical or depth checks. Every processed report
has an audit entry, including reports with no successful estimate.

| Coverage per report | CPT | SPT |
| --- | ---: | ---: |
| Every combination (14 CPT / 4 SPT) | 30,725 | 5,518 |
| Some combinations | 12,506 | 1,139 |
| No successful combination | 6,258 | 9,655 |

The source has 54,691 CPT reports and 22,036 SPT reports in total. The remaining
5,202 CPT and 5,724 SPT reports contain no measurements and were not queued.

These are labelled correlation alternatives, not a geological applicability
certification. In particular, applying an age-specific or loess correlation does
not establish that a site has that geology. See [the runbook](vs30_database_population.md)
for the precise input/default/depth policies.

## Counts by correlation

For profiles reaching 30 m, the two Boore-labelled rows have the same directly
integrated central value; they are not two independent observations.

| Input correlation | Boore 2004 | Boore 2011 |
| --- | ---: | ---: |
| CPT Andrus 2007 Holocene | 30,824 | 40,980 |
| CPT Andrus 2007 Pleistocene | 30,824 | 40,980 |
| CPT Andrus 2007 Tertiary Cooper Marl | 33,353 | 42,825 |
| CPT Hegazy 2006 | 30,824 | 40,980 |
| CPT McGann 2015 | 32,451 | 43,074 |
| CPT McGann 2018 | 32,451 | 43,074 |
| CPT Robertson 2009 | 30,824 | 40,980 |
| SPT Brandenberg 2010 | 5,518 | 6,657 |
| SPT Kwak 2015 | 5,518 | 6,657 |

## Preserved artifacts and reproducibility

Production run directory:
`/home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/vs30_estimates_20260919_v2`

- `estimates.sqlite`: complete report assumptions, all method outcomes and
  exclusion reasons, timings, source identity, scientific-code/resource SHA-256
  hashes, environment versions, lookup IDs and publication record.
- `source_before_vs30.db`: full pre-population SQLite backup, with both estimate
  tables empty; 6,377,701,376 bytes. Do not overwrite or delete it casually.
- `run.lock`: coordinator lock file; its presence alone does not mean a process
  is still running. The operating-system lock is released when the process ends.

The earlier sibling directory `vs30_estimates_20260919` is preserved for audit,
marked invalidated, and blocked from publication. It contains the 200-report
pilot and the interrupted first run. No rows from that run were imported.

The Vs30 implementation used the `andrew-latest` worktree at
`/home/arr65/src/Vs30/.worktrees/andrew-latest`; NZGD used
`/home/arr65/src/nzgd`. Working-tree code hashes are recorded, so reproduction
does not depend on assuming those checkouts were clean commits.

The shared runner is `nzgd/scripts/estimate_vs30/batch.py`. The existing
`estimate_vs30_from_cpt.py` and `estimate_vs30_from_spt.py` entry points now
delegate to it and are safe to import. Neither uses the old extraction-default
database implicitly.

## Verification performed

- 84 VsViewer tests passed, including the CPT performance regression test.
- 19 batch tests passed: depth boundaries, input-array isolation, missing u2,
  nearest-layer handling, soil ambiguity, SPT defaults, numerical cancellation,
  resume, changed-source rejection, backup preservation, NULL uncertainty,
  metadata flags, idempotence and transaction rollback across both tables.
- Scientific calculations matched direct legacy/VsViewer calls exactly for
  42 CPT estimates and 58 SPT estimates. CPT checks used reversed correlation
  order and included 15 exact normalized-parameter array comparisons.
- 158 SPT effective-stress values matched independent layer integration.
- The final audit has exactly 14 outcomes per CPT and four per SPT. All report
  IDs and NZGD IDs matched the source before import.
- Publication created and checked a full backup, then inserted both estimate
  tables in one transaction with foreign-key enforcement.
- Post-import SQLite `quick_check` returned `ok`; estimate-table foreign-key
  checks passed; neither table has duplicate natural keys.
- Every inserted value exactly matches its successful audit result; no populated
  uncertainty values exist. Both backup estimate tables remain empty.
- Row counts in all 24 non-estimate tables match the backup. The database has
  no triggers, and the import only inserts into the two estimate tables.
- Ruff checks and whitespace checks passed on the changed execution code.

The target database retained its original file size because SQLite could reuse
existing free pages; unchanged file size does not mean the insert was absent.

## Quality-control findings and follow-up priorities

### Numerical cancellation prevented before import

Several decimal CPT inputs mathematically cancelled to zero corrected tip
resistance but became tiny positive floating-point values. They generated
near-zero Vs30 values in preliminary QA. A relative roundoff guard now rejects
these inputs; tests cover three cancellation examples, and four real problematic
reports were checked explicitly. The production run was restarted with fresh
policy-version-2 provenance. No arbitrary output-velocity clipping was added.

### SPT soil-log ambiguity is the largest coverage limitation

8,154 SPT reports were skipped because a layer top had multiple soil labels.
Other input exclusions included 1,313 reports with insufficient positive-N/depth
coverage, 86 with conflicting N values at a depth, and 98 with missing, invalid
or conflicting layer geometry. Further depth checks after SPT correction also
exclude combinations. All individual reasons are in the audit.

Do not address mixed soil labels by picking the first row. A future policy needs
to distinguish mixtures, duplicate descriptions and genuinely conflicting logs.
The approved nearest-layer rule fills spatial gaps; it does not resolve these
classification ambiguities.

Among the 6,657 SPT reports with estimates:

- 158 used the clearly flagged no-log/default-clay fallback.
- 6,649 assumed groundwater, 5,299 assumed efficiency, and 6,583 assumed diameter.
- 2,722 had explicit soil-log gaps filled by the nearest-layer rule.
- 6,142 used profiles containing supplied unit weights with saturated weight
  below unsaturated weight. The supplied lookup was preserved, not recalibrated.

These are report counts, not independent uncertainties or additional rows.

### Low estimates and unlogged near-surface intervals

CPT estimates range from 18.60 to 1,706.09 m/s; SPT estimates range from 73.30 to
546.08 m/s. Using **50 m/s only as a review threshold**, there are 31 low CPT
estimates across ten reports, listed below. None exceeds the illustrative upper
review threshold of 2,000 m/s. These thresholds are not scientific validity
limits; the remaining finite predictions were retained for later review as
requested, without clipping their values.

| CPT ID | NZGD ID | Lowest estimate (m/s) | Estimates below 50 m/s |
| --- | ---: | ---: | ---: |
| 153471 | 173954 | 18.60 | 6 |
| 73951 | 36260 | 25.90 | 4 |
| 153473 | 173957 | 33.32 | 3 |
| 139853 | 141677 | 34.49 | 6 |
| 139857 | 141679 | 36.70 | 4 |
| 99777 | 57846 | 37.99 | 2 |
| 149219 | 161825 | 45.66 | 1 |
| 139855 | 141678 | 46.34 | 2 |
| 139859 | 141680 | 46.37 | 2 |
| 114133 | 88794 | 48.39 | 1 |

523 CPT reports with estimates have the first usable qc/fs sample deeper than
2 m. The existing profile integration extends that first speed to the surface.
For example, CPT 73951 starts at 13 m with a very small first cone resistance;
that extrapolation strongly affects its low estimates. Review source files and
predrill/near-surface assumptions before using these cases in downstream models.

### Remaining scientific work

The Boore sigma error, full uncertainty propagation, first-sample CPT groundwater
pressure, soil/unit-weight assumptions, and geological applicability remain
documented in [the follow-up proposal](https://github.com/ucgmsim/Vs30/blob/dev_scripts_archive/dev/docs/vs_calc/vs30_uncertainty_followup.md). The initial
population does not resolve them or certify these values for engineering design.
