# Verifying the `cptreport` count reduction after dedup

*Investigation date: 2026-06-29*

A researcher worried that data was lost because the deduped DB has fewer
`cptreport` rows. This note records the check that the reduction is expected and
fully explained by the deduplication/consolidation process.

Comparison: `uc_nzgd_v0p8p1_20260625.db` → `uc_nzgd_v0p8p1_20260625_deduped.db`
(the single recorded `dedup_run`; `source_db_path` matches, run finished cleanly,
`script_version` = current `HEAD`, no `dedup_failures.csv` / orphan artifacts).

## Result: the count drop is 100% accounted for

| Check | Value |
|---|---|
| `cptreport` rows, source → deduped | 177,791 → 56,127 (**−121,664**) |
| Distinct `cpt_id` that vanished | 121,664 |
| New `cpt_id` that appeared | 0 |
| `cpt_id`s logged as deleted in `dedup_audit` | 121,664 |
| **Vanished set == audit-deleted set** | exact match |
| Unexplained loss / phantom deletions | 0 / 0 |
| Cross-check: `dedup_report.csv` CPT deletions | 121,664 (matches) |

This is an **identity-level** match: the exact rows that disappeared are exactly
the rows the dedup log says it deleted. Nothing vanished outside the process.

What the 121,664 deleted rows were:

| Pass | Deleted | Nature |
|---|---|---|
| `within_record` | 120,692 | duplicate/empty sheets of one investigation |
| cross-record `hash` | 876 | bit-identical traces |
| cross-record `fuzzy` | 96 | near-identical traces |
| `supplemental_consolidation` | 0 | fills NULL cells only |

Of these, **91,955 (75.6%) were empty rows** (no trace data) and 13,147 were
bit-identical `hash` duplicates — both lossless.

## Caveat: ~46 records lost real depth coverage (fuzzy pass)

Checking every data-bearing deletion against the canonical that replaced it, **46
records (~0.1%) lost >0.5 m of depth coverage** (13 deep, 35 shallow). All come
from the **fuzzy** passes; `hash` passes lose nothing. Cause: fuzzy matching
collapses traces of very unequal length, and the canonical selector keeps the
lowest `cpt_id`, which is sometimes the shorter fragment. Worst confirmed cases:

| nzgd_id | deleted trace | survives | lost |
|---|---|---|---|
| 8809 | 0–22.39 m | 0–0.14 m (15 pts) | ~22 m |
| 193652 | 0–26.1 m | 0–6.86 m | ~19 m |
| 186134 / 186131 | 0–15.5 m | 0.3–3.7 m | ~12 m each |
| 118991 | 0–15 m | 0–5.5 m + 8.0–8.8 m | gap + deep |

**Bottom line:** the row-count reduction is fully attributable to dedup with no
silent loss; ~99.9% of removals are lossless (empty rows or verified duplicates).
The ~46 fuzzy over-collapses are genuine content loss worth reviewing — a possible
fix is a trace-length/coverage guard in the fuzzy pass plus a canonical selector
that prefers the most-complete row (max depth-span / measurement count) over the
lowest id.

## Reproduce

Counts and identity comparison come from `cptreport` in each DB and the
`dedup_audit` / `dedup_run` tables in the deduped DB (`report_pairs_json` lists
the deleted reports per pass); the same totals appear in `dedup_report.csv`. Depth
coverage uses `cptreport.min_depth_m` / `max_depth_m` (cross-checked against
`cptmeasurements`).
