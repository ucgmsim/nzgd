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

> **Resolved 2026-07-07** — both over-collapse mechanisms were fixed and a full
> re-sweep confirms **zero** real coverage loss. See the update at the end.

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

## Update (2026-07-07): fixes shipped — full re-sweep shows zero real loss

Both over-collapse mechanisms flagged in the caveat were fixed and validated on
real data:

- **Cross-record fuzzy pass** — completeness-aware canonical selection plus a
  containment / reparent-if-not-contained guard (`pass2_fuzzy.py`,
  `selection.py`).
- **Within-record Pass 0** — `default_within_record_canonical` now ranks by
  depth span (widest coverage wins), and a report is deleted only when its depth
  extent is *contained* in the canonical's — otherwise it is kept as a separate
  report (`canonical_selectors.py`, `pass0_within_record.py`). The sole Pass 0
  deletion site iterates the absorbed list, so within-record consolidation is
  now provably no-loss.

Re-running the full pipeline on `uc_nzgd_v0p8p1_20260625.db` and re-sweeping
**every** nzgd_id — coverage per record = union of `[min_depth_m, max_depth_m]`
intervals over its `has_cpt_data` reports, source vs deduped:

| Metric | Value |
|---|---|
| Source nzgd_ids with CPT data | 49,773 |
| Records with ≤0.5 m coverage change | 47,270 |
| Records flagged (>0.5 m own-nzgd_id reduction) | 2,503 |
| — intentional quality-filter rejections | 1,658 |
| — cross-record moves (data survives under the canonical nzgd_id) | 845 |
| — **real, unexplained loss** | **0** |
| **Max unexplained residual, any single record** | **0.0000 m** |

Every metre of source depth coverage is still present in the deduped DB — either
under the same nzgd_id, or (for cross-record duplicates) under the canonical
nzgd_id it merged into. The four worst within-record cases are fully restored
(8809 → 22.39 m, 118991 → 14.99 m, 186131 / 186134 → 15.50 m; each
`deduped span == source span`). The worst cross-record case, nzgd 193652's
6.86 m, is contained in its canonical nzgd 193678, whose deep 26.1 m trace is
preserved intact (source span 26.1 m == deduped span 26.1 m). The caveat above is
therefore resolved: **no dedup pass — hash, within-record, or cross-record
fuzzy — reduces real depth coverage.**

The 2,503 flagged records break down entirely into two expected, non-loss
categories. Quality-filter rejections (`quality_reject` /
`quality_reject_record`, e.g. nzgd 7691's 0–20 m constant-column report) are
deliberate removals of degenerate traces. Cross-record moves (`dedup_audit`
`canonical_nzgd_id` / `merged_nzgd_id`) relocate a duplicate to its canonical;
the depth still exists, just attributed to the canonical record.

Method note: stored `min_depth_m` / `max_depth_m` were re-validated against
`cptmeasurements` over a 300-report random sample (worst |Δ| = 0.0000 m), so the
sweep reads the stored extents rather than scanning all 99 M measurement rows.
