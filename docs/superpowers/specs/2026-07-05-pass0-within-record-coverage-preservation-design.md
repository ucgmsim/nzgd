# Pass 0 within-record coverage preservation — design

**Date:** 2026-07-05
**Status:** Approved approach (Approach B), pending spec review
**Scope:** `nzgd/dedup/` Pass 0 within-record consolidation and its canonical selector.

## 1. Goal

Stop **Pass 0 within-record consolidation** from dropping real depth coverage. The
2026-07-04 fuzzy-pass fix (`docs/superpowers/specs/2026-07-04-fuzzy-pass-coverage-preservation-design.md`)
made the *cross-record* fuzzy pass no-loss, but a full re-run of `uc_nzgd_v0p8p1`
showed 4 of the 5 worst flagged records (8809, 186134, 186131, 118991) still lose
10–22 m — they are collapsed by **Pass 0**, not the cross-record fuzzy pass. After
this change, **no within-record consolidation reduces depth coverage**: a report is
deleted only when its depth range is contained in the surviving (canonical) report;
otherwise it is kept as a separate report.

## 2. Background — root cause

Pass 0 clusters a record's report rows by source-file stem + trace identity
(`_cluster_within_stem`, cross-stem edges via `best_trace_score`), then collapses
each cluster to one canonical row, deleting the rest. Two facts combine into loss:

1. **The canonical selector ignores trace size.**
   `default_within_record_canonical` (`nzgd/dedup/canonical_selectors.py`) picks
   `min(report_id)` among data-bearing rows. When a cluster contains a short trace
   and a long trace that fuzzy-match on their overlap, and the *short* one has the
   smaller `cpt_id`, the short fragment becomes canonical.
2. **Every non-canonical report in the cluster is deleted.**
   `apply_within_record_consolidation_plan` calls `executor.delete_report` on each
   `absorbed_report`. The long trace is deleted; only the short fragment survives.

`best_trace_score` compares only the overlapping depth range (documented root cause
of the fuzzy over-collapse), so a 0–22 m trace and a 0–0.14 m fragment match on
their 0–0.14 m overlap and land in the same cluster. This is the same defect the
fuzzy fix addressed, one pass earlier and with a *different* selector — the fuzzy
fix explicitly scoped Pass 0 out ("Non-goal: Pass 0 within-record uses a different
selector … left as a possible follow-up"). This spec is that follow-up.

## 3. Decisions (approved — "Approach B")

- **Two levers, mirroring the fuzzy fix:** completeness-aware canonical selection
  **and** a keep-if-not-contained safeguard.
- **Completeness metric = depth *span*** (`max_depth − min_depth` extent), not
  measurement-row count — targets coverage directly, consistent with the fuzzy fix.
- **Partial overlap → keep both reports** (never delete a non-contained trace).
  Within one record all reports already share the same `nzgd_id`, so "keep" is
  simply *not deleting* — no reparenting is needed (unlike the cross-record fuzzy
  pass, which had to reparent onto a different `nzgd_id`).
- **No new config, no new selector.** The default selector is fixed *in place*; the
  old lowest-id behavior is buggy and not worth preserving as an option. Config still
  resolves `within_record.canonical_selector` to `default_within_record_canonical`,
  so the re-run picks up the fix with no driver change.
- **Scope:** `default_within_record_canonical` + `generate_within_record_consolidation_plan`.
  The hash-identical and no-data absorptions are already no-loss and are unchanged.

## 4. Design

### 4.1 Completeness-aware canonical selection — `nzgd/dedup/canonical_selectors.py`

`ClusterRow` gains one field:

```python
@dataclass(frozen=True)
class ClusterRow:
    report_id: int
    has_data: bool
    measurement_row_count: int
    metadata_non_null_count: int
    depth_span: float               # max_depth − min_depth of the report's trace; 0.0 if no finite depth
```

The selector ranks by span, then id:

```python
def default_within_record_canonical(cluster_rows, table_cfg) -> int:
    """v2 default: prefer has_data rows; among them widest depth span; tiebreaker smallest report_id."""
    candidates = [r for r in cluster_rows if r.has_data] or list(cluster_rows)
    return min(candidates, key=lambda r: (-r.depth_span, r.report_id)).report_id
```

- `has_data` stays the primary discriminator (a data-bearing row still beats a
  no-data row even if the latter's degenerate span ties at 0.0).
- Among data-bearing rows, the widest depth coverage wins; identical/equal-length
  traces tie on span and fall back to smallest `report_id` — **identical to today's
  behavior for every equal-length cluster**.
- `depth_span` has **no default** on the dataclass, so any un-updated construction
  fails loudly rather than silently reverting to lowest-id ranking.

### 4.2 Keep-if-not-contained safeguard — `nzgd/dedup/pass0_within_record.py`

`generate_within_record_consolidation_plan` already receives `traces`
(`{report_id: ndarray}`, depth in column 0) from `_build_clusters_for_nzgd`. For
each cluster, compute depth extents once with the existing helper and derive spans:

```python
extent_by_id = {
    rid: (trace_depth_extent(traces[rid]) if rid in traces else (math.nan, math.nan))
    for rid in cluster_report_ids
}
span_by_id = {
    rid: (hi - lo if math.isfinite(lo) and math.isfinite(hi) else 0.0)
    for rid, (lo, hi) in extent_by_id.items()
}
```

`span_by_id[rid]` feeds `ClusterRow(..., depth_span=span_by_id[rid])`.

After the selector picks `canonical_id`, gate what enters `absorbed`:

```python
canonical_lo, canonical_hi = extent_by_id[canonical_id]
absorbed = []
for rid in cluster_report_ids:
    if rid == canonical_id:
        continue
    lo_a, hi_a = extent_by_id[rid]
    contained = not (math.isfinite(lo_a) and math.isfinite(hi_a)) or (
        math.isfinite(canonical_lo) and canonical_lo <= lo_a and hi_a <= canonical_hi
    )
    if not contained:
        continue  # genuine partial overlap: keep as a separate report (no loss)
    absorbed.append(_AbsorbedReport(...))   # unchanged construction
if not absorbed:
    continue                                # nothing left to absorb → no consolidation
```

- A report with **no finite depth** (no measurement rows, or all-NaN depths) is
  contained by definition — deleting it loses no coverage; its metadata is still
  preserved by the existing enrichment step for reports that *are* absorbed.
- A **contained** report (fragment ⊆ canonical, or hash-identical → equal extent →
  contained) is absorbed and deleted exactly as today.
- A **not-contained** report (partial overlap; canonical's extent does not cover it)
  is dropped from `absorbed` → never deleted → survives untouched under the same
  `nzgd_id`.

### 4.3 Airtight no-loss argument

The only deletion in Pass 0 is `executor.delete_report` iterating
`consolidation.absorbed_reports` in `apply_within_record_consolidation_plan`. §4.2
lets a report enter `absorbed_reports` only when its depth extent is contained in the
canonical (the survivor), or it has no finite depth. Therefore **every Pass-0-deleted
report is provably contained in a survivor**, so no within-record consolidation can
reduce depth coverage. `apply_within_record_consolidation_plan` requires no change.

### 4.4 What is deliberately *not* changed

- **Clustering / edge formation** (`_cluster_within_stem`, cross-stem edges) is
  untouched. A partial-overlap pair may still be *clustered*, but §4.2 keeps it — so
  no containment guard at edge-formation time is needed for no-loss. Leaving
  clustering alone keeps the change minimal and cannot create new merges.
- **Metadata enrichment, audit rows, savepoints, SPT cascade** — unchanged. Kept
  (not-contained) reports simply never appear in a consolidation's `absorbed_reports`,
  so they are absent from the audit, consistent with "nothing happened to them."
- **`trace_compare.py`** — no change; `trace_depth_extent` already exists (added by
  the fuzzy fix) and is imported into Pass 0.

## 5. Files changed

| File | Change |
|------|--------|
| `nzgd/dedup/canonical_selectors.py` | `ClusterRow` += `depth_span: float`; selector key → `(-depth_span, report_id)`; docstring |
| `nzgd/dedup/pass0_within_record.py` | import `trace_depth_extent`; build `extent_by_id`/`span_by_id`; pass `depth_span` into `ClusterRow`; containment-filter `absorbed`; `continue` when empty |
| `tests/dedup/test_dedup_pipeline.py` | +2 integration tests (§6) |

No change to `trace_compare.py`, `config.yaml`, or `deduplicate.py`.

## 6. Testing

Two integration tests against the real SQLite schema (`fresh_db`), same helpers and
style as the existing Pass 0 tests (`add_cpt_record`/`add_cpt_report`, `_run_pass0`):

1. **Containment → widest kept (the 8809-style bug).** One record, one stem: a short
   fragment with the **smaller** `cpt_id` (depths 0.1–0.4) and a long trace with the
   larger `cpt_id` (depths 0.1–2.0) that are identical on the 0.1–0.4 overlap, so they
   fuzzy-match and cluster. Assert the **long** report survives (`remaining == [long_id]`)
   and its trace still reaches ~2.0 m. **Fails on today's code** (short/lowest-id wins,
   long trace deleted).
2. **Partial overlap → keep both (the 118991-style case).** One record, one stem: two
   traces 0.1–1.0 and 0.6–1.6, identical on their 0.6–1.0 overlap, so they cluster but
   neither contains the other. Assert **both** reports survive and `n_clusters == 0`.
   **Fails on today's code** (second report deleted, 1.0–1.6 m lost).

Regression: the 12 existing Pass 0 tests were hand-checked to be behavior-unchanged —
they use identical or equal-length traces, or no-data absorptions, none of which the
fix alters. They must still pass unmodified.

Post-build real-data validation: re-run the full dedup on a backup of
`uc_nzgd_v0p8p1` and confirm the flagged records (8809, 186134, 186131, 118991, and
the fuzzy-fixed 193652) retain their source depth coverage in the deduped DB — the
same source-vs-deduped span comparison used to validate the fuzzy fix.

## 7. Risks & rollback

- **Fewer within-record absorptions.** Genuine partial-overlap reports are now kept,
  so the deduped DB has slightly more report rows. Intended (they cover different
  depth). Magnitude quantified from the re-run's Pass 0 counts.
- **Shared `ClusterRow`.** Only one construction site exists (`pass0_within_record.py`)
  and one selector consumes it; `select_canonical` (hash/fuzzy) is a separate function
  and is unaffected. Adding a non-defaulted field is caught at construction.
- **Gappy traces.** Containment is extent-based (`max − min`), so a canonical whose
  trace has an internal depth gap but still *spans* an absorbed report's range counts
  as containing it. Acceptable first-order (the flagged cases are simple fragments),
  refinable to covered-length later — same trade-off accepted for the fuzzy fix.
- **SPT.** `load_traces` puts depth in column 0 for SPT too (blow-count strings coerce
  to NaN elsewhere), so span/extent are well-defined; the all-NaN-depth SPT case yields
  `(nan, nan)` → span 0.0 → contained, matching today's no-crash behavior.
- **Reversible.** The source DB is never modified; re-running dedup regenerates the
  deduped DB.

## 8. Impact summary

Makes Pass 0 within-record consolidation strictly no-loss, closing the dominant
remaining source of dedup depth-coverage loss (8809, 186134, 186131, 118991).
Combined with the 2026-07-04 fuzzy-pass fix, **no dedup pass — hash, within-record,
or cross-record fuzzy — reduces depth coverage.** Slightly more report rows survive
(partial-overlap pairs kept separate). Clustering, enrichment, audit, and config are
unchanged.
