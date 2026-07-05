# Pass 0 within-record coverage preservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Pass 0 within-record consolidation strictly no-loss so it never deletes a report whose depth coverage isn't contained in the surviving canonical.

**Architecture:** Two combining levers, mirroring the 2026-07-04 cross-record fuzzy fix one pass earlier. (1) The within-record canonical selector ranks by depth *span* (widest coverage wins) instead of lowest `report_id`. (2) A report is absorbed/deleted only if its depth extent is contained in the canonical's; a genuine partial-overlap report is kept as a separate report under the same `nzgd_id`. The only deletion site in Pass 0 iterates `absorbed_reports`, so gating what enters that list makes the pass provably no-loss.

**Tech Stack:** Python 3, SQLite (`sqlite3`), numpy, pytest. Reuses the existing `nzgd.dedup.trace_compare.trace_depth_extent` helper.

**Spec:** `docs/superpowers/specs/2026-07-05-pass0-within-record-coverage-preservation-design.md`

## Global Constraints

- Python interpreter / test runner: `/home/arr65/venvs/dev_nzgd_venv/bin/python` (the project `.venv` lacks deps). Run pytest as `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest ...`.
- Import style: prefer module imports with dot notation; the exceptions in use here are `import math`, `import numpy as np`. `pass0_within_record.py` already uses named imports from `nzgd.dedup.trace_compare` — extend that existing line, don't add a new import style.
- No unnecessary aliases; prefer inline code over one-off helpers (project CLAUDE.md).
- Do **not** sweep unrelated pre-existing lint. Only touch the lines this plan specifies. Run `ruff` only on the files you change and confirm no *new* findings.
- Baseline: `tests/dedup/test_dedup_pipeline.py` is green (39 passed) before you start. All 39 must remain green after every task (the fix is behavior-preserving for equal-length / identical / no-data clusters).
- Metric is depth **span/extent** (`max_depth - min_depth`), never measurement-row count.
- Do not modify `nzgd/dedup/trace_compare.py`, `nzgd/resources/config.yaml`, or `nzgd/scripts/db/deduplicate.py` — the default selector is fixed in place and config already resolves to it.

---

### Task 1: Completeness-aware within-record canonical selection

Rank the within-record canonical by depth span (widest coverage) instead of lowest `report_id`. This alone restores the *containment* cases (a short fragment that fuzzy-matches a long trace no longer wins canonical). Adds `depth_span` to `ClusterRow` and plumbs it from the already-loaded traces in the plan generator.

**Files:**
- Modify: `nzgd/dedup/canonical_selectors.py` (ClusterRow dataclass + `default_within_record_canonical`)
- Modify: `nzgd/dedup/pass0_within_record.py` (import; cluster loop in `generate_within_record_consolidation_plan`)
- Test: `tests/dedup/test_dedup_pipeline.py` (add one integration test in the Pass 0 section)

**Interfaces:**
- Consumes: `nzgd.dedup.trace_compare.trace_depth_extent(arr: np.ndarray) -> tuple[float, float]` (already exists — returns `(min, max)` of finite depths in column 0, or `(nan, nan)`); `load_traces(...) -> dict[int, np.ndarray]` (already imported).
- Produces: `ClusterRow` gains field `depth_span: float`. Selector signature unchanged: `default_within_record_canonical(cluster_rows: Sequence[ClusterRow], table_cfg: TableConfig) -> int`. Task 2 relies on the local dict `extent_by_id: dict[int, tuple[float, float]]` being built inside the cluster loop.

- [ ] **Step 1: Write the failing test**

Add this function to the Pass 0 section of `tests/dedup/test_dedup_pipeline.py` (near the other `test_pass0_*` functions, e.g. after `test_pass0_fuzzy_within_record_match`). It uses only existing module-level helpers (`add_cpt_record`, `add_cpt_report`, `_run_pass0`, `CPT_TABLE_CONFIG`).

```python
def test_pass0_canonical_prefers_widest_depth_span(fresh_db: sqlite3.Connection) -> None:
    """A short fragment (smaller cpt_id) and a long trace identical on their overlap
    cluster together; the long trace must become canonical and survive (old code kept
    the lowest-id fragment and deleted the long trace)."""
    def qc(d: float) -> float:
        return 1.0 + 0.5 * d
    short_trace = [(round(0.1 * i, 1), qc(0.1 * i), 0.01, 0.0) for i in range(1, 5)]   # 0.1..0.4
    long_trace = [(round(0.1 * i, 1), qc(0.1 * i), 0.01, 0.0) for i in range(1, 21)]   # 0.1..2.0
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=short_trace, source_file="A.xlsx_sheet_Frag")
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=1, trace=long_trace,  source_file="A.xlsx_sheet_Full")

    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (1, 1)
    remaining = [r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport ORDER BY cpt_id")]
    assert remaining == [20]  # the long trace survives, not the lowest-id fragment
    max_depth = fresh_db.execute("SELECT MAX(depth_m) FROM cptmeasurements WHERE cpt_id = 20").fetchone()[0]
    assert max_depth == 2.0  # full depth coverage preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest "tests/dedup/test_dedup_pipeline.py::test_pass0_canonical_prefers_widest_depth_span" -v`
Expected: FAIL — `assert remaining == [20]` fails because today's selector keeps the lowest-id fragment (`remaining == [10]`, and the long trace's rows are deleted).

- [ ] **Step 3: Add `depth_span` to `ClusterRow`**

In `nzgd/dedup/canonical_selectors.py`, add the field to the dataclass. Replace:

```python
@dataclass(frozen=True)
class ClusterRow:
    """Compact summary of one cptreport/sptreport row for selector input."""

    report_id: int                  # cpt_id or spt_id
    has_data: bool                  # has_cpt_data=1 for CPT; measurement_row_count > 0 for SPT
    measurement_row_count: int
    metadata_non_null_count: int    # non-NULL fields in cptreport/sptreport metadata
```

with:

```python
@dataclass(frozen=True)
class ClusterRow:
    """Compact summary of one cptreport/sptreport row for selector input."""

    report_id: int                  # cpt_id or spt_id
    has_data: bool                  # has_cpt_data=1 for CPT; measurement_row_count > 0 for SPT
    measurement_row_count: int
    metadata_non_null_count: int    # non-NULL fields in cptreport/sptreport metadata
    depth_span: float               # max_depth - min_depth of the trace; 0.0 if no finite depth
```

- [ ] **Step 4: Rank the selector by depth span**

In `nzgd/dedup/canonical_selectors.py`, replace:

```python
def default_within_record_canonical(
    cluster_rows: Sequence[ClusterRow],
    table_cfg: TableConfig,
) -> int:
    """v1 default: prefer rows with has_data=True; tiebreaker smallest report_id."""
    candidates = [r for r in cluster_rows if r.has_data] or list(cluster_rows)
    return min(candidates, key=lambda r: r.report_id).report_id
```

with:

```python
def default_within_record_canonical(
    cluster_rows: Sequence[ClusterRow],
    table_cfg: TableConfig,
) -> int:
    """v2 default: prefer has_data rows; among them the widest depth span; tiebreaker smallest report_id."""
    candidates = [r for r in cluster_rows if r.has_data] or list(cluster_rows)
    return min(candidates, key=lambda r: (-r.depth_span, r.report_id)).report_id
```

- [ ] **Step 5: Import the depth-extent helper into Pass 0**

In `nzgd/dedup/pass0_within_record.py`, extend the existing trace_compare import. Replace:

```python
from nzgd.dedup.trace_compare import best_trace_score, load_traces
```

with:

```python
from nzgd.dedup.trace_compare import best_trace_score, load_traces, trace_depth_extent
```

- [ ] **Step 6: Compute per-report extents/spans and pass `depth_span` into `ClusterRow`**

In `nzgd/dedup/pass0_within_record.py`, inside `generate_within_record_consolidation_plan`, replace the top of the cluster loop:

```python
        for cluster_report_ids in clusters:
            if len(cluster_report_ids) <= 1:
                continue
            cluster_rows = [
                ClusterRow(
                    report_id=rid,
                    has_data=measurement_count_by_id[rid] > 0,
                    measurement_row_count=measurement_count_by_id[rid],
                    metadata_non_null_count=_metadata_non_null_count(conn, rid, table_cfg),
                )
                for rid in cluster_report_ids
            ]
            canonical_id = canonical_selector(cluster_rows, table_cfg)
```

with:

```python
        for cluster_report_ids in clusters:
            if len(cluster_report_ids) <= 1:
                continue
            extent_by_id = {
                rid: (trace_depth_extent(traces[rid]) if rid in traces else (math.nan, math.nan))
                for rid in cluster_report_ids
            }
            span_by_id = {
                rid: (hi - lo if math.isfinite(lo) and math.isfinite(hi) else 0.0)
                for rid, (lo, hi) in extent_by_id.items()
            }
            cluster_rows = [
                ClusterRow(
                    report_id=rid,
                    has_data=measurement_count_by_id[rid] > 0,
                    measurement_row_count=measurement_count_by_id[rid],
                    metadata_non_null_count=_metadata_non_null_count(conn, rid, table_cfg),
                    depth_span=span_by_id[rid],
                )
                for rid in cluster_report_ids
            ]
            canonical_id = canonical_selector(cluster_rows, table_cfg)
```

- [ ] **Step 7: Run the new test — verify it passes**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest "tests/dedup/test_dedup_pipeline.py::test_pass0_canonical_prefers_widest_depth_span" -v`
Expected: PASS.

- [ ] **Step 8: Run the full dedup suite + ruff — verify no regressions**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -q`
Expected: `40 passed` (39 baseline + 1 new).

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m ruff check nzgd/dedup/canonical_selectors.py nzgd/dedup/pass0_within_record.py tests/dedup/test_dedup_pipeline.py`
Expected: no new findings (clean, or unchanged pre-existing count — do not fix unrelated pre-existing findings).

- [ ] **Step 9: Commit**

```bash
git add nzgd/dedup/canonical_selectors.py nzgd/dedup/pass0_within_record.py tests/dedup/test_dedup_pipeline.py
git commit -m "feat(dedup): rank within-record canonical by depth span

ClusterRow gains depth_span; default_within_record_canonical prefers the
widest depth coverage (tiebreak smallest report_id) so a short fragment
no longer wins canonical over a long trace it fuzzy-matches on the overlap.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Keep-if-not-contained safeguard

Absorb (delete) a non-canonical report only if its depth extent is contained in the canonical's; otherwise keep it as a separate report. This closes the genuine partial-overlap case (neither trace contains the other) and makes Pass 0 provably no-loss.

**Files:**
- Modify: `nzgd/dedup/pass0_within_record.py` (the absorbed-report loop in `generate_within_record_consolidation_plan`)
- Test: `tests/dedup/test_dedup_pipeline.py` (add one integration test in the Pass 0 section)

**Interfaces:**
- Consumes: `extent_by_id: dict[int, tuple[float, float]]` built in Task 1 inside the cluster loop; `canonical_id: int` from the selector. `math` is already imported in `pass0_within_record.py`.
- Produces: no signature changes. `apply_within_record_consolidation_plan` is unchanged (it deletes exactly what is in `absorbed_reports`).

- [ ] **Step 1: Write the failing test**

Add this function to the Pass 0 section of `tests/dedup/test_dedup_pipeline.py` (near the Task 1 test).

```python
def test_pass0_keeps_partial_overlap_reports(fresh_db: sqlite3.Connection) -> None:
    """Two traces that are identical on their overlap but where neither contains the
    other (0.1-1.0 vs 0.6-1.6) cluster together, yet must both survive — deleting
    either would lose the depth the other lacks."""
    def qc(d: float) -> float:
        return 2.0 + d
    trace_a = [(round(0.1 * i, 1), qc(0.1 * i), 0.02, 0.0) for i in range(1, 11)]    # 0.1..1.0
    trace_b = [(round(0.1 * i, 1), qc(0.1 * i), 0.02, 0.0) for i in range(6, 17)]    # 0.6..1.6
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0)
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace_a, source_file="A.xlsx_sheet_Lower")
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=1, trace=trace_b, source_file="A.xlsx_sheet_Upper")

    n_clusters, n_records = _run_pass0(fresh_db, CPT_TABLE_CONFIG)
    assert (n_clusters, n_records) == (0, 0)  # partial overlap -> no absorption
    remaining = [r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport ORDER BY cpt_id")]
    assert remaining == [10, 20]  # both reports kept
    deep = fresh_db.execute("SELECT MAX(depth_m) FROM cptmeasurements WHERE cpt_id = 20").fetchone()[0]
    assert deep == 1.6  # the deeper report's coverage is intact
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest "tests/dedup/test_dedup_pipeline.py::test_pass0_keeps_partial_overlap_reports" -v`
Expected: FAIL — after Task 1 the two reports still cluster and collapse (canonical 10 by tie-break, report 20 deleted), so `remaining == [10]` and `n_clusters == 1`.

- [ ] **Step 3: Add the containment filter to the absorbed loop**

In `nzgd/dedup/pass0_within_record.py`, inside `generate_within_record_consolidation_plan`, replace:

```python
            canonical_id = canonical_selector(cluster_rows, table_cfg)
            has_data_by_id = {rid: measurement_count_by_id[rid] > 0 for rid in cluster_report_ids}
            absorbed = []
            for rid in cluster_report_ids:
                if rid == canonical_id:
                    continue
                absorbed.append(_AbsorbedReport(
                    absorbed_report_id=rid,
                    absorbed_source_file=source_file_by_id.get(rid) or "",
                    trace_match=_classify_match(
                        canonical_id, rid, has_data_by_id, traces,
                        thresholds["trace_score_max"], thresholds["trace_resample_step_m"],
                    ),
                ))
            plans.append(WithinRecordConsolidation(
```

with:

```python
            canonical_id = canonical_selector(cluster_rows, table_cfg)
            canonical_lo, canonical_hi = extent_by_id[canonical_id]
            has_data_by_id = {rid: measurement_count_by_id[rid] > 0 for rid in cluster_report_ids}
            absorbed = []
            for rid in cluster_report_ids:
                if rid == canonical_id:
                    continue
                lo_a, hi_a = extent_by_id[rid]
                contained = not (math.isfinite(lo_a) and math.isfinite(hi_a)) or (
                    math.isfinite(canonical_lo) and canonical_lo <= lo_a and hi_a <= canonical_hi
                )
                if not contained:
                    # genuine partial overlap: keep as a separate report under the same nzgd_id
                    continue
                absorbed.append(_AbsorbedReport(
                    absorbed_report_id=rid,
                    absorbed_source_file=source_file_by_id.get(rid) or "",
                    trace_match=_classify_match(
                        canonical_id, rid, has_data_by_id, traces,
                        thresholds["trace_score_max"], thresholds["trace_resample_step_m"],
                    ),
                ))
            if not absorbed:
                continue
            plans.append(WithinRecordConsolidation(
```

- [ ] **Step 4: Run the new test — verify it passes**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest "tests/dedup/test_dedup_pipeline.py::test_pass0_keeps_partial_overlap_reports" -v`
Expected: PASS.

- [ ] **Step 5: Run the full dedup suite + ruff — verify no regressions**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -q`
Expected: `41 passed` (39 baseline + 2 new). In particular the containment-case tests (`test_pass0_typical_multi_sheet_collapse`, `test_pass0_all_no_data_cluster`, `test_pass0_stem_only_attachment`, `test_pass0_spt_consolidation_cascade`, `test_pass0_single_file_multi_cpt_split`) still pass — their absorbed reports are all contained (identical/equal-length traces or no finite depth).

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m ruff check nzgd/dedup/pass0_within_record.py tests/dedup/test_dedup_pipeline.py`
Expected: no new findings.

- [ ] **Step 6: Commit**

```bash
git add nzgd/dedup/pass0_within_record.py tests/dedup/test_dedup_pipeline.py
git commit -m "feat(dedup): keep within-record reports not contained in the canonical

A non-canonical report is absorbed (deleted) only when its depth extent is
contained in the canonical's (or it has no finite depth); a genuine
partial-overlap report is kept as a separate report. The only Pass 0
deletion iterates absorbed_reports, so within-record consolidation is now
provably no-loss.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Validation (post-merge, not a code task)

After both tasks land, confirm the real-data impact — the same source-vs-deduped span check used for the fuzzy fix.

- [ ] **V1: Back up the current deduped DB** (per the re-run procedure in memory `project_cpt_constant_column_filter`): copy/rename the existing `uc_nzgd_v0p8p1_..._deduped.db` to a dated `.backup-2026-07-05` before regenerating.
- [ ] **V2: Re-run the full dedup** on a copy of the source DB via `nzgd/scripts/db/deduplicate.py --source <source_db>` (never modify the source; the script writes a fresh deduped DB). Expect exit 0.
- [ ] **V3: Verify coverage restored** for the flagged records — for each of nzgd 8809, 186134, 186131, 118991 (and the already-fixed 193652), compare max depth-span across the record's reports in the source DB vs the new deduped DB; assert no material loss (the pre-fix losses were 10–22 m). Report the before/after spans and the change in Pass 0 within-record absorbed counts.

---

## Self-Review

**Spec coverage:**
- §4.1 completeness-aware selection → Task 1 (Steps 3-4: `depth_span` field + `(-depth_span, report_id)` key). ✓
- §4.2 keep-if-not-contained (extent computation, span plumbing) → Task 1 Step 6 (`extent_by_id`/`span_by_id`) + Task 2 Step 3 (containment filter, `if not absorbed: continue`). ✓
- §4.3 airtight argument (only-delete-if-contained) → Task 2 Step 3 gates `absorbed_reports`; `apply_` unchanged. ✓
- §4.4 clustering/enrichment/audit/`trace_compare.py` unchanged → no task touches them. ✓
- §6 tests: containment (widest kept) → Task 1 test; partial overlap (keep both) → Task 2 test; regression 12 Pass 0 tests → Steps 8/5 assert full suite green. Real-data validation → Validation section. ✓
- §3 no config / no new selector → Global Constraints forbid touching config/deduplicate.py; selector fixed in place. ✓

**Placeholder scan:** none — every code step shows complete old/new blocks and exact commands.

**Type consistency:** `depth_span: float` defined in Task 1 Step 3, produced by `span_by_id[rid]` (float) in Step 6, consumed by the selector key in Step 4. `extent_by_id: dict[int, tuple[float, float]]` defined in Task 1 Step 6, consumed in Task 2 Step 3 (`canonical_lo, canonical_hi = extent_by_id[canonical_id]`, `lo_a, hi_a = extent_by_id[rid]`). `trace_depth_extent` return type matches its existing definition. Test helper names (`add_cpt_record`, `add_cpt_report`, `_run_pass0`, `CPT_TABLE_CONFIG`) match existing module-level usages. ✓
