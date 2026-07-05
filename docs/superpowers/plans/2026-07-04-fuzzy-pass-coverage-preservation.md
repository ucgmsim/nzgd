# Fuzzy-Pass Depth-Coverage Preservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the dedup fuzzy pass from dropping depth coverage — the deleted trace is always contained in the survivor, or the pair isn't merged.

**Architecture:** Two levers. (1) `select_canonical` gains an optional `completeness` map (depth-span per nzgd_id) as its primary rank key, so the most-complete trace survives; the fuzzy pass builds this from its already-loaded trace cache. Inert for the hash pass (passes no map). (2) A containment guard in the fuzzy predicate rejects pairs whose shorter trace isn't ~contained in the longer's depth extent. A shared `trace_depth_extent` helper backs both.

**Tech Stack:** Python 3.12, numpy, SQLite (`sqlite3`), pytest. No new dependencies.

## Global Constraints

- **Python / tests:** use `/home/arr65/venvs/dev_nzgd_venv/bin/python` for everything. Run tests with `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest`.
- **Style:** ruff, numpy docstrings, type hints. The project enables ruff `ANN` (not test-exempt), so **every new test function needs type-annotated params + `-> None`**, and **all test imports go at the top of the file** (no `E402`). Lint every changed file including tests.
- **Completeness metric = depth *span*** (`max_depth − min_depth` of finite depths). `containment_frac = 0.9` default.
- **`select_canonical` stays backward-compatible:** the new `completeness` argument defaults to `None`; when `None` the ranking is byte-for-byte the current behavior (so the hash pass and any other caller are unaffected). Its return index changes from `scored[0][2]` to `scored[0][3]` because the sort key gains a leading element — do not miss this.
- **Scope:** `trace_compare.py`, `selection.py`, `pass2_fuzzy.py`, `config.yaml`, and the dedup tests. Do NOT change the hash pass, Pass 0, `executor.py`, or the driver (`deduplicate.py` already spreads `fuzzy_pass` config into the thresholds dict, so `containment_frac` threads through automatically).
- **Guard is strictly conservative:** it only ever *removes* fuzzy edges; it can never create a new merge.
- **Import style (dedup package):** import names directly; `pass2_fuzzy.py` aliases trace-compare imports with a leading underscore (`_best_trace_score`, `_load_traces`, …) — follow that for the new import.
- **Spec:** `docs/superpowers/specs/2026-07-04-fuzzy-pass-coverage-preservation-design.md`.

---

### Task 1: Completeness-aware canonical selection

**Files:**
- Modify: `nzgd/dedup/trace_compare.py` (add `trace_depth_extent`)
- Modify: `nzgd/dedup/selection.py` (`select_canonical` gains `completeness`)
- Modify: `nzgd/dedup/pass2_fuzzy.py` (build completeness map, pass it in)
- Test: `tests/dedup/test_dedup_pipeline.py`

**Interfaces:**
- Produces:
  - `trace_depth_extent(arr: np.ndarray) -> tuple[float, float]` — `(min, max)` of finite depths (column 0); `(nan, nan)` if none finite.
  - `select_canonical(conn, cluster_nzgd_ids, matched_pairs, table_cfg, completeness: dict[int, float] | None = None) -> int` — ranks by `(-completeness, -unique_rows, -meta_count, nzgd_id)`.
  - `_record_completeness(traces: dict[int, np.ndarray]) -> float` in `pass2_fuzzy.py` — max depth span across a record's traces, `0.0` if none finite.

- [ ] **Step 1: Write the failing unit test for `trace_depth_extent`**

Add `import math` and `import numpy as np` to the top import block of `tests/dedup/test_dedup_pipeline.py` (with the stdlib / third-party groups), and `from nzgd.dedup.trace_compare import trace_depth_extent` to the `nzgd.dedup` import group. Then append this test to the end of the file:

```python
def test_trace_depth_extent() -> None:
    arr = np.array([[0.1, 1.0], [0.5, 2.0], [0.3, 3.0]])
    assert trace_depth_extent(arr) == (0.1, 0.5)
    nan_depths = np.array([[math.nan, 1.0], [math.nan, 2.0]])
    lo, hi = trace_depth_extent(nan_depths)
    assert math.isnan(lo) and math.isnan(hi)
    lo, hi = trace_depth_extent(np.empty((0, 2)))
    assert math.isnan(lo) and math.isnan(hi)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_trace_depth_extent -v`
Expected: FAIL — `ImportError: cannot import name 'trace_depth_extent'`.

- [ ] **Step 3: Implement `trace_depth_extent`**

In `nzgd/dedup/trace_compare.py`, add after `coerce_to_float` (the module already imports `math` and `numpy as np`):

```python
def trace_depth_extent(arr: np.ndarray) -> tuple[float, float]:
    """Return (min, max) of a trace's finite depths (column 0).

    Depth is column 0 per ``load_traces``. Returns ``(nan, nan)`` if the trace
    is empty or has no finite depth.
    """
    if arr.shape[0] == 0:
        return math.nan, math.nan
    depths = arr[:, 0]
    finite = depths[np.isfinite(depths)]
    if finite.size == 0:
        return math.nan, math.nan
    return float(finite.min()), float(finite.max())
```

- [ ] **Step 4: Run it to verify it passes**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_trace_depth_extent -v`
Expected: PASS.

- [ ] **Step 5: Write the failing canonical-selection integration test**

Append to `tests/dedup/test_dedup_pipeline.py`:

```python
def test_fuzzy_keeps_most_complete_trace(fresh_db: sqlite3.Connection) -> None:
    # nzgd 1 (smaller id) holds only a 0.1-0.3 m fragment; nzgd 2 holds the full
    # 0.1-3.0 m trace. They match on the fragment's overlap and merge. Today's
    # smallest-id tiebreak wrongly keeps the fragment; completeness must keep the
    # full trace.
    full = [(d / 10, 1.0 + 0.1 * d, 0.01 + 0.001 * d, 0.001 * d) for d in range(1, 31)]
    frag = full[:3]
    add_cpt_record(fresh_db, 1, -41.0, 174.0, "Site X", "2024-01-01")
    add_cpt_record(fresh_db, 2, -41.0, 174.00001, "Site X", "2024-01-02")
    add_cpt_report(fresh_db, 10, 1, frag)
    add_cpt_report(fresh_db, 20, 2, full)

    total_c, total_r = _run_both_passes(fresh_db, CPT_TABLE_CONFIG, _DEFAULT_THRESHOLDS)

    assert (total_c, total_r) == (1, 1)
    survivor = fresh_db.execute(
        "SELECT nzgd_id FROM nzgdrecord WHERE merged_into_nzgd_id IS NULL"
    ).fetchone()[0]
    assert survivor == 2  # the full-coverage record
    # the surviving trace is the full 30-row one; the fragment was deleted
    assert fresh_db.execute("SELECT COUNT(*) FROM cptmeasurements").fetchone()[0] == 30
```

- [ ] **Step 6: Run it to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_fuzzy_keeps_most_complete_trace -v`
Expected: FAIL — today the fragment record (nzgd 1) wins canonical, so `survivor == 1` and `cptmeasurements` count is 3.

- [ ] **Step 7: Implement completeness in `select_canonical` and wire it from the fuzzy pass**

In `nzgd/dedup/selection.py`, replace the `select_canonical` signature and body's scoring loop/return. The function becomes:

```python
def select_canonical(
    conn: sqlite3.Connection,
    cluster_nzgd_ids: Iterable[int],
    matched_pairs: Iterable[tuple[int, int, int, int]],
    table_cfg: TableConfig,
    completeness: dict[int, float] | None = None,
) -> int:
    """Pick the canonical nzgd_id from a cluster of nzgd_ids per the spec rule.

    When `completeness` (a `{nzgd_id: depth-coverage}` map) is provided, coverage
    is the primary key so the most-complete trace survives; ties fall through to
    the original rule (most unique measurement rows, then most non-null metadata,
    then smallest nzgd_id). When `completeness` is None the ranking is unchanged.
    """
    pairs = list(matched_pairs)
    nzgd_ids = list(cluster_nzgd_ids)
    scored = []
    for nz in nzgd_ids:
        matched_ids = _matched_report_ids_for_nzgd(nz, pairs)
        unique_rows = _unique_measurement_row_count(conn, nz, matched_ids, table_cfg)
        meta_count = _non_null_metadata_count(conn, nz)
        cov = completeness.get(nz, 0.0) if completeness is not None else 0.0
        # Sort key: maximise coverage, then unique_rows, then meta_count; minimise nzgd_id
        scored.append((-cov, -unique_rows, -meta_count, nz))
    scored.sort()
    return scored[0][3]
```

(Leave the `Parameters/Returns` numpy-doc block that follows if present, or fold the note above into it — keep the docstring accurate. The critical change beyond the new arg is `return scored[0][3]`, not `[2]`.)

In `nzgd/dedup/pass2_fuzzy.py`, extend the trace-compare import to add the new helper:

```python
from nzgd.dedup.trace_compare import (
    best_trace_score as _best_trace_score,
    coerce_to_float as _coerce_to_float,
    load_traces as _load_traces,
    trace_depth_extent as _trace_depth_extent,
    trace_score as _trace_score,
)
```

Add a module-level helper (near the other private helpers, e.g. after `_load_active_records`):

```python
def _record_completeness(traces: dict[int, np.ndarray]) -> float:
    """Widest depth span (max - min extent) across a record's traces; 0.0 if none finite."""
    best = 0.0
    for arr in traces.values():
        lo, hi = _trace_depth_extent(arr)
        if math.isfinite(lo) and math.isfinite(hi):
            best = max(best, hi - lo)
    return best
```

In `generate_fuzzy_merge_plan`, replace the canonical-selection call inside the cluster loop:

```python
        completeness = {nz: _record_completeness(traces_for(nz)) for nz in nzgd_ids}
        canonical = select_canonical(
            conn, nzgd_ids, matched_pairs_for_selection, table_cfg, completeness=completeness
        )
```

(`traces_for` returns the cached traces already loaded during the candidate-pair loop; every cluster member participated in an edge, so its traces are present.)

- [ ] **Step 8: Run the new test, the full suite, and lint**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_fuzzy_keeps_most_complete_trace -v`
Expected: PASS (`survivor == 2`, 30 measurement rows).

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/ -q`
Expected: PASS — all pre-existing tests plus the two new ones (the hash tests confirm `completeness=None` left hash unchanged).

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m ruff check nzgd/dedup/trace_compare.py nzgd/dedup/selection.py nzgd/dedup/pass2_fuzzy.py tests/dedup/test_dedup_pipeline.py`
Expected: `All checks passed!`. Fix any issue and re-run.

- [ ] **Step 9: Commit**

```bash
git add nzgd/dedup/trace_compare.py nzgd/dedup/selection.py nzgd/dedup/pass2_fuzzy.py tests/dedup/test_dedup_pipeline.py
git commit -m "fix(dedup): keep the most-complete trace as fuzzy canonical"
```

---

### Task 2: Containment guard in the fuzzy predicate

**Files:**
- Modify: `nzgd/dedup/pass2_fuzzy.py` (`overlap_containment` feature + `_predicate` check)
- Modify: `nzgd/resources/config.yaml` (`fuzzy_pass.containment_frac`)
- Test: `tests/dedup/test_dedup_pipeline.py` (`_DEFAULT_THRESHOLDS` + partial-overlap test)

**Interfaces:**
- Consumes: `_trace_depth_extent` (Task 1).
- Produces: fuzzy pairs are rejected when `overlap_containment < containment_frac`; `overlap_containment` appears in the feature dict / calibration.

- [ ] **Step 1: Write the failing partial-overlap test**

Append to `tests/dedup/test_dedup_pipeline.py`:

```python
def test_fuzzy_guard_rejects_partial_overlap(fresh_db: sqlite3.Connection) -> None:
    # Traces 0.0-2.0 m and 1.5-3.5 m: they agree on the 1.5-2.0 m overlap, but
    # neither contains the other (overlap 0.5 m / min span 2.0 m = 0.25 < 0.9),
    # so the guard must keep them as two separate records.
    a = [(d / 10, 1.0 + 0.1 * d, 0.01, 0.001 * d) for d in range(0, 21)]
    b = [(d / 10, 1.0 + 0.1 * d, 0.01, 0.001 * d) for d in range(15, 36)]
    add_cpt_record(fresh_db, 1, -41.0, 174.0, "Site Y", "2024-01-01")
    add_cpt_record(fresh_db, 2, -41.0, 174.00001, "Site Y", "2024-01-02")
    add_cpt_report(fresh_db, 10, 1, a)
    add_cpt_report(fresh_db, 20, 2, b)

    total_c, _ = _run_both_passes(fresh_db, CPT_TABLE_CONFIG, _DEFAULT_THRESHOLDS)

    assert total_c == 0
    merged = fresh_db.execute(
        "SELECT COUNT(*) FROM nzgdrecord WHERE merged_into_nzgd_id IS NOT NULL"
    ).fetchone()[0]
    assert merged == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_fuzzy_guard_rejects_partial_overlap -v`
Expected: FAIL — without the guard the two traces match on their overlap and merge, so `total_c == 1` and one record is merged.

- [ ] **Step 3: Add `containment_frac` config and to the test thresholds**

In `nzgd/resources/config.yaml`, add one line to the `deduplication.fuzzy_pass` block (after `trace_resample_step_m: 0.05`):

```yaml
    containment_frac: 0.9
```

In `tests/dedup/test_dedup_pipeline.py`, add the same key to `_DEFAULT_THRESHOLDS`:

```python
_DEFAULT_THRESHOLDS = {
    "spatial_radius_m": 50,
    "date_window_days": 90,
    "name_similarity_min": 80,
    "trace_score_max": 0.05,
    "trace_resample_step_m": 0.05,
    "containment_frac": 0.9,
}
```

- [ ] **Step 4: Implement the containment feature and guard in `pass2_fuzzy.py`**

In `generate_fuzzy_merge_plan`, in the candidate-pair loop, right after `trace_score, best_pair = _best_trace_score(ta, tb, step)` and the `max_depth_diff` block, compute the containment and add it to `features`:

```python
        overlap_containment = 0.0
        if best_pair is not None:
            ra_best, rb_best = best_pair
            lo_a, hi_a = _trace_depth_extent(ta[ra_best])
            lo_b, hi_b = _trace_depth_extent(tb[rb_best])
            if math.isfinite(lo_a) and math.isfinite(lo_b):
                span_a, span_b = hi_a - lo_a, hi_b - lo_b
                overlap = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
                min_span = min(span_a, span_b)
                overlap_containment = 1.0 if min_span == 0 else overlap / min_span
        features = {
            "spatial_m": spatial,
            "date_days": date_days,
            "name_sim": name_sim,
            "max_depth_diff_m": max_depth_diff,
            "trace_score": trace_score,
            "overlap_containment": overlap_containment,
        }
```

(The `features = { ... }` above replaces the existing five-key `features` dict — it adds the `overlap_containment` key.)

Then extend `_predicate` — add this check as the last one before `return True`:

```python
    if features["overlap_containment"] < thresholds["containment_frac"]:
        return False
    return True
```

- [ ] **Step 5: Run the partial-overlap test**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_fuzzy_guard_rejects_partial_overlap -v`
Expected: PASS (`total_c == 0`, no records merged).

- [ ] **Step 6: Run the full suite (regression) and lint**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/ -q`
Expected: PASS — in particular `test_slight_perturbation_pair_is_merged_by_fuzzy` and `test_fuzzy_keeps_most_complete_trace` still pass (near-identical and containment pairs have `overlap_containment ≈ 1`, so the guard allows them).

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m ruff check nzgd/dedup/pass2_fuzzy.py tests/dedup/test_dedup_pipeline.py`
Expected: `All checks passed!`.

- [ ] **Step 7: Commit**

```bash
git add nzgd/dedup/pass2_fuzzy.py nzgd/resources/config.yaml tests/dedup/test_dedup_pipeline.py
git commit -m "fix(dedup): guard fuzzy pass against merging partial-overlap traces"
```

---

## Optional real-data validation (not a committed test)

After both tasks, optionally confirm the `containment_frac = 0.9` default against the flagged records by re-running dedup on a real DB and checking that the previously-lost records (8809, 193652, 186134, 118991, …) either keep the fuller trace or stay unmerged:

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -m nzgd.scripts.db.deduplicate \
    --source /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p8p1_20260625.db \
    --target /tmp/cov_smoke_deduped.db --skip-spt
```
Then compare `cptreport.max_depth_m - min_depth_m` for those nzgd_ids against the source. Delete `/tmp/cov_smoke_deduped.db` afterwards.

---

## Self-Review

**Spec coverage:** completeness-aware `select_canonical` (spec §4.1) → Task 1. `trace_depth_extent` helper (§4.2) → Task 1. Completeness map (§4.2) → Task 1. Containment guard + `overlap_containment` feature (§4.2) → Task 2. Config `containment_frac` (§4.3) → Task 2. Tests for the three regimes (§6): fragment-vs-full → Task 1; partial-overlap → Task 2; near-identical → covered by the existing `test_slight_perturbation_pair_is_merged_by_fuzzy` confirmed in Task 2's full-suite run. Hash-unaffected guarantee (§7) → `completeness=None` default + existing hash tests in the suite run.

**Placeholder scan:** No TBD/TODO; every code step has complete code; every run step has an exact command and expected result.

**Type consistency:** `trace_depth_extent(arr) -> tuple[float, float]` defined in Task 1, used in Task 1 (`_record_completeness`) and Task 2 (guard) identically. `select_canonical(..., completeness=None)` defined and called with `completeness=` in Task 1. `_record_completeness(traces) -> float` defined and used in Task 1. `overlap_containment` feature key written in Task 2's `features` dict and read in Task 2's `_predicate`. `containment_frac` added to both `config.yaml` and `_DEFAULT_THRESHOLDS` in Task 2, read in `_predicate`. The `select_canonical` return-index change (`[2]`→`[3]`) is called out in Global Constraints and Task 1 Step 7.

---

### Task 3: Reparent (never delete) a matched report the canonical doesn't cover

Added after the final whole-branch review to close I1: `best_trace_score` picks the most-*similar* report pair, which for a **multi-report** record can pair the canonical's short report with the merged record's long report — and the long report would then be deleted, dropping the coverage it has beyond the survivor. Fix: delete a matched merged report only if its depth extent is contained in one of the canonical's reports; otherwise reparent it (keep it).

**Files:**
- Modify: `nzgd/dedup/pass2_fuzzy.py` (filter `matched_pairs` by containment before building `unique_ids`)
- Test: `tests/dedup/test_dedup_pipeline.py`

**Interfaces:**
- Consumes: `_trace_depth_extent` (Task 1); `traces_for` closure; `ReportPairMatch`; `math` (all already in `pass2_fuzzy.py`).
- Produces: in a fuzzy merge, a matched merged report whose depth extent is NOT contained in any canonical report is moved from `matched_pairs` (delete) to `unique_merged_report_ids` (reparent).

- [ ] **Step 1: Write the failing multi-report test**

Append to `tests/dedup/test_dedup_pipeline.py`:

```python
def test_fuzzy_reparents_uncontained_merged_report(fresh_db: sqlite3.Connection) -> None:
    # Multi-report canonical: nzgd 1 has a SHORT report (0.1-3.0 m) and a LONG,
    # unrelated report (10-35 m, distinct values). nzgd 2 has one LONG report
    # (0.1-20.0 m) that matches nzgd 1's SHORT report on 0.1-3.0 m. Completeness
    # makes nzgd 1 canonical (span 25 > 20); best_trace_score pairs (short, B_long),
    # so B_long would be deleted -- but it is NOT contained in any of nzgd 1's
    # reports, so it must be reparented (kept), not deleted.
    short = [(d / 10, 1.0 + 0.1 * d, 0.01, 0.001 * d) for d in range(1, 31)]
    b_long = [(d / 10, 1.0 + 0.1 * d, 0.01, 0.001 * d) for d in range(1, 201)]
    a_long = [(d / 10, 40.0 + 0.1 * d, 0.5, 0.0) for d in range(100, 351)]
    add_cpt_record(fresh_db, 1, -41.0, 174.0, "Multi Site", "2024-01-01")
    add_cpt_record(fresh_db, 2, -41.0, 174.00001, "Multi Site", "2024-01-02")
    add_cpt_report(fresh_db, 10, 1, short)
    add_cpt_report(fresh_db, 11, 1, a_long)
    add_cpt_report(fresh_db, 20, 2, b_long)

    total_c, total_r = _run_both_passes(fresh_db, CPT_TABLE_CONFIG, _DEFAULT_THRESHOLDS)

    assert (total_c, total_r) == (1, 1)
    assert fresh_db.execute(
        "SELECT merged_into_nzgd_id FROM nzgdrecord WHERE nzgd_id = 2"
    ).fetchone()[0] == 1
    # B_long (cpt 20) is KEPT (reparented to nzgd 1), not deleted -> its rows survive
    assert fresh_db.execute("SELECT COUNT(*) FROM cptreport WHERE cpt_id = 20").fetchone()[0] == 1
    assert fresh_db.execute("SELECT nzgd_id FROM cptreport WHERE cpt_id = 20").fetchone()[0] == 1
    assert fresh_db.execute("SELECT COUNT(*) FROM cptmeasurements WHERE cpt_id = 20").fetchone()[0] == 200
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_fuzzy_reparents_uncontained_merged_report -v`
Expected: FAIL — without the fix, `cpt_id 20` (B_long) is a matched pair and gets deleted, so `SELECT COUNT(*) ... WHERE cpt_id = 20` is `0` (assert `0 == 1`).

- [ ] **Step 3: Implement the containment filter**

In `nzgd/dedup/pass2_fuzzy.py`, inside `generate_fuzzy_merge_plan`'s `for merged_nz in ordered_merged:` loop, insert this block immediately AFTER the `matched_pairs` list is built (right after the `if key in per_pair_match:` / `matched_pairs.append(...)` block) and BEFORE the `cur.execute("SELECT ... report ids for merged_nz")`:

```python
            # Delete a matched merged report only if its depth extent is contained
            # in one of the canonical's reports; otherwise keep it (it falls into
            # unique_ids below and is reparented). best_trace_score pairs by
            # similarity, not length, so on a multi-report record the merged side's
            # matched report can be longer than the canonical report it matched --
            # reparenting instead of deleting guarantees no depth coverage is lost.
            canonical_extents = [
                _trace_depth_extent(arr) for arr in traces_for(canonical).values()
            ]
            merged_traces = traces_for(merged_nz)
            kept_matched: list[ReportPairMatch] = []
            for p in matched_pairs:
                lo_m, hi_m = _trace_depth_extent(merged_traces[p.merged_report_id])
                covered = not (math.isfinite(lo_m) and math.isfinite(hi_m)) or any(
                    lo_c <= lo_m and hi_m <= hi_c
                    for lo_c, hi_c in canonical_extents
                    if math.isfinite(lo_c)
                )
                if covered:
                    kept_matched.append(p)
            matched_pairs = kept_matched
```

The existing `matched_merged_ids = {p.merged_report_id for p in matched_pairs}` and `unique_ids = sorted(merged_reports - matched_merged_ids)` then automatically reparent any report dropped from `matched_pairs`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_fuzzy_reparents_uncontained_merged_report -v`
Expected: PASS (`cpt_id 20` kept, reparented to nzgd 1, 200 rows intact; nzgd 2 merged into 1).

- [ ] **Step 5: Run the full suite (regression) and lint**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/ -q`
Expected: PASS — in particular the fragment-vs-full and near-identical fuzzy tests still merge-and-delete (their matched report IS contained in the canonical, so `covered` is True).

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m ruff check nzgd/dedup/pass2_fuzzy.py tests/dedup/test_dedup_pipeline.py`
Expected: `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add nzgd/dedup/pass2_fuzzy.py tests/dedup/test_dedup_pipeline.py
git commit -m "fix(dedup): reparent (not delete) a fuzzy-matched report the canonical doesn't cover"
```
