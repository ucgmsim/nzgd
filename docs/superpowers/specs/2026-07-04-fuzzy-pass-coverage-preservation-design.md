# Fuzzy-pass depth-coverage preservation — design

**Date:** 2026-07-04
**Status:** Approved design, pending spec review
**Scope:** `nzgd/dedup/` cross-record fuzzy pass (Pass 2) and the shared canonical selector.

## 1. Goal

Stop the deduplication fuzzy pass from dropping real depth coverage. Today ~46 CPT
records (~0.1%) lose >0.5 m of depth because the fuzzy pass merges traces of very
unequal length and then keeps the shorter one. After this change, **no fuzzy merge
reduces depth coverage** — a report is deleted only when its depth range is
contained in a surviving (canonical) report; otherwise it is reparented (kept), or
the pair is not merged at all.

## 2. Background — root cause (two combining defects)

The over-collapse is documented in `docs/cptreport_count_reduction_verification.md`
(46 records lost >0.5 m, all from the fuzzy passes; hash loses nothing). In code:

1. **Fuzzy matching compares only the overlapping depth range.** `trace_score`
   (`trace_compare.py`) computes RMSE on `[max(min_depths), min(max_depths)]`. So a
   full **0–22 m** trace and a **0–0.14 m** fragment match perfectly on their
   0–0.14 m overlap and fire the predicate. (`max_depth_diff_m` is computed but the
   predicate never checks it.)
2. **Canonical selection ignores the matched trace's size.** `select_canonical`
   (`selection.py`) ranks by **unique (non-matched) measurement rows**, deliberately
   excluding the matched report. When each record's only report *is* the matched
   pair, both score 0 unique rows and the tie falls through metadata to **smallest
   `nzgd_id`** — which can be the fragment. The 22 m trace is deleted; the 0.14 m one
   survives.

3 of the 4 worst cases (8809, 193652, 186134) are *containment* — the survivor's
range sits inside the deleted trace — so fixing canonical selection alone removes
those losses. Only genuine partial overlap (e.g. 118991) additionally needs a guard.

## 3. Decisions (approved)

- **Approach A — both levers:** completeness-aware canonical selection **and** a
  containment guard in the fuzzy predicate.
- **Completeness metric = depth *span*** (`max_depth − min_depth` extent), not
  measurement count — it targets coverage directly.
- **`containment_frac = 0.9`** default merge cutoff, config-tunable, validated
  against the flagged records during implementation.
- **Partial overlap → keep as two separate records** (never delete a non-contained
  trace).
- **Scope:** `select_canonical` + the fuzzy pass. Hash pass is unaffected (its traces
  are bit-identical). **Non-goal:** Pass 0 within-record uses a different selector and
  was not flagged as lossy — left as a possible follow-up.

## 4. Design

### 4.1 Completeness-aware canonical selection — `nzgd/dedup/selection.py`

`select_canonical` gains an optional argument:

```python
def select_canonical(conn, cluster_nzgd_ids, matched_pairs, table_cfg,
                     completeness: dict[int, float] | None = None) -> int:
```

The sort key gains a new **primary** term:

```python
cov = completeness.get(nz, 0.0) if completeness is not None else 0.0
scored.append((-cov, -unique_rows, -meta_count, nz))
```

- When `completeness is None` (the hash pass call), `cov = 0.0` for every record, so
  the key reduces to the current `(-unique_rows, -meta_count, nz)` — **behavior
  unchanged for hash**.
- When provided (fuzzy), the record with the widest depth coverage wins, so the
  most-complete trace survives.

### 4.2 Completeness map + containment guard — `nzgd/dedup/pass2_fuzzy.py`

**Depth extent helper** (small, in `trace_compare.py`, reused by both):

```python
def trace_depth_extent(arr: np.ndarray) -> tuple[float, float]:
    """(min, max) of finite depths (column 0); (nan, nan) if none finite."""
```

**Completeness map:** the fuzzy pass already loads every candidate record's traces
into `trace_cache`. Before/at `select_canonical`, build
`completeness[nz] = max over the record's traces of (max_extent − min_extent)`,
0.0 if a record has no finite-depth trace, and pass it into `select_canonical`.

**Containment guard:** for the best-matching report pair `best_pair = (ra, rb)`
(only computed when `best_pair is not None`, i.e. the traces overlap and
`trace_score` is finite), compute from the two matched traces:

```
span_a, span_b = extent spans of ra, rb
overlap_extent = max(0.0, min(max_a, max_b) − max(min_a, min_b))
overlap_containment = 1.0 if min(span_a, span_b) == 0 else overlap_extent / min(span_a, span_b)
```

`overlap_containment` is always present in the `features` dict — it is `0.0` when
`best_pair is None` (no overlap), so the key always exists. Extend `_predicate`:

```python
if features["overlap_containment"] < thresholds["containment_frac"]:
    return False
```

(When `best_pair is None`, `trace_score` is `inf`, so `_predicate` already returns
`False` at the earlier `trace_score` check, and `predicate_matched` separately
requires `best_pair is not None` — the containment term is belt-and-suspenders.)

- Contained pair (0.14 m ⊆ 22 m; 21.9 m vs 22.0 m) → ratio ≈ 1 → **merged**
  (and §4.1 keeps the longer).
- Partial overlap (0–15 m vs 5.5–20 m → 9.5/14.5 ≈ 0.66) → **not merged** →
  both records kept.
- A degenerate zero-span trace that overlaps → `overlap_containment = 1.0` → merged,
  and §4.1 keeps the longer (deleting the point loses nothing).
- The guard only ever *removes* edges, so it is strictly conservative — it can never
  create a new (wrong) merge.

`overlap_containment` is a feature, so it appears in the calibration report
alongside the others.

### 4.2b Delete-vs-reparent — the airtight guarantee

The guard proves the shorter of the *two matched reports* is contained in the
longer. But `best_trace_score` picks the most-*similar* pair, not the longest, so
on a multi-report record it can pair the canonical's short report with the merged
record's long report — and deleting that long report would drop the coverage it
has beyond the survivor. To close this, in `generate_fuzzy_merge_plan` a matched
merged report is **deleted only if its depth extent is contained in one of the
canonical's reports**; otherwise it is dropped from `matched_pairs`, which routes
it into `unique_merged_report_ids` and it is **reparented** (kept under the
canonical). This makes "a fuzzy-deleted report is always contained in a survivor"
hold unconditionally, including multi-report records. Reparenting a not-contained
report is strictly safe (it only ever keeps more data). A report with no finite
depth extent counts as contained (deleting it loses nothing).

### 4.3 Config — `nzgd/resources/config.yaml`

Add under `deduplication.fuzzy_pass`:

```yaml
    containment_frac: 0.9
```

`deduplicate.py` already builds `fuzzy_thresholds` by spreading `fuzzy_pass`, so the
value threads through to `generate_fuzzy_merge_plan` → `_predicate` with no driver
change. (Existing pass tests build their own `_DEFAULT_THRESHOLDS` dict; those add
`"containment_frac": 0.9` too.)

## 5. Files changed

| File | Change |
|------|--------|
| `nzgd/dedup/trace_compare.py` | add `trace_depth_extent` helper |
| `nzgd/dedup/selection.py` | `select_canonical` optional `completeness`; coverage as primary sort key |
| `nzgd/dedup/pass2_fuzzy.py` | build completeness map; compute `overlap_containment`; add it to features; guard in `_predicate`; pass completeness to `select_canonical` |
| `nzgd/resources/config.yaml` | `fuzzy_pass.containment_frac: 0.9` |
| test module | integration tests (see §6) |

## 6. Testing

Integration tests against the real SQLite schema (`fresh_db`), three regimes:

1. **Fragment vs full (containment → keep full):** two spatially-close, same-name
   records — one with a full 0–3 m varying trace, one with a 0–0.3 m fragment of it.
   Run the fuzzy pass. Assert they **merge** and the **full-coverage record is
   canonical** (the fragment's report is deleted, not the full one). Fails on
   today's code (fragment/lowest-id wins).
2. **Partial overlap (guard → keep both):** two records, traces 0–2 m and 1.5–3.5 m
   (overlap 0.5 m, spans 2 m → containment 0.25 < 0.9). Assert **no merge** — both
   records survive.
3. **Near-identical full (regression → still merge):** two near-identical full
   traces → still merged (containment ≈ 1), so normal dedup is preserved.

Plus a verification step during implementation: check `containment_frac = 0.9`
against the flagged `nzgd_id`s (8809, 193652, 186134, 118991, …) — confirm each is
either contained (merge, longer kept) or guard-blocked, with no depth loss.

## 7. Risks & rollback

- **Shared `select_canonical`:** the new arg defaults to `None`, so the hash pass and
  any other caller are unaffected — verified by the existing hash tests still passing.
- **Fewer merges:** the guard keeps genuine partial-overlap pairs as separate records,
  so the deduped DB has slightly more CPT records. This is intended (they cover
  different depth). Magnitude quantified during implementation.
- **Gappy traces:** containment is extent-based (`max − min`), so a canonical report
  with an internal depth gap that still *spans* the merged report's range counts as
  containing it even if its sample density differs there. Acceptable first-order (the
  flagged cases are simple fragments); refinable later to covered-length.
- **Chained reparenting (non-loss):** the completeness map and canonical extents are a
  pre-execution snapshot, so within one cluster a later merged record can't "see" a
  report reparented onto the canonical by an earlier one. This only ever keeps *more*
  data (never deletes something uncontained), so it cannot lose coverage — it can just
  occasionally forgo a dedup opportunity a chained check would have found.
- **Tuning:** `containment_frac` is config-tunable; the default is validated against
  the flagged records.
- **Reversible:** source DB is never modified; re-running dedup regenerates the
  deduped DB.

## 8. Impact summary

Eliminates the ~46 fuzzy over-collapse coverage losses (guaranteed: no fuzzy merge
drops depth coverage). Slightly fewer fuzzy merges (partial-overlap pairs kept
separate). Hash pass, Pass 0, and within-record consolidation are unchanged.
