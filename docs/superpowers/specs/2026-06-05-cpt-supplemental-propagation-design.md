# CPT Supplemental-Value Propagation Design

Date: 2026-06-05
Status: Draft for review

## Background

CPT supplemental metadata (`predrill_depth_m`, `extracted_gwl_m`,
`tip_net_area_ratio`, `termination_reason_id`, `gwl_method_id`) is extracted by
`scripts/extract/cpt/extract_all_potential_cpt_supplemental_values.py` and
filtered by `filter_potential_cpt_supplemental_values.py`. A separate external
backfill (`scripts/db/put_maxim_cpt_metadata_into_db.py`) was recently added to
fill values that the native pipeline appeared to "miss" — 6,929 `cptreport`
rows (predrill 3,274 / GWL 6,042 / tip 293).

An investigation (2026-06-05) found that the premise — that the *extractor*
misses these values — is wrong. Tracing all 6,929 backfilled rows end-to-end
(candidate CSV → filtered CSV → pre-dedup DB → deduped DB):

| Where the value is | share |
| --- | --- |
| **Already extracted AND filtered, then lost at the dedup merge/association step** | **98.5%** |
| Found as a candidate but dropped by the filter | 0.9% |
| Genuine extraction miss (no candidate ever found) | 0.6% |

**Root cause.** A record holds the same CPT in several files
(`CPT_3_AGS01.ags`, `CPT_3_AGS01.xls`, `CPT_3_RAW01.txt`). The supplemental
extractor reads the spreadsheet/text siblings and *does* find the values, but
`db/cpt_ids.py:assign()` joins supplemental → trace on the exact key
`nzgd_id_AND_filename_AND_sheetname`, so a value from a format-sibling never
reaches the deduplicated trace row (attributed to the `.ags` file). The
existing Pass 0 within-record carry-forward (`dedup/pass0_within_record.py`)
only enriches *within a single trace cluster*; when a record's siblings differ
in trace content (e.g. record 3: `.ags` vs `.xls` differ only in the u2 channel
→ `trace_score` 3.089 ≫ 0.05 → separate clusters, both survive), nothing
propagates and the canonical row stays NULL.

Verification (zero disagreements, conflicts benign):
- In 100% of recoverable cases our own extracted value *includes* Maxim's exact
  value (predrill 96.4% unique-match + 1.9% match-with-conflict; GWL 90.0% +
  8.6%; tip 100%).
- Within-record value conflicts across siblings: predrill 1.98%, GWL 10.1%
  (dominated by a `0.0` placeholder vs a real value), tip 0.05%.

### GWL `0.0` and the consolidation principle

68.9% of the GWL "misses" are `0.0`. The evidence
(`docs/gwl_zero_is_placeholder.md`) shows `0` is *statistically* a placeholder:
the value distribution spikes at exactly `0.0` (7,860 values) with a near-empty
band just above it; sampled sources write `0` under a `Water level`
template-default label (92%) or `Nil` (3%); and 5% carry a *positive* GWL in
another sibling. Only 2 of 7,860 zeros even carry a `measured` method, and those
are still `Measured GWL: 0` template fields indistinguishable from defaults.

**Guiding principle (settled with the user).** This DB is primarily an accurate
*consolidation of what the source files state*; it does not pass judgement on a
value's validity, **except** to prefer a more-valid alternative when one exists
for the same record. Concretely for GWL: prefer an in-range value if any source
for that `nzgd_id` has one; otherwise preserve the literal `0` the source wrote;
and yield NULL only when no value was extracted at all (`Nil`/blank/absent). The
configured range `extracted_gwl_m = [0.01, 50]` is used only to **rank**
competing values (in-range beats `0`), never to null a lone value — exactly the
rule the 2026-05-22 within-record spec already states ("plausibility consulted
only when picking between competing values… not to retroactively clean up values
that have no useful alternative"). The `Nil`-vs-`0` distinction falls out for
free: a literal `0` is numeric → extracted → preserved; `Nil`/blank is not a
number → never extracted → NULL.

## Goal

Fix the **association**, not the search: surface each record's already-extracted
supplemental values onto its surviving deduped rows, *from our own data*.
Concretely:

1. **Consolidate** each record's supplemental values onto its surviving rows:
   prefer the single in-range value; otherwise the value the source actually
   states (e.g. a literal `0`); NULL when nothing was extracted. Recovers the
   ~98.5% of the gap that is already in our own extraction.
2. Close the small genuine-extraction tail (49 predrill cells written as `Nil`).
3. Measure the difference vs Maxim and decide whether the Maxim backfill can be
   retired (deferred).

**Reproducing Maxim exactly is explicitly _not_ a goal.** The aim is a faithful
consolidation of the sources; where Maxim mis-assumes (e.g. it treats a blank /
`Nil` GWL as `0`) our result will and *should* differ — we record `NULL`. There
is **no value scrubbing** — lone source values (including a literal `0`) are
preserved.

## Scope

In scope:
- A new within-record **supplemental consolidation** step in `nzgd/dedup/`,
  wired into `scripts/db/deduplicate.py` to run after Pass 2 (the agreed "step
  in dedup pipeline" placement), for CPT records.
- Extraction/filter tail: accept a predrill cell value of `Nil` as `0.0`
  (predrill only); filter the GWL negative "no-water" sentinels (`-30`, `-60`,
  `-100`) to NULL at the filter stage (before `np.abs()`).
- `dedup_audit.match_pass` CHECK constraint widened to record the new step.
- Validation re-run measuring residual vs the Maxim-filled DB.

Not in scope:
- Re-engineering the supplemental extractor's keyword search (the investigation
  found it is not the cause; no missing-keyword cases were found).
- Any value scrub / DB-wide sentinel cleanup (the earlier draft proposed one;
  removed in favour of source-fidelity).
- A deeper fuzzy-matcher change to merge u2-divergent format-siblings into one
  cluster (record-3 case). Consolidation makes it unnecessary for this goal.
- Retiring `put_maxim_cpt_metadata_into_db.py` — deferred until residual measured.
- SPT (`sptreport`) consolidation — CPT only for v1 (the helper is table-generic).
- Investigating or fixing the **genuine residual extraction gaps** the
  measurement surfaces (e.g. values living only in native `.ags` or sheets we
  don't read). These are **reported only** and considered after this
  consolidation work is complete.

## Decisions settled during design

| Decision | Choice | Reason |
| --- | --- | --- |
| Root-cause fix location | Post-dedup within-record consolidation step in `deduplicate.py` (after Pass 2) | User-selected. Only option that catches both-survive, no-data-sibling, and cross-record re-parented cases; isolated from the fragile clustering passes |
| Grouping | All **surviving** `cptreport` rows sharing an `nzgd_id` (across clusters) | The gap is precisely siblings landing in *different* Pass-0 clusters |
| Consolidation rule | Per field: prefer the single in-range value; else the single recorded value (e.g. `0`); else NULL. Valid-value conflicts → B3 (below). Fill only NULL/non-useful cells; never override a valid value | Faithful to sources; prefers a more-valid alternative; never invents or nulls |
| GWL `0` semantics | Preserved when it is the only value; overridden by an in-range sibling value; `Nil`/blank/absent → NULL | User principle: DB reflects the sources; plausibility only ranks. `Nil`-vs-`0` distinction is automatic via the numeric extractor |
| No value scrub | The earlier draft's sentinel scrub is **removed** | Reverts to the 2026-05-22 principle (plausibility ranks, never scrubs a lone value) |
| Conflict handling (B3) | Valid-value conflict: small spread (≤ per-field threshold) → pick via the selector; large spread → skip + log (likely artifact) | Large spreads flag artifacts (e.g. `22`); small spreads are rounding-level, any source value is fine |
| Conflict selector | Most-corroborated value (mode) → tiebreak most-decimals (rounded 3 dp) → smallest `cpt_id` | Faithful, no directional bias, deterministic. Rejected: "smallest GWL" (shallow bias), most-decimals-alone (float noise) |
| `gwl_method_id` coupling | Taken from the same source row as the chosen `extracted_gwl_m` | The method describes that specific GWL value |
| Extractor tail | Accept predrill `Nil` → `0.0`; leave GWL `Nil` unmatched (→ NULL) | predrill `0` = "no pre-drilling", in range; GWL `Nil` is no value |
| GWL negative sentinels | Filter `-30`/`-60`/`-100` to NULL at the filter stage, before `np.abs()` (configurable list) | The RAW01.txt "no-water" defaults; `np.abs` fabricates `30`/`60`/`100`. Evidence in `docs/gwl_zero_is_placeholder.md` Part 2 |
| Maxim backfill | Keep for now; decide after measuring residual | User-selected |

## Architecture

A new module `nzgd/dedup/supplemental_consolidation.py` provides
`consolidate_within_record_supplemental(conn, table_cfg)`, invoked once per CPT
pass from `scripts/db/deduplicate.py` immediately after Pass 2 (fuzzy), before
the `dedup_run` finalisation:

```
Pass 0 (within-record consolidation)   [existing]
Pass 1 (cross-record hash)             [existing]
Pass 2 (cross-record fuzzy)            [existing]
→ consolidate_within_record_supplemental(conn, table_cfg)   [NEW]
```

It runs on the *survivors* (`cptreport` rows still present after all
deletions/re-parenting), so "the rows of a record" is final and stable. It is
idempotent and reuses `dedup.plausibility.is_useful_value` purely for ranking.
Because it is plain SELECT + narrow UPDATE over an existing deduped DB, it can
also run standalone against an already-deduped DB without a full re-dedup (used
for the measurement run and to regenerate the current DB).

### `consolidate_within_record_supplemental(conn, table_cfg)`

Supplemental columns (CPT): `extracted_gwl_m`, `gwl_method_id`,
`tip_net_area_ratio`, `predrill_depth_m`, `termination_reason_id`.

For each `nzgd_id` with ≥2 surviving rows, and for each column, compute the
record's **consolidated value** from the values present across its rows:

1. `useful = distinct values v where is_useful_value(v, "cptreport", col)`.
2. `recorded = distinct non-NULL values` (includes out-of-range values like `0`).
3. Resolve to a single **consolidated value** (or skip):
   - `len(useful) == 1` → consolidated = that value.
   - `len(useful) >= 2` → conflict among valid values. Let `spread = max(useful)
     - min(useful)`:
       - `spread <=` the field's `small_spread_threshold` → a rounding-level
         variant: consolidated = `select(rows holding a useful value)` (selector
         below).
       - `spread >` threshold → likely contains an artifact (e.g. a `22` from
         free text): **skip** this column for this record and add it to the
         conflict report; change nothing.
   - `len(useful) == 0` and `len(recorded) == 1` → consolidated = that recorded
     value (e.g. a literal `0`) — faithful, no better alternative.
   - `len(useful) == 0` and `len(recorded) >= 2` → skip + log (no in-range value
     to prefer among non-valid candidates).
   - `len(recorded) == 0` → nothing (all NULL).
4. **Fill, never override a valid value.** Set every cell that is NULL or holds a
   non-useful (out-of-range) value to `consolidated`; a cell already holding a
   useful value is left untouched. So a NULL canonical row receives the value and
   a sentinel `0` yields to an in-range value, but two conflicting *valid*
   siblings each keep their own reading.

   **Selector** (`select`, for a small-spread valid conflict): the value with the
   most corroboration — appearing on the most rows (mode) — breaking ties by
   (a) most decimal places after rounding to 3 dp, then (b) smallest `cpt_id`.
   Only ever returns a value present in the sources; unbiased in every field;
   deterministic. (Rejected: "smallest GWL" — biases the water table shallow;
   most-decimals alone — float/unit-conversion noise can fake precision.)
5. `extracted_gwl_m` and `gwl_method_id` move together: compute the GWL source
   row (smallest `cpt_id` holding the consolidated GWL), then set
   `gwl_method_id` on the record's rows from that same source row.
   `gwl_method_id` is not consolidated independently.

`termination_reason_id` (FK, no plausibility range) uses the same rule with
`useful == recorded` (every non-NULL value is "useful" for a column with no
configured range, per `is_useful_value`).

Note this never overrides an in-range value with `0`, and never nulls a value:
the only overrides are NULL→value and out-of-range→in-range.

### Audit & reporting

- `dedup_audit`: widen the `match_pass` CHECK constraint to add
  `'supplemental_consolidation'` (same idempotent table-recreate migration the
  2026-05-22 spec used for `'within_record'`). One audit row per affected
  `nzgd_id`: `canonical_nzgd_id == merged_nzgd_id == nzgd_id`, empty
  `report_pairs_json`, `metadata_copied_json =
  {"<col>": {"value": v, "source_report_id": Z, "target_report_ids": [...]}}`.
- A new CSV report `supplemental_consolidation_report.csv`: per-field counts of
  cells filled (NULL→value), overridden (out-of-range→in-range), and small-spread
  conflicts resolved by the selector; plus the **skipped-conflict list**
  (`nzgd_id`, column, the distinct values, spread). The large-spread entries in
  that list are the artifact-detection feed for the precision pass.

### Extraction / filter tail

**Predrill `Nil` → `0.0`** (predrill only): a candidate cell value of exactly
`Nil`/`nil`/`NIL` (optionally surrounded by whitespace), matched against a
predrill label, yields `predrill_depth = 0.0`. The extract stage currently emits
no candidate for a non-numeric value, so the predrill value-pattern must
additionally admit `nil`, with the filter mapping it to `0.0` (exact
extract-vs-filter placement resolved during planning). GWL `Nil` is deliberately
**not** mapped (stays unmatched → NULL). Affects ~49 records; takes effect only
on a full re-extraction; lowest priority, may be deferred.

**GWL negative "no-water" sentinels → NULL** (filter stage): before `np.abs()`,
drop a GWL candidate whose raw value is in a configurable sentinel list —
confirmed members `-30` (291 records), `-60` (113), `-100` (86), the RAW01.txt
`Waterlevel: -30/-60/-100` defaults. `np.abs()` otherwise fabricates
`30`/`60`/`100`; the `-30` case is the most harmful because `30` lands *inside*
`[0.01, 50]` and pollutes the useful tier (`-60`/`-100` → `60`/`100` are already
out of range, but would otherwise be recorded for records whose only GWL is the
sentinel). Filter-only (the negatives are already captured as candidates), so
re-running the filter regenerates corrected values without re-extraction.
Evidence: `docs/gwl_zero_is_placeholder.md` Part 2.

### Out-of-scope precision issue: free-text false positives

Some candidates are numbers pulled from prose where a keyword appears in a
comment — e.g. `22` from *"22 Tonne MAN Truck"*, or `250` from an equipment-ID
cell `MH250` adjacent to an `SWL …` comment. **No clean filter rule separates
these from legitimate values:** NZGD supplemental values are themselves
frequently semi-free-text (`SWL = 0.9m.b.g.l`, `Water Level 2.5m` inside a
comment), so a value-cell "noise" test flags ~14.9k records (mostly legitimate),
and most prose-label artifacts already produce out-of-range numbers the
plausibility tier excludes. The harmful *in-range* residue (like `22`) is small
(~37 lone-`22` records; the rest are large-spread conflicts B3 already skips) and
is listed in the skipped-conflict log. This precision work needs its own
validated design and is **out of scope here**.

## Config

One new block — the per-field `small_spread_threshold` used by the B3 conflict
selector:

```yaml
within_record_supplemental:
  small_spread_threshold:
    predrill_depth_m:   0.5    # metres
    extracted_gwl_m:    0.5    # metres
    tip_net_area_ratio: 0.05   # dimensionless ratio
```

Loaded into `constants.DEDUP_CONFIG` per the existing pattern; starting values,
tuned after the first real-data run. Otherwise consolidation reuses the existing
`field_plausibility_ranges.cptreport` block (for ranking only).

## Module layout

```
nzgd/dedup/
    plausibility.py                 # unchanged (is_useful_value reused for ranking)
    schema.py                       # widen dedup_audit.match_pass CHECK
    supplemental_consolidation.py   # NEW — consolidate_within_record_supplemental
    reports.py                      # tiny edit — write supplemental_consolidation_report.csv
nzgd/scripts/db/
    deduplicate.py                  # invoke consolidation after Pass 2 (CPT)
nzgd/scripts/extract/cpt/
    filter_potential_cpt_supplemental_values.py       # predrill Nil → 0.0 (tail)
    extract_all_potential_cpt_supplemental_values.py  # admit "nil" predrill value (tail)
tests/dedup/
    test_dedup_pipeline.py          # +integration scenarios
```

No new runtime dependencies.

## Testing strategy

Same philosophy as the existing dedup specs (minimise tests; integration over
unit; no implicit library testing). Append scenarios to
`tests/dedup/test_dedup_pipeline.py`, using the existing synthetic-DB fixtures:

1. **Cross-cluster consolidation (record-3 shape)** — one `nzgd_id`, two
   surviving rows in different clusters: `.ags` with NULL supplemental, `.xls`
   with predrill + GWL `1.3`. Verify: the `.ags` row receives predrill and GWL
   `1.3`; audit records the copy.
2. **Split-across-siblings (record-170 shape)** — predrill+GWL on one surviving
   row, tip-ratio on another; both rows end up with all three values.
3. **GWL `0` preserved when it is the only value** — a record whose only GWL
   value is `0` (others NULL); after consolidation every row reads `0` (NOT
   nulled).
4. **GWL `0` overridden by a positive sibling** — one row GWL `0`, another GWL
   `1.5`; both rows end up `1.5` (in-range beats `0`).
5a. **Small-spread conflict → selector** — canonical NULL; siblings with predrill
   `0.75` and `0.80` (spread 0.05 ≤ 0.5). Verify: the NULL canonical is filled
   with the selected value; each sibling keeps its own reading; selection logged.
5b. **Corroboration** — canonical NULL; rows `0.8`, `0.8`, `0.75`. Verify:
   canonical filled with `0.8` (mode), not `0.75`.
5c. **Large-spread conflict → skip** — GWL `1.5` vs `22.0` (spread 20.5 > 0.5).
   Verify: nothing changes; canonical stays NULL; conflict reported (artifact feed).
6. **predrill `0` untouched** — predrill `0` (in range) with no other value stays
   `0` and is not treated as a sentinel.
7. **gwl_method coupling** — GWL consolidated from a sibling also pulls that
   sibling's `gwl_method_id`; when the consolidated GWL is a preserved `0`, the
   method comes from the `0`'s source row.
8. **Idempotence** — running the step twice produces no further changes.

### Behaviour we explicitly do not test
SQLite transaction/CHECK semantics, `pandas`/stdlib internals, and the existing
Pass 0/1/2 behaviour (covered by their own tests).

### Real-data validation / measurement
Maxim is a useful diagnostic, **not** ground truth — reproducing it exactly is
not the goal. Run the step standalone against
`uc_nzgd_v0p7p0_20260528_deduped_NO_FILL_WITH_MAXIM_VALUES.db` (a working copy)
and **categorise every difference** vs the Maxim-filled `..._deduped.db`, per
field:

- **Match** — we produce Maxim's in-range value (expected: predrill ~2,794, tip
  293, GWL>0 ~1,737).
- **Preserved `0`** — both we and Maxim record a literal `0` (match).
- **Intended improvement** — Maxim `0`, we record a positive from a sibling
  (~5% of GWL `0`s).
- **Intended difference** — Maxim `0` (from a blank/`Nil` source), we record
  `NULL`. Expected and desired; not a regression.
- **Genuine residual gap** — Maxim has an in-range value we lack (NULL/absent).
  These are the only differences that warrant investigation: each is either a
  real extraction gap (a value only in a sheet/format we don't read) or a bug.
  Expected small (~49 predrill `Nil` + a handful); enumerate and explain each.
- **Conflict** — multiple distinct in-range values; left unchanged + reported.

Success criterion: every difference falls into an intended/expected bucket, and
the genuine-residual-gap bucket is fully enumerated and explained. The size of
that bucket informs the Maxim-retirement decision (and any further extractor
work).

## Applying to the current database

The step is idempotent and runs standalone, so the current deduped DB is brought
into line by running consolidation on the NO-FILL deduped DB (our own
extraction): this fills canonical rows, overrides any `0` that has a positive
sibling, preserves lone `0`s, and yields NULL where only `Nil` was present —
superseding Maxim's values for these fields. The predrill-`Nil` tail (extract
change) only lands on the next full extraction run. Whether to regenerate from
scratch vs apply the standalone correction is a small operational choice
deferred to planning, alongside the Maxim-retirement decision.

## Deferred / out of scope
- Merging u2-divergent format-siblings in the fuzzy matcher (record-3 cluster
  split). Consolidation handles the symptom; the deeper trace-dedup change is
  riskier and unnecessary here.
- SPT supplemental consolidation (function is table-generic; enable later).
- Retiring the Maxim backfill (pending residual measurement).
