# CPT ground-water level placeholder values (`0` and `-30`)

*Investigation date: 2026-06-05*

This note records the evidence behind how the consolidation pipeline treats two
placeholder conventions for ground-water level (GWL): a literal `0` (preserved
as written) and a `-30` "no-water" sentinel (filtered out). The DB is primarily
a faithful consolidation of what the source files state; it does not judge a
value's validity, **except** to prefer a more-valid alternative when one exists
for the same record.

## Part 1 — a GWL of `0` is (almost always) a placeholder, but we preserve it

While investigating missing CPT supplemental values, we found **~3,845 records
had GWL backfilled as `0.0` m**. The evidence below shows an extracted GWL of `0`
is overwhelmingly a **placeholder / unfilled-template default**, not a genuine
shallow water table — but, per the consolidation principle, we still record it
faithfully (rather than scrub it) and only override it with a more-valid value.

**Decision.**
- prefer an in-range GWL if any source for that `nzgd_id` records one (the
  statistical evidence below is the basis for that preference);
- otherwise **preserve the literal `0`** the source wrote — it is unambiguously
  what is recorded;
- yield `NULL` only when no GWL value was extracted at all (`Nil`/blank/absent —
  `Nil` is not numeric, so it is never extracted).

The range `field_plausibility_ranges.cptreport.extracted_gwl_m = [0.01, 50]` is
used only to **rank** competing values (in-range beats `0`), never to null a lone
value.

### 1. The value distribution has a placeholder signature

Distribution of all extracted GWL values (filtered output, n = 36,032):

| GWL range (m) | count |
|---|---|
| < 0 | 0 |
| **== 0.0** | **7,860** |
| (0, 0.1) | 45 |
| [0.1, 0.2) | 221 |
| [0.2, 0.5) | 1,059 |
| [0.5, 1) | 3,466 |
| [1, 2) | 10,840 |
| [2, 5) | 10,278 |
| ≥ 5 | 2,263 |

A spike of **7,860 values at exactly `0.0`** (~22%), with a **near-empty band
immediately above it** (45 in (0, 0.1), 221 in [0.1, 0.2)), then a smooth,
physically-plausible hump from ~0.5 m. A genuine shallow-water-table population
would fill the 0.05–0.4 m bins; a sharp spike at exactly `0` with a gap above it
is the classic signature of a default/placeholder.

### 2. The source files write `0` as a template default

Sample of 400 of the 3,845 `GWL == 0.0` records, by how each writes GWL:

| What the source shows | records | share |
|---|---|---|
| literal `0` / `0.00` (almost always labelled `Water level`) | 368 | 92% |
| **a positive GWL in another sibling file** | 21 | 5% |
| literal `Nil` | 11 | 3% |

The dominant pattern is `Water level: 0` — the default of an unfilled template
field. The 5% that carry a *positive* GWL in a sibling prove `0` is the wrong
value for those records (and consolidation will prefer the positive there).

### 3. No reliable signal identifies a *genuine* measured 0

We checked whether the extraction method (`gwl_method`) could carve out
genuinely-measured zeros. It cannot. Of the 7,860 `GWL == 0` values: `(none)`
7,158 (91%), `assumed` 677 (8.6%), `collapsed` 21, `derived` 2, **`measured` 2**,
`dipper` 0. By contrast `GWL > 0` carries real-observation methods far more often
(dipper 1,339, measured 480, derived 429). The two `measured` zeros are
`Measured GWL: 0` template fields beside `Pre-Drill: 0` — still the unfilled
pattern; the `measured` flag only reflects label wording. There is no reliable
way to distinguish a genuine measured `0` from a default — which is why we do not
try to judge individual zeros, and instead record every literal `0` uniformly as
written (preferring a more-valid sibling value when one exists).

## Part 2 — the `-30` "no-water" sentinel (filtered out)

A placeholder *family* appears in the RAW01.txt template, which writes
`Waterlevel: -30.00` / `-60.00` / `-100.00` (beside `Groundlevel: 0.00`) when no
water level was recorded. The supplemental filter applies `np.abs()` to extracted
values — correct for the common "below ground = negative" sign convention (e.g.
`-1.2` → `1.2`) — but this turns these sentinels into fabricated `30`/`60`/`100`.
The `-30` case is the most harmful: `30` lands *inside* `[0.01, 50]` and
masquerades as a real deep water table; `-60`/`-100` → `60`/`100` are already out
of range, but would still be wrongly *recorded* for records whose only GWL is the
sentinel.

Evidence these are sentinels, not real deep readings:

- **They spike at exactly `-30`/`-60`/`-100`:** 291 / 113 / 86 records carry a
  GWL candidate of `-30` / `-60` / `-100` respectively — the dominant
  large-magnitude negatives (the next most common negatives are the genuine
  `-1.x`/`-2.x` below-ground values). After `np.abs()` the `-30` becomes the
  `30.0` spike (298) in the distribution, with essentially nothing between 5 m
  and 30 m.
- **It is contradicted by the record's own data:** the same records hold a real,
  shallow GWL in a sibling (e.g. records 25/27: `Assumed GWL: 2` / `3.4` in the
  `.xls`, but `Waterlevel: -30.00` in the `.txt`). A water table cannot be both
  ~2–3 m and 30 m.
- **30 m is implausibly deep** for these (predominantly Christchurch) sites whose
  real GWLs cluster at 0.5–2.5 m.

**Decision:** filter the negative GWL sentinels (`-30`, `-60`, `-100`) to *no
value* (`NULL`) at the **filter** stage, **before** `np.abs()` runs (a
configurable sentinel list). This differs from the `0` treatment in Part 1: a
literal `0` is a value the source actually wrote and is preserved, whereas
`-30 → 30` (etc.) is a **transformation artifact** — recording it would assert a
value the source never meant. Because the negatives are already captured as
candidates, this is a **filter-only** change: re-running the filter on the
existing candidate CSV regenerates corrected values without re-extraction. It
removes the spurious `30`/`60`/`100`s and most GWL false-conflicts. (A handful of
*literal* out-of-range values — e.g. 13 literal `100`s — are left as written, per
the preserve-what-the-source-states principle; being out of range they are never
preferred over a real value.)

## Reproduce

- Script (read-only): `supplemental_value_analysis/gwl_zero_placeholder_analysis.py`
  in the extraction output directory.
- Inputs: `extracted_supplemental_values/cpt_supplemental_values_v0p7p0_20260528.csv`,
  `extracted_supplemental_values/all_potential_cpt_supplemental_values_v0p7p0_20260528.csv`
  (raw candidates, retains the `-30` sign), `supplemental_value_analysis/missing_values_breakdown.csv`,
  and source files under `downloads/nzgd_source_files/<id>/`.

*Caveat:* the source-file categorisation reads only the metadata block (first 30
rows of each spreadsheet sheet, first 40 lines of csv/txt); `.ags` files are not
parsed for GWL in this sample because the readable spreadsheet siblings carry the
same metadata.
