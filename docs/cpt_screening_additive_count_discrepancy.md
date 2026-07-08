# Why the researcher's CPT screening returns *fewer* records on newer DB versions

*Investigation date: 2026-07-07*

## Question

A researcher screens the CPT database with a fixed script
(`nzgd/scripts/temp/CPT_DB_screening_V26.py`) to select traces suitable for their
work. They expect database releases to be **additive** (each newer release should
yield ≥ the passing-record count of an older one), but observed **fewer** passing
records on a newer release. They also remarked that the discrepancy might be
eliminated by *filtering CPTs with constant trace columns differently*. This note
investigates how such a discrepancy arises and whether the latest
dedup/consolidation pipeline is immune.

Comparison anchored (per the researcher): baseline
`uc_nzgd_v0p6p0_20260403.db` (raw) → newer
`uc_nzgd_v0p8p1_20260625_deduped.db.backup-2026-07-05` (a *pre-quality-filter*
deduped build). We also test the **current** deduped build
`uc_nzgd_v0p8p1_20260625.db` (post coverage + quality-filter improvements).

## Executive summary

1. **The "additive" premise is false by construction.** Each release is a *fresh
   re-extraction*, not a superset (raw CPT report counts even fell 153,514 →
   144,742 from v0p4p3 → v0p6p0). More fundamentally, the dedup pipeline's *job*
   is to **reduce** the record count — it merges genuine duplicates and discards
   degenerate (constant-column) reports. A naive unique-record count therefore
   *must* drop after dedup; that is correct behaviour, not loss.

2. **The drop the researcher saw is ~93% expected de-duplication.** Reproducing
   their screening, v0p6p0 → backup loses 426 passes, of which **397 (93%) are
   genuine duplicate CPTs** (one physical sounding uploaded under several NZGD
   IDs) correctly collapsed into a canonical that still passes. No data is lost —
   only the ID count falls. The remaining 29 are a known, since-fixed bug (below).

3. **The *current* pipeline reverses the discrepancy into a gain.** v0p6p0 →
   current is **+353 passes** (21,826 → 22,179). The known within-record
   over-collapse bug is fixed (records 8809, 186131, 186134 restored from
   0.14/3.85/3.70 m back to 22.39/15.51/15.51 m), and **≈760 records are newly
   gained** because the new quality filter removes constant-u2 "decoy" reports
   that were winning the researcher's max-depth selection and masking a good
   report.

4. **The constant-u2 channel is the lever the researcher intuited.** 99.85 % of
   the DB's constant columns are u2 (pore pressure), and *every* constant-column
   screening failure in v0p6p0 is u2-only. The discrepancy's **direction depends
   on whether the researcher's constant-column rule agrees with the pipeline's**:
   when both reject constant-u2 (the script's current behaviour) the newer DB has
   *more* passes; when the researcher instead *keeps* constant-u2 the newer DB
   appears to lose 311 records that the pipeline deliberately discarded.

5. **No real data loss remains.** Of the 11 residual QC flips in current, **zero
   lose depth coverage** — every one keeps its full depth extent; they fail only
   because consolidation left a duplicate-depth trace (7) or the canonical trips a
   gate (4). The latest pipeline is therefore "immune" in the sense that matters
   (no coverage loss; discrepancy reversed), though a *naive ID count* will always
   legitimately shrink under de-duplication.

## The reproduced screening trend

The screening was reproduced faithfully against each DB's traces (validated:
reproduced max-depth matches stored `cptreport.max_depth_m` to a median |Δ| of
0.0000 m; Filter-2 verdict agrees on 100 % of records). Three constant-column
policies are scored: **current** = reject if qc, fs, *or* u2 constant (the
script's Filter 4); **no-u2** = reject only on qc/fs; **none** = no constant
filter.

| DB | groups | pass (qc/fs/u2) | pass (no-u2) | pass (none) |
|---|---|---|---|---|
| v0p6p0 (raw, baseline) | 49,769 | 21,826 | 22,908 | 22,908 |
| v0p8p1 (raw) | 49,773 | 21,826 | 22,908 | 22,908 |
| backup (deduped, **no** quality filter) | 48,858 | **21,400** | 22,430 | 22,430 |
| **current (deduped, post-fix)** | 47,231 | **22,179** | 22,179 | 22,179 |

Two features drive everything:

- **v0p6p0 → backup falls (21,826 → 21,400).** This is the researcher's observed
  discrepancy.
- **backup → current rises (21,400 → 22,179)** and all three policies **converge**
  in current — because the quality filter has already removed every constant-u2
  report, so the researcher's Filter 4 has nothing left to reject.

## How the discrepancy arises (three mechanisms)

**(a) Re-extraction — not the cause here.** v0p6p0 → v0p8p1 (raw→raw) changed the
pass count by **0** (21,826 → 21,826) and dropped 0 has-data records. Re-extraction
*can* move records across thresholds in general, but in this version pair it did
not.

**(b) De-duplication (intended count reduction).** The dedup pipeline merges
cross-record duplicates. In the researcher's metric (one representative per NZGD
group), two passing duplicates become one passing canonical — the count drops by
one with no data loss. This is 397 of the 426 lost passes in v0p6p0 → backup.

**(c) Constant-column (u2) handling.** The current pipeline's quality filter
discards any report with a constant depth/qc/fs/**u2** column (≥3 equal non-null
values). Census of the current build: **5,222 reports rejected, 5,214 (99.85 %)
for constant u2**, 8 for fs, 1 for qc — touching 4,246 NZGD IDs (955 whole-record
removals + 716 that lost their only trace report). This *aligns the DB with the
researcher's own Filter 4*, and additionally removes "decoy" reports (see below).

## Why the current build gains records (the shadowing effect)

The researcher's script picks, per NZGD group, the **deepest** report as
representative, then applies QC. When a group contains a deep report whose u2 is a
flat placeholder *and* a good report, the deep placeholder wins the selection and
the group fails on constant-u2 — the good report is never seen.

The current build's quality filter deletes those flat-u2 reports *before*
selection, so the good report becomes the representative and passes. Empirically,
of the ~760 records newly passing in current, **761 failed in v0p6p0 with reason
`constant:u`** — direct confirmation of the shadowing mechanism.

## Decomposition of each comparison (merge-aware)

Merges are resolved via `dedup_audit`, so a record merged into a *passing*
canonical is not counted as lost.

**v0p6p0 → backup, current policy (net −426):**

| Category | Count | Nature |
|---|---|---|
| Duplicate collapse (into passing canonical) | 397 | Expected — no data loss |
| Merged into a failing canonical | 12 | Old-pipeline QC flip |
| Genuine QC flip | 17 | Incl. 8809/186131/186134 (over-collapse **bug**) |
| Gains | 0 | — |

**v0p6p0 → current, current policy (net +353):**

| Category | Count | Nature |
|---|---|---|
| Gains (constant-u2 decoy removed) | ≈760 | 761 were `constant:u` fails in v0p6p0 |
| Duplicate collapse (into passing canonical) | 409 | Expected — no data loss |
| Genuine QC flip | 11 | **Zero lose coverage** (see below) |

**v0p6p0 → current, no-u2 policy (net −729):** here the researcher *keeps*
constant-u2 records, so the 311 u2 reports the DB deleted now count as "lost"
(plus 419 duplicate collapses, 11 flips, −12 gains). This is the exact scenario in
which "filtering constant-column CPTs differently" changes the answer: **keeping
the u2 rejection (as V26 does) eliminates the discrepancy; dropping it re-creates
it.**

## The 11 residual QC flips are not data loss

Every one of the 11 v0p6p0→current genuine flips keeps its **full depth extent** in
current. Seven fail on **duplicate depths**; four merged into a canonical that
trips a gate. Root cause of the duplicate-depth cases: the group contains a clean
report and a sibling with a few stacked rows (two measurements at one depth) that
reach the *same* max depth. The researcher's screening tie-breaks to the clean
report; the dedup within-record consolidation kept the duplicate-containing report
as canonical and removed the clean sibling. Example — nzgd 11853: raw reports
cpt_id 30721 (2000 rows, 0 duplicates) and 30725 (2004 rows, 4 duplicates), both
to 20.0 m; current keeps 30725 → duplicate-depths → fail. Coverage is preserved;
only trace *quality* differs.

This same-depth-row phenomenon is not specific to these 7 records, nor is it a
dedup regression: it is a pre-existing **extraction** artifact (a report stores
*conflicting* measurements at one depth — e.g. nzgd 11853 has qc 22.76 and 23.99
at 10.34 m), and it is stable across versions — Filter 3 rejects 183 such
representatives in v0p6p0 and 177 in current (dedup slightly *reduced* it). No
depth coverage is lost in any of them. Because the repeated rows carry conflicting
values (not exact duplicates), collapsing them is a lossy domain choice best made
by the consumer: dropping or averaging same-depth rows in the screening (a one-line
change before Filter 3) recovers these ~177 records. A pipeline change was
considered and **deliberately declined** — it would modify measurement data for all
users to satisfy one downstream filter while recovering no lost data. If those
records ever matter, the correct fix is upstream in extraction, not in dedup.

## Is the latest pipeline immune?

- **To real data / coverage loss: yes.** No dedup pass reduces depth coverage
  (independently proven), re-extraction dropped 0 records, and all residual flips
  preserve their full extent. The over-collapse bug that caused the backup's worst
  losses is fixed.
- **To a falling *naive record count*: no — and it should not be.** De-duplication
  legitimately reduces the number of distinct NZGD IDs by collapsing true
  duplicates. Any screening that counts unique representatives will see fewer after
  dedup even though no usable CPT was lost.

In short: the current pipeline resolves the researcher's specific discrepancy
(v0p6p0 → current is **+353**, not a loss), and the only way it "loses" records is
a policy mismatch on constant-u2 or the correct collapsing of duplicates.

## Recommendations for the researcher

1. **Count physical CPTs, not raw IDs.** When comparing versions, resolve merges
   through `dedup_audit` (a merged record's data lives under its canonical).
   Comparing a raw export to a deduped export conflates duplicate-removal with loss.
2. **Keep the constant-u2 rejection** (V26 already does) — it matches the pipeline's
   quality filter and makes the count stable/increasing. If instead the goal is to
   *retain* no-pore-pressure soundings (a constant/absent u2 is physically
   legitimate), note the deduped DB has already removed them; screen the raw DB or
   ask the pipeline to retain constant-u2 reports.
3. **Screen the deduped DB, not multi-report raw exports.** It removes the
   representative-selection nondeterminism and the constant-u2 shadowing that make
   raw-export counts unstable across versions.
4. **Duplicate-depth rejections are yours to tune, not a pipeline bug.** Filter 3
   rejects ~177 otherwise-usable traces that carry a few same-depth rows (an
   extraction artifact, not dedup; no coverage lost). Drop or average same-depth
   rows in the screening to recover them; a pipeline change was assessed and
   declined as low-value.

## Method and caveats

- **Exporter not available.** The tool that writes the researcher's
  `EstimatedVs_*/CPT_*_result.csv` files is not on this machine, so the screening
  was reproduced in *essence*: the V26 gate chain applied directly to each report's
  `depth_m/qc_MPa/fs_MPa/u2_MPa` from `cptmeasurements`. Assumptions: one CSV per
  `has_cpt_data` report, raw trace (no pre-filter), group = NZGD ID, tie-break =
  lowest id. Filter 5 (sampling ratio) rejects 0–1 records and is immaterial.
- **Tie-break sensitivity.** For groups whose deepest reports tie on max depth
  (e.g. 11853), the v0p6p0 "pass" depends on the tie-break landing on the clean
  report; the researcher's real suffix order may differ for a handful of records.
- **Reproduction validated** against stored extents (max |Δ| = 0.01 m; 100 %
  Filter-2 agreement) and internally (fail reasons sum exactly to group counts).

## Reproduce

Scripts (scratchpad): `qc_repro.py` (per-DB verdicts via one indexed `GROUP BY
cpt_id` aggregate scan + Filter-5 median stage), `qc_compare.py` (merge-aware
attribution), `forensic.py` (loss characterisation + shadowing confirmation),
`structural_accounting.py` (record-level v0p6p0→deduped decomposition). Per-report
constant-column census reads `quality_reject.constant_columns_json` in the current
deduped DB.
