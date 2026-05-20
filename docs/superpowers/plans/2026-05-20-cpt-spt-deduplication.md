# CPT/SPT Cross-Record Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone Python script that reads a post-extraction NZGD SQLite DB and produces a deduplicated copy in which records uploaded under multiple NZGD IDs are merged into a single canonical record, with full audit trail.

**Architecture:** New `nzgd/dedup/` module providing two passes (hash-based exact match, then metadata-blocked fuzzy match) and a shared merge executor. CLI entry point at `nzgd/scripts/db/deduplicate.py`. The source DB is read-only; mutation happens on a copy.

**Tech Stack:** Python 3.9+, SQLite (stdlib `sqlite3`), `hashlib.blake2b` + `struct` for fingerprinting, `scipy.sparse.csgraph.connected_components` for clustering, `sklearn.neighbors.BallTree` for spatial blocking, `rapidfuzz` for name similarity, `typer` for CLI, `pytest` for tests. Reference spec: `docs/superpowers/specs/2026-05-20-cpt-spt-deduplication-design.md`.

**Environment:** Python at `/home/arr65/venvs/dev_nzgd_venv/bin/python`. Source DB at `/home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p6p0_20260403.db`. Run all commands relative to repo root `/home/arr65/src/nzgd`.

---

## File Structure

**New files (under `nzgd/dedup/`):**
- `__init__.py` — empty package marker
- `data_types.py` — `MergePlanEntry`, `ReportPairMatch` dataclasses + table-config record
- `schema.py` — DDL: adds `merged_into_nzgd_id` column to `nzgdrecord`, creates `dedup_run` and `dedup_audit` tables
- `fingerprint.py` — `compute_trace_hash(rows)` deterministic per-trace blake2b digest
- `cluster.py` — `connected_components_from_edges(edges)` thin scipy wrapper turning an edge list of nzgd_id pairs into cluster IDs
- `selection.py` — `select_canonical(...)` picks canonical from a cluster per the spec's rule
- `pass1_hash.py` — `generate_hash_merge_plan(conn, table_cfg)` produces `list[MergePlanEntry]`
- `pass2_fuzzy.py` — `generate_fuzzy_merge_plan(conn, table_cfg)` produces `list[MergePlanEntry]`
- `executor.py` — `apply_merge_plan(conn, plan, run_id, table_cfg)` applies a plan, writing audit rows
- `reports.py` — `write_dedup_report(conn, path)`, `write_calibration_report(...)`, `write_failures_report(...)`

**New CLI script:**
- `nzgd/scripts/db/deduplicate.py` — typer command: `--source`, `--target`, `--skip-cpt`, `--skip-spt`

**New tests:**
- `tests/dedup/__init__.py` — empty
- `tests/dedup/conftest.py` — synthetic SQLite DB fixtures
- `tests/dedup/test_dedup_pipeline.py` — end-to-end integration scenarios

**Modifications:**
- `requirements.txt` — add `rapidfuzz`, `scipy`, `scikit-learn`
- `nzgd/resources/config.yaml` — add `deduplication` section
- `nzgd/constants.py` — load dedup config

`scipy` and `scikit-learn` are present in the dev venv but missing from `requirements.txt`; adding them makes the dep set match the venv.

`pass1_hash.py` and `pass2_fuzzy.py` both produce `list[MergePlanEntry]` so the executor consumes either uniformly.

`table_cfg` is a small object naming the per-record-type tables (CPT or SPT) so each module can be called twice — once for CPT, once for SPT — without branching on record type internally.

---

## Task 1: Scaffolding

**Files:**
- Create: `nzgd/dedup/__init__.py`
- Create: `tests/dedup/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Create directories and empty package files**

```bash
mkdir -p nzgd/dedup tests/dedup
touch nzgd/dedup/__init__.py tests/dedup/__init__.py
```

- [ ] **Step 2: Add new dependencies to requirements.txt**

Append the following lines to `requirements.txt`:

```
rapidfuzz
scipy
scikit-learn
```

- [ ] **Step 3: Install rapidfuzz into the dev venv**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/pip install rapidfuzz
```

Expected: success, version printed.

- [ ] **Step 4: Verify imports work**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "import rapidfuzz, scipy.sparse.csgraph, sklearn.neighbors; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add nzgd/dedup/ tests/dedup/ requirements.txt
git commit -m "Scaffold dedup module + add rapidfuzz/scipy/sklearn deps"
```

---

## Task 2: Config and constants

**Files:**
- Modify: `nzgd/resources/config.yaml` (append section)
- Modify: `nzgd/constants.py` (append loads)

- [ ] **Step 1: Append `deduplication` section to config.yaml**

Append to `nzgd/resources/config.yaml`:

```yaml

deduplication:
  hash_pass:
    stream_chunk_size: 100000

  fuzzy_pass:
    spatial_radius_m: 50
    date_window_days: 90
    name_similarity_min: 80
    trace_score_max: 0.05
    trace_resample_step_m: 0.05

  calibration:
    random_pair_sample_size: 5000

  output:
    deduped_db_suffix: "_deduped"
    report_filename: "dedup_report.csv"
    calibration_report_filename: "calibration_report.csv"
    failures_filename: "dedup_failures.csv"
```

- [ ] **Step 2: Append dedup-config loading to constants.py**

Append to `nzgd/constants.py`:

```python

# Deduplication configuration (see docs/superpowers/specs/2026-05-20-cpt-spt-deduplication-design.md)
DEDUP_CONFIG = CONFIG["deduplication"]
```

- [ ] **Step 3: Verify it loads**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "from nzgd import constants; print(constants.DEDUP_CONFIG['fuzzy_pass']['spatial_radius_m'])"
```

Expected: `50`.

- [ ] **Step 4: Commit**

```bash
git add nzgd/resources/config.yaml nzgd/constants.py
git commit -m "Add dedup config block and constants loader"
```

---

## Task 3: Data types

**Files:**
- Create: `nzgd/dedup/data_types.py`

- [ ] **Step 1: Write `data_types.py`**

Create `nzgd/dedup/data_types.py` with:

```python
"""Data types shared across dedup passes and the executor."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TableConfig:
    """Per-record-type table names so dedup code is parameterised over CPT vs SPT."""

    record_type: str  # 'CPT' or 'BH'
    report_table: str  # 'cptreport' or 'sptreport'
    measurement_table: str  # 'cptmeasurements' or 'sptmeasurements'
    report_id_column: str  # 'cpt_id' or 'spt_id'
    measurement_value_columns: tuple[str, ...]
    # Tables that must be deleted in this order before deleting the report row
    # itself. Each entry is (table_name, fk_column). Innermost dependents first.
    dependent_tables: tuple[tuple[str, str], ...]


CPT_TABLE_CONFIG = TableConfig(
    record_type="CPT",
    report_table="cptreport",
    measurement_table="cptmeasurements",
    report_id_column="cpt_id",
    measurement_value_columns=("depth_m", "qc_MPa", "fs_MPa", "u2_MPa"),
    dependent_tables=(
        ("cptvs30estimates", "cpt_id"),
        ("cptmeasurements", "cpt_id"),
    ),
)

SPT_TABLE_CONFIG = TableConfig(
    record_type="BH",
    report_table="sptreport",
    measurement_table="sptmeasurements",
    report_id_column="spt_id",
    measurement_value_columns=("depth_m", "ISPT_MAIN", "ISPT_NVAL", "ISPT_REP"),
    dependent_tables=(
        # soilmeasurementsoiltype joins by soil_measurement_id; the executor
        # handles its cascade explicitly via a subquery rather than a flat list.
        ("soilmeasurements", "spt_id"),
        ("densitymeasurements", "spt_id"),
        ("sptvs30estimates", "spt_id"),
        ("sptmeasurements", "spt_id"),
    ),
)


@dataclass(frozen=True)
class ReportPairMatch:
    """One (canonical_report_id, merged_report_id) pair identified by a pass."""

    canonical_report_id: int
    merged_report_id: int
    # Free-form metrics: {"hash": "<hex>"} for hash matches; full feature vector for fuzzy.
    metrics: dict[str, Any]


@dataclass(frozen=True)
class MergePlanEntry:
    """One (canonical, merged) pair within a cluster. A 3-way cluster produces 2 entries."""

    cluster_id: int
    canonical_nzgd_id: int
    merged_nzgd_id: int
    record_type: str
    match_pass: str  # 'hash' or 'fuzzy'
    matched_pairs: list[ReportPairMatch]  # reports to delete from merged record
    unique_merged_report_ids: list[int] = field(default_factory=list)  # reports to re-parent
```

- [ ] **Step 2: Verify it imports**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "from nzgd.dedup.data_types import MergePlanEntry, ReportPairMatch, CPT_TABLE_CONFIG, SPT_TABLE_CONFIG; print(CPT_TABLE_CONFIG.report_id_column, SPT_TABLE_CONFIG.report_id_column)"
```

Expected: `cpt_id spt_id`.

- [ ] **Step 3: Commit**

```bash
git add nzgd/dedup/data_types.py
git commit -m "Add dedup data types (MergePlanEntry, TableConfig)"
```

---

## Task 4: Schema migration module

**Files:**
- Create: `nzgd/dedup/schema.py`

- [ ] **Step 1: Write `schema.py`**

Create `nzgd/dedup/schema.py` with:

```python
"""DDL for dedup-specific schema additions on the target (deduped) DB."""

import sqlite3


_ADD_MERGED_INTO_COLUMN = (
    "ALTER TABLE nzgdrecord "
    "ADD COLUMN merged_into_nzgd_id INTEGER REFERENCES nzgdrecord(nzgd_id)"
)

_INDEX_MERGED_INTO = (
    "CREATE INDEX IF NOT EXISTS idx_nzgdrecord_merged_into "
    "ON nzgdrecord(merged_into_nzgd_id)"
)

_CREATE_DEDUP_RUN = """
CREATE TABLE IF NOT EXISTS dedup_run (
    run_id               INTEGER PRIMARY KEY,
    started_at           TEXT NOT NULL,
    finished_at          TEXT,
    source_db_path       TEXT NOT NULL,
    script_version       TEXT NOT NULL,
    config_snapshot_json TEXT NOT NULL,
    n_clusters_merged    INTEGER,
    n_records_merged     INTEGER
)
"""

_CREATE_DEDUP_AUDIT = """
CREATE TABLE IF NOT EXISTS dedup_audit (
    audit_id                INTEGER PRIMARY KEY,
    run_id                  INTEGER NOT NULL REFERENCES dedup_run(run_id),
    cluster_id              INTEGER NOT NULL,
    canonical_nzgd_id       INTEGER NOT NULL,
    merged_nzgd_id          INTEGER NOT NULL,
    record_type             TEXT NOT NULL CHECK(record_type IN ('CPT', 'BH')),
    match_pass              TEXT NOT NULL CHECK(match_pass IN ('hash', 'fuzzy')),
    report_pairs_json       TEXT NOT NULL,
    metadata_copied_json    TEXT,
    metadata_conflicts_json TEXT,
    merged_at               TEXT NOT NULL
)
"""

_INDEX_AUDIT_CANONICAL = "CREATE INDEX IF NOT EXISTS idx_dedup_audit_canonical ON dedup_audit(canonical_nzgd_id)"
_INDEX_AUDIT_MERGED    = "CREATE INDEX IF NOT EXISTS idx_dedup_audit_merged    ON dedup_audit(merged_nzgd_id)"
_INDEX_AUDIT_CLUSTER   = "CREATE INDEX IF NOT EXISTS idx_dedup_audit_cluster   ON dedup_audit(cluster_id)"


def apply_dedup_schema(conn: sqlite3.Connection) -> None:
    """Apply dedup-specific schema additions to a deduped target DB.

    Adds `nzgdrecord.merged_into_nzgd_id`, creates `dedup_run` and
    `dedup_audit` tables, and creates supporting indexes. Idempotent: if a
    second invocation runs against an already-migrated DB, the ALTER TABLE
    will fail with "duplicate column"; the function catches that case and
    proceeds. All other DDL is `IF NOT EXISTS`.

    Foreign-key enforcement requires `PRAGMA foreign_keys = ON` on the
    connection; callers should issue that before invoking this function.
    """
    cur = conn.cursor()
    try:
        cur.execute(_ADD_MERGED_INTO_COLUMN)
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise
    cur.execute(_INDEX_MERGED_INTO)
    cur.execute(_CREATE_DEDUP_RUN)
    cur.execute(_CREATE_DEDUP_AUDIT)
    cur.execute(_INDEX_AUDIT_CANONICAL)
    cur.execute(_INDEX_AUDIT_MERGED)
    cur.execute(_INDEX_AUDIT_CLUSTER)
    conn.commit()
```

- [ ] **Step 2: Smoke check on an in-memory DB**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
import sqlite3
from nzgd.dedup.schema import apply_dedup_schema
conn = sqlite3.connect(':memory:')
conn.execute('PRAGMA foreign_keys = ON')
conn.execute('CREATE TABLE nzgdrecord (nzgd_id INTEGER PRIMARY KEY)')
apply_dedup_schema(conn)
cur = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name\")
print([r[0] for r in cur.fetchall()])
"
```

Expected: includes `dedup_run` and `dedup_audit`.

- [ ] **Step 3: Commit**

```bash
git add nzgd/dedup/schema.py
git commit -m "Add dedup schema-migration module"
```

---

## Task 5: Fingerprint module

**Files:**
- Create: `nzgd/dedup/fingerprint.py`

- [ ] **Step 1: Write `fingerprint.py`**

Create `nzgd/dedup/fingerprint.py` with:

```python
"""Per-trace bit-deterministic fingerprint for the dedup hash pass."""

import hashlib
import math
import struct
from typing import Iterable, Sequence

# Fixed 8-byte sentinel for NULL/NaN values. Distinct from any IEEE 754 double
# representation by construction (the leading 4 bytes are zero, the trailing
# 4 are ASCII 'NaN_').
_NULL_SENTINEL = b"\x00\x00\x00\x00NaN_"


def _encode_value(v: float | None) -> bytes:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return _NULL_SENTINEL
    return struct.pack("<d", float(v))


def compute_trace_hash(rows: Iterable[Sequence[float | None]]) -> bytes:
    """Compute a 16-byte blake2b digest of a sorted measurement trace.

    `rows` must already be sorted by depth (ascending). Each row is a tuple of
    floats (or `None`/`NaN`) in a fixed column order; the caller chooses the
    order (e.g. depth, qc, fs, u2 for CPT). NaN and NULL are both mapped to a
    fixed 8-byte sentinel so they hash identically. Finite floats are packed
    as little-endian IEEE 754 doubles; no rounding is applied.

    Two traces producing the same digest are byte-identical after this
    normalisation, which is the strongest possible "same data" claim.
    """
    h = hashlib.blake2b(digest_size=16)
    for row in rows:
        for v in row:
            h.update(_encode_value(v))
    return h.digest()
```

- [ ] **Step 2: Smoke check**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
from nzgd.dedup.fingerprint import compute_trace_hash
import math
a = [(0.0, 1.0, 0.01, 0.0), (0.1, 1.1, 0.011, 0.0)]
b = [(0.0, 1.0, 0.01, 0.0), (0.1, 1.1, 0.011, 0.0)]
c = [(0.0, 1.0, 0.01, 0.0), (0.1, 1.1, 0.012, 0.0)]  # one value differs
d = [(0.0, 1.0, 0.01, None), (0.1, 1.1, 0.011, math.nan)]  # NaN vs None
e = [(0.0, 1.0, 0.01, None), (0.1, 1.1, 0.011, None)]
print('equal a==b:', compute_trace_hash(a) == compute_trace_hash(b))
print('different a!=c:', compute_trace_hash(a) != compute_trace_hash(c))
print('NaN==None d==e:', compute_trace_hash(d) == compute_trace_hash(e))
"
```

Expected: all three lines print `True`.

- [ ] **Step 3: Commit**

```bash
git add nzgd/dedup/fingerprint.py
git commit -m "Add per-trace fingerprint module for dedup hash pass"
```

---

## Task 6: Cluster utility

**Files:**
- Create: `nzgd/dedup/cluster.py`

- [ ] **Step 1: Write `cluster.py`**

Create `nzgd/dedup/cluster.py` with:

```python
"""Edge-list → connected-components helper. Thin wrapper around scipy."""

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def connected_components_from_edges(
    edges: Iterable[tuple[int, int]],
) -> dict[int, int]:
    """Return a mapping `{node_id: cluster_label}` for each node appearing in any edge.

    Edges are undirected; duplicates and self-loops are ignored. Cluster labels are
    consecutive integers starting at 1. Nodes that do not appear in any edge are not
    in the returned mapping.
    """
    edge_list = [(a, b) for (a, b) in edges if a != b]
    if not edge_list:
        return {}
    nodes_sorted = sorted({n for pair in edge_list for n in pair})
    index_of = {n: i for i, n in enumerate(nodes_sorted)}
    rows = np.fromiter((index_of[a] for a, _ in edge_list), dtype=np.int64)
    cols = np.fromiter((index_of[b] for _, b in edge_list), dtype=np.int64)
    data = np.ones(len(edge_list), dtype=np.int8)
    n = len(nodes_sorted)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(graph, directed=False)
    # Relabel components to 1..n_components in order of first occurrence
    seen: dict[int, int] = {}
    next_id = 1
    out: dict[int, int] = {}
    for node, lbl in zip(nodes_sorted, labels):
        if lbl not in seen:
            seen[int(lbl)] = next_id
            next_id += 1
        out[node] = seen[int(lbl)]
    return out
```

- [ ] **Step 2: Smoke check**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
from nzgd.dedup.cluster import connected_components_from_edges
# {A,B,C} connected; {D,E} connected; F isolated (not in edges)
edges = [(1, 2), (2, 3), (4, 5)]
print(connected_components_from_edges(edges))
# transitive: A-B, A-D → {A,B,D} one cluster
print(connected_components_from_edges([(1, 2), (1, 4)]))
"
```

Expected: two outputs. First like `{1: 1, 2: 1, 3: 1, 4: 2, 5: 2}`. Second like `{1: 1, 2: 1, 4: 1}`.

- [ ] **Step 3: Commit**

```bash
git add nzgd/dedup/cluster.py
git commit -m "Add scipy-backed connected-components cluster utility"
```

---

## Task 7: Canonical selection

**Files:**
- Create: `nzgd/dedup/selection.py`

- [ ] **Step 1: Write `selection.py`**

Create `nzgd/dedup/selection.py` with:

```python
"""Canonical selection for a dedup cluster.

Picks the canonical nzgd_id per the spec rule:
  1. Highest count of measurement rows in reports with no matched-pair counterpart.
  2. Tiebreaker: most non-null nzgdrecord columns.
  3. Tiebreaker: smallest nzgd_id.
"""

import sqlite3
from typing import Iterable

from nzgd.dedup.data_types import TableConfig


_NZGDRECORD_METADATA_COLUMNS = (
    "type_id", "latitude", "longitude",
    "model_vs30_foster_2019_km_per_s", "model_vs30_stddev_foster_2019_km_per_s",
    "model_gwl_westerhoff_2018_m", "model_gwl_nlm_2025_m", "model_gwl_nlm_2025_stddev_m",
    "original_investigation_name", "investigation_date", "published_date",
    "region_id", "district_id", "city_id", "suburb_id",
)


def _matched_report_ids_for_nzgd(
    nzgd_id: int, matched_pairs: Iterable[tuple[int, int, int]]
) -> set[int]:
    """Return set of report ids in this nzgd_id that appear in any matched pair.

    matched_pairs is iterable of (nzgd_id_a, report_id_a, nzgd_id_b_or_report_id_b...)
    For our use, a pair is represented as (nzgd_a, report_a, nzgd_b, report_b).
    """
    out: set[int] = set()
    for nzgd_a, report_a, nzgd_b, report_b in matched_pairs:
        if nzgd_a == nzgd_id:
            out.add(report_a)
        if nzgd_b == nzgd_id:
            out.add(report_b)
    return out


def _unique_measurement_row_count(
    conn: sqlite3.Connection,
    nzgd_id: int,
    matched_report_ids: set[int],
    table_cfg: TableConfig,
) -> int:
    """Count measurement rows for `nzgd_id` in reports NOT in `matched_report_ids`."""
    cur = conn.cursor()
    if matched_report_ids:
        placeholders = ",".join("?" * len(matched_report_ids))
        query = (
            f"SELECT COUNT(*) FROM {table_cfg.measurement_table} m "
            f"JOIN {table_cfg.report_table} r ON r.{table_cfg.report_id_column} = m.{table_cfg.report_id_column} "
            f"WHERE r.nzgd_id = ? "
            f"AND r.{table_cfg.report_id_column} NOT IN ({placeholders})"
        )
        cur.execute(query, (nzgd_id, *matched_report_ids))
    else:
        query = (
            f"SELECT COUNT(*) FROM {table_cfg.measurement_table} m "
            f"JOIN {table_cfg.report_table} r ON r.{table_cfg.report_id_column} = m.{table_cfg.report_id_column} "
            f"WHERE r.nzgd_id = ?"
        )
        cur.execute(query, (nzgd_id,))
    return cur.fetchone()[0]


def _non_null_metadata_count(conn: sqlite3.Connection, nzgd_id: int) -> int:
    cur = conn.cursor()
    cols_sql = ", ".join(_NZGDRECORD_METADATA_COLUMNS)
    cur.execute(f"SELECT {cols_sql} FROM nzgdrecord WHERE nzgd_id = ?", (nzgd_id,))
    row = cur.fetchone()
    if row is None:
        return 0
    return sum(1 for v in row if v is not None)


def select_canonical(
    conn: sqlite3.Connection,
    cluster_nzgd_ids: Iterable[int],
    matched_pairs: Iterable[tuple[int, int, int, int]],
    table_cfg: TableConfig,
) -> int:
    """Pick the canonical nzgd_id from a cluster of nzgd_ids per the spec rule.

    Parameters
    ----------
    conn
        Open SQLite connection to the target DB.
    cluster_nzgd_ids
        nzgd_ids in the cluster.
    matched_pairs
        Iterable of (nzgd_id_a, report_id_a, nzgd_id_b, report_id_b) — the matched
        report pairs across nzgd_ids in this cluster (as identified by the pass).
    table_cfg
        Per-record-type table configuration.

    Returns
    -------
    int
        The selected canonical nzgd_id.
    """
    pairs = list(matched_pairs)
    nzgd_ids = list(cluster_nzgd_ids)
    scored = []
    for nz in nzgd_ids:
        matched_ids = _matched_report_ids_for_nzgd(nz, pairs)
        unique_rows = _unique_measurement_row_count(conn, nz, matched_ids, table_cfg)
        meta_count = _non_null_metadata_count(conn, nz)
        # Sort key: maximise unique_rows, then meta_count; minimise nzgd_id
        scored.append((-unique_rows, -meta_count, nz))
    scored.sort()
    return scored[0][2]
```

- [ ] **Step 2: Smoke check on in-memory DB**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
import sqlite3
from nzgd.dedup.data_types import CPT_TABLE_CONFIG
from nzgd.dedup.selection import select_canonical

conn = sqlite3.connect(':memory:')
conn.executescript('''
CREATE TABLE nzgdrecord (nzgd_id INTEGER PRIMARY KEY, latitude REAL, longitude REAL);
CREATE TABLE cptreport (cpt_id INTEGER PRIMARY KEY, nzgd_id INTEGER);
CREATE TABLE cptmeasurements (measurement_id INTEGER PRIMARY KEY, cpt_id INTEGER, depth_m REAL, qc_MPa REAL, fs_MPa REAL, u2_MPa REAL);
INSERT INTO nzgdrecord VALUES (1, -41.0, 174.0), (2, -41.0, 174.0);
INSERT INTO cptreport VALUES (10, 1), (11, 1), (20, 2);
-- record 1: 2 reports; report 10 is matched with report 20; report 11 unique with 3 rows
INSERT INTO cptmeasurements (cpt_id, depth_m, qc_MPa, fs_MPa, u2_MPa) VALUES
    (10, 0.1, 1.0, 0.01, 0.0), (10, 0.2, 1.1, 0.011, 0.0),
    (11, 0.1, 2.0, 0.02, 0.0), (11, 0.2, 2.1, 0.021, 0.0), (11, 0.3, 2.2, 0.022, 0.0),
    (20, 0.1, 1.0, 0.01, 0.0), (20, 0.2, 1.1, 0.011, 0.0);
''')
# matched: cpt_id 10 (nzgd 1) ↔ cpt_id 20 (nzgd 2)
canonical = select_canonical(conn, [1, 2], [(1, 10, 2, 20)], CPT_TABLE_CONFIG)
print('canonical:', canonical)
# Expected: 1, because nzgd 1 has 3 unique rows (cpt_id 11), nzgd 2 has 0.
"
```

Expected: `canonical: 1`.

- [ ] **Step 3: Commit**

```bash
git add nzgd/dedup/selection.py
git commit -m "Add canonical-selection function for dedup clusters"
```

---

## Task 8: Hash pass

**Files:**
- Create: `nzgd/dedup/pass1_hash.py`

- [ ] **Step 1: Write `pass1_hash.py`**

Create `nzgd/dedup/pass1_hash.py` with:

```python
"""Pass 1: bit-exact match via per-trace fingerprint hashing."""

import sqlite3
from collections import defaultdict
from itertools import combinations

from tqdm import tqdm

from nzgd.dedup.cluster import connected_components_from_edges
from nzgd.dedup.data_types import (
    MergePlanEntry,
    ReportPairMatch,
    TableConfig,
)
from nzgd.dedup.fingerprint import compute_trace_hash
from nzgd.dedup.selection import select_canonical


def _compute_report_hashes(
    conn: sqlite3.Connection, table_cfg: TableConfig
) -> dict[int, bytes]:
    """Compute one fingerprint per non-merged report.

    Returns a dict `{report_id: hash}`. Only reports whose nzgd_id has
    `merged_into_nzgd_id IS NULL` are processed (so a re-run after pass 1
    doesn't re-hash already-merged data). Streams the full measurement table
    in (report_id, depth) order and filters in Python — using an `IN (?, ?, ...)`
    clause is not viable with ~144k cpt_ids (exceeds SQLite's host-parameter limit).
    """
    report_id_col = table_cfg.report_id_column
    cur = conn.cursor()
    cur.execute(
        f"SELECT r.{report_id_col} "
        f"FROM {table_cfg.report_table} r "
        f"JOIN nzgdrecord n ON n.nzgd_id = r.nzgd_id "
        f"WHERE n.merged_into_nzgd_id IS NULL"
    )
    active_report_ids: set[int] = {r[0] for r in cur.fetchall()}
    if not active_report_ids:
        return {}

    select_cols = ", ".join(table_cfg.measurement_value_columns)
    cur.execute(
        f"SELECT {report_id_col}, {select_cols} "
        f"FROM {table_cfg.measurement_table} "
        f"ORDER BY {report_id_col}, depth_m"
    )

    hashes: dict[int, bytes] = {}
    current_id: int | None = None
    current_rows: list[tuple] = []
    for row in tqdm(cur, desc=f"hashing {table_cfg.measurement_table}"):
        rid = row[0]
        if rid not in active_report_ids:
            continue
        if rid != current_id:
            if current_id is not None:
                hashes[current_id] = compute_trace_hash(current_rows)
                current_rows = []
            current_id = rid
        current_rows.append(row[1:])
    if current_id is not None:
        hashes[current_id] = compute_trace_hash(current_rows)
    return hashes


def generate_hash_merge_plan(
    conn: sqlite3.Connection, table_cfg: TableConfig
) -> list[MergePlanEntry]:
    """Produce a list of MergePlanEntry for byte-identical traces across nzgd_ids."""
    hashes = _compute_report_hashes(conn, table_cfg)
    if not hashes:
        return []

    # Build report_id → nzgd_id lookup by scanning the full report table and
    # filtering in Python (an `IN (...)` clause would exceed the host-parameter limit).
    cur = conn.cursor()
    cur.execute(
        f"SELECT {table_cfg.report_id_column}, nzgd_id FROM {table_cfg.report_table}"
    )
    report_to_nzgd = {rid: nz for rid, nz in cur.fetchall() if rid in hashes}

    # Bucket reports by hash; drop singletons; drop buckets confined to one nzgd_id.
    by_hash: dict[bytes, list[int]] = defaultdict(list)
    for rid, h in hashes.items():
        by_hash[h].append(rid)
    cross_nzgd_buckets = []
    for h, rids in by_hash.items():
        if len(rids) < 2:
            continue
        nz_set = {report_to_nzgd[r] for r in rids}
        if len(nz_set) >= 2:
            cross_nzgd_buckets.append((h, rids))

    if not cross_nzgd_buckets:
        return []

    # Build edges between nzgd_ids that share a bucket.
    edges: list[tuple[int, int]] = []
    bucket_pair_meta: dict[tuple[int, int], list[ReportPairMatch]] = defaultdict(list)
    for h, rids in cross_nzgd_buckets:
        for a, b in combinations(rids, 2):
            nz_a, nz_b = report_to_nzgd[a], report_to_nzgd[b]
            if nz_a == nz_b:
                continue
            edges.append((nz_a, nz_b))
            key = (min(nz_a, nz_b), max(nz_a, nz_b))
            bucket_pair_meta[key].append(
                ReportPairMatch(
                    canonical_report_id=a if nz_a < nz_b else b,
                    merged_report_id=b if nz_a < nz_b else a,
                    metrics={"hash": h.hex()},
                )
            )

    nzgd_to_cluster = connected_components_from_edges(edges)
    clusters: dict[int, list[int]] = defaultdict(list)
    for nz, cl in nzgd_to_cluster.items():
        clusters[cl].append(nz)

    plan: list[MergePlanEntry] = []
    for cluster_id, nzgd_ids in clusters.items():
        # All matched_pairs in this cluster, in (nz_a, rep_a, nz_b, rep_b) form.
        matched_pairs_sql: list[tuple[int, int, int, int]] = []
        for h, rids in cross_nzgd_buckets:
            for a, b in combinations(rids, 2):
                nz_a, nz_b = report_to_nzgd[a], report_to_nzgd[b]
                if nz_a in nzgd_ids and nz_b in nzgd_ids and nz_a != nz_b:
                    matched_pairs_sql.append((nz_a, a, nz_b, b))

        canonical = select_canonical(conn, nzgd_ids, matched_pairs_sql, table_cfg)
        # For each non-canonical nzgd_id, build a MergePlanEntry.
        for merged_nz in nzgd_ids:
            if merged_nz == canonical:
                continue
            # Matched pairs between (canonical, merged_nz): orient with canonical first.
            entry_matched_pairs: list[ReportPairMatch] = []
            for nz_a, rep_a, nz_b, rep_b in matched_pairs_sql:
                if {nz_a, nz_b} == {canonical, merged_nz}:
                    if nz_a == canonical:
                        entry_matched_pairs.append(
                            ReportPairMatch(rep_a, rep_b, {"hash": True})
                        )
                    else:
                        entry_matched_pairs.append(
                            ReportPairMatch(rep_b, rep_a, {"hash": True})
                        )
            # Reports unique to merged_nz: query its report ids, subtract matched.
            cur.execute(
                f"SELECT {table_cfg.report_id_column} FROM {table_cfg.report_table} "
                f"WHERE nzgd_id = ?",
                (merged_nz,),
            )
            merged_reports = {r[0] for r in cur.fetchall()}
            matched_merged_ids = {p.merged_report_id for p in entry_matched_pairs}
            unique_ids = sorted(merged_reports - matched_merged_ids)
            plan.append(
                MergePlanEntry(
                    cluster_id=cluster_id,
                    canonical_nzgd_id=canonical,
                    merged_nzgd_id=merged_nz,
                    record_type=table_cfg.record_type,
                    match_pass="hash",
                    matched_pairs=entry_matched_pairs,
                    unique_merged_report_ids=unique_ids,
                )
            )
    return plan
```

- [ ] **Step 2: Smoke check on in-memory DB**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
import sqlite3
from nzgd.dedup.data_types import CPT_TABLE_CONFIG
from nzgd.dedup.pass1_hash import generate_hash_merge_plan

conn = sqlite3.connect(':memory:')
conn.executescript('''
CREATE TABLE nzgdrecord (nzgd_id INTEGER PRIMARY KEY, merged_into_nzgd_id INTEGER);
CREATE TABLE cptreport (cpt_id INTEGER PRIMARY KEY, nzgd_id INTEGER);
CREATE TABLE cptmeasurements (measurement_id INTEGER PRIMARY KEY, cpt_id INTEGER, depth_m REAL, qc_MPa REAL, fs_MPa REAL, u2_MPa REAL);
INSERT INTO nzgdrecord VALUES (1, NULL), (2, NULL);
INSERT INTO cptreport VALUES (10, 1), (20, 2);
INSERT INTO cptmeasurements (cpt_id, depth_m, qc_MPa, fs_MPa, u2_MPa) VALUES
    (10, 0.1, 1.0, 0.01, 0.0), (10, 0.2, 1.1, 0.011, 0.0),
    (20, 0.1, 1.0, 0.01, 0.0), (20, 0.2, 1.1, 0.011, 0.0);
''')
plan = generate_hash_merge_plan(conn, CPT_TABLE_CONFIG)
print(f'plan length: {len(plan)}')
for e in plan:
    print(f'  cluster={e.cluster_id} canonical={e.canonical_nzgd_id} merged={e.merged_nzgd_id} pairs={len(e.matched_pairs)}')
"
```

Expected: plan length 1; one entry merging nzgd 2 into nzgd 1 (or vice versa — both have 0 unique data, smallest nzgd_id wins) with 1 matched pair.

- [ ] **Step 3: Commit**

```bash
git add nzgd/dedup/pass1_hash.py
git commit -m "Add Pass 1: hash-based exact-match dedup plan generator"
```

---

## Task 9: Merge executor

**Files:**
- Create: `nzgd/dedup/executor.py`

- [ ] **Step 1: Write `executor.py`**

Create `nzgd/dedup/executor.py` with:

```python
"""Apply a list of MergePlanEntry to a deduped target DB."""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterable

from tqdm import tqdm

from nzgd.dedup.data_types import MergePlanEntry, TableConfig


_NZGDRECORD_METADATA_COLUMNS = (
    "type_id", "latitude", "longitude",
    "model_vs30_foster_2019_km_per_s", "model_vs30_stddev_foster_2019_km_per_s",
    "model_gwl_westerhoff_2018_m", "model_gwl_nlm_2025_m", "model_gwl_nlm_2025_stddev_m",
    "original_investigation_name", "investigation_date", "published_date",
    "region_id", "district_id", "city_id", "suburb_id",
)


def _enrich_canonical_metadata(
    conn: sqlite3.Connection, canonical_nzgd_id: int, merged_nzgd_ids: list[int]
) -> tuple[dict, dict]:
    """Merge non-null metadata from merged records into canonical.

    Returns `(copied, conflicts)`. `copied` is `{column: {"value": v, "source_nzgd_id": id}}`.
    `conflicts` is `{column: [{"value": v, "source_nzgd_id": id}, ...]}`.
    """
    cur = conn.cursor()
    cols_sql = ", ".join(_NZGDRECORD_METADATA_COLUMNS)
    cur.execute(f"SELECT {cols_sql} FROM nzgdrecord WHERE nzgd_id = ?", (canonical_nzgd_id,))
    canon_row = cur.fetchone()
    canon_vals = dict(zip(_NZGDRECORD_METADATA_COLUMNS, canon_row))

    # Build {column: [(value, source_nzgd_id), ...]} of non-null values from merged records
    merged_vals: dict[str, list[tuple]] = {c: [] for c in _NZGDRECORD_METADATA_COLUMNS}
    for nz in merged_nzgd_ids:
        cur.execute(f"SELECT {cols_sql} FROM nzgdrecord WHERE nzgd_id = ?", (nz,))
        row = cur.fetchone()
        if row is None:
            continue
        for col, val in zip(_NZGDRECORD_METADATA_COLUMNS, row):
            if val is not None:
                merged_vals[col].append((val, nz))

    copied: dict[str, dict] = {}
    conflicts: dict[str, list[dict]] = {}
    updates: dict[str, object] = {}
    for col in _NZGDRECORD_METADATA_COLUMNS:
        if canon_vals[col] is not None:
            continue  # canonical wins
        candidates = merged_vals[col]
        if not candidates:
            continue  # nothing to copy
        distinct = {c[0] for c in candidates}
        if len(distinct) == 1:
            chosen = candidates[0]
            updates[col] = chosen[0]
            copied[col] = {"value": chosen[0], "source_nzgd_id": chosen[1]}
        else:
            # Multiple distinct values; pick the one from the smallest merged nzgd_id.
            chosen = min(candidates, key=lambda c: c[1])
            updates[col] = chosen[0]
            copied[col] = {"value": chosen[0], "source_nzgd_id": chosen[1]}
            conflicts[col] = [
                {"value": v, "source_nzgd_id": nz} for v, nz in candidates
            ]

    if updates:
        set_clause = ", ".join(f"{c} = ?" for c in updates)
        cur.execute(
            f"UPDATE nzgdrecord SET {set_clause} WHERE nzgd_id = ?",
            (*updates.values(), canonical_nzgd_id),
        )
    return copied, conflicts


def _delete_report(
    conn: sqlite3.Connection, report_id: int, table_cfg: TableConfig
) -> None:
    """Delete a report row and all its dependent rows in the spec-mandated order."""
    cur = conn.cursor()
    # Special-case SPT: soilmeasurementsoiltype joins soilmeasurements via soil_measurement_id.
    if table_cfg.record_type == "BH":
        cur.execute(
            "DELETE FROM soilmeasurementsoiltype "
            "WHERE soil_measurement_id IN (SELECT soil_measurement_id FROM soilmeasurements WHERE spt_id = ?)",
            (report_id,),
        )
    for tbl, fk in table_cfg.dependent_tables:
        cur.execute(f"DELETE FROM {tbl} WHERE {fk} = ?", (report_id,))
    cur.execute(
        f"DELETE FROM {table_cfg.report_table} WHERE {table_cfg.report_id_column} = ?",
        (report_id,),
    )


def _reparent_report(
    conn: sqlite3.Connection, report_id: int, new_nzgd_id: int, table_cfg: TableConfig
) -> None:
    cur = conn.cursor()
    cur.execute(
        f"UPDATE {table_cfg.report_table} SET nzgd_id = ? "
        f"WHERE {table_cfg.report_id_column} = ?",
        (new_nzgd_id, report_id),
    )
    if table_cfg.record_type == "CPT":
        cur.execute(
            "UPDATE cptvs30estimates SET nzgd_id = ? WHERE cpt_id = ?",
            (new_nzgd_id, report_id),
        )


def apply_merge_plan(
    conn: sqlite3.Connection,
    plan: Iterable[MergePlanEntry],
    run_id: int,
    table_cfg: TableConfig,
    failures: list[dict] | None = None,
) -> tuple[int, int]:
    """Apply a merge plan and write audit rows. Returns (n_clusters, n_records).

    Each cluster is applied in its own SAVEPOINT so a single bad cluster only
    rolls back itself. Failures are appended to `failures` as dicts if provided.
    """
    plan = list(plan)
    if not plan:
        return 0, 0
    # Group entries by cluster_id so we apply per-cluster transactions.
    by_cluster: dict[int, list[MergePlanEntry]] = {}
    for e in plan:
        by_cluster.setdefault(e.cluster_id, []).append(e)

    n_clusters_ok = 0
    n_records_merged_ok = 0
    cur = conn.cursor()
    for cluster_id, entries in tqdm(by_cluster.items(), desc=f"merging {table_cfg.record_type}"):
        canonical = entries[0].canonical_nzgd_id
        merged_ids = [e.merged_nzgd_id for e in entries]
        savepoint = f"cluster_{cluster_id}"
        cur.execute(f"SAVEPOINT {savepoint}")
        try:
            copied, conflicts = _enrich_canonical_metadata(conn, canonical, merged_ids)
            for entry in entries:
                for pair in entry.matched_pairs:
                    _delete_report(conn, pair.merged_report_id, table_cfg)
                for rid in entry.unique_merged_report_ids:
                    _reparent_report(conn, rid, canonical, table_cfg)
                cur.execute(
                    "UPDATE nzgdrecord SET merged_into_nzgd_id = ? WHERE nzgd_id = ?",
                    (canonical, entry.merged_nzgd_id),
                )
                pairs_json = json.dumps([
                    {
                        "canonical_report_id": p.canonical_report_id,
                        "merged_report_id": p.merged_report_id,
                        "metrics": p.metrics,
                    }
                    for p in entry.matched_pairs
                ])
                cur.execute(
                    "INSERT INTO dedup_audit (run_id, cluster_id, canonical_nzgd_id, merged_nzgd_id, "
                    "record_type, match_pass, report_pairs_json, metadata_copied_json, "
                    "metadata_conflicts_json, merged_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        entry.cluster_id,
                        entry.canonical_nzgd_id,
                        entry.merged_nzgd_id,
                        entry.record_type,
                        entry.match_pass,
                        pairs_json,
                        json.dumps(copied) if copied else None,
                        json.dumps(conflicts) if conflicts else None,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                n_records_merged_ok += 1
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            n_clusters_ok += 1
        except Exception as exc:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            if failures is not None:
                failures.append({
                    "cluster_id": cluster_id,
                    "canonical_nzgd_id": canonical,
                    "merged_nzgd_ids": merged_ids,
                    "record_type": table_cfg.record_type,
                    "error": repr(exc),
                })
    conn.commit()
    return n_clusters_ok, n_records_merged_ok
```

- [ ] **Step 2: Smoke check using the Task 8 fixture**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
import sqlite3
from nzgd.dedup.schema import apply_dedup_schema
from nzgd.dedup.data_types import CPT_TABLE_CONFIG
from nzgd.dedup.pass1_hash import generate_hash_merge_plan
from nzgd.dedup.executor import apply_merge_plan

conn = sqlite3.connect(':memory:')
conn.execute('PRAGMA foreign_keys = ON')
conn.executescript('''
CREATE TABLE nzgdrecord (nzgd_id INTEGER PRIMARY KEY,
    type_id INTEGER, latitude REAL, longitude REAL,
    model_vs30_foster_2019_km_per_s REAL, model_vs30_stddev_foster_2019_km_per_s REAL,
    model_gwl_westerhoff_2018_m REAL, model_gwl_nlm_2025_m REAL, model_gwl_nlm_2025_stddev_m REAL,
    original_investigation_name TEXT, investigation_date TEXT, published_date TEXT,
    region_id INTEGER, district_id INTEGER, city_id INTEGER, suburb_id INTEGER);
CREATE TABLE cptreport (cpt_id INTEGER PRIMARY KEY, nzgd_id INTEGER);
CREATE TABLE cptmeasurements (measurement_id INTEGER PRIMARY KEY, cpt_id INTEGER, depth_m REAL, qc_MPa REAL, fs_MPa REAL, u2_MPa REAL);
CREATE TABLE cptvs30estimates (vs30_id INTEGER PRIMARY KEY, cpt_id INTEGER, nzgd_id INTEGER);
INSERT INTO nzgdrecord (nzgd_id, type_id, latitude, longitude) VALUES (1, 1, -41.0, 174.0), (2, 1, -41.0, 174.0);
INSERT INTO cptreport VALUES (10, 1), (20, 2);
INSERT INTO cptmeasurements (cpt_id, depth_m, qc_MPa, fs_MPa, u2_MPa) VALUES
    (10, 0.1, 1.0, 0.01, 0.0), (10, 0.2, 1.1, 0.011, 0.0),
    (20, 0.1, 1.0, 0.01, 0.0), (20, 0.2, 1.1, 0.011, 0.0);
''')
apply_dedup_schema(conn)
conn.execute(
    'INSERT INTO dedup_run (run_id, started_at, source_db_path, script_version, config_snapshot_json) VALUES (1, ?, ?, ?, ?)',
    ('2026-05-20T00:00:00Z', ':memory:', 'test', '{}'),
)
plan = generate_hash_merge_plan(conn, CPT_TABLE_CONFIG)
n_clusters, n_records = apply_merge_plan(conn, plan, run_id=1, table_cfg=CPT_TABLE_CONFIG)
print(f'clusters={n_clusters} records={n_records}')
print('nzgdrecord:', list(conn.execute('SELECT nzgd_id, merged_into_nzgd_id FROM nzgdrecord')))
print('cptreport:', list(conn.execute('SELECT * FROM cptreport')))
print('cptmeasurements count:', conn.execute('SELECT COUNT(*) FROM cptmeasurements').fetchone()[0])
print('dedup_audit:', list(conn.execute('SELECT cluster_id, canonical_nzgd_id, merged_nzgd_id, match_pass FROM dedup_audit')))
"
```

Expected:
- `clusters=1 records=1`
- nzgdrecord: nzgd 2 has `merged_into_nzgd_id = 1`
- cptreport: only cpt_id 10 remains (cpt_id 20 deleted)
- cptmeasurements count: 2 (cpt_id 20's rows deleted)
- dedup_audit: one row with `match_pass='hash'`

- [ ] **Step 3: Commit**

```bash
git add nzgd/dedup/executor.py
git commit -m "Add merge executor with per-cluster savepoints and audit logging"
```

---

## Task 10: Reports

**Files:**
- Create: `nzgd/dedup/reports.py`

- [ ] **Step 1: Write `reports.py`**

Create `nzgd/dedup/reports.py` with:

```python
"""CSV writers for dedup script outputs."""

import csv
import json
import sqlite3
from pathlib import Path


def write_dedup_report(conn: sqlite3.Connection, run_id: int, path: Path) -> None:
    """Flatten dedup_audit rows for a given run into a CSV."""
    cur = conn.cursor()
    cur.execute(
        "SELECT cluster_id, canonical_nzgd_id, merged_nzgd_id, record_type, "
        "match_pass, report_pairs_json, metadata_copied_json, metadata_conflicts_json, merged_at "
        "FROM dedup_audit WHERE run_id = ? "
        "ORDER BY record_type, cluster_id, merged_nzgd_id",
        (run_id,),
    )
    rows = cur.fetchall()
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster_id", "canonical_nzgd_id", "merged_nzgd_id", "record_type",
            "match_pass", "matched_pair_count", "metadata_copied_fields",
            "metadata_conflict_fields", "merged_at",
        ])
        for r in rows:
            (cluster, canon, merged, rtype, mpass, pairs_json,
             copied_json, conflicts_json, merged_at) = r
            pair_count = len(json.loads(pairs_json))
            copied_fields = ",".join(json.loads(copied_json).keys()) if copied_json else ""
            conflict_fields = ",".join(json.loads(conflicts_json).keys()) if conflicts_json else ""
            writer.writerow([cluster, canon, merged, rtype, mpass, pair_count,
                             copied_fields, conflict_fields, merged_at])


def write_calibration_report(
    positive_rows: list[dict],
    negative_rows: list[dict],
    path: Path,
) -> None:
    """Write feature distributions for positive and negative fuzzy-pass examples."""
    fieldnames = ["group", "spatial_m", "date_days", "name_sim",
                  "max_depth_diff_m", "trace_score"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in positive_rows:
            writer.writerow({"group": "positive", **{k: r.get(k) for k in fieldnames if k != "group"}})
        for r in negative_rows:
            writer.writerow({"group": "negative", **{k: r.get(k) for k in fieldnames if k != "group"}})


def write_failures_report(failures: list[dict], path: Path) -> None:
    """Write rolled-back-cluster failures to CSV with full JSON detail."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "canonical_nzgd_id", "merged_nzgd_ids",
                         "record_type", "error"])
        for fail in failures:
            writer.writerow([
                fail.get("cluster_id"),
                fail.get("canonical_nzgd_id"),
                json.dumps(fail.get("merged_nzgd_ids")),
                fail.get("record_type"),
                fail.get("error"),
            ])
```

- [ ] **Step 2: Smoke check on an in-memory DB**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
import sqlite3, json, tempfile, pathlib
from nzgd.dedup.schema import apply_dedup_schema
from nzgd.dedup.reports import write_dedup_report, write_failures_report

conn = sqlite3.connect(':memory:')
conn.execute('CREATE TABLE nzgdrecord (nzgd_id INTEGER PRIMARY KEY)')
apply_dedup_schema(conn)
conn.execute(
    'INSERT INTO dedup_run (run_id, started_at, source_db_path, script_version, config_snapshot_json) VALUES (1, ?, ?, ?, ?)',
    ('2026-05-20T00:00:00Z', ':memory:', 'test', '{}'),
)
conn.execute(
    'INSERT INTO dedup_audit (run_id, cluster_id, canonical_nzgd_id, merged_nzgd_id, record_type, match_pass, report_pairs_json, merged_at) VALUES (1, 1, 100, 200, ?, ?, ?, ?)',
    ('CPT', 'hash', json.dumps([{'canonical_report_id': 10, 'merged_report_id': 20, 'metrics': {}}]), '2026-05-20T00:00:00Z'),
)
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / 'r.csv'
    write_dedup_report(conn, 1, p)
    print(p.read_text())
    pf = pathlib.Path(d) / 'f.csv'
    write_failures_report([{'cluster_id': 5, 'canonical_nzgd_id': 1, 'merged_nzgd_ids': [2, 3], 'record_type': 'CPT', 'error': 'boom'}], pf)
    print(pf.read_text())
"
```

Expected: two CSVs printed; the first has a row with `1,100,200,CPT,hash,1,...`; the second has a row with `5,1,\"[2, 3]\",CPT,boom`.

- [ ] **Step 3: Commit**

```bash
git add nzgd/dedup/reports.py
git commit -m "Add CSV report writers for dedup audit/calibration/failures"
```

---

## Task 11: First integration test (exact-duplicate scenario)

**Files:**
- Create: `tests/dedup/conftest.py`
- Create: `tests/dedup/test_dedup_pipeline.py`

- [ ] **Step 1: Write `conftest.py`**

Create `tests/dedup/conftest.py` with:

```python
"""Synthetic-DB fixtures for dedup integration tests."""

import sqlite3
from pathlib import Path

import pytest


_FULL_SCHEMA_SQL = """
CREATE TABLE type (id INTEGER PRIMARY KEY, value TEXT);
INSERT INTO type VALUES (1, 'CPT'), (2, 'BH');

CREATE TABLE nzgdrecord (
    nzgd_id INTEGER PRIMARY KEY,
    type_id INTEGER NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    model_vs30_foster_2019_km_per_s REAL,
    model_vs30_stddev_foster_2019_km_per_s REAL,
    model_gwl_westerhoff_2018_m REAL,
    model_gwl_nlm_2025_m REAL,
    model_gwl_nlm_2025_stddev_m REAL,
    original_investigation_name TEXT,
    investigation_date TEXT,
    published_date TEXT,
    region_id INTEGER NOT NULL DEFAULT 0,
    district_id INTEGER NOT NULL DEFAULT 0,
    city_id INTEGER NOT NULL DEFAULT 0,
    suburb_id INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE cptreport (
    cpt_id INTEGER PRIMARY KEY,
    nzgd_id INTEGER NOT NULL,
    max_depth_m REAL,
    min_depth_m REAL,
    extracted_gwl_m REAL,
    gwl_method_id INTEGER,
    tip_net_area_ratio REAL,
    predrill_depth_m REAL,
    termination_reason_id INTEGER,
    has_cpt_data INTEGER NOT NULL DEFAULT 1,
    cpt_data_duplicate_of_cpt_id INTEGER,
    did_explicit_unit_conversion INTEGER,
    did_inferred_unit_conversion INTEGER,
    source_file TEXT NOT NULL DEFAULT 'synthetic.xls'
);

CREATE TABLE cptmeasurements (
    measurement_id INTEGER PRIMARY KEY,
    cpt_id INTEGER NOT NULL,
    depth_m REAL,
    qc_MPa REAL,
    fs_MPa REAL,
    u2_MPa REAL
);

CREATE TABLE cptvs30estimates (
    vs30_id INTEGER PRIMARY KEY,
    cpt_id INTEGER,
    nzgd_id INTEGER,
    cpt_to_vs_correlation_id INTEGER,
    vs_to_vs30_correlation_id INTEGER,
    vs30 REAL,
    vs30_stddev REAL
);

CREATE TABLE sptreport (
    spt_id INTEGER PRIMARY KEY,
    nzgd_id INTEGER NOT NULL,
    efficiency REAL,
    extracted_gwl_m REAL,
    borehole_diameter REAL,
    casing_diameter REAL,
    source_file TEXT NOT NULL DEFAULT 'synthetic.ags'
);

CREATE TABLE sptmeasurements (
    spt_measurement_id INTEGER PRIMARY KEY,
    spt_id INTEGER NOT NULL,
    depth_m REAL,
    ISPT_MAIN INTEGER,
    ISPT_NVAL INTEGER,
    ISPT_REP INTEGER
);

CREATE TABLE soilmeasurements (
    soil_measurement_id INTEGER PRIMARY KEY,
    spt_id INTEGER NOT NULL,
    top_depth_m REAL,
    bottom_depth_m REAL
);

CREATE TABLE densitymeasurements (
    density_measurement_id INTEGER PRIMARY KEY,
    spt_id INTEGER NOT NULL,
    top_depth_m REAL,
    bottom_depth_m REAL,
    density_keyword TEXT
);

CREATE TABLE soilmeasurementsoiltype (
    soil_measurement_id INTEGER NOT NULL,
    soil_type_id INTEGER NOT NULL,
    PRIMARY KEY (soil_measurement_id, soil_type_id)
);

CREATE TABLE sptvs30estimates (
    vs30_id INTEGER PRIMARY KEY,
    spt_id INTEGER,
    spt_to_vs_correlation_id INTEGER,
    vs_to_vs30_correlation_id INTEGER,
    assumed_borehole_diameter_mm REAL,
    assumed_hammer_type_id INTEGER,
    estimate_used_extracted_efficiency INTEGER,
    estimate_used_extracted_layer_soil_types INTEGER,
    vs30 REAL,
    vs30_stddev REAL
);

CREATE INDEX cptmeasurements_cpt_id ON cptmeasurements(cpt_id);
CREATE INDEX cptreport_nzgd_id ON cptreport(nzgd_id);
CREATE INDEX sptmeasurements_spt_id ON sptmeasurements(spt_id);
CREATE INDEX sptreport_nzgd_id ON sptreport(nzgd_id);
CREATE INDEX soilmeasurements_spt_id ON soilmeasurements(spt_id);
CREATE INDEX densitymeasurements_spt_id ON densitymeasurements(spt_id);
CREATE INDEX soilmeasurementsoiltype_soil_measurement_id ON soilmeasurementsoiltype(soil_measurement_id);
"""


def _make_fresh_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_FULL_SCHEMA_SQL)
    return conn


def add_cpt_record(
    conn: sqlite3.Connection,
    nzgd_id: int,
    lat: float = -41.0,
    lon: float = 174.0,
    investigation_name: str | None = None,
    investigation_date: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO nzgdrecord (nzgd_id, type_id, latitude, longitude, "
        "original_investigation_name, investigation_date) VALUES (?,1,?,?,?,?)",
        (nzgd_id, lat, lon, investigation_name, investigation_date),
    )


def add_bh_record(
    conn: sqlite3.Connection,
    nzgd_id: int,
    lat: float = -41.0,
    lon: float = 174.0,
    investigation_name: str | None = None,
    investigation_date: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO nzgdrecord (nzgd_id, type_id, latitude, longitude, "
        "original_investigation_name, investigation_date) VALUES (?,2,?,?,?,?)",
        (nzgd_id, lat, lon, investigation_name, investigation_date),
    )


def add_cpt_report(
    conn: sqlite3.Connection,
    cpt_id: int,
    nzgd_id: int,
    trace: list[tuple[float, float, float, float]],
    source_file: str = "synthetic.xls",
) -> None:
    conn.execute(
        "INSERT INTO cptreport (cpt_id, nzgd_id, max_depth_m, min_depth_m, has_cpt_data, source_file) "
        "VALUES (?,?,?,?,1,?)",
        (cpt_id, nzgd_id,
         max(r[0] for r in trace) if trace else None,
         min(r[0] for r in trace) if trace else None,
         source_file),
    )
    conn.executemany(
        "INSERT INTO cptmeasurements (cpt_id, depth_m, qc_MPa, fs_MPa, u2_MPa) VALUES (?,?,?,?,?)",
        [(cpt_id, *row) for row in trace],
    )


def add_spt_report(
    conn: sqlite3.Connection,
    spt_id: int,
    nzgd_id: int,
    trace: list[tuple[float, int, int, int]],
    source_file: str = "synthetic.ags",
) -> None:
    conn.execute(
        "INSERT INTO sptreport (spt_id, nzgd_id, source_file) VALUES (?,?,?)",
        (spt_id, nzgd_id, source_file),
    )
    conn.executemany(
        "INSERT INTO sptmeasurements (spt_id, depth_m, ISPT_MAIN, ISPT_NVAL, ISPT_REP) VALUES (?,?,?,?,?)",
        [(spt_id, *row) for row in trace],
    )


@pytest.fixture
def fresh_db(tmp_path: Path) -> sqlite3.Connection:
    """A blank schema-loaded DB. Each test populates it as needed."""
    db_path = tmp_path / "fresh.db"
    conn = _make_fresh_db(db_path)
    yield conn
    conn.close()
```

- [ ] **Step 2: Write the first integration test**

Create `tests/dedup/test_dedup_pipeline.py` with:

```python
"""End-to-end integration scenarios for the dedup pipeline."""

import json
import sqlite3

import pytest

from nzgd.dedup.data_types import CPT_TABLE_CONFIG, SPT_TABLE_CONFIG
from nzgd.dedup.executor import apply_merge_plan
from nzgd.dedup.pass1_hash import generate_hash_merge_plan
from nzgd.dedup.schema import apply_dedup_schema
from tests.dedup.conftest import (
    add_cpt_record,
    add_cpt_report,
    add_bh_record,
    add_spt_report,
)


def _start_run(conn: sqlite3.Connection) -> int:
    apply_dedup_schema(conn)
    cur = conn.execute(
        "INSERT INTO dedup_run (started_at, source_db_path, script_version, config_snapshot_json) "
        "VALUES (?, ?, ?, ?)",
        ("2026-05-20T00:00:00Z", ":memory:", "test", "{}"),
    )
    return cur.lastrowid


def test_exact_duplicate_pair_is_merged(fresh_db: sqlite3.Connection) -> None:
    trace = [(0.10, 1.00, 0.010, 0.000),
             (0.20, 1.10, 0.011, 0.000),
             (0.30, 1.20, 0.012, 0.000)]
    add_cpt_record(fresh_db, nzgd_id=1, lat=-41.0, lon=174.0,
                   investigation_name="Site A", investigation_date="2024-01-15")
    add_cpt_record(fresh_db, nzgd_id=2, lat=-41.0, lon=174.0,
                   investigation_name=None, investigation_date="2024-02-01")
    add_cpt_report(fresh_db, cpt_id=10, nzgd_id=1, trace=trace)
    add_cpt_report(fresh_db, cpt_id=20, nzgd_id=2, trace=trace)

    run_id = _start_run(fresh_db)
    plan = generate_hash_merge_plan(fresh_db, CPT_TABLE_CONFIG)
    n_clusters, n_records = apply_merge_plan(fresh_db, plan, run_id, CPT_TABLE_CONFIG)

    assert n_clusters == 1 and n_records == 1
    # nzgd 1 has more non-null metadata (investigation_name set), so it wins.
    merged_into = fresh_db.execute(
        "SELECT merged_into_nzgd_id FROM nzgdrecord WHERE nzgd_id = 2"
    ).fetchone()[0]
    assert merged_into == 1
    assert fresh_db.execute(
        "SELECT merged_into_nzgd_id FROM nzgdrecord WHERE nzgd_id = 1"
    ).fetchone()[0] is None
    # cpt_id 20 deleted; cpt_id 10 still there
    remaining = [r[0] for r in fresh_db.execute("SELECT cpt_id FROM cptreport")]
    assert remaining == [10]
    # cptmeasurements for cpt_id 20 deleted
    assert fresh_db.execute("SELECT COUNT(*) FROM cptmeasurements WHERE cpt_id = 20").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM cptmeasurements WHERE cpt_id = 10").fetchone()[0] == 3
    # audit row exists with hash pass
    audit = fresh_db.execute(
        "SELECT cluster_id, canonical_nzgd_id, merged_nzgd_id, match_pass FROM dedup_audit"
    ).fetchall()
    assert audit == [(1, 1, 2, "hash")]
```

- [ ] **Step 3: Run the test**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py::test_exact_duplicate_pair_is_merged -v
```

Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/dedup/conftest.py tests/dedup/test_dedup_pipeline.py
git commit -m "Add dedup integration-test fixture + first scenario (exact duplicate)"
```

---

## Task 12: Fuzzy pass

**Files:**
- Create: `nzgd/dedup/pass2_fuzzy.py`

- [ ] **Step 1: Write `pass2_fuzzy.py`**

Create `nzgd/dedup/pass2_fuzzy.py` with:

```python
"""Pass 2: metadata-blocked fuzzy match over post-pass-1 survivors."""

import datetime as _dt
import math
import random
import sqlite3
from collections import defaultdict
from itertools import combinations

import numpy as np
from rapidfuzz import fuzz
from sklearn.neighbors import BallTree
from tqdm import tqdm

from nzgd.dedup.cluster import connected_components_from_edges
from nzgd.dedup.data_types import MergePlanEntry, ReportPairMatch, TableConfig
from nzgd.dedup.selection import select_canonical


def _load_active_records(
    conn: sqlite3.Connection, table_cfg: TableConfig
) -> list[dict]:
    """Return one dict per nzgd_id of the given record type that still has active reports."""
    type_id = 1 if table_cfg.record_type == "CPT" else 2
    cur = conn.cursor()
    cur.execute(
        "SELECT n.nzgd_id, n.latitude, n.longitude, n.investigation_date, "
        "n.original_investigation_name "
        "FROM nzgdrecord n "
        f"WHERE n.merged_into_nzgd_id IS NULL AND n.type_id = ? "
        f"AND EXISTS (SELECT 1 FROM {table_cfg.report_table} r WHERE r.nzgd_id = n.nzgd_id)",
        (type_id,),
    )
    return [
        {"nzgd_id": r[0], "lat": r[1], "lon": r[2], "date": r[3], "name": r[4]}
        for r in cur.fetchall()
    ]


def _load_traces(
    conn: sqlite3.Connection, nzgd_id: int, table_cfg: TableConfig
) -> dict[int, np.ndarray]:
    """Return `{report_id: ndarray of shape (n_rows, len(value_columns))}` for one nzgd_id.

    Columns appear in the order given by `table_cfg.measurement_value_columns`;
    the first must be `depth_m` so column 0 of each array is depth.
    """
    value_cols_with_m = ", ".join(f"m.{c}" for c in table_cfg.measurement_value_columns)
    cur = conn.cursor()
    cur.execute(
        f"SELECT r.{table_cfg.report_id_column}, {value_cols_with_m} "
        f"FROM {table_cfg.measurement_table} m "
        f"JOIN {table_cfg.report_table} r "
        f"ON r.{table_cfg.report_id_column} = m.{table_cfg.report_id_column} "
        f"WHERE r.nzgd_id = ? "
        f"ORDER BY r.{table_cfg.report_id_column}, m.depth_m",
        (nzgd_id,),
    )
    rows_by_report: dict[int, list[tuple]] = defaultdict(list)
    for row in cur.fetchall():
        rid = row[0]
        rows_by_report[rid].append(row[1:])
    return {rid: np.array(rows, dtype=float) for rid, rows in rows_by_report.items()}


def _trace_score(a: np.ndarray, b: np.ndarray, step: float) -> float:
    """Aligned normalised-RMSE sum across non-depth channels.

    Returns inf if either trace has fewer than 2 points or their depth ranges
    do not overlap. Each channel's RMSE is divided by the mean absolute value
    across both traces (floored at 1e-6) before being summed across channels,
    so the score is comparable across qc/fs/u2 which have different magnitudes.
    """
    if a.shape[0] < 2 or b.shape[0] < 2:
        return math.inf
    lo = max(a[:, 0].min(), b[:, 0].min())
    hi = min(a[:, 0].max(), b[:, 0].max())
    if hi <= lo:
        return math.inf
    grid = np.arange(lo, hi + step / 2, step)
    if grid.size < 2:
        return math.inf
    total = 0.0
    for ch in range(1, a.shape[1]):
        ai = np.interp(grid, a[:, 0], a[:, ch])
        bi = np.interp(grid, b[:, 0], b[:, ch])
        denom = max((np.abs(ai).mean() + np.abs(bi).mean()) / 2.0, 1e-6)
        rmse = float(np.sqrt(np.mean((ai - bi) ** 2)))
        total += rmse / denom
    return total


def _best_trace_score(
    traces_a: dict[int, np.ndarray],
    traces_b: dict[int, np.ndarray],
    step: float,
) -> tuple[float, tuple[int, int] | None]:
    """Best (lowest) trace_score over all cross-record report pairs, plus the winning pair."""
    best_score = math.inf
    best_pair: tuple[int, int] | None = None
    for ra, ta in traces_a.items():
        for rb, tb in traces_b.items():
            s = _trace_score(ta, tb, step)
            if s < best_score:
                best_score = s
                best_pair = (ra, rb)
    return best_score, best_pair


def _predicate(features: dict, thresholds: dict) -> bool:
    """Spec's conjunctive match predicate. Missing date/name are 'no signal'."""
    if features["spatial_m"] >= thresholds["spatial_radius_m"]:
        return False
    if features["date_days"] is not None and features["date_days"] >= thresholds["date_window_days"]:
        return False
    if features["name_sim"] is not None and features["name_sim"] <= thresholds["name_similarity_min"]:
        return False
    if features["trace_score"] >= thresholds["trace_score_max"]:
        return False
    return True


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return float(2 * r * math.asin(math.sqrt(a)))


def _date_diff_days(a: str | None, b: str | None) -> float | None:
    if a is None or b is None:
        return None
    try:
        da = _dt.date.fromisoformat(a)
        db = _dt.date.fromisoformat(b)
    except (TypeError, ValueError):
        return None
    return float(abs((da - db).days))


def _name_sim(a: str | None, b: str | None) -> float | None:
    if a is None or b is None:
        return None
    return float(fuzz.token_set_ratio(a, b))


def _blocked_candidate_pairs(
    records: list[dict], radius_m: float
) -> list[tuple[int, int]]:
    """BallTree haversine-radius query → unique unordered nzgd_id pairs."""
    if len(records) < 2:
        return []
    rad = np.radians(np.array([[r["lat"], r["lon"]] for r in records]))
    tree = BallTree(rad, metric="haversine")
    earth_r = 6371008.8
    radius_rad = radius_m / earth_r
    pairs: set[tuple[int, int]] = set()
    for i, point in enumerate(rad):
        idxs = tree.query_radius(point.reshape(1, -1), r=radius_rad)[0]
        for j in idxs:
            if j == i:
                continue
            a, b = records[i]["nzgd_id"], records[int(j)]["nzgd_id"]
            pairs.add((min(a, b), max(a, b)))
    return list(pairs)


def generate_fuzzy_merge_plan(
    conn: sqlite3.Connection,
    table_cfg: TableConfig,
    thresholds: dict,
    calibration_collector: dict | None = None,
    random_seed: int = 0,
) -> list[MergePlanEntry]:
    """Produce MergePlanEntry list for fuzzy near-duplicates.

    Entries within a cluster are sorted by trace_score (best-match first) so the
    executor's per-step metadata-enrichment respects the spec's score-order rule.

    `calibration_collector`, if provided, is populated with two lists:
      - `positive`: feature dicts for pairs that fired the predicate.
      - `negative`: feature dicts for a random sample of pairs that did not fire.
    Caller passes both to `reports.write_calibration_report`.
    """
    records = _load_active_records(conn, table_cfg)
    by_nzgd = {r["nzgd_id"]: r for r in records}
    candidate_pairs = _blocked_candidate_pairs(records, thresholds["spatial_radius_m"])

    step = thresholds["trace_resample_step_m"]
    trace_cache: dict[int, dict[int, np.ndarray]] = {}

    def traces_for(nz: int) -> dict[int, np.ndarray]:
        if nz not in trace_cache:
            trace_cache[nz] = _load_traces(conn, nz, table_cfg)
        return trace_cache[nz]

    positive_examples: list[dict] = []
    negative_examples: list[dict] = []
    rng = random.Random(random_seed)
    negative_target = thresholds.get("random_pair_sample_size", 5000)

    edges: list[tuple[int, int]] = []
    pair_metadata: dict[tuple[int, int], dict] = {}

    for nz_a, nz_b in tqdm(candidate_pairs, desc=f"fuzzy {table_cfg.record_type}"):
        rec_a = by_nzgd[nz_a]
        rec_b = by_nzgd[nz_b]
        spatial = _haversine_m(rec_a["lat"], rec_a["lon"], rec_b["lat"], rec_b["lon"])
        date_days = _date_diff_days(rec_a["date"], rec_b["date"])
        name_sim = _name_sim(rec_a["name"], rec_b["name"])
        ta, tb = traces_for(nz_a), traces_for(nz_b)
        trace_score, best_pair = _best_trace_score(ta, tb, step)
        max_depth_diff: float | None = None
        if ta and tb:
            a_depths = [t[:, 0].max() for t in ta.values() if t.shape[0] > 0]
            b_depths = [t[:, 0].max() for t in tb.values() if t.shape[0] > 0]
            if a_depths and b_depths:
                max_depth_diff = float(abs(max(a_depths) - max(b_depths)))
        features = {
            "spatial_m": spatial,
            "date_days": date_days,
            "name_sim": name_sim,
            "max_depth_diff_m": max_depth_diff,
            "trace_score": trace_score,
        }

        predicate_matched = _predicate(features, thresholds) and best_pair is not None
        if predicate_matched:
            edges.append((nz_a, nz_b))
            key = (min(nz_a, nz_b), max(nz_a, nz_b))
            best_pair_oriented = best_pair if nz_a < nz_b else (best_pair[1], best_pair[0])
            pair_metadata[key] = {"features": features, "best_pair": best_pair_oriented}

        if calibration_collector is not None:
            if predicate_matched:
                positive_examples.append(features)
            elif len(negative_examples) < negative_target and rng.random() < 0.5:
                negative_examples.append(features)

    if calibration_collector is not None:
        calibration_collector["positive"] = positive_examples
        calibration_collector["negative"] = negative_examples

    if not edges:
        return []

    nzgd_to_cluster = connected_components_from_edges(edges)
    clusters: dict[int, list[int]] = defaultdict(list)
    for nz, cl in nzgd_to_cluster.items():
        clusters[cl].append(nz)

    cur = conn.cursor()
    plan: list[MergePlanEntry] = []
    for cluster_id, nzgd_ids in clusters.items():
        matched_pairs_for_selection: list[tuple[int, int, int, int]] = []
        per_pair_match: dict[tuple[int, int], ReportPairMatch] = {}
        for a, b in combinations(sorted(nzgd_ids), 2):
            key = (min(a, b), max(a, b))
            if key not in pair_metadata:
                continue
            ra, rb = pair_metadata[key]["best_pair"]
            matched_pairs_for_selection.append((a, ra, b, rb))
            per_pair_match[key] = ReportPairMatch(
                canonical_report_id=ra,
                merged_report_id=rb,
                metrics=dict(pair_metadata[key]["features"]),
            )

        canonical = select_canonical(conn, nzgd_ids, matched_pairs_for_selection, table_cfg)

        # Sort the merged nzgd_ids by trace_score with the canonical (best first),
        # so the executor applies metadata enrichment in score order.
        def _score_with_canonical(merged_nz: int) -> float:
            key = (min(canonical, merged_nz), max(canonical, merged_nz))
            return pair_metadata.get(key, {}).get("features", {}).get("trace_score", math.inf)

        ordered_merged = sorted(
            (nz for nz in nzgd_ids if nz != canonical),
            key=_score_with_canonical,
        )

        for merged_nz in ordered_merged:
            key = (min(canonical, merged_nz), max(canonical, merged_nz))
            matched_pairs: list[ReportPairMatch] = []
            if key in per_pair_match:
                pm = per_pair_match[key]
                if canonical < merged_nz:
                    matched_pairs.append(pm)
                else:
                    matched_pairs.append(
                        ReportPairMatch(pm.merged_report_id, pm.canonical_report_id, pm.metrics)
                    )

            cur.execute(
                f"SELECT {table_cfg.report_id_column} FROM {table_cfg.report_table} "
                f"WHERE nzgd_id = ?",
                (merged_nz,),
            )
            merged_reports = {r[0] for r in cur.fetchall()}
            matched_merged_ids = {p.merged_report_id for p in matched_pairs}
            unique_ids = sorted(merged_reports - matched_merged_ids)
            plan.append(
                MergePlanEntry(
                    cluster_id=cluster_id,
                    canonical_nzgd_id=canonical,
                    merged_nzgd_id=merged_nz,
                    record_type=table_cfg.record_type,
                    match_pass="fuzzy",
                    matched_pairs=matched_pairs,
                    unique_merged_report_ids=unique_ids,
                )
            )
    return plan
```

- [ ] **Step 2: Smoke check fuzzy pass on a near-duplicate fixture**

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
import sqlite3
from nzgd.dedup.schema import apply_dedup_schema
from nzgd.dedup.data_types import CPT_TABLE_CONFIG
from nzgd.dedup.pass2_fuzzy import generate_fuzzy_merge_plan
from nzgd.dedup.executor import apply_merge_plan
from nzgd import constants

DEDUP_THRESHOLDS = constants.DEDUP_CONFIG['fuzzy_pass']

conn = sqlite3.connect(':memory:')
conn.execute('PRAGMA foreign_keys = ON')
conn.executescript('''
CREATE TABLE type (id INTEGER PRIMARY KEY, value TEXT);
INSERT INTO type VALUES (1, 'CPT');
CREATE TABLE nzgdrecord (nzgd_id INTEGER PRIMARY KEY, type_id INTEGER NOT NULL,
    latitude REAL NOT NULL, longitude REAL NOT NULL,
    model_vs30_foster_2019_km_per_s REAL, model_vs30_stddev_foster_2019_km_per_s REAL,
    model_gwl_westerhoff_2018_m REAL, model_gwl_nlm_2025_m REAL, model_gwl_nlm_2025_stddev_m REAL,
    original_investigation_name TEXT, investigation_date TEXT, published_date TEXT,
    region_id INTEGER DEFAULT 0, district_id INTEGER DEFAULT 0, city_id INTEGER DEFAULT 0, suburb_id INTEGER DEFAULT 0);
CREATE TABLE cptreport (cpt_id INTEGER PRIMARY KEY, nzgd_id INTEGER NOT NULL, max_depth_m REAL, source_file TEXT NOT NULL DEFAULT 'x', has_cpt_data INTEGER NOT NULL DEFAULT 1);
CREATE TABLE cptmeasurements (measurement_id INTEGER PRIMARY KEY, cpt_id INTEGER, depth_m REAL, qc_MPa REAL, fs_MPa REAL, u2_MPa REAL);
CREATE TABLE cptvs30estimates (vs30_id INTEGER PRIMARY KEY, cpt_id INTEGER, nzgd_id INTEGER);
INSERT INTO nzgdrecord (nzgd_id, type_id, latitude, longitude, original_investigation_name, investigation_date) VALUES
    (1, 1, -41.0, 174.0, 'Site A', '2024-01-01'),
    (2, 1, -41.0, 174.0001, 'Site A', '2024-01-02');
INSERT INTO cptreport (cpt_id, nzgd_id) VALUES (10, 1), (20, 2);
INSERT INTO cptmeasurements (cpt_id, depth_m, qc_MPa, fs_MPa, u2_MPa) VALUES
    (10, 0.1, 1.0, 0.01, 0.0), (10, 0.2, 1.1, 0.011, 0.0), (10, 0.3, 1.2, 0.012, 0.0),
    (20, 0.1, 1.001, 0.0101, 0.0), (20, 0.2, 1.101, 0.0111, 0.0), (20, 0.3, 1.201, 0.0121, 0.0);
''')
apply_dedup_schema(conn)
conn.execute('INSERT INTO dedup_run (run_id, started_at, source_db_path, script_version, config_snapshot_json) VALUES (1, ?, ?, ?, ?)',
             ('t', ':memory:', 't', '{}'))
plan = generate_fuzzy_merge_plan(conn, CPT_TABLE_CONFIG, DEDUP_THRESHOLDS)
print(f'plan length: {len(plan)}')
for p in plan:
    print(f'  cluster={p.cluster_id} canonical={p.canonical_nzgd_id} merged={p.merged_nzgd_id} pairs={len(p.matched_pairs)}')
n_c, n_r = apply_merge_plan(conn, plan, run_id=1, table_cfg=CPT_TABLE_CONFIG)
print(f'merged: clusters={n_c} records={n_r}')
"
```

Expected: plan length 1; one fuzzy merge; `merged: clusters=1 records=1`.

- [ ] **Step 3: Commit**

```bash
git add nzgd/dedup/pass2_fuzzy.py
git commit -m "Add Pass 2: metadata-blocked fuzzy-match dedup plan generator"
```

---

## Task 13: CLI script

**Files:**
- Create: `nzgd/scripts/db/deduplicate.py`

- [ ] **Step 1: Write `deduplicate.py`**

Create `nzgd/scripts/db/deduplicate.py` with:

```python
"""CLI entry point for cross-record CPT/SPT deduplication.

Copies a source NZGD SQLite DB to a target path, then applies hash and fuzzy
deduplication passes to the copy. The source DB is never modified.
"""

import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

from nzgd import constants
from nzgd.dedup.data_types import CPT_TABLE_CONFIG, SPT_TABLE_CONFIG
from nzgd.dedup.executor import apply_merge_plan
from nzgd.dedup.pass1_hash import generate_hash_merge_plan
from nzgd.dedup.pass2_fuzzy import generate_fuzzy_merge_plan
from nzgd.dedup.reports import (
    write_calibration_report,
    write_dedup_report,
    write_failures_report,
)
from nzgd.dedup.schema import apply_dedup_schema


app = typer.Typer(help=__doc__)


def _script_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
            cwd=Path(__file__).resolve().parents[3],
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


@app.command()
def main(
    source: Path = typer.Option(..., "--source", help="Source SQLite DB (read-only)."),
    target: Path = typer.Option(None, "--target", help="Target deduped DB path. Defaults to '<source>_deduped.db'."),
    skip_cpt: bool = typer.Option(False, "--skip-cpt", help="Skip CPT deduplication."),
    skip_spt: bool = typer.Option(False, "--skip-spt", help="Skip SPT deduplication."),
) -> None:
    """Run the dedup pipeline against `source`, producing a deduped DB at `target`."""
    if target is None:
        suffix = constants.DEDUP_CONFIG["output"]["deduped_db_suffix"]
        target = source.with_name(source.stem + suffix + ".db")
    if target.exists():
        typer.echo(f"Target {target} already exists; refusing to overwrite. Delete it and rerun.", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Copying {source} → {target} ...")
    shutil.copyfile(source, target)

    conn = sqlite3.connect(target)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_dedup_schema(conn)

    config_snapshot = json.dumps(constants.DEDUP_CONFIG)
    started = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO dedup_run (started_at, source_db_path, script_version, config_snapshot_json) "
        "VALUES (?, ?, ?, ?)",
        (started, str(source), _script_version(), config_snapshot),
    )
    run_id = cur.lastrowid
    conn.commit()

    out_dir = target.parent
    fuzzy_thresholds = {
        **constants.DEDUP_CONFIG["fuzzy_pass"],
        "random_pair_sample_size": constants.DEDUP_CONFIG["calibration"]["random_pair_sample_size"],
    }

    all_failures: list[dict] = []
    total_clusters = 0
    total_records = 0

    for cfg, skip in ((CPT_TABLE_CONFIG, skip_cpt), (SPT_TABLE_CONFIG, skip_spt)):
        if skip:
            typer.echo(f"Skipping {cfg.record_type} per CLI flag.")
            continue
        typer.echo(f"[{cfg.record_type}] Pass 1: hash ...")
        hash_plan = generate_hash_merge_plan(conn, cfg)
        c1, r1 = apply_merge_plan(conn, hash_plan, run_id, cfg, failures=all_failures)
        typer.echo(f"[{cfg.record_type}] Pass 1: merged {r1} records across {c1} clusters.")

        typer.echo(f"[{cfg.record_type}] Pass 2: fuzzy ...")
        calibration: dict = {}
        fuzzy_plan = generate_fuzzy_merge_plan(conn, cfg, fuzzy_thresholds, calibration_collector=calibration)
        c2, r2 = apply_merge_plan(conn, fuzzy_plan, run_id, cfg, failures=all_failures)
        typer.echo(f"[{cfg.record_type}] Pass 2: merged {r2} records across {c2} clusters.")

        # Write a per-record-type calibration file when there's content
        if calibration.get("positive") or calibration.get("negative"):
            cal_path = out_dir / f"{cfg.record_type.lower()}_{constants.DEDUP_CONFIG['output']['calibration_report_filename']}"
            write_calibration_report(calibration.get("positive", []), calibration.get("negative", []), cal_path)

        total_clusters += c1 + c2
        total_records += r1 + r2

    finished = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE dedup_run SET finished_at = ?, n_clusters_merged = ?, n_records_merged = ? WHERE run_id = ?",
        (finished, total_clusters, total_records, run_id),
    )
    conn.commit()

    report_path = out_dir / constants.DEDUP_CONFIG["output"]["report_filename"]
    write_dedup_report(conn, run_id, report_path)

    if all_failures:
        failures_path = out_dir / constants.DEDUP_CONFIG["output"]["failures_filename"]
        write_failures_report(all_failures, failures_path)
        typer.echo(f"{len(all_failures)} cluster(s) failed; see {failures_path}")

    typer.echo(f"Done. Deduped DB at {target}. Report at {report_path}.")
    conn.close()


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Smoke check CLI on a synthetic copy of the production DB**

(Run only against a copy — the CLI writes to a target path next to the source by default. Make sure free disk allows the ~5 GB copy. This step also validates the script works on real data shape, not just the test fixture.)

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -c "
# Mini-smoke: just confirm the module imports and typer wires up.
from nzgd.scripts.db.deduplicate import app
print('module ok')
"
```

Expected: `module ok`.

- [ ] **Step 3: Commit**

```bash
git add nzgd/scripts/db/deduplicate.py
git commit -m "Add deduplicate.py CLI orchestrating both passes for CPT and SPT"
```

---

## Task 14: Additional CPT integration scenarios

**Files:**
- Modify: `tests/dedup/test_dedup_pipeline.py` (append tests)

- [ ] **Step 1: Append the additional CPT scenarios**

Append to `tests/dedup/test_dedup_pipeline.py`:

```python


def _run_both_passes(conn: sqlite3.Connection, cfg, thresholds) -> tuple[int, int]:
    run_id = _start_run(conn)
    from nzgd.dedup.pass2_fuzzy import generate_fuzzy_merge_plan
    hash_plan = generate_hash_merge_plan(conn, cfg)
    c1, r1 = apply_merge_plan(conn, hash_plan, run_id, cfg)
    fuzzy_plan = generate_fuzzy_merge_plan(conn, cfg, thresholds)
    c2, r2 = apply_merge_plan(conn, fuzzy_plan, run_id, cfg)
    return c1 + c2, r1 + r2


_DEFAULT_THRESHOLDS = {
    "spatial_radius_m": 50,
    "date_window_days": 90,
    "name_similarity_min": 80,
    "trace_score_max": 0.05,
    "trace_resample_step_m": 0.05,
}


def test_slight_perturbation_pair_is_merged_by_fuzzy(fresh_db: sqlite3.Connection) -> None:
    trace_a = [(d, 1.0 + 0.1 * d, 0.01 + 0.001 * d, 0.0) for d in [0.1, 0.2, 0.3, 0.4]]
    trace_b = [(d, 1.001 + 0.1 * d, 0.0101 + 0.001 * d, 0.0) for d in [0.1, 0.2, 0.3, 0.4]]
    add_cpt_record(fresh_db, 1, -41.0, 174.0, "Wellington Site A", "2024-01-01")
    add_cpt_record(fresh_db, 2, -41.0, 174.0001, "Wellington Site A", "2024-01-05")
    add_cpt_report(fresh_db, 10, 1, trace_a)
    add_cpt_report(fresh_db, 20, 2, trace_b)
    total_c, total_r = _run_both_passes(fresh_db, CPT_TABLE_CONFIG, _DEFAULT_THRESHOLDS)
    assert (total_c, total_r) == (1, 1)
    audit = fresh_db.execute("SELECT match_pass FROM dedup_audit").fetchall()
    assert audit == [("fuzzy",)]


def test_nearby_but_distinct_sites_are_not_merged(fresh_db: sqlite3.Connection) -> None:
    # Same lat/lon (within radius) but very different traces and names → no merge
    trace_a = [(d, 1.0, 0.01, 0.0) for d in [0.1, 0.2, 0.3]]
    trace_b = [(d, 50.0, 0.5, 0.0) for d in [0.1, 0.2, 0.3]]  # qc 50x larger
    add_cpt_record(fresh_db, 1, -41.0, 174.0, "Site Alpha", "2024-01-01")
    add_cpt_record(fresh_db, 2, -41.0, 174.0001, "Site Beta", "2024-06-01")
    add_cpt_report(fresh_db, 10, 1, trace_a)
    add_cpt_report(fresh_db, 20, 2, trace_b)
    total_c, _ = _run_both_passes(fresh_db, CPT_TABLE_CONFIG, _DEFAULT_THRESHOLDS)
    assert total_c == 0
    merged = fresh_db.execute("SELECT COUNT(*) FROM nzgdrecord WHERE merged_into_nzgd_id IS NOT NULL").fetchone()[0]
    assert merged == 0


def test_3way_transitive_cluster_via_hash(fresh_db: sqlite3.Connection) -> None:
    # nzgd 1 has reports r10, r11. nzgd 2 has r20 (matches r10). nzgd 3 has r30 (matches r11).
    trace_x = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    trace_y = [(0.1, 2.0, 0.02, 0.0), (0.2, 2.1, 0.021, 0.0)]
    add_cpt_record(fresh_db, 1, -41.0, 174.0, "Has both", "2024-01-01")
    add_cpt_record(fresh_db, 2, -41.0, 174.0, None, None)
    add_cpt_record(fresh_db, 3, -41.0, 174.0, None, None)
    add_cpt_report(fresh_db, 10, 1, trace_x)
    add_cpt_report(fresh_db, 11, 1, trace_y)
    add_cpt_report(fresh_db, 20, 2, trace_x)
    add_cpt_report(fresh_db, 30, 3, trace_y)
    run_id = _start_run(fresh_db)
    plan = generate_hash_merge_plan(fresh_db, CPT_TABLE_CONFIG)
    apply_merge_plan(fresh_db, plan, run_id, CPT_TABLE_CONFIG)
    audit = fresh_db.execute(
        "SELECT cluster_id, canonical_nzgd_id, merged_nzgd_id FROM dedup_audit ORDER BY merged_nzgd_id"
    ).fetchall()
    # Two rows, both sharing cluster_id, both pointing to canonical 1
    assert len(audit) == 2
    assert audit[0][0] == audit[1][0]  # same cluster_id
    assert audit[0][1] == 1 and audit[1][1] == 1
    assert {audit[0][2], audit[1][2]} == {2, 3}


def test_partial_overlap_reparents_unique_reports(fresh_db: sqlite3.Connection) -> None:
    # nzgd 1: r10 (shared), r11 (unique). nzgd 2: r20 (shared with r10), r21 (unique).
    shared = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    unique_1 = [(0.1, 5.0, 0.05, 0.0), (0.2, 5.1, 0.051, 0.0), (0.3, 5.2, 0.052, 0.0)]  # 3 rows
    unique_2 = [(0.1, 7.0, 0.07, 0.0)]  # 1 row
    add_cpt_record(fresh_db, 1)
    add_cpt_record(fresh_db, 2)
    add_cpt_report(fresh_db, 10, 1, shared)
    add_cpt_report(fresh_db, 11, 1, unique_1)
    add_cpt_report(fresh_db, 20, 2, shared)
    add_cpt_report(fresh_db, 21, 2, unique_2)
    run_id = _start_run(fresh_db)
    plan = generate_hash_merge_plan(fresh_db, CPT_TABLE_CONFIG)
    apply_merge_plan(fresh_db, plan, run_id, CPT_TABLE_CONFIG)
    # nzgd 1 has more unique data (3 rows) → canonical; nzgd 2 merged into 1.
    assert fresh_db.execute("SELECT merged_into_nzgd_id FROM nzgdrecord WHERE nzgd_id = 2").fetchone()[0] == 1
    # r20 deleted; r21 re-parented to nzgd 1.
    reports = dict(fresh_db.execute("SELECT cpt_id, nzgd_id FROM cptreport").fetchall())
    assert reports == {10: 1, 11: 1, 21: 1}
    # cptmeasurements for r21 still present (re-parent doesn't touch measurements)
    assert fresh_db.execute("SELECT COUNT(*) FROM cptmeasurements WHERE cpt_id = 21").fetchone()[0] == 1


def test_metadata_conflict_picks_one_and_records_conflict(fresh_db: sqlite3.Connection) -> None:
    trace = [(0.1, 1.0, 0.01, 0.0), (0.2, 1.1, 0.011, 0.0)]
    # nzgd 1: NULL investigation_name, but set investigation_date so its non-null
    # metadata count matches the other two records (each has investigation_name).
    # All three then tie on tiebreaker 1; smallest nzgd_id picks nzgd 1 as canonical.
    add_cpt_record(fresh_db, 1, investigation_name=None, investigation_date="2024-01-01")
    add_cpt_record(fresh_db, 2, investigation_name="Foo", investigation_date=None)
    add_cpt_record(fresh_db, 3, investigation_name="Bar", investigation_date=None)
    add_cpt_report(fresh_db, 10, 1, trace)
    add_cpt_report(fresh_db, 20, 2, trace)
    add_cpt_report(fresh_db, 30, 3, trace)
    run_id = _start_run(fresh_db)
    plan = generate_hash_merge_plan(fresh_db, CPT_TABLE_CONFIG)
    apply_merge_plan(fresh_db, plan, run_id, CPT_TABLE_CONFIG)
    # Canonical's investigation_name is now non-NULL (one of Foo/Bar).
    name = fresh_db.execute("SELECT original_investigation_name FROM nzgdrecord WHERE nzgd_id = 1").fetchone()[0]
    assert name in {"Foo", "Bar"}
    # Conflict recorded in at least one audit row.
    rows = fresh_db.execute(
        "SELECT metadata_copied_json, metadata_conflicts_json FROM dedup_audit"
    ).fetchall()
    found_conflict = False
    for copied_json, conflict_json in rows:
        if conflict_json and "original_investigation_name" in conflict_json:
            found_conflict = True
            data = json.loads(conflict_json)["original_investigation_name"]
            values = {entry["value"] for entry in data}
            assert {"Foo", "Bar"}.issubset(values)
    assert found_conflict


def test_predicate_accepts_when_date_or_name_is_null(fresh_db: sqlite3.Connection) -> None:
    trace_a = [(d, 1.0 + 0.1 * d, 0.01 + 0.001 * d, 0.0) for d in [0.1, 0.2, 0.3, 0.4]]
    trace_b = [(d, 1.001 + 0.1 * d, 0.0101 + 0.001 * d, 0.0) for d in [0.1, 0.2, 0.3, 0.4]]
    add_cpt_record(fresh_db, 1, -41.0, 174.0, None, None)  # both sentinels
    add_cpt_record(fresh_db, 2, -41.0, 174.0001, None, None)
    add_cpt_report(fresh_db, 10, 1, trace_a)
    add_cpt_report(fresh_db, 20, 2, trace_b)
    total_c, total_r = _run_both_passes(fresh_db, CPT_TABLE_CONFIG, _DEFAULT_THRESHOLDS)
    assert (total_c, total_r) == (1, 1)
```

- [ ] **Step 2: Run all CPT tests**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -v
```

Expected: all 7 tests pass (1 from Task 11 + 6 new).

- [ ] **Step 3: Commit**

```bash
git add tests/dedup/test_dedup_pipeline.py
git commit -m "Add CPT integration scenarios: fuzzy, no-merge, transitive, partial, conflict, sentinel"
```

---

## Task 15: SPT integration scenario with dependent tables

**Files:**
- Modify: `tests/dedup/test_dedup_pipeline.py` (append)

- [ ] **Step 1: Append the SPT scenario**

Append to `tests/dedup/test_dedup_pipeline.py`:

```python


def test_spt_cluster_cascades_dependent_table_deletes(fresh_db: sqlite3.Connection) -> None:
    trace = [(1.0, 5, 5, 5), (2.0, 7, 7, 7), (3.0, 10, 10, 10)]
    add_bh_record(fresh_db, 1)
    add_bh_record(fresh_db, 2)
    add_spt_report(fresh_db, 100, 1, trace)
    add_spt_report(fresh_db, 200, 2, trace)
    # Add dependents on the duplicate (spt_id 200) so we can verify cascade
    fresh_db.execute("INSERT INTO soilmeasurements (soil_measurement_id, spt_id, top_depth_m, bottom_depth_m) VALUES (1000, 200, 0.0, 1.0)")
    fresh_db.execute("INSERT INTO soilmeasurements (soil_measurement_id, spt_id, top_depth_m, bottom_depth_m) VALUES (1001, 200, 1.0, 2.0)")
    fresh_db.execute("INSERT INTO densitymeasurements (density_measurement_id, spt_id, top_depth_m, bottom_depth_m, density_keyword) VALUES (2000, 200, 0.0, 1.0, 'loose')")
    fresh_db.execute("INSERT INTO soilmeasurementsoiltype (soil_measurement_id, soil_type_id) VALUES (1000, 1), (1000, 2), (1001, 3)")
    # And one on the canonical to confirm we don't touch it
    fresh_db.execute("INSERT INTO soilmeasurements (soil_measurement_id, spt_id, top_depth_m, bottom_depth_m) VALUES (1002, 100, 0.0, 1.0)")
    fresh_db.execute("INSERT INTO soilmeasurementsoiltype (soil_measurement_id, soil_type_id) VALUES (1002, 1)")

    run_id = _start_run(fresh_db)
    plan = generate_hash_merge_plan(fresh_db, SPT_TABLE_CONFIG)
    apply_merge_plan(fresh_db, plan, run_id, SPT_TABLE_CONFIG)

    # spt_id 200 and all its dependents are gone
    assert fresh_db.execute("SELECT COUNT(*) FROM sptreport WHERE spt_id = 200").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM sptmeasurements WHERE spt_id = 200").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM soilmeasurements WHERE spt_id = 200").fetchone()[0] == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM densitymeasurements WHERE spt_id = 200").fetchone()[0] == 0
    assert fresh_db.execute(
        "SELECT COUNT(*) FROM soilmeasurementsoiltype WHERE soil_measurement_id IN (1000, 1001)"
    ).fetchone()[0] == 0
    # Canonical untouched
    assert fresh_db.execute("SELECT COUNT(*) FROM soilmeasurements WHERE spt_id = 100").fetchone()[0] == 1
    assert fresh_db.execute(
        "SELECT COUNT(*) FROM soilmeasurementsoiltype WHERE soil_measurement_id = 1002"
    ).fetchone()[0] == 1
    # Merge recorded with record_type = 'BH'
    assert fresh_db.execute("SELECT record_type FROM dedup_audit").fetchall() == [("BH",)]
```

- [ ] **Step 2: Run all tests**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/dedup/test_dedup_pipeline.py -v
```

Expected: 8 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/dedup/test_dedup_pipeline.py
git commit -m "Add SPT integration scenario verifying dependent-table cascade"
```

---

## Task 16: Real-data validation run

**Files:** none (manual command + observation)

- [ ] **Step 1: Run dedup against the production DB**

```bash
cd /home/arr65/src/nzgd && /home/arr65/venvs/dev_nzgd_venv/bin/python -m nzgd.scripts.db.deduplicate \
    --source /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p6p0_20260403.db
```

Expected: progress bars; final summary `Done. Deduped DB at ..._deduped.db. Report at ...dedup_report.csv.`

Approximate runtime expectation: hash pass scans 82.8M cptmeasurements rows + 280k sptmeasurements rows; fuzzy pass queries a BallTree on ~50k canonicals and computes RMSE on candidate pairs. Total order-of-magnitude: tens of minutes.

- [ ] **Step 2: Inspect the dedup report**

```bash
head -20 /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/dedup_report.csv
wc -l /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/dedup_report.csv
```

Spot-check 5–10 reported merges:
- For each, query both nzgd_ids' source_file from cptreport in the *source* DB and confirm the data plausibly comes from the same physical investigation (same site name, same approximate date, similar source-file naming).
- If false-positive rate >~1%, flag as the trigger condition described in the spec's deferred-scope section.

- [ ] **Step 3: Inspect the calibration report and tune thresholds if needed**

```bash
ls /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/*calibration*.csv
```

Open the calibration report(s) in pandas:
```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
import pandas as pd
df = pd.read_csv('/home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/cpt_calibration_report.csv')
print(df.groupby('group').describe())
"
```

If the positive (hash-match) distribution and negative (random) distribution overlap heavily on any feature, adjust the corresponding threshold in `nzgd/resources/config.yaml`, delete the target DB, and re-run.

- [ ] **Step 4: No commit unless thresholds were adjusted**

If `config.yaml` thresholds were tuned during step 3, commit only that change:

```bash
git add nzgd/resources/config.yaml
git commit -m "Tune fuzzy-pass thresholds based on real-data calibration"
```

---

## Notes

- **Idempotency of re-runs:** the script refuses to overwrite an existing target DB, by design. To re-run with different thresholds, delete the target first. The source DB is read-only.
- **`PRAGMA foreign_keys = ON`:** issued by the script before any DDL. Tests that bypass the CLI must also enable it; the test fixture does so already.
- **`dedup_run.config_snapshot_json`:** dumped as a JSON string of the entire `DEDUP_CONFIG` dict, so every audit row can be traced back to the exact thresholds in effect.
