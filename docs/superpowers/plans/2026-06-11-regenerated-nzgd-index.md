# Regenerated NZGD Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the frozen `nzgd_metadata_from_coordinates_22_august_2025.csv` with a regenerated, git-tracked `nzgd_index.csv.gz` rebuilt from the API investigation catalog, and correct four inaccurate column names across the repo.

**Architecture:** A build script derives the index from `current_nzgd_investigation_catalog.csv.gz` plus an append-only location sidecar (LINZ shapefile point-in-polygon for new IDs only) plus fresh GeoTIFF raster samples. All consumers already load via `config.yaml`'s `nzgd_index_file_name`, so the switch is a config flip plus two script edits. A one-time verification script proves no information is lost before legacy artifacts are deleted.

**Tech Stack:** Python 3, pandas, geopandas (sjoin), rasterio (sample), qcore (NZTM transform), pytest.

**Spec:** `docs/superpowers/specs/2026-06-11-regenerated-nzgd-index-design.md` (read it first).

> **As-built note (executed 2026-06-22 on branch `feat/regenerated-nzgd-index`).**
> Two reviewed deviations from the code listings below:
> 1. **Task 6** — the seed's `KNOWN_CORRECTIONS` blanks **both 229630 and 229545**
>    to `unclassified`. 229545 is a 0,0 null-island CPT test record
>    (`ISAB-124bbbf444`) with a fabricated Dunsandel classification, surfaced by
>    the out-of-bbox report and confirmed by the user; 8753 (a real record with
>    corrupted coordinates but a plausible classification) was deliberately
>    preserved. Commit `5a66c68`.
> 2. **Task 7** — the verification uses `SEED_CORRECTED_IDS = {229630, 229545}`,
>    and **Check 5 (NZTM allclose) excludes `KNOWN_DIVERGENT_IDS`** so reassigned
>    IDs (229775, 229822), whose upstream coordinates changed to 0,0, don't trip
>    it. The Check 5 listing below predates that fix. Commit `1f02f7c`.

---

## Context for the implementer

- Repo root: `/home/arr65/src/nzgd`. Work on branch `feat/regenerated-nzgd-index` (already created).
- Python: `/home/arr65/venvs/dev_nzgd_venv/bin/python` (has the nzgd package installed editable, plus geopandas 1.1.1, rasterio 1.4.3, qcore). Run pytest as `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest`.
- `nzgd/constants.py` loads `nzgd/resources/config.yaml` at import time; `constants.RESOURCE_PATH` is `nzgd/resources/`.
- Ground-truth catalog: `nzgd/resources/nzgd_catalogs_from_api/current_nzgd_investigation_catalog.csv.gz` — 197,649 rows, ID column named `Id`.
- Legacy index: `nzgd/resources/nzgd_metadata_from_coordinates_22_august_2025.csv` — 197,647 rows, ID column named `nzgd_id`. Do not modify it; it is deleted only in Task 12 after verification.
- Shapefiles (verified field names):
  - `/home/arr65/data/nzgd/resources/shapefiles/lds-nz-land-districts-SHP_WGS84_EPSG_4326/nz-land-districts.shp` — fields `id`, `name` (CRS EPSG:4326).
  - `/home/arr65/data/nzgd/resources/shapefiles/lds-nz-suburbs-and-localities-SHP_WGS84_EPSG_4326/nz-suburbs-and-localities.shp` — fields include `territoria`, `major_na_2`, `name_ascii` (CRS EPSG:4326).
  - Mapping (from the legacy classification code): district-shapefile `name` → `region`; suburbs `territoria` → `district`, `major_na_2` → `city`, `name_ascii` → `suburb`.
- GeoTIFF paths are already config keys: `westerhoff_2018_model_path`, `nlm_gwd_path`, `nlm_gw_std_path`, `foster_2019_vs30_model_path` (Foster has 2 bands: band 1 Vs30 in m/s, band 2 natural-log sigma).
- `qcore.coordinates.wgs_depth_to_nztm(array_of_[lat, lon])` returns an array of `[northing, easting]` = `[nztm_y, nztm_x]`.
- NEVER call the NZGD API. NEVER modify NZGD source files under `/home/arr65/data/nzgd/downloads/`.
- Ruff is configured (numpy docstrings, type annotations on args, naming). Follow it.
- Commit after every task; messages end with the project's usual co-author trailer.

## File map

| File | Action | Responsibility |
| --- | --- | --- |
| `nzgd/metadata/__init__.py` | create | package marker |
| `nzgd/metadata/io_utils.py` | create | atomic CSV-gz writes |
| `nzgd/metadata/location.py` | create | point-in-polygon classification, name sanitization, NZ bbox |
| `nzgd/metadata/rasters.py` | create | GeoTIFF sampling with nodata→NaN |
| `nzgd/metadata/build.py` | create | pure build logic: load, sidecar update, nztm, assemble, guards |
| `nzgd/scripts/metadata/build_nzgd_index.py` | create | orchestrator CLI |
| `nzgd/scripts/metadata/one_time/seed_location_sidecar_from_legacy_index.py` | create | one-time sidecar seed |
| `nzgd/scripts/metadata/one_time/verify_new_index_against_legacy.py` | create | one-time no-loss proof |
| `tests/metadata/test_*.py` | create | unit tests |
| `nzgd/resources/config.yaml` | modify | shapefile keys; later index-name flip + plausibility key renames |
| `nzgd/constants.py` | modify | shapefile path constants |
| `nzgd/db/orm.py`, `nzgd/dedup/selection.py`, `nzgd/dedup/executor.py`, `tests/dedup/conftest.py` | modify | column renames |
| `nzgd/scripts/db/put_nzgd_metadata.py` | modify | read model columns from index; renames |
| `nzgd/scripts/metadata/make_metadata_summary_csv.py` | modify | repoint input; renames |
| legacy index + 2 scripts + gitignored intermediates | delete | Task 12, after verification |
| `/home/arr65/src/api_nzgd/README_to_download_nzgd_updates.md` | modify | Stage 4 docs (separate repo) |

---

### Task 1: metadata package + atomic writes

**Files:**
- Create: `nzgd/metadata/__init__.py`
- Create: `nzgd/metadata/io_utils.py`
- Test: `tests/metadata/__init__.py` (empty), `tests/metadata/test_io_utils.py`

- [ ] **Step 1: Write the failing test**

`tests/metadata/__init__.py`: empty file. `tests/metadata/test_io_utils.py`:

```python
"""Tests for atomic CSV-gz writing."""

import pandas as pd

from nzgd.metadata.io_utils import atomic_write_csv_gz


def test_atomic_write_csv_gz_roundtrip(tmp_path):
    """A written file reads back equal to the input frame."""
    df = pd.DataFrame({"nzgd_id": [1, 2], "region": ["Canterbury", "unclassified"]})
    out = tmp_path / "x.csv.gz"
    atomic_write_csv_gz(df, out)
    assert pd.read_csv(out).equals(df)


def test_atomic_write_csv_gz_overwrites(tmp_path):
    """Writing over an existing file replaces it, leaving no temp file behind."""
    out = tmp_path / "x.csv.gz"
    atomic_write_csv_gz(pd.DataFrame({"a": [1]}), out)
    atomic_write_csv_gz(pd.DataFrame({"a": [2]}), out)
    assert pd.read_csv(out)["a"].tolist() == [2]
    assert list(tmp_path.iterdir()) == [out]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/metadata/test_io_utils.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'nzgd.metadata'`

- [ ] **Step 3: Write implementation**

`nzgd/metadata/__init__.py`:

```python
"""Locally computed NZGD index metadata: location classification, raster sampling, index build."""
```

`nzgd/metadata/io_utils.py`:

```python
"""Atomic file-writing helpers for index artifacts."""

import os
from pathlib import Path

import pandas as pd


def atomic_write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to gzipped CSV atomically (temp file + rename).

    Parameters
    ----------
    df : pd.DataFrame
        Frame to write (index is not written).
    path : Path
        Destination path ending in .csv.gz.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    df.to_csv(tmp_path, index=False, compression="gzip")
    os.replace(tmp_path, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/metadata/test_io_utils.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add nzgd/metadata/ tests/metadata/
git commit -m "feat: add nzgd.metadata package with atomic csv.gz writer"
```

---

### Task 2: location classification module

**Files:**
- Create: `nzgd/metadata/location.py`
- Test: `tests/metadata/test_location.py`

- [ ] **Step 1: Write the failing test**

`tests/metadata/test_location.py`:

```python
"""Tests for point-in-polygon location classification."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from nzgd.metadata.location import classify_locations, coords_outside_nz, replace_chars


def _square(lon0: float, lat0: float, size: float = 1.0) -> Polygon:
    return Polygon(
        [(lon0, lat0), (lon0 + size, lat0), (lon0 + size, lat0 + size), (lon0, lat0 + size)]
    )


def _toy_gdfs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    district_gdf = gpd.GeoDataFrame(
        {"name": ["Canterbury Land District"]},
        geometry=[_square(172.0, -44.0)],
        crs="EPSG:4326",
    )
    suburbs_gdf = gpd.GeoDataFrame(
        {
            "territoria": ["Christchurch City"],
            "major_na_2": ["Christchurch"],
            "name_ascii": ["Aranui"],
        },
        geometry=[_square(172.0, -44.0)],
        crs="EPSG:4326",
    )
    return district_gdf, suburbs_gdf


def test_classify_point_inside_polygons():
    """A point inside both polygons gets all four names, sanitized."""
    district_gdf, suburbs_gdf = _toy_gdfs()
    points = pd.DataFrame({"nzgd_id": [10], "Latitude": [-43.5], "Longitude": [172.5]})
    out = classify_locations(points, district_gdf, suburbs_gdf)
    row = out.iloc[0]
    assert row["region"] == "Canterbury_Land_District"
    assert row["district"] == "Christchurch_City"
    assert row["city"] == "Christchurch"
    assert row["suburb"] == "Aranui"


def test_classify_point_outside_is_unclassified():
    """Points outside the polygons (or with NaN coords) become unclassified."""
    district_gdf, suburbs_gdf = _toy_gdfs()
    points = pd.DataFrame(
        {"nzgd_id": [1, 2], "Latitude": [12.123, None], "Longitude": [34.123, None]}
    )
    out = classify_locations(points, district_gdf, suburbs_gdf)
    assert (out[["region", "district", "city", "suburb"]] == "unclassified").all().all()
    assert out["nzgd_id"].tolist() == [1, 2]


def test_replace_chars():
    """Spaces, apostrophes, commas, and slashes become underscores."""
    assert replace_chars("Hawke's Bay, NZ/Aotearoa") == "Hawke_s_Bay__NZ_Aotearoa"


def test_coords_outside_nz():
    """Out-of-bbox coords are flagged; NZ coords and NaN coords are not."""
    df = pd.DataFrame(
        {"Latitude": [-43.5, 12.123, 0.0, None], "Longitude": [172.6, 34.123, 0.0, None]}
    )
    assert coords_outside_nz(df).tolist() == [False, True, True, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/metadata/test_location.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError` (no `nzgd.metadata.location`)

- [ ] **Step 3: Write implementation**

`nzgd/metadata/location.py`:

```python
"""Classify NZGD record locations against local LINZ shapefiles.

Port of the legacy `find_region` logic from the retired nzgd_data_extraction
repo, vectorized with a single spatial join instead of a per-row pool.
"""

import re

import geopandas as gpd
import pandas as pd

LOCATION_COLUMNS = ["region", "district", "city", "suburb"]
UNCLASSIFIED = "unclassified"

# All of New Zealand has negative latitude; anything outside this box is suspect.
NZ_LAT_BOUNDS = (-48.5, -33.5)
NZ_LON_BOUNDS = (165.0, 180.0)


def replace_chars(old_string: str) -> str:
    """Replace space, apostrophe, comma, and slash with underscores.

    Matches the sanitization used when the legacy classifications were built,
    so new classifications stay vocabulary-compatible with seeded ones.
    """
    return re.sub(r"[ ',/]", "_", old_string)


def coords_outside_nz(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask of rows whose Latitude/Longitude fall outside NZ.

    Rows with missing coordinates are not flagged (NaN comparisons are False).
    """
    return (
        (df["Latitude"] < NZ_LAT_BOUNDS[0])
        | (df["Latitude"] > NZ_LAT_BOUNDS[1])
        | (df["Longitude"] < NZ_LON_BOUNDS[0])
        | (df["Longitude"] > NZ_LON_BOUNDS[1])
    )


def classify_locations(
    points_df: pd.DataFrame,
    district_gdf: gpd.GeoDataFrame,
    suburbs_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Classify points into region/district/city/suburb by point-in-polygon.

    Parameters
    ----------
    points_df : pd.DataFrame
        Columns `nzgd_id`, `Latitude`, `Longitude` (WGS84). NaN or
        out-of-polygon coordinates classify as "unclassified".
    district_gdf : gpd.GeoDataFrame
        LINZ land districts; field `name` becomes `region`.
    suburbs_gdf : gpd.GeoDataFrame
        LINZ suburbs and localities; `territoria` becomes `district`,
        `major_na_2` becomes `city`, `name_ascii` becomes `suburb`.

    Returns
    -------
    pd.DataFrame
        One row per input row: `nzgd_id` plus the four sanitized location
        columns, in input order.
    """
    points = gpd.GeoDataFrame(
        points_df[["nzgd_id"]].copy(),
        geometry=gpd.points_from_xy(points_df["Longitude"], points_df["Latitude"]),
        crs="EPSG:4326",
    )

    district_hits = gpd.sjoin(
        points, district_gdf[["name", "geometry"]], how="left", predicate="within"
    ).drop_duplicates(subset="nzgd_id")
    suburb_hits = gpd.sjoin(
        points,
        suburbs_gdf[["territoria", "major_na_2", "name_ascii", "geometry"]],
        how="left",
        predicate="within",
    ).drop_duplicates(subset="nzgd_id")

    out = points_df[["nzgd_id"]].copy()
    out = out.merge(
        district_hits[["nzgd_id", "name"]].rename(columns={"name": "region"}),
        on="nzgd_id",
        how="left",
    )
    out = out.merge(
        suburb_hits[["nzgd_id", "territoria", "major_na_2", "name_ascii"]].rename(
            columns={"territoria": "district", "major_na_2": "city", "name_ascii": "suburb"}
        ),
        on="nzgd_id",
        how="left",
    )
    for col in LOCATION_COLUMNS:
        out[col] = out[col].fillna(UNCLASSIFIED).astype(str).map(replace_chars)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/metadata/test_location.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add nzgd/metadata/location.py tests/metadata/test_location.py
git commit -m "feat: shapefile-based location classification for index build"
```

---

### Task 3: raster sampling module

**Files:**
- Create: `nzgd/metadata/rasters.py`
- Test: `tests/metadata/test_rasters.py`

- [ ] **Step 1: Write the failing test**

`tests/metadata/test_rasters.py`:

```python
"""Tests for GeoTIFF sampling with nodata handling."""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from nzgd.metadata.rasters import sample_band


def _write_tif(path, nodata=-32767.0):
    """A 10x10 EPSG:2193 raster covering x 1000-2000, y 5000-6000, value 7.0.

    Cell (0, 0) (top-left, i.e. x 1000-1100, y 5900-6000) is nodata.
    """
    data = np.full((10, 10), 7.0, dtype="float32")
    data[0, 0] = nodata
    transform = from_origin(1000.0, 6000.0, 100.0, 100.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:2193",
        transform=transform,
        nodata=nodata,
    ) as ds:
        ds.write(data, 1)


def test_sample_band_inside_nodata_and_outside(tmp_path):
    """In-bounds cells return values; nodata and out-of-bounds return NaN."""
    tif = tmp_path / "t.tif"
    _write_tif(tif)
    xy = [(1550.0, 5550.0), (1050.0, 5950.0), (999999.0, 999999.0)]
    values = sample_band(tif, xy, band=1)
    assert values[0] == 7.0
    assert np.isnan(values[1])
    assert np.isnan(values[2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/metadata/test_rasters.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError` (no `nzgd.metadata.rasters`)

- [ ] **Step 3: Write implementation**

`nzgd/metadata/rasters.py`:

```python
"""Sample model GeoTIFFs at NZTM points, mapping nodata/out-of-bounds to NaN.

Replaces the duplicated sampling code formerly in put_nzgd_metadata.py. The
legacy index baked nodata sentinels (e.g. -32767) into its model columns;
this module never lets a sentinel through.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from nzgd import constants

MODEL_COLUMNS = [
    "model_gwl_westerhoff_2018_m",
    "model_gwl_nlm_2025_m",
    "model_gwl_nlm_2025_stddev_m",
    "model_vs30_foster_2019_m_per_s",
    "model_vs30_stddev_foster_2019_ln",
]


def sample_band(raster_path: Path, xy_pairs: list, band: int = 1) -> list:
    """Sample one raster band at (easting, northing) points.

    Parameters
    ----------
    raster_path : Path
        GeoTIFF path (CRS must match the point coordinates, EPSG:2193 here).
    xy_pairs : list
        Sequence of (x, y) = (easting, northing) tuples with finite values.
    band : int
        1-based band index.

    Returns
    -------
    list
        One float per point; NaN for nodata, masked, or out-of-bounds samples.
    """
    values = []
    with rasterio.open(raster_path) as ds:
        nodata = ds.nodatavals[band - 1]
        for sample in ds.sample(xy_pairs, indexes=[band], masked=True):
            value = sample[0]
            if np.ma.is_masked(value):
                values.append(np.nan)
                continue
            value = float(value)
            if not np.isfinite(value) or (nodata is not None and value == nodata):
                values.append(np.nan)
            else:
                values.append(value)
    return values


def sample_model_columns(nztm_df: pd.DataFrame) -> pd.DataFrame:
    """Sample all five model columns for a frame with `nztm_x`/`nztm_y`.

    Rows with missing NZTM coordinates get NaN in every model column. Raster
    paths come from `nzgd.constants` (config.yaml).

    Returns
    -------
    pd.DataFrame
        The five MODEL_COLUMNS, aligned to `nztm_df.index`.
    """
    out = pd.DataFrame(index=nztm_df.index, columns=MODEL_COLUMNS, dtype=float)
    valid = nztm_df["nztm_x"].notna() & nztm_df["nztm_y"].notna()
    if not valid.any():
        return out
    xy = list(zip(nztm_df.loc[valid, "nztm_x"], nztm_df.loc[valid, "nztm_y"]))
    out.loc[valid, "model_gwl_westerhoff_2018_m"] = sample_band(
        constants.WESTERHOFF_2018_MODEL_PATH, xy, band=1
    )
    out.loc[valid, "model_gwl_nlm_2025_m"] = sample_band(constants.NLM_GWD_PATH, xy, band=1)
    out.loc[valid, "model_gwl_nlm_2025_stddev_m"] = sample_band(
        constants.NLM_GW_STD_PATH, xy, band=1
    )
    out.loc[valid, "model_vs30_foster_2019_m_per_s"] = sample_band(
        constants.FOSTER_2019_VS30_MODEL_PATH, xy, band=1
    )
    out.loc[valid, "model_vs30_stddev_foster_2019_ln"] = sample_band(
        constants.FOSTER_2019_VS30_MODEL_PATH, xy, band=2
    )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/metadata/test_rasters.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add nzgd/metadata/rasters.py tests/metadata/test_rasters.py
git commit -m "feat: raster sampling with nodata->NaN for index build"
```

---

### Task 4: pure build logic

**Files:**
- Create: `nzgd/metadata/build.py`
- Test: `tests/metadata/test_build.py`

- [ ] **Step 1: Write the failing test**

`tests/metadata/test_build.py`:

```python
"""Tests for the pure index-build logic."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from nzgd.metadata import build


def _square(lon0: float, lat0: float, size: float = 1.0) -> Polygon:
    return Polygon(
        [(lon0, lat0), (lon0 + size, lat0), (lon0 + size, lat0 + size), (lon0, lat0 + size)]
    )


def _toy_gdfs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    district_gdf = gpd.GeoDataFrame(
        {"name": ["Canterbury Land District"]}, geometry=[_square(172.0, -44.0)], crs="EPSG:4326"
    )
    suburbs_gdf = gpd.GeoDataFrame(
        {
            "territoria": ["Christchurch City"],
            "major_na_2": ["Christchurch"],
            "name_ascii": ["Aranui"],
        },
        geometry=[_square(172.0, -44.0)],
        crs="EPSG:4326",
    )
    return district_gdf, suburbs_gdf


def _catalog(ids: list) -> pd.DataFrame:
    df = pd.DataFrame({"nzgd_id": ids})
    for col in build.RAW_CATALOG_COLUMNS:
        if col != "nzgd_id":
            df[col] = None
    df["Latitude"] = -43.5
    df["Longitude"] = 172.5
    return df


def test_load_catalog_rejects_duplicate_ids(tmp_path):
    """Duplicate Id values in the catalog abort the build."""
    path = tmp_path / "cat.csv.gz"
    pd.DataFrame({"Id": [1, 1], "Type": ["SCP", "SCP"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        build.load_catalog(path)


def test_check_id_superset():
    """A disappearing ID raises; pure additions pass."""
    build.check_id_superset(pd.Series([1, 2]), pd.Series([1, 2, 3]))
    with pytest.raises(RuntimeError, match="disappear"):
        build.check_id_superset(pd.Series([1, 2]), pd.Series([1, 3]))


def test_update_sidecar_appends_only_missing_ids():
    """Existing rows are untouched; only missing IDs get classified."""
    district_gdf, suburbs_gdf = _toy_gdfs()
    sidecar = pd.DataFrame(
        {
            "nzgd_id": [1],
            "region": ["ManualRegion"],
            "district": ["ManualDistrict"],
            "city": ["ManualCity"],
            "suburb": ["ManualSuburb"],
        }
    )
    combined, new_rows = build.update_sidecar(sidecar, _catalog([1, 2]), district_gdf, suburbs_gdf)
    assert new_rows["nzgd_id"].tolist() == [2]
    assert combined["nzgd_id"].tolist() == [1, 2]
    # ID 1's manual values survive even though classification would differ.
    assert combined.loc[combined["nzgd_id"] == 1, "region"].item() == "ManualRegion"
    assert combined.loc[combined["nzgd_id"] == 2, "region"].item() == "Canterbury_Land_District"


def test_assemble_index_columns_and_rows():
    """The assembled index has exactly INDEX_COLUMNS, one row per catalog row."""
    catalog = _catalog([1, 2])
    sidecar = pd.DataFrame(
        {
            "nzgd_id": [1, 2],
            "region": ["a", "a"],
            "district": ["b", "b"],
            "city": ["c", "c"],
            "suburb": ["d", "d"],
        }
    )
    nztm = pd.DataFrame({"nztm_y": [5.0, 6.0], "nztm_x": [1.0, 2.0]}, index=catalog.index)
    models = pd.DataFrame(
        {col: [0.1, 0.2] for col in build.MODEL_COLUMNS}, index=catalog.index
    )
    out = build.assemble_index(catalog, sidecar, nztm, models)
    assert list(out.columns) == build.INDEX_COLUMNS
    assert len(out) == 2


def test_nan_aware_neq():
    """NaN == NaN; NaN vs value differs; equal values match."""
    a = pd.Series([1.0, None, 3.0])
    b = pd.Series([1.0, None, 4.0])
    assert build.nan_aware_neq(a, b).tolist() == [False, False, True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/metadata/test_build.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError` (no `nzgd.metadata.build`)

- [ ] **Step 3: Write implementation**

`nzgd/metadata/build.py`:

```python
"""Pure logic for building nzgd_index.csv.gz from the API investigation catalog."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from qcore import coordinates

from nzgd.metadata.location import LOCATION_COLUMNS, classify_locations
from nzgd.metadata.rasters import MODEL_COLUMNS

# The 26 catalog columns, with Id renamed to nzgd_id, in catalog order.
RAW_CATALOG_COLUMNS = [
    "nzgd_id", "State", "InvestigationId", "Type", "TypeDisplay",
    "Latitude", "Longitude", "Northings", "Eastings", "EpsgCode",
    "FinalDepth", "GroundLevel", "MethodOfGroundLevel",
    "MethodOfGroundLevelDisplay", "HasGroundImprovementConducted",
    "HasReportedIssues", "ElevationDatum", "ElevationDatumDisplay",
    "Remarks", "LocationDescription", "MethodOfLocation",
    "MethodOfLocationDisplay", "ProjectId", "EndDate", "CreatedOn",
    "LastModifiedOn",
]

NZTM_COLUMNS = ["nztm_y", "nztm_x"]
INDEX_COLUMNS = RAW_CATALOG_COLUMNS + LOCATION_COLUMNS + NZTM_COLUMNS + MODEL_COLUMNS


def load_catalog(path: Path) -> pd.DataFrame:
    """Load the investigation catalog, rename Id to nzgd_id, reject duplicates."""
    df = pd.read_csv(path, low_memory=False).rename(columns={"Id": "nzgd_id"})
    duplicates = sorted(df.loc[df["nzgd_id"].duplicated(), "nzgd_id"].unique())
    if duplicates:
        raise ValueError(f"catalog has duplicate nzgd_id values: {duplicates[:20]}")
    return df


def compute_nztm(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """NZTM northing/easting for rows with coordinates; NaN otherwise."""
    out = pd.DataFrame(index=catalog_df.index, columns=NZTM_COLUMNS, dtype=float)
    valid = catalog_df["Latitude"].notna() & catalog_df["Longitude"].notna()
    if valid.any():
        northing_easting = coordinates.wgs_depth_to_nztm(
            catalog_df.loc[valid, ["Latitude", "Longitude"]].values
        )
        out.loc[valid, "nztm_y"] = northing_easting[:, 0]
        out.loc[valid, "nztm_x"] = northing_easting[:, 1]
    return out


def update_sidecar(
    sidecar_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    district_gdf: gpd.GeoDataFrame,
    suburbs_gdf: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify catalog IDs missing from the sidecar; never touch existing rows.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (updated sidecar sorted by nzgd_id, the newly classified rows).
    """
    missing = catalog_df.loc[
        ~catalog_df["nzgd_id"].isin(sidecar_df["nzgd_id"]),
        ["nzgd_id", "Latitude", "Longitude"],
    ]
    if missing.empty:
        empty = pd.DataFrame(columns=["nzgd_id", *LOCATION_COLUMNS])
        return sidecar_df, empty
    new_rows = classify_locations(missing, district_gdf, suburbs_gdf)
    combined = (
        pd.concat([sidecar_df, new_rows], ignore_index=True)
        .sort_values("nzgd_id")
        .reset_index(drop=True)
    )
    before = sidecar_df.sort_values("nzgd_id").reset_index(drop=True)
    after = (
        combined[combined["nzgd_id"].isin(before["nzgd_id"])].reset_index(drop=True)
    )
    if not after.equals(before):
        raise RuntimeError("append-only violation: an existing sidecar row changed")
    return combined, new_rows


def check_id_superset(old_ids: pd.Series, new_ids: pd.Series) -> None:
    """Raise if any previously indexed ID would disappear from the new index."""
    missing = sorted(set(old_ids) - set(new_ids))
    if missing:
        raise RuntimeError(
            f"{len(missing)} previously indexed nzgd_id values would disappear "
            f"(preservation invariant violated): {missing[:20]}"
        )


def assemble_index(
    catalog_df: pd.DataFrame,
    sidecar_df: pd.DataFrame,
    nztm_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join catalog + location + nztm + model columns into INDEX_COLUMNS order."""
    out = catalog_df.merge(sidecar_df, on="nzgd_id", how="left", validate="one_to_one")
    out = pd.concat([out, nztm_df, model_df], axis=1)
    return out[INDEX_COLUMNS]


def nan_aware_neq(a: pd.Series, b: pd.Series) -> pd.Series:
    """Elementwise inequality where NaN equals NaN."""
    return (a != b) & ~(a.isna() & b.isna())


def diff_raw_columns(prev_index_df: pd.DataFrame, new_index_df: pd.DataFrame) -> dict:
    """Per-column changed-ID lists for the raw catalog columns, common IDs only."""
    merged = prev_index_df.merge(
        new_index_df, on="nzgd_id", suffixes=("_prev", "_new"), how="inner"
    )
    changed = {}
    for col in RAW_CATALOG_COLUMNS:
        if col == "nzgd_id":
            continue
        mask = nan_aware_neq(merged[f"{col}_prev"], merged[f"{col}_new"])
        if mask.any():
            changed[col] = merged.loc[mask, "nzgd_id"].tolist()
    return changed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/metadata/test_build.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add nzgd/metadata/build.py tests/metadata/test_build.py
git commit -m "feat: pure build logic for regenerated NZGD index"
```

---

### Task 5: config keys, constants, and the build script

**Files:**
- Modify: `nzgd/resources/config.yaml` (after the `foster_2019_vs30_model_path` line, ~line 48)
- Modify: `nzgd/constants.py` (after `FOSTER_2019_VS30_MODEL_PATH`, ~line 59)
- Modify: `requirements.txt` (only if geopandas absent)
- Create: `nzgd/scripts/metadata/build_nzgd_index.py`

- [ ] **Step 1: Add config keys**

In `nzgd/resources/config.yaml`, directly below the `foster_2019_vs30_model_path` line, add:

```yaml
# Paths to LINZ shapefiles for region/district/city/suburb classification
district_shapefile_path: "/home/arr65/data/nzgd/resources/shapefiles/lds-nz-land-districts-SHP_WGS84_EPSG_4326/nz-land-districts.shp"
suburbs_shapefile_path: "/home/arr65/data/nzgd/resources/shapefiles/lds-nz-suburbs-and-localities-SHP_WGS84_EPSG_4326/nz-suburbs-and-localities.shp"
```

- [ ] **Step 2: Add constants**

In `nzgd/constants.py`, directly below the `FOSTER_2019_VS30_MODEL_PATH` block, add:

```python
# Paths to LINZ shapefiles for location classification
DISTRICT_SHAPEFILE_PATH = Path(CONFIG["district_shapefile_path"])
SUBURBS_SHAPEFILE_PATH = Path(CONFIG["suburbs_shapefile_path"])
```

- [ ] **Step 3: Ensure geopandas is a declared dependency**

Run: `grep -i geopandas /home/arr65/src/nzgd/requirements.txt || echo "geopandas" >> /home/arr65/src/nzgd/requirements.txt`
Expected: either an existing line is printed, or the append happens silently.

- [ ] **Step 4: Verify constants resolve**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -c "from nzgd import constants; print(constants.DISTRICT_SHAPEFILE_PATH.exists(), constants.SUBURBS_SHAPEFILE_PATH.exists())"`
Expected: `True True`

- [ ] **Step 5: Write the build script**

`nzgd/scripts/metadata/build_nzgd_index.py`:

```python
"""Rebuild nzgd_index.csv.gz from the current API investigation catalog.

Run after each NZGD sync (Stage 4 of the update workflow):

    python nzgd/scripts/metadata/build_nzgd_index.py

Inputs are all local: the catalog, the location sidecar, LINZ shapefiles,
and the model GeoTIFFs. No NZGD API calls are made.
"""

import sys

import geopandas as gpd
import pandas as pd

from nzgd import constants
from nzgd.metadata import build
from nzgd.metadata.io_utils import atomic_write_csv_gz
from nzgd.metadata.location import LOCATION_COLUMNS, UNCLASSIFIED
from nzgd.metadata.rasters import MODEL_COLUMNS, sample_model_columns

INDEX_PATH = constants.RESOURCE_PATH / "nzgd_index.csv.gz"
SIDECAR_PATH = constants.RESOURCE_PATH / "nzgd_id_to_location.csv.gz"


def main() -> int:
    """Build the index; return a process exit code."""
    catalog = build.load_catalog(constants.NZGD_API_INVESTIGATION_CATALOG_PATH_CURRENT)
    if not SIDECAR_PATH.exists():
        print(
            f"ERROR: {SIDECAR_PATH} not found. Run "
            "nzgd/scripts/metadata/one_time/seed_location_sidecar_from_legacy_index.py first."
        )
        return 1
    sidecar = pd.read_csv(SIDECAR_PATH)
    district_gdf = gpd.read_file(constants.DISTRICT_SHAPEFILE_PATH)
    suburbs_gdf = gpd.read_file(constants.SUBURBS_SHAPEFILE_PATH)

    sidecar, new_rows = build.update_sidecar(sidecar, catalog, district_gdf, suburbs_gdf)
    nztm = build.compute_nztm(catalog)
    print("Sampling model rasters...")
    models = sample_model_columns(nztm)
    index_df = build.assemble_index(catalog, sidecar, nztm, models)

    added_ids: list = []
    changed: dict = {}
    if INDEX_PATH.exists():
        prev = pd.read_csv(INDEX_PATH, low_memory=False)
        build.check_id_superset(prev["nzgd_id"], index_df["nzgd_id"])
        added_ids = sorted(set(index_df["nzgd_id"]) - set(prev["nzgd_id"]))
        changed = build.diff_raw_columns(prev, index_df)

    if len(new_rows):
        atomic_write_csv_gz(sidecar, SIDECAR_PATH)
    atomic_write_csv_gz(index_df, INDEX_PATH)

    print(f"Wrote {INDEX_PATH} ({len(index_df):,} rows).")
    print(f"Newly classified IDs appended to sidecar: {len(new_rows)}")
    if len(new_rows):
        print(f"  IDs: {new_rows['nzgd_id'].tolist()}")
        fully_unclassified = new_rows[
            (new_rows[LOCATION_COLUMNS] == UNCLASSIFIED).all(axis=1)
        ]
        print(f"  of which fully unclassified: {fully_unclassified['nzgd_id'].tolist()}")
    if added_ids:
        print(f"IDs added since previous index ({len(added_ids)}): {added_ids[:50]}")
    if changed:
        print("Catalog-column changes since previous index (column: changed IDs):")
        for col, ids in changed.items():
            shown = ids if len(ids) <= 20 else ids[:20] + ["..."]
            print(f"  {col}: {shown}")
    print("NaN model-value counts:")
    print(index_df[MODEL_COLUMNS].isna().sum().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Commit**

```bash
git add nzgd/resources/config.yaml nzgd/constants.py requirements.txt nzgd/scripts/metadata/build_nzgd_index.py
git commit -m "feat: build script and config plumbing for regenerated index"
```

---

### Task 6: seed the location sidecar (one-time, real data)

**Files:**
- Create: `nzgd/scripts/metadata/one_time/seed_location_sidecar_from_legacy_index.py`
- Creates artifact: `nzgd/resources/nzgd_id_to_location.csv.gz`

- [ ] **Step 1: Write the seed script**

`nzgd/scripts/metadata/one_time/seed_location_sidecar_from_legacy_index.py`:

```python
"""One-time: seed nzgd_id_to_location.csv.gz from the legacy 22-Aug-2025 index.

Preserves the manually assembled location classifications as a first-class,
git-tracked artifact. Reports rows whose coordinates fall outside the NZ
bounding box yet carry a real classification, for case-by-case review.
"""

import sys

import pandas as pd

from nzgd import constants
from nzgd.metadata.io_utils import atomic_write_csv_gz
from nzgd.metadata.location import LOCATION_COLUMNS, UNCLASSIFIED, coords_outside_nz

LEGACY_INDEX_PATH = (
    constants.RESOURCE_PATH / "nzgd_metadata_from_coordinates_22_august_2025.csv"
)
SIDECAR_PATH = constants.RESOURCE_PATH / "nzgd_id_to_location.csv.gz"

# 229630 is an upstream NZGD test record at 12.123/34.123 (Sudan); the legacy
# index classified it Canterbury/Christchurch/Casebrook in error.
# See docs/superpowers/specs/2026-06-11-regenerated-nzgd-index-design.md.
KNOWN_CORRECTIONS = {229630: UNCLASSIFIED}


def main() -> int:
    """Seed the sidecar; return a process exit code."""
    legacy = pd.read_csv(LEGACY_INDEX_PATH, low_memory=False)
    sidecar = legacy[["nzgd_id", *LOCATION_COLUMNS]].copy()

    outside = coords_outside_nz(legacy)
    classified = (sidecar[LOCATION_COLUMNS] != UNCLASSIFIED).any(axis=1)
    flagged = legacy.loc[
        outside & classified, ["nzgd_id", "Latitude", "Longitude", *LOCATION_COLUMNS]
    ]
    if len(flagged):
        print("Out-of-NZ coordinates but carrying a classification (review case-by-case):")
        print(flagged.to_string(index=False))

    for nzgd_id, value in KNOWN_CORRECTIONS.items():
        sidecar.loc[sidecar["nzgd_id"] == nzgd_id, LOCATION_COLUMNS] = value
        print(f"Applied documented correction: {nzgd_id} -> {value}")

    sidecar = sidecar.sort_values("nzgd_id").reset_index(drop=True)
    atomic_write_csv_gz(sidecar, SIDECAR_PATH)
    print(
        f"Wrote {SIDECAR_PATH} with {len(sidecar):,} rows "
        f"({int(classified.sum()):,} carrying a classification)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it on real data**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python nzgd/scripts/metadata/one_time/seed_location_sidecar_from_legacy_index.py`
Expected: the flagged-rows report (at design time 229630 appears with
Canterbury/Christchurch_City/Christchurch/Casebrook; other upstream test
records may appear if they carry classifications), the
`Applied documented correction: 229630 -> unclassified` line, and a final
line reporting 197,647 rows written.

**STOP if any flagged row other than 229630 carries a classification that
looks wrong — ask the user before correcting anything else. Only 229630 has a
pre-approved correction.**

- [ ] **Step 3: Sanity-check the sidecar**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -c "
import pandas as pd
df = pd.read_csv('nzgd/resources/nzgd_id_to_location.csv.gz')
print(len(df), df['nzgd_id'].is_unique)
print(df.loc[df['nzgd_id'] == 229630].to_string(index=False))"`
Expected: `197647 True` and the 229630 row showing `unclassified` in all four columns.

- [ ] **Step 4: Commit (script and artifact)**

```bash
git add nzgd/scripts/metadata/one_time/seed_location_sidecar_from_legacy_index.py nzgd/resources/nzgd_id_to_location.csv.gz
git commit -m "feat: seed location sidecar from legacy index (229630 corrected)"
```

---

### Task 7: build the index and verify no loss (real data)

**Files:**
- Create: `nzgd/scripts/metadata/one_time/verify_new_index_against_legacy.py`
- Creates artifacts: `nzgd/resources/nzgd_index.csv.gz`, updated `nzgd/resources/nzgd_id_to_location.csv.gz`

- [ ] **Step 1: Run the build**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python nzgd/scripts/metadata/build_nzgd_index.py`
Expected (a few minutes): `Wrote .../nzgd_index.csv.gz (197,649 rows).`, sidecar gains
2 newly classified IDs `[230469, 230470]` (both fully unclassified — their
coordinates are the 0,0 placeholder), no previous-index diff section (first
build), and NaN model-value counts printed per column.

- [ ] **Step 2: Write the verification script**

`nzgd/scripts/metadata/one_time/verify_new_index_against_legacy.py`:

```python
"""One-time: prove the regenerated index loses nothing vs the legacy index.

Run before deleting any legacy artifact. Non-zero exit on any failure.
See docs/superpowers/specs/2026-06-11-regenerated-nzgd-index-design.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nzgd import constants
from nzgd.metadata.build import RAW_CATALOG_COLUMNS, nan_aware_neq
from nzgd.metadata.location import LOCATION_COLUMNS

LEGACY_INDEX_PATH = (
    constants.RESOURCE_PATH / "nzgd_metadata_from_coordinates_22_august_2025.csv"
)
NEW_INDEX_PATH = constants.RESOURCE_PATH / "nzgd_index.csv.gz"
ARCHIVE_DIR = Path(
    "/home/arr65/data/nzgd/downloads/nzgd_source_files_of_overwritten_nzgd_ids"
)
# IDs whose raw-column divergence is understood and archived (see spec).
KNOWN_DIVERGENT_IDS = {16, 229775, 229822}
# IDs whose location was corrected at seed time (see seed script).
SEED_CORRECTED_IDS = {229630}
# Legacy model column -> new model column (renamed; values sampled fresh).
MODEL_RENAME_MAP = {
    "model_gwl_westerhoff_2018": "model_gwl_westerhoff_2018_m",
    "model_vs30_foster_2019": "model_vs30_foster_2019_m_per_s",
    "model_vs30_std_foster_2019": "model_vs30_stddev_foster_2019_ln",
}


def main() -> int:
    """Run all no-loss checks; return a process exit code."""
    legacy = pd.read_csv(LEGACY_INDEX_PATH, low_memory=False)
    new = pd.read_csv(NEW_INDEX_PATH, low_memory=False)
    failures: list[str] = []

    # 1. Legacy IDs must all survive.
    missing = sorted(set(legacy["nzgd_id"]) - set(new["nzgd_id"]))
    if missing:
        failures.append(f"{len(missing)} legacy IDs missing: {missing[:20]}")
    added = sorted(set(new["nzgd_id"]) - set(legacy["nzgd_id"]))
    print(f"IDs added vs legacy ({len(added)}): {added[:50]}")

    # 2. Raw catalog columns: divergence only on known, archived IDs.
    merged = legacy.merge(new, on="nzgd_id", suffixes=("_old", "_new"), how="inner")
    divergent: set = set()
    for col in RAW_CATALOG_COLUMNS:
        if col == "nzgd_id":
            continue
        mask = nan_aware_neq(merged[f"{col}_old"], merged[f"{col}_new"])
        divergent |= set(merged.loc[mask, "nzgd_id"])
    unexpected = divergent - KNOWN_DIVERGENT_IDS
    if unexpected:
        failures.append(
            "unexpected divergent IDs (review upstream change, archive the old "
            f"row if it is a reassignment, then extend KNOWN_DIVERGENT_IDS): "
            f"{sorted(unexpected)[:20]}"
        )
    print(f"Divergent IDs on raw columns: {sorted(divergent)}")
    show_cols = [
        "nzgd_id", "State", "Type", "TypeDisplay", "InvestigationId",
        "Latitude", "Longitude", "CreatedOn", "LastModifiedOn",
    ]
    for nzgd_id in sorted(divergent):
        print(f"--- {nzgd_id} legacy ---")
        print(legacy.loc[legacy["nzgd_id"] == nzgd_id, show_cols].to_string(index=False))
        print(f"--- {nzgd_id} new ---")
        print(new.loc[new["nzgd_id"] == nzgd_id, show_cols].to_string(index=False))

    # 3. Each divergent ID must have an archived metadata row.
    for nzgd_id in sorted(divergent):
        archive = ARCHIVE_DIR / str(nzgd_id)
        has_row = archive.is_dir() and any(
            "metadata_row" in p.name for p in archive.iterdir()
        )
        if not has_row:
            failures.append(f"no archived metadata row for divergent ID {nzgd_id}")

    # 4. Location columns: identical except documented seed-time corrections.
    loc_old = legacy[["nzgd_id", *LOCATION_COLUMNS]].set_index("nzgd_id")
    loc_new = new[["nzgd_id", *LOCATION_COLUMNS]].set_index("nzgd_id").loc[loc_old.index]
    loc_diff_ids = set(loc_old.index[(loc_old != loc_new).any(axis=1)])
    if loc_diff_ids - SEED_CORRECTED_IDS:
        failures.append(
            f"location changed beyond documented corrections: "
            f"{sorted(loc_diff_ids - SEED_CORRECTED_IDS)[:20]}"
        )
    print(f"Location diffs (expected only {sorted(SEED_CORRECTED_IDS)}): {sorted(loc_diff_ids)}")

    # 5. NZTM coordinates: allclose where both sides have values.
    nztm = legacy[["nzgd_id", "nztm_y", "nztm_x"]].merge(
        new[["nzgd_id", "nztm_y", "nztm_x"]], on="nzgd_id", suffixes=("_old", "_new")
    )
    both = nztm.drop(columns="nzgd_id").notna().all(axis=1)
    nztm_ok = bool(
        np.allclose(
            nztm.loc[both, ["nztm_y_old", "nztm_x_old"]].values,
            nztm.loc[both, ["nztm_y_new", "nztm_x_new"]].values,
        )
    )
    if not nztm_ok:
        failures.append("nztm values not allclose to legacy")
    print(f"nztm allclose over {int(both.sum()):,} rows: {nztm_ok}")

    # 6. Model columns: reported, not asserted (fresh sampling, corrected names;
    #    legacy baked nodata sentinels like -32767 which are now NaN).
    for old_col, new_col in MODEL_RENAME_MAP.items():
        pair = legacy[["nzgd_id", old_col]].merge(new[["nzgd_id", new_col]], on="nzgd_id")
        both_present = pair[old_col].notna() & pair[new_col].notna()
        delta = (pair.loc[both_present, new_col] - pair.loc[both_present, old_col]).abs()
        print(
            f"{old_col} -> {new_col}: n_both={int(both_present.sum()):,} "
            f"max|delta|={delta.max():.6g} legacy_nan={int(pair[old_col].isna().sum())} "
            f"new_nan={int(pair[new_col].isna().sum())}"
        )

    if failures:
        print("\nVERIFICATION FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll no-loss checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run the verification**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python nzgd/scripts/metadata/one_time/verify_new_index_against_legacy.py`
Expected: added IDs `[230469, 230470]`; divergent IDs `[16, 229775, 229822]`
with their old/new rows printed; location diffs `[229630]`; nztm allclose
`True`; model deltas tiny (max|delta| near 0) with `new_nan` larger than
`legacy_nan` for the Vs30 columns (legacy baked -32767 sentinels, and the
legacy Westerhoff column baked 0.0 for out-of-raster points); final line
`All no-loss checks passed.` and exit code 0.

**STOP and consult the user if any check fails. Do not delete anything.**

- [ ] **Step 3b: Port-validation spot check (location classification)**

Recompute the location of a random sample of already-classified IDs with the
new code and real shapefiles, and report the agreement rate against the
sidecar. High agreement expected, not asserted at 100% (shapefile vintages and
boundary edge cases differ from the legacy batches).

Run:

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python - <<'EOF'
import geopandas as gpd
import pandas as pd

from nzgd import constants
from nzgd.metadata.location import LOCATION_COLUMNS, UNCLASSIFIED, classify_locations

index_df = pd.read_csv(constants.RESOURCE_PATH / "nzgd_index.csv.gz", low_memory=False)
classified = index_df[(index_df[LOCATION_COLUMNS] != UNCLASSIFIED).any(axis=1)]
sample = classified.sample(n=500, random_state=42)

district_gdf = gpd.read_file(constants.DISTRICT_SHAPEFILE_PATH)
suburbs_gdf = gpd.read_file(constants.SUBURBS_SHAPEFILE_PATH)
recomputed = classify_locations(
    sample[["nzgd_id", "Latitude", "Longitude"]], district_gdf, suburbs_gdf
)

merged = sample[["nzgd_id", *LOCATION_COLUMNS]].merge(
    recomputed, on="nzgd_id", suffixes=("_sidecar", "_recomputed")
)
for col in LOCATION_COLUMNS:
    agree = (merged[f"{col}_sidecar"] == merged[f"{col}_recomputed"]).mean()
    print(f"{col}: {agree:.1%} agreement over {len(merged)} sampled IDs")
EOF
```

Expected: agreement well above 90% per column. This is a report, not a gate —
but if agreement is very low (say under 80%), something is wrong with the port
(e.g. a field mapping); STOP and investigate before continuing.

- [ ] **Step 4: Commit script and artifacts**

```bash
git add nzgd/scripts/metadata/one_time/verify_new_index_against_legacy.py nzgd/resources/nzgd_index.csv.gz nzgd/resources/nzgd_id_to_location.csv.gz
git commit -m "feat: first regenerated nzgd_index.csv.gz, verified lossless vs legacy"
```

---

### Task 8: column renames (ORM, dedup, config, tests)

**Files:**
- Modify: `nzgd/db/orm.py` (NZGDRecord class, ~lines 179-230)
- Modify: `nzgd/dedup/selection.py` (~line 15), `nzgd/dedup/executor.py` (~line 14), `nzgd/dedup/pass2_fuzzy.py` (~line 33)
- Modify: `nzgd/resources/config.yaml` (`field_plausibility_ranges`, ~line 246)
- Modify: `tests/dedup/conftest.py` and any test using the old names

The four renames (values unchanged everywhere — pure renames):
`model_vs30_foster_2019_km_per_s` → `model_vs30_foster_2019_m_per_s`;
`model_vs30_stddev_foster_2019_km_per_s` → `model_vs30_stddev_foster_2019_ln`;
`investigation_date` → `record_created_on`; `published_date` → `record_last_modified_on`.

- [ ] **Step 1: Rename ORM fields and fix their docstrings**

In `nzgd/db/orm.py`, NZGDRecord class, replace:

```python
    model_vs30_foster_2019_km_per_s = FloatField(null=True)
    """float: The modelled Vs30 value from Foster et al. (2019), at this record's
    location."""

    model_vs30_stddev_foster_2019_km_per_s = FloatField(null=True)
    """float: The modelled Vs30 standard deviation from Foster et al. (2019), at this
    record's location."""
```

with:

```python
    model_vs30_foster_2019_m_per_s = FloatField(null=True)
    """float: The modelled Vs30 value from Foster et al. (2019), at this record's
    location, in metres per second."""

    model_vs30_stddev_foster_2019_ln = FloatField(null=True)
    """float: The modelled Vs30 standard deviation from Foster et al. (2019), at
    this record's location, in natural-log units (dimensionless)."""
```

and replace:

```python
    investigation_date = DateField(formats=["%Y-%m-%d"], null=True)
    """date: The date the investigation was conducted."""

    published_date = DateField(formats=["%Y-%m-%d"], null=True)
    """date: The date the record was published."""
```

with:

```python
    record_created_on = DateField(formats=["%Y-%m-%d"], null=True)
    """date: When the record was created in NZGD (NZGD CreatedOn). Not the
    investigation date: NZGD stores that, sparsely, in EndDate."""

    record_last_modified_on = DateField(formats=["%Y-%m-%d"], null=True)
    """date: When the record was last modified in NZGD (NZGD LastModifiedOn)."""
```

and in `class Meta` replace `(("model_vs30_foster_2019_km_per_s",), False),`
with `(("model_vs30_foster_2019_m_per_s",), False),`.

- [ ] **Step 2: Rename in the dedup modules**

In BOTH `nzgd/dedup/selection.py` and `nzgd/dedup/executor.py`, replace the
identical tuple:

```python
_NZGDRECORD_METADATA_COLUMNS = (
    "type_id", "latitude", "longitude",
    "model_vs30_foster_2019_km_per_s", "model_vs30_stddev_foster_2019_km_per_s",
    "model_gwl_westerhoff_2018_m", "model_gwl_nlm_2025_m", "model_gwl_nlm_2025_stddev_m",
    "original_investigation_name", "investigation_date", "published_date",
    "region_id", "district_id", "city_id", "suburb_id",
)
```

with:

```python
_NZGDRECORD_METADATA_COLUMNS = (
    "type_id", "latitude", "longitude",
    "model_vs30_foster_2019_m_per_s", "model_vs30_stddev_foster_2019_ln",
    "model_gwl_westerhoff_2018_m", "model_gwl_nlm_2025_m", "model_gwl_nlm_2025_stddev_m",
    "original_investigation_name", "record_created_on", "record_last_modified_on",
    "region_id", "district_id", "city_id", "suburb_id",
)
```

In `nzgd/dedup/pass2_fuzzy.py` (~line 33) the SQL selects
`n.investigation_date`; change it to `n.record_created_on`, and update any
downstream reference to that result column inside the same file (grep the file
for `investigation_date` and rename every occurrence).

- [ ] **Step 3: Rename the config plausibility keys**

In `nzgd/resources/config.yaml` under `field_plausibility_ranges: nzgdrecord:`, replace:

```yaml
      model_vs30_foster_2019_km_per_s:           [50.0, 2000.0]
      model_vs30_stddev_foster_2019_km_per_s:    [0.0, 10.0]
```

with:

```yaml
      model_vs30_foster_2019_m_per_s:            [50.0, 2000.0]
      model_vs30_stddev_foster_2019_ln:          [0.0, 10.0]
```

- [ ] **Step 4: Rename in tests, then sweep for stragglers**

In `tests/dedup/conftest.py` rename the two `km_per_s` schema columns and the
`investigation_date`/`published_date` columns and helper parameters to the new
names; then sweep:

Run: `grep -rn "km_per_s\|investigation_date\|published_date" nzgd/ tests/ --include="*.py" | grep -v scripts/db/put_nzgd_metadata.py | grep -v scripts/metadata/make_metadata_summary_csv.py`
Expected: no output (the two excluded scripts are rewritten in Tasks 9-10).
Fix any hits this reveals (e.g. other tests) before proceeding.

- [ ] **Step 5: Run the test suite**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/ -v`
Expected: all tests pass (dedup suite green with renamed columns).

- [ ] **Step 6: Commit**

```bash
git add nzgd/db/orm.py nzgd/dedup/ nzgd/resources/config.yaml tests/
git commit -m "refactor: correct inaccurate column names (vs30 units, log sigma, record dates)"
```

---

### Task 9: rewrite put_nzgd_metadata.py to read model columns from the index

**Files:**
- Modify: `nzgd/scripts/db/put_nzgd_metadata.py` (replace entire file)

The current file hardcodes the index filename and duplicates ~230 lines of
raster sampling that now lives in `nzgd/metadata/rasters.py` (sampling happens
once, at index build time). Replace the whole file with:

- [ ] **Step 1: Replace the file**

```python
"""Serialize NZGD record metadata from the index into the database.

Model values (Vs30, groundwater) come straight from the regenerated NZGD
index, which samples the GeoTIFFs at build time (nzgd/metadata/rasters.py).
"""

import sqlite3

import pandas as pd
from tqdm import tqdm

from nzgd import constants


def serialize_record_metadata(
    metadata_df: pd.DataFrame, spt_ids: set, cpt_ids: set, conn: sqlite3.Connection
) -> None:
    """Insert or replace nzgdrecord rows from index metadata.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Rows of the NZGD index (nzgd_id, Latitude, Longitude, model columns,
        InvestigationId, CreatedOn, LastModifiedOn, region/district/city/suburb).
    spt_ids : set
        nzgd_id values that correspond to SPT records (type_id 2).
    cpt_ids : set
        nzgd_id values that correspond to CPT records (type_id 1).
    conn : sqlite3.Connection
        Open database connection (modified in place).
    """
    cursor = conn.cursor()

    location_categories = ["region", "district", "city", "suburb"]
    location_id_maps = {}
    for category in location_categories:
        cursor.execute(f"SELECT id, value FROM {category}")
        location_id_maps[category] = {name: id_ for id_, name in cursor.fetchall()}

    for category in location_categories:
        metadata_df[f"{category}_id"] = metadata_df[category].map(
            location_id_maps[category],
        )

    for _, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
        if row["nzgd_id"] in cpt_ids:
            type_id = 1
        elif row["nzgd_id"] in spt_ids:
            type_id = 2
        else:
            raise ValueError(
                f"nzgd_id {row['nzgd_id']} is not in either cpt_ids or spt_ids"
            )

        cursor.execute(
            """
            INSERT OR REPLACE INTO nzgdrecord (
                nzgd_id, type_id, latitude, longitude,
                model_vs30_foster_2019_m_per_s, model_vs30_stddev_foster_2019_ln,
                model_gwl_westerhoff_2018_m, model_gwl_nlm_2025_m,
                model_gwl_nlm_2025_stddev_m, original_investigation_name,
                record_created_on, record_last_modified_on,
                region_id, district_id, city_id, suburb_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row["nzgd_id"]),
                type_id,
                row["Latitude"],
                row["Longitude"],
                row["model_vs30_foster_2019_m_per_s"],
                row["model_vs30_stddev_foster_2019_ln"],
                row["model_gwl_westerhoff_2018_m"],
                row["model_gwl_nlm_2025_m"],
                row["model_gwl_nlm_2025_stddev_m"],
                row["InvestigationId"],
                row["CreatedOn"],
                row["LastModifiedOn"],
                row["region_id"],
                row["district_id"],
                row["city_id"],
                row["suburb_id"],
            ),
        )


if __name__ == "__main__":
    with sqlite3.connect(constants.OUTPUT_DB_PATH) as db:
        sptreport_df = pd.read_sql_query("SELECT * FROM sptreport", db)
        cpt_report_df = pd.read_sql_query("SELECT * FROM cptreport", db)

    spt_ids = set(sptreport_df["nzgd_id"].unique().tolist())
    cpt_ids = set(cpt_report_df["nzgd_id"].unique().tolist())
    nzgd_ids_in_db = spt_ids.union(cpt_ids)

    metadata_df = pd.read_csv(constants.INDEX_FILE_PATH, low_memory=False)
    metadata_df = metadata_df[metadata_df["nzgd_id"].isin(nzgd_ids_in_db)]

    with sqlite3.connect(constants.OUTPUT_DB_PATH) as db:
        serialize_record_metadata(metadata_df, spt_ids, cpt_ids, db)
```

NOTE: pandas NaN values must reach sqlite as NULL. The old code had the same
property for its sampled values via `None`; with `row[...]` a NaN float is
passed through and sqlite3 stores it as a REAL NaN, not NULL. Convert before
the loop — insert this line immediately before `for _, row in tqdm(...)`:

```python
    metadata_df = metadata_df.astype(object).where(pd.notna(metadata_df), None)
```

- [ ] **Step 2: Compile-check and lint**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m py_compile nzgd/scripts/db/put_nzgd_metadata.py && /home/arr65/venvs/dev_nzgd_venv/bin/python -m ruff check nzgd/scripts/db/put_nzgd_metadata.py`
Expected: no output / "All checks passed". (A full run needs a freshly built
DB with the new schema — that happens at the next normal DB rebuild, not here.
If ruff is not installed in the venv, run `ruff check` from wherever it is
installed, or skip the lint half.)

- [ ] **Step 3: Commit**

```bash
git add nzgd/scripts/db/put_nzgd_metadata.py
git commit -m "refactor: put_nzgd_metadata reads model columns from regenerated index"
```

---

### Task 10: repoint make_metadata_summary_csv.py at the index

**Files:**
- Modify: `nzgd/scripts/metadata/make_metadata_summary_csv.py`

- [ ] **Step 1: Replace the input block**

Replace the current top-of-file input block (the
`nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl_df = pd.read_csv(...)`
call and the `.rename(...)` that follows it, lines ~35-53) with:

```python
nzgd_index_df = pd.read_csv(constants.INDEX_FILE_PATH, low_memory=False)
nzgd_index_df = nzgd_index_df.rename(
    columns={
        "State": "availability_status",
        "InvestigationId": "original_investigation_name",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "CreatedOn": "record_created_on",
        "LastModifiedOn": "record_last_modified_on",
    }
)
```

(The old map also renamed `nlm_gwl_m`/`nlm_gwl_stddev_m` — those columns came
from the deleted intermediate; the index already carries
`model_gwl_nlm_2025_m`/`model_gwl_nlm_2025_stddev_m` directly.)

- [ ] **Step 2: Rename every later use of the long variable**

Replace all remaining occurrences of
`nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl_df` with
`nzgd_index_df` (they appear in the `columns_not_in_db` block, ~lines 258-283).

Run: `grep -n "with_nlm_gwl" nzgd/scripts/metadata/make_metadata_summary_csv.py`
Expected: no output.

- [ ] **Step 3: Update the SQL column names**

In both SQL queries, replace:
- `nz.model_vs30_foster_2019_km_per_s,` → `nz.model_vs30_foster_2019_m_per_s,`
- `nz.model_vs30_stddev_foster_2019_km_per_s,` → `nz.model_vs30_stddev_foster_2019_ln,`
- `nz.investigation_date,` → `nz.record_created_on,`
- `nz.published_date,` → `nz.record_last_modified_on,`

- [ ] **Step 4: Compile-check, lint, sweep**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m py_compile nzgd/scripts/metadata/make_metadata_summary_csv.py && grep -rn "km_per_s\|investigation_date\|published_date" nzgd/ tests/ --include="*.py"`
Expected: py_compile silent; the grep returns no output anywhere in the repo's
Python code. (A real summary run needs a DB built with the new schema — next
normal DB rebuild.)

- [ ] **Step 5: Commit**

```bash
git add nzgd/scripts/metadata/make_metadata_summary_csv.py
git commit -m "refactor: metadata summary reads regenerated index directly"
```

---

### Task 11: flip the config to the new index

**Files:**
- Modify: `nzgd/resources/config.yaml` (~lines 26-27)

- [ ] **Step 1: Flip the index filename**

Replace:

```yaml
nzgd_index_file_name: "nzgd_metadata_from_coordinates_22_august_2025.csv"
#nzgd_index_file_name: "only_127920_nzgd_metadata_from_coordinates_22_august_2025.csv"
```

with:

```yaml
nzgd_index_file_name: "nzgd_index.csv.gz"
```

(The commented line referenced a file that no longer exists.)

- [ ] **Step 2: Smoke-check every consumer path**

Run:

```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python - <<'EOF'
import pandas as pd
from nzgd import constants
from nzgd.extract.bh import ags_miner

new = pd.read_csv(constants.INDEX_FILE_PATH, low_memory=False)
legacy = pd.read_csv(
    constants.RESOURCE_PATH / "nzgd_metadata_from_coordinates_22_august_2025.csv",
    low_memory=False,
)
print("shape:", new.shape)

new_scp = set(new.loc[new["Type"] == "SCP", "nzgd_id"])
old_scp = set(legacy.loc[legacy["Type"] == "SCP", "nzgd_id"])
print("SCP gained:", sorted(new_scp - old_scp), "lost:", sorted(old_scp - new_scp))

bh_displays = constants.NZGD_TypeDisplay_VALUES_FOR_BOREHOLES
new_bh = set(new.loc[new["TypeDisplay"].isin(bh_displays), "nzgd_id"])
old_bh = set(legacy.loc[legacy["TypeDisplay"].isin(bh_displays), "nzgd_id"])
print("BH gained:", sorted(new_bh - old_bh), "lost:", sorted(old_bh - new_bh))

print("ags_miner loads:", len(ags_miner._load_index_data()))
EOF
```

Expected — these are exactly the spec's "behavior deltas":
```
shape: (197649, 37)
SCP gained: [230470] lost: [229822]
BH gained: [229775, 229822] lost: []
ags_miner loads: 197649
```
(229822's CPT files live in the overwritten-IDs archive, not the main download
tree, so neither extraction pipeline processes stale files. ID 16 is SCP in
both old and new, so it does not appear in the deltas; it was already in the
CPT set and now has files on disk.)

**STOP if the deltas differ — the catalog may have been re-synced since the
verification; re-run Task 7 and review.**

- [ ] **Step 3: Commit**

```bash
git add nzgd/resources/config.yaml
git commit -m "feat: switch nzgd_index_file_name to regenerated nzgd_index.csv.gz"
```

---

### Task 12: delete superseded legacy artifacts

Only after Task 7's verification passed and Task 11's smoke checks matched.

**Files:**
- Delete (git): `nzgd/resources/nzgd_metadata_from_coordinates_22_august_2025.csv`,
  `nzgd/scripts/metadata/update_nzgd_metadata_for_past_and_current_nzgd_investigations.py`,
  `nzgd/scripts/temp/assemble_nzgd_metadata_from_coordinates.py`
- Delete (disk only, gitignored): the `with_nlm_gwl` CSVs and `nzgd_id_to_region.csv`
- Modify: `.gitignore` (~lines 199-201)

- [ ] **Step 1: Remove tracked legacy files**

```bash
git rm nzgd/resources/nzgd_metadata_from_coordinates_22_august_2025.csv
git rm nzgd/scripts/metadata/update_nzgd_metadata_for_past_and_current_nzgd_investigations.py
git rm nzgd/scripts/temp/assemble_nzgd_metadata_from_coordinates.py
```

(Their content stays in git history; the legacy index's initial commit is
`20933a0`. The one_time seed/verify scripts will no longer be runnable —
expected; they are kept as the audit record.)

- [ ] **Step 2: Remove gitignored intermediates from disk**

```bash
rm -f nzgd/resources/nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl.csv
rm -f nzgd/resources/nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl.csv.gz
rm -f nzgd/resources/nzgd_id_to_region.csv
```

- [ ] **Step 3: Clean .gitignore**

Remove these three lines from `.gitignore`:

```
nzgd/resources/nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl.csv
nzgd/resources/nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl.csv.gz
nzgd/resources/nzgd_id_to_region.csv
```

- [ ] **Step 4: Optional — the retired repo's duplicate copy**

`/home/arr65/src/nzgd_data_extraction/nzgd_data_extraction/resources/nzgd_metadata_from_coordinates_22_august_2025.csv`
is an md5-identical copy in the retired repo. The user has said that repo may
break, but deleting outside this repo is not this plan's call to make
silently: **ask the user** whether to delete it, and only then `rm` it.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "chore: remove superseded legacy index and broken intermediate chain"
```

---

### Task 13: document Stage 4 in the api_nzgd sync README

**Files:**
- Modify: `/home/arr65/src/api_nzgd/README_to_download_nzgd_updates.md` (separate repo, branch `dev`)

- [ ] **Step 1: Replace the stale reconcile instruction**

Read the file first. Around line 70 it says:

```markdown
Then manually reconcile `nzgd_metadata_from_coordinates_22_august_2025.csv` per `NOTES.md`.
```

Replace that line with:

```markdown
## Stage 4: rebuild the NZGD index (nzgd repo)

After the catalogs are updated, regenerate the derived index used by the
extraction and DB scripts:

    cd /home/arr65/src/nzgd
    /home/arr65/venvs/dev_nzgd_venv/bin/python nzgd/scripts/metadata/build_nzgd_index.py

This rebuilds `nzgd/resources/nzgd_index.csv.gz` from
`current_nzgd_investigation_catalog.csv.gz` (plus the location sidecar and
local model rasters), reports exactly what changed, and refuses to drop any
previously indexed record. Commit the updated `nzgd_index.csv.gz` and
`nzgd_id_to_location.csv.gz` in the nzgd repo — their git history records
when the mirror was last updated.
```

Integrate with the surrounding text (numbering/headings) as the file's
structure requires — read before editing.

- [ ] **Step 2: Commit in api_nzgd**

```bash
git -C /home/arr65/src/api_nzgd status -sb   # confirm on dev, clean
git -C /home/arr65/src/api_nzgd add README_to_download_nzgd_updates.md
git -C /home/arr65/src/api_nzgd commit -m "docs: Stage 4 - rebuild regenerated NZGD index after sync"
```

Do not push unless the user asks.

---

### Task 14: final verification

- [ ] **Step 1: Full test suite**

Run: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 2: Remnant sweep**

Run: `grep -rn "metadata_from_coordinates\|km_per_s\|investigation_date\|published_date" nzgd/ tests/ --include="*.py" --include="*.yaml"`
Expected: hits only in `nzgd/scripts/metadata/one_time/` (the audit-record
scripts intentionally reference the legacy filename). Anything else is a miss
— fix it.

- [ ] **Step 3: Review the branch**

Run: `git log --oneline feat/cpt-supplemental-consolidation..HEAD`
Expected: the spec/plan commits plus one commit per task above.

- [ ] **Step 4: Hand off**

Implementation complete. Use superpowers:finishing-a-development-branch to
decide merge/PR/cleanup with the user. Remind the user of the two follow-ups
that are intentionally NOT in this plan: the next versioned DB rebuild picks
up the renamed schema (and only then can the summary CSV be regenerated), and
the map webapps are separate focused work.
