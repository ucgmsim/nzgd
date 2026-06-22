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
