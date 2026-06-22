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
