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
