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
