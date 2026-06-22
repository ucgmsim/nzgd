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
