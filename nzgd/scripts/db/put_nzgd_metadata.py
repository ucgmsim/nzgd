import sqlite3

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from nzgd import constants
from qcore import coordinates


def sample_nlm_gwl_values(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Sample NLM groundwater level values from GeoTIFF files.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame with latitude and longitude columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with added model_gwl_nlm_2025_m and model_gwl_nlm_2025_stddev_m columns.
    """
    # Filter to rows with valid coordinates
    has_coords = metadata_df["Latitude"].notna() & metadata_df["Longitude"].notna()
    coords_df = metadata_df[has_coords].copy()

    if len(coords_df) == 0:
        print("No records with valid coordinates to sample NLM values.")
        metadata_df["model_gwl_nlm_2025_m"] = None
        metadata_df["model_gwl_nlm_2025_stddev_m"] = None
        return metadata_df

    print(f"Sampling NLM values for {len(coords_df)} records with coordinates...")

    # Convert latitude and longitude to NZTM coordinates
    nztm_yx = coordinates.wgs_depth_to_nztm(
        coords_df[["Latitude", "Longitude"]].values,
    )

    # nztm_yx returns (northing, easting) = (y, x)
    # For rasterio.sample(), we need (easting, northing) = (x, y) pairs
    coords_df["nztm_northing"] = nztm_yx[:, 0]  # y = northing
    coords_df["nztm_easting"] = nztm_yx[:, 1]  # x = easting

    # Make a list of (easting, northing) pairs for rasterio.sample()
    nztm_xy_list = []
    for row_idx in range(len(coords_df)):
        nztm_xy_list.append(
            np.array(
                [
                    coords_df.iloc[row_idx]["nztm_easting"],
                    coords_df.iloc[row_idx]["nztm_northing"],
                ],
            ),
        )

    # Sample from NLM GWD (Groundwater Depth)
    print("  Sampling from NLM GWD (Groundwater Depth)...")
    if not constants.NLM_GWD_PATH.exists():
        raise FileNotFoundError(f"NLM GWD file not found: {constants.NLM_GWD_PATH}")

    with rasterio.open(constants.NLM_GWD_PATH) as dataset:
        nlm_gwd_values_in_array = list(dataset.sample(nztm_xy_list))

    # Convert the sampled values to a list of floats, handling masked/NaN values
    # rasterio.sample() returns masked arrays for coordinates outside raster bounds
    nlm_gwd_values = []
    for x in nlm_gwd_values_in_array:
        try:
            # Try to get the value, handling masked arrays
            val = float(x[0]) if not pd.isna(x[0]) else None
        except (ValueError, IndexError, TypeError):
            # Handle cases where x[0] is masked or invalid
            val = None
        nlm_gwd_values.append(val)

    coords_df["model_gwl_nlm_2025_m"] = nlm_gwd_values

    # Sample from NLM GW STD (Groundwater Standard Deviation without seasonal)
    print("  Sampling from NLM GW STD (without seasonal)...")
    if not constants.NLM_GW_STD_PATH.exists():
        raise FileNotFoundError(
            f"NLM GW STD file not found: {constants.NLM_GW_STD_PATH}"
        )

    with rasterio.open(constants.NLM_GW_STD_PATH) as dataset:
        nlm_gw_std_values_in_array = list(dataset.sample(nztm_xy_list))

    # Convert the sampled values to a list of floats, handling masked/NaN values
    # rasterio.sample() returns masked arrays for coordinates outside raster bounds
    nlm_gw_std_values = []
    for x in nlm_gw_std_values_in_array:
        try:
            # Try to get the value, handling masked arrays
            val = float(x[0]) if not pd.isna(x[0]) else None
        except (ValueError, IndexError, TypeError):
            # Handle cases where x[0] is masked or invalid
            val = None
        nlm_gw_std_values.append(val)

    coords_df["model_gwl_nlm_2025_stddev_m"] = nlm_gw_std_values

    # Merge back into original dataframe
    metadata_df = metadata_df.merge(
        coords_df[["nzgd_id", "model_gwl_nlm_2025_m", "model_gwl_nlm_2025_stddev_m"]],
        on="nzgd_id",
        how="left",
    )

    print(f"  Successfully sampled NLM values for {len(coords_df)} records")
    return metadata_df


def sample_foster_2019_vs30_values(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Sample Foster et al. (2019) Vs30 values from GeoTIFF file.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame with latitude and longitude columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with added model_vs30_foster_2019_km_per_s and model_vs30_std_foster_2019_km_per_s columns.
    """
    # Filter to rows with valid coordinates
    has_coords = metadata_df["Latitude"].notna() & metadata_df["Longitude"].notna()
    coords_df = metadata_df[has_coords].copy()

    if len(coords_df) == 0:
        print("No records with valid coordinates to sample Foster 2019 Vs30 values.")
        metadata_df["model_vs30_foster_2019_km_per_s"] = None
        metadata_df["model_vs30_std_foster_2019_km_per_s"] = None
        return metadata_df

    print(
        f"Sampling Foster 2019 Vs30 values for {len(coords_df)} records with coordinates..."
    )

    # Convert latitude and longitude to NZTM coordinates
    nztm_yx = coordinates.wgs_depth_to_nztm(
        coords_df[["Latitude", "Longitude"]].values,
    )

    # nztm_yx returns (northing, easting) = (y, x)
    # For rasterio.sample(), we need (easting, northing) = (x, y) pairs
    coords_df["nztm_northing"] = nztm_yx[:, 0]  # y = northing
    coords_df["nztm_easting"] = nztm_yx[:, 1]  # x = easting

    # Make a list of (easting, northing) pairs for rasterio.sample()
    nztm_xy_list = []
    for row_idx in range(len(coords_df)):
        nztm_xy_list.append(
            np.array(
                [
                    coords_df.iloc[row_idx]["nztm_easting"],
                    coords_df.iloc[row_idx]["nztm_northing"],
                ],
            ),
        )

    # Sample from Foster 2019 Vs30 model
    print("  Sampling from Foster 2019 Vs30 model...")
    if not constants.FOSTER_2019_VS30_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Foster 2019 Vs30 model file not found: {constants.FOSTER_2019_VS30_MODEL_PATH}"
        )

    with rasterio.open(constants.FOSTER_2019_VS30_MODEL_PATH) as foster_ds:
        # gather samples; each sample is an array with length == number of bands
        foster_samples = list(foster_ds.sample(nztm_xy_list))

    # Convert to numpy array (n_points x n_bands), ensuring float and flattened arrays
    # handle masked arrays if present
    foster_arr = np.array(
        [np.asarray(s).astype(float).flatten() for s in foster_samples]
    )

    # Ensure 2D shape
    if foster_arr.ndim == 1:
        foster_arr = foster_arr.reshape(-1, 1)

    # Assign to DataFrame with safety checks: expect band 0 = vs30, band 1 = vs30_std
    if foster_arr.shape[1] >= 1:
        # Convert masked/NaN values to None
        vs30_values = []
        for val in foster_arr[:, 0]:
            try:
                val_float = float(val) if not pd.isna(val) else None
            except (ValueError, TypeError):
                val_float = None
            vs30_values.append(val_float)
        coords_df["model_vs30_foster_2019_km_per_s"] = vs30_values
    else:
        coords_df["model_vs30_foster_2019_km_per_s"] = None

    if foster_arr.shape[1] >= 2:
        # Convert masked/NaN values to None
        vs30_std_values = []
        for val in foster_arr[:, 1]:
            try:
                val_float = float(val) if not pd.isna(val) else None
            except (ValueError, TypeError):
                val_float = None
            vs30_std_values.append(val_float)
        coords_df["model_vs30_std_foster_2019_km_per_s"] = vs30_std_values
    else:
        # if std band not present, fill with None
        coords_df["model_vs30_std_foster_2019_km_per_s"] = None

    # Merge back into original dataframe
    metadata_df = metadata_df.merge(
        coords_df[
            [
                "nzgd_id",
                "model_vs30_foster_2019_km_per_s",
                "model_vs30_std_foster_2019_km_per_s",
            ]
        ],
        on="nzgd_id",
        how="left",
    )

    print(
        f"  Successfully sampled Foster 2019 Vs30 values for {len(coords_df)} records"
    )
    return metadata_df


def sample_westerhoff_2018_gwl_values(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Sample Westerhoff 2018 groundwater level values from GeoTIFF file.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame with latitude and longitude columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with added model_gwl_westerhoff_2018_m column.
    """
    # Filter to rows with valid coordinates
    has_coords = metadata_df["Latitude"].notna() & metadata_df["Longitude"].notna()
    coords_df = metadata_df[has_coords].copy()

    if len(coords_df) == 0:
        print("No records with valid coordinates to sample Westerhoff 2018 values.")
        metadata_df["model_gwl_westerhoff_2018_m"] = None
        return metadata_df

    print(
        f"Sampling Westerhoff 2018 values for {len(coords_df)} records with coordinates..."
    )

    # Convert latitude and longitude to NZTM coordinates
    nztm_yx = coordinates.wgs_depth_to_nztm(
        coords_df[["Latitude", "Longitude"]].values,
    )

    # nztm_yx returns (northing, easting) = (y, x)
    # For rasterio.sample(), we need (easting, northing) = (x, y) pairs
    coords_df["nztm_northing"] = nztm_yx[:, 0]  # y = northing
    coords_df["nztm_easting"] = nztm_yx[:, 1]  # x = easting

    # Make a list of (easting, northing) pairs for rasterio.sample()
    nztm_xy_list = []
    for row_idx in range(len(coords_df)):
        nztm_xy_list.append(
            np.array(
                [
                    coords_df.iloc[row_idx]["nztm_easting"],
                    coords_df.iloc[row_idx]["nztm_northing"],
                ],
            ),
        )

    # Sample from Westerhoff 2018 model
    print("  Sampling from Westerhoff 2018 model...")
    if not constants.WESTERHOFF_2018_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Westerhoff 2018 model file not found: {constants.WESTERHOFF_2018_MODEL_PATH}"
        )

    with rasterio.open(constants.WESTERHOFF_2018_MODEL_PATH) as dataset:
        westerhoff_values_in_array = list(dataset.sample(nztm_xy_list))

    # Convert the sampled values to a list of floats, handling masked/NaN values
    # rasterio.sample() returns masked arrays for coordinates outside raster bounds
    westerhoff_values = []
    for x in westerhoff_values_in_array:
        try:
            # Try to get the value, handling masked arrays
            val = float(x[0]) if not pd.isna(x[0]) else None
        except (ValueError, IndexError, TypeError):
            # Handle cases where x[0] is masked or invalid
            val = None
        westerhoff_values.append(val)

    coords_df["model_gwl_westerhoff_2018_m"] = westerhoff_values

    # Merge back into original dataframe
    metadata_df = metadata_df.merge(
        coords_df[["nzgd_id", "model_gwl_westerhoff_2018_m"]],
        on="nzgd_id",
        how="left",
    )

    print(f"  Successfully sampled Westerhoff 2018 values for {len(coords_df)} records")
    return metadata_df


def serialize_record_metadata(
    metadata_df: pd.DataFrame, spt_ids: set, cpt_ids: set, conn: sqlite3.Connection
):
    """Serialize NZGD record metadata to the database.

    This function processes a DataFrame containing NZGD metadata and inserts or
    updates records in the nzgdrecord table. It maps location strings (region,
    district, city, suburb) to their corresponding database IDs and determines
    the investigation type (CPT or SPT) based on the nzgd_id.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame containing NZGD metadata with columns including:
        - nzgd_id: Unique identifier for the NZGD record
        - Latitude, Longitude: Geographic coordinates
        - model_vs30_foster_2019_km_per_s: Foster 2019 Vs30 model value
        - model_vs30_std_foster_2019_km_per_s: Foster 2019 Vs30 standard deviation
        - model_gwl_westerhoff_2018_m: Westerhoff 2018 groundwater level model value
        - model_gwl_nlm_2025_m: NLM 2025 groundwater level model value
        - model_gwl_nlm_2025_stddev_m: NLM 2025 groundwater level standard deviation
        - InvestigationId: Original investigation name
        - CreatedOn: Investigation date
        - LastModifiedOn: Published date
        - region, district, city, suburb: Location strings to be mapped to IDs
    spt_ids : set
        Set of nzgd_id values that correspond to Standard Penetration Test (SPT)
        records. Records with nzgd_id in this set will be assigned type_id = 2.
    cpt_ids : set
        Set of nzgd_id values that correspond to Cone Penetration Test (CPT)
        records. Records with nzgd_id in this set will be assigned type_id = 1.
    conn : sqlite3.Connection
        SQLite database connection object used to execute database operations.

    Returns
    -------
    None
        This function modifies the database in-place and does not return a value.

    Notes
    -----
    - The function uses INSERT OR REPLACE to update existing records if they
      already exist in the database.
    - Location strings are mapped to IDs by querying the region, district, city,
      and suburb tables in the database.
    - Progress is displayed using tqdm progress bar during iteration.
    """
    cursor = conn.cursor()

    # Fetch location id mappings from the database
    location_categories = ["region", "district", "city", "suburb"]
    location_id_maps = {}
    for category in location_categories:
        cursor.execute(f"SELECT id, value FROM {category}")
        location_id_maps[category] = {name: id_ for id_, name in cursor.fetchall()}

    # Map string columns in metadata_df to their corresponding ids
    for category in location_categories:
        metadata_df[f"{category}_id"] = metadata_df[category].map(
            location_id_maps[category],
        )

    for _, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
        if row["nzgd_id"] in cpt_ids:
            type_id = 1  # for CPT
        elif row["nzgd_id"] in spt_ids:
            type_id = 2  # for SPT
        else:
            raise ValueError(
                f"nzgd_id {row['nzgd_id']} is not in either cpt_ids or spt_ids"
            )

        cursor.execute(
            """
            INSERT OR REPLACE INTO nzgdrecord (nzgd_id, type_id, latitude, longitude, model_vs30_foster_2019_km_per_s, model_vs30_stddev_foster_2019_km_per_s, model_gwl_westerhoff_2018_m, model_gwl_nlm_2025_m, model_gwl_nlm_2025_stddev_m, original_investigation_name, investigation_date, published_date, region_id, district_id, city_id, suburb_id)
            VALUES (?, ? , ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                int(row["nzgd_id"]),
                type_id,
                row["Latitude"],
                row["Longitude"],
                row["model_vs30_foster_2019_km_per_s"],
                row["model_vs30_std_foster_2019_km_per_s"],
                row.get("model_gwl_westerhoff_2018_m"),
                row.get("model_gwl_nlm_2025_m"),
                row.get("model_gwl_nlm_2025_stddev_m"),
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

    nzgd_ids_in_db = set(sptreport_df["nzgd_id"].unique().tolist()).union(
        set(cpt_report_df["nzgd_id"].unique().tolist()),
    )

    metadata_df = pd.read_csv(
        str(
            constants.RESOURCE_PATH
            / "nzgd_metadata_from_coordinates_22_august_2025.csv",
        )
    )

    # Filter metadata to keep only rows where nzgd_id is in the database
    metadata_df = metadata_df[metadata_df["nzgd_id"].isin(nzgd_ids_in_db)]

    # Sample Foster 2019 Vs30 values from GeoTIFF file
    metadata_df = sample_foster_2019_vs30_values(metadata_df)

    # Sample Westerhoff 2018 groundwater level values from GeoTIFF file
    metadata_df = sample_westerhoff_2018_gwl_values(metadata_df)

    # Sample NLM groundwater level values from GeoTIFF files
    metadata_df = sample_nlm_gwl_values(metadata_df)

    with sqlite3.connect(constants.OUTPUT_DB_PATH) as db:
        serialize_record_metadata(metadata_df, spt_ids, cpt_ids, db)
