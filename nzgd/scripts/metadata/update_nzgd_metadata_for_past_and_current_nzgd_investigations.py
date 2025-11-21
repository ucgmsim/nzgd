"""Add NLM groundwater level values to the NZGD index file."""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from natsort import natsort_keygen

from nzgd import constants

nzgd_metadata_currently_available_on_nzgd_df = pd.read_csv(
    "/home/arr65/src/api_nzgd/api_nzgd/resources/available_nzgd_investigations.csv.gz",
    compression="gzip",
)
nzgd_metadata_currently_available_on_nzgd_df = (
    nzgd_metadata_currently_available_on_nzgd_df.rename(columns={"Id": "nzgd_id"})
)

nzgd_metadata_past_and_present_df = pd.read_csv(
    constants.RESOURCE_PATH / "nzgd_metadata_from_coordinates_22_august_2025.csv.gz",
    compression="gzip",
)
nzgd_metadata_past_and_present_df["nlm_gwl_m"] = np.nan
nzgd_metadata_past_and_present_df["nlm_gwl_stddev_m"] = np.nan

# Check for nzgd_id in currently_available that are not in past_and_present
currently_available_ids = set(nzgd_metadata_currently_available_on_nzgd_df["nzgd_id"])
past_and_present_ids = set(nzgd_metadata_past_and_present_df["nzgd_id"])
missing_ids = currently_available_ids - past_and_present_ids

if missing_ids:
    print(
        f"Found {len(missing_ids)} nzgd_id values in nzgd_metadata_currently_available_on_nzgd_df that are not in nzgd_metadata_past_and_present_df"
    )
    print(f"Missing IDs: {sorted(missing_ids)}")
else:
    print(
        "All nzgd_id values in nzgd_metadata_currently_available_on_nzgd_df are present in nzgd_metadata_past_and_present_df"
    )

print()

all_nzgd_metadata_df = pd.concat(
    [nzgd_metadata_past_and_present_df, nzgd_metadata_currently_available_on_nzgd_df],
    ignore_index=True,
)
all_nzgd_metadata_df = all_nzgd_metadata_df.drop_duplicates(
    subset="nzgd_id", keep="first"
)
all_nzgd_metadata_df = all_nzgd_metadata_df.sort_values(
    by="nzgd_id", key=natsort_keygen()
)

# Check for rows with missing nztm_x or nztm_y coordinates
missing_nztm_x = all_nzgd_metadata_df["nztm_x"].isna()
missing_nztm_y = all_nzgd_metadata_df["nztm_y"].isna()
missing_nztm_x_or_y = missing_nztm_x | missing_nztm_y

if missing_nztm_x_or_y.any():
    print(
        f"Found {missing_nztm_x_or_y.sum()} rows with missing nztm_x or nztm_y coordinates:"
    )
    print(f"  - {missing_nztm_x.sum()} rows missing nztm_x")
    print(f"  - {missing_nztm_y.sum()} rows missing nztm_y")
    print(f"  - {(missing_nztm_x & missing_nztm_y).sum()} rows missing both")

    rows_with_missing_coords = all_nzgd_metadata_df[missing_nztm_x_or_y]
    print("\nRows with missing coordinates (showing nzgd_id):")
    print(rows_with_missing_coords[["nzgd_id", "nztm_x", "nztm_y"]].to_string())
else:
    print("All rows have valid nztm_x and nztm_y coordinates.")

print()

# Check for rows with missing region, district, city, or suburb
missing_region = all_nzgd_metadata_df["region"].isna()
missing_district = all_nzgd_metadata_df["district"].isna()
missing_city = all_nzgd_metadata_df["city"].isna()
missing_suburb = all_nzgd_metadata_df["suburb"].isna()
missing_location_info = (
    missing_region | missing_district | missing_city | missing_suburb
)

if missing_location_info.any():
    print(
        f"Found {missing_location_info.sum()} rows with missing location information:"
    )
    print(f"  - {missing_region.sum()} rows missing region")
    print(f"  - {missing_district.sum()} rows missing district")
    print(f"  - {missing_city.sum()} rows missing city")
    print(f"  - {missing_suburb.sum()} rows missing suburb")

    rows_with_missing_location = all_nzgd_metadata_df[missing_location_info]
    print(
        "\nRows with missing location information (showing nzgd_id and location columns):"
    )
    print(
        rows_with_missing_location[
            ["nzgd_id", "region", "district", "city", "suburb"]
        ].to_string()
    )
else:
    print("All rows have valid region, district, city, and suburb values.")

# Paths to GeoTIFF files
westerhoff_2018_model_path = Path(
    "/home/arr65/data/nzgd/resources/national_water_table_model_data_2022/nwt_wtd_NZ_20220825.tif"
)
nlm_gwd_path = Path(
    "/home/arr65/data/national_liquefaction_model/NLM files/v2025.0_rc4/GW/NLM_gwd.tif"
)
nlm_gw_std_path = Path(
    "/home/arr65/data/national_liquefaction_model/NLM files/v2025.0_rc4/GW/NLM_gw_std_wo_seasonal.tif"
)

# Check which rows need sampling
missing_westerhoff = all_nzgd_metadata_df["model_gwl_westerhoff_2018"].isna()
missing_nlm_gwl = all_nzgd_metadata_df["nlm_gwl_m"].isna()
missing_nlm_gwl_stddev = all_nzgd_metadata_df["nlm_gwl_stddev_m"].isna()

# Get rows that need sampling and have valid coordinates
has_coords = (
    all_nzgd_metadata_df["nztm_x"].notna() & all_nzgd_metadata_df["nztm_y"].notna()
)

# Sample from Westerhoff 2018 model for rows missing model_gwl_westerhoff_2018
needs_westerhoff = missing_westerhoff & has_coords
if needs_westerhoff.any():
    print(f"Sampling from Westerhoff 2018 model for {needs_westerhoff.sum()} rows...")
    westerhoff_rows = all_nzgd_metadata_df[needs_westerhoff]
    westerhoff_nztm_xy_list = [
        np.array([row["nztm_x"], row["nztm_y"]])
        for _, row in westerhoff_rows.iterrows()
    ]

    with rasterio.open(westerhoff_2018_model_path) as dataset:
        westerhoff_values_in_array = list(dataset.sample(westerhoff_nztm_xy_list))

    westerhoff_values = [float(x[0]) for x in westerhoff_values_in_array]
    all_nzgd_metadata_df.loc[needs_westerhoff, "model_gwl_westerhoff_2018"] = (
        westerhoff_values
    )
    print(f"  Sampled {len(westerhoff_values)} values for model_gwl_westerhoff_2018")

# Sample from NLM GWD for rows missing nlm_gwl_m
needs_nlm_gwl = missing_nlm_gwl & has_coords
if needs_nlm_gwl.any():
    print(f"Sampling from NLM GWD for {needs_nlm_gwl.sum()} rows...")
    nlm_gwl_rows = all_nzgd_metadata_df[needs_nlm_gwl]
    nlm_gwl_nztm_xy_list = [
        np.array([row["nztm_x"], row["nztm_y"]]) for _, row in nlm_gwl_rows.iterrows()
    ]

    with rasterio.open(nlm_gwd_path) as dataset:
        nlm_gwd_values_in_array = list(dataset.sample(nlm_gwl_nztm_xy_list))

    nlm_gwd_values = [float(x[0]) for x in nlm_gwd_values_in_array]
    all_nzgd_metadata_df.loc[needs_nlm_gwl, "nlm_gwl_m"] = nlm_gwd_values
    print(f"  Sampled {len(nlm_gwd_values)} values for nlm_gwl_m")

# Sample from NLM GW STD for rows missing nlm_gwl_stddev_m
needs_nlm_gwl_stddev = missing_nlm_gwl_stddev & has_coords
if needs_nlm_gwl_stddev.any():
    print(
        f"Sampling from NLM GW STD (without seasonal) for {needs_nlm_gwl_stddev.sum()} rows..."
    )
    nlm_gwl_stddev_rows = all_nzgd_metadata_df[needs_nlm_gwl_stddev]
    nlm_gwl_stddev_nztm_xy_list = [
        np.array([row["nztm_x"], row["nztm_y"]])
        for _, row in nlm_gwl_stddev_rows.iterrows()
    ]

    with rasterio.open(nlm_gw_std_path) as dataset:
        nlm_gw_std_values_in_array = list(dataset.sample(nlm_gwl_stddev_nztm_xy_list))

    nlm_gw_std_values = [float(x[0]) for x in nlm_gw_std_values_in_array]
    all_nzgd_metadata_df.loc[needs_nlm_gwl_stddev, "nlm_gwl_stddev_m"] = (
        nlm_gw_std_values
    )
    print(f"  Sampled {len(nlm_gw_std_values)} values for nlm_gwl_stddev_m")

if needs_westerhoff.any() or needs_nlm_gwl.any() or needs_nlm_gwl_stddev.any():
    print("Sampling complete.")
else:
    print("No missing values found that need sampling.")

all_nzgd_metadata_df.to_csv(
    constants.RESOURCE_PATH
    / "nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl.csv.gz",
    compression="gzip",
    index=False,
)
