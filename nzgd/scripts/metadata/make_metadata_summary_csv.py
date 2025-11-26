"""Script to create the metadata summary CSV file from the UC NZGD SQLite database and the NZGD index file.

For a NZGD IDs with several CPT IDs, metadata from each CPT is combined into a single row.
If multiple CPT IDs all have valid metadata values, the first valid value it finds is used.

Many columns are renamed from the SQLite database and the NZGD index file so they match and can be concatenated.
"""

import sqlite3
from collections.abc import Iterator
from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd
from natsort import natsort_keygen

from nzgd import constants

if TYPE_CHECKING:
    from pandas import DataFrame

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(
        iterable: Any, desc: str | None = None, total: int | None = None
    ) -> Iterator[Any]:
        """Fallback progress bar when tqdm is not available."""
        if desc:
            print(f"{desc}...")
        return iter(iterable)


nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl_df = pd.read_csv(
    constants.RESOURCE_PATH
    / "nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl.csv.gz",
    compression="gzip",
)
nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl_df = (
    nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl_df.rename(
        columns={
            "State": "availability_status",
            "InvestigationId": "original_investigation_name",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "CreatedOn": "investigation_date",
            "LastModifiedOn": "published_date",
            "nlm_gwl_m": "model_gwl_nlm_2025_m",
            "nlm_gwl_stddev_m": "model_gwl_nlm_2025_stddev_m",
        }
    )
)


print("Loading data from database...")
with sqlite3.connect(constants.OUTPUT_DB_PATH) as db:
    print("  Loading cptreport data...")
    my_extractions_df = pd.read_sql_query("SELECT * FROM cptreport", db)
    print(f"    Loaded {len(my_extractions_df)} CPT reports")

    print("  Loading CPT metadata summary...")
    cpt_metadata_summary_df = pd.read_sql_query(
        """
        SELECT 
            cr.cpt_id,
            cr.nzgd_id,
            cr.max_depth_m,
            cr.min_depth_m,
            cr.extracted_gwl_m,
            gwl.value AS gwl_method,
            cr.tip_net_area_ratio,
            cr.predrill_depth_m,
            cr.source_file,
            nz.latitude,
            nz.longitude,
            nz.model_vs30_foster_2019,
            nz.model_vs30_stddev_foster_2019,
            nz.model_gwl_westerhoff_2018_m,
            nz.model_gwl_nlm_2025_m,
            nz.model_gwl_nlm_2025_stddev_m,
            nz.original_investigation_name,
            nz.investigation_date,
            nz.published_date,
            r.value AS region_name,
            d.value AS district_name,
            c.value AS city_name,
            s.value AS suburb_name,
            t.value AS type,
            tr.value AS termination_reason,
            cr.has_cpt_data
        FROM cptreport cr
        LEFT JOIN nzgdrecord nz ON cr.nzgd_id = nz.nzgd_id
        LEFT JOIN type t ON nz.type_id = t.id
        LEFT JOIN region r ON nz.region_id = r.id
        LEFT JOIN district d ON nz.district_id = d.id
        LEFT JOIN city c ON nz.city_id = c.id
        LEFT JOIN suburb s ON nz.suburb_id = s.id
        LEFT JOIN terminationreason tr ON cr.termination_reason_id = tr.id
        LEFT JOIN cptgroundwaterlevelmethod gwl ON cr.gwl_method_id = gwl.id
        """,
        db,
    )
    print(f"    Loaded {len(cpt_metadata_summary_df)} CPT metadata rows")

    print("  Loading SPT metadata summary...")
    spt_metadata_summary_df = pd.read_sql_query(
        """
        SELECT 
            sr.*,
            nz.latitude,
            nz.longitude,
            nz.model_vs30_foster_2019,
            nz.model_vs30_stddev_foster_2019,
            nz.model_gwl_westerhoff_2018_m,
            nz.model_gwl_nlm_2025_m,
            nz.model_gwl_nlm_2025_stddev_m,
            nz.original_investigation_name,
            nz.investigation_date,
            nz.published_date,
            r.value AS region_name,
            d.value AS district_name,
            c.value AS city_name,
            s.value AS suburb_name,
            t.value AS type
        FROM sptreport sr
        LEFT JOIN nzgdrecord nz ON sr.nzgd_id = nz.nzgd_id
        LEFT JOIN type t ON nz.type_id = t.id
        LEFT JOIN region r ON nz.region_id = r.id
        LEFT JOIN district d ON nz.district_id = d.id
        LEFT JOIN city c ON nz.city_id = c.id
        LEFT JOIN suburb s ON nz.suburb_id = s.id
        """,
        db,
    )
    spt_metadata_summary_df["has_extracted_spt_trace"] = 1
    print(f"    Loaded {len(spt_metadata_summary_df)} SPT metadata rows")
print("Data loading complete.\n")


def fill_reference_row_per_nzgd_id(
    df: "DataFrame", name: str = "DataFrame"
) -> "DataFrame":
    """
    For each nzgd_id, find the row with the most filled columns (reference row)
    and fill missing values from other rows with the same nzgd_id.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame to process.
    name : str
        Name of the DataFrame for progress reporting.

    Returns
    -------
    DataFrame
        A DataFrame with one row per nzgd_id.
    """
    if df.empty or "nzgd_id" not in df.columns:
        return df

    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Get unique nzgd_ids for progress tracking
    unique_nzgd_ids = df["nzgd_id"].unique()
    total_groups = len(unique_nzgd_ids)

    print(f"Processing {name}: {total_groups} unique nzgd_ids from {len(df)} rows...")

    result_rows = []
    filled_count = 0

    # Group by nzgd_id with progress bar
    for nzgd_id in tqdm(
        unique_nzgd_ids, desc=f"  Processing {name}", total=total_groups
    ):
        group_df = df[df["nzgd_id"] == nzgd_id]

        if group_df.empty:
            continue

        # Find the row with the most non-null values (reference row)
        non_null_counts = group_df.notna().sum(axis=1)
        reference_idx = non_null_counts.idxmax()
        reference_row = group_df.loc[reference_idx].copy()

        # Fill missing values in reference row from other rows
        for col in group_df.columns:
            if pd.isna(reference_row[col]):
                # Find non-null values in this column from other rows
                other_values = group_df.loc[
                    group_df.index != reference_idx, col
                ].dropna()
                if not other_values.empty:
                    # Use the first non-null value found
                    reference_row[col] = other_values.iloc[0]
                    filled_count += 1

        result_rows.append(reference_row)

    print(f"  Completed: {len(result_rows)} rows created, {filled_count} values filled")
    print(f"  Original rows: {len(df)}, Final rows: {len(result_rows)}")

    if result_rows:
        return pd.DataFrame(result_rows)
    else:
        return df.iloc[0:0].copy()  # Return empty DataFrame with same structure


# Process both DataFrames
print("\n" + "=" * 60)
print("Processing DataFrames to fill reference rows")
print("=" * 60)
cpt_metadata_summary_df_filled = fill_reference_row_per_nzgd_id(
    cpt_metadata_summary_df, name="CPT metadata"
)
print()
spt_metadata_summary_df_filled = fill_reference_row_per_nzgd_id(
    spt_metadata_summary_df, name="SPT metadata"
)
print("\n" + "=" * 60)
print("Processing complete!")
print("=" * 60)
print(f"CPT summary: {len(cpt_metadata_summary_df_filled)} rows")
print(f"SPT summary: {len(spt_metadata_summary_df_filled)} rows")
print()

metadata_summary_from_db_df = pd.concat(
    [cpt_metadata_summary_df_filled, spt_metadata_summary_df_filled], ignore_index=True
)
# Remove the simple `type` column (CPT or BH) that comes from nzgdrecord in the SQLite database so that the
# more detailed "Type" and "TypeDisplay" columns can be used from the NZGD index file.
metadata_summary_from_db_df = metadata_summary_from_db_df.drop(columns=["type"])

metadata_summary_from_db_df = metadata_summary_from_db_df.drop(
    columns=["cpt_id", "spt_id"]
)


metadata_summary_from_db_df = metadata_summary_from_db_df.rename(
    columns={
        "tip_net_area_ratio": "cpt_tip_net_area_ratio",
        "predrill_depth_m": "cpt_predrill_depth_m",
        "termination_reason": "cpt_termination_reason",
        "efficiency": "spt_efficiency",
        "region_name": "region",
        "district_name": "district",
        "city_name": "city",
        "suburb_name": "suburb",
    }
)

# Create a copy with only nzgd_id and columns not in metadata_summary_from_db_df
db_columns = set(metadata_summary_from_db_df.columns)
# Preserve the order from the original DataFrame
columns_not_in_db = [
    col
    for col in nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl_df.columns
    if col not in db_columns
]
# Always include nzgd_id first, then other columns in their original order
columns_to_select = ["nzgd_id"] + [col for col in columns_not_in_db if col != "nzgd_id"]
nzgd_metadata_additional_columns_df = (
    nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl_df[
        columns_to_select
    ].copy()
)


all_metadata_for_records_in_db_df = metadata_summary_from_db_df.merge(
    nzgd_metadata_additional_columns_df, on="nzgd_id", how="left"
)

nzgd_ids_in_db = set(all_metadata_for_records_in_db_df["nzgd_id"])
nzgd_metadata_not_in_db_df = (
    nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl_df[
        ~nzgd_metadata_for_past_and_current_nzgd_investigations_with_nlm_gwl_df[
            "nzgd_id"
        ].isin(nzgd_ids_in_db)
    ].copy()
)

all_metadata_from_db_and_index = pd.concat(
    [all_metadata_for_records_in_db_df, nzgd_metadata_not_in_db_df], ignore_index=True
)


all_metadata_from_db_and_index = all_metadata_from_db_and_index.sort_values(
    by="nzgd_id", key=natsort_keygen()
)

all_metadata_from_db_and_index = all_metadata_from_db_and_index.rename(
    columns={"has_cpt_data": "has_extracted_cpt_trace"}
)

output_filename = f"uc_nzgd_metadata_summary_{date.today().isoformat()}.csv"
output_path = constants.OUTPUT_DB_PATH.parent / output_filename

all_metadata_from_db_and_index.to_csv(output_path, index=False)

print("Metadata summary saved to:")
print(output_path)
print()

print("Processing complete!")
print("=" * 60)
print(f"Metadata summary: {len(all_metadata_from_db_and_index)} rows")
print()
