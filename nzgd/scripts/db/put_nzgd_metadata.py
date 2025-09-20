import sqlite3

import pandas as pd
from tqdm import tqdm

from nzgd import constants


def serialize_record_metadata(
    metadata_df: pd.DataFrame, spt_ids: set, cpt_ids: set, conn: sqlite3.Connection
):
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
        else:
            type_id = 2  # for SPT

        cursor.execute(
            """
            INSERT OR REPLACE INTO nzgdrecord (nzgd_id, type_id, latitude, longitude, model_vs30_foster_2019, model_vs30_stddev_foster_2019, model_gwl_westerhoff_2018, original_investigation_name, investigation_date, published_date, region_id, district_id, city_id, suburb_id)
            VALUES (?, ? , ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                int(row["nzgd_id"]),
                type_id,
                row["Latitude"],
                row["Longitude"],
                row["model_vs30_foster_2019"],
                row["model_vs30_std_foster_2019"],
                row["model_gwl_westerhoff_2018"],
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
        constants.RESOURCE_PATH / "nzgd_metadata_from_coordinates_22_august_2025.csv",
    )

    # Filter metadata to keep only rows where nzgd_id is in the database
    metadata_df = metadata_df[metadata_df["nzgd_id"].isin(nzgd_ids_in_db)]

    with sqlite3.connect(constants.OUTPUT_DB_PATH) as db:
        serialize_record_metadata(metadata_df, spt_ids, cpt_ids, db)
