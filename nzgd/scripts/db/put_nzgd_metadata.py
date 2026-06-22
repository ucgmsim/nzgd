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

    # pandas NaN must reach sqlite as NULL, not a REAL NaN. Convert the whole
    # frame to object with None in place of every NaN before inserting.
    metadata_df = metadata_df.astype(object).where(pd.notna(metadata_df), None)

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
