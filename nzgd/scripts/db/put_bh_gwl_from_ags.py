"""Put borehole groundwater levels extracted from AGS files into the database."""

import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from nzgd import constants


def serialize_spt_gwl_data(
    metadata_df: pd.DataFrame,
    conn: sqlite3.Connection,
):
    """Serialize SPT ground water level data to the SQLite database.

    Parameters
    ----------
    spt_gwl_df : pd.DataFrame
        A DataFrame containing SPT ground water level data with columns 'nzgd_id_integer',
        'investigation_number', 'measured_gwl', 'measured_gwl_minus_model_gwl', and 'gwl_method_id_integer'.
    conn : sqlite3.Connection
        A connection object to the SQLite database.

    Returns
    -------
    None
        This function does not return anything.

    """
    sptreport_df = pd.read_sql_query("SELECT * FROM sptreport", conn)

    spt_nzgd_ids = set(sptreport_df["nzgd_id"].unique().tolist())

    cursor = conn.cursor()

    for _, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
        if row["nzgd_id"] not in spt_nzgd_ids:
            continue

        cursor.execute(
            """
            UPDATE sptreport
            SET extracted_gwl = ?, gwl_residual = ?
            WHERE nzgd_id = ?
            """,
            (row["extracted_gwl"], row["gwl_residual"], row["nzgd_id"]),
        )


if __name__ == "__main__":
    bh_gwl_df = pd.read_csv(
        "/home/arr65/data/nzgd/extracted_single_values/borehole_explicit_ground_water_levels.csv"
    )

    bh_gwl_df["nzgd_id"] = (
        bh_gwl_df["record_name"].str.extract(r"BH_(\d+)", expand=False).astype(int)
    )

    bh_gwl_df.rename(
        columns={
            "adopted_value": "extracted_gwl",
            "measured_minus_model": "gwl_residual",
        },
        inplace=True,
    )

    output_path = Path(
        constants.OUTPUT_DB_PATH,
    )

    with sqlite3.connect(output_path) as db:
        serialize_spt_gwl_data(
            bh_gwl_df,
            db,
        )
