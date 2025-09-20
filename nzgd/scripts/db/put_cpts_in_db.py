"""Script to serialize only CPT data into a SQLite database using existing cpt_id mapping and metadata summary."""

import sqlite3
from pathlib import Path

import natsort
import pandas as pd
from tqdm import tqdm

from nzgd import constants


def serialize_cpt_reports(
    extracted_cpt_dir: Path,
    cpt_id_df: pd.DataFrame,
    model_gwl_df: pd.DataFrame,
    conn: sqlite3.Connection,
):
    """Serialize only CPT data to the SQLite database.

    Parameters
    ----------
    extracted_cpt_path : list
        The path to the extracted CPT data files.
    cpt_id_df : pd.DataFrame
        DataFrame mapping (nzgd_id, investigation_number) to cpt_id.
    conn : sqlite3.Connection
        SQLite database connection.

    """
    cursor = conn.cursor()

    extracted_cpt_files = natsort.natsorted(list(extracted_cpt_dir.glob("*.parquet")))
    nzgd_ids_with_extractions = {int(cpt_file.stem) for cpt_file in extracted_cpt_files}

    for _, cpt_id_df_row in tqdm(cpt_id_df.iterrows(), total=len(cpt_id_df)):
        # Initialize variables to None
        termination_reason_id = None
        gwl_method_id = None
        max_depth = None
        min_depth = None
        has_cpt_data = 0
        cpt_data_duplicate_of_cpt_id = None
        did_explicit_unit_conversion = 0
        did_inferred_unit_conversion = 0

        # Retrieve values from the cpt_id index file
        if pd.notna(cpt_id_df_row["termination_reason"]):
            termination_reason_id = constants.CPT_TERMINATION_REASON_TO_ID[
                cpt_id_df_row["termination_reason"]
            ]

        if pd.notna(cpt_id_df_row["gwl_method"]):
            gwl_method_id = constants.GROUND_WATER_LEVEL_METHOD_TO_ID[
                cpt_id_df_row["gwl_method"]
            ]

        # Get the Westerhoff 2018 model GWL for this nzgd_id
        model_gwl = model_gwl_df[model_gwl_df["nzgd_id"] == cpt_id_df_row["nzgd_id"]][
            "model_gwl_westerhoff_2018"
        ].iloc[0]

        _, source_file_name, source_sheet_name = cpt_id_df_row[
            "nzgd_id_AND_filename_AND_sheetname"
        ].split("_AND_")

        # Retrieve values from the extracted CPT data
        if cpt_id_df_row["nzgd_id"] in nzgd_ids_with_extractions:
            cpt_data_df = pd.read_parquet(
                extracted_cpt_dir / f"{cpt_id_df_row['nzgd_id']}.parquet",
            )

            cpt_data_df_for_sheet = cpt_data_df[
                (cpt_data_df["file_name"] == source_file_name)
                & (cpt_data_df["sheet_name"] == source_sheet_name)
            ]

            if len(cpt_data_df_for_sheet) > 0:
                max_depth = cpt_data_df_for_sheet["Depth"].max()
                min_depth = cpt_data_df_for_sheet["Depth"].min()
                has_cpt_data = 1

                explicit_unit_conversions = cpt_data_df_for_sheet[
                    "explicit_unit_conversions"
                ].unique()

                if (
                    len(explicit_unit_conversions) > 0
                    and len(explicit_unit_conversions[0]) > 0
                ):
                    did_explicit_unit_conversion = 1

                inferred_unit_conversions = cpt_data_df_for_sheet[
                    "inferred_unit_conversions"
                ].unique()

                if (
                    len(inferred_unit_conversions) > 0
                    and len(inferred_unit_conversions[0]) > 0
                ):
                    did_inferred_unit_conversion = 1

            else:
                removed_duplicates = cpt_data_df["removed_duplicates"].unique()
                if len(removed_duplicates) > 0:
                    removed_duplicates_str = removed_duplicates[0]
                    if (removed_duplicates_str is not None) and (
                        f"{source_file_name}_AND_{source_sheet_name}"
                        in removed_duplicates_str
                    ):
                        max_depth = cpt_data_df["Depth"].max()
                        min_depth = cpt_data_df["Depth"].min()

                        file_name_of_cpt_data = cpt_data_df["file_name"].unique()[0]
                        sheet_name_of_cpt_data = cpt_data_df["sheet_name"].unique()[0]

                        cpt_data_duplicate_of_cpt_id = int(
                            cpt_id_df[
                                cpt_id_df["nzgd_id_AND_filename_AND_sheetname"]
                                == f"{cpt_id_df_row['nzgd_id']}_AND_{file_name_of_cpt_data}_AND_{sheet_name_of_cpt_data}"
                            ]["cpt_id"].iloc[0],
                        )

        # Insert into cptreport
        cursor.execute(
            """
            INSERT OR REPLACE INTO cptreport (
                cpt_id, nzgd_id, max_depth, min_depth, gwl, gwl_method_id, gwl_residual,
                tip_net_area_ratio, termination_reason_id, has_cpt_data, cpt_data_duplicate_of_cpt_id,
                did_explicit_unit_conversion, did_inferred_unit_conversion, source_file
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cpt_id_df_row["cpt_id"],
                cpt_id_df_row["nzgd_id"],
                max_depth,
                min_depth,
                cpt_id_df_row["ground_water_level"],
                gwl_method_id,
                cpt_id_df_row["ground_water_level"] - model_gwl,
                cpt_id_df_row["tip_net_area_ratio"],
                termination_reason_id,
                has_cpt_data,
                cpt_data_duplicate_of_cpt_id,
                did_explicit_unit_conversion,
                did_inferred_unit_conversion,
                f"{source_file_name}_sheet_{source_sheet_name}",
            ),
        )


def serialize_cpt_data_arrays(
    extracted_cpt_dir: Path,
    cpt_id_df: pd.DataFrame,
    conn: sqlite3.Connection,
):
    """Serialize the CPT data arrays to the SQLite database.

    Parameters
    ----------
    extracted_cpt_path : list
        The path to the extracted CPT data files.
    cpt_id_df : pd.DataFrame
        DataFrame mapping (nzgd_id, investigation_number) to cpt_id.
    conn : sqlite3.Connection
        SQLite database connection.

    """
    cursor = conn.cursor()

    extracted_cpt_files = natsort.natsorted(list(extracted_cpt_dir.glob("*.parquet")))
    nzgd_ids_with_extractions = {int(cpt_file.stem) for cpt_file in extracted_cpt_files}

    cpt_measurements_id = 1

    for _, cpt_id_df_row in tqdm(cpt_id_df.iterrows(), total=len(cpt_id_df)):
        _, source_file_name, source_sheet_name = cpt_id_df_row[
            "nzgd_id_AND_filename_AND_sheetname"
        ].split("_AND_")

        # Retrieve values from the extracted CPT data
        if cpt_id_df_row["nzgd_id"] in nzgd_ids_with_extractions:
            cpt_data_df = pd.read_parquet(
                extracted_cpt_dir / f"{cpt_id_df_row['nzgd_id']}.parquet",
            )

            cpt_data_df_for_sheet = cpt_data_df[
                (cpt_data_df["file_name"] == source_file_name)
                & (cpt_data_df["sheet_name"] == source_sheet_name)
            ]

            if len(cpt_data_df_for_sheet) > 0:
                # Insert CPT measurements
                for _, row in cpt_data_df_for_sheet.iterrows():
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO cptmeasurements (
                            measurement_id, cpt_id, depth, qc, fs, u2
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            cpt_measurements_id,
                            cpt_id_df_row["cpt_id"],
                            row["Depth"],
                            row["qc"],
                            row["fs"],
                            row["u"],
                        ),
                    )
                    cpt_measurements_id += 1


if __name__ == "__main__":
    cpt_id_and_single_values_df = pd.read_csv(
        "/home/arr65/src/nzgd_data_extraction/nzgd_data_extraction/nzgd_sqlite/cpt_ids_multi_single_values.csv",
    )

    model_gwl_df = pd.read_csv(
        "/home/arr65/src/nzgd_data_extraction/nzgd_data_extraction/resources/nzgd_metadata_from_coordinates_22_august_2025.csv",
    )

    # The metadata (including lat, lon) has been lost for nzgd_id = 187732.
    # Unknown how this happened. Perhaps NZGD removed it after the files were
    # downloaded. Only keep records for which we have metadata.
    cpt_id_and_single_values_df = cpt_id_and_single_values_df[
        cpt_id_and_single_values_df["nzgd_id"].isin(model_gwl_df["nzgd_id"])
    ]

    extracted_cpt_dir = Path(
        "/home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/extracted_data_per_record",
    )

    # cpt_id_and_single_values_df = cpt_id_and_single_values_df.iloc[124054 : 124054 + 1]
    # cpt_id_and_single_values_df = cpt_id_and_single_values_df.iloc[0:10]

    output_path = Path(
        constants.OUTPUT_DB_PATH,
    )

    with sqlite3.connect(output_path) as db:
        serialize_cpt_reports(
            extracted_cpt_dir,
            cpt_id_and_single_values_df,
            model_gwl_df,
            db,
        )

    with sqlite3.connect(output_path) as db:
        serialize_cpt_data_arrays(
            extracted_cpt_dir,
            cpt_id_and_single_values_df,
            db,
        )
