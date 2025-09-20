"""Create an empty SQLite database and fill support tables."""

import sqlite3

import pandas as pd
from tqdm import tqdm

from nzgd import constants
from nzgd.db import orm

def serialize_record_metadata(metadata_df: pd.DataFrame, conn: sqlite3.Connection):
    cursor = conn.cursor()

    # Fetch location id mappings from the database
    location_categories = ["region", "district", "city", "suburb"]
    location_id_maps = {}
    for category in location_categories:
        cursor.execute(f"SELECT {category}_id, name FROM {category}")
        location_id_maps[category] = {name: id_ for id_, name in cursor.fetchall()}

    # Map string columns in metadata_df to their corresponding ids
    for category in location_categories:
        metadata_df[f"{category}_id_db"] = metadata_df[category].map(
            location_id_maps[category],
        )

    for _, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
        cursor.execute(
            """
            INSERT OR REPLACE INTO nzgdrecord (nzgd_id, type_prefix, original_investigation_name, investigation_date, published_date, latitude, longitude, region_id, district_id, city_id, suburb_id, model_vs30_foster_2019, model_vs30_stddev_foster_2019, model_gwl_westerhoff_2018)
            VALUES (?, ? , ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                int(row["nzgd_id"]),
                row["Type"],
                row["InvestigationId"],
                row["CreatedOn"],
                row["LastModifiedOn"],
                row["Latitude"],
                row["Longitude"],
                row["region_id_db"],
                row["district_id_db"],
                row["city_id_db"],
                row["suburb_id_db"],
                row["model_vs30_foster_2019"],
                row["model_vs30_std_foster_2019"],
                row["model_gwl_westerhoff_2018"],
            ),
        )


def serialize_correlation_tables(conn: sqlite3.Connection):
    """Serialize correlation strings to the SQLite database.

    This function processes the vs30 DataFrame to extract unique correlation strings
    and inserts or replaces them into the corresponding tables in the SQLite database.

    Parameters
    ----------
    vs30_df : pd.DataFrame
        A DataFrame containing vs30 correlation data with columns 'cpt_vs_correlation_id_integer',
        'cpt_vs_correlation', 'vs30_correlation_id_integer', and 'vs30_correlation'.
    conn : sqlite3.Connection
        A connection object to the SQLite database.

    Returns
    -------
    None
        This function does not return anything.

    """
    cursor = conn.cursor()

    # Populate CPTToVsCorrelation
    for name, id_ in constants.CPT_TO_VS_CORRELATION_TO_ID.items():
        cursor.execute(
            "INSERT OR REPLACE INTO CPTToVsCorrelation (id, value) VALUES (?, ?)",
            (id_, name),
        )

    # Populate SPTToVsCorrelation
    for name, id_ in constants.SPT_TO_VS_CORRELATION_TO_ID.items():
        cursor.execute(
            "INSERT OR REPLACE INTO SPTToVsCorrelation (id, value) VALUES (?, ?)",
            (id_, name),
        )

    # Populate VsToVs30Correlation
    for name, id_ in constants.VS_TO_VS30_CORRELATION_TO_ID.items():
        cursor.execute(
            "INSERT OR REPLACE INTO VsToVs30Correlation (id, value) VALUES (?, ?)",
            (id_, name),
        )

    conn.commit()


def serialize_investigation_type_table(conn: sqlite3.Connection):
    """Serialize correlation strings to the SQLite database.

    This function processes the vs30 DataFrame to extract unique correlation strings
    and inserts or replaces them into the corresponding tables in the SQLite database.

    Parameters
    ----------
    conn : sqlite3.Connection
        A connection object to the SQLite database.

    Returns
    -------
    None
        This function does not return anything.

    """
    cursor = conn.cursor()

    # Populate CPTToVsCorrelation
    for name, id_ in constants.TYPE_TO_ID.items():
        cursor.execute(
            "INSERT OR REPLACE INTO Type (id, value) VALUES (?, ?)",
            (id_, name),
        )

    conn.commit()


def serialize_spt_hammer_type_table(conn: sqlite3.Connection):
    """Serialize SPT hammer types to the SQLite database.

    Parameters
    ----------
    conn : sqlite3.Connection
        A connection object to the SQLite database.

    Returns
    -------
    None
        This function does not return anything.

    """
    cursor = conn.cursor()

    # Populate CPTToVsCorrelation
    for name, id_ in constants.HAMMER_TYPE_TO_ID.items():
        cursor.execute(
            "INSERT OR REPLACE INTO SPTToVs30HammerType (id, value) VALUES (?, ?)",
            (id_, name),
        )

    conn.commit()


def serialize_spt_soil_type_table(
    conn: sqlite3.Connection,
):
    """Serialize soil type data to the SQLite database.

    Parameters
    ----------
    conn : sqlite3.Connection
        A connection object to the SQLite database.

    Returns
    -------
    None
        This function does not return anything.

    """
    cursor = conn.cursor()

    for value, value_id in tqdm(
        constants.SOIL_TYPE_TO_ID.items(),
        total=len(constants.SOIL_TYPE_TO_ID),
    ):
        cursor.execute(
            """
            INSERT OR REPLACE INTO soiltypes (id, name)
            VALUES (?, ?)
        """,
            (value_id, value),
        )


def serialize_cpt_termination_reason_table(
    conn: sqlite3.Connection,
):
    """Serialize CPT termination reason data to the SQLite database.

    Parameters
    ----------
    cpt_termination_reason_id_map : pd.DataFrame
        A DataFrame containing CPT termination reason data with columns 'termination_reason_id' and 'termination_reason'.

    Returns
    -------
    None
        This function does not return anything.

    """
    cursor = conn.cursor()

    for value, value_id in tqdm(
        constants.CPT_TERMINATION_REASON_TO_ID.items(),
        total=len(constants.CPT_TERMINATION_REASON_TO_ID),
    ):
        cursor.execute(
            """
            INSERT OR REPLACE INTO terminationreason (id, value)
            VALUES (?, ?)
        """,
            (value_id, value),
        )


def serialize_ground_water_level_method_table(
    conn: sqlite3.Connection,
):
    """Serialize ground water level method data to the SQLite database.

    Parameters
    ----------
    conn : sqlite3.Connection
        A connection object to the SQLite database.

    Returns
    -------
    None
        This function does not return anything.

    """
    cursor = conn.cursor()

    for value, value_id in tqdm(
        constants.GROUND_WATER_LEVEL_METHOD_TO_ID.items(),
        total=len(constants.GROUND_WATER_LEVEL_METHOD_TO_ID),
    ):
        cursor.execute(
            """
            INSERT OR REPLACE INTO CPTGroundWaterLevelMethod (id, value)
            VALUES (?, ?)
        """,
            (value_id, value),
        )


def serialize_location_name_tables(metadata_df: pd.DataFrame, conn: sqlite3.Connection):
    """Serialize location strings to the SQLite database.

    This function processes the metadata DataFrame to extract unique location strings
    for different location categories and inserts or replaces them into the corresponding
    tables in the SQLite database.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        A DataFrame containing metadata with location information.
    conn : sqlite3.Connection
        A connection object to the SQLite database.

    Returns
    -------
    None
        This function does not return anything.

    """
    cursor = conn.cursor()

    location_categories = ["region", "district", "city", "suburb"]

    for location_category in location_categories:
        print(f"serializing {location_category} table")
        location_table_series = (
            metadata_df[location_category]
            .dropna()
            .drop_duplicates(keep="first")
            .sort_values()
            .reset_index(drop=True)
        )

        # Create DataFrame with id and values columns
        location_table_df = pd.DataFrame(
            {
                "id": location_table_series.index + 1,
                "value": location_table_series.values,
            },
        )

        for _, row in tqdm(
            location_table_df.iterrows(),
            total=location_table_df.shape[0],
        ):
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {location_category} (id, value)
                VALUES (?, ?)
            """,
                (row["id"], row["value"]),
            )


if __name__ == "__main__":
    metadata_from_location_coordinates = pd.read_csv(
        "/home/arr65/src/nzgd_data_extraction/nzgd_data_extraction/resources/nzgd_metadata_from_coordinates_22_august_2025.csv",
    )

    orm.initialize_db()

    print()

    with sqlite3.connect(constants.OUTPUT_DB_PATH) as db:
        ### needs to be in the db for Jake's SPT mining code to work
        serialize_spt_soil_type_table(
            db,
        )
        serialize_cpt_termination_reason_table(
            db,
        )
        serialize_ground_water_level_method_table(
            db,
        )

        serialize_correlation_tables(db)

        serialize_spt_hammer_type_table(db)

        serialize_investigation_type_table(db)

        serialize_location_name_tables(metadata_from_location_coordinates, db)

        ## Support tables end here

        # serialize_record_metadata(meta_data_df, db)