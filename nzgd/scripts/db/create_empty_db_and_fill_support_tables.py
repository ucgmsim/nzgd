"""Create an empty SQLite database and fill support tables."""

import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from nzgd import constants
from nzgd.db import orm


def initialize_database_at_path(db_path: Path):
    """Initialize a database at the specified path with all required tables.
    
    Parameters
    ----------
    db_path : Path
        The path where the database should be created.
    """
    # Create the database file if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary database connection for initialization
    temp_db = sqlite3.connect(str(db_path))
    
    # Create all tables using the ORM models
    # We need to temporarily change the database path in the ORM
    original_db_path = orm.db.database
    orm.db.init(str(db_path))
    
    try:
        with orm.db:
            orm.db.create_tables([
                orm.Type,
                orm.Region,
                orm.District,
                orm.City,
                orm.Suburb,
                orm.CPTToVsCorrelation,
                orm.SPTToVsCorrelation,
                orm.VsToVs30Correlation,
                orm.SPTToVs30HammerType,
                orm.TerminationReason,
                orm.CPTGroundWaterLevelMethod,
                orm.SoilTypes,
                orm.NZGDRecord,
                orm.SPTReport,
                orm.SoilMeasurements,
                orm.SoilMeasurementSoilType,
                orm.SPTMeasurements,
                orm.CPTReport,
                orm.CPTMeasurements,
                orm.CPTVs30Estimates,
                orm.SPTVs30Estimates,
            ])
    finally:
        # Restore the original database path
        orm.db.init(original_db_path)
    
    temp_db.close()

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


def populate_database(db_path: Path, metadata_df: pd.DataFrame):
    """Populate a database with support tables and metadata.
    
    Parameters
    ----------
    db_path : Path
        The path to the database to populate.
    metadata_df : pd.DataFrame
        The metadata DataFrame containing location information.
    """
    print(f"Populating database at {db_path}")
    
    with sqlite3.connect(str(db_path)) as db:
        # needs to be in the db for Jake's SPT mining code to work
        serialize_spt_soil_type_table(db)
        serialize_cpt_termination_reason_table(db)
        serialize_ground_water_level_method_table(db)

        serialize_correlation_tables(db)

        serialize_spt_hammer_type_table(db)

        serialize_investigation_type_table(db)

        serialize_location_name_tables(metadata_df, db)

if __name__ == "__main__":
    metadata_from_location_coordinates = pd.read_csv(
        constants.INDEX_FILE_PATH,
    )

    # List of all database paths to create
    database_paths = [
        constants.OUTPUT_DB_PATH,
        constants.TEMP_SPT_PDF_DB_PATH,
        constants.TEMP_SPT_AGS_DB_PATH,
    ]

    print("Creating and initializing databases...")
    for db_path in database_paths:
        print(f"Creating database at {db_path}")
        initialize_database_at_path(db_path)
    
    print("Populating databases with support tables...")
    for db_path in database_paths:
        populate_database(db_path, metadata_from_location_coordinates)
    
    print("Database creation and population completed successfully!")
