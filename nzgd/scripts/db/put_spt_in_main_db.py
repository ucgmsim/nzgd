"""Copy SPT data from temporary databases to the main database.

This script copies SPT-related data from both temporary SPT databases (AGS and PDF)
into the main database. It handles borehole_id conflicts by incrementing the ID
when necessary.
"""

import sqlite3
from pathlib import Path

from nzgd import constants


def get_next_available_borehole_id(main_conn: sqlite3.Connection) -> int:
    """Get the next available borehole_id in the main database.

    Parameters
    ----------
    main_conn : sqlite3.Connection
        Connection to the main database.

    Returns
    -------
    int
        The next available borehole_id.
    """
    cursor = main_conn.cursor()
    cursor.execute("SELECT MAX(borehole_id) FROM sptreport")
    result = cursor.fetchone()
    max_id = result[0] if result[0] is not None else 0
    return max_id + 1


def copy_spt_reports(
    temp_conn: sqlite3.Connection, main_conn: sqlite3.Connection, source_name: str
) -> dict[int, int]:
    """Copy SPT reports from temporary database to main database.

    Parameters
    ----------
    temp_conn : sqlite3.Connection
        Connection to the temporary database.
    main_conn : sqlite3.Connection
        Connection to the main database.
    source_name : str
        Name of the source (e.g., "AGS" or "PDF") for logging.

    Returns
    -------
    dict[int, int]
        Mapping from original borehole_id to new borehole_id.
    """
    temp_cursor = temp_conn.cursor()
    main_cursor = main_conn.cursor()

    # Get all SPT reports from temporary database
    temp_cursor.execute(
        "SELECT borehole_id, nzgd_id, efficiency, extracted_gwl, gwl_residual, source_file FROM sptreport"
    )
    temp_reports = temp_cursor.fetchall()

    # Get existing borehole_ids in main database
    main_cursor.execute("SELECT borehole_id FROM sptreport")
    existing_borehole_ids = {row[0] for row in main_cursor.fetchall()}

    # Create mapping from original borehole_id to new borehole_id
    borehole_id_mapping = {}
    next_available_id = get_next_available_borehole_id(main_conn)

    for temp_report in temp_reports:
        (
            original_borehole_id,
            nzgd_id,
            efficiency,
            extracted_gwl,
            gwl_residual,
            source_file,
        ) = temp_report

        # Always use next available ID to ensure uniqueness
        # This allows the same nzgd_id to appear multiple times with different borehole_ids
        new_borehole_id = next_available_id
        next_available_id += 1

        if original_borehole_id in existing_borehole_ids:
            print(
                f"Borehole ID {original_borehole_id} already exists, using {new_borehole_id}"
            )
        else:
            print(
                f"Using borehole ID {new_borehole_id} for original {original_borehole_id}"
            )

        borehole_id_mapping[original_borehole_id] = new_borehole_id

        # Insert into main database
        main_cursor.execute(
            """
            INSERT INTO sptreport 
            (borehole_id, nzgd_id, efficiency, extracted_gwl, gwl_residual, source_file)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                new_borehole_id,
                nzgd_id,
                efficiency,
                extracted_gwl,
                gwl_residual,
                source_file if source_file else "",
            ),
        )

    print(f"Copied {len(temp_reports)} SPT reports from {source_name} database")
    return borehole_id_mapping


def copy_spt_measurements(
    temp_conn: sqlite3.Connection,
    main_conn: sqlite3.Connection,
    borehole_id_mapping: dict[int, int],
    source_name: str,
):
    """Copy SPT measurements from temporary database to main database.

    Parameters
    ----------
    temp_conn : sqlite3.Connection
        Connection to the temporary database.
    main_conn : sqlite3.Connection
        Connection to the main database.
    borehole_id_mapping : dict[int, int]
        Mapping from original borehole_id to new borehole_id.
    source_name : str
        Name of the source for logging.
    """
    temp_cursor = temp_conn.cursor()
    main_cursor = main_conn.cursor()

    # Get the next available measurement_id in main database
    main_cursor.execute("SELECT MAX(measurement_id) FROM sptmeasurements")
    result = main_cursor.fetchone()
    next_measurement_id = (result[0] + 1) if result[0] is not None else 1

    # Get all SPT measurements from temporary database
    temp_cursor.execute("SELECT borehole_id, depth, n FROM sptmeasurements")
    temp_measurements = temp_cursor.fetchall()

    for temp_measurement in temp_measurements:
        original_borehole_id, depth, n = temp_measurement

        if original_borehole_id in borehole_id_mapping:
            new_borehole_id = borehole_id_mapping[original_borehole_id]

            # Insert into main database with new measurement_id
            main_cursor.execute(
                """
                INSERT INTO sptmeasurements (measurement_id, borehole_id, depth, n)
                VALUES (?, ?, ?, ?)
            """,
                (next_measurement_id, new_borehole_id, depth, n),
            )
            next_measurement_id += 1

    print(
        f"Copied {len(temp_measurements)} SPT measurements from {source_name} database"
    )


def copy_soil_measurements_and_types(
    temp_conn: sqlite3.Connection,
    main_conn: sqlite3.Connection,
    borehole_id_mapping: dict[int, int],
    source_name: str,
):
    """Copy soil measurements and their associated soil types from temporary database to main database.

    This function combines the copying of soil measurements and soil measurement soil types
    to ensure the same measurement_id is used in both tables.

    Parameters
    ----------
    temp_conn : sqlite3.Connection
        Connection to the temporary database.
    main_conn : sqlite3.Connection
        Connection to the main database.
    borehole_id_mapping : dict[int, int]
        Mapping from original borehole_id to new borehole_id.
    source_name : str
        Name of the source for logging.
    """
    temp_cursor = temp_conn.cursor()
    main_cursor = main_conn.cursor()

    # Get the next available measurement_id in main database
    main_cursor.execute("SELECT MAX(measurement_id) FROM soilmeasurements")
    result = main_cursor.fetchone()
    next_measurement_id = (result[0] + 1) if result[0] is not None else 1

    # Get all soil measurements from temporary database
    temp_cursor.execute(
        "SELECT measurement_id, report_id, top_depth FROM soilmeasurements"
    )
    temp_soil_measurements = temp_cursor.fetchall()

    # Get all soil measurement soil types from temporary database
    temp_cursor.execute(
        "SELECT soil_measurement_id, soil_type_id FROM soilmeasurementsoiltype"
    )
    temp_soil_types = temp_cursor.fetchall()

    # Create mapping from original measurement_id to (report_id, top_depth)
    temp_measurement_map = {row[0]: (row[1], row[2]) for row in temp_soil_measurements}

    # Create mapping from original measurement_id to list of soil_type_ids
    temp_measurement_soil_types = {}
    for row in temp_soil_types:
        original_measurement_id = row[0]
        soil_type_id = row[1]
        if original_measurement_id not in temp_measurement_soil_types:
            temp_measurement_soil_types[original_measurement_id] = []
        temp_measurement_soil_types[original_measurement_id].append(soil_type_id)

    # Mapping from original measurement_id to new measurement_id
    measurement_id_mapping = {}

    # Copy soil measurements and track the measurement_id mapping
    for original_measurement_id, (
        original_borehole_id,
        top_depth,
    ) in temp_measurement_map.items():
        if original_borehole_id in borehole_id_mapping:
            new_borehole_id = borehole_id_mapping[original_borehole_id]

            # Insert into main database with new measurement_id
            main_cursor.execute(
                """
                INSERT INTO soilmeasurements (measurement_id, report_id, top_depth)
                VALUES (?, ?, ?)
            """,
                (next_measurement_id, new_borehole_id, top_depth),
            )

            # Store the mapping
            measurement_id_mapping[original_measurement_id] = next_measurement_id

            # Copy associated soil types using the same measurement_id
            if original_measurement_id in temp_measurement_soil_types:
                for soil_type_id in temp_measurement_soil_types[
                    original_measurement_id
                ]:
                    main_cursor.execute(
                        """
                        INSERT OR IGNORE INTO soilmeasurementsoiltype 
                        (soil_measurement_id, soil_type_id)
                        VALUES (?, ?)
                    """,
                        (next_measurement_id, soil_type_id),
                    )

            next_measurement_id += 1

    copied_count = len(measurement_id_mapping)
    print(
        f"Copied {copied_count} soil measurements and their types from {source_name} database"
    )


def copy_spt_data_from_temp_db(
    temp_db_path: Path, main_conn: sqlite3.Connection, source_name: str
):
    """Copy all SPT data from a temporary database to the main database.

    Parameters
    ----------
    temp_db_path : Path
        Path to the temporary database.
    main_conn : sqlite3.Connection
        Connection to the main database.
    source_name : str
        Name of the source (e.g., "AGS" or "PDF").
    """
    if not temp_db_path.exists():
        print(f"Temporary database {temp_db_path} does not exist, skipping...")
        return

    with sqlite3.connect(str(temp_db_path)) as temp_conn:
        print(f"Copying SPT data from {source_name} database...")

        # Copy SPT reports and get borehole_id mapping
        borehole_id_mapping = copy_spt_reports(temp_conn, main_conn, source_name)

        # Copy related data using the mapping
        copy_spt_measurements(temp_conn, main_conn, borehole_id_mapping, source_name)
        copy_soil_measurements_and_types(
            temp_conn, main_conn, borehole_id_mapping, source_name
        )


def main():
    """Main function to copy SPT data from temporary databases to main database."""
    # Connect to main database
    with sqlite3.connect(str(constants.OUTPUT_DB_PATH)) as main_conn:
        # Copy from AGS temporary database
        copy_spt_data_from_temp_db(constants.TEMP_SPT_AGS_DB_PATH, main_conn, "AGS")

        # Copy from PDF temporary database
        copy_spt_data_from_temp_db(constants.TEMP_SPT_PDF_DB_PATH, main_conn, "PDF")

    print("SPT data copying completed successfully!")


if __name__ == "__main__":
    main()
