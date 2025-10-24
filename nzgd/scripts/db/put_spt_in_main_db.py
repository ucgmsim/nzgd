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


def copy_spt_reports(temp_conn: sqlite3.Connection, main_conn: sqlite3.Connection, 
                     source_name: str) -> dict[int, int]:
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
    temp_cursor.execute("SELECT * FROM sptreport")
    temp_reports = temp_cursor.fetchall()
    
    # Get column names for the temp database
    temp_cursor.execute("PRAGMA table_info(sptreport)")
    temp_columns = [col[1] for col in temp_cursor.fetchall()]
    
    # Get existing borehole_ids in main database
    main_cursor.execute("SELECT borehole_id FROM sptreport")
    existing_borehole_ids = {row[0] for row in main_cursor.fetchall()}
    
    # Create mapping from original borehole_id to new borehole_id
    borehole_id_mapping = {}
    next_available_id = get_next_available_borehole_id(main_conn)
    
    for temp_report in temp_reports:
        temp_report_dict = dict(zip(temp_columns, temp_report))
        original_borehole_id = temp_report_dict['borehole_id']
        
        # Always use next available ID to ensure uniqueness
        # This allows the same nzgd_id to appear multiple times with different borehole_ids
        new_borehole_id = next_available_id
        next_available_id += 1
        
        if original_borehole_id in existing_borehole_ids:
            print(f"Borehole ID {original_borehole_id} already exists, using {new_borehole_id}")
        else:
            print(f"Using borehole ID {new_borehole_id} for original {original_borehole_id}")
        
        borehole_id_mapping[original_borehole_id] = new_borehole_id
        
        # Insert into main database
        main_cursor.execute("""
            INSERT INTO sptreport 
            (borehole_id, nzgd_id, efficiency, extracted_gwl, gwl_residual, source_file)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            new_borehole_id,
            temp_report_dict['nzgd_id'],  # nzgd_id should be same as original borehole_id
            temp_report_dict.get('efficiency'),
            temp_report_dict.get('extracted_gwl'),
            temp_report_dict.get('gwl_residual'),
            temp_report_dict.get('source_file', ''),
        ))
    
    print(f"Copied {len(temp_reports)} SPT reports from {source_name} database")
    return borehole_id_mapping


def copy_spt_measurements(temp_conn: sqlite3.Connection, main_conn: sqlite3.Connection,
                          borehole_id_mapping: dict[int, int], source_name: str):
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
    temp_cursor.execute("SELECT * FROM sptmeasurements")
    temp_measurements = temp_cursor.fetchall()
    
    # Get column names for the temp database
    temp_cursor.execute("PRAGMA table_info(sptmeasurements)")
    temp_columns = [col[1] for col in temp_cursor.fetchall()]
    
    for temp_measurement in temp_measurements:
        temp_measurement_dict = dict(zip(temp_columns, temp_measurement))
        original_borehole_id = temp_measurement_dict['borehole_id']
        
        if original_borehole_id in borehole_id_mapping:
            new_borehole_id = borehole_id_mapping[original_borehole_id]
            
            # Insert into main database with new measurement_id
            main_cursor.execute("""
                INSERT INTO sptmeasurements (measurement_id, borehole_id, depth, n)
                VALUES (?, ?, ?, ?)
            """, (
                next_measurement_id,
                new_borehole_id,
                temp_measurement_dict.get('depth'),
                temp_measurement_dict.get('n')
            ))
            next_measurement_id += 1
    
    print(f"Copied {len(temp_measurements)} SPT measurements from {source_name} database")


def copy_soil_measurements(temp_conn: sqlite3.Connection, main_conn: sqlite3.Connection,
                          borehole_id_mapping: dict[int, int], source_name: str):
    """Copy soil measurements from temporary database to main database.
    
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
    temp_cursor.execute("SELECT * FROM soilmeasurements")
    temp_soil_measurements = temp_cursor.fetchall()
    
    # Get column names for the temp database
    temp_cursor.execute("PRAGMA table_info(soilmeasurements)")
    temp_columns = [col[1] for col in temp_cursor.fetchall()]
    
    for temp_soil_measurement in temp_soil_measurements:
        temp_soil_dict = dict(zip(temp_columns, temp_soil_measurement))
        original_borehole_id = temp_soil_dict['report_id']  # report_id is the borehole_id
        
        if original_borehole_id in borehole_id_mapping:
            new_borehole_id = borehole_id_mapping[original_borehole_id]
            
            # Insert into main database with new measurement_id
            main_cursor.execute("""
                INSERT INTO soilmeasurements (measurement_id, report_id, top_depth)
                VALUES (?, ?, ?)
            """, (
                next_measurement_id,
                new_borehole_id,
                temp_soil_dict.get('top_depth')
            ))
            next_measurement_id += 1
    
    print(f"Copied {len(temp_soil_measurements)} soil measurements from {source_name} database")


def copy_soil_measurement_soil_types(temp_conn: sqlite3.Connection, main_conn: sqlite3.Connection,
                                    borehole_id_mapping: dict[int, int], source_name: str):
    """Copy soil measurement soil type relationships from temporary database to main database.
    
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
    
    # Get all soil measurement soil types from temporary database
    temp_cursor.execute("SELECT * FROM soilmeasurementsoiltype")
    temp_soil_types = temp_cursor.fetchall()
    
    # Get column names for the temp database
    temp_cursor.execute("PRAGMA table_info(soilmeasurementsoiltype)")
    temp_columns = [col[1] for col in temp_cursor.fetchall()]
    
    # Get soil measurement IDs in main database
    main_cursor.execute("SELECT measurement_id, report_id FROM soilmeasurements")
    main_soil_measurements = {row[1]: row[0] for row in main_cursor.fetchall()}
    
    for temp_soil_type in temp_soil_types:
        temp_soil_type_dict = dict(zip(temp_columns, temp_soil_type))
        original_measurement_id = temp_soil_type_dict['soil_measurement_id']
        
        # Find the corresponding soil measurement in main database
        # We need to find the soil measurement by report_id (borehole_id)
        temp_cursor.execute("""
            SELECT report_id FROM soilmeasurements 
            WHERE measurement_id = ?
        """, (original_measurement_id,))
        result = temp_cursor.fetchone()
        
        if result:
            original_borehole_id = result[0]
            if original_borehole_id in borehole_id_mapping:
                new_borehole_id = borehole_id_mapping[original_borehole_id]
                
                # Find the new measurement_id in main database
                if new_borehole_id in main_soil_measurements:
                    new_measurement_id = main_soil_measurements[new_borehole_id]
                    
                    # Insert into main database
                    main_cursor.execute("""
                        INSERT OR IGNORE INTO soilmeasurementsoiltype 
                        (soil_measurement_id, soil_type_id)
                        VALUES (?, ?)
                    """, (
                        new_measurement_id,
                        temp_soil_type_dict.get('soil_type_id')
                    ))
    
    print(f"Copied soil measurement soil type relationships from {source_name} database")


def copy_spt_data_from_temp_db(temp_db_path: Path, main_conn: sqlite3.Connection, source_name: str):
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
        copy_soil_measurements(temp_conn, main_conn, borehole_id_mapping, source_name)
        copy_soil_measurement_soil_types(temp_conn, main_conn, borehole_id_mapping, source_name)


def main():
    """Main function to copy SPT data from temporary databases to main database."""
    # Connect to main database
    with sqlite3.connect(str(constants.OUTPUT_DB_PATH)) as main_conn:
        # Copy from AGS temporary database
        copy_spt_data_from_temp_db(
            constants.TEMP_SPT_AGS_DB_PATH, 
            main_conn, 
            "AGS"
        )
        
        # Copy from PDF temporary database
        copy_spt_data_from_temp_db(
            constants.TEMP_SPT_PDF_DB_PATH, 
            main_conn, 
            "PDF"
        )
    
    print("SPT data copying completed successfully!")


if __name__ == "__main__":
    main()
