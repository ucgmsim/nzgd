"""Fill missing CPT metadata values from Maxim's extractions.

This script reads Maxim's extracted CPT metadata from CSV files and updates
missing values in the cptreport table for:
- predrill_depth_m
- extracted_gwl_m
- tip_net_area_ratio
"""

import sqlite3
from pathlib import Path

import pandas as pd

from nzgd import constants

maxim_cpt_output_dir = Path(
    "/home/arr65/data/nzgd/maxim_liquepy_extracted_cpt_20250930/liquepy_extracted_cpt"
)

# Apply updates to the de-duplicated DB produced by deduplicate.py, which names its
# output by appending the deduped suffix to the source DB stem.
DEDUPED_DB_PATH = constants.OUTPUT_DB_PATH.with_name(
    constants.OUTPUT_DB_PATH.stem
    + constants.DEDUP_CONFIG["output"]["deduped_db_suffix"]
    + ".db"
)


def extract_metadata_from_maxim_csv(csv_path: Path, delimiter: str = ","):
    """
    Extract pre-drill depth, groundwater level, and cone area ratio from Maxim's CSV file.

    Based on liquepy format where metadata is in the first 24 lines:
    - "Pre-Drill:" contains predrill depth
    - "Assumed GWL:" contains groundwater level
    - "aratio" contains cone area ratio
    """
    predrill_depth = None
    gwl = None
    aratio = None

    try:
        with open(csv_path, "r") as f:
            lines = f.readlines()

        # Read first 24 lines (metadata section)
        for line in lines[:24]:
            parts = line.strip().split(delimiter)
            if len(parts) >= 2:
                key = parts[0].strip()
                value = parts[1].strip()

                # Check for Pre-Drill (case-insensitive, handles variations like "Pre-Drill:" or "Pre-Drill")
                if key.lower().startswith("pre-drill"):
                    try:
                        predrill_depth = float(value) if value and value != "" else None
                    except (ValueError, TypeError):
                        predrill_depth = None

                # Check for Assumed GWL (case-insensitive)
                if "assumed gwl" in key.lower():
                    try:
                        if value == "-" or value == "":
                            gwl = None
                        else:
                            gwl = float(value)
                    except (ValueError, TypeError):
                        gwl = None

                # Check for aratio (case-insensitive)
                if "aratio" in key.lower():
                    try:
                        aratio = float(value) if value and value != "" else None
                    except (ValueError, TypeError):
                        aratio = None
    except (IOError, OSError, UnicodeDecodeError) as e:
        print(f"Error reading {csv_path}: {e}")

    return predrill_depth, gwl, aratio


def main():
    """Main function to update missing CPT metadata from Maxim's extractions."""
    print("=" * 80)
    print("FILLING MISSING CPT METADATA FROM MAXIM'S EXTRACTIONS")
    print("=" * 80)
    print()

    # Extract metadata from Maxim's CSV files
    print("Extracting metadata from Maxim's CSV files...")
    maxim_csv_files = {}
    for csv_file in maxim_cpt_output_dir.rglob("CPT_*.csv"):
        # Extract nzgd_id from filename like "CPT_123.csv"
        filename = csv_file.name
        if filename.startswith("CPT_") and filename.endswith(".csv"):
            try:
                nzgd_id_str = filename[4:-4]  # Remove "CPT_" prefix and ".csv" suffix
                nzgd_id = int(nzgd_id_str)
                maxim_csv_files[nzgd_id] = csv_file
            except ValueError:
                continue

    print(f"Found {len(maxim_csv_files)} CSV files")
    print()

    # Extract metadata from CSV files
    maxim_metadata = {}
    for nzgd_id, csv_file in maxim_csv_files.items():
        predrill, gwl, aratio = extract_metadata_from_maxim_csv(csv_file)
        maxim_metadata[nzgd_id] = {
            "predrill_depth_m": predrill,
            "extracted_gwl_m": gwl,
            "tip_net_area_ratio": aratio,
        }

    print(f"Extracted metadata for {len(maxim_metadata)} CPTs")
    print()

    # Connect to database and get current CPT reports
    print(f"Reading current CPT reports from de-duplicated database {DEDUPED_DB_PATH}...")
    with sqlite3.connect(DEDUPED_DB_PATH) as db:
        cpt_reports_df = pd.read_sql_query("SELECT * FROM cptreport", db)

        # Find records that need updating
        updates = []
        stats = {
            "predrill_depth_m": {"found": 0, "updated": 0},
            "extracted_gwl_m": {"found": 0, "updated": 0},
            "tip_net_area_ratio": {"found": 0, "updated": 0},
        }

        for idx, row in cpt_reports_df.iterrows():
            nzgd_id = row["nzgd_id"]
            cpt_id = row["cpt_id"]

            if nzgd_id not in maxim_metadata:
                continue

            maxim_vals = maxim_metadata[nzgd_id]
            update_fields = {}

            # Check each field
            for field_name in [
                "predrill_depth_m",
                "extracted_gwl_m",
                "tip_net_area_ratio",
            ]:
                maxim_val = maxim_vals[field_name]
                current_val = row[field_name]

                # Only update if current value is NULL and Maxim has a value
                if pd.isna(current_val) and maxim_val is not None:
                    stats[field_name]["found"] += 1
                    update_fields[field_name] = maxim_val

            if update_fields:
                updates.append({"cpt_id": cpt_id, "fields": update_fields})

        # Print statistics
        print("Update statistics:")
        for field_name, field_stats in stats.items():
            print(
                f"  {field_name}: {field_stats['found']} records found with missing values"
            )
        print(f"  Total records to update: {len(updates)}")
        print()

        if not updates:
            print(
                "No updates needed. All values are already present or Maxim has no values."
            )
            return

        # Perform updates
        print("Updating database...")
        cursor = db.cursor()
        update_count = 0

        for update in updates:
            cpt_id = update["cpt_id"]
            fields = update["fields"]

            # Build UPDATE statement
            set_clauses = []
            values = []
            for field_name, field_value in fields.items():
                set_clauses.append(f"{field_name} = ?")
                values.append(field_value)
                stats[field_name]["updated"] += 1

            values.append(cpt_id)  # For WHERE clause

            update_sql = f"""
                UPDATE cptreport
                SET {", ".join(set_clauses)}
                WHERE cpt_id = ?
            """

            cursor.execute(update_sql, values)
            update_count += 1

        db.commit()

        print(f"Successfully updated {update_count} CPT records")
        print()

        # Print final statistics
        print("Final update statistics:")
        for field_name, field_stats in stats.items():
            print(f"  {field_name}: {field_stats['updated']} records updated")
        print()

        print("Update complete!")


if __name__ == "__main__":
    main()
