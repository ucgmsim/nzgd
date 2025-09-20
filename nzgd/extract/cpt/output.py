"""Output functions for handling extracted geotechnical data."""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from nzgd_data_extraction import (
    constants,
    data_structures,
)
from nzgd_data_extraction.nzgd_sqlite.orm import (
    City,
    CPTMeasurements,
    CPTReport,
    District,
    NZGDRecord,
    Region,
    Suburb,
    initialize_db,
)


def write_extracted_data(
    extractions: list[data_structures.SheetExtractionResult],
) -> None:
    """Write the extracted data to parquet files.

    Parameters
    ----------
    extractions : list[data_structures.SheetExtractionResult]
        A list of SheetExtractionResult objects containing the extracted data and
        failed extractions.

    Returns
    -------
    None

    """
    # file_path is initialized to Path(), which has the string representation "." so
    # file_path is only valid if its string representation has length > 1
    for extraction in extractions:
        if len(str(extraction.file_path)) > 1:
            first_valid_path = extraction.file_path
            break

    # Add columns to indicate the source file and sheet for each row of data
    for extraction_result_index, extraction_result in enumerate(extractions):
        if isinstance(
            extraction_result.extraction,
            data_structures.ExtractedDataAndColInfo,
        ):
            extractions[extraction_result_index].extraction.data_df.loc[
                :,
                "file_name",
            ] = extraction_result.file_path.name

            extractions[extraction_result_index].extraction.data_df.loc[
                :,
                "sheet_name",
            ] = extraction_result.sheet_name

            # concatenate the list of removed duplicates into a single string
            if extraction_result.removed_duplicates:
                extraction_result.removed_duplicates = "___".join(
                    extraction_result.removed_duplicates,
                )

            extractions[extraction_result_index].extraction.data_df.loc[
                :,
                "removed_duplicates",
            ] = extraction_result.removed_duplicates

            extractions[extraction_result_index].extraction.data_df.loc[
                :,
                "explicit_unit_conversions",
            ] = extraction_result.explicit_unit_conversions

            extractions[extraction_result_index].extraction.data_df.loc[
                :,
                "inferred_unit_conversions",
            ] = extraction_result.inferred_unit_conversions

        else:
            extractions[extraction_result_index].extraction.loc[
                :,
                "file_name",
            ] = extraction_result.file_path.name

            extractions[extraction_result_index].extraction.loc[
                :,
                "sheet_name",
            ] = extraction_result.sheet_name

    successful_extraction_dfs = [
        extraction_result.extraction.data_df
        for extraction_result in extractions
        if isinstance(
            extraction_result.extraction,
            data_structures.ExtractedDataAndColInfo,
        )
    ]

    failed_extraction_dfs = [
        extraction_result.extraction
        for extraction_result in extractions
        if not isinstance(
            extraction_result.extraction,
            data_structures.ExtractedDataAndColInfo,
        )
    ]

    # Add an index to the DataFrames to indicate that the data comes from different
    # measurements
    for investigation_number in range(len(successful_extraction_dfs)):
        successful_extraction_dfs[investigation_number].loc[
            :,
            "investigation_number",
        ] = investigation_number

    for failed_extraction_index in range(len(failed_extraction_dfs)):
        failed_extraction_dfs[failed_extraction_index].loc[
            :,
            "failed_extraction_index",
        ] = failed_extraction_index

    extracted_data_per_record_output_path = (
        constants.OUTPUT_DIRECTORY / "extracted_data_per_record"
    )
    extraction_failures_per_record_output_path = (
        constants.OUTPUT_DIRECTORY / "extraction_failures_per_record"
    )

    extracted_data_per_record_output_path.mkdir(exist_ok=True, parents=True)
    extraction_failures_per_record_output_path.mkdir(exist_ok=True, parents=True)

    # If there are any dataframes of extracted data or failed extractions,
    # save them to parquet files

    if len(successful_extraction_dfs) > 0:
        all_extracted_data = pd.concat(successful_extraction_dfs)
        all_extracted_data.to_parquet(
            extracted_data_per_record_output_path
            / f"{first_valid_path.parent.name.split('_')[1]}.parquet",
        )

    if len(failed_extraction_dfs) > 0:
        all_failed_extractions = pd.concat(failed_extraction_dfs)
        all_failed_extractions.to_parquet(
            extraction_failures_per_record_output_path
            / f"{first_valid_path.parent.name.split('_')[1]}.parquet",
        )


def output_cpt_to_db(
    extraction_results: list[data_structures.SheetExtractionResult],
) -> None:
    """Output the extracted CPT/SCPT data to the database.

    Parameters
    ----------
    extraction_results : list[data_structures.SheetExtractionResult]
        A list of SheetExtractionResult objects containing the extracted data and
        failed extraction_results.

    Returns
    -------
    None

    """
    # Check if database file exists, if not initialize it
    db_path = Path(constants.OUTPUT_DB_PATH)
    if not db_path.exists():
        initialize_db()
        # Also initialize location tables when creating new database
        populate_location_tables()
    else:
        # Check if location tables are populated, if not populate them
        try:
            if Region.select().count() == 0:
                populate_location_tables()
        except Exception:
            # If tables don't exist or there's an error, populate them
            populate_location_tables()

    # Filter out extraction_results without successful results
    successful_extractions = [
        extraction_result
        for extraction_result in extraction_results
        if isinstance(
            extraction_result.extraction,
            data_structures.ExtractedDataAndColInfo,
        )
    ]

    if not successful_extractions:
        return
    try:
        max_cpt_id_query = CPTReport.select().order_by(CPTReport.cpt_id.desc()).first()
        next_cpt_id = max_cpt_id_query.cpt_id + 1 if max_cpt_id_query else 1
    except Exception:  # noqa: BLE001
        next_cpt_id = 1

    # Process each extraction_result
    for extraction_result in successful_extractions:
        # Type assertions since we filtered for non-None values
        if not isinstance(
            extraction_result.extraction,
            data_structures.ExtractedDataAndColInfo,
        ):
            error_message = "Expected ExtractedDataAndColInfo type"
            raise TypeError(error_message)
        # Extract type prefix and nzgd_id from parent directory name
        parent_dir_name = extraction_result.file_path.parent.name
        # Extract prefix and numeric part from directory name
        # (e.g., "CPT_1" -> ("CPT", 1) or "SCPT_98767" -> ("SCPT", 98767))
        match = re.search(r"(CPT|SCPT)_(\d+)", parent_dir_name)
        if not match:
            continue  # Skip if no valid format found

        type_prefix = match.group(1)
        nzgd_id = int(match.group(2))

        # Get location IDs for this record
        record_id = f"{type_prefix}_{nzgd_id}"
        location_ids = get_location_ids_for_record(record_id)

        # Get metadata from CSV
        metadata = _get_metadata_from_csv(nzgd_id, type_prefix)

        # Create or update NZGDRecord if it doesn't exist
        nzgd_record, created = NZGDRecord.get_or_create(
            nzgd_id=nzgd_id,
            defaults={
                "type_prefix": type_prefix,
                "latitude": metadata["latitude"],
                "longitude": metadata["longitude"],
                "original_investigation_name": metadata["original_investigation_name"],
                "investigation_date": metadata["investigation_date"],
                "published_date": metadata["published_date"],
                "region_id": location_ids["region_id"],
                "district_id": location_ids["district_id"],
                "city_id": location_ids["city_id"],
                "suburb_id": location_ids["suburb_id"],
            },
        )

        # Extract depth information from the DataFrame
        extracted_data_df = extraction_result.extraction.data_df

        # Calculate deepest and shallowest depths (depth column is guaranteed to be "Depth")
        depth_values = extracted_data_df["Depth"].dropna()
        if len(depth_values) > 0:
            deepest_depth = float(depth_values.max())
            shallowest_depth = float(depth_values.min())
        else:
            deepest_depth = 0.0
            shallowest_depth = 0.0

        # Create CPTReport entry
        cpt_report = CPTReport.create(
            cpt_id=next_cpt_id,
            nzgd_id=nzgd_record,
            cpt_file=extraction_result.file_path.name,
            deepest_depth=deepest_depth,
            shallowest_depth=shallowest_depth,
        )

        # Insert CPT measurement data into CPTMeasurements table
        _insert_cpt_measurements(extracted_data_df, next_cpt_id)

        next_cpt_id += 1


def _insert_cpt_measurements(extracted_data_df: pd.DataFrame, cpt_id: int) -> None:
    """Insert CPT measurement data into the CPTMeasurements table.

    Parameters
    ----------
    extracted_data_df : pd.DataFrame
        DataFrame containing the extracted CPT measurement data with columns:
        "Depth", "qc", "fs", "u".
    cpt_id : int
        The CPT ID to associate with these measurements.

    Returns
    -------
    None

    """
    # Prepare measurement data for bulk insert
    measurements_data = []
    for _, row in extracted_data_df.iterrows():
        # Extract values, handling potential NaN values
        depth_val = row["Depth"] if pd.notna(row["Depth"]) else None
        qc_val = row["qc"] if pd.notna(row["qc"]) else None
        fs_val = row["fs"] if pd.notna(row["fs"]) else None
        u2_val = row["u"] if pd.notna(row["u"]) else None

        # Only add measurement if at least depth is available
        if depth_val is not None:
            measurements_data.append(
                {
                    "cpt_id": cpt_id,
                    "depth": float(depth_val),
                    "qc": float(qc_val) if qc_val is not None else None,
                    "fs": float(fs_val) if fs_val is not None else None,
                    "u2": float(u2_val) if u2_val is not None else None,
                },
            )

    # Bulk insert measurements if we have any data
    if measurements_data:
        CPTMeasurements.insert_many(measurements_data).execute()


def _get_metadata_from_csv(nzgd_record_id: int, type_prefix: str) -> dict:
    """Extract metadata from CSV for a given NZGD record ID.

    Parameters
    ----------
    nzgd_record_id : int
        The ID of the NZGD record to get metadata for.
    type_prefix : str
        The type prefix (CPT or SCPT) for the record.

    Returns
    -------
    dict
        Dictionary containing metadata with keys:
        'latitude', 'longitude', 'original_investigation_name',
        'investigation_date', 'published_date'

    Raises
    ------
    FileNotFoundError
        If the metadata CSV file is not found
    RuntimeError
        If the CSV file cannot be read
    ValueError
        If no metadata is found for the given NZGD record ID

    """
    # Load the CSV file with metadata
    csv_path = (
        Path(__file__).parent
        / "resources"
        / "NZGD_Investigation_Report_08112024_1017.csv"
    )

    if not csv_path.exists():
        msg = f"Metadata CSV file not found at {csv_path}"
        raise FileNotFoundError(msg)

    # Read the CSV file
    try:
        metadata_df = pd.read_csv(
            csv_path,
            usecols=[
                "ID",
                "Type",
                "OriginalReference",
                "InvestigationDate",
                "TotalDepth",
                "PublishedDate",
                "Latitude",
                "Longitude",
            ],
        )
    except Exception as e:
        msg = f"Error reading metadata CSV: {e}"
        raise RuntimeError(msg) from e

    # Find the row for this NZGD ID
    target_id = f"{type_prefix}_{nzgd_record_id}"
    matching_rows = metadata_df[metadata_df["ID"] == target_id]

    if matching_rows.empty:
        msg = (
            f"No metadata found for NZGD ID {nzgd_record_id} with prefix {type_prefix}"
        )
        raise ValueError(msg)

    # Get the first matching row
    metadata_row = matching_rows.iloc[0]

    # Parse dates (handle different date formats)
    investigation_date = None
    published_date = None

    try:
        if pd.notna(metadata_row["InvestigationDate"]):
            # Parse format like "12/May/2011"
            investigation_date = datetime.strptime(
                metadata_row["InvestigationDate"],
                "%d/%b/%Y",
            ).date()
    except (ValueError, TypeError):
        pass

    try:
        if pd.notna(metadata_row["PublishedDate"]):
            # Parse format like "13/Jun/2012"
            published_date = datetime.strptime(
                metadata_row["PublishedDate"],
                "%d/%b/%Y",
            ).date()
    except (ValueError, TypeError):
        pass

    # Extract latitude and longitude
    latitude = (
        float(metadata_row["Latitude"]) if pd.notna(metadata_row["Latitude"]) else 0.0
    )
    longitude = (
        float(metadata_row["Longitude"]) if pd.notna(metadata_row["Longitude"]) else 0.0
    )

    # Extract original reference
    original_investigation_name = (
        metadata_row["OriginalReference"]
        if pd.notna(metadata_row["OriginalReference"])
        else None
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "original_investigation_name": original_investigation_name,
        "investigation_date": investigation_date,
        "published_date": published_date,
    }


def populate_location_tables() -> None:
    """Populate the Region, City, District, and Suburb tables from CSV data.

    This function reads the regions CSV file and extracts unique values
    for each location type, then populates the corresponding database tables.

    Column mappings:
    - Region table ← "district" column in CSV
    - District table ← "territor_1" column in CSV
    - City table ← "major_na_2" column in CSV
    - Suburb table ← "name" column in CSV

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the regions CSV file is not found
    RuntimeError
        If the CSV file cannot be read

    """
    # Load the CSV file with location data
    csv_path = (
        Path(__file__).parent
        / "resources"
        / "regions_NZGD_Investigation_Report_08112024_1017.csv"
    )

    if not csv_path.exists():
        msg = f"Regions CSV file not found at {csv_path}"
        raise FileNotFoundError(msg)

    # Read the CSV file
    try:
        regions_df = pd.read_csv(csv_path)
    except Exception as e:
        msg = f"Error reading regions CSV: {e}"
        raise RuntimeError(msg) from e

    # Populate each table
    _populate_regions_table(regions_df)
    _populate_districts_table(regions_df)
    _populate_cities_table(regions_df)
    _populate_suburbs_table(regions_df)


def _populate_regions_table(regions_df: pd.DataFrame) -> None:
    """Populate the Region table from the CSV data.

    Parameters
    ----------
    regions_df : pd.DataFrame
        DataFrame containing the regions data

    Returns
    -------
    None

    """
    # Get unique regions from the "district" column
    unique_regions = regions_df["district"].dropna().unique()

    # Insert regions with sequential IDs starting from 1
    for idx, region_name in enumerate(sorted(unique_regions), start=1):
        Region.get_or_create(
            region_id=idx,
            defaults={"name": region_name},
        )


def _populate_districts_table(regions_df: pd.DataFrame) -> None:
    """Populate the District table from the CSV data.

    Parameters
    ----------
    regions_df : pd.DataFrame
        DataFrame containing the regions data

    Returns
    -------
    None

    """
    # Get unique districts from the "territor_1" column
    unique_districts = regions_df["territor_1"].dropna().unique()

    # Insert districts with sequential IDs starting from 1
    for idx, district_name in enumerate(sorted(unique_districts), start=1):
        District.get_or_create(
            district_id=idx,
            defaults={"name": district_name},
        )


def _populate_cities_table(regions_df: pd.DataFrame) -> None:
    """Populate the City table from the CSV data.

    Parameters
    ----------
    regions_df : pd.DataFrame
        DataFrame containing the regions data

    Returns
    -------
    None

    """
    # Get unique cities from the "major_na_2" column
    unique_cities = regions_df["major_na_2"].dropna().unique()

    # Insert cities with sequential IDs starting from 1
    for idx, city_name in enumerate(sorted(unique_cities), start=1):
        City.get_or_create(
            city_id=idx,
            defaults={"name": city_name},
        )


def _populate_suburbs_table(regions_df: pd.DataFrame) -> None:
    """Populate the Suburb table from the CSV data.

    Parameters
    ----------
    regions_df : pd.DataFrame
        DataFrame containing the regions data

    Returns
    -------
    None

    """
    # Get unique suburbs from the "name" column
    unique_suburbs = regions_df["name"].dropna().unique()

    # Insert suburbs with sequential IDs starting from 1
    for idx, suburb_name in enumerate(sorted(unique_suburbs), start=1):
        Suburb.get_or_create(
            suburb_id=idx,
            defaults={"name": suburb_name},
        )


def get_location_ids_for_record(record_id: str) -> dict[str, int]:
    """Get the location IDs for a given record ID from the CSV data.

    This function looks up the location information for a specific record
    and returns the corresponding database IDs for region, district, city, and suburb.

    Parameters
    ----------
    record_id : str
        The record ID (e.g., "CPT_7") to look up

    Returns
    -------
    dict[str, int]
        Dictionary containing the location IDs:
        {"region_id": int, "district_id": int, "city_id": int, "suburb_id": int}

    Raises
    ------
    FileNotFoundError
        If the regions CSV file is not found
    ValueError
        If the record ID is not found in the CSV or location data is missing
    RuntimeError
        If location tables are not properly populated

    """
    # Ensure location tables are populated
    _ensure_location_tables_populated()

    # Load the CSV file with location data
    csv_path = (
        Path(__file__).parent
        / "resources"
        / "regions_NZGD_Investigation_Report_08112024_1017.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"Regions CSV file not found at {csv_path}")

    regions_df = pd.read_csv(csv_path)

    # Find the row for this record ID
    matching_rows = regions_df[regions_df["record_id"] == record_id]

    if matching_rows.empty:
        raise ValueError(f"Record ID '{record_id}' not found in regions CSV")

    # Get the first matching row
    row = matching_rows.iloc[0]

    # Extract location names - all must be present
    region_name = row["district"] if pd.notna(row["district"]) else None
    district_name = row["territor_1"] if pd.notna(row["territor_1"]) else None
    city_name = row["major_na_2"] if pd.notna(row["major_na_2"]) else None
    suburb_name = row["name"] if pd.notna(row["name"]) else None

    if not region_name:
        raise ValueError(f"Region (district) not found for record {record_id}")
    if not district_name:
        raise ValueError(f"District (territor_1) not found for record {record_id}")
    if not city_name:
        raise ValueError(f"City (major_na_2) not found for record {record_id}")
    if not suburb_name:
        raise ValueError(f"Suburb (name) not found for record {record_id}")

    # Look up region ID
    region = Region.get(Region.name == region_name)

    # Look up district ID
    district = District.get(District.name == district_name)

    # Look up city ID
    city = City.get(City.name == city_name)

    # Look up suburb ID
    suburb = Suburb.get(Suburb.name == suburb_name)

    return {
        "region_id": region.region_id,
        "district_id": district.district_id,
        "city_id": city.city_id,
        "suburb_id": suburb.suburb_id,
    }


def _ensure_location_tables_populated() -> None:
    """Ensure that location tables are populated.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If location tables cannot be populated or are empty

    """
    # Check if tables are populated by counting regions
    region_count = Region.select().count()
    if region_count == 0:
        # Try to populate tables
        populate_location_tables()
        # Verify they were populated successfully
        if Region.select().count() == 0:
            raise RuntimeError(
                "Failed to populate location tables - Region table is still empty",
            )
