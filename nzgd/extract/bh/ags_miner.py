"""Borehole Report Processor
--------------------------

This script is a command-line interface tool for processing borehole PDF reports
to extract Standard Penetration Test (SPT) values and associated soil classifications.
It consolidates the extracted data into a structured format, which is saved as a
Parquet file for further analysis.

Features
--------
- Extracts depth, SPT values, and soil classifications from borehole PDF reports.
- Supports bulk processing of multiple reports in a directory.
- Outputs consolidated data in a Parquet format for efficient storage and retrieval.

Usage
-----
Run the script from the command line with the required arguments. Example usage:

    python miner.py /path/to/reports /path/to/output.parquet

Positional Arguments
---------------------
report_directory : Path
    Path to the directory containing borehole PDF reports.
output_path : Path
    Path to save the consolidated output as a Parquet file.

Dependencies
------------
- Python >= 3.8
- pdfminer.six
- pandas
- numpy
- typer
- tqdm

Notes
-----
- Ensure that the input PDF reports are formatted in a way that the script can parse.
- The script attempts to extract data robustly but may fail for non-standard or
  corrupted reports.
- Warnings are emitted for reports that cannot be processed, but execution will
  continue for other reports.

"""

import json
import multiprocessing
import re
import sqlite3
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import chardet
import numpy as np
import pandas as pd
import tqdm
import typer

from nzgd.constants import INDEX_FILE_PATH
from nzgd.extract.bh.ags_parser import load_ags_tables
from nzgd.extract.bh.utils import SPTReport, extract_soil_report

# Initialize Typer app
app = typer.Typer()

# Configure warnings
warnings.simplefilter("error", np.exceptions.RankWarning)


@lru_cache(maxsize=1)
def _load_index_data() -> pd.DataFrame:
    """Load NZGD index metadata."""
    try:
        return pd.read_csv(INDEX_FILE_PATH)
    except FileNotFoundError:
        warnings.warn(f"Index file not found at {INDEX_FILE_PATH}")
    except Exception as exc:
        warnings.warn(f"Failed to load index file {INDEX_FILE_PATH}: {exc}")
    return pd.DataFrame(columns=["nzgd_id", "InvestigationId"])


def _get_investigation_ids(nzgd_id: int) -> list[str]:
    """Return potential LOCA_ID values for the provided NZGD id."""
    index_df = _load_index_data()
    if index_df.empty or "InvestigationId" not in index_df.columns:
        return []

    matches = index_df.loc[index_df["nzgd_id"] == nzgd_id, "InvestigationId"]
    if matches.empty:
        return []

    value = matches.iloc[0]
    if pd.isna(value) or value == "":
        return []

    raw_value = str(value).strip()
    candidates: list[str] = []

    bracket_ids = re.findall(r"\[([^\]]+)\]", raw_value)
    candidates.extend(part.strip() for part in bracket_ids if part.strip())

    if raw_value and raw_value not in candidates:
        candidates.append(raw_value)

    return candidates


def _filter_by_investigation(
    df: pd.DataFrame,
    investigation_ids: list[str],
) -> pd.DataFrame:
    """Filter a table to rows matching the investigation id if needed."""
    if "LOCA_ID" not in df.columns:
        return df

    unique_loca_ids = {
        str(value) for value in df["LOCA_ID"].dropna().unique() if value != ""
    }

    if len(unique_loca_ids) <= 1:
        return df

    if not investigation_ids:
        warnings.warn(
            "Multiple LOCA_ID values present but no InvestigationId found; "
            "retaining all rows."
        )
        return df

    filtered = df[df["LOCA_ID"].astype(str).isin(investigation_ids)]
    if filtered.empty:
        warnings.warn(
            "InvestigationId(s) %s not found in table; retaining all rows."
            % ", ".join(investigation_ids)
        )
        return df

    return filtered


def _first_numeric_value(df: pd.DataFrame, column: str) -> float | None:
    """Return the first value in `column` convertible to float."""
    if df.empty or column not in df.columns:
        return None

    series = df[column].dropna()
    for value in series:
        if value == "" or pd.isna(value):
            continue

        candidate = value.strip() if isinstance(value, str) else value
        if candidate == "":
            continue

        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue

    return None


def process_borehole(borehole_id: int, report: Path) -> SPTReport:
    """Process a borehole report to extract SPT values and soil types.

    Parameters
    ----------
    borehole_id : int
        The borehole ID.
    report : Path
        The path to the borehole report AGS.

    Returns
    -------
    SPTReport
        The extracted SPT report.

    """
    tables, headings = load_ags_tables(report)

    investigation_ids = _get_investigation_ids(borehole_id)

    ispt_columns = [
        "LOCA_ID",
        "ISPT_TOP",
        "ISPT_MAIN",
        "ISPT_NVAL",
        "ISPT_WAT",
        "ISPT_ERAT",
    ]
    geology_columns = ["LOCA_ID", "GEOL_TOP", "GEOL_DESC"]
    desired_ispt_columns = [
        "LOCA_ID",
        "Depth",
        "ISPT_MAIN",
        "ISPT_NVAL",
        "ISPT_WAT",
        "ISPT_ERAT",
    ]

    # Initialize SPT table with empty DataFrame
    spt_table = pd.DataFrame(columns=desired_ispt_columns)
    # Try to extract SPT data if ISPT table exists
    if "ISPT" in tables:
        ispt_df = tables["ISPT"].iloc[2:].copy()
        ispt_df = _filter_by_investigation(ispt_df, investigation_ids)
        ispt_df = ispt_df.reindex(columns=ispt_columns, fill_value=pd.NA)

        spt_table = ispt_df.rename(columns={"ISPT_TOP": "Depth"}).reindex(
            columns=desired_ispt_columns, fill_value=pd.NA
        )
        # If all N values are empty, create empty DataFrame with NaN values
        if spt_table["ISPT_MAIN"].eq("").all() and spt_table["ISPT_NVAL"].eq("").all():
            spt_table = pd.DataFrame(columns=desired_ispt_columns)
            warnings.warn(
                f"No SPT ISPT_MAIN values found in {report}, creating empty SPT measurements"
            )

    # Initialize geology table
    geology_table = pd.DataFrame(columns=geology_columns)

    # Try to extract geology data if GEOL table exists
    if "GEOL" in tables and tables.get("GEOL") is not None:
        geol_df = tables["GEOL"][geology_columns].iloc[2:].copy()
        geol_df = _filter_by_investigation(geol_df, investigation_ids)
        geology_table = geol_df.rename(
            columns={"GEOL_TOP": "top_depth", "GEOL_DESC": "soil_types"}
        )
        geology_table["soil_types"] = geology_table["soil_types"].apply(
            extract_soil_report,
        )

    efficiency = _first_numeric_value(spt_table, "ISPT_ERAT")
    groundwater_level = _first_numeric_value(spt_table, "ISPT_WAT")

    # Try to extract efficiency from report text only if not already found
    if efficiency is None:
        try:
            report_data = report.read_bytes()
            encoding = chardet.detect(report_data)
            report_text = report_data.decode(encoding["encoding"])

            if efficiencies := list(re.finditer(RATIO_RE, report_text)):
                label = re.search(LABEL_RE, report_text)
                if label:
                    label_start = label.start(0)
                    label_end = label.end(0)
                    efficiency = float(
                        min(
                            efficiencies,
                            # Hausdorff distance between label spans to find the
                            # one that is most likely to be the hammer energy
                            # efficiency ratio.
                            key=lambda m: max(
                                abs(m.start(0) - label_start),
                                abs(m.end(0) - label_end),
                            ),
                        ).group(1),
                    )
        except Exception as e:
            warnings.warn(f"Could not extract efficiency from {report}: {e}")

    # Check if any meaningful data was extracted
    has_spt_data = not spt_table.empty and not spt_table["ISPT_MAIN"].eq("").all()
    has_soil_data = not geology_table.empty
    has_efficiency = efficiency is not None

    if not (has_spt_data or has_soil_data or has_efficiency):
        raise ValueError(
            f"No meaningful data extracted from {report}: no SPT measurements, soil measurements, or efficiency found"
        )

    return SPTReport(
        borehole_id=borehole_id,
        nzgd_id=borehole_id,
        efficiency=efficiency,
        extracted_gwl=groundwater_level,
        source_file=report,
        spt_measurements=spt_table,
        soil_measurements=geology_table,
    )


RATIO_RE = re.compile(r"(\d{1,3}(\.\d+)?)\s*%")
LABEL_RE = re.compile(r"\b(ratio|efficien(t|cy)|hammer\s+energy)\b", re.IGNORECASE)


def process_borehole_no_exceptions(
    borehole_file_tuple: tuple[int, Path],
) -> SPTReport | None:
    """Process a borehole report while suppressing exceptions.

    Parameters
    ----------
    borehole_file_tuple : tuple[int, Path]
        A tuple containing (borehole_id, file_path).

    Returns
    -------
    Optional[SPTReport]
        A SPTReport with borehole data, or None if an exception occurs.

    """
    borehole_id, report = borehole_file_tuple
    try:
        return process_borehole(borehole_id, report)
    except Exception as e:
        warnings.warn(f"Failed to process {report}: {e}")
        return None


def serialize_reports(reports: list[SPTReport], conn: sqlite3.Connection):
    """Persist extracted borehole data into the SQLite schema.

    Parameters
    ----------
    reports : list of SPTReport
        Collection of parsed borehole reports containing SPT measurements,
        soil classifications, and optional efficiency/groundwater metadata.
    conn : sqlite3.Connection
        Open database connection pointing at the NZGD extraction database.

    Notes
    -----
    The function performs the following steps:

    1. Upsert each `SPTReport` into the `sptreport` table.
    2. Collate all unique soil types, ensuring that supporting lookups exist.
    3. Insert SPT measurements while normalising missing numeric values.
    4. Insert geology intervals and link them to their soil classifications.
    """
    cursor = conn.cursor()

    # Insert SPTReports
    report_data = [
        (
            report.borehole_id,
            report.borehole_id,
            report.efficiency,
            report.extracted_gwl,
            report.source_file.name,
        )
        for report in reports
    ]
    cursor.executemany(
        """
        INSERT OR REPLACE INTO sptreport (spt_id, nzgd_id, efficiency, extracted_gwl_m, source_file)
        VALUES (?, ?, ?, ?, ?)
    """,
        report_data,
    )

    # Insert SoilTypes and retrieve their IDs
    soil_type_data = set()
    for report in reports:
        for _, row in report.soil_measurements.iterrows():
            for soil_type in row["soil_types"]:
                soil_type_data.add((soil_type,))

    cursor.execute("SELECT id, value FROM SoilTypes")
    soil_type_id_map = {
        value: soil_type_id for soil_type_id, value in cursor.fetchall()
    }

    # Insert SPTMeasurements and SPTMeasurementSoilTypes
    for report in reports:
        # Only insert SPT measurements if the DataFrame is not empty
        if not report.spt_measurements.empty:
            for _, row in report.spt_measurements.iterrows():
                # Handle NaN/None values by converting to None for SQLite
                depth = row["Depth"] if pd.notna(row["Depth"]) else None
                ispt_main_n = (
                    row["ISPT_MAIN"]
                    if pd.notna(row["ISPT_MAIN"]) and row["ISPT_MAIN"] != ""
                    else None
                )
                ispt_nval = (
                    row["ISPT_NVAL"]
                    if "ISPT_NVAL" in row
                    and pd.notna(row["ISPT_NVAL"])
                    and row["ISPT_NVAL"] != ""
                    else None
                )
                cursor.execute(
                    """
                    INSERT INTO sptmeasurements (spt_id, depth_m, ISPT_MAIN, ISPT_NVAL)
                    VALUES (?, ?, ?, ?)
                """,
                    (report.borehole_id, depth, ispt_main_n, ispt_nval),
                )
        # Only process soil measurements if the DataFrame is not empty
        if not report.soil_measurements.empty:
            for _, row in report.soil_measurements.iterrows():
                if not row["soil_types"]:
                    continue
                # Handle NaN/None values for top_depth
                top_depth = row["top_depth"] if pd.notna(row["top_depth"]) else None
                cursor.execute(
                    """
                                   INSERT INTO soilmeasurements (spt_id, top_depth_m)
                                   VALUES (?, ?)
                               """,
                    (report.borehole_id, top_depth),
                )
                measurement_id = cursor.lastrowid
                for soil_type in row["soil_types"]:
                    if soil_type in soil_type_id_map:
                        cursor.execute(
                            """ INSERT OR IGNORE INTO soilmeasurementsoiltype
                                       VALUES (?, ?)
                            """,
                            (measurement_id, soil_type_id_map[soil_type]),
                        )


@app.command(
    help="Mine an individual borehole PDF and output a JSON file.",
    name="single",
)
def mine_individual_borehole(
    borehole_pdf: Annotated[
        Path,
        typer.Argument(
            help="Path to borehole PDF file to read.",
            exists=True,
            readable=True,
            dir_okay=False,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(
            help="Path to save the output (as a JSON file).",
            writable=True,
            dir_okay=False,
        ),
    ],
):
    """Extract SPT readings from a single borehole log file.

    Parameters
    ----------
    borehole_pdf : Path
        Path to the borehole log PDF file.
    output_path : Path
        Path to the output file (a JSON file).

    """
    spt_report = process_borehole(borehole_pdf)
    with open(output_path, "w") as output:
        json.dump(
            {
                "Borehole Id": spt_report.borehole_id,
                "Borehole File": str(spt_report.source_file),
                "Efficiency": spt_report.efficiency,
                "Measurements": spt_report.spt_measurements.sort_values(
                    by="Depth",
                ).to_dict("records"),
            },
            output,
            indent=4,
        )


def mine_borehole_log(
    input_file_tuples: list[tuple[int, Path]],
    output_path: Path,
) -> None:
    """Extract and consolidate borehole log data from a directory of reports.

    Parameters
    ----------
    input_file_tuples : list[tuple[int, Path]]
        List of tuples of borehole ID and path to the borehole log AGS files.
    output_path : Path
        Path to the output SQLite database.

    """

    with multiprocessing.Pool() as pool:
        reports = [
            report
            for report in tqdm.tqdm(
                pool.imap(process_borehole_no_exceptions, input_file_tuples),
                total=len(input_file_tuples),
            )
            if report is not None
        ]

    with sqlite3.connect(output_path) as db:
        serialize_reports(reports, db)


if __name__ == "__main__":
    app()
