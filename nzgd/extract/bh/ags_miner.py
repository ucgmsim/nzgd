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
from pathlib import Path
from typing import Annotated

import chardet
import numpy as np
import pandas as pd
import tqdm
import typer
from python_ags4 import AGS4

from nzgd.constants import SOIL_TYPE_TO_ID
from nzgd.extract.bh.data_structures import SPTReport

# Initialize Typer app
app = typer.Typer()

# Configure warnings
warnings.simplefilter("error", np.exceptions.RankWarning)


def extract_soil_report(description: str) -> set[str]:
    """Extract soil types mentioned in a description.

    Parameters
    ----------
    description : str
        The input text to search for soil types.

    Returns
    -------
    set[str]
        A set of identified soil types from the input.

    """
    soil_types = set(SOIL_TYPE_TO_ID.keys())

    # Initially try to find soil types that are already written in upper case
    found_soil_types = soil_types & {word.strip(",.;") for word in description.split()}

    # If no soil types were found, try to find soil types that are written in lower case
    if len(found_soil_types) == 0:
        found_soil_types = soil_types & {
            word.strip(",.;").upper() for word in description.split()
        }

    return found_soil_types


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
    tables, headings = AGS4.AGS4_to_dataframe(report)

    # Initialize SPT table with empty DataFrame
    spt_table = pd.DataFrame(columns=["Depth", "N"])

    # Try to extract SPT data if ISPT table exists
    if "ISPT" in tables:
        spt_table = (
            tables["ISPT"][["ISPT_TOP", "ISPT_MAIN"]]
            .iloc[2:]
            .rename(columns={"ISPT_TOP": "Depth", "ISPT_MAIN": "N"})
        )
        # If all N values are empty, create empty DataFrame with NaN values
        if spt_table["N"].eq("").all():
            spt_table = pd.DataFrame(columns=["Depth", "N"])
            warnings.warn(
                f"No SPT N values found in {report}, creating empty SPT measurements"
            )

    # Initialize geology table
    geology_table = pd.DataFrame(columns=["top_depth", "soil_types"])

    # Try to extract geology data if GEOL table exists
    if "GEOL" in tables and tables.get("GEOL") is not None:
        geology_table = (
            tables["GEOL"][["GEOL_TOP", "GEOL_DESC"]]
            .iloc[2:]
            .rename(columns={"GEOL_TOP": "top_depth", "GEOL_DESC": "soil_types"})
        )
        geology_table["soil_types"] = geology_table["soil_types"].apply(
            extract_soil_report,
        )

    # Try to extract efficiency from report text
    efficiency = None
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
    has_spt_data = not spt_table.empty and not spt_table["N"].eq("").all()
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
        extracted_gwl=None,
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
                n_value = row["N"] if pd.notna(row["N"]) and row["N"] != "" else None
                cursor.execute(
                    """
                    INSERT INTO sptmeasurements (spt_id, depth_m, n)
                    VALUES (?, ?, ?)
                """,
                    (report.borehole_id, depth, n_value),
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
