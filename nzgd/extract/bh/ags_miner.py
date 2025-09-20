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
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import chardet
import numpy as np
import pandas as pd
import tqdm
import typer
from python_ags4 import AGS4

# Initialize Typer app
app = typer.Typer()

# Configure warnings
warnings.simplefilter("error", np.exceptions.RankWarning)


@dataclass
class SPTReport:
    borehole_id: int
    """The borehole ID number for this report."""
    nzgd_id: int
    """The NZGD D number for this report."""
    efficiency: float | None
    """The hammer efficiency ratio."""

    borehole_diameter: float | None
    """The diameter of the borehole."""

    extracted_gwl: float | None
    """The extracted ground water level for the SPT (borehole) report."""

    source_file: Path
    """The path to the report."""

    spt_measurements: pd.DataFrame
    """The SPT record. A data frame with columns Depth, and N."""

    soil_measurements: pd.DataFrame
    """The SPT soil measurements. A dataframe with columns 'top_depth', and 'soil_types'"""


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
    soil_types = {"SAND", "SILT", "CLAY", "GRAVEL", "COBBLES", "BOULDERS"}
    return soil_types & {word.strip(",.;") for word in description.split()}


def borehole_id(report: Path) -> int:
    """Extract the borehole ID from a report filename.

    Parameters
    ----------
    report : Path
        The path to the borehole report.

    Returns
    -------
    int
        The extracted borehole ID.

    Raises
    ------
    ValueError
        If the report name does not follow the expected format.

    """
    # Borehole PDF names have format Borehole_<Borehole ID>_(Raw/Rep)01.ags
    if match := re.search(r"_(\d+)_", report.stem):
        return int(match.group(1))
    raise ValueError(f"Report name {report.stem} lacks proper structure")


def process_borehole(report: Path) -> SPTReport:
    """Process a borehole report to extract SPT values and soil types.

    Parameters
    ----------
    report : Path
        The path to the borehole report AGS.

    Returns
    -------
    SPTReport
        The extracted SPT report.

    Raises
    ------
    ValueError
        If depth column or SPT values are missing.

    """
    tables, headings = AGS4.AGS4_to_dataframe(report)
    spt_table = (
        tables["ISPT"][["ISPT_TOP", "ISPT_MAIN"]]
        .iloc[2:]
        .rename(columns={"ISPT_TOP": "Depth", "ISPT_MAIN": "N"})
    )
    if spt_table["N"].eq("").all():
        raise ValueError("SPT N values are empty strings")

    geology_table = tables.get("GEOL")
    if geology_table is not None:
        geology_table = (
            geology_table[["GEOL_TOP", "GEOL_DESC"]]
            .iloc[2:]
            .rename(columns={"GEOL_TOP": "top_depth", "GEOL_DESC": "soil_types"})
        )
        geology_table["soil_types"] = geology_table["soil_types"].apply(
            extract_soil_report,
        )
        report_data = report.read_bytes()
        encoding = chardet.detect(report_data)
        report_text = report_data.decode(encoding["encoding"])

        efficiency = None
        if efficiencies := list(re.finditer(RATIO_RE, report_text)):
            label = re.search(LABEL_RE, report_text)
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
    else:
        geology_table = pd.DataFrame(columns=["top_depth", "soil_types"])

    return SPTReport(
        borehole_id=borehole_id(report),
        nzgd_id=borehole_id(report),
        efficiency=efficiency,
        borehole_diameter=None,
        extracted_gwl=None,
        source_file=report,
        spt_measurements=spt_table,
        soil_measurements=geology_table,
    )


RATIO_RE = re.compile(r"(\d{1,3}(\.\d+)?)\s*%")
LABEL_RE = re.compile(r"\b(ratio|efficien(t|cy)|hammer\s+energy)\b", re.IGNORECASE)


def process_borehole_no_exceptions(report: Path) -> SPTReport | None:
    """Process a borehole report while suppressing exceptions.

    Parameters
    ----------
    report : Path
        The path to the borehole report.

    Returns
    -------
    Optional[pd.DataFrame]
        A DataFrame with borehole data, or None if an exception occurs.

    """
    try:
        return process_borehole(report)
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
            report.borehole_diameter,
            report.extracted_gwl,
            report.source_file.name,
        )
        for report in reports
    ]
    cursor.executemany(
        """
        INSERT OR REPLACE INTO sptreport (borehole_id, nzgd_id, efficiency, borehole_diameter, extracted_gwl, source_file)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        report_data,
    )

    # Insert SoilTypes and retrieve their IDs
    soil_type_data = set()
    for report in reports:
        for _, row in report.soil_measurements.iterrows():
            for soil_type in row["soil_types"]:
                soil_type_data.add((soil_type,))

    cursor.execute("SELECT id, name FROM SoilTypes")
    soil_type_id_map = {name: soil_type_id for soil_type_id, name in cursor.fetchall()}

    # Insert SPTMeasurements and SPTMeasurementSoilTypes
    for report in reports:
        for _, row in report.spt_measurements.iterrows():
            cursor.execute(
                """
                INSERT INTO sptmeasurements (borehole_id, depth, n)
                VALUES (?, ?, ?)
            """,
                (report.borehole_id, row["Depth"], row["N"]),
            )
        for _, row in report.soil_measurements.iterrows():
            if not row["soil_types"]:
                continue
            cursor.execute(
                """
                               INSERT INTO soilmeasurements (report_id, top_depth)
                               VALUES (?, ?)
                           """,
                (report.borehole_id, row["top_depth"]),
            )
            measurement_id = cursor.lastrowid
            for soil_type in row["soil_types"]:
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


@app.command(help="Extract borehole SPT data from a directory.", name="directory")
def mine_borehole_log(
    report_directory: Annotated[
        Path,
        typer.Argument(
            help="Path to the directory containing borehole PDF reports.",
            exists=True,
            readable=True,
            file_okay=False,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(
            help="Path to save the consolidated output as a database.",
            writable=True,
            exists=False,
            dir_okay=False,
        ),
    ],
) -> None:
    """Extract and consolidate borehole log data from a directory of reports.

    Parameters
    ----------
    report_directory : Path
        Path to the directory containing borehole PDF reports.
    output_path : Path
        Path to save the consolidated output as a Parquet file.

    """
    pdfs = list(report_directory.rglob("*.ags"))
    with multiprocessing.Pool() as pool:
        reports_including_incorrect = [
            report
            for report in tqdm.tqdm(
                pool.imap(process_borehole_no_exceptions, pdfs),
                total=len(pdfs),
            )
            if report is not None
        ]

    # Jake applied a filtering as some incorrect extractions are not in his database.
    # Jake's filtering criteria are not known, the extraction code and input data have
    # not changed, so we only keep extractions that are also in his database.

    jake_conn = sqlite3.connect("/home/arr65/Downloads/jake_geodata.db")
    jake_sptreport_df = pd.read_sql_query("SELECT * FROM sptreport", jake_conn)
    jake_conn.close()
    jake_extracted_borehole_ids = jake_sptreport_df["borehole_id"].tolist()

    reports = [
        report
        for report in reports_including_incorrect
        if report.borehole_id in jake_extracted_borehole_ids
    ]

    with sqlite3.connect(output_path) as db:
        serialize_reports(reports, db)


if __name__ == "__main__":
    app()
