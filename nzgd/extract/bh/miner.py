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
import os
import re
import sqlite3
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import tqdm
import typer
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTFigure, LTTextBoxHorizontal, LTTextBoxVertical
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFParser

# Initialize Typer app
app = typer.Typer()

# Configure warnings
warnings.simplefilter("error", np.exceptions.RankWarning)


@dataclass
class TextObject:
    """Represents a text object with positional and textual data.

    Attributes
    ----------
    y0 : float
        The lower y-coordinate of the text object.
    y1 : float
        The upper y-coordinate of the text object.
    x0 : float
        The left x-coordinate of the text object.
    x1 : float
        The right x-coordinate of the text object.
    text : str
        The textual content of the object.

    """

    y0: float
    y1: float
    x0: float
    x1: float
    text: str

    @property
    def yc(self) -> float:
        """float: The vertical centre coordinate."""
        return (self.y1 + self.y0) / 2

    @property
    def xc(self) -> float:
        """float: The horizontal centre coordinate."""
        return (self.x1 + self.x0) / 2


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


def extract_spt_value(text: str) -> int | None:
    """Extract the SPT (Standard Penetration Test) value from text.

    Parameters
    ----------
    text : str
        The input text containing the SPT value.

    Returns
    -------
    Optional[int]
        The extracted SPT value, or None if not found.

    """
    # Searches for "N" as a whole word (e.g., not part of another word
    # like "Nail"), followed by "=" or ">", and a number, allowing
    # spaces between.
    # Valid match: "N = 42" or "N>50"
    # Invalid match: "Northing=42" or "NN = 50"
    if match := re.search(r"\bNc?\s*(=|\>)\s*(\d+)", text):
        return int(match.group(2))
    return None


def extract_pdf_text_objects(lt_objs: list[Any]) -> Generator[TextObject, None, None]:
    """Recursively extract text objects from PDF layout elements.

    Parameters
    ----------
    lt_objs : list[Any]
        A list of PDF layout objects.

    Yields
    ------
    TextObject
        Extracted text objects with positional and text data.

    """
    for obj in lt_objs:
        if isinstance(obj, (LTTextBoxHorizontal, LTTextBoxVertical)):
            yield TextObject(
                y0=obj.y0,
                y1=obj.y1,
                x0=obj.x0,
                x1=obj.x1,
                text=obj.get_text(),
            )
        elif isinstance(obj, LTFigure):
            yield from extract_pdf_text_objects(obj._objs)


def is_valid_depth_measurement(text: str, max_depth_cutoff: float = 60) -> bool:
    """Check if a given string represents a valid depth measurement.

    A valid depth measurement is a number that is between 0 and 60m.
    The 60m cutoff heuristic is a generous assumption on the deepest
    reasonable depths observed in borehole PDFs.

    Parameters
    ----------
    text : str
        The string to check.
    max_depth_cutoff : float, optional
        Use an alternative maximum depth cutoff value. Default is 60m.

    Returns
    -------
    bool
        True if the string is a valid number, otherwise False.

    """
    try:
        x = float(text.strip(" \nm"))

        return 0 <= x <= max_depth_cutoff
    except ValueError:
        return False


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
    # Borehole PDF names have format Borehole_<Borehole ID>_(Raw/Rep)01.pdf
    if match := re.search(r"_(\d+)_", report.stem):
        return int(match.group(1))
    raise ValueError(f"Report name {report.stem} lacks proper structure")


def process_borehole(report: Path) -> SPTReport:
    """Process a borehole report to extract SPT values and soil types.

    Parameters
    ----------
    report : Path
        The path to the borehole report PDF.

    Returns
    -------
    SPTReport
        The extracted SPT report.

    Raises
    ------
    PDFTextExtractionNotAllowed
        If text extraction is not permitted for the PDF.
    ValueError
        If depth column or SPT values are missing.

    """
    text_objects = []
    with report.open("rb") as file:
        parser = PDFParser(file)
        document = PDFDocument(parser)

        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed(
                f"Text extraction not allowed in {report}",
            )

        rsrcmgr = PDFResourceManager()
        laparams = LAParams(detect_vertical=True)
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        for page in PDFPage.create_pages(document):
            interpreter.process_page(page)
            layout = device.get_result()
            page_text_objects = sorted(
                extract_pdf_text_objects(layout._objs),
                key=lambda obj: (obj.y0, obj.x0),
            )
            text_objects.append(page_text_objects)

    return _analyze_text_objects(text_objects, report)


def invalid_scale(scale: list[float]) -> bool:
    """Detect if a depth scale is invalid.

    A depth scale is a list of depth values extracted from the PDF.
    The depth scale is considered invalid if it is not strictly
    increasing and non-negative, i.e.

        scale[i] > scale[i - 1] and
        scale[0] >= 0.

    Parameters
    ----------
    scale : list[float]
        The list of depth values to check.


    Returns
    -------
    bool
        True if the depth scale is strictly increasing.

    """
    return all(scale[i] > scale[i - 1] for i in range(1, len(scale))) and scale[0] >= 0


def extract_depth_scale(
    depth_node: TextObject,
    nodes: list[TextObject],
    eps_max: float = 5,
    max_iterations: int = 10,
) -> npt.NDArray[np.float64]:
    """Extracts the depth scaling parameters for a given depth column by iteratively
    refining the search space to identify valid depth values.

    Parameters
    ----------
    depth_node : TextObject
        The text object representing the depth column, with `x0`, `x1`, and `yc`
        attributes specifying its bounding box and centre.
    nodes : list[TextObject]
        A list of text objects containing potential depth values, typically all
        the text nodes on a page.
    eps_max : float, optional
        The maximum allowed horizontal deviation from the centre of the `depth_node`
        during the search for valid depth values. Default is 5.
    max_iterations : int, optional
        The maximum number of iterations to perform when refining the search space.
        Default is 10.

    Returns
    -------
    npt.NDArray[np.float64]
        A 1D array of two values `[m, c]` representing the depth scale equation:
        `depth = m * y + c`, where `y` is the PDF y-coordinate of the text object.

    Raises
    ------
    ValueError
        If the function fails to converge to a valid set of depth values within the
        specified number of iterations.

    Notes
    -----
    This function identifies text objects aligned horizontally with the depth column,
    refines the search bounds using binary search to filter valid depth values, and
    computes a linear fit.

    The data_extraction for extraction is illustrated below:


          +-------------------------+         ^
          |    Depth   Col 1  Col 2 |         | y-axis
          |                         |         |
          |       100         -1    |         |
          |                         |         |
          |             150         |         |
          |                         |         |
          |       200               |         |
          +-------------------------+         v
    <-------- eps_max ---------->  (Initial bounds include everything)
      <------ eps_1 ------>         (Bounds narrowed, some invalid values excluded)
             <eps_2>               (Bounds too narrow, missing valid values)
         <--- eps_3 -->      (Final valid bounds after adjustment)

    The diagram illustrates:
    - `eps_max` starts as the widest range.
    - Iterative refinement adjusts the bounds (`eps_1`, `eps_2`, `eps_3`) using binary search.
    - Extra numbers (e.g., `-1` and `150` in `Col 2`) are excluded from the depth column.

    If the search bounds accidentally shrink too far (as in `eps_1` ->
    `eps_2`), the algorithm expands the bounds again to include valid
    depth values. A successful result is achieved when the data_extraction
    converges to the smallest possible epsilon that collects a valid
    depth scale.

    """
    eps_max += (depth_node.x1 - depth_node.x0) / 2
    eps_low = 0.0
    eps_high = eps_max
    eps = eps_max / 2
    params: npt.NDArray[np.float64] | None = None
    for _ in range(max_iterations):
        depth_values = [
            obj
            for obj in nodes
            if (not (depth_node.x1 < obj.xc - eps or obj.xc + eps < depth_node.x0))
            and is_valid_depth_measurement(obj.text)
        ]
        if len(depth_values) < 2:
            # Not enough depth values found, widen the search space for depth values.
            eps_low = eps
            eps = (eps_high + eps_low) / 2
            continue
        scale = [float(obj.text.strip(" \nm")) for obj in depth_values]
        if invalid_scale(scale):
            # The depth scale doesn't make sense: this likely means we have
            # included values in our depth that are not part of the depth
            # column. Narrow the search to filter them out.

            eps_high = eps
            eps = (eps_high + eps_low) / 2
            continue

        params = np.polyfit(
            [node.yc for node in depth_values],
            scale,
            1,
        )
    if isinstance(params, np.ndarray):
        return params
    raise ValueError("Failed to converge to a valid depth column")


RATIO_RE = re.compile(r"(\d{1,3}(\.\d+)?)\s*%")
LABEL_RE = re.compile(r"\b(ratio|efficien(t|cy)|hammer\s+energy)\b", re.IGNORECASE)


def get_ratio_near(node: TextObject, page: list[TextObject]) -> float | None:
    """Extract the energy ratio from the given node or nearby.

    Parameters
    ----------
    node : TextObject
        The reference node.
    page : list[TextObject]
        The page of text objects to search.


    Returns
    -------
    Optional[float]
        Either the efficiency ratio, or None if it could be found.

    """
    if efficiencies := list(re.finditer(RATIO_RE, node.text)):
        label = re.search(LABEL_RE, node.text)
        label_start = label.start(0)
        label_end = label.end(0)
        return float(
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

    efficiency_nodes = [
        other for other in page if not (node.y1 < other.y0 or other.y1 < node.y0)
    ]
    for efficiency_node in efficiency_nodes:
        if efficiency := re.search(RATIO_RE, efficiency_node.text):
            return float(efficiency.group(1))

    return None


def _analyze_text_objects(
    text_objects: list[list[TextObject]],
    report: Path,
) -> SPTReport:
    """Analyse extracted text objects to extract borehole data.

    Parameters
    ----------
    text_objects : list[list[TextObject]]
        A list of pages containing lists of TextObjects.
    report : Path
        The path to the borehole report.

    Returns
    -------
    pd.DataFrame
        A DataFrame with borehole data.

    Raises
    ------
    ValueError
        If depth column or SPT values are not found.

    """
    spt_values, extracted_soil_depths, soil_types = [], [], []

    try:
        depth_node = next(
            node
            for page in text_objects
            for node in page
            # Matches "depth" or "length" (case-insensitive),
            # optionally followed by "(m)" with or without spaces in
            # between.
            if re.match(r"(depth|length)\s*(\(m\))?", node.text.lower())
        )
    except StopIteration as exc:
        raise ValueError(f"Depth column not found in {report}") from exc

    hammer_efficiency = None
    for page in text_objects:
        for node in page:
            if re.search(LABEL_RE, node.text.lower()):
                hammer_efficiency = get_ratio_near(node, page)

    for page in text_objects:
        try:
            m, c = extract_depth_scale(depth_node, page)
        except ValueError:
            continue
        for node in sorted(page, key=lambda n: n.yc, reverse=True):
            depth = m * node.yc + c
            if soil_report := extract_soil_report(node.text):
                extracted_soil_depths.append(depth)
                soil_types.append(soil_report)

        for node in page:
            depth = m * node.yc + c
            depth = round(depth / 0.25) * 0.25

            n = extract_spt_value(node.text)
            if n is not None:
                spt_values.append(
                    {
                        "Depth": round(depth, 2),
                        "N": n,
                    },
                )
    if not spt_values:
        raise ValueError(f"No SPT values found in {report}")

    df = pd.DataFrame(spt_values)
    min_depth = df["Depth"].min()
    max_depth = df["Depth"].max()
    if min_depth < 0 or max_depth > 70:
        raise ValueError(
            f"Invalid depth calculation detected (minimum depth = {min_depth}, max depth = {max_depth}).",
        )
    soil_measurements = pd.DataFrame(
        {"top_depth": extracted_soil_depths, "soil_types": soil_types},
    )
    print()
    return SPTReport(
        borehole_id=borehole_id(report),
        nzgd_id=borehole_id(report),
        efficiency=hammer_efficiency,
        borehole_diameter=None,
        extracted_gwl=None,
        source_file=report,
        spt_measurements=df,
        soil_measurements=soil_measurements,
    )


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
        warnings.warn(f"Failed to data_extraction {report}: {e}")
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
                    """ INSERT INTO soilmeasurementsoiltype
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
    pdfs = list(report_directory.glob("*.pdf"))
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
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

    print()

    with sqlite3.connect(output_path) as db:
        serialize_reports(reports, db)


if __name__ == "__main__":
    app()
