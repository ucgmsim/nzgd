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
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import chardet
import numpy as np
import pandas as pd
import tqdm
import typer

from nzgd.constants import INDEX_FILE_PATH, SPT_AGS_LOG_FILE_PATH
from nzgd.extract.bh.ags_parser import load_ags_tables
from nzgd.extract.bh.utils import SPTReport, extract_soil_report

# Initialize Typer app
app = typer.Typer()

# Configure warnings
warnings.simplefilter("error", np.exceptions.RankWarning)


@dataclass
class LocaDiagnostics:
    """Holds metadata for LOCA_ID filtering and logging.

    Attributes
    ----------
    nzgd_id : int
        Identifier of the NZGD record being processed.
    ags_file_name : str
        Source AGS file name.
    investigation_raw : str | None
        Original `InvestigationId` string retrieved from the NZGD index.
    has_multiple : bool
        Indicates whether more than one LOCA_ID was detected.
    found_match : bool | None
        True when a LOCA_ID matches the investigation id; None when not applicable.
    matched_loca_id : str | None
        The matched LOCA_ID value when available.
    all_loca_ids : str | None
        Pipe-delimited list of all observed LOCA_ID values when multiples exist.
    warned_no_InvestigationID : bool
        Flag that the missing-investigation warning has been emitted.
    warned_no_match : bool
        Flag that the missing-match warning has been emitted.
    investigation_tokens : tuple[str, ...] | None
        Cached, normalized InvestigationId tokens used during matching.
    """

    nzgd_id: int
    ags_file_name: str
    investigation_raw: str | None
    has_multiple: bool = False
    found_match: bool | None = None
    matched_loca_id: str | None = None
    all_loca_ids: str | None = None
    warned_no_InvestigationID: bool = False
    warned_no_match: bool = False
    investigation_tokens: tuple[str, ...] | None = None

    def to_log_row(self) -> dict[str, Any]:
        """Return the diagnostics state formatted for CSV logging."""
        return {
            "nzgd_id": self.nzgd_id,
            "AGS_file_name": self.ags_file_name,
            "InvestigationId": self.investigation_raw
            if self.investigation_raw
            else pd.NA,
            "has_multiple_LOCA_ID": self.has_multiple,
            "found_LOCA_ID_match": (self.found_match if self.has_multiple else pd.NA),
            "matched_LOCA_ID": (
                self.matched_loca_id
                if self.has_multiple and self.matched_loca_id
                else pd.NA
            ),
            "all_LOCA_ID": (
                self.all_loca_ids if self.has_multiple and self.all_loca_ids else pd.NA
            ),
            "warned_no_InvestigationID": self.warned_no_InvestigationID,
            "warning_no_match": self.warned_no_match,
        }


@dataclass
class BoreholeProcessingResult:
    """Aggregates the extracted report and its diagnostic log row."""

    report: SPTReport
    log_row: dict[str, Any]


LOG_COLUMNS = [
    "nzgd_id",
    "AGS_file_name",
    "InvestigationId",
    "has_multiple_LOCA_ID",
    "found_LOCA_ID_match",
    "matched_LOCA_ID",
    "all_LOCA_ID",
    "warned_no_InvestigationID",
    "warning_no_match",
]


_NON_ALNUM_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")
_RANGE_TOKEN_RE = re.compile(r"^([A-Z]+)(\d+)-(\d+)$")
_SUFFIX_RE = re.compile(r"(\d+[A-Z]*)$")


def _classify_token(value: str) -> int:
    """Return a category for token specificity."""
    has_alpha = any(ch.isalpha() for ch in value)
    has_digit = any(ch.isdigit() for ch in value)
    if has_alpha and has_digit:
        return 0
    if has_digit:
        return 1
    if has_alpha:
        return 2
    return 3


def _numeric_value(value: str) -> int | None:
    """Extract the integer represented in the token, if any."""
    digits = "".join(ch for ch in value if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            return None
    return None


def _token_variants(token: str) -> list[str]:
    """Generate ordered token variants with decreasing specificity."""
    variants: list[str] = []
    seen: set[str] = set()

    def push(value: str) -> None:
        value = value.strip().upper()
        if value and value not in seen:
            variants.append(value)
            seen.add(value)

    push(token)
    base = re.sub(r"[^A-Z0-9]", "", token.upper())
    if base:
        push(base)
        trimmed_base = base.lstrip("0")
        if trimmed_base and trimmed_base != base:
            push(trimmed_base)

        no_prefix = re.sub(r"^[A-Z]+", "", base)
        if no_prefix and no_prefix != base:
            push(no_prefix)
            trimmed = no_prefix.lstrip("0")
            if trimmed and trimmed != no_prefix:
                push(trimmed)
        else:
            trimmed = base.lstrip("0")
            if trimmed and trimmed != base:
                push(trimmed)

        suffix_match = _SUFFIX_RE.search(base)
        if suffix_match:
            suffix = suffix_match.group(1)
            push(suffix)
            trimmed = suffix.lstrip("0")
            if trimmed and trimmed != suffix:
                push(trimmed)

    return variants


def _expand_range_token(token: str) -> list[str]:
    """Expand tokens of the form PREFIX00-05 into individual identifiers."""
    match = _RANGE_TOKEN_RE.match(token.upper())
    if not match:
        return []

    prefix, start_str, end_str = match.groups()
    start = int(start_str)
    end = int(end_str)
    step = 1 if end >= start else -1
    width = max(len(start_str), len(end_str))

    expanded = [f"{prefix}{num:0{width}d}" for num in range(start, end + step, step)]
    return expanded


def _generate_investigation_tokens(raw_value: str) -> list[str]:
    """Generate ordered matching tokens from the InvestigationId string."""
    tokens: list[str] = []
    seen: set[str] = set()

    def push(value: str) -> None:
        for variant in _token_variants(value):
            if variant not in seen:
                tokens.append(variant)
                seen.add(variant)

    if not raw_value:
        return tokens

    push(raw_value)

    for match in re.finditer(r"([A-Z]+)(\d+)-(\d+)", raw_value.upper()):
        expanded = _expand_range_token(
            f"{match.group(1)}{match.group(2)}-{match.group(3)}"
        )
        for value in expanded:
            push(value)

    for bracket_token in re.findall(r"\[([^\]]+)\]", raw_value):
        push(bracket_token)

    for part in _NON_ALNUM_SPLIT_RE.split(raw_value):
        if not part:
            continue
        push(part)
        for expanded in _expand_range_token(part):
            push(expanded)

    return tokens


def _generate_loca_variants(loca_id: str) -> list[str]:
    """Generate ordered variants for a LOCA_ID suitable for matching."""
    variants: list[str] = []
    seen: set[str] = set()

    for variant in _token_variants(loca_id):
        if variant not in seen:
            variants.append(variant)
            seen.add(variant)

    for part in _NON_ALNUM_SPLIT_RE.split(loca_id):
        if not part:
            continue
        for variant in _token_variants(part):
            if variant not in seen:
                variants.append(variant)
                seen.add(variant)

    return variants


def _find_matching_locas(
    loca_ids: Iterable[str],
    investigation_tokens: Iterable[str],
) -> list[str]:
    """Return LOCA_IDs that align with the provided investigation tokens."""
    tokens = [token.strip().upper() for token in investigation_tokens if token]
    if not tokens:
        return []

    loca_list = list(loca_ids)
    if not loca_list:
        return []

    loca_variant_positions: list[tuple[str, list[str], dict[str, int]]] = []
    for loca in loca_list:
        variant_list = _generate_loca_variants(loca)
        variant_positions = {variant: idx for idx, variant in enumerate(variant_list)}
        loca_variant_positions.append((loca, variant_list, variant_positions))

    best_match: tuple[tuple[int, int, int, int, int, int], str] | None = None
    for token_idx, token in enumerate(tokens):
        token_upper = token.strip().upper()
        if not token_upper:
            continue
        token_category = _classify_token(token_upper)
        token_numeric = _numeric_value(token_upper)
        numeric_priority = -(token_numeric if token_numeric is not None else -1)
        length_priority = -len(token_upper)

        for loca_idx, (loca, variant_list, variant_positions) in enumerate(
            loca_variant_positions,
        ):
            if token_upper not in variant_positions:
                continue

            variant_index = variant_positions[token_upper]
            primary_variant = variant_list[0] if variant_list else ""
            primary_flag = 0 if token_upper == primary_variant else 1

            score = (
                token_category,
                primary_flag,
                numeric_priority,
                length_priority,
                token_idx,
                variant_index,
                loca_idx,
            )
            if best_match is None or score < best_match[0]:
                best_match = (score, loca)

    if best_match is None:
        return []

    return [best_match[1]]


@lru_cache(maxsize=1)
def _load_index_data() -> pd.DataFrame:
    """Load NZGD index metadata."""
    try:
        return pd.read_csv(INDEX_FILE_PATH, low_memory=False)
    except FileNotFoundError:
        warnings.warn(f"Index file not found at {INDEX_FILE_PATH}")
    except Exception as exc:
        warnings.warn(f"Failed to load index file {INDEX_FILE_PATH}: {exc}")
    return pd.DataFrame(columns=["nzgd_id", "InvestigationId"])


def _get_investigation_ids(nzgd_id: int) -> tuple[list[str], str | None]:
    """Return potential LOCA_ID values and the raw InvestigationId."""
    index_df = _load_index_data()
    if index_df.empty or "InvestigationId" not in index_df.columns:
        return [], None

    matches = index_df.loc[index_df["nzgd_id"] == nzgd_id, "InvestigationId"]
    if matches.empty:
        return [], None

    value = matches.iloc[0]
    if pd.isna(value) or value == "":
        return [], None

    raw_value = str(value).strip()
    tokens = _generate_investigation_tokens(raw_value)

    return tokens, raw_value


def _filter_by_investigation(
    df: pd.DataFrame,
    investigation_ids: list[str],
    diagnostics: LocaDiagnostics | None = None,
) -> pd.DataFrame:
    """Filter a table to rows matching the investigation id if needed."""
    if "LOCA_ID" not in df.columns:
        return df

    unique_loca_ids = sorted(
        {
            str(value).strip()
            for value in df["LOCA_ID"].dropna().unique()
            if str(value).strip()
        }
    )

    if len(unique_loca_ids) <= 1:
        return df

    matched_loca_id = diagnostics.matched_loca_id if diagnostics else None
    if matched_loca_id:
        mask = (
            df["LOCA_ID"].astype(str).str.strip().str.upper()
            == matched_loca_id.strip().upper()
        )
        if mask.any():
            return df[mask]

    if diagnostics and diagnostics.investigation_tokens is None and investigation_ids:
        diagnostics.investigation_tokens = tuple(investigation_ids)

    if not investigation_ids:
        if diagnostics:
            if diagnostics.warned_no_InvestigationID:
                return df
            diagnostics.warned_no_InvestigationID = True
        warnings.warn(
            "Multiple LOCA_ID values present but no InvestigationId found; "
            "retaining all rows."
        )
        return df

    matches = _find_matching_locas(unique_loca_ids, investigation_ids)
    if matches:
        matched = matches[0]
        mask = (
            df["LOCA_ID"].astype(str).str.strip().str.upper() == matched.strip().upper()
        )
        if mask.any():
            if diagnostics:
                diagnostics.matched_loca_id = matched
                diagnostics.found_match = True
            return df[mask]

    if diagnostics:
        if diagnostics.warned_no_match:
            return df
        diagnostics.warned_no_match = True

    warnings.warn(
        "InvestigationId(s) %s not found in table; retaining all rows."
        % ", ".join(investigation_ids)
    )
    return df


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


def _collect_loca_ids(tables: dict[str, pd.DataFrame]) -> set[str]:
    loca_ids: set[str] = set()
    for table in tables.values():
        if not isinstance(table, pd.DataFrame) or "LOCA_ID" not in table.columns:
            continue
        series = table["LOCA_ID"]
        if "HEADING" in table.columns and len(series) > 2:
            series = series.iloc[2:]
        for value in series.dropna():
            value_str = str(value).strip()
            if value_str:
                loca_ids.add(value_str)
    return loca_ids


def _build_loca_diagnostics(
    tables: dict[str, pd.DataFrame],
    investigation_ids: list[str],
    investigation_raw: str | None,
    nzgd_id: int,
    ags_file_name: str,
) -> LocaDiagnostics:
    diagnostics = LocaDiagnostics(
        nzgd_id=nzgd_id,
        ags_file_name=ags_file_name,
        investigation_raw=investigation_raw,
    )

    unique_ids = sorted(_collect_loca_ids(tables))
    diagnostics.has_multiple = len(unique_ids) > 1

    if diagnostics.has_multiple and unique_ids:
        diagnostics.all_loca_ids = "|".join(unique_ids)
        if investigation_ids:
            diagnostics.investigation_tokens = tuple(investigation_ids)
            matched_candidates = _find_matching_locas(unique_ids, investigation_ids)
            if matched_candidates:
                diagnostics.found_match = True
                diagnostics.matched_loca_id = matched_candidates[0]
            else:
                diagnostics.found_match = False
        else:
            diagnostics.found_match = False
    else:
        diagnostics.all_loca_ids = None
        diagnostics.found_match = None
        diagnostics.matched_loca_id = None

    return diagnostics


def process_borehole(borehole_id: int, report: Path) -> BoreholeProcessingResult:
    """Process a borehole report to extract SPT values and soil types.

    Parameters
    ----------
    borehole_id : int
        The borehole ID.
    report : Path
        The path to the borehole report AGS.

    Returns
    -------
    BoreholeProcessingResult
        Aggregated result containing the extracted SPT report and metadata
        captured for LOCA_ID logging.

    """
    tables, headings = load_ags_tables(report)

    investigation_ids, investigation_raw = _get_investigation_ids(borehole_id)
    diagnostics = _build_loca_diagnostics(
        tables,
        investigation_ids,
        investigation_raw,
        borehole_id,
        report.name,
    )

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
        ispt_df = _filter_by_investigation(ispt_df, investigation_ids, diagnostics)
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
        geol_df = _filter_by_investigation(geol_df, investigation_ids, diagnostics)
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

    return BoreholeProcessingResult(
        report=SPTReport(
            borehole_id=borehole_id,
            nzgd_id=borehole_id,
            efficiency=efficiency,
            extracted_gwl=groundwater_level,
            source_file=report,
            spt_measurements=spt_table,
            soil_measurements=geology_table,
        ),
        log_row=diagnostics.to_log_row(),
    )


RATIO_RE = re.compile(r"(\d{1,3}(\.\d+)?)\s*%")
LABEL_RE = re.compile(r"\b(ratio|efficien(t|cy)|hammer\s+energy)\b", re.IGNORECASE)


def process_borehole_no_exceptions(
    borehole_file_tuple: tuple[int, Path],
) -> BoreholeProcessingResult | None:
    """Process a borehole report while suppressing exceptions.

    Parameters
    ----------
    borehole_file_tuple : tuple[int, Path]
        A tuple containing (borehole_id, file_path).

    Returns
    -------
    Optional[BoreholeProcessingResult]
        Aggregated borehole output containing both the SPT report and its
        logging metadata, or None if an exception occurs.

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
    match = re.search(r"(\d+)", borehole_pdf.stem)
    if not match:
        raise ValueError(
            f"Could not determine NZGD ID from file name: {borehole_pdf.name}"
        )
    borehole_id = int(match.group(1))

    result = process_borehole(borehole_id, borehole_pdf)
    spt_report = result.report
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
        results = [
            result
            for result in tqdm.tqdm(
                pool.imap(process_borehole_no_exceptions, input_file_tuples),
                total=len(input_file_tuples),
            )
            if result is not None
        ]

    reports = [result.report for result in results]
    log_rows = [result.log_row for result in results]

    with sqlite3.connect(output_path) as db:
        serialize_reports(reports, db)

    log_df = pd.DataFrame(log_rows, columns=LOG_COLUMNS)
    log_df.sort_values(by=["nzgd_id", "AGS_file_name"], inplace=True, ignore_index=True)
    SPT_AGS_LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(SPT_AGS_LOG_FILE_PATH, index=False)


if __name__ == "__main__":
    app()
