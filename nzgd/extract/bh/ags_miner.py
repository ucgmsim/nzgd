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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import chardet
import numpy as np
import pandas as pd
import tqdm
import typer

from nzgd import constants
from nzgd.extract.bh.ags_parser import load_ags_tables
from nzgd.extract.bh.utils import SPTReport, extract_soil_report

# Initialize Typer app
app = typer.Typer()

# Configure warnings
warnings.simplefilter("error", np.exceptions.RankWarning)


class ExtractionStatusLogger:
    """Tracks extraction status and writes to a file."""

    def __init__(self, output_file: Path, nzgd_id: int, file_name: str):
        """Initialize the extraction status logger.

        Parameters
        ----------
        output_file : Path
            Path to the CSV file where extraction status will be written.
        nzgd_id : int
            The NZGD ID for this file.
        file_name : str
            The name of the file being processed.
        """
        self.output_file = output_file
        self.nzgd_id = nzgd_id
        self.file_name = file_name
        self.original_showwarning = warnings.showwarning
        # Initialize extraction status - default to 1 (success) for all
        self.extraction_status = {
            "nzgd_id": nzgd_id,
            "file_name": file_name,
            "extracted_spt_measurements": 1,
            "extracted_soil_measurements": 1,
            "extracted_density_measurements": 1,
            "extracted_efficiency": 1,
            "extracted_diameter": 1,
        }

    def __enter__(self):
        """Set up the warning handler."""
        warnings.showwarning = self._showwarning
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Restore the original warning handler and write extraction status to file."""
        warnings.showwarning = self.original_showwarning
        self._write_extraction_status_to_file()

    def _showwarning(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        """Custom warning handler that captures warnings and updates extraction status."""
        # Call the original showwarning to print to terminal
        self.original_showwarning(message, category, filename, lineno, file, line)

        # Update extraction status based on warning messages
        warning_text = str(message) if isinstance(message, Warning) else message

        # Check for specific warning patterns that indicate extraction failures
        if "No SPT" in warning_text or "no SPT measurements" in warning_text:
            self.extraction_status["extracted_spt_measurements"] = 0
        if (
            "no soil measurements" in warning_text
            or "No meaningful data" in warning_text
        ):
            self.extraction_status["extracted_soil_measurements"] = 0
        if "no density" in warning_text.lower():
            self.extraction_status["extracted_density_measurements"] = 0
        if (
            "no efficiency" in warning_text
            or "efficiency" in warning_text.lower()
            and "not found" in warning_text.lower()
        ):
            self.extraction_status["extracted_efficiency"] = 0
        if "no diameter" in warning_text.lower():
            self.extraction_status["extracted_diameter"] = 0

    def update_extraction_status(
        self,
        extracted_spt: bool | None = None,
        extracted_soil: bool | None = None,
        extracted_density: bool | None = None,
        extracted_efficiency: bool | None = None,
        extracted_diameter: bool | None = None,
    ):
        """Update extraction status based on actual extraction results.

        Parameters
        ----------
        extracted_spt : bool | None
            Whether SPT measurements were extracted.
        extracted_soil : bool | None
            Whether soil measurements were extracted.
        extracted_density : bool | None
            Whether density measurements were extracted.
        extracted_efficiency : bool | None
            Whether efficiency was extracted.
        extracted_diameter : bool | None
            Whether diameter was extracted.
        """
        if extracted_spt is not None:
            self.extraction_status["extracted_spt_measurements"] = (
                1 if extracted_spt else 0
            )
        if extracted_soil is not None:
            self.extraction_status["extracted_soil_measurements"] = (
                1 if extracted_soil else 0
            )
        if extracted_density is not None:
            self.extraction_status["extracted_density_measurements"] = (
                1 if extracted_density else 0
            )
        if extracted_efficiency is not None:
            self.extraction_status["extracted_efficiency"] = (
                1 if extracted_efficiency else 0
            )
        if extracted_diameter is not None:
            self.extraction_status["extracted_diameter"] = (
                1 if extracted_diameter else 0
            )

    def _write_extraction_status_to_file(self):
        """Write extraction status to a CSV file."""
        # Ensure the output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to CSV
        status_df = pd.DataFrame([self.extraction_status])
        status_df.to_csv(self.output_file, index=False)


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


def _try_single_prefix_match(
    loca_ids_with_counts: list[tuple[str, int]],
    investigation_id: str,
) -> str | None:
    """Try to match using single prefix strategy.

    When a prefix appears in InvestigationId and only one LOCA_ID has that prefix,
    use that LOCA_ID.

    This fallback method was developed to address problematic NZGD IDs where
    simple substring matching failed:
    - NZGD ID 31205 (InvestigationId: "91 River Road BH, HAs, DCPs") → BH01
      In this case, if only one of the LOCA_IDs has "BH" prefix, that one is selected.
    - NZGD ID 97660 (InvestigationId: "BH-t2") → BH2
      In this case, if only one of the LOCA_IDs has "BH" prefix, that one is selected.

    Parameters
    ----------
    loca_ids_with_counts : list[tuple[str, int]]
        List of (LOCA_ID, table_count) tuples.
    investigation_id : str
        The InvestigationId string.

    Returns
    -------
    str | None
        Matched LOCA_ID if found, None otherwise.
    """
    investigation_upper = investigation_id.upper()
    loca_ids = [loca_id for loca_id, _count in loca_ids_with_counts]

    # Count occurrences of each prefix in LOCA_IDs
    prefix_counts: dict[str, list[str]] = {}
    for loca_id in loca_ids:
        prefix_match = re.match(r"^([A-Za-z]+)", loca_id)
        if prefix_match:
            prefix = prefix_match.group(1).upper()
            if prefix in investigation_upper:
                if prefix not in prefix_counts:
                    prefix_counts[prefix] = []
                prefix_counts[prefix].append(loca_id)

    # Find prefixes that appear only once
    for prefix, matching_loca_ids in prefix_counts.items():
        if len(matching_loca_ids) == 1:
            return matching_loca_ids[0]

    return None


def _try_bh_t_pattern_match(
    loca_ids_with_counts: list[tuple[str, int]],
    investigation_id: str,
) -> str | None:
    """Try to match using BH-t pattern strategy.

    When InvestigationId has pattern "BH-tX" or "BH_tX", extract number X and
    find LOCA_ID with same number (preferring BH prefix).

    This fallback method was developed to address problematic NZGD IDs where
    simple substring matching failed:
    - NZGD ID 99137 (InvestigationId: "BH-t5") → BH5
      (matches "BH-t5" in InvestigationId to "BH5" in LOCA_IDs)

    Parameters
    ----------
    loca_ids_with_counts : list[tuple[str, int]]
        List of (LOCA_ID, table_count) tuples, sorted by table_count descending.
    investigation_id : str
        The InvestigationId string.

    Returns
    -------
    str | None
        Matched LOCA_ID if found, None otherwise.
    """
    bh_t_match = re.search(r"BH[-_]t(\d+)", investigation_id, re.IGNORECASE)
    if not bh_t_match:
        return None

    number = bh_t_match.group(1)
    loca_ids = [loca_id for loca_id, _count in loca_ids_with_counts]

    # Find LOCA_IDs with this number, preferring BH prefix
    candidates_with_bh = []
    candidates_without_bh = []

    for loca_id in loca_ids:
        loca_number_match = re.search(r"(\d+)", loca_id)
        if loca_number_match and loca_number_match.group(1) == number:
            if "BH" in loca_id.upper():
                candidates_with_bh.append(loca_id)
            else:
                candidates_without_bh.append(loca_id)

    # Prefer BH-prefixed matches, then others
    if candidates_with_bh:
        return candidates_with_bh[0]
    if candidates_without_bh:
        return candidates_without_bh[0]

    return None


def _try_bracket_pattern_match(
    loca_ids_with_counts: list[tuple[str, int]],
    investigation_id: str,
) -> str | None:
    """Try to match using bracket pattern strategy.

    When InvestigationId has pattern "[PREFIX_BHX]" or "[PREFIX-BHX]", extract
    prefix and number, find LOCA_ID with same prefix and number.

    This fallback method is a general-purpose strategy for matching bracket patterns
    in InvestigationId. It extracts content from brackets and attempts to match
    prefixes and numbers to LOCA_IDs. Note that for the 7 problematic NZGD IDs
    (124644, 124645, 124646, 124647), the context-based match runs first and handles
    these cases before bracket pattern matching is attempted. This method serves as
    a fallback for cases with brackets but no recognized location keywords.

    Parameters
    ----------
    loca_ids_with_counts : list[tuple[str, int]]
        List of (LOCA_ID, table_count) tuples, sorted by table_count descending.
    investigation_id : str
        The InvestigationId string.

    Returns
    -------
    str | None
        Matched LOCA_ID if found, None otherwise.
    """
    bracket_match = re.search(r"\[([^\]]+)\]", investigation_id)
    if not bracket_match:
        return None

    bracket_content = bracket_match.group(1)
    parts = re.split(r"[-_]", bracket_content)
    number_match = re.search(r"(\d+)", bracket_content)

    if not number_match:
        return None

    number = int(number_match.group(1))  # Convert to int to handle leading zeros
    prefix = parts[0].upper() if len(parts) >= 2 else None

    loca_ids = [loca_id for loca_id, _count in loca_ids_with_counts]

    # Find LOCA_IDs with this number
    candidates = []
    for loca_id in loca_ids:
        loca_number_match = re.search(r"(\d+)", loca_id)
        if loca_number_match and int(loca_number_match.group(1)) == number:
            candidates.append(loca_id)

    if not candidates:
        return None

    # Filter by prefix if available
    if prefix:
        prefix_candidates = [
            lid
            for lid in candidates
            if prefix in lid.upper() or prefix[:2] in lid.upper()
        ]
        if prefix_candidates:
            return prefix_candidates[0]

    # Return first candidate if no prefix match
    return candidates[0]


def _try_context_based_match(
    loca_ids_with_counts: list[tuple[str, int]],
    investigation_id: str,
) -> str | None:
    """Try to match using context-based strategy.

    When InvestigationId contains location/context keywords that map to prefixes,
    use that prefix along with number from bracket.

    This fallback method was developed to address problematic NZGD IDs where
    simple substring matching failed:
    - NZGD ID 124644 (InvestigationId: "Michael Fowler Centre GENZWELL16138AA [MF_BH3]")
      → MF_Boring3 (matches "Michael Fowler" keyword to "MF" prefix)
    - NZGD ID 124645 (InvestigationId: "Michael Fowler Centre GENZWELL16138AA [MF_BH4]")
      → MF_Boring4 (matches "Michael Fowler" keyword to "MF" prefix)
    - NZGD ID 124646 (InvestigationId: "Michael Fowler Centre GENZWELL16138AA [MF_BH5]")
      → MF_Boring5 (matches "Michael Fowler" keyword to "MF" prefix)
    - NZGD ID 124647 (InvestigationId: "Wellington Town Hall GENZWELL16138AA [Aurecon-BH02]")
      → TH_BH02 (matches "Town Hall" keyword to "TH" prefix)

    Parameters
    ----------
    loca_ids_with_counts : list[tuple[str, int]]
        List of (LOCA_ID, table_count) tuples, sorted by table_count descending.
    investigation_id : str
        The InvestigationId string.

    Returns
    -------
    str | None
        Matched LOCA_ID if found, None otherwise.
    """
    # Keyword to prefix mapping
    keyword_to_prefix = {
        "town hall": "TH",
        "michael fowler": "MF",
        # Add more mappings as needed
    }

    investigation_upper = investigation_id.upper()

    # Find matching keyword
    matched_prefix = None
    for keyword, prefix in keyword_to_prefix.items():
        if keyword.upper() in investigation_upper:
            matched_prefix = prefix
            break

    if not matched_prefix:
        return None

    # Extract number from bracket
    bracket_match = re.search(r"\[([^\]]+)\]", investigation_id)
    if not bracket_match:
        return None

    bracket_content = bracket_match.group(1)
    number_match = re.search(r"(\d+)", bracket_content)
    if not number_match:
        return None

    number = int(number_match.group(1))

    # Find LOCA_IDs with this prefix and number
    loca_ids = [loca_id for loca_id, _count in loca_ids_with_counts]
    prefix_loca_ids = [
        lid for lid in loca_ids if lid.upper().startswith(matched_prefix + "_")
    ]

    matching_loca_ids = [lid for lid in prefix_loca_ids if str(number) in lid]

    if matching_loca_ids:
        return matching_loca_ids[0]

    return None


def _find_matching_loca_from_investigation(
    loca_ids_with_counts: list[tuple[str, int]],
    investigation_id: str | None,
) -> str | None:
    """Find the best matching LOCA_ID by checking if it appears in InvestigationId.

    This function works through LOCA_IDs sorted by the number of tables they appear in
    (descending order), considering only ISPT and GEOL tables. For each LOCA_ID, it checks
    if the LOCA_ID appears in the InvestigationId string. The first match is returned,
    prioritizing LOCA_IDs that appear in more tables (ISPT and/or GEOL).

    If simple substring matching fails, alternative strategies are tried as fallbacks:
    1. Context-based matching (e.g., "Town Hall" → "TH", "Michael Fowler" → "MF")
    2. Single prefix match (e.g., NZGD IDs 31205, 97660) - if only one LOCA_ID has
       a prefix that appears in InvestigationId, that one is selected
    3. BH-t pattern matching (e.g., "BH-t5" → "BH5" for NZGD ID 99137)
    4. Bracket pattern matching

    These fallback methods successfully match all 7 previously problematic NZGD IDs:
    - 31205 ("91 River Road BH, HAs, DCPs") → BH01
      (only one LOCA_ID has "BH" prefix, so that one is selected)
    - 97660 ("BH-t2") → BH2
      (only one LOCA_ID has "BH" prefix, so that one is selected)
    - 99137 ("BH-t5") → BH5
    - 124644 ("Michael Fowler Centre GENZWELL16138AA [MF_BH3]") → MF_Boring3
    - 124645 ("Michael Fowler Centre GENZWELL16138AA [MF_BH4]") → MF_Boring4
    - 124646 ("Michael Fowler Centre GENZWELL16138AA [MF_BH5]") → MF_Boring5
    - 124647 ("Wellington Town Hall GENZWELL16138AA [Aurecon-BH02]") → TH_BH02

    Parameters
    ----------
    loca_ids_with_counts : list[tuple[str, int]]
        List of (LOCA_ID, table_count) tuples, sorted by table_count descending.
    investigation_id : str | None
        The InvestigationId string to search in.

    Returns
    -------
    str | None
        The first LOCA_ID that appears in the InvestigationId string, or None if
        no match is found or investigation_id is None/empty.
    """
    if not investigation_id:
        return None

    investigation_upper = investigation_id.upper()

    # Strategy 1: Simple substring match (primary method)
    # Work through LOCA_IDs in order (already sorted by table count descending)
    # For each LOCA_ID, check if it appears as a substring in the InvestigationId
    # Example: LOCA_ID "BH434" should be found in InvestigationId "City Rail Link Stage 4 [BH434]"
    for loca_id, _table_count in loca_ids_with_counts:
        loca_upper = loca_id.upper()
        # Check if LOCA_ID is contained within InvestigationId (not the other way around)
        # This checks: is "BH434" in "City Rail Link Stage 4 [BH434]"?
        if loca_upper in investigation_upper:
            return loca_id

    # Strategy 2: Single prefix match (fallback)
    match = _try_single_prefix_match(loca_ids_with_counts, investigation_id)
    if match:
        return match

    # Strategy 3: BH-t pattern matching (fallback)
    match = _try_bh_t_pattern_match(loca_ids_with_counts, investigation_id)
    if match:
        return match

    # Strategy 4: Context-based matching (fallback - try before bracket pattern)
    # This should run before bracket pattern to prioritize context when available
    match = _try_context_based_match(loca_ids_with_counts, investigation_id)
    if match:
        return match

    # Strategy 5: Bracket pattern matching (fallback)
    match = _try_bracket_pattern_match(loca_ids_with_counts, investigation_id)
    if match:
        return match

    return None


@lru_cache(maxsize=1)
def _load_index_data() -> pd.DataFrame:
    """Load NZGD index metadata."""
    try:
        return pd.read_csv(constants.INDEX_FILE_PATH, low_memory=False)
    except FileNotFoundError:
        warnings.warn(f"Index file not found at {constants.INDEX_FILE_PATH}")
    except Exception as exc:
        warnings.warn(f"Failed to load index file {constants.INDEX_FILE_PATH}: {exc}")
    return pd.DataFrame(columns=["nzgd_id", "InvestigationId"])


def _get_investigation_id(nzgd_id: int) -> str | None:
    """Return the raw InvestigationId string for the given nzgd_id.

    Parameters
    ----------
    nzgd_id : int
        The NZGD ID to look up.

    Returns
    -------
    str | None
        The InvestigationId string if found, None otherwise.
    """
    index_df = _load_index_data()
    if index_df.empty or "InvestigationId" not in index_df.columns:
        return None

    matches = index_df.loc[index_df["nzgd_id"] == nzgd_id, "InvestigationId"]
    if matches.empty:
        return None

    value = matches.iloc[0]
    if pd.isna(value) or value == "":
        return None

    return str(value).strip()


def _filter_by_investigation(
    df: pd.DataFrame,
    investigation_id: str | None,
    diagnostics: LocaDiagnostics | None = None,
) -> pd.DataFrame:
    """Filter a table to rows matching the matched location ID (LOCA_ID or HOLE_ID) from diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    investigation_id : str | None
        The InvestigationId string (kept for compatibility, but matching is done via diagnostics).
    diagnostics : LocaDiagnostics | None
        Diagnostics object containing the matched location ID.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows with the matched location ID.
        Returns empty DataFrame if:
        - No match found in InvestigationId, OR
        - Matched location ID doesn't exist in this table
        Returns original DataFrame if only one location ID exists (no filtering needed).
    """
    location_id_col = _get_location_id_column_name(df)
    if not location_id_col:
        return df

    unique_loca_ids = sorted(
        {
            str(value).strip()
            for value in df[location_id_col].dropna().unique()
            if str(value).strip()
        }
    )

    # If only one location ID exists, no filtering needed
    if len(unique_loca_ids) <= 1:
        return df

    # Use the matched location ID from diagnostics if available
    matched_loca_id = diagnostics.matched_loca_id if diagnostics else None

    # If no match was found in InvestigationId, return empty DataFrame
    if not matched_loca_id:
        if diagnostics:
            if not investigation_id:
                if not diagnostics.warned_no_InvestigationID:
                    diagnostics.warned_no_InvestigationID = True
                    warnings.warn(
                        f"Multiple {location_id_col} values present but no InvestigationId found; "
                        "returning empty DataFrame."
                    )
            else:
                if not diagnostics.warned_no_match:
                    diagnostics.warned_no_match = True
                    warnings.warn(
                        f"No {location_id_col} found in InvestigationId '{investigation_id}'; "
                        "returning empty DataFrame."
                    )
        else:
            # Fallback: warn if no diagnostics available
            if not investigation_id:
                warnings.warn(
                    f"Multiple {location_id_col} values present but no InvestigationId found; "
                    "returning empty DataFrame."
                )
            else:
                warnings.warn(
                    f"No {location_id_col} found in InvestigationId '{investigation_id}'; "
                    "returning empty DataFrame."
                )
        # Return empty DataFrame with same columns
        return df.iloc[0:0].copy()

    # Try to filter by matched location ID
    mask = (
        df[location_id_col].astype(str).str.strip().str.upper()
        == matched_loca_id.strip().upper()
    )

    if mask.any():
        # Matched location ID exists in this table, return filtered rows
        return df[mask]
    else:
        # Matched location ID doesn't exist in this table, return empty DataFrame
        if diagnostics and not diagnostics.warned_no_match:
            diagnostics.warned_no_match = True
            warnings.warn(
                f"Matched {location_id_col} '{matched_loca_id}' from InvestigationId "
                f"'{investigation_id}' not found in this table; returning empty DataFrame."
            )
        # Return empty DataFrame with same columns
        return df.iloc[0:0].copy()


def _extract_density_from_description(description: str) -> list[str]:
    """Extract density-related keywords from a soil description string.

    Extracts all density-related keywords from DENSITY_RELATED_KEYWORDS along with
    their modifiers (e.g., "very", "high", "low", "medium", "well", "poorly", etc.).
    Patterns are matched in order from most specific to least specific to avoid
    double-counting (e.g., "very dense" is matched before standalone "dense").

    Parameters
    ----------
    description : str
        Soil description text (e.g., from GEOL_DESC).

    Returns
    -------
    list[str]
        List of density keywords found with modifiers (e.g., ['very dense', 'loose', 'well compacted']).
        Compound phrases are prioritized over standalone terms.
    """
    if pd.isna(description) or not description:
        return []

    description_str = str(description).lower()

    # Build patterns: (pattern_regex, expected_output_text)
    patterns = []

    # Separate single-word and multi-word keywords
    single_word_keywords = [
        kw
        for kw in constants.DENSITY_RELATED_KEYWORDS
        if " " not in kw and "-" not in kw
    ]
    multi_word_keywords = [
        kw for kw in constants.DENSITY_RELATED_KEYWORDS if " " in kw or "-" in kw
    ]

    # First, add patterns with modifiers + single-word keywords (most specific)
    for modifier_pattern, modifier_text in constants.DENSITY_MODIFIERS:
        for keyword in single_word_keywords:
            keyword_escaped = re.escape(keyword)
            pattern = rf"{modifier_pattern}\s+{keyword_escaped}\b"
            patterns.append((pattern, f"{modifier_text} {keyword}"))

    # Then add multi-word keywords (they may already contain modifiers)
    for keyword in multi_word_keywords:
        keyword_escaped = re.escape(keyword)
        pattern = rf"{keyword_escaped}"
        patterns.append((pattern, keyword))

    # Finally, add standalone single-word keywords (least specific)
    for keyword in single_word_keywords:
        keyword_escaped = re.escape(keyword)
        pattern = rf"\b{keyword_escaped}\b"
        patterns.append((pattern, keyword))

    found_phrases = []
    matched_positions = set()  # Track positions to avoid overlapping matches

    # Sort patterns by length (longest first) to prioritize more specific matches
    patterns.sort(key=lambda x: len(x[0]), reverse=True)

    for pattern, expected_text in patterns:
        matches = re.finditer(pattern, description_str, re.IGNORECASE)
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()

            # Check if this position overlaps with a previously matched phrase
            overlaps = any(
                start_pos < prev_end and end_pos > prev_start
                for prev_start, prev_end in matched_positions
            )

            if not overlaps:
                # Extract the actual matched text and normalize whitespace
                matched_text = match.group().strip()
                matched_text = re.sub(r"\s+", " ", matched_text)
                found_phrases.append(matched_text)
                matched_positions.add((start_pos, end_pos))

    return found_phrases


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


def _get_location_id_column_name(df: pd.DataFrame) -> str | None:
    """Get the location ID column name (LOCA_ID or HOLE_ID) from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check for location ID columns.

    Returns
    -------
    str | None
        "LOCA_ID" if present, "HOLE_ID" if LOCA_ID is not present but HOLE_ID is,
        or None if neither is present.
    """
    if "LOCA_ID" in df.columns:
        return "LOCA_ID"
    elif "HOLE_ID" in df.columns:
        return "HOLE_ID"
    return None


def _collect_loca_ids_and_table_counts(
    tables: dict[str, pd.DataFrame],
) -> list[tuple[str, int]]:
    """Collect all location IDs (LOCA_ID or HOLE_ID) and count how many tables each appears in.

    Only counts occurrences in ISPT and GEOL tables, ignoring all other tables.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Dictionary of table names to DataFrames loaded from an AGS file.

    Returns
    -------
    list[tuple[str, int]]
        List of (location_id, table_count) tuples, sorted by table_count descending,
        then by location_id alphabetically. Location IDs appearing in more tables come first.
        Only counts occurrences in ISPT and GEOL tables.
    """
    loca_id_to_tables: dict[str, set[str]] = {}

    # Find only ISPT and GEOL tables that have a location ID column (LOCA_ID or HOLE_ID)
    tables_with_loca_id = {}
    for table_name, table in tables.items():
        if isinstance(table, pd.DataFrame) and table_name in ("ISPT", "GEOL"):
            location_id_col = _get_location_id_column_name(table)
            if location_id_col:
                tables_with_loca_id[table_name] = (table, location_id_col)

    # Process each table
    for table_name, (table, location_id_col) in tables_with_loca_id.items():
        series = table[location_id_col]
        # Skip the first 2 rows if HEADING column exists (UNIT and TYPE rows)
        if "HEADING" in table.columns and len(series) > 2:
            series = series.iloc[2:]
        for value in series.dropna():
            value_str = str(value).strip()
            if value_str:
                # Track which table this location ID appears in
                if value_str not in loca_id_to_tables:
                    loca_id_to_tables[value_str] = set()
                loca_id_to_tables[value_str].add(table_name)

    # Convert to list of tuples and sort by table count (descending), then by location ID
    result = [
        (loca_id, len(tables_set)) for loca_id, tables_set in loca_id_to_tables.items()
    ]
    result.sort(key=lambda x: (-x[1], x[0]))  # Negative for descending order

    return result


def _build_loca_diagnostics(
    tables: dict[str, pd.DataFrame],
    investigation_id: str | None,
    nzgd_id: int,
    ags_file_name: str,
) -> LocaDiagnostics:
    """Build LOCA_ID diagnostics using the new approach.

    The new approach:
    1. Collects all LOCA_IDs and sorts them by number of tables they appear in (descending)
    2. Checks if each LOCA_ID appears in the InvestigationId string
    3. Uses the first matching LOCA_ID (prioritizing those in more tables)

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Dictionary of table names to DataFrames loaded from an AGS file.
    investigation_id : str | None
        The raw InvestigationId string from the index.
    nzgd_id : int
        The NZGD ID being processed.
    ags_file_name : str
        The name of the AGS file being processed.

    Returns
    -------
    LocaDiagnostics
        Diagnostics object with matching information.
    """
    diagnostics = LocaDiagnostics(
        nzgd_id=nzgd_id,
        ags_file_name=ags_file_name,
        investigation_raw=investigation_id,
    )

    # Collect LOCA_IDs sorted by table count (descending)
    loca_ids_with_counts = _collect_loca_ids_and_table_counts(tables)
    unique_ids = [loca_id for loca_id, _count in loca_ids_with_counts]

    diagnostics.has_multiple = len(unique_ids) > 1

    if diagnostics.has_multiple and unique_ids:
        diagnostics.all_loca_ids = "|".join(unique_ids)
        # Find matching LOCA_ID by checking if it appears in InvestigationId
        matched_loca_id = _find_matching_loca_from_investigation(
            loca_ids_with_counts, investigation_id
        )
        if matched_loca_id:
            diagnostics.found_match = True
            diagnostics.matched_loca_id = matched_loca_id
        else:
            diagnostics.found_match = False
            diagnostics.matched_loca_id = None
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

    investigation_id = _get_investigation_id(borehole_id)

    diagnostics = _build_loca_diagnostics(
        tables,
        investigation_id,
        borehole_id,
        report.name,
    )

    # Determine which location ID column name to use (LOCA_ID or HOLE_ID) for each table
    ispt_location_id_col = None
    geol_location_id_col = None

    if "ISPT" in tables and isinstance(tables["ISPT"], pd.DataFrame):
        ispt_location_id_col = _get_location_id_column_name(tables["ISPT"])
    if "GEOL" in tables and isinstance(tables["GEOL"], pd.DataFrame):
        geol_location_id_col = _get_location_id_column_name(tables["GEOL"])

    # Build column lists, only including location ID column if it exists
    # Copy the constant list (using list()) so we can safely prepend the location ID column.
    ispt_columns = list(constants.ISPT_COLUMNS)
    if ispt_location_id_col:
        ispt_columns.insert(0, ispt_location_id_col)

    geology_columns = ["GEOL_TOP", "GEOL_BASE", "GEOL_DESC"]
    # Also include GEOL_NAME if it exists, as it may contain soil type information
    if "GEOL" in tables and isinstance(tables["GEOL"], pd.DataFrame):
        if "GEOL_NAME" in tables["GEOL"].columns:
            geology_columns.append("GEOL_NAME")
    if geol_location_id_col:
        geology_columns.insert(0, geol_location_id_col)

    # Copy the constant list (using list()) so we can safely prepend the location ID column.
    desired_ispt_columns = list(constants.DESIRED_ISPT_COLUMNS)
    if ispt_location_id_col:
        desired_ispt_columns.insert(0, ispt_location_id_col)

    # Initialize SPT table with empty DataFrame
    spt_table = pd.DataFrame(columns=desired_ispt_columns)
    # Try to extract SPT data if ISPT table exists
    if "ISPT" in tables:
        ispt_df = tables["ISPT"].iloc[2:].copy()
        ispt_df = _filter_by_investigation(ispt_df, investigation_id, diagnostics)
        ispt_df = ispt_df.reindex(columns=ispt_columns, fill_value=pd.NA)

        spt_table = ispt_df.rename(columns={"ISPT_TOP": "Depth"}).reindex(
            columns=desired_ispt_columns, fill_value=pd.NA
        )
        # If all N values are empty, create empty DataFrame with NaN values
        if spt_table["ISPT_MAIN"].eq("").all() and spt_table["ISPT_NVAL"].eq("").all():
            spt_table = pd.DataFrame(columns=desired_ispt_columns)
            warnings.warn(
                f"No SPT ISPT_MAIN or ISPT_NVAL values found in {report}, creating empty SPT measurements"
            )

    # Initialize geology table
    geology_table = pd.DataFrame(columns=geology_columns)

    # Initialize density measurements table (depth-specific from GEOL)
    density_measurements = pd.DataFrame(
        columns=[
            "top_depth_m",
            "bottom_depth_m",
            "density_description",
        ]
    )

    # Try to extract geology data if GEOL table exists
    if "GEOL" in tables and tables.get("GEOL") is not None:
        geol_df = tables["GEOL"][geology_columns].iloc[2:].copy()
        geol_df = _filter_by_investigation(geol_df, investigation_id, diagnostics)

        # Rename columns, including GEOL_BASE if it exists
        rename_dict = {"GEOL_TOP": "top_depth", "GEOL_DESC": "soil_types"}
        if "GEOL_BASE" in geol_df.columns:
            rename_dict["GEOL_BASE"] = "bottom_depth"
        geology_table = geol_df.rename(columns=rename_dict)

        # Extract soil types from GEOL_DESC
        geology_table["soil_types"] = geology_table["soil_types"].apply(
            extract_soil_report,
        )

        # If GEOL_NAME exists, also extract soil types from it and combine with GEOL_DESC results
        if "GEOL_NAME" in geology_table.columns:

            def extract_from_both(
                desc_soil_types: set[str], geol_name: Any
            ) -> set[str]:
                """Extract soil types from both GEOL_DESC and GEOL_NAME."""
                # Start with soil types from GEOL_DESC
                combined_soil_types = (
                    desc_soil_types.copy() if desc_soil_types else set()
                )

                # Also extract from GEOL_NAME if it exists and is not empty
                if pd.notna(geol_name) and str(geol_name).strip():
                    name_soil_types = extract_soil_report(str(geol_name))
                    combined_soil_types.update(name_soil_types)

                return combined_soil_types

            # Combine soil types from both columns
            geology_table["soil_types"] = geology_table.apply(
                lambda row: extract_from_both(
                    row["soil_types"], row.get("GEOL_NAME", "")
                ),
                axis=1,
            )

        # Extract density measurements from GEOL descriptions
        # Use GEOL_DESC if available, otherwise try GEOL_NAME
        density_rows = []
        for idx, row in geol_df.iterrows():
            description = row.get("GEOL_DESC", "")
            # If GEOL_DESC is empty, try GEOL_NAME as fallback
            if (
                not description
                or pd.isna(description)
                or str(description).strip() == ""
            ) and "GEOL_NAME" in row:
                description = row.get("GEOL_NAME", "")
            top_depth = row.get("GEOL_TOP", None)

            # Try to get bottom depth if GEOL_BASE exists
            bottom_depth = None
            if "GEOL_BASE" in geol_df.columns:
                bottom_depth = row.get("GEOL_BASE", None)

            # Extract density keywords (with modifiers like "very", "high", etc.)
            density_keywords = _extract_density_from_description(description)

            # Try to convert depths to float
            top_depth_float = None
            bottom_depth_float = None

            try:
                if top_depth is not None and str(top_depth).strip():
                    top_depth_float = float(str(top_depth).strip())
            except (ValueError, TypeError):
                pass

            try:
                if bottom_depth is not None and str(bottom_depth).strip():
                    bottom_depth_float = float(str(bottom_depth).strip())
            except (ValueError, TypeError):
                pass

            # Only add density rows if we have a valid top_depth (required field)
            if top_depth_float is not None:
                # Create rows for density keywords
                for density_keyword in density_keywords:
                    density_rows.append(
                        {
                            "top_depth_m": top_depth_float,
                            "bottom_depth_m": bottom_depth_float,
                            "density_description": density_keyword,  # Column name kept for compatibility
                        }
                    )

        if density_rows:
            density_measurements = pd.DataFrame(density_rows)

    diameter = None
    if "HDIA" in tables and tables.get("HDIA") is not None:
        hdia_df = tables["HDIA"].iloc[2:].copy()
        hdia_location_id_col = _get_location_id_column_name(tables["HDIA"])
        if hdia_location_id_col:
            hdia_df = _filter_by_investigation(hdia_df, investigation_id, diagnostics)
        diameter = _first_numeric_value(hdia_df, "HDIA_DIAM")

    efficiency = _first_numeric_value(spt_table, "ISPT_ERAT")
    groundwater_level = _first_numeric_value(spt_table, "ISPT_WAT")

    # Validate groundwater_level against allowed bounds
    if groundwater_level is not None:
        if (
            groundwater_level > constants.MAX_ALLOWED_GWL
            or groundwater_level < constants.MIN_ALLOWED_GWL
        ):
            groundwater_level = None

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
    # Check SPT data: either ISPT_MAIN or ISPT_NVAL should have values
    has_spt_data = False
    if not spt_table.empty:
        has_main = (
            "ISPT_MAIN" in spt_table.columns and not spt_table["ISPT_MAIN"].eq("").all()
        )
        has_nval = (
            "ISPT_NVAL" in spt_table.columns and not spt_table["ISPT_NVAL"].eq("").all()
        )
        has_spt_data = has_main or has_nval

    # Check if geology_table has any rows with non-empty soil_types
    has_soil_data = (
        not geology_table.empty
        and geology_table["soil_types"]
        .apply(lambda x: len(x) > 0 if isinstance(x, set) else False)
        .any()
    )
    has_density_data = not density_measurements.empty
    has_efficiency = efficiency is not None
    has_diameter_data = diameter is not None

    if not (has_spt_data or has_soil_data or has_density_data or has_efficiency or has_diameter_data):
        raise ValueError(
            f"No meaningful data extracted from {report}: no SPT measurements, soil measurements, density measurements, efficiency, or diameter found"
        )

    return BoreholeProcessingResult(
        report=SPTReport(
            borehole_id=borehole_id,
            nzgd_id=borehole_id,
            efficiency=efficiency,
            extracted_gwl=groundwater_level,
            extracted_diameter=diameter,
            source_file=report,
            spt_measurements=spt_table,
            soil_measurements=geology_table,
            density_measurements=density_measurements,
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

    # Create a unique extraction status file for this worker process
    # Use process ID and borehole ID to ensure uniqueness
    process_id = os.getpid()
    status_file = (
        constants.SPT_AGS_EXTRACTION_WARNINGS_DIR
        / f"extraction_status_{process_id}_{borehole_id}.csv"
    )

    # Track extraction status for this file
    status_logger = ExtractionStatusLogger(status_file, borehole_id, report.name)
    with status_logger:
        try:
            result = process_borehole(borehole_id, report)

            # Update extraction status based on actual results
            if result:
                report_obj = result.report
                # Check SPT data: either ISPT_MAIN or ISPT_NVAL should have values
                extracted_spt = False
                if not report_obj.spt_measurements.empty:
                    has_main = (
                        "ISPT_MAIN" in report_obj.spt_measurements.columns
                        and not report_obj.spt_measurements["ISPT_MAIN"].eq("").all()
                    )
                    has_nval = (
                        "ISPT_NVAL" in report_obj.spt_measurements.columns
                        and not report_obj.spt_measurements["ISPT_NVAL"].eq("").all()
                    )
                    extracted_spt = has_main or has_nval
                extracted_soil = (
                    not report_obj.soil_measurements.empty
                    and report_obj.soil_measurements["soil_types"]
                    .apply(lambda x: len(x) > 0 if isinstance(x, set) else False)
                    .any()
                )
                extracted_density = not report_obj.density_measurements.empty
                extracted_efficiency = report_obj.efficiency is not None
                extracted_diameter = report_obj.extracted_diameter is not None

                # Update status before context exits
                status_logger.update_extraction_status(
                    extracted_spt=extracted_spt,
                    extracted_soil=extracted_soil,
                    extracted_density=extracted_density,
                    extracted_efficiency=extracted_efficiency,
                    extracted_diameter=extracted_diameter,
                )

            return result
        except Exception as e:
            warnings.warn(f"Failed to process {report}: {e}")
            # Update status to indicate failure - all extractions failed
            status_logger.update_extraction_status(
                extracted_spt=False,
                extracted_soil=False,
                extracted_density=False,
                extracted_efficiency=False,
                extracted_diameter=False,
            )
            return None


def _consolidate_warning_files():
    """Consolidate all extraction status CSV files from worker processes into a single file."""
    if not constants.SPT_AGS_EXTRACTION_WARNINGS_DIR.exists():
        return

    # Find all extraction status CSV files
    status_files = list(
        constants.SPT_AGS_EXTRACTION_WARNINGS_DIR.glob("extraction_status_*.csv")
    )

    if not status_files:
        return

    # Read and combine all status files
    all_statuses: list[dict[str, Any]] = []
    for status_file in status_files:
        try:
            status_df = pd.read_csv(status_file)
            all_statuses.extend(status_df.to_dict("records"))
        except Exception:
            # Skip files that can't be read
            continue

    if not all_statuses:
        return

    # Create consolidated DataFrame and write to main file
    consolidated_df = pd.DataFrame(all_statuses)
    consolidated_df.sort_values(
        by=["nzgd_id", "file_name"], inplace=True, ignore_index=True
    )

    consolidated_file = (
        constants.SPT_AGS_EXTRACTION_WARNINGS_DIR / "all_spt_ags_extraction_status.csv"
    )
    consolidated_df.to_csv(consolidated_file, index=False)

    # Clean up individual status files
    for status_file in status_files:
        try:
            status_file.unlink()
        except Exception:
            # Ignore errors when deleting files
            pass


def _row_value_or_none(row: pd.Series, col: str) -> Any:
    """Return the value of a column from a row, or None if it is missing or empty.

    Parameters
    ----------
    row : pd.Series
        A row from a pandas DataFrame, typically obtained via ``iterrows()``.
    col : str
        The column name to look up in ``row``.

    Returns
    -------
    Any
        The value at ``row[col]`` if the column is present, the value is not
        null (``pd.isna``), and the value is not an empty string.
        Returns ``None`` otherwise.
    """
    if col not in row or pd.isna(row[col]) or row[col] == "":
        return None
    return row[col]


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
            report.extracted_diameter,
            report.source_file.name,
        )
        for report in reports
    ]
    cursor.executemany(
        """
        INSERT OR REPLACE INTO sptreport (spt_id, nzgd_id, efficiency, extracted_gwl_m, borehole_diameter, source_file)
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

    cursor.execute("SELECT id, value FROM SoilTypes")
    soil_type_id_map = {
        value: soil_type_id for soil_type_id, value in cursor.fetchall()
    }

    # Insert DensityMeasurements (depth-specific from GEOL)
    density_data = []
    for report in reports:
        for _, row in report.density_measurements.iterrows():
            top_depth = row.get("top_depth_m")
            bottom_depth = row.get("bottom_depth_m")
            density_keyword = row.get("density_description")

            # Store density keyword directly as text (no normalization)
            density_keyword_str = None
            if density_keyword and pd.notna(density_keyword):
                density_keyword_str = str(density_keyword).strip()

            # Ensure top_depth is not None (required field)
            if top_depth is None or pd.isna(top_depth):
                continue  # Skip rows without depth (shouldn't happen for GEOL data)

            density_data.append(
                (
                    report.borehole_id,
                    float(top_depth),
                    float(bottom_depth)
                    if bottom_depth is not None and pd.notna(bottom_depth)
                    else None,
                    density_keyword_str,
                )
            )

    if density_data:
        cursor.executemany(
            """
            INSERT INTO densitymeasurements 
            (spt_id, top_depth_m, bottom_depth_m, density_keyword)
            VALUES (?, ?, ?, ?)
        """,
            density_data,
        )

    # Insert SPTMeasurements and SPTMeasurementSoilTypes
    for report in reports:
        # Only insert SPT measurements if the DataFrame is not empty
        if not report.spt_measurements.empty:
            for _, row in report.spt_measurements.iterrows():
                # Handle NaN/None values by converting to None for SQLite
                depth = row["Depth"] if pd.notna(row["Depth"]) else None

                ispt_main_n = _row_value_or_none(row, "ISPT_MAIN")
                ispt_nval = _row_value_or_none(row, "ISPT_NVAL")
                ispt_rep = _row_value_or_none(row, "ISPT_REP")

                cursor.execute(
                    """
                    INSERT INTO sptmeasurements (spt_id, depth_m, ISPT_MAIN, ISPT_NVAL, ISPT_REP)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        report.borehole_id,
                        depth,
                        ispt_main_n,
                        ispt_nval,
                        ispt_rep,
                    ),
                )

        # Only process soil measurements if the DataFrame is not empty
        if not report.soil_measurements.empty:
            for _, row in report.soil_measurements.iterrows():
                if not row["soil_types"]:
                    continue
                # Handle NaN/None values for top_depth and bottom_depth
                top_depth = _row_value_or_none(row, "top_depth")
                bottom_depth = None
                if "bottom_depth" in row:
                    bottom_depth = _row_value_or_none(row, "bottom_depth")

                cursor.execute(
                    """
                                   INSERT INTO soilmeasurements (spt_id, top_depth_m, bottom_depth_m)
                                   VALUES (?, ?, ?)
                               """,
                    (report.borehole_id, top_depth, bottom_depth),
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
                "Borehole Diameter": spt_report.extracted_diameter,
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
    # Ensure warnings directory exists
    constants.SPT_AGS_EXTRACTION_WARNINGS_DIR.mkdir(parents=True, exist_ok=True)

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
    constants.SPT_AGS_LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(constants.SPT_AGS_LOG_FILE_PATH, index=False)

    # Consolidate all warning files into a single file
    _consolidate_warning_files()


if __name__ == "__main__":
    app()
