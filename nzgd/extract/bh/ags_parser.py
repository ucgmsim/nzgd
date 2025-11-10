from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import chardet
import pandas as pd
from python_ags4 import AGS4
from python_ags4.AGS4 import AGS4Error

logger = logging.getLogger(__name__)


@dataclass
class ParsedAGSTables:
    tables: Dict[str, pd.DataFrame]
    headings: Dict[str, List[str]]


def load_ags_tables(
    filepath: Path,
    *,
    encoding: str | None = None,
    rename_duplicate_headers: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """Load AGS tables into pandas DataFrames.

    Attempts to use the official `AGS4.AGS4_to_dataframe` implementation first.
    If that returns empty results (common when GROUP headers are missing) or
    raises an error, a more tolerant parser is used that can handle AGS-like
    files that omit the ``GROUP``/``HEADING`` rows, such as
    ``Borehole_127920_AGS01.ags``.
    """

    try:
        tables, headings = AGS4.AGS4_to_dataframe(
            filepath,
            encoding=encoding or "utf-8",
            rename_duplicate_headers=rename_duplicate_headers,
        )
        if tables:
            return tables, headings
    except AGS4Error as exc:
        logger.debug("Standard AGS4 parsing failed for %s: %s", filepath, exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Unexpected error parsing %s with python_ags4: %s", filepath, exc
        )

    parsed = _parse_nonstandard_ags(filepath, encoding=encoding)
    return parsed.tables, parsed.headings


def _parse_nonstandard_ags(
    filepath: Path,
    *,
    encoding: str | None = None,
) -> ParsedAGSTables:
    detected_encoding = encoding or _detect_encoding(filepath)
    tables: Dict[str, pd.DataFrame] = {}
    headings: Dict[str, List[str]] = {}

    current_group: str | None = None
    current_headings: List[str] = []
    current_rows: List[List[str]] = []
    collecting_headings = False

    def flush_group() -> None:
        nonlocal current_group, current_headings, current_rows, collecting_headings
        if not current_group or not current_headings:
            current_group = None
            current_headings = []
            current_rows = []
            collecting_headings = False
            return

        if not any(row for row in current_rows):
            current_group = None
            current_headings = []
            current_rows = []
            collecting_headings = False
            return

        # Insert placeholder TYPE row if it is missing so downstream logic
        # that slices off UNIT/TYPE continues to operate.
        if not any(row[0] == "TYPE" for row in current_rows):
            current_rows.insert(1, ["TYPE"] + [""] * len(current_headings))

        columns = ["HEADING"] + current_headings
        normalized_rows = [_pad_or_truncate(row, len(columns)) for row in current_rows]
        tables[current_group] = pd.DataFrame(normalized_rows, columns=columns)
        headings[current_group] = columns

        current_group = None
        current_headings = []
        current_rows = []
        collecting_headings = False

    with filepath.open(
        encoding=detected_encoding,
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            row = _strip_trailing_blanks(raw_row)
            if not row:
                continue

            first_cell = row[0].strip()

            if first_cell.startswith("**"):
                flush_group()
                current_group = first_cell.lstrip("*")
                current_headings = []
                current_rows = []
                collecting_headings = True
                continue

            if first_cell.startswith("*") and collecting_headings:
                cleaned = [
                    cell.lstrip("*").strip() for cell in row if cell and cell.strip()
                ]
                current_headings.extend(cleaned)
                continue

            collecting_headings = False

            if current_group is None or not current_headings:
                logger.debug(
                    "Encountered data row before headings in %s: %s",
                    filepath,
                    row,
                )
                continue

            heading, values = _extract_row_heading_and_values(row)
            values = _ensure_length(values, len(current_headings))
            current_rows.append([heading] + values)

    flush_group()

    return ParsedAGSTables(tables=tables, headings=headings)


def _extract_row_heading_and_values(row: List[str]) -> Tuple[str, List[str]]:
    first_cell = row[0].strip()
    if first_cell.startswith("<") and first_cell.endswith(">"):
        tag = first_cell.strip("<>").strip().upper()
        if tag.endswith("S"):
            tag = tag[:-1]
        tag = {"UNIT": "UNIT", "TYPE": "TYPE", "DATA": "DATA"}.get(tag, tag)
        return tag, row[1:]

    return "DATA", row


def _ensure_length(values: List[str], expected: int) -> List[str]:
    if len(values) < expected:
        return values + [""] * (expected - len(values))
    if len(values) > expected:
        return values[:expected]
    return values


def _pad_or_truncate(row: List[str], expected_len: int) -> List[str]:
    if len(row) < expected_len:
        return row + [""] * (expected_len - len(row))
    if len(row) > expected_len:
        return row[:expected_len]
    return row


def _strip_trailing_blanks(row: Iterable[str]) -> List[str]:
    trimmed = list(row)
    while trimmed and trimmed[-1] == "":
        trimmed.pop()
    return trimmed


def _detect_encoding(filepath: Path) -> str:
    try:
        raw = filepath.read_bytes()
    except OSError:
        logger.debug("Failed to read bytes for encoding detection: %s", filepath)
        return "utf-8"

    detection = chardet.detect(raw)
    encoding = detection.get("encoding") or "utf-8"
    return encoding
