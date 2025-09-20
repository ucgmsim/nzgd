"""Data structures for handling CPT data extraction."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import pandas as pd


class FilePathList(NamedTuple):
    """Named tuple to store file paths organized by file type.

    Attributes
    ----------
    spreadsheet : list[Path]
        List of paths to spreadsheet files (.xls, .xlsx, .csv, .txt).
    ags : list[Path]
        List of paths to AGS format files.
    pdf : list[Path]
        List of paths to PDF files.
    other : list[Path]
        List of paths to other file types.

    """

    spreadsheets: list[Path]
    ags: list[Path]
    nondata: list[Path]


class SearchForColResults(NamedTuple):
    """Named tuple to store search results.

    Attributes
    ----------
    col_index_in_line: int
        The index in the delimited line where the search term was found.
    search_term: str
        The search term that was used to find the match.
    matched_string: str
        The string that matched the search term in the line.

    """

    col_index_in_line: int
    search_term: str
    matched_string: str


class AllCPTColsSearchResults(NamedTuple):
    """Named tuple to store the search for each CPT column.

    Attributes
    ----------
    col1_search_result: list[SearchForColResults]
        The search result for the first column.
    col2_search_result: list[SearchForColResults]
        The search result for the second column.
    col3_search_result: list[SearchForColResults]
        The search result for the third column.
    col4_search_result: list[SearchForColResults]
        The search result for the fourth column.

    """

    col1_search_result: list[SearchForColResults]
    col2_search_result: list[SearchForColResults]
    col3_search_result: list[SearchForColResults]
    col4_search_result: list[SearchForColResults]

    @property
    def all_search_results(self) -> list[SearchForColResults]:
        """Get all search results combined into a single list.

        Returns
        -------
        list[SearchForColResults]
            A list containing all search results from all columns.

        """
        return (
            self.col1_search_result
            + self.col2_search_result
            + self.col3_search_result
            + self.col4_search_result
        )

    @property
    def col_idx_to_name(self) -> dict[int, str]:
        """Get a dictionary mapping column indices to their matched strings.

        Returns
        -------
        dict[int, str]
            A dictionary with column indices as keys and matched strings as values.

        Notes
        -----
        If multiple search results have the same column index, the last one encountered
        will be used in the dictionary.

        """
        mapping = {}
        for result in self.all_search_results:
            mapping[result.col_index_in_line] = result.matched_string
        return mapping


@dataclass
class ExtractedDataAndColInfo:
    """Stores the extracted columns and associated information."""

    data_df: pd.DataFrame
    col_info: AllCPTColsSearchResults


@dataclass
class SheetExtractionResult:
    """Dataclass to store the outcome of an extraction attempt from one sheet.

    Attributes
    ----------
    extraction : ExtractedDataAndColInfo | pd.DataFrame
        If the extraction was successful, this will be an ExtractedDataAndColInfo object
        If the extraction failed, this will be a DataFrame summarizing the failure.
    file_path: Path | NoneType
        Path to the file being processed.
    sheet_name: str, default="0"
        Name of the sheet being processed. CSV and TXT files cannot have multiple
        sheets so their sheet name is always "0". Excel files can have multiple sheets.

    """

    extraction: ExtractedDataAndColInfo | pd.DataFrame
    file_path: Path = Path()
    sheet_name: str = "0"
    removed_duplicates: list | str | None = None
    explicit_unit_conversions: str | None = None
    inferred_unit_conversions: str | None = None


@dataclass
class NumStrAndFloat:
    """Dataclass to store surplus of numeric values in a DataFrame.

    Attributes
    ----------
    numerical_surplus_per_row : npt.NDArray[np.float]
        Surplus of numeric values per row.
    numerical_surplus_per_col : npt.NDArray[np.float]
        Surplus of numeric values per column.
    num_numerical_values : np.float
        Total number of numerical values in the DataFrame.

    """

    num_numerical_values_per_row: pd.Series
    num_numerical_values_per_col: pd.Series
    numerical_surplus_per_row: pd.Series
    numerical_surplus_per_col: pd.Series
    num_numerical_values: float


@dataclass
class SearchableSheet:
    """A class to hold the results of a searchable sheet."""

    nzgd_id: int
    sheet: list[list[str | float]]
    file_path: Path
    sheet_name: str = "0"  # Default "0" for CSV and TXT files which do not have sheets


class DataOrientation(StrEnum):
    """Represent the orientation of the data in the sheet."""

    COLUMNS = "columns"
    ROWS = "rows"


@dataclass
class SearchResult:
    """Store search results."""

    search_assumption: str
    assumed_orientation: DataOrientation | None
    field_label_str: str | None
    value_str: str | None


@dataclass
class SearchInfoAndResult:
    """Store search information and results."""

    nzgd_id: int
    file_name: str
    sheet_name: str
    search_term: str
    likely_orientation: DataOrientation
    search_results: list[SearchResult]


class CellToCheck(StrEnum):
    """Options for checking neighboring cells."""

    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"
