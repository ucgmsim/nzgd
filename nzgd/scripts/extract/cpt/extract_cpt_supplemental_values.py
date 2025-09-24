"""Extract supplemental values extracted from CPT investigations.

These supplemental values are ground water level, tip net area ratio, and the reason
for terminating the CPT investigation.
"""

import multiprocessing as mp
import re
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from nzgd import constants
from nzgd.extract.cpt import data_structures, errors, info, search, tasks


def get_csv_txt_iterable(
    nzgd_id: int,
    file_path: Path,
) -> data_structures.SearchableSheet:
    """Get an easily searchable representation of a CSV or TXT file.

    Parameters
    ----------
    nzgd_id : int
        The NZGD ID number for which to get searchable sheets.
    file_path : Path
        The path to the CSV or TXT file.

    Returns
    -------
    data_structures.SearchableSheet
        A searchable representation of the file.

    """
    file_encoding = info.find_encoding(file_path)

    # Type-guard against file_encoding=None
    if file_encoding is None:
        error_message = f"Could not determine encoding for {file_path}"
        raise errors.FileProcessingError(error_message)

    # Immediately return if the file is empty
    with Path.open(file_path, encoding=file_encoding) as file:
        content = file.read()
        if len(content) == 0:
            return data_structures.SearchableSheet(
                nzgd_id,
                sheet=[],
                file_path=file_path,
            )

    delimiter = info.determine_delimiter(file_path, file_encoding)

    lines_and_cells_iterable = tasks.get_csv_or_txt_split_readlines(
        file_path,
        file_encoding,
        delimiter,
    )

    return data_structures.SearchableSheet(
        nzgd_id=nzgd_id,
        sheet=lines_and_cells_iterable,
        file_path=file_path,
    )


def get_xls_xlsx_iterable(
    nzgd_id: int,
    file_path: Path,
) -> list[data_structures.SearchableSheet]:
    """Get an easily searchable representation of an XLS or XLSX file.

    Parameters
    ----------
    nzgd_id : int
        The NZGD ID number for which to get searchable sheets.
    file_path : Path
        The path to the XLS or XLSX file.

    Returns
    -------
    list[data_structures.SearchableSheet]
        A list of searchable sheets.

    """
    iterable_per_sheet = []

    try:
        sheet_names, found_engine = tasks.get_xls_sheet_names(file_path)
    except errors.InvalidExcelFileError:
        return iterable_per_sheet

    for sheet_name in sheet_names:
        try:
            sheet_as_df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=None,
                parse_dates=False,
                engine=found_engine,
            )
            iterable_per_sheet.append(
                data_structures.SearchableSheet(
                    nzgd_id=nzgd_id,
                    sheet=sheet_as_df.to_numpy().tolist(),
                    file_path=file_path,
                    sheet_name=sheet_name,
                ),
            )
        except errors.InvalidExcelFileError:
            continue

    return iterable_per_sheet


def get_searchable_sheets(nzgd_id: int) -> list[data_structures.SearchableSheet]:
    """Load spreadsheet data for straightforward searching.

    Parameters
    ----------
    nzgd_id : int
        The NZGD ID number for which to get searchable sheets.

    Returns
    -------
    list[data_structures.SearchableSheet]
        A list of searchable sheets.

    """
    if nzgd_id in constants.CPT_IDS:
        record_ffp = constants.NZGD_SOURCE_DATA_DIR / "cpt" / f"CPT_{nzgd_id}"
    if nzgd_id in constants.SCPT_IDS:
        record_ffp = constants.NZGD_SOURCE_DATA_DIR / "scpt" / f"SCPT_{nzgd_id}"

    available_files_for_record = list(record_ffp.glob("*.*"))

    csv_txt_iterables = []
    xls_xlsx_iterables = []

    csv_txt_files = [
        file
        for file in available_files_for_record
        if file.suffix.lower() in [".csv", ".txt"]
    ]

    xls_xlsx_files = [
        file
        for file in available_files_for_record
        if file.suffix.lower() in [".xls", ".xlsx"]
    ]

    for file in csv_txt_files:
        csv_txt_iterables.append(get_csv_txt_iterable(nzgd_id, file))

    for file in xls_xlsx_files:
        sheets_as_iterables = get_xls_xlsx_iterable(nzgd_id, file)

        for sheet_iterable in sheets_as_iterables:
            xls_xlsx_iterables.append(sheet_iterable)

    return csv_txt_iterables + xls_xlsx_iterables


def skip_leading_nans_in_cells_to_check(
    cells_to_check: list[str | float],
    max_num_blank_spaces_to_try_skipping: int,
) -> str | float:
    """Skip leading NaN values in a list of cells to check.

    Parameters
    ----------
    cells_to_check: list[str | float]
        The cells to check.

    max_num_blank_spaces_to_try_skipping: int
        The maximum number of empty cells to skip over.

    Returns
    -------
    str | float
        The first non-NaN value in the list or np.nan if all values are NaN.

    """
    allowed_blank_iterations = list(
        range(max_num_blank_spaces_to_try_skipping),
    )

    for blank_iteration in allowed_blank_iterations:
        if blank_iteration < len(cells_to_check):
            if isinstance(
                cells_to_check[blank_iteration],
                float,
            ) and np.isnan(
                cells_to_check[blank_iteration],
            ):
                continue
            return cells_to_check[blank_iteration]
    return np.nan


def is_horiz_or_vert(
    searchable_sheet: data_structures.SearchableSheet,
) -> data_structures.DataOrientation:
    """Check if the cell is part of a horizontal or vertical line."""
    # Convert list[list[str]] to DataFrame with consistent column count
    # Find the maximum number of columns
    max_cols = (
        max(len(row) for row in searchable_sheet.sheet) if searchable_sheet.sheet else 0
    )

    # Pad shorter rows with NaN values
    padded_data = []
    for row in searchable_sheet.sheet:
        padded_row = row + [np.nan] * (max_cols - len(row))
        padded_data.append(padded_row)

    as_df = pd.DataFrame(padded_data)

    num_str_and_float = info.count_num_str_float(as_df)

    num_str_per_row = (
        num_str_and_float.num_numerical_values_per_row
        - num_str_and_float.numerical_surplus_per_row
    )

    num_str_per_col = (
        num_str_and_float.num_numerical_values_per_col
        - num_str_and_float.numerical_surplus_per_col
    )

    if np.max(num_str_per_col) > np.max(num_str_per_row):
        return data_structures.DataOrientation.COLUMNS
    return data_structures.DataOrientation.ROWS


def find_header_line_index(
    searchable_sheet: data_structures.SearchableSheet,
) -> int:
    """Find the index of the header line in the sheet.

    Parameters
    ----------
    searchable_sheet : data_structures.SearchableSheet
        The sheet to search.

    Returns
    -------
    int
        The index of the header line, or the length of the sheet if not found.

    """
    for line_index, line in enumerate(searchable_sheet.sheet):
        initial_header_search = search.search_line_for_all_needed_cells(line)

        validated_search_results = []
        for search_results_for_col_index, search_results_for_col in enumerate(
            initial_header_search,
        ):
            validated_search_results_for_col = search.remove_col_name_false_positives(
                search_results_for_col,
                search_results_for_col_index,
            )
            validated_search_results.append(validated_search_results_for_col)
        # This code will ignore all lines below the returned header_line_index
        # as these are usually just the CPT data. However, if it is < 2, then its not
        # possible for the ground water level header and value to be above the main
        # header row, so we need to ensure that potential ground water level values
        # are preserved
        if all(len(x) > 0 for x in validated_search_results) and (line_index > 1):
            return line_index
    # If no header line is found, return the length of the sheet so that all lines
    # can be selected with the notation `searchable_sheet.sheet[:header_line_index]`
    # (up to but not including the index).
    return len(searchable_sheet.sheet)


def get_check_cell(
    searchable_sheet: data_structures.SearchableSheet,
    line_index: int,
    col_index: int,
    orientation: data_structures.DataOrientation,
    direction_to_cell: data_structures.CellToCheck,
):
    """Get the cell to check based on the current reference cell and data orientation.

    This function retrieves a neighboring cell relative to a reference cell position,
    taking into account the data orientation (COLUMNS or ROWS) and the desired
    direction to check. It handles edge cases where the requested cell is out of
    bounds and can skip over NaN values to find the nearest non-empty cell.

    Parameters
    ----------
    searchable_sheet : data_structures.SearchableSheet
        The searchable sheet containing the data grid to examine.
    line_index : int
        The row index of the reference cell (0-based).
    col_index : int
        The column index of the reference cell (0-based).
    orientation : data_structures.DataOrientation
        The data orientation, either COLUMNS or ROWS, which determines how
        directional searches are interpreted.
    direction_to_cell : data_structures.CellToCheck
        The direction to search for the cell relative to the reference position.
        Valid values are LEFT, RIGHT, ABOVE, or BELOW.

    Returns
    -------
    str
        The string representation of the found cell value. Returns "nan" if the
        cell to check is out of bounds bounds or if no non-NaN
        value is found within the allowed skip distance.

    Notes
    -----
    The function respects the MAX_NUM_BLANK_SPACES_TO_TRY_SKIPPING constant
    to determine how many NaN/empty cells to skip when searching for a valid
    cell value. The behavior varies based on orientation:

    - For COLUMNS orientation: LEFT and RIGHT directions are used
    - For ROWS orientation: ABOVE and BELOW directions are used

    """
    line = searchable_sheet.sheet[line_index]

    if orientation == data_structures.DataOrientation.COLUMNS:
        if direction_to_cell == data_structures.CellToCheck.LEFT:
            # check the cell to the left
            if col_index == 0:
                cell_to_check = np.nan
            else:
                cell_to_check = skip_leading_nans_in_cells_to_check(
                    line[0:col_index],
                    constants.MAX_NUM_BLANK_SPACES_TO_TRY_SKIPPING,
                )

        if direction_to_cell == data_structures.CellToCheck.RIGHT:
            if col_index == len(line) - 1:
                cell_to_check = np.nan
            else:
                # Check the cell to the right
                cell_to_check = skip_leading_nans_in_cells_to_check(
                    line[col_index + 1 :],
                    constants.MAX_NUM_BLANK_SPACES_TO_TRY_SKIPPING,
                )

    if orientation == data_structures.DataOrientation.ROWS:
        if direction_to_cell == data_structures.CellToCheck.ABOVE:
            # check the cell above
            if line_index == 0:
                cell_to_check = np.nan
            else:
                temp_column = [
                    searchable_sheet.sheet[i][col_index]
                    if len(searchable_sheet.sheet[i]) > col_index
                    else np.nan
                    for i in range(line_index)
                ]

                cell_to_check = skip_leading_nans_in_cells_to_check(
                    temp_column[
                        ::-1
                    ],  # reversing list to get the first non-NaN value above
                    constants.MAX_NUM_BLANK_SPACES_TO_TRY_SKIPPING,
                )

        if direction_to_cell == data_structures.CellToCheck.BELOW:
            # Check the cell below
            if line_index == len(searchable_sheet.sheet) - 1:
                cell_to_check = np.nan

            else:
                temp_col = [
                    searchable_sheet.sheet[i][col_index]
                    if len(searchable_sheet.sheet[i]) > col_index
                    else np.nan
                    for i in range(
                        line_index + 1,
                        len(searchable_sheet.sheet),
                    )
                ]

                cell_to_check = skip_leading_nans_in_cells_to_check(
                    temp_col,
                    constants.MAX_NUM_BLANK_SPACES_TO_TRY_SKIPPING,
                )
    return str(cell_to_check)


def sub_search(
    searchable_sheet: data_structures.SearchableSheet,
    search_term: str,
) -> data_structures.SearchInfoAndResult | None:
    """Find key information.

    Parameters
    ----------
    searchable_sheet : data_structures.SearchableSheet
        The sheet to search.
    search_term : str
        The term to search for.

    Returns
    -------
    data_structures.SearchInfoAndResult
        The search information and results

    """
    # Check if search term exists in any cell using regex word boundary matching
    # Regex pattern explanation:
    # (?:^|\s) - Non-capturing group: match start of string (^) OR whitespace (\s)
    # {re.escape(search_term.lower())} - The escaped search term (handles special regex chars)
    # (?:\s|$|:|=) - Non-capturing group: match whitespace (\s) OR end of string ($)
    #                OR colon (:) OR equals (=)
    # This ensures the search term is a complete word, not part of another word

    if not any(
        re.search(
            rf"(?:^|\s){re.escape(search_term.lower())}(?:\s|$|:|=|\()",
            str(cell).lower(),
        )
        for line in searchable_sheet.sheet
        for cell in line
    ):
        return None

    # Identify the header row indices so the data rows can be excluded, as the data
    # sometimes contains columns of string values that make it difficult to tell if
    # the metadata is arranged in rows or columns.
    # We exclude all rows below the header, which assumes that the required metadata
    # is always above the data rows. This assumption may not always be valid.

    header_line_index = find_header_line_index(searchable_sheet)

    searchable_sheet.sheet = searchable_sheet.sheet[
        :header_line_index
    ]  # select up to but not including the index

    likely_orientation = is_horiz_or_vert(searchable_sheet)

    # Collect the metadata and search result in one object for later use
    search_info_and_results = data_structures.SearchInfoAndResult(
        nzgd_id=searchable_sheet.nzgd_id,
        file_name=searchable_sheet.file_path.name,
        sheet_name=searchable_sheet.sheet_name,
        search_term=search_term,
        likely_orientation=likely_orientation,
        search_results=[],
    )

    for line_index, line in enumerate(searchable_sheet.sheet):
        for col_index, cell in enumerate(line):
            # Check if the found cell alone can be used to determine the reason
            # for termination.
            # Use regex to match search_term as whole word (bounded by spaces or string boundaries)
            cell_str = str(cell).lower()
            # Regex pattern breakdown:
            # (?:^|\s) - Non-capturing group: match start of string (^) OR whitespace (\s)
            # {re.escape(search_term.lower())} - The escaped search term (prevents regex injection)
            # (?:\s|$|:|=) - Non-capturing group: match whitespace (\s) OR end of string ($)
            #                OR colon (:) OR equals (=)
            # Examples: "factor" matches in "factor", " factor ", "factor test", "test factor"
            # But NOT in "factorial", "cofactor", "alpha_factor"
            search_pattern = rf"(?:^|\s){re.escape(search_term.lower())}(?:\s|$|:|=|\()"

            if re.search(search_pattern, cell_str):
                for assumption in constants.term_dict[search_term]:
                    # Just skip to the next assumption case if there are no search
                    # terms
                    if len(constants.term_dict[search_term][assumption]) == 0:
                        continue

                    ## a factor is checked last so the false positives of "Alpha factor" and "Beta factor"
                    # ("a factor" is in both) can be safely ignored by continuing the loop
                    if search_term == "a factor":
                        if "alpha factor" in str(cell).lower():
                            continue
                        if "beta factor" in str(cell).lower():
                            continue

                    pattern = rf"(?:{r'|'.join(constants.term_dict[search_term][assumption])})"

                    if assumption == "assuming_cell_is_standalone":
                        match = re.search(
                            pattern,
                            str(cell),
                            flags=re.IGNORECASE,
                        )
                        if match:
                            search_info_and_results.search_results.append(
                                data_structures.SearchResult(
                                    search_assumption=assumption,
                                    assumed_orientation=None,
                                    field_label_str=None,
                                    value_str=str(cell),
                                ),
                            )

                            continue

                    for assumed_orientation in [
                        data_structures.DataOrientation.COLUMNS,
                        data_structures.DataOrientation.ROWS,
                    ]:
                        if (
                            assumption
                            == "assuming_cell_is_a_value_in_need_of_field_name_to_confirm"
                        ):
                            if (
                                assumed_orientation
                                == data_structures.DataOrientation.COLUMNS
                            ):
                                cell_to_check = get_check_cell(
                                    searchable_sheet,
                                    line_index,
                                    col_index,
                                    assumed_orientation,
                                    data_structures.CellToCheck.LEFT,
                                )
                            else:
                                cell_to_check = get_check_cell(
                                    searchable_sheet,
                                    line_index,
                                    col_index,
                                    assumed_orientation,
                                    data_structures.CellToCheck.ABOVE,
                                )

                            match = re.search(
                                pattern,
                                str(cell_to_check),
                                flags=re.IGNORECASE,
                            )
                            if match:
                                search_info_and_results.search_results.append(
                                    data_structures.SearchResult(
                                        search_assumption=assumption,
                                        assumed_orientation=assumed_orientation,
                                        field_label_str=cell_to_check,
                                        value_str=str(cell),
                                    ),
                                )

                        if (
                            assumption
                            == "assuming_cell_is_a_field_name_in_need_of_a_value"
                        ):
                            if (
                                assumed_orientation
                                == data_structures.DataOrientation.COLUMNS
                            ):
                                cell_to_check = get_check_cell(
                                    searchable_sheet,
                                    line_index,
                                    col_index,
                                    assumed_orientation,
                                    data_structures.CellToCheck.RIGHT,
                                )
                            else:
                                cell_to_check = get_check_cell(
                                    searchable_sheet,
                                    line_index,
                                    col_index,
                                    assumed_orientation,
                                    data_structures.CellToCheck.BELOW,
                                )

                            match = re.search(
                                pattern,
                                str(cell_to_check),
                                flags=re.IGNORECASE,
                            )

                            if match:
                                search_info_and_results.search_results.append(
                                    data_structures.SearchResult(
                                        search_assumption=assumption,
                                        assumed_orientation=assumed_orientation,
                                        field_label_str=str(cell),
                                        value_str=str(cell_to_check),
                                    ),
                                )

    return search_info_and_results


def do_search(
    nzgd_id: int,
) -> list[data_structures.SearchInfoAndResult]:
    """Search CPT or SCPT record files for reasons for termination.

    Parameters
    ----------
    nzgd_id : int
        The NZGD ID for which to search for termination reasons.

    Returns
    -------
    dict[tuple[str, str], TerminationReasonSearchResult]
        A dictionary mapping (file_name, sheet_name) tuples to their corresponding
        termination reasons.

    """
    searchable_sheets = get_searchable_sheets(nzgd_id)

    search_results = []

    for search_term in constants.term_dict:
        for searchable_sheet in searchable_sheets:
            search_results.append(
                sub_search(searchable_sheet, search_term),
            )

    return search_results


if __name__ == "__main__":
    # investigation_type = "scpt"
    investigation_type = "cpt"

    ground_water_level_strings = []

    cpt_list = natsorted(
        list(
            Path(
                f"/home/arr65/data/nzgd/downloads_and_metadata/nzgd_investigation_source_files_from_webcrawler_11_Sep_2025/{investigation_type}",
            ).glob("*"),
        ),
    )

    output_dir = Path(
        "/home/arr65/data/nzgd/extracted_single_values_V3/all_possible_values",
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    nzgd_id_list = natsorted([int(cpt_id.name.split("_")[1]) for cpt_id in cpt_list])

    results_with_none = []
    num_workers = 7
    with mp.Pool(processes=num_workers) as pool:
        results_with_none.extend(
            list(
                tqdm(
                    pool.imap(do_search, nzgd_id_list),
                    total=len(nzgd_id_list),
                ),
            ),
        )

    results = []
    for result in results_with_none:
        subresults_without_none = []
        for subresult in result:
            if subresult is not None:
                subresults_without_none.append(subresult)
        results.append(subresults_without_none)

    results_df = pd.DataFrame()

    for record_result in results:
        for search_term_result in record_result:
            for assumed_orientation_result in search_term_result.search_results:
                results_df = pd.concat(
                    [
                        results_df,
                        pd.DataFrame(
                            {
                                "nzgd_id": search_term_result.nzgd_id,
                                "file_name": search_term_result.file_name,
                                "sheet_name": search_term_result.sheet_name,
                                "likely_orientation": search_term_result.likely_orientation,
                                "search_term": search_term_result.search_term,
                                "search_assumption": assumed_orientation_result.search_assumption,
                                "assumed_orientation": assumed_orientation_result.assumed_orientation,
                                "field_label": assumed_orientation_result.field_label_str,
                                "value": assumed_orientation_result.value_str,
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

    results_df.to_csv(
        output_dir / f"{investigation_type}_v10.csv",
        index=False,
    )
