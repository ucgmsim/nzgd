import numpy as np
import numpy.typing as npt
import pandas as pd

from nzgd import constants
from nzgd.extract.cpt import data_structures, errors, tasks


def find_cell_with_exact_match_in_line(
    line: list[str] | pd.Series | pd.Index,
    search_character: str,
) -> data_structures.SearchForColResults | None:
    """Find the first index in a delimited line that exactly matches a given character.

    This function iterates through a list of cells and returns the index of the first
    cell that exactly matches the specified character, ignoring case.

    Parameters
    ----------
    line : list[str] | pd.Series | pd.Index
        The line of cells to search through.
    search_character : str
        The character to match exactly.

    Returns
    -------
    int
        The first index in the delimited line that exactly matches a given character,
        or None if no match is found.

    """
    for i, cell_content in enumerate(line):
        if isinstance(cell_content, str) and cell_content.lower() == search_character:
            return data_structures.SearchForColResults(
                i,
                search_character,
                cell_content,
            )

    return None


def find_cell_in_line_that_contains_string(
    line: list[str] | pd.Series | pd.Index,
    search_string: str,
) -> data_structures.SearchForColResults | None:
    """Find the first index in a delimited line that contains the search search_string.

    This function iterates through a list of cells and returns the index of the first
    cell that contains the search search_string, ignoring case.

    Parameters
    ----------
    line : list[str] | pd.Series | pd.Index
        The line of cells to search through.
    search_string : str
        The search_string to search for within the cells.

    Returns
    -------
    int
        the first index in a delimited line that that contains the search string,
        or None if no match is found.

    """
    for i, cell_content in enumerate(line):
        if isinstance(cell_content, str) and search_string in cell_content.lower():
            return data_structures.SearchForColResults(i, search_string, cell_content)
    return None


def find_cells_in_line_that_contains_string(
    line: list[str] | pd.Series | pd.Index,
    search_string: str,
) -> list[data_structures.SearchForColResults]:
    """Find the indices in a delimited line that contain the search search_string.

    This function iterates through a list of cells and returns the indices of all cells
    that contain the specified search_string, ignoring case.

    Parameters
    ----------
    line : list[str] | pd.Series | pd.Index
        The line of cells to search through.
    search_string : str
        The search_string to search for within the cells.

    Returns
    -------
    list[data_structures.SearchForColResults]
        A list of indices of cells that contain the given search_string.

    """
    searches_results_to_return = []
    for i, cell_content in enumerate(line):
        if isinstance(cell_content, str) and search_string in cell_content.lower():
            searches_results_to_return.append(
                data_structures.SearchForColResults(i, search_string, cell_content),
            )
    return searches_results_to_return


def find_all_matching_cells_in_line(
    line: list[str | float] | pd.Series | pd.Index,
    characters: tuple[str],
    substrings: tuple[str],
) -> list[data_structures.SearchForColResults]:
    """Search for cells in a line that match given characters or substrings.

    This function searches through the cells in a line and finds the indices of cells
    that either exactly match any of the given characters or contain any of the given
    substrings.

    Parameters
    ----------
    line : list[str] | pd.Series | pd.Index
        The line of cells to search through.
    characters : list[str]
        A list of characters to match exactly against cell contents.
    substrings : list[str]
        A list of substrings to search for within cell contents.

    Returns
    -------
    list[data_structures.SearchForColResults]
        A list of search results containing information about matching cells,
        sorted by column index position.

    """
    # Get exact matches
    char_matches = [
        find_cell_with_exact_match_in_line(line, char) for char in characters
    ]

    # Get substring matches
    substring_matches = [
        match
        for substring in substrings
        for match in find_cells_in_line_that_contains_string(line, substring)
    ]

    # Combine, filter None values, and sort by position
    return sorted(
        {match for match in char_matches + substring_matches if match is not None},
        key=lambda x: x.col_index_in_line,
    )


def remove_repeated_finds_of_same_column_per_param(
    search_results: list[data_structures.SearchForColResults],
) -> list[data_structures.SearchForColResults]:
    """Remove repeated finds of the same column from search results.

    This function takes a list of SearchForColResults and removes any repeated finds
    of the same column, keeping only the first occurrence. For example, often the
    "depth" search term and the "h " search term will both match the same
    "Depth" column, so this function will only keep one.

    Parameters
    ----------
    search_results : list[data_structures.SearchForColResults]
        The list of search results to filter.

    Returns
    -------
    list[data_structures.SearchForColResults]
        A filtered list with repeated finds removed.

    """
    seen_indices = set()
    filtered_results = []
    for result in search_results:
        if result.col_index_in_line not in seen_indices:
            seen_indices.add(result.col_index_in_line)
            filtered_results.append(result)
    return filtered_results


def remove_repeated_column_finds_across_all_params(
    search_results: list[list[data_structures.SearchForColResults]]
    | data_structures.AllCPTColsSearchResults,
) -> data_structures.AllCPTColsSearchResults:
    """Remove repeated finds of the same column across all parameters.

    This function takes a list of SearchForColResults and removes any repeated finds
    of the same column across all parameters, keeping only the first occurrence.

    Parameters
    ----------
    search_results : list[data_structures.SearchForColResults]
        The list of search results to filter.

    Returns
    -------
    list[data_structures.SearchForColResults]
        A filtered list with repeated finds removed.

    """
    identified_column_names = set()
    unique_col_names = []
    for col_info_for_param in search_results:
        unique_col_names_for_param = []
        for col_info in col_info_for_param:
            if col_info.matched_string not in identified_column_names:
                unique_col_names_for_param.append(col_info)
                identified_column_names.add(col_info.matched_string)
        unique_col_names.append(unique_col_names_for_param)

    return data_structures.AllCPTColsSearchResults(
        col1_search_result=unique_col_names[0],
        col2_search_result=unique_col_names[1],
        col3_search_result=unique_col_names[2],
        col4_search_result=unique_col_names[3],
    )


def remove_col_name_false_positives(
    search_results_for_col: list[data_structures.SearchForColResults],
    column_index: int,
) -> list[data_structures.SearchForColResults]:
    """Remove false positives from column name search results.

    This function takes a list of SearchForColResults and removes any results that
    match with known false positives in `constants.header_search_negating_terms`.

    Parameters
    ----------
    search_results_for_col : list[data_structures.SearchForColResults]
        The list of search results for the specific column.
    column_index : int
        The index of the column being processed.
         - depth is 0, qc is 1, fs is 2, and u is 3.

    Returns
    -------
    list[data_structures.SearchForColResults]
        A filtered list with false positives removed.

    """
    header_search_negating_terms = constants.header_search_false_positives[column_index]

    validated_search_results_for_col = []
    any_negating_term = False
    for term_search_result in search_results_for_col:
        for negating_term in header_search_negating_terms:
            if negating_term in term_search_result.matched_string.lower():
                any_negating_term = True
        if not any_negating_term:
            validated_search_results_for_col.append(term_search_result)

    return validated_search_results_for_col


def search_line_for_all_needed_cells(
    line: list[str | float] | pd.Series | pd.Index,
) -> data_structures.AllCPTColsSearchResults:
    """Search an iterable line for all needed CPT column headers.

    This function searches through the cells in a line and identifies cells that
    match patterns for the four required CPT columns: depth, qc (cone resistance),
    fs (sleeve friction), and u (pore pressure). It returns detailed information
    about all matching cells for each column type.

    Parameters
    ----------
    line : line: list[str] | pd.Series | pd.Index
        The line of cells to search through.
    file_path : Path, optional
        The file path of the original data file, used for error reporting.
    sheet_name : str, optional
        The name of the sheet being processed, used for error reporting.

    Returns
    -------
    data_structures.AllCPTColsSearchResults
        A named tuple containing search results for each of the four CPT columns.
        Each column result is a list of SearchForColResults objects, where each
        SearchForColResults contains the column index, search term used, and the
        actual matched string from the line.

    Raises
    ------
    errors.FileProcessingError
        If any of the four required columns (depth, qc, fs, u) cannot be found
        in the line.

    Notes
    -----
    The function searches for exact character matches and substring matches based
    on patterns defined in constants.SEARCH_PATTERNS_CONFIG for each column type.
    All matching cells are returned, not just the first match, allowing the caller
    to choose the most appropriate column if multiple candidates exist.

    """
    characters1 = tuple(constants.SEARCH_PATTERNS_CONFIG["depth"]["characters"])
    substrings1 = tuple(constants.SEARCH_PATTERNS_CONFIG["depth"]["substrings"])

    characters2 = tuple(constants.SEARCH_PATTERNS_CONFIG["qc"]["characters"])
    substrings2 = tuple(constants.SEARCH_PATTERNS_CONFIG["qc"]["substrings"])

    characters3 = tuple(constants.SEARCH_PATTERNS_CONFIG["fs"]["characters"])
    substrings3 = tuple(constants.SEARCH_PATTERNS_CONFIG["fs"]["substrings"])

    characters4 = tuple(constants.SEARCH_PATTERNS_CONFIG["u"]["characters"])
    substrings4 = tuple(constants.SEARCH_PATTERNS_CONFIG["u"]["substrings"])

    col1_search_results = remove_col_name_false_positives(
        remove_repeated_finds_of_same_column_per_param(
            find_all_matching_cells_in_line(
                line,
                characters1,
                substrings1,
            ),
        ),
        0,
    )

    col2_search_results = remove_col_name_false_positives(
        remove_repeated_finds_of_same_column_per_param(
            find_all_matching_cells_in_line(
                line,
                characters2,
                substrings2,
            ),
        ),
        1,
    )
    col3_search_results = remove_col_name_false_positives(
        remove_repeated_finds_of_same_column_per_param(
            find_all_matching_cells_in_line(
                line,
                characters3,
                substrings3,
            ),
        ),
        2,
    )
    col4_search_results = remove_col_name_false_positives(
        remove_repeated_finds_of_same_column_per_param(
            find_all_matching_cells_in_line(
                line,
                characters4,
                substrings4,
            ),
        ),
        3,
    )

    return data_structures.AllCPTColsSearchResults(
        col1_search_result=col1_search_results,
        col2_search_result=col2_search_results,
        col3_search_result=col3_search_results,
        col4_search_result=col4_search_results,
    )


def find_one_header_row_index_from_column_names(
    lines_and_cells_iterable: list[str] | pd.Series | pd.Index,
) -> int:
    """Find the one header row index from column names in lines_and_cells_iterable.

    This function searches for the line (row index) that most likely contains the
    column names.

    Parameters
    ----------
    lines_and_cells_iterable : list[str] | pd.Series | pd.Index
        The input data as a list of lists or a DataFrame.

    Returns
    -------
    int
        The index of the header row if found, otherwise -1.

    """
    # make an array of row indices to check and roll the array such the first searched
    # row is the one with the highest values of text cells

    if isinstance(lines_and_cells_iterable, pd.DataFrame):
        lines_and_cells_iterable = tasks.convert_dataframe_to_list_of_lists(
            lines_and_cells_iterable,
        )

    num_text_cells_per_line = get_number_of_numeric_or_text_cells_per_line(
        lines_and_cells_iterable,
        constants.NumOrText.TEXT,
    )
    num_numerical_cells_per_line = get_number_of_numeric_or_text_cells_per_line(
        lines_and_cells_iterable,
        constants.NumOrText.NUMERIC,
    )

    text_surplus_per_line = num_text_cells_per_line - num_numerical_cells_per_line

    # roll the array such that the first row to be checked is the one with the highest
    # number of text cells as it is most likely to contain the column names. This will
    # reduce the chance of accidentally choosing the wrong row because it coincidentally
    # contained the keywords
    check_rows = np.roll(
        np.arange(0, len(lines_and_cells_iterable) - 1),
        -np.argmax(text_surplus_per_line),
    )
    best_partial_header_row_idx = -1
    num_headers_in_most_complete_row = -1

    for check_row in check_rows:
        line_check = search_line_for_all_needed_cells(
            lines_and_cells_iterable[check_row],
        )

        # Type narrowing: line_check is always a tuple in this context
        if isinstance(line_check, tuple):
            num_found_header_cols = np.sum([len(x) > 0 for x in line_check])
        else:
            # This branch should not occur based on the function implementation
            error_message = "Unexpected type for line_check, expected tuple."
            raise errors.FileProcessingError(error_message)
        # If at least one possible column name was found for each column,
        # immediately return the row
        if num_found_header_cols == constants.REQUIRED_NUMBER_OF_COLUMNS:
            return check_row

        # If more than one possible column name was found for the current row,
        # update the best row
        if (num_found_header_cols > 0) and (
            num_found_header_cols > num_headers_in_most_complete_row
        ):
            best_partial_header_row_idx = check_row
            num_headers_in_most_complete_row = num_found_header_cols

    # Return the most complete header row that was found.
    # If not even a partial header row was found, returns -1.
    return best_partial_header_row_idx


def get_number_of_numeric_or_text_cells_per_line(
    iterable: list[str] | pd.Series | pd.Index,
    numeric_or_text: constants.NumOrText,
) -> npt.NDArray[np.int_]:
    """Get the number of numeric or text cells per line in an iterable.

    This function iterates through each line in the provided iterable and counts the
    number of cells that are either numeric or text, based on the specified type.

    Parameters
    ----------
    iterable : list[str] | pd.Series | pd.Index
        The input data as a list of lists or a DataFrame.
    numeric_or_text : constants.NumOrText
        The type of cells to count (numeric or text).

    Returns
    -------
    npt.NDArray[np.int_]
        An array containing the count of numeric or text cells for each line.

    """
    if isinstance(iterable, pd.DataFrame):
        iterable = [iterable.iloc[i].to_list() for i in range(len(iterable))]
    iterable = tasks.convert_numerical_str_cells_to_float(iterable)

    num_x_cells_per_line = np.zeros(len(iterable), dtype=int)

    for row_idx, line in enumerate(iterable):
        filtered_line = [cell for cell in line if "nan" not in str(cell).lower()]

        if numeric_or_text == constants.NumOrText.TEXT:
            num_x_cells_per_line[row_idx] = np.sum(
                [isinstance(cell, str) for cell in filtered_line],
            )
        elif numeric_or_text == constants.NumOrText.NUMERIC:
            num_x_cells_per_line[row_idx] = np.sum(
                [isinstance(cell, int | float) for cell in filtered_line],
            )

    return num_x_cells_per_line


def find_row_indices_of_header_lines(
    lines_and_cells_iterable: pd.DataFrame | list,
) -> list:
    """Find the row indices of the header lines.

    Parameters
    ----------
    lines_and_cells_iterable : Union[pd.DataFrame, list]
        The input data as a list of lists containing strings or a DataFrame.

    Returns
    -------
    np.ndarray
        An array containing the row index of each header row.

    """
    if isinstance(lines_and_cells_iterable, pd.DataFrame):
        lines_and_cells_iterable = tasks.convert_dataframe_to_list_of_lists(
            lines_and_cells_iterable,
        )

    # Some CSV files such as CPT_135862_Raw01.csv have empty cells represented as empty
    # strings between commas such as "","","","","" which should be removed before
    # counting the number of text cells per line

    lines_and_cells_iterable_no_empty_cell_issues = []
    for line in lines_and_cells_iterable:
        line_no_empty_cells = [cell for cell in line if cell not in ['""', '""\n']]
        lines_and_cells_iterable_no_empty_cell_issues.append(line_no_empty_cells)

    lines_and_cells_iterable = lines_and_cells_iterable_no_empty_cell_issues

    num_text_cells_per_line = get_number_of_numeric_or_text_cells_per_line(
        lines_and_cells_iterable,
        constants.NumOrText.TEXT,
    )
    num_numeric_cells_per_line = get_number_of_numeric_or_text_cells_per_line(
        lines_and_cells_iterable,
        constants.NumOrText.NUMERIC,
    )
    text_surplus_per_line = num_text_cells_per_line - num_numeric_cells_per_line

    initial_header_row = find_one_header_row_index_from_column_names(
        lines_and_cells_iterable,
    )

    ## If no header row was found, there will be a single value of -1 in the
    # header_rows list so return an empty list which will later raise an exception
    if initial_header_row == -1:
        return []

    additional_header_rows = []

    ## Check for header rows above the one first identified
    current_row_idx = initial_header_row - 1
    while current_row_idx >= 0:
        # Most spreadsheets have general notes above the header rows so to robustly
        # identify additional header rows above the first header row found, the number
        # of text cells in the rows should be the same as in the first one found
        # This would not identify a header row above if it is missing one or more cells
        # but this is a reasonable trade-off to avoid identifying general notes as
        # header rows as none of those cases have been encountered in the data so far.
        # Additionally, it is unlikely that the first identified header row is not the
        # first header row in the data, anyway.

        if (
            num_text_cells_per_line[current_row_idx]
            != num_text_cells_per_line[initial_header_row]
        ):
            break

        additional_header_rows.append(current_row_idx)
        current_row_idx -= 1

    ## Check for header rows below the one first identified
    current_row_idx = initial_header_row + 1
    while current_row_idx < len(lines_and_cells_iterable) - 1:
        if text_surplus_per_line[current_row_idx] > 0:
            additional_header_rows.append(int(current_row_idx))
        else:
            break
        current_row_idx += 1

    # Add the initial header row to the list of header rows and sort.
    all_header_rows = [initial_header_row, *additional_header_rows]

    # TODO: Why crash if some header rows are repeated? Why not just remove duplicates?
    # check that all identified header row indices are unique
    if len(all_header_rows) != len(set(all_header_rows)):
        error_message = (
            "header_row_idx_repeated - some header rows were identified twice"
        )
        raise ValueError(error_message)

    return sorted(all_header_rows)


def find_all_candidate_header_rows(
    lines_and_cells_iterable: pd.DataFrame | list,
) -> list[int]:
    """Find all rows that could potentially be header rows.

    Unlike find_row_indices_of_header_lines which starts from one initial row
    and only extends to adjacent rows with matching text cell counts, this function
    returns ALL rows where text_surplus > 0 in the region near the data. This
    catches candidate header rows that have different text cell counts from each other,
    which is important for files with conflicting multi-row headers.

    Parameters
    ----------
    lines_and_cells_iterable : pd.DataFrame | list
        The input data as a DataFrame or list of lists.

    Returns
    -------
    list[int]
        Sorted list of row indices that could be header rows.

    """
    if isinstance(lines_and_cells_iterable, pd.DataFrame):
        lines_and_cells_iterable = tasks.convert_dataframe_to_list_of_lists(
            lines_and_cells_iterable,
        )

    # Remove empty string cells that can cause issues
    cleaned_lines = []
    for line in lines_and_cells_iterable:
        line_no_empty_cells = [cell for cell in line if cell not in ['""', '""\n']]
        cleaned_lines.append(line_no_empty_cells)

    num_text_cells_per_line = get_number_of_numeric_or_text_cells_per_line(
        cleaned_lines,
        constants.NumOrText.TEXT,
    )
    num_numeric_cells_per_line = get_number_of_numeric_or_text_cells_per_line(
        cleaned_lines,
        constants.NumOrText.NUMERIC,
    )
    text_surplus_per_line = num_text_cells_per_line - num_numeric_cells_per_line

    # Find the start of the main contiguous data block. We look for the first
    # row that starts a run of at least min_contiguous_data_rows consecutive
    # rows with numerical surplus > 0. This avoids being tricked by isolated
    # metadata rows that happen to contain a few numbers.
    min_contiguous_data_rows = constants.COLUMN_DATA_VALIDATION[
        "min_data_rows_for_scoring"
    ]
    first_data_row = len(cleaned_lines)
    for i in range(len(cleaned_lines)):
        # Check if there's a contiguous run of numerical rows starting at i
        is_contiguous = True
        for j in range(i, min(i + min_contiguous_data_rows, len(cleaned_lines))):
            if num_numeric_cells_per_line[j] < constants.MIN_NUMERICAL_SURPLUS_PER_ROW:
                is_contiguous = False
                break
        if is_contiguous and i + min_contiguous_data_rows <= len(cleaned_lines):
            first_data_row = i
            break

    # Collect rows with text surplus > 0 near the data block. We search
    # backwards from the first data row to find the contiguous block of
    # text-heavy rows immediately above the data (the actual header region).
    # We stop when we hit a row with no text surplus (likely the boundary
    # between metadata and headers) or a gap of non-text rows.
    candidate_rows = []
    max_gap = 2  # Allow small gaps (e.g., an empty row between header rows)
    gap_count = 0
    for i in range(first_data_row - 1, -1, -1):
        if text_surplus_per_line[i] > 0 and num_text_cells_per_line[i] >= 2:
            candidate_rows.append(i)
            gap_count = 0
        else:
            gap_count += 1
            if gap_count > max_gap:
                break

    return sorted(candidate_rows)


def _combine_header_rows_to_list(
    lines_and_cells_iterable: list[list],
    row_indices: list[int],
) -> list[str]:
    """Combine multiple header rows into a single list of concatenated strings.

    For each column position, concatenates the text from all specified rows
    with a space separator, skipping NaN/empty values.

    Parameters
    ----------
    lines_and_cells_iterable : list[list]
        The full data as a list of lists.
    row_indices : list[int]
        The indices of the rows to combine.

    Returns
    -------
    list[str]
        A list of combined header strings, one per column.

    """
    if len(row_indices) == 0:
        return []

    # Find the maximum number of columns across all header rows
    max_cols = max(len(lines_and_cells_iterable[r]) for r in row_indices)

    combined = []
    for col_idx in range(max_cols):
        parts = []
        for row_idx in row_indices:
            row = lines_and_cells_iterable[row_idx]
            if col_idx < len(row):
                val = row[col_idx]
                val_str = str(val).strip() if val is not None else ""
                if val_str.lower() not in ("nan", ""):
                    parts.append(val_str)
        combined.append(" ".join(parts) if parts else "")

    return combined


def score_column_assignment(
    data_rows: list[list],
    depth_col_idx: int,
    qc_col_idx: int,
    fs_col_idx: int,
    u_col_idx: int,
) -> float:
    """Score a proposed column assignment based on data magnitude characteristics.

    Uses physical properties of CPT measurements to determine how well the
    proposed column mapping matches expected data patterns:
    - Depth should be monotonically increasing and non-negative
    - u2 (pore pressure) should have the most negative values
    - fs (sleeve friction) should have few negative values
    - All columns should be distinct

    Parameters
    ----------
    data_rows : list[list]
        The numerical data rows (below the header), as a list of lists.
    depth_col_idx : int
        Proposed column index for depth.
    qc_col_idx : int
        Proposed column index for cone resistance (qc).
    fs_col_idx : int
        Proposed column index for sleeve friction (fs).
    u_col_idx : int
        Proposed column index for pore pressure (u).

    Returns
    -------
    float
        A score indicating how well the data matches the proposed assignment.
        Higher scores indicate better matches.

    """
    min_rows = constants.COLUMN_DATA_VALIDATION["min_data_rows_for_scoring"]
    mono_threshold = constants.COLUMN_DATA_VALIDATION["depth_monotonicity_threshold"]
    max_neg_fs = constants.COLUMN_DATA_VALIDATION["max_negative_fraction_for_fs"]

    # Column uniqueness check
    col_indices = [depth_col_idx, qc_col_idx, fs_col_idx, u_col_idx]
    if len(set(col_indices)) != 4:
        return -20.0

    # Extract numerical values for each proposed column
    def get_numeric_values(col_idx: int) -> list[float]:
        vals = []
        for row in data_rows:
            if col_idx < len(row):
                val = row[col_idx]
                if isinstance(val, int | float) and not np.isnan(val):
                    vals.append(float(val))
                elif isinstance(val, str):
                    try:
                        vals.append(float(val))
                    except (ValueError, TypeError):
                        pass
        return vals

    depth_vals = get_numeric_values(depth_col_idx)
    qc_vals = get_numeric_values(qc_col_idx)
    fs_vals = get_numeric_values(fs_col_idx)
    u_vals = get_numeric_values(u_col_idx)

    # Need sufficient data for scoring
    if any(len(v) < min_rows for v in [depth_vals, qc_vals, fs_vals, u_vals]):
        return 0.0

    score = 0.0

    # 1. Depth monotonicity (+10)
    if len(depth_vals) >= 2:
        diffs = [depth_vals[i + 1] - depth_vals[i] for i in range(len(depth_vals) - 1)]
        fraction_increasing = sum(1 for d in diffs if d > 0) / len(diffs)
        if fraction_increasing >= mono_threshold:
            score += 10.0

    # 2. Depth non-negative (+3)
    if all(v >= 0 for v in depth_vals):
        score += 3.0

    # 3. u2 should have the most negative minimum value (+5)
    min_depth = min(depth_vals)
    min_qc = min(qc_vals)
    min_fs = min(fs_vals)
    min_u = min(u_vals)
    if min_u <= min(min_depth, min_qc, min_fs):
        score += 5.0

    # 4. fs should have few negative values (+3)
    neg_fs_count = sum(1 for v in fs_vals if v < 0)
    if neg_fs_count / len(fs_vals) < max_neg_fs:
        score += 3.0

    # 5. All 4 columns have sufficient data (+3)
    if all(
        len(v) >= constants.MIN_NUM_DATA_POINTS_PER_COLUMN
        for v in [depth_vals, qc_vals, fs_vals, u_vals]
    ):
        score += 3.0

    return score


def find_best_header_combination(
    extracted_data_df: pd.DataFrame,
    candidate_header_rows: list[int],
) -> tuple[int, list[str], data_structures.AllCPTColsSearchResults] | None:
    """Try multiple combinations of candidate header rows and pick the best.

    For each combination, concatenates the header rows, searches for CPT column
    names, and scores the resulting column assignment using data magnitude
    validation.

    Parameters
    ----------
    extracted_data_df : pd.DataFrame
        The full DataFrame loaded with header=None.
    candidate_header_rows : list[int]
        List of row indices that could be header rows.

    Returns
    -------
    tuple[int, list[str], AllCPTColsSearchResults] | None
        A tuple of (header_row_index, combined_header_list, column_search_results),
        or None if no valid combination is found.

    """
    from itertools import combinations

    if len(candidate_header_rows) == 0:
        return None

    lines = tasks.convert_dataframe_to_list_of_lists(extracted_data_df)
    lines = tasks.convert_numerical_str_cells_to_float(lines)

    best_score = -float("inf")
    best_result = None

    # Generate all non-empty subsets up to size 3
    all_combos = []
    for size in range(1, min(4, len(candidate_header_rows) + 1)):
        all_combos.extend(combinations(candidate_header_rows, size))

    for combo in all_combos:
        combo_rows = sorted(combo)

        # Combine the header rows
        combined_header = _combine_header_rows_to_list(lines, combo_rows)

        if len(combined_header) == 0:
            continue

        # Search the combined header for column names
        col_search = search_line_for_all_needed_cells(combined_header)
        col_search = remove_repeated_column_finds_across_all_params(col_search)

        # Count how many of the 4 required columns were found
        found_params = [i for i, results in enumerate(col_search) if len(results) > 0]
        num_found = len(found_params)

        if num_found < 3:
            continue

        # The data starts after the lowest header row in this combination
        lowest_header_row = max(combo_rows)
        data_start = lowest_header_row + 1

        # Get data rows
        data_rows = []
        for i in range(data_start, len(lines)):
            data_rows.append(lines[i])

        if len(data_rows) < constants.COLUMN_DATA_VALIDATION["min_data_rows_for_scoring"]:
            continue

        # Get the first candidate column index for each found parameter
        col_indices = {}
        param_names = ["depth", "qc", "fs", "u"]
        for param_idx in found_params:
            col_indices[param_names[param_idx]] = (
                col_search[param_idx][0].col_index_in_line
            )

        # If a parameter is missing, try to infer from data
        missing_params = [p for p in param_names if p not in col_indices]
        assigned_col_indices = set(col_indices.values())

        for missing_param in missing_params:
            inferred_col = _infer_missing_column_from_data(
                data_rows,
                missing_param,
                assigned_col_indices,
                max_cols=len(combined_header),
            )
            if inferred_col is not None:
                col_indices[missing_param] = inferred_col
                assigned_col_indices.add(inferred_col)

                # Create a synthetic SearchForColResults for the inferred column
                inferred_header_text = (
                    combined_header[inferred_col]
                    if inferred_col < len(combined_header)
                    else ""
                )
                if not inferred_header_text:
                    inferred_header_text = f"inferred_{missing_param}"
                inferred_result = data_structures.SearchForColResults(
                    col_index_in_line=inferred_col,
                    search_term=f"data_inferred_{missing_param}",
                    matched_string=inferred_header_text,
                )
                # Add to the appropriate position in col_search
                param_idx = param_names.index(missing_param)
                col_search = data_structures.AllCPTColsSearchResults(
                    col1_search_result=(
                        col_search[0]
                        if param_idx != 0
                        else [inferred_result, *col_search[0]]
                    ),
                    col2_search_result=(
                        col_search[1]
                        if param_idx != 1
                        else [inferred_result, *col_search[1]]
                    ),
                    col3_search_result=(
                        col_search[2]
                        if param_idx != 2
                        else [inferred_result, *col_search[2]]
                    ),
                    col4_search_result=(
                        col_search[3]
                        if param_idx != 3
                        else [inferred_result, *col_search[3]]
                    ),
                )

        # Check that all 4 parameters are now assigned
        if len(col_indices) != 4:
            continue

        # Score this assignment
        current_score = score_column_assignment(
            data_rows,
            col_indices["depth"],
            col_indices["qc"],
            col_indices["fs"],
            col_indices["u"],
        )

        # On ties, prefer combinations with more header rows (more unit info)
        tiebreaker = len(combo_rows) * 0.1
        current_score += tiebreaker

        if current_score > best_score:
            best_score = current_score
            # Build clean col_info with only the winning column index per parameter.
            # This prevents ambiguous matches (e.g., depth at both col 0 and col 3)
            # from confusing downstream column selection.
            clean_col_search = _build_clean_col_info(
                col_search, col_indices, param_names, combined_header,
            )
            best_result = (lowest_header_row, combined_header, clean_col_search)

    return best_result


def _build_clean_col_info(
    col_search: data_structures.AllCPTColsSearchResults,
    col_indices: dict[str, int],
    param_names: list[str],
    combined_header: list[str],
) -> data_structures.AllCPTColsSearchResults:
    """Build clean AllCPTColsSearchResults with exactly one match per parameter.

    After scoring validates a specific column assignment, this function creates
    a clean col_info with only the winning column index for each parameter,
    removing spurious matches that could confuse downstream processing.

    """
    clean_results = []
    for param_idx, param_name in enumerate(param_names):
        winning_col_idx = col_indices[param_name]
        # Find the search result that matches the winning column index
        matching_result = None
        for r in col_search[param_idx]:
            if r.col_index_in_line == winning_col_idx:
                matching_result = r
                break
        if matching_result is None:
            # Create a synthetic result (e.g., for inferred columns)
            header_text = (
                combined_header[winning_col_idx]
                if winning_col_idx < len(combined_header)
                else f"inferred_{param_name}"
            )
            matching_result = data_structures.SearchForColResults(
                col_index_in_line=winning_col_idx,
                search_term=f"data_inferred_{param_name}",
                matched_string=header_text,
            )
        clean_results.append([matching_result])

    return data_structures.AllCPTColsSearchResults(
        col1_search_result=clean_results[0],
        col2_search_result=clean_results[1],
        col3_search_result=clean_results[2],
        col4_search_result=clean_results[3],
    )


def _infer_missing_column_from_data(
    data_rows: list[list],
    missing_param: str,
    assigned_col_indices: set[int],
    max_cols: int,
) -> int | None:
    """Infer a missing column from data characteristics.

    When a header row matches only 3 out of 4 required parameters, this function
    tries to identify the missing parameter from the data by looking at numerical
    characteristics of unassigned columns.

    Parameters
    ----------
    data_rows : list[list]
        The numerical data rows.
    missing_param : str
        The parameter name to infer ("depth", "qc", "fs", or "u").
    assigned_col_indices : set[int]
        Column indices already assigned to other parameters.
    max_cols : int
        Maximum number of columns in the data.

    Returns
    -------
    int | None
        The inferred column index, or None if inference failed.

    """
    mono_threshold = constants.COLUMN_DATA_VALIDATION["depth_monotonicity_threshold"]

    # Get unassigned column indices
    unassigned = [i for i in range(max_cols) if i not in assigned_col_indices]

    if len(unassigned) == 0:
        return None

    def get_numeric_values(col_idx: int) -> list[float]:
        vals = []
        for row in data_rows:
            if col_idx < len(row):
                val = row[col_idx]
                if isinstance(val, int | float) and not np.isnan(val):
                    vals.append(float(val))
                elif isinstance(val, str):
                    try:
                        vals.append(float(val))
                    except (ValueError, TypeError):
                        pass
        return vals

    if missing_param == "depth":
        # Look for monotonically increasing column
        best_col = None
        best_mono = -1.0
        for col_idx in unassigned:
            vals = get_numeric_values(col_idx)
            if len(vals) < 10:
                continue
            diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
            if len(diffs) == 0:
                continue
            frac_increasing = sum(1 for d in diffs if d > 0) / len(diffs)
            if frac_increasing > best_mono and frac_increasing >= mono_threshold:
                best_mono = frac_increasing
                best_col = col_idx
        return best_col

    if missing_param == "u":
        # Look for column with most negative values
        best_col = None
        most_negative_min = 0.0
        for col_idx in unassigned:
            vals = get_numeric_values(col_idx)
            if len(vals) < 10:
                continue
            col_min = min(vals)
            if col_min < most_negative_min:
                most_negative_min = col_min
                best_col = col_idx
        return best_col

    return None


def find_col_name_from_substring(
    df: pd.DataFrame,
    substrings: list[str],
    remaining_cols_to_search: list[str],
    target_column_name: str,
) -> tuple[str | None, pd.DataFrame, list[str]]:
    """Find a column name containing a substring in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing the loaded xls file.
    substrings : list[str]
        A list of substrings to search for in the column names.
    remaining_cols_to_search : list[str]
        A list of column names to search for the substring in.
    target_column_name : str
        The name of the column being searched for.

    Returns
    -------
    tuple[str | None, pd.DataFrame, list[str]]
        A tuple containing:
        - The name of the column containing the substring (None if no match found)
        - The updated DataFrame with metadata attributes and potential unit conversions
        - The updated list of columns to search (with matched column removed if found)

    """
    candidate_col_names = []
    for col_name in remaining_cols_to_search:
        if isinstance(col_name, str):
            for substring in substrings:
                if (
                    substring in col_name.lower()
                    and col_name not in candidate_col_names
                ):
                    candidate_col_names.append(col_name)

    # no relevant columns were found
    if len(candidate_col_names) == 0:
        col = None
        if "missing_columns" not in df.attrs:
            df.attrs["missing_columns"] = [target_column_name]
        else:
            df.attrs["missing_columns"].append(target_column_name)

    ## Check that there are some candidate column names
    else:
        col = candidate_col_names[0]

        # check for "Clean" which is sometimes used for a cleaned version of the
        # same data
        if len(candidate_col_names) > 1:
            for candidate_name in candidate_col_names:
                ## some "clean" columns are full of nans (no data) so also check that
                # the number of nans in the "clean" column is less than or equal to
                # the number of nans in the current column
                if ("clean" in candidate_name.lower()) and (
                    np.sum(pd.isna(df[candidate_name])) <= np.sum(pd.isna(df[col]))
                ):
                    col = candidate_name
                    break

        df.attrs[f"candidate_{target_column_name}_column_names_in_original_file"] = (
            candidate_col_names
        )
        df.attrs[f"adopted_{target_column_name}_column_name_in_original_file"] = col
        remaining_cols_to_search.remove(col)

        # if the column is in kPa, convert to MPa
        if (target_column_name != "Depth") and ("kpa" in col.lower()):
            df.loc[:, col] /= 1000
        # if the depth column is in cm, convert to m
        if (target_column_name == "Depth") and ("cm" in col):
            df.loc[:, col] /= 100

    return col, df, remaining_cols_to_search
