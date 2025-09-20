import numpy as np
import numpy.typing as npt
import pandas as pd

from nzgd_data_extraction import constants, data_structures, errors, tasks


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
