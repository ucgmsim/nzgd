import re
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import toml

from nzgd_data_extraction import constants, data_structures, search


def count_num_str_float(
    extracted_data_df: pd.DataFrame,
) -> data_structures.NumStrAndFloat:
    """Get info about the types of data in the DataFrame.

    Parameters
    ----------
    extracted_data_df : pd.DataFrame
        The DataFrame containing the data.

    Returns
    -------
    NumStrAndFloat
        An object containing information about the type of data in the DataFrame.


    """
    df_for_counting_num_of_str = extracted_data_df.map(
        lambda x: 1.0 if isinstance(x, str) else 0,
    )

    df_nan_to_str = extracted_data_df.infer_objects(copy=False).fillna("nan")
    df_for_counting_num_of_num = df_nan_to_str.map(
        lambda x: 1.0 if isinstance(x, int | float) else 0,
    )

    return data_structures.NumStrAndFloat(
        num_numerical_values_per_row=df_for_counting_num_of_num.sum(axis=1),
        num_numerical_values_per_col=df_for_counting_num_of_num.sum(axis=0),
        numerical_surplus_per_row=df_for_counting_num_of_num.sum(axis=1)
        - df_for_counting_num_of_str.sum(axis=1),
        numerical_surplus_per_col=df_for_counting_num_of_num.sum(axis=0)
        - df_for_counting_num_of_str.sum(axis=0),
        num_numerical_values=df_for_counting_num_of_num.sum().sum(),
    )


def nth_highest_value(array: npt.NDArray, n: int) -> float:
    """Find the nth highest value in an array.

    Parameters
    ----------
    array : npt.ArrayLike
        The input array.
    n : int
        The value of n to select the nth highest value.

    Returns
    -------
    float
        The nth highest value in the array.

    """
    ## Filter out any nan depth values (such as at the end of a file) and sort the array
    sorted_array = np.sort(array[np.isfinite(array)])

    ## indexing a numpy array returns a numpy scalar which is not the same as a Python
    # float so explicitly convert to satisfy type checking
    return float(sorted_array[-n])


def round_to_nearest_power_of_10(number: float) -> float:
    """Round a number to the nearest power of 10.

    Parameters
    ----------
    number : float
        The input number to round.

    Returns
    -------
    float
        The nearest power of 10. Returns 0 if the input is 0.

    """
    if number == 0:
        return 0
    power = round(np.log10(abs(number)))
    return 10**power


def get_depth_step_size(depth_array: npt.ArrayLike) -> float:
    """Calculate the median depth step, correcting for repeated consecutive values.

    Parameters
    ----------
    depth_array : npt.ArrayLike
        An array of depth values.

    Returns
    -------
    float
        The median step size in the depth values.

    """
    conditioned_depths = np.abs(
        depth_array,
    )  # Take the absolute value of the depth array
    conditioned_depths = conditioned_depths[
        np.isfinite(conditioned_depths)
    ]  # Remove non-finite values

    median_step_size = float(
        np.median(np.diff(conditioned_depths)),
    )

    if median_step_size == 0:
        # The depth array likely has many repeated consecutive values, so try this
        # function again, but only using unique depth values

        unique_depths = (
            pd.Series(np.abs(depth_array)).drop_duplicates(keep="first").to_numpy()
        )

        if len(unique_depths) < 2:
            return 1.0  # Not enough unique depth values to calculate a step size

        median_step_size = get_depth_step_size(unique_depths)

    return median_step_size


def str_cannot_become_float(value: str) -> bool:
    """Check if a string cannot be converted to a float.

    Parameters
    ----------
    value : str
        The string to check.

    Returns
    -------
    bool
        True if the string cannot be converted to a float, False otherwise.

    """
    try:
        float(value)
    except ValueError:
        return True
    else:
        return False


def get_column_names(loaded_data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Find names of the required columns in the DataFrame.

    Parameters
    ----------
    loaded_data_df : pd.DataFrame
        The DataFrame containing the data with potential column names.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        A tuple containing the DataFrame with updated attributes and a list of final
        column names.

    Raises
    ------
    ValueError
        If a column name is repeated or if a known false positive column name is used.

    """
    known_false_positive_col_names = toml.load(
        Path(__file__).parent.parent
        / "resources"
        / "known_false_positive_column_names.toml",
    )

    col_index_to_name = {0: "Depth", 1: "qc", 2: "fs", 3: "u"}

    new_col_search_results = search.search_line_for_all_needed_cells(
        loaded_data_df.columns.tolist(),
    )

    all_possible_col_indices = []
    for search_results_for_col in new_col_search_results:
        col_indices = []
        for search_result in search_results_for_col:
            if search_result.col_index_in_line is not None:
                col_indices.append(search_result.col_index_in_line)
        all_possible_col_indices.append(col_indices)

    final_col_names = []

    for col_number, possible_cols_info_per_col_type in enumerate(
        all_possible_col_indices,
    ):
        if len(possible_cols_info_per_col_type) == 0:
            final_col_names.append(None)

            if "missing_columns" not in loaded_data_df.attrs:
                loaded_data_df.attrs["missing_columns"] = [
                    col_index_to_name[col_number],
                ]
            else:
                loaded_data_df.attrs["missing_columns"].append(
                    col_index_to_name[col_number],
                )

        else:
            possible_col_names_with_info = [
                loaded_data_df.columns[int(idx)]
                for idx in possible_cols_info_per_col_type
            ]

            candidate_col_name = possible_col_names_with_info[0]

            ## Check that the selected column name is used only once
            if len(loaded_data_df[candidate_col_name].shape) > 1:
                error_message = (
                    "repeated_col_names_in_source - sheet has multiple"
                    f"columns with the name {candidate_col_name}"
                )
                raise ValueError(
                    error_message,
                )
            # For every possible column, get the number of finite values
            num_finite_per_col_list = []
            for col_name in possible_col_names_with_info:
                data_col = loaded_data_df[col_name]
                ## If the shape of data col is like (x, y), instead of just (x,)
                # there are multiple columns with the same name so raise an error

                error_message = (
                    "repeated_col_names_in_source - sheet has multiple"
                    f"columns with the name {col_name}"
                )

                if len(data_col.shape) > 1:
                    raise ValueError(
                        error_message,
                    )
                finite_data_col = np.isfinite(data_col)
                num_finite = np.sum(finite_data_col)
                num_finite_per_col_list.append(num_finite)

            num_finite_per_col = np.array(num_finite_per_col_list)

            ## Valid possible column names will have at least one finite value
            valid_possible_col_names = np.array(possible_col_names_with_info)[
                num_finite_per_col > 0
            ]

            ## Initially set the column name to the first valid column name
            if len(valid_possible_col_names) == 0:
                error_message = (
                    "no_valid_column_names - sheet has no valid column"
                    f"names for column {col_index_to_name[col_number]}"
                )
                raise ValueError(error_message)
            col_name = valid_possible_col_names[0]

            for possible_col_name in valid_possible_col_names:
                # If another valid column name does not include "clean" or "corrected"
                # then use that column name instead, as the "clean" or "corrected"
                # columns may have been processed such that the
                # correlations are no longer valid
                if ("clean" not in possible_col_name.lower()) & (
                    "corrected" not in possible_col_name.lower()
                ):
                    col_name = possible_col_name
                    break

            final_col_names.append(col_name)

            loaded_data_df.attrs[
                f"candidate_{col_index_to_name[col_number]}_column_names_in_original_file"
            ] = list(valid_possible_col_names)
            loaded_data_df.attrs[
                f"adopted_{col_index_to_name[col_number]}_column_name_in_original_file"
            ] = col_name

    ## Check if any of the identified column names are known false positives
    for col_name in final_col_names:
        if col_name in known_false_positive_col_names:
            error_message = (
                f"false_positive_column_name - Using a column named [{col_name}] "
                "which is a known false positive for column "
                f"[{known_false_positive_col_names[col_name]}]"
            )
            raise ValueError(error_message)

    return loaded_data_df, final_col_names


def find_encoding(file_path: Path) -> str | None:
    """Determine the encoding of a text file.

    This function attempts to open the file using several possible encodings.
    If the file can be opened without a UnicodeDecodeError, the encoding is returned.

    Parameters
    ----------
    file_path : Path
        The path to the text file whose encoding is to be determined.

    Returns
    -------
    str | None
        The encoding that successfully opens the file without errors or
        None if no encoding works.

    """
    for encoding in constants.ENCODINGS_TO_TRY:
        try:
            # open the text file
            with file_path.open(encoding=encoding) as file:
                # Check that the file can be read with this encoding
                _ = file.readlines()
            return encoding
        except UnicodeDecodeError:
            continue
    return None


def determine_delimiter(file_path: Path, file_encoding: str) -> str:
    r"""Determine the delimiter used in a text file.

    This function reads the content of a text file and counts the occurrences of
    each possible delimiter. It returns the delimiter that appears most frequently in
    the file.

    Parameters
    ----------
    file_path : Path
        The path to the text file.
    file_encoding : str
        The encoding needed to read the file.
    possible_delimiters : list[str], optional
        A list of possible delimiters to check for in the file (excluding white space r"\\s+"). Default is [r",", r";", r"\t"].

    Returns
    -------
    str
        The delimiter that appears most frequently in the file.

    Raises
    ------
    ValueError
        If no delimiter is found in the file.

    """
    possible_delimiters = [r",", r";", r"\t"]

    # Open the file and read its content using the specified encoding
    with Path.open(file_path, encoding=file_encoding) as file:
        content = file.read()

    # Count the occurrences of each possible delimiter in the file content
    num_delimiters = [len(re.findall(delim, content)) for delim in possible_delimiters]

    num_white_space_delimiters = len(re.findall(r"\s+", content)) - len(
        re.findall(r"\t", content),
    )
    # Add white space delimiter at this point to not interfere with the previous count
    possible_delimiters.append(r"\s+")
    num_delimiters.append(num_white_space_delimiters)
    num_delimiters = np.array(num_delimiters)

    # Raise an error if no delimiter is found in the file.
    if np.sum(num_delimiters) == 0:
        error_message = (
            f"no_delimiter_found - Unable to find a delimiter in file {file_path.name}"
        )
        raise ValueError(error_message)

    # Return the delimiter that appears most frequently
    return possible_delimiters[np.argmax(num_delimiters)]


def can_convert_str_to_float(value: str) -> bool:
    """Check if a string can be converted to a float.

    Parameters
    ----------
    value : str
        The string to check.

    Returns
    -------
    bool
        True if the string can be converted to a float, False otherwise.

    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def array_is_like_an_index(
    array: pd.Series,
) -> bool:
    """Check if the array is like an index."""
    # Check if all entries are integers and in ascending order
    return bool(np.issubdtype(array.dtype, np.integer) & np.all(np.diff(array) >= 0))
