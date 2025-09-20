"""Data conditioning and preprocessing for NZGD extractions."""

import numpy as np
import pandas as pd

from nzgd_data_extraction import data_structures, tasks


def data_conditioning_for_one_sheet(
    extraction_from_sheet: data_structures.SheetExtractionResult,
) -> data_structures.SheetExtractionResult:
    """Apply complete data conditioning pipeline to a single sheet extraction.

    Parameters
    ----------
    extraction_from_sheet : data_structures.SheetExtractionResult
        The sheet extraction result to be conditioned.

    Returns
    -------
    data_structures.SheetExtractionResult
        The conditioned sheet extraction result.

    """
    extraction_from_sheet = tasks.enforce_positive_depth_for_sheet(
        extraction_from_sheet,
    )

    extraction_from_sheet = tasks.explicit_unit_conversions_in_sheet(
        extraction_from_sheet,
    )

    extraction_from_sheet = tasks.standardize_column_names_in_sheet(
        extraction_from_sheet,
    )

    return tasks.infer_unit_conversions_for_sheet(
        extraction_from_sheet,
    )


def get_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Get rows that are not all zeros or NaNs.

    Filters out rows from a DataFrame where all values are either zero, NaN,
    or a combination of both. Returns a clean DataFrame with only rows that
    contain at least one valid (non-zero, non-NaN) value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to filter for valid rows.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only rows with at least one valid value,
        with reset indices starting from 0.

    Notes
    -----
    The function considers a row invalid if:
    - All values are zero
    - All values are NaN
    - All values are either zero or NaN

    """
    df_reset = df.reset_index(drop=True)
    valid_mask = ~(
        (df_reset == 0).all(axis=1)
        | df_reset.isna().all(axis=1)
        | ((df_reset == 0) | df_reset.isna()).all(axis=1)
    )
    return df_reset[valid_mask].reset_index(drop=True)


def is_contiguous_subset(smaller_df: pd.DataFrame, larger_df: pd.DataFrame) -> bool:
    """Check if smaller_df exists as contiguous subset in larger_df.

    Determines whether the smaller DataFrame appears as a contiguous block
    of rows within the larger DataFrame. Both DataFrames must have the same
    number of columns, and the comparison is done row-by-row using exact
    equality.

    Parameters
    ----------
    smaller_df : pd.DataFrame
        The DataFrame to search for as a subset. Must have the same number
        of columns as larger_df.
    larger_df : pd.DataFrame
        The DataFrame to search within. Must have at least as many rows
        as smaller_df.

    Returns
    -------
    bool
        True if smaller_df exists as a contiguous subset in larger_df,
        False otherwise.

    Notes
    -----
    The function returns False immediately if:
    - Either DataFrame is empty
    - The DataFrames have different numbers of columns
    - The smaller DataFrame has more rows than the larger one

    The search uses sliding window comparison, checking each possible
    starting position in the larger DataFrame.

    """
    if smaller_df.empty or larger_df.empty or smaller_df.shape[1] != larger_df.shape[1]:
        return False

    smaller_rows = len(smaller_df)
    for start_idx in range(len(larger_df) - smaller_rows + 1):
        if larger_df.iloc[start_idx : start_idx + smaller_rows].equals(smaller_df):
            return True
    return False


def remove_duplicate_extractions(
    extractions_from_sheets: list[data_structures.SheetExtractionResult],
) -> list[data_structures.SheetExtractionResult]:
    """Remove duplicate extractions based on DataFrame content.

    Identifies and removes duplicate sheet extractions by comparing the actual
    data content of their extracted DataFrames. The function handles exact
    duplicates as well as cases where one DataFrame is a contiguous subset
    of another, keeping the larger/more complete version.

    Parameters
    ----------
    extractions_from_sheets : list[data_structures.SheetExtractionResult]
        List of sheet extraction results to deduplicate. Each extraction
        may contain a DataFrame with extracted data.

    Returns
    -------
    list[data_structures.SheetExtractionResult]
        List of unique extractions with duplicates removed. The order
        may differ from the input list, and the length will be less than
        or equal to the input length.

    Notes
    -----
    The deduplication process:
    1. Skips extractions without successful results
    2. Filters out invalid rows (all zeros/NaNs) before comparison
    3. Compares DataFrames with matching column counts
    4. Handles three types of duplicates:
       - Exact matches: removes the duplicate
       - Smaller subset of larger: removes the smaller one
       - Larger superset of smaller: replaces smaller with larger

    Only extractions with the same number of columns are compared.
    Empty DataFrames after filtering are excluded from the results.

    """
    seen_dataframes = []
    unique_extractions = []

    def get_file_sheet_str(sheet_extraction):
        file_name = sheet_extraction.file_path.name
        nzgd_id = int(file_name.split("_")[1])
        sheet_name = sheet_extraction.sheet_name

        return f"{nzgd_id}_AND_{file_name}_AND_{sheet_name}"

    for _, sheet_extraction in enumerate(extractions_from_sheets):
        if not isinstance(
            sheet_extraction.extraction,
            data_structures.ExtractedDataAndColInfo,
        ):
            unique_extractions.append(sheet_extraction)
            continue

        current_df = get_valid_rows(
            sheet_extraction.extraction.data_df,
        )

        if current_df.empty:
            continue

        # Check against all seen DataFrames
        duplicate_found = False
        for j, seen_df in enumerate(seen_dataframes):
            if current_df.shape[1] != seen_df.shape[1]:
                continue

            curr_rows, seen_rows = len(current_df), len(seen_df)

            # Helper: ensure removed_duplicates is a list
            if getattr(unique_extractions[j], "removed_duplicates", None) is None:
                unique_extractions[j].removed_duplicates = []

            if current_df.equals(seen_df):
                # Exact duplicate: add info to reference's removed_duplicates
                unique_extractions[j].removed_duplicates.append(
                    get_file_sheet_str(sheet_extraction),
                )
                duplicate_found = True
                break

            if curr_rows < seen_rows and is_contiguous_subset(current_df, seen_df):
                # Subset duplicate: add info to reference's removed_duplicates
                unique_extractions[j].removed_duplicates.append(
                    get_file_sheet_str(sheet_extraction),
                )
                duplicate_found = True
                break

            if curr_rows > seen_rows and is_contiguous_subset(seen_df, current_df):
                # The previous unique extraction is now a duplicate, so transfer its removed_duplicates to the new one
                old_unique = unique_extractions[j]
                # Prepare new removed_duplicates list
                new_removed = []
                # Add old unique's own file+sheet string
                new_removed.append(get_file_sheet_str(old_unique))
                # Add any previous removed_duplicates
                if getattr(old_unique, "removed_duplicates", None):
                    new_removed.extend(old_unique.removed_duplicates)
                # Set on the new unique extraction
                sheet_extraction.removed_duplicates = new_removed
                seen_dataframes[j] = current_df
                unique_extractions[j] = sheet_extraction
                duplicate_found = True
                break

        if not duplicate_found:
            seen_dataframes.append(current_df)
            unique_extractions.append(sheet_extraction)

    # if any of the removed_duplicates lists are empty, set to None
    for extraction_result in unique_extractions:
        if getattr(extraction_result, "removed_duplicates", None) == []:
            extraction_result.removed_duplicates = None

    return unique_extractions


def remove_non_numerical_data_for_one_sheet(
    extraction_from_sheet: data_structures.SheetExtractionResult,
) -> data_structures.SheetExtractionResult:
    """Remove non-numerical data from a single sheet extraction.

    Parameters
    ----------
    extraction_from_sheet : data_structures.SheetExtractionResult
        The sheet extraction result to process.

    Returns
    -------
    data_structures.SheetExtractionResult
        The modified sheet extraction result with non-numerical data removed.

    """
    # Type guard: Immediately return if there is no successful extraction to check
    if not isinstance(
        extraction_from_sheet.extraction,
        data_structures.ExtractedDataAndColInfo,
    ):
        return extraction_from_sheet

    extraction_from_sheet = remove_nondata_rows_after_data_in_sheet(
        extraction_from_sheet,
    )

    # Force any remaining non-numerical data to NaN
    data_df = extraction_from_sheet.extraction.data_df.apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not isinstance(data_df, pd.DataFrame):
        error_str = "Expected a DataFrame"
        raise TypeError(error_str)
    extraction_from_sheet.extraction.data_df = data_df

    return extraction_from_sheet


def remove_nondata_rows_after_data_in_sheet(
    sheet_extraction: data_structures.SheetExtractionResult,
) -> data_structures.SheetExtractionResult:
    """Remove non-data rows after the data in a DataFrame.

    Some records contain other data below the main data, that is separated by a header
    line that interrupts the leftmost (Depth) column. This function removes all rows
    after the first row that contains a string in the Depth column.

    Parameters
    ----------
    sheet_extraction : data_structures.SheetExtractionResult
        The SheetExtractionResult object containing the extraction results.

    Returns
    -------
    data_structures.SheetExtractionResult
        The SheetExtractionResult object with non-data rows removed.

    """
    # Type guard: Immediately return if there is no successful extraction to modify
    if not isinstance(
        sheet_extraction.extraction,
        data_structures.ExtractedDataAndColInfo,
    ):
        return sheet_extraction

    data_df = sheet_extraction.extraction.data_df
    depth_col_name = sheet_extraction.extraction.col_info.col1_search_result[
        0
    ].matched_string

    # Find rows in the specified column that contain string values
    str_in_depth = data_df[
        data_df[depth_col_name].map(
            lambda x: isinstance(x, str),
        )
    ]
    if len(str_in_depth) > 0:
        idx_of_str_in_depth = np.min(str_in_depth.index)
    else:
        idx_of_str_in_depth = len(data_df)

    # Find rows where all columns are NaN
    all_nan_rows = data_df[data_df.isna().all(axis=1)]

    if len(all_nan_rows) > 0:
        idx_of_nan_rows = np.min(all_nan_rows.index)
    else:
        idx_of_nan_rows = len(data_df)

    # Determine the first index after the data
    first_idx_after_data = min(idx_of_str_in_depth, idx_of_nan_rows)

    # Initialize a DataFrame attribute to store the index of the last data row
    # if there are non-data rows after the data. Leave as NaN if there are no
    # non-data rows after the data.
    data_df.attrs["ignoring_rows_after_this_row_index"] = np.nan

    # Truncate the DataFrame up to the first index after the data
    if first_idx_after_data < len(data_df):
        data_df = data_df.iloc[:first_idx_after_data]

        ## Storing first_idx_after_data in the attrs as a float so it can be written to parquet
        data_df.attrs["ignoring_rows_after_this_row_index"] = float(
            first_idx_after_data,
        )

    sheet_extraction.extraction.data_df = data_df

    return sheet_extraction
