"""Data conditioning and preprocessing for NZGD extractions."""

import numpy as np
import pandas as pd

from nzgd.extract.cpt import data_structures, tasks


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
