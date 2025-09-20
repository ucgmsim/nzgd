"""Functions to select columns from sheet extraction results."""

import re
from collections import defaultdict

import numpy as np

from nzgd_data_extraction import (
    constants,
    data_structures,
    errors,
    info,
    validation,
)


def select_columns_for_one_sheet(
    sheet_extraction_result: data_structures.SheetExtractionResult,
) -> data_structures.SheetExtractionResult:
    """Select columns for a single sheet based on the successful extraction results.

    Parameters
    ----------
    sheet_extraction_result : data_structures.SheetExtractionResult
        A SheetExtractionResult object containing the extraction results.

    Returns
    -------
    data_structures.SheetExtractionResult
        A SheetExtractionResult object with selected columns.

    """
    # Type guard: Immediately return if there is no successful extraction to modify
    if not isinstance(
        sheet_extraction_result.extraction,
        data_structures.ExtractedDataAndColInfo,
    ):
        return sheet_extraction_result

    # Keep only one of repeated column names in the spreadsheet.
    sheet_extraction_result = keep_one_of_repeated_column_names(
        sheet_extraction_result,
    )

    # Remove invalid column options
    sheet_extraction_result = remove_invalid_column_options(
        sheet_extraction_result,
    )

    # Check if required columns were removed
    return validation.identify_missing_columns_in_sheet(
        sheet_extraction_result,
    )


def keep_one_of_repeated_column_names(
    sheet_extraction_result: data_structures.SheetExtractionResult,
) -> data_structures.SheetExtractionResult:
    """Keep one of repeated column names.

    Some CPT source files such as CPT_2720_Raw01.CSV have repeated columns names.
    This function will remove all but one of the repeated columns from the extracted
    col_info. However, note that this will not remove the repeated columns from the
    extracted DataFrame.

    If the repeated column names have a different number of data points, the one with
    the most data points will be kept. If they have the same number of data points,
    the first one will be kept.

    Parameters
    ----------
    sheet_extraction_result : data_structures.SheetExtractionResult
        A SheetExtractionResult object containing the extraction results.

    Returns
    -------
    data_structures.SheetExtractionResult
        A SheetExtractionResult object with invalid columns removed.

    """
    # If this sheet did not provide a successful extraction, return immediately
    if not isinstance(
        sheet_extraction_result.extraction,
        data_structures.ExtractedDataAndColInfo,
    ):
        return sheet_extraction_result

    data_df = sheet_extraction_result.extraction.data_df

    # Rename columns by appending _X where X is the column index
    data_df.columns = [f"{col}COLINDEX{i}" for i, col in enumerate(data_df.columns)]

    col_options_per_param = sheet_extraction_result.extraction.col_info

    data_type_count = info.count_num_str_float(
        data_df,
    )

    all_params_col_options_to_discard = []
    for col_options_for_this_param in col_options_per_param:
        # Find repeated column names
        col_name_counts = defaultdict(lambda: 0)
        repeated_col_names_with_metadata = []
        repeated_col_names_num_data_points = []

        for col_option in col_options_for_this_param:
            col_name_counts[col_option.matched_string] += 1

        repeated_col_names = [k for k, v in col_name_counts.items() if v > 1]

        if len(repeated_col_names) > 0:
            for col_option in col_options_for_this_param:
                if col_option.matched_string in repeated_col_names:
                    repeated_col_names_with_metadata.append(col_option)

            # Get the corresponding number of data points
            for col_option in repeated_col_names_with_metadata:
                col_name = col_option.matched_string
                col_index = col_option.col_index_in_line

                repeated_col_names_num_data_points.append(
                    data_type_count.num_numerical_values_per_col[
                        f"{col_name}COLINDEX{col_index}"
                    ],
                )

            repeated_col_names_num_data_points = np.array(
                repeated_col_names_num_data_points,
            )

            num_points_and_metadata_tuple = zip(
                repeated_col_names_with_metadata,
                repeated_col_names_num_data_points,
                strict=False,
            )

            # If all repeated columns have the same number of data points,
            # we will keep the first one.
            # If one has more data points than the others, it will be kept.

            # Sort by number of data points in descending order and then sort by -column
            # index in ascending order so that the first column with the most data
            # points is kept. The second sort is done with a descending (reverse) sort
            # by sorting on negative column index (-x[0].col_index_in_line).
            num_points_and_metadata_tuple = sorted(
                num_points_and_metadata_tuple,
                key=lambda x: (x[1], -x[0].col_index_in_line),
                reverse=True,
            )

            col_options_to_discard = [x[0] for x in num_points_and_metadata_tuple][1:]
            all_params_col_options_to_discard.extend(col_options_to_discard)

            col_names_to_discard = [
                f"{x.matched_string}COLINDEX{x.col_index_in_line}"
                for x in col_options_to_discard
            ]

            data_df = data_df.drop(columns=col_names_to_discard)

    # Strip COLINDEXN from column names
    data_df.columns = [re.sub(r"COLINDEX\d+$", "", col) for col in data_df.columns]

    updated_col_options_per_param = data_structures.AllCPTColsSearchResults(
        col1_search_result=[],
        col2_search_result=[],
        col3_search_result=[],
        col4_search_result=[],
    )

    for param_idx, col_options_for_this_param in enumerate(col_options_per_param):
        for col_option in col_options_for_this_param:
            if col_option not in all_params_col_options_to_discard:
                updated_col_options_per_param[param_idx].append(col_option)

    sheet_extraction_result.extraction.data_df = data_df
    sheet_extraction_result.extraction.col_info = updated_col_options_per_param

    return sheet_extraction_result


def remove_invalid_column_options(
    sheet_extraction_result: data_structures.SheetExtractionResult,
) -> data_structures.SheetExtractionResult:
    """Remove columns options that contain invalid data.

    Parameters
    ----------
    sheet_extraction_result : data_structures.SheetExtractionResult
        A SheetExtractionResult object containing the extraction results.

    Returns
    -------
    data_structures.SheetExtractionResult
        A SheetExtractionResult object with invalid columns removed.

    """
    # If this sheet did not provide a successful extraction, return immediately
    if not isinstance(
        sheet_extraction_result.extraction,
        data_structures.ExtractedDataAndColInfo,
    ):
        return sheet_extraction_result

    valid_cols_for_all_params = []

    col_options_per_param = sheet_extraction_result.extraction.col_info

    data_type_count = info.count_num_str_float(
        sheet_extraction_result.extraction.data_df,
    )

    already_seen_col_names = []
    for col_options_for_this_param in col_options_per_param:
        valid_cols_for_this_param = []

        for col_option in col_options_for_this_param:
            # Check that the column has the minimum number of data points and
            # is not like an index like [0, 1, 2, 3...] and
            # has not been seen before, so all column names are unique

            if (
                (
                    data_type_count.num_numerical_values_per_col[
                        col_option.matched_string
                    ]
                    > constants.MIN_NUM_DATA_POINTS_PER_COLUMN
                )
                and (
                    not info.array_is_like_an_index(
                        sheet_extraction_result.extraction.data_df[
                            col_option.matched_string
                        ],
                    )
                )
                and col_option.matched_string not in already_seen_col_names
            ):
                valid_cols_for_this_param.append(col_option)
                already_seen_col_names.append(col_option.matched_string)

        valid_cols_for_all_params.append(valid_cols_for_this_param)

    # If no valid columns are found for a parameter, its corresponding
    # entry in valid_cols_for_all_params is set to an empty list.
    # Therefore, valid_cols_for_all_params should always have a length of 4.
    if len(valid_cols_for_all_params) != constants.REQUIRED_NUMBER_OF_COLUMNS:
        raise errors.FileProcessingError

    valid_cols = data_structures.AllCPTColsSearchResults(
        col1_search_result=valid_cols_for_all_params[0],
        col2_search_result=valid_cols_for_all_params[1],
        col3_search_result=valid_cols_for_all_params[2],
        col4_search_result=valid_cols_for_all_params[3],
    )

    # After removing invalid columns, check if any required columns are missing
    sheet_extraction_result.extraction.col_info = valid_cols
    sheet_extraction_result = validation.identify_missing_columns_in_sheet(
        sheet_extraction_result,
    )
    # Return without proceeding if required columns were removed
    if not isinstance(
        sheet_extraction_result.extraction,
        data_structures.ExtractedDataAndColInfo,
    ):
        return sheet_extraction_result

    sheet_extraction_result.extraction.col_info = valid_cols

    valid_column_names = [
        col_info[0].matched_string
        for col_info in sheet_extraction_result.extraction.col_info
    ]

    sheet_extraction_result.extraction.data_df = (
        sheet_extraction_result.extraction.data_df[valid_column_names]
    )
    return sheet_extraction_result
