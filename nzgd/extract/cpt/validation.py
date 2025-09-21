"""Functions to validate extracted CPT data."""

from nzgd import constants
from nzgd.extract.cpt import (
    data_structures,
    info,
    tasks,
)


def validate_initial_extraction_of_sheet(
    extraction_from_sheet: data_structures.SheetExtractionResult,
) -> data_structures.SheetExtractionResult:
    """Perform basic validation checks of the raw extractions.

    Parameters
    ----------
    extraction_from_sheet : data_structures.SheetExtractionResult
        The extraction result from a single sheet.

    Returns
    -------
    data_structures.SheetExtractionResult
        The updated extraction result after validation checks.

    """
    # Type guard: Immediately return if there is no successful extraction to validate
    if not isinstance(
        extraction_from_sheet.extraction,
        data_structures.ExtractedDataAndColInfo,
    ):
        return extraction_from_sheet

    # Count different data types in rows and columns
    data_type_count = info.count_num_str_float(
        extraction_from_sheet.extraction.data_df,
    )

    # Identify rows that have sufficient numerical data
    keep_row_mask = (
        data_type_count.numerical_surplus_per_row
        >= constants.MIN_NUMERICAL_SURPLUS_PER_ROW
    )

    # Only keep rows that have sufficient numerical data
    extraction_from_sheet.extraction.data_df = extraction_from_sheet.extraction.data_df[
        keep_row_mask
    ]

    # Count the number of rows and columns after filtering
    data_type_count = info.count_num_str_float(
        extraction_from_sheet.extraction.data_df,
    )

    ## Check if the dataframe has any remaining numeric data after filtering
    if data_type_count.num_numerical_values == 0:
        extraction_from_sheet.extraction = tasks.make_error_summary_df(
            file_path=extraction_from_sheet.file_path,
            error_category="no_numeric_data",
            error_details="has no numeric data",
            sheet_name=extraction_from_sheet.sheet_name.replace("-", "_"),
        )

        return extraction_from_sheet

    return extraction_from_sheet


def identify_missing_columns_in_sheet(
    extraction_from_sheet: data_structures.SheetExtractionResult,
) -> data_structures.SheetExtractionResult:
    """Check that every parameter has at least one possible column name."""
    # If this sheet did not result in a successful extraction, return immediately
    if not isinstance(
        extraction_from_sheet.extraction,
        data_structures.ExtractedDataAndColInfo,
    ):
        return extraction_from_sheet

    missing_col_indices = [
        not len(possible_col_names) > 0
        for possible_col_names in extraction_from_sheet.extraction.col_info
    ]

    # Identify the missing columns by indexing into a numpy array of column names
    missing_cols = constants.COLUMN_CPT_DATA_TYPES[missing_col_indices]
    if len(missing_cols) > 0:
        extraction_from_sheet.extraction = tasks.make_error_summary_df(
            file_path=extraction_from_sheet.file_path,
            error_category="missing_columns",
            error_details=(
                f"{extraction_from_sheet.file_path.name.replace('-', '_')} sheet "
                f"{extraction_from_sheet.sheet_name} "
                f"is missing [{' & '.join(missing_cols)}]"
            ),
            sheet_name=extraction_from_sheet.sheet_name.replace(
                "-",
                "_",
            ),
        )

    return extraction_from_sheet
