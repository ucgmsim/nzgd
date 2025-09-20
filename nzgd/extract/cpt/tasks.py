import copy
import re
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import xlrd

from nzgd_data_extraction import (
    constants,
    data_structures,
    errors,
    info,
    search,
)


def apply_func_to_all_sheets(
    func: Callable,
    sheet_extraction_results: list[data_structures.SheetExtractionResult],
) -> list[data_structures.SheetExtractionResult]:
    """Apply a function to all sheets in a list of SheetExtractionResult objects.

    Parameters
    ----------
    func : Callable
        The function to apply to each SheetExtractionResult object.
    sheet_extraction_results : list[data_structures.SheetExtractionResult]
        A list of SheetExtractionResult objects to process.

    Returns
    -------
    list[data_structures.SheetExtractionResult]
        The list of SheetExtractionResult objects after applying the function.

    """
    modified_sheet_extractions = copy.copy(sheet_extraction_results)

    for idx, sheet_extraction in enumerate(sheet_extraction_results):
        # If this item is not the result of a successful extraction,
        # just continue with the next item in the list
        if not isinstance(
            sheet_extraction.extraction,
            data_structures.ExtractedDataAndColInfo,
        ):
            continue

        modified_sheet_extractions[idx] = func(sheet_extraction)

    return modified_sheet_extractions


def enforce_positive_depth_for_sheet(
    sheet_extraction: data_structures.SheetExtractionResult,
) -> data_structures.SheetExtractionResult:
    """Enforce positive depth values extracted from a sheet.

    Parameters
    ----------
    sheet_extraction : data_structures.SheetExtractionResult
        The SheetExtractionResult object containing the extraction results.

    Returns
    -------
    data_structures.SheetExtractionResult
        The SheetExtractionResult object with positive depth values.

    """
    # Type guard: Immediately return if there is no successful extraction to modify
    if not isinstance(
        sheet_extraction.extraction,
        data_structures.ExtractedDataAndColInfo,
    ):
        return sheet_extraction

    depth_col_name = sheet_extraction.extraction.col_info.col1_search_result[
        0
    ].matched_string

    data_df = sheet_extraction.extraction.data_df

    # If any of the values in the depth column are negative, take the absolute value
    # and set the attribute to indicate that the depth was originally negative
    if (data_df[depth_col_name] < 0).any():
        data_df[depth_col_name] = data_df[depth_col_name].abs()
        data_df.attrs["depth_originally_defined_as_negative"] = True

    sheet_extraction.extraction.data_df = data_df

    return sheet_extraction


def explicit_unit_conversions_in_sheet(
    sheet_extraction: data_structures.SheetExtractionResult,
) -> data_structures.SheetExtractionResult:
    """Explicit unit conversions in sheet.

    Converts data in columns with explicit indications of cm and kPa units to
    m and MPa, respectively.

    Parameters
    ----------
    sheet_extraction : data_structures.SheetExtractionResult
        The SheetExtractionResult object containing the extraction results.

    Returns
    -------
    data_structures.SheetExtractionResult
        The SheetExtractionResult object with converted units.

    """
    # Type guard: Immediately return if there is no successful extraction to check
    if not isinstance(
        sheet_extraction.extraction,
        data_structures.ExtractedDataAndColInfo,
    ):
        return sheet_extraction

    explicit_unit_conversions = []

    col_names = [
        col_info[0].matched_string for col_info in sheet_extraction.extraction.col_info
    ]

    converted_units_data_df = sheet_extraction.extraction.data_df.copy()

    placeholder_mask = converted_units_data_df.isin(
        constants.KNOWN_MISSING_VALUE_PLACEHOLDERS,
    )

    for col_index, col_name in enumerate(col_names):
        if col_name is not None:
            if col_index == 0:
                # checking the depth column
                if "cm" in col_name.lower():
                    converted_units_data_df.loc[:, col_name] /= 100.0
                    explicit_unit_conversions.append(
                        f"{col_name} was converted from cm to m",
                    )

            # checking the other columns
            elif "kpa" in col_name.lower():
                converted_units_data_df.loc[:, col_name] /= 1000.0
                explicit_unit_conversions.append(
                    f"{col_name} was converted from kPa to MPa",
                )

    # converted_units_data_df.attrs["explicit_unit_conversions"] = " & ".join(
    #     explicit_unit_conversions,
    # )

    # Where placeholder_mask is True, replace with original values
    converted_units_data_df = converted_units_data_df.mask(
        placeholder_mask,
        sheet_extraction.extraction.data_df,
    )

    sheet_extraction.extraction.data_df = converted_units_data_df
    sheet_extraction.explicit_unit_conversions = "_AND_".join(explicit_unit_conversions)

    return sheet_extraction


def standardize_column_names_in_sheet(
    sheet_extraction: data_structures.SheetExtractionResult,
) -> data_structures.SheetExtractionResult:
    """Standardize the column names extracted from a sheet.

    Parameters
    ----------
    sheet_extraction : data_structures.SheetExtractionResult
        The SheetExtractionResult object containing the data to be processed.

    Returns
    -------
    data_structures.SheetExtractionResult
        The SheetExtractionResult object with standardized column names.

    """
    # Type guard: Immediately return if there is no successful extraction to check
    if not isinstance(
        sheet_extraction.extraction,
        data_structures.ExtractedDataAndColInfo,
    ):
        return sheet_extraction

    sheet_extraction.extraction.data_df.columns = list(
        constants.COLUMN_DESCRIPTIONS,
    )[0:4]

    return sheet_extraction


def convert_dataframe_to_list_of_lists(df: pd.DataFrame) -> list[list[str]]:
    """Convert a DataFrame to a list of lists.

    This function converts a DataFrame to a list of lists, where each list represents a
    row in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to convert.

    Returns
    -------
    list[list[str]]
        A list of lists, where each list represents a row in the DataFrame.

    """
    return [df.iloc[i].to_list() for i in range(len(df))]


def convert_num_as_str_to_float(
    val: float | str,
) -> float | str:
    """Convert a numerical string to a float.

    This function attempts to convert a string to a float. If the conversion fails, it
    returns the original string.

    Parameters
    ----------
    val : Union[float, str, None]
        A value to try to convert to a float.

    Returns
    -------
    Union[float, str]
        The converted float if the value can be converted, otherwise the original value.

    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return val


def convert_numerical_str_cells_to_float(
    iterable: pd.Series | list,
) -> list[list]:
    """Convert numerical string cells to float in an iterable.

    This function iterates through each cell in the provided iterable and converts cells that are numerical strings to
    floats.

    Parameters
    ----------
    iterable : Union[pd.Series, list]
        The input data as a list of lists or a DataFrame.

    Returns
    -------
    list[list]
        The modified iterable with numerical strings converted to floats.

    """
    if isinstance(iterable, pd.DataFrame):
        iterable = [iterable.iloc[i].to_list() for i in range(len(iterable))]

    iterable_no_numerical_str = copy.copy(iterable)

    for row_idx, line in enumerate(iterable_no_numerical_str):
        for col_idx, cell in enumerate(line):
            ## some cells are read by pd.read_xls() as type datetime so they need to be converted to str
            cell = str(cell)
            if info.can_convert_str_to_float(cell):
                iterable_no_numerical_str[row_idx][col_idx] = float(cell)

    return iterable_no_numerical_str


def excel_skip_nondata_rows_at_start(
    extracted_data_df: pd.DataFrame,
    header_row_index: int,
) -> pd.DataFrame:
    """Skip non-data rows at the start of the DataFrame.

    Parameters
    ----------
    extracted_data_df : pd.DataFrame
        The DataFrame containing the data.
    header_row_index : int
        The index of the header row.

    Returns
    -------
    pd.DataFrame
        The DataFrame with non-data rows at the start skipped or
        an empty DataFrame if no data rows are found.

    """
    # Get the surplus of numeric values in each row
    data_type_count_info = info.count_num_str_float(
        extracted_data_df,
    )
    # Find the first data row after the header
    data_row_indices = np.where(
        (
            np.arange(len(data_type_count_info.numerical_surplus_per_row))
            > header_row_index
        )
        & (data_type_count_info.numerical_surplus_per_row > 0),
    )[0]

    # if there are no data rows after the header row, return an empty DataFrame
    if len(data_row_indices) == 0:
        return pd.DataFrame()

    first_data_row_index = data_row_indices[0]

    # Find empty rows between header and data
    empty_rows_between = extracted_data_df[
        (extracted_data_df.isna().all(axis=1))
        & (extracted_data_df.index > header_row_index)
        & (extracted_data_df.index < first_data_row_index)
    ].index

    # Calculate the number of rows to skip
    rows_to_skip = header_row_index + len(empty_rows_between)

    # The column names are stored as the DataFrame's column names so we skip the
    # original header row and any empty rows between the header and the first row of
    # data. The first included row is (rows_to_skip + 1) to start from the first
    # data row
    return extracted_data_df.iloc[rows_to_skip + 1 :].reset_index(
        drop=True,
    )


def combine_multiple_header_rows(
    loaded_data_df: pd.DataFrame,
    header_row_indices: npt.ArrayLike,
) -> tuple[pd.DataFrame, int]:
    """Combine multiple header rows into a single header row in a DataFrame.

    Parameters
    ----------
    loaded_data_df : pd.DataFrame
        The DataFrame containing the data with multiple header rows.
    header_row_indices : npt.ArrayLike
        An array of indices representing the header rows to be combined.

    Returns
    -------
    tuple[pd.DataFrame, int]
        A tuple containing the DataFrame with combined header rows and the index of the final header row.

    """
    ## take the header_row_index as the maximum of the header_row_indices
    ## which is the lowest row in the spreadsheet
    header_row_index = np.max(header_row_indices)

    ## copy the column names from the rows above the lowest header row
    loaded_data_df_with_combined_header_rows = loaded_data_df.copy()
    for row_idx in header_row_indices:
        for col_idx in range(loaded_data_df.shape[1]):
            if row_idx != header_row_index:
                loaded_data_df_with_combined_header_rows.iloc[
                    header_row_index,
                    col_idx,
                ] = (
                    str(loaded_data_df.iloc[header_row_index, col_idx])
                    + " "
                    + str(loaded_data_df.iloc[row_idx, col_idx])
                )

    return loaded_data_df_with_combined_header_rows, header_row_index


def infer_unit_conversions_for_sheet(
    sheet_extraction: data_structures.SheetExtractionResult,
) -> data_structures.SheetExtractionResult:
    """Infer from numerical values whether units should be converted.

    Assumes that only conversions to m or to MPa are needed.

    Parameters
    ----------
    sheet_extraction : data_structures.SheetExtractionResult
        The SheetExtractionResult object containing the extraction results.

    Returns
    -------
    data_structures.SheetExtractionResult
        The corrected DataFrame with inferred unit conversions and updated attributes.

    """
    # Type guard: Immediately return if there is no successful extraction to check
    if not isinstance(
        sheet_extraction.extraction,
        data_structures.ExtractedDataAndColInfo,
    ):
        return sheet_extraction

    converted_units_data_df = sheet_extraction.extraction.data_df.copy()

    placeholder_mask = sheet_extraction.extraction.data_df.isin(
        constants.KNOWN_MISSING_VALUE_PLACEHOLDERS,
    )

    # Remove placeholder values before wrong units are inferred as the placeholder
    # values have large magnitudes that could skew the results
    converted_units_data_df = converted_units_data_df.infer_objects(copy=False).replace(
        constants.KNOWN_MISSING_VALUE_PLACEHOLDERS,
        np.nan,
    )

    inferred_unit_conversions = []

    percent_depths_between_cm_thresh_and_mm_thresh = (
        100
        * np.sum(
            (
                converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[0]]
                > constants.INFER_WRONG_UNITS_THRESHOLDS["cm_threshold"]
            )
            & (
                converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[0]]
                < constants.INFER_WRONG_UNITS_THRESHOLDS["mm_threshold"]
            ),
        )
        / len(converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[0]])
    )

    percent_depths_exceeding_mm_thresh = (
        100
        * np.sum(
            converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[0]]
            > constants.INFER_WRONG_UNITS_THRESHOLDS["mm_threshold"],
        )
        / len(converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[0]])
    )

    percentage_qc_exceeding_kpa_threshold = (
        100
        * np.sum(
            converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[1]]
            > constants.INFER_WRONG_UNITS_THRESHOLDS["qc_kpa_threshold"],
        )
        / len(converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[1]])
    )

    percentage_fs_exceeding_kpa_threshold = (
        100
        * np.sum(
            converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[2]]
            > constants.INFER_WRONG_UNITS_THRESHOLDS["fs_kpa_threshold"],
        )
        / len(converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[2]])
    )

    percentage_u_exceeding_kpa_threshold = (
        100
        * np.sum(
            np.abs(converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[3]])
            > constants.INFER_WRONG_UNITS_THRESHOLDS["u_kpa_threshold"],
        )
        / len(converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[3]])
    )

    median_depth_step_size = info.get_depth_step_size(
        converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[0]],
    )

    if (
        percent_depths_between_cm_thresh_and_mm_thresh
        > constants.INFER_WRONG_UNITS_THRESHOLDS["percent_exceeding_threshold"]
    ):
        # depth values are likely in cm
        converted_units_data_df.loc[:, list(constants.COLUMN_DESCRIPTIONS)[0]] /= 100.0
        median_depth_step_size = info.get_depth_step_size(
            converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[0]],
        )
        inferred_unit_conversions.append(
            f"{list(constants.COLUMN_DESCRIPTIONS)[0]} was converted from cm to m",
        )

    elif (
        percent_depths_exceeding_mm_thresh
        > constants.INFER_WRONG_UNITS_THRESHOLDS["percent_exceeding_threshold"]
    ):
        # depth values are likely in mm
        converted_units_data_df.loc[:, list(constants.COLUMN_DESCRIPTIONS)[0]] /= 1000.0
        median_depth_step_size = info.get_depth_step_size(
            converted_units_data_df[list(constants.COLUMN_DESCRIPTIONS)[0]],
        )
        inferred_unit_conversions.append(
            f"{list(constants.COLUMN_DESCRIPTIONS)[0]} was converted from mm to m",
        )

    # CPT_24476, CPT_24479, and CPT_24481 indicate units of cm, so they are divided
    # by 100 by the explicit unit conversion function.  However, the depth values are
    # actually in m, so they become far too small if dividing by 100.  This check is to
    # catch such cases.

    # If the median step size in depth values is less than 1mm, then the depth values
    # are likely not in m, as a typical step size in depth values is around 1cm.
    # Therefore, we estimate a scaling factor to multiply the depth values by to convert
    # them to metres.
    elif median_depth_step_size < float(
        constants.INFER_WRONG_UNITS_THRESHOLDS[
            "smaller_depth_step_size_implies_not_in_m"
        ],
    ):
        guessed_scaling_factor = info.round_to_nearest_power_of_10(
            float(constants.INFER_WRONG_UNITS_THRESHOLDS["typical_depth_step_size_m"])
            / median_depth_step_size,
        )
        converted_units_data_df.loc[:, list(constants.COLUMN_DESCRIPTIONS)[0]] *= (
            guessed_scaling_factor
        )
        inferred_unit_conversions.append(
            f"{list(constants.COLUMN_DESCRIPTIONS)[0]} was multiplied by a guessed scaling factor of {guessed_scaling_factor}",
        )

    if (
        percentage_qc_exceeding_kpa_threshold
        > constants.INFER_WRONG_UNITS_THRESHOLDS["percent_exceeding_threshold"]
    ):
        # qc values are likely in kPa
        converted_units_data_df.loc[:, list(constants.COLUMN_DESCRIPTIONS)[1]] /= 1000.0
        inferred_unit_conversions.append(
            f"{list(constants.COLUMN_DESCRIPTIONS)[1]} was converted from kPa to MPa",
        )
    if (
        percentage_fs_exceeding_kpa_threshold
        > constants.INFER_WRONG_UNITS_THRESHOLDS["fs_kpa_threshold"]
    ):
        # fs values are likely in kPa
        converted_units_data_df.loc[:, list(constants.COLUMN_DESCRIPTIONS)[2]] /= 1000.0
        inferred_unit_conversions.append(
            f"{list(constants.COLUMN_DESCRIPTIONS)[2]} was converted from kPa to MPa",
        )

    if (
        percentage_u_exceeding_kpa_threshold
        > constants.INFER_WRONG_UNITS_THRESHOLDS["percent_exceeding_threshold"]
    ):
        # u values are likely in kPa
        converted_units_data_df.loc[:, list(constants.COLUMN_DESCRIPTIONS)[3]] /= 1000.0
        inferred_unit_conversions.append(
            f"{list(constants.COLUMN_DESCRIPTIONS)[3]} was converted from kPa to MPa",
        )

    # converted_units_data_df.attrs["inferred_unit_conversions"] = " & ".join(
    #     inferred_unit_conversions,
    # )

    # restore the original placeholder values (where placeholder_mask is True)
    converted_units_data_df = converted_units_data_df.mask(
        placeholder_mask,
        sheet_extraction.extraction.data_df,
    )

    sheet_extraction.extraction.data_df = converted_units_data_df
    sheet_extraction.inferred_unit_conversions = "_AND_".join(
        inferred_unit_conversions,
    )

    return sheet_extraction


def ensure_positive_depth(loaded_data_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that the depth column has positive values.

    Parameters
    ----------
    loaded_data_df : pd.DataFrame
        The input DataFrame containing the data to be checked.

    Returns
    -------
    pd.DataFrame
        The corrected DataFrame with positive depth values and no negative values in qc and fs columns.

    """
    ## Ensure that the depth column is defined as positive (some have depth as negative)
    if loaded_data_df[list(constants.COLUMN_DESCRIPTIONS)[0]].min() < 0:
        loaded_data_df.loc[:, list(constants.COLUMN_DESCRIPTIONS)[0]] = np.abs(
            loaded_data_df[list(constants.COLUMN_DESCRIPTIONS)[0]],
        )
        loaded_data_df.attrs["depth_originally_defined_as_negative"] = True
    else:
        loaded_data_df.attrs["depth_originally_defined_as_negative"] = False

    return loaded_data_df


def make_summary_df_per_record(
    record_dir_name: str,
    file_was_loaded: bool,
    loaded_file_type: str,
    loaded_file_name: str,
    pdf_file_list: list,
    cpt_file_list: list,
    ags_file_list: list,
    xls_file_list: list,
    xlsx_file_list: list,
    csv_file_list: list,
    txt_file_list: list,
    unknown_list: list,
):
    """Create a summary DataFrame with information about the loaded files.

    Parameters
    ----------
    record_dir_name : str
        The name of the record directory.
    file_was_loaded : bool
        A flag indicating whether a file was successfully loaded.
    loaded_file_type : str
        The type of the loaded file.
    loaded_file_name : str
        The name of the loaded file.
    pdf_file_list : list
        A list of PDF files.
    cpt_file_list : list
        A list of CPT files.
    ags_file_list : list
        A list of AGS files.
    xls_file_list : list
        A list of XLS files.
    xlsx_file_list : list
        A list of XLSX files.
    csv_file_list : list
        A list of CSV files.
    txt_file_list : list
        A list of TXT files.
    unknown_list : list
        A list of files with unknown types.

    Returns
    -------
    pd.DataFrame
        The concatenated summary DataFrame with the new information added.

    """
    if (
        (len(pdf_file_list) > 0)
        & (len(cpt_file_list) == 0)
        & (len(ags_file_list) == 0)
        & (len(xls_file_list) == 0)
        & (len(xlsx_file_list) == 0)
        & (len(csv_file_list) == 0)
        & (len(txt_file_list) == 0)
        & (len(unknown_list) == 0)
    ):
        has_only_pdf = True
    else:
        has_only_pdf = False

    loading_summary = pd.DataFrame(
        {
            "record_name": record_dir_name,
            "file_was_loaded": file_was_loaded,
            "loaded_file_type": loaded_file_type,
            "loaded_file_name": loaded_file_name,
            "only_has_pdf": has_only_pdf,
            "num_pdf_files": len(pdf_file_list),
            "num_cpt_files": len(cpt_file_list),
            "num_ags_files": len(ags_file_list),
            "num_xls_files": len(xls_file_list),
            "num_xlsx_files": len(xlsx_file_list),
            "num_csv_files": len(csv_file_list),
            "num_txt_files": len(txt_file_list),
            "num_other_files": len(unknown_list),
        },
        index=[0],
    )

    return loading_summary


def make_column_names_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Make DataFrame column names unique by adding suffixes to duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with potentially duplicate column names.

    Returns
    -------
    pd.DataFrame
        The DataFrame with unique column names.

    """
    columns = df.columns.tolist()
    seen = {}
    new_columns = []

    for col in columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 1
            new_columns.append(col)

    df_copy = df.copy()
    df_copy.columns = new_columns
    return df_copy


def identify_cols_to_avoid_if_possible(
    possible_cols_for_parameter: list[data_structures.SearchForColResults],
) -> npt.NDArray[np.bool_]:
    """Check if a column should be avoided if possible.

    Parameters
    ----------
    possible_cols_for_parameter : list[data_structures.SearchForColResults]
        The possible column names with additional information.

    Returns
    -------
    npt.NDArray[np.bool_]
        An array indicating whether each column should be avoided.

    """
    should_avoid_col_if_possible = np.zeros(
        len(possible_cols_for_parameter),
        dtype=np.bool_,
    )

    for possible_col_idx, possible_col in enumerate(
        possible_cols_for_parameter,
    ):
        for avoid_word in constants.AVOID_COLUMN_NAMES_CONTAINING_IF_POSSIBLE:
            if avoid_word in possible_col.matched_string.lower():
                should_avoid_col_if_possible[possible_col_idx] = True
                break

    return should_avoid_col_if_possible


def select_columns_to_extract_from_one_sheet(
    sheet_extraction_result: data_structures.SheetExtractionResult,
) -> data_structures.AllCPTColsSearchResults:
    """Select column names from among the identified options.

    Parameters
    ----------
    sheet_extraction_result : data_structures.SheetExtractionResult
        The SheetExtractionResult object containing the extraction results.

    Returns
    -------
    data_structures.AllCPTColsSearchResults
        The column search results.

    """
    selected_cols = []

    for param_col_options in sheet_extraction_result.extraction.col_info:
        should_avoid_col_idx = identify_cols_to_avoid_if_possible(
            param_col_options,
        )

        # If there is only one option or all columns should be avoided,
        # select the first column.
        if len(param_col_options) == 1 or np.all(should_avoid_col_idx):
            selected_cols.append(param_col_options[0])

        # Otherwise, select the first column that should not be avoided.
        else:
            selected_cols.append(
                param_col_options[np.argmax(~should_avoid_col_idx)],
            )

    # Check for the required number of unique column names
    col_names_str_set = {selected_col.matched_string for selected_col in selected_cols}
    if len(col_names_str_set) != constants.REQUIRED_NUMBER_OF_COLUMNS:
        raise errors.IncorrectNumberOfColumnsError

    # Create a new instance of AllCPTColsSearchResults with the selected columns
    return data_structures.AllCPTColsSearchResults(
        col1_search_result=[selected_cols[0]],
        col2_search_result=[selected_cols[1]],
        col3_search_result=[selected_cols[2]],
        col4_search_result=[selected_cols[3]],
    )


def get_xls_sheet_names(
    file_path: Path,
) -> tuple[list[str], constants.ExcelEngine]:
    """Get the sheet names from an Excel file and determine the engine used to read the file.

    This function attempts to read the sheet names from an Excel file using the xlrd and openpyxl engines.
    If the file is not a valid .xls or .xlsx file, it raises a FileProcessingError.

    Parameters
    ----------
    file_path : Path
        The path to the Excel file.

    Returns
    -------
    tuple[list[str], str]
        A tuple containing a list of sheet names and the engine used to read the file.

    Raises
    ------
    FileProcessingError
        If the file is not a valid .xls or .xlsx file.

    """
    if file_path.suffix.lower() == ".xls":
        engine = constants.ExcelEngine.xlrd
    else:
        engine = constants.ExcelEngine.openpyxl

    # Some .xls files are actually xlsx files and need to be opened with openpyxl
    try:
        sheet_names = pd.ExcelFile(file_path, engine=engine).sheet_names
        return sheet_names, engine

    except (xlrd.biffh.XLRDError, TypeError):
        if engine == constants.ExcelEngine.xlrd:
            other_engine = constants.ExcelEngine.openpyxl
        else:
            other_engine = constants.ExcelEngine.xlrd

        engine = other_engine
        try:
            sheet_names = pd.ExcelFile(file_path, engine=engine).sheet_names
        except:
            error_message = (
                f"invalid_excel_file - file {file_path.name} is not a "
                "valid xls or xlsx file"
            )
            raise errors.InvalidExcelFileError(error_message)

        return sheet_names, engine

    except zipfile.BadZipFile:
        error_message = (
            f"invalid_excel_file - file {file_path.name} is not a "
            "valid xls or xlsx file"
        )
        raise errors.InvalidExcelFileError(error_message)

    except xlrd.compdoc.CompDocError:
        error_message = (
            f"invalid_excel_file - file {file_path.name} has MSAT extension corruption"
        )
        raise errors.InvalidExcelFileError(error_message)


def find_missing_cols_for_best_sheet(missing_columns_per_sheet: list[list]) -> list:
    """Find the sheet with the fewest missing columns.

    Parameters
    ----------
    missing_columns_per_sheet : list[list]
        A list of lists, where each inner list contains the missing columns for a sheet.

    Returns
    -------
    list
        The list of missing columns for the sheet with the fewest missing columns.

    """
    final_num_missing_cols = 5
    final_missing_cols = []
    for missing_cols in missing_columns_per_sheet:
        if len(missing_cols) < final_num_missing_cols:
            final_num_missing_cols = len(missing_cols)
            final_missing_cols = missing_cols
    return final_missing_cols


# Remove delimiter characters that are between quote characters before splitting
def remove_delimiters_between_quotes(text: str, delimiter: str) -> str:
    """Remove delimiter characters that are between quote characters using regex.

    This function processes text to remove specified delimiter characters that occur
    within quoted strings (both single and double quotes) while preserving the
    quoted strings and their quote characters. Text outside of quotes remains
    unchanged. This is useful for preprocessing delimited text files where
    delimiter characters within quoted fields should not be treated as field
    separators.

    Parameters
    ----------
    text : str
        The input text string to process. May contain quoted and unquoted sections.
    delimiter : str
        The delimiter character(s) to remove from within quoted strings. Can be
        a single character (e.g., ',', ';') or multiple characters (e.g., ', ').

    Returns
    -------
    str
        The processed text with delimiter characters removed from within quoted
        strings. Quote characters and text outside quotes are preserved unchanged.

    """
    # Pattern to match content within quotes (both single and double)
    # This pattern captures quoted strings and preserves them
    quote_pattern = r'("[^"]*"|\'[^\']*\')'

    # Split text into quoted and unquoted parts
    parts = re.split(quote_pattern, text)

    result_parts = []
    for i, part in enumerate(parts):
        # Even indices are unquoted text, odd indices are quoted text
        if i % 2 == 0:
            # Keep unquoted parts as is
            result_parts.append(part)
        # For quoted parts, remove delimiters inside but keep the quotes
        elif part.startswith('"') and part.endswith('"'):
            inner_content = part[1:-1].replace(delimiter, "")
            result_parts.append(f'"{inner_content}"')
        elif part.startswith("'") and part.endswith("'"):
            inner_content = part[1:-1].replace(delimiter, "")
            result_parts.append(f"'{inner_content}'")
        else:
            result_parts.append(part)

    return "".join(result_parts)


def get_csv_or_txt_split_readlines(
    file_path: Path,
    encoding: str,
    delimiter: str,
) -> list[list[str | float]]:
    """Load a CSV or TXT file as a list of lists of strings.

    Each list represents a line in the file. Each item in the
    list represents a cell in the line.

    Parameters
    ----------
    file_path : Path
        The path to the file to be read.
    encoding : str
        The encoding to use when reading the file.
    delimiter : str
        The delimiter to use when reading the file.

    Returns
    -------
    list[list[str]]
        A list of lists containing strings. Each list represents a line in the file. Each item in the list represents a cell in
        the line.

    Raises
    ------
    FileProcessingError
        If the file contains only one line.

    """
    with open(file_path, encoding=encoding) as file:
        lines = file.readlines()

    # Possibly don't need this check here
    # if len(lines) == 1:
    #     raise errors.FileProcessingError(
    #         f"only_one_line - sheet (0) has only one line with first cell of {lines[0]}",
    #     )

    ## If the delimiter is white space (r"\s+"), special handling is required for cases such as CPT_59209,
    ## which does not give a unit for Depth, so has an empty cell at the start of its second header row, as shown below:
    ## Depth    qc   fs    u2
    ##         [MPa] [MPa] [MPa]
    ## If no special consideration is taken, the [MPa] unit will be associated with the Depth column.
    if delimiter == r"\s+":
        for line_index in range(len(lines)):
            line_string = lines[line_index]
            line_string_white_space_stripped = line_string.strip(" ")
            ## Skip blank lines
            if line_string_white_space_stripped != "\n":
                ## Regex expression to match white space (\s+) at the start of the line (^)
                if re.match(r"^\s+", line_string):
                    lines[line_index] = "placeholder" + line_string

    # Apply delimiter removal to each line before splitting
    processed_lines = [
        remove_delimiters_between_quotes(line, delimiter) for line in lines
    ]

    # Split the processed lines and clean up
    split_lines = [re.split(delimiter, line) for line in processed_lines]

    # Strip newlines and replace empty strings with NaN
    cleaned_lines = [
        [cell.strip("\n\r") if cell.strip("\n\r") != "" else np.nan for cell in line]
        for line in split_lines
    ]

    return cleaned_lines


def make_error_summary_df(
    file_path: Path,
    error_category: str,
    error_details: str,
    sheet_name: str = "0",
) -> pd.DataFrame:
    """Create a DataFrame summarizing the errors encountered during file processing.

    Parameters
    ----------
    file_path : Path
        The path to the file.
    error_category : str
        The category of error encountered.
    error_details : str
        Additional details about the error.
    sheet_name : str, optional with default "0"
        The name of the sheet. For CSV and TXT files, which do not have multiple sheets,
        the default value of "0" is used.

    Returns
    -------
    pd.DataFrame
        A DataFrame summarizing the errors.

    """
    return pd.DataFrame(
        {
            "record_name": file_path.parent.name,
            "file_name": file_path.name,
            "sheet_name": sheet_name.replace("-", "_"),
            "category": error_category,
            "details": error_details,
        },
        index=[0],
    )


def load_excel_sheet(
    file_path: Path,
    sheet_name: str,
    engine: constants.ExcelEngine,
) -> data_structures.SheetExtractionResult:
    """Load an Excel sheet and return a DataFrame with the required columns."""
    extracted_data_df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        engine=engine.value,
        parse_dates=False,
    )

    # Check that the DataFrame contains data
    if extracted_data_df.empty:
        return data_structures.SheetExtractionResult(
            extraction=make_error_summary_df(
                file_path=file_path,
                error_category="no_data",
                error_details="sheet has no data",
                sheet_name=sheet_name.replace("-", "_"),
            ),
            file_path=file_path,
            sheet_name=sheet_name.replace("-", "_"),
        )

    if extracted_data_df.shape[0] == 1:
        return data_structures.SheetExtractionResult(
            extraction=make_error_summary_df(
                file_path=file_path,
                error_category="only_one_line",
                error_details=f"has only one line with first cell of {extracted_data_df.iloc[0][0]}",
                sheet_name=sheet_name.replace("-", "_"),
            ),
            file_path=file_path,
            sheet_name=sheet_name.replace("-", "_"),
        )

    # data_type_count = info.count_num_str_float(
    #     extracted_data_df,
    # )

    # # Drop columns that have more text than numeric data
    # extracted_data_df = extracted_data_df.loc[
    #     :,
    #     data_type_count.numerical_surplus_per_col >= 0,
    # ]

    try:
        multi_row_header_indices = search.find_row_indices_of_header_lines(
            extracted_data_df,
        )
    except errors.FileProcessingError as e:
        return data_structures.SheetExtractionResult(
            extraction=make_error_summary_df(
                file_path=file_path,
                error_category=str(e).split("-")[0],
                error_details=str(e).split("-")[1].strip(),
                sheet_name=sheet_name.replace("-", "_"),
            ),
            file_path=file_path,
            sheet_name=sheet_name.replace("-", "_"),
        )

    if len(multi_row_header_indices) == 0:
        return data_structures.SheetExtractionResult(
            extraction=make_error_summary_df(
                file_path=file_path,
                error_category="no_header_row",
                error_details="sheet has no header row",
                sheet_name=sheet_name.replace("-", "_"),
            ),
            file_path=file_path,
            sheet_name=sheet_name.replace("-", "_"),
        )

    extracted_data_df, header_row_index = combine_multiple_header_rows(
        extracted_data_df,
        multi_row_header_indices,
    )

    #######################################################
    # Set dataframe's headers/column names. Note that .to_numpy() is used so that the
    # row's index in the DataFrame is not included in the header
    extracted_data_df.columns = extracted_data_df.iloc[header_row_index].to_numpy()

    # Make column names unique by adding suffixes to duplicates
    extracted_data_df = make_column_names_unique(extracted_data_df)

    possible_col_names_with_info = (
        search.remove_repeated_column_finds_across_all_params(
            search.search_line_for_all_needed_cells(
                list(extracted_data_df.columns),
            ),
        )
    )

    extracted_data_df = excel_skip_nondata_rows_at_start(
        extracted_data_df,
        header_row_index,
    )

    # Check that there are remaining data rows after skipping non-data rows
    if extracted_data_df.empty:
        return data_structures.SheetExtractionResult(
            extraction=make_error_summary_df(
                file_path=file_path,
                error_category="no_data",
                error_details="sheet has no data after skipping non-data rows",
                sheet_name=sheet_name.replace("-", "_"),
            ),
            file_path=file_path,
            sheet_name=sheet_name.replace("-", "_"),
        )

    # col_names = select_best_option_for_column_extraction(
    #     possible_col_names_with_info,
    #     extracted_data_df,
    #     file_path.name,
    #     sheet_name,
    # )

    # Make a copy that contains strings so that interruptions in the Depth column
    # can be identified.     # after this point to exclude different kinds
    # of data that is included at the bottom of some files
    # extracted_data_df_still_containing_strings = extracted_data_df.copy()

    #     ### Some records contain other data sets in the same file but in these cases, the Depth column is
    #     ### interrupted by a string, so we find the first row that contains a string in the Depth column
    #     ### and remove all rows after this point
    #     if final_col_names[0] is not None:
    #         extracted_data_df = data_structures.remove_nondata_rows_after_data(
    #             extracted_data_df,
    #             extracted_data_df_still_containing_strings,
    #             final_col_names,
    #         )

    return data_structures.SheetExtractionResult(
        extraction=data_structures.ExtractedDataAndColInfo(
            data_df=extracted_data_df,
            col_info=possible_col_names_with_info,
        ),
        file_path=file_path,
        sheet_name=sheet_name.replace("-", "_"),
    )


def safe_load_csv_or_txt(file_path: Path) -> data_structures.SheetExtractionResult:
    """Load CSV or TXT files that have a different number of columns in each row.

    The "safe" part of the name indicates that this function is designed to handle
    files with a different number of columns in each row, without raising an error,
    by only loading the columns that are present in every row.

    Parameters
    ----------
    file_path : Path
        The path to the CSV or TXT file.

    Returns
    -------
    data_structures.SheetExtractionResult
        The outcome of the attempted extraction

    Raises
    ------
    errors.FileProcessingError
        If the file is empty or if no header rows are found.

    """
    file_encoding = info.find_encoding(file_path)

    if file_encoding is None:
        return data_structures.SheetExtractionResult(
            extraction=make_error_summary_df(
                file_path=file_path,
                error_category="unknown_file_encoding",
                error_details="could not determine file encoding",
            ),
            file_path=file_path,
        )

    with Path.open(file_path, encoding=file_encoding) as file:
        content = file.read()
        if len(content) == 0:
            return data_structures.SheetExtractionResult(
                extraction=make_error_summary_df(
                    file_path=file_path,
                    error_category="no_data",
                    error_details=f"sheet ({file_path.name.replace('-', '_')}) has no "
                    "data",
                ),
                file_path=file_path,
            )

    delimiter = info.determine_delimiter(file_path, file_encoding)

    lines_and_cells_iterable = get_csv_or_txt_split_readlines(
        file_path,
        file_encoding,
        delimiter,
    )

    if len(lines_and_cells_iterable) == 0:
        return data_structures.SheetExtractionResult(
            extraction=make_error_summary_df(
                file_path=file_path,
                error_category="no_data",
                error_details=f"sheet ({file_path.name.replace('-', '_')}) has no data",
            ),
            file_path=file_path,
        )

    header_row_indices_in_csv_or_txt_file = search.find_row_indices_of_header_lines(
        lines_and_cells_iterable,
    )

    if len(header_row_indices_in_csv_or_txt_file) == 0:
        return data_structures.SheetExtractionResult(
            extraction=make_error_summary_df(
                file_path=file_path,
                error_category="no_header_row",
                error_details=f"no_header_row - sheet \
                               ({file_path.name.replace('-', '_')}) has no header row",
            ),
            file_path=file_path,
        )

    if len(header_row_indices_in_csv_or_txt_file) == 1:
        possible_col_names_with_info = search.search_line_for_all_needed_cells(
            lines_and_cells_iterable[header_row_indices_in_csv_or_txt_file[0]],
        )

    if len(header_row_indices_in_csv_or_txt_file) > 1:
        # combine multiple header rows into a single header row
        num_cols_in_header_rows = np.array(
            [
                len(lines_and_cells_iterable[header_row_indices_in_csv_or_txt_file[i]])
                for i in range(len(header_row_indices_in_csv_or_txt_file))
            ],
        )

        # Some header rows such as for CPT_59209 have two header rows with the
        # parameter name in the first row and the unit in the second row. However, often
        # a unit is not given for every parameter making it appear as though there are a
        # different number of columns in the header rows. The first several columns
        # typically correspond, and these are the ones we are interested in, so we only
        # loop over the header columns up to the minimum number of columns in the header
        # rows.

        lines_and_cells_iterable_handled_multi_header_rows = (
            lines_and_cells_iterable.copy()
        )

        for header_row_idx in header_row_indices_in_csv_or_txt_file[0:-1]:
            for header_col_idx in range(np.min(num_cols_in_header_rows)):
                ## Modify the lower header row to include the upper header row(s)
                lines_and_cells_iterable_handled_multi_header_rows[
                    header_row_indices_in_csv_or_txt_file[-1]
                ][header_col_idx] = (
                    f"{lines_and_cells_iterable[header_row_idx][header_col_idx]}_"
                    f"{lines_and_cells_iterable[header_row_indices_in_csv_or_txt_file[-1]][header_col_idx]}"
                )
        # replace the original lines_and_cells_iterable with the modified one
        lines_and_cells_iterable = lines_and_cells_iterable_handled_multi_header_rows

        # To identify the column types, we need to pass the header row into
        # search_line_for_all_needed_cells(). If there are multiple header rows,
        # all information is copied into the last header row, so we can just pass
        # the last header row into search_line_for_all_needed_cells() with index -1.
        # If there is only one header row, we can also use just use index -1.
        possible_col_names_with_info = search.search_line_for_all_needed_cells(
            lines_and_cells_iterable[header_row_indices_in_csv_or_txt_file[-1]],
        )

    highest_safe_column_index = len(
        lines_and_cells_iterable[header_row_indices_in_csv_or_txt_file[-1]],
    )

    loaded_as_df = False
    # Some .TXT and .CSV files have an inconsistent number of columns in each row,
    # which causes pd.read_csv() to raise a ParserError. Many inspections of several
    # of these problematic files show that the first several columns are consistent
    # across all rows, so we try to load the file with a decreasing number of columns
    # until it succeeds or we reach the minimum number of columns that we require.
    # The minimum number of columns is defined in constants.REQUIRED_NUMBER_OF_COLUMNS.
    # If the file cannot be loaded with the required number of columns an extraction
    # failure is recorded in the SheetExtractionResult object.
    while (
        not loaded_as_df
        and highest_safe_column_index > constants.REQUIRED_NUMBER_OF_COLUMNS
    ):
        try:
            extracted_df = pd.read_csv(
                file_path,
                header=None,
                encoding=file_encoding,
                sep=delimiter,
                # +1 to skiprows to convert from index to number of lines
                skiprows=header_row_indices_in_csv_or_txt_file[-1] + 1,
                usecols=list(range(highest_safe_column_index)),
            )
            loaded_as_df = True
        # if the file cannot be loaded with the current number of columns,
        # reduce the number of columns by 1 and try again
        except (pd.errors.ParserError, ValueError):
            highest_safe_column_index -= 1

    if not loaded_as_df:
        return data_structures.SheetExtractionResult(
            extraction=make_error_summary_df(
                file_path=file_path,
                error_category="unable_to_load_csv_or_txt",
                error_details="unable_to_load_csv_or_txt",
            ),
            file_path=file_path,
        )

    extracted_df.columns = lines_and_cells_iterable[
        header_row_indices_in_csv_or_txt_file[-1]
    ][0:highest_safe_column_index]

    # Make column names unique by adding suffixes to duplicates
    extracted_df = make_column_names_unique(extracted_df)

    # Update the column names now that inconsistent columns have been removed
    possible_col_names_with_info = search.search_line_for_all_needed_cells(
        extracted_df.columns,
    )

    return data_structures.SheetExtractionResult(
        extraction=data_structures.ExtractedDataAndColInfo(
            data_df=extracted_df,
            col_info=possible_col_names_with_info,
        ),
        file_path=file_path,
    )
