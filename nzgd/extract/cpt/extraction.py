from pathlib import Path

import pandas as pd
from python_ags4 import AGS4

from nzgd import constants
from nzgd.extract.cpt import data_structures, errors, info, search, tasks


def get_file_list_for_record(record_dir: Path) -> data_structures.FilePathList:
    """Get a list of all data files for a given record.

    Parameters
    ----------
    record_dir : Path
        The directory containing the data files for a given NZGD record.

    Returns
    -------
    FileList
        An object with attributes for each file type containing lists of Paths.

    """
    all_files = list(record_dir.glob("*"))

    return data_structures.FilePathList(
        spreadsheets=[
            f
            for f in all_files
            if f.suffix.lower() in constants.FILE_EXTENSIONS["spreadsheets"]
        ],
        ags=[
            f for f in all_files if f.suffix.lower() in constants.FILE_EXTENSIONS["ags"]
        ],
        nondata=[
            f
            for f in all_files
            if f.suffix.lower() in constants.FILE_EXTENSIONS["nondata"]
        ],
    )


def extract_all_data_for_one_record(
    record_dir: Path,
) -> list[data_structures.SheetExtractionResult]:
    """Try to extract all data for a single record.

    Parameters
    ----------
    record_dir : Path
        The directory containing the data files for a given NZGD record.
    extraction_failures_per_record_output_path : Optional[Path]
        The location to save information about failed extractions in a file named after
        the record.

    Returns
    -------
    list[data_structures.SheetExtractionResult]
        A list of extraction results for each sheet in the record's data files.

    """
    paths = get_file_list_for_record(record_dir)

    # Check for the presence of data files
    if (len(paths.spreadsheets) + len(paths.ags)) == 0:
        if (len(paths.spreadsheets) + len(paths.ags) + len(paths.nondata)) == 0:
            # no files at all
            error_as_string = "no_files - no files in the record directory"
        else:
            # some files but no data files
            error_as_string = "no_data_files - no data files (may contain pdf and cpt)"

            return [
                data_structures.SheetExtractionResult(
                    extraction=pd.DataFrame(
                        {
                            "record_name": record_dir.name,
                            "file_name": None,
                            "sheet_name": None,
                            "category": error_as_string.split("-")[0].strip(),
                            "details": error_as_string.split("-")[1].strip(),
                        },
                        index=[0],
                    ),
                    file_path=record_dir / "placeholder_for_no_data_files.txt",
                ),
            ]

    preliminary_sheet_extractions = []

    for file_to_try in paths.spreadsheets:
        preliminary_sheet_extractions.extend(
            extract_sheets_in_file(
                file_to_try,
            ),
        )

    for file_to_try in paths.ags:
        preliminary_sheet_extractions.extend(extract_ags(file_to_try))

    return preliminary_sheet_extractions


def extract_sheets_in_file(
    file_path: Path,
) -> list[data_structures.SheetExtractionResult]:
    """Load the results of a Cone Penetration Test (CPT) from a spreadsheet file.

    This function can be used for the following types of spreadsheet files:
    .xls, .xlsx, .csv, .txt

    Parameters
    ----------
    file_path : Path
        The path to the spreadsheet file (e.g., .xls, .xlsx, .csv, .txt).

    Returns
    -------
    data_structures.ExtractionResultsForFile

    """
    if "_" not in file_path.name:
        return [
            data_structures.SheetExtractionResult(
                extraction=tasks.make_error_summary_df(
                    file_path=file_path,
                    error_category="bad_file_name_format",
                    error_details=(
                        f"{file_path.name} does not contain an NZGD format file name"
                    ),
                ),
                file_path=file_path,
            ),
        ]

    record_name = f"{file_path.name.split('_')[0]}_{file_path.name.split('_')[1]}"

    if record_name in constants.KNOWN_SPECIAL_CASES:
        return [
            data_structures.SheetExtractionResult(
                extraction=tasks.make_error_summary_df(
                    file_path=file_path,
                    error_category=constants.KNOWN_SPECIAL_CASES[record_name]
                    .split("-")[0]
                    .strip(),
                    error_details=constants.KNOWN_SPECIAL_CASES[record_name]
                    .split("-")[1]
                    .strip(),
                ),
                file_path=file_path,
            ),
        ]

    if file_path.suffix.lower() in [".csv", ".txt"]:
        # CSV and TXT files do not have multiple sheets, so there will only be one
        # successful extraction or failure per file. These single outcomes are put
        # in a list of length 1 so they can be treated in the same way as the
        # list of extractions from an XLS/XLSX file that has sheets.
        try:
            return [tasks.safe_load_csv_or_txt(file_path)]

        except errors.FileProcessingError as e:
            return [
                data_structures.SheetExtractionResult(
                    extraction=tasks.make_error_summary_df(
                        file_path=file_path,
                        error_category="unknown_category",
                        error_details=str(e),
                    ),
                    file_path=file_path,
                ),
            ]

    elif file_path.suffix.lower() in [".xls", ".xlsx"]:
        # If the Excel file has no sheets, immediately return
        # a failed extraction summary in a list of length 1 so it can be treated
        # in the same way as the extractions from a normal XLS/XLSX file that has
        # several sheets
        try:
            sheet_names, _ = tasks.get_xls_sheet_names(file_path)

        except errors.InvalidExcelFileError as e:
            return [
                data_structures.SheetExtractionResult(
                    extraction=tasks.make_error_summary_df(
                        file_path=file_path,
                        error_category=str(e).split("-")[0].strip(),
                        error_details=str(e).split("-")[1].strip(),
                    ),
                    file_path=file_path,
                ),
            ]

        # Attempt to extract every sheet in the XLS/XLSX file
        _, engine = tasks.get_xls_sheet_names(file_path)
        return [
            tasks.load_excel_sheet(
                file_path,
                sheet_name,
                engine,
            )
            for sheet_name in sheet_names
        ]

    raise errors.UnsupportedInputFileTypeError


def extract_ags(
    file_path: Path,
) -> list[data_structures.SheetExtractionResult]:
    """Load an AGS file.

    Parameters
    ----------
    file_path : Path
        The path to the AGS file.

    Returns
    -------
    list[data_structures.SheetExtractionResult]
        The CPT data from the AGS file.

    """
    try:
        # Access AGS with: tables, headings = AGS4.AGS4_to_dataframe(file_path)
        tables, _ = AGS4.AGS4_to_dataframe(file_path)
    except UnboundLocalError:
        # Found the meaning of this UnboundLocalError by uploading one of these AGS
        # files to the AGS file conversion tool on https://agsapi.bgs.ac.uk
        return [
            data_structures.SheetExtractionResult(
                extraction=tasks.make_error_summary_df(
                    file_path=file_path,
                    error_category="ags_duplicate_headers",
                    error_details="AGS file contains duplicate headers",
                ),
                file_path=file_path,
            ),
        ]

    if len(tables) == 0:
        return [
            data_structures.SheetExtractionResult(
                extraction=tasks.make_error_summary_df(
                    file_path=file_path,
                    error_category="no_ags_data_tables",
                    error_details="no data tables found in the AGS file",
                ),
                file_path=file_path,
            ),
        ]

    ## Check that the SCPT table is present in the ags file
    try:
        tables["SCPT"]
    except KeyError:
        return [
            data_structures.SheetExtractionResult(
                extraction=tasks.make_error_summary_df(
                    file_path=file_path,
                    error_category="ags_missing_table",
                    error_details="AGS file is missing the required SCPT table",
                ),
                file_path=file_path,
            ),
        ]

    ## Check if any required columns are missing from the ags file
    required_ags_column_names = ["SCPT_DPTH", "SCPT_RES", "SCPT_FRES", "SCPT_PWP2"]
    for required_column_name in required_ags_column_names:
        if required_column_name not in tables["SCPT"].columns:
            return [
                data_structures.SheetExtractionResult(
                    extraction=tasks.make_error_summary_df(
                        file_path=file_path,
                        error_category="ags_missing_columns",
                        error_details=(
                            f"AGS file is missing {required_column_name} "
                            "(and possibly other columns)"
                        ),
                    ),
                    file_path=file_path,
                ),
            ]

    ## The first two data rows contain units and the number of decimal places
    ## For example:
    #     Depth      qc      fs    u
    # 0       m     MPa     MPa  MPa
    # 1     2DP     3DP     4DP  4DP

    # Construct column names by from the column descriptions and units so the units
    # will be picked up by the data conditioning tasks
    extracted_data_df = pd.DataFrame(
        {
            list(constants.COLUMN_DESCRIPTIONS)[0]
            + " "
            + str(tables["SCPT"]["SCPT_DPTH"][0]): tables["SCPT"]["SCPT_DPTH"],
            list(constants.COLUMN_DESCRIPTIONS)[1]
            + " "
            + str(tables["SCPT"]["SCPT_RES"][0]): tables["SCPT"]["SCPT_RES"],
            list(constants.COLUMN_DESCRIPTIONS)[2]
            + " "
            + str(tables["SCPT"]["SCPT_FRES"][0]): tables["SCPT"]["SCPT_FRES"],
            list(constants.COLUMN_DESCRIPTIONS)[3]
            + " "
            + str(tables["SCPT"]["SCPT_PWP2"][0]): tables["SCPT"]["SCPT_PWP2"],
        },
    )

    extracted_data_df = extracted_data_df.iloc[2:]

    # Convert string values to floats where possible
    for col in extracted_data_df.columns:
        for idx in extracted_data_df.index:
            value = extracted_data_df.loc[idx, col]
            if isinstance(value, str) and info.can_convert_str_to_float(value):
                extracted_data_df.loc[idx, col] = float(value)

    return [
        data_structures.SheetExtractionResult(
            extraction=data_structures.ExtractedDataAndColInfo(
                data_df=extracted_data_df,
                col_info=search.search_line_for_all_needed_cells(
                    extracted_data_df.columns,
                ),
            ),
            file_path=file_path,
        ),
    ]
