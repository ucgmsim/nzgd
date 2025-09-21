"""Output functions for handling extracted geotechnical data."""

import pandas as pd

from nzgd import constants
from nzgd.extract.cpt import (
    data_structures,
)


def write_extracted_data(
    extractions: list[data_structures.SheetExtractionResult],
) -> None:
    """Write the extracted data to parquet files.

    Parameters
    ----------
    extractions : list[data_structures.SheetExtractionResult]
        A list of SheetExtractionResult objects containing the extracted data and
        failed extractions.

    Returns
    -------
    None

    """
    # file_path is initialized to Path(), which has the string representation "." so
    # file_path is only valid if its string representation has length > 1
    for extraction in extractions:
        if len(str(extraction.file_path)) > 1:
            first_valid_path = extraction.file_path
            break

    # Add columns to indicate the source file and sheet for each row of data
    for extraction_result_index, extraction_result in enumerate(extractions):
        if isinstance(
            extraction_result.extraction,
            data_structures.ExtractedDataAndColInfo,
        ):
            extractions[extraction_result_index].extraction.data_df.loc[
                :,
                "file_name",
            ] = extraction_result.file_path.name

            extractions[extraction_result_index].extraction.data_df.loc[
                :,
                "sheet_name",
            ] = extraction_result.sheet_name

            # concatenate the list of removed duplicates into a single string
            if extraction_result.removed_duplicates:
                extraction_result.removed_duplicates = "___".join(
                    extraction_result.removed_duplicates,
                )

            extractions[extraction_result_index].extraction.data_df.loc[
                :,
                "removed_duplicates",
            ] = extraction_result.removed_duplicates

            extractions[extraction_result_index].extraction.data_df.loc[
                :,
                "explicit_unit_conversions",
            ] = extraction_result.explicit_unit_conversions

            extractions[extraction_result_index].extraction.data_df.loc[
                :,
                "inferred_unit_conversions",
            ] = extraction_result.inferred_unit_conversions

        else:
            extractions[extraction_result_index].extraction.loc[
                :,
                "file_name",
            ] = extraction_result.file_path.name

            extractions[extraction_result_index].extraction.loc[
                :,
                "sheet_name",
            ] = extraction_result.sheet_name

    successful_extraction_dfs = [
        extraction_result.extraction.data_df
        for extraction_result in extractions
        if isinstance(
            extraction_result.extraction,
            data_structures.ExtractedDataAndColInfo,
        )
    ]

    failed_extraction_dfs = [
        extraction_result.extraction
        for extraction_result in extractions
        if not isinstance(
            extraction_result.extraction,
            data_structures.ExtractedDataAndColInfo,
        )
    ]

    # Add an index to the DataFrames to indicate that the data comes from different
    # measurements
    for investigation_number in range(len(successful_extraction_dfs)):
        successful_extraction_dfs[investigation_number].loc[
            :,
            "investigation_number",
        ] = investigation_number

    for failed_extraction_index in range(len(failed_extraction_dfs)):
        failed_extraction_dfs[failed_extraction_index].loc[
            :,
            "failed_extraction_index",
        ] = failed_extraction_index

    extracted_data_per_record_output_path = (
        constants.OUTPUT_DIRECTORY / "extracted_data_per_record"
    )
    extraction_failures_per_record_output_path = (
        constants.OUTPUT_DIRECTORY / "extraction_failures_per_record"
    )

    extracted_data_per_record_output_path.mkdir(exist_ok=True, parents=True)
    extraction_failures_per_record_output_path.mkdir(exist_ok=True, parents=True)

    # If there are any dataframes of extracted data or failed extractions,
    # save them to parquet files
    if len(successful_extraction_dfs) > 0:
        all_extracted_data = pd.concat(successful_extraction_dfs)
        all_extracted_data.to_parquet(
            extracted_data_per_record_output_path
            / f"{first_valid_path.parent.name.split('_')[1]}.parquet",
        )

    if len(failed_extraction_dfs) > 0:
        all_failed_extractions = pd.concat(failed_extraction_dfs)
        all_failed_extractions.to_parquet(
            extraction_failures_per_record_output_path
            / f"{first_valid_path.parent.name.split('_')[1]}.parquet",
        )
