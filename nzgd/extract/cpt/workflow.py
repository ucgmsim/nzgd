from pathlib import Path

from nzgd.extract.cpt import (
    conditioning,
    extraction,
    output,
    select_columns,
    tasks,
    validation,
)


def process_one_record(
    record_dir: Path,
) -> None:
    """Manage the main flow of the data extraction process.

    Parameters
    ----------
    record_dir : Path
        The directory containing the data files for a given NZGD record.
    investigation_type : cpt_data_extraction_helpers.InvestigationType
        The type of investigation being processed (CPT or SCPT)
    extracted_data_per_record_output_path : Optional[Path]
        The location to save the extracted data in a file named after the record.
    extraction_failures_per_record_output_path : Optional[Path]
        The location to save information about failed extractions in a file named after
        the record.

    Returns
    -------
    None

    """

    sheet_extractions = extraction.extract_all_data_for_one_record(
        record_dir=record_dir,
    )

    sheet_extractions = tasks.apply_func_to_all_sheets(
        validation.identify_missing_columns_in_sheet,
        sheet_extractions,
    )

    sheet_extractions = tasks.apply_func_to_all_sheets(
        conditioning.remove_non_numerical_data_for_one_sheet,
        sheet_extractions,
    )

    sheet_extractions = tasks.apply_func_to_all_sheets(
        select_columns.select_columns_for_one_sheet,
        sheet_extractions,
    )

    validated_extractions = tasks.apply_func_to_all_sheets(
        validation.validate_initial_extraction_of_sheet,
        sheet_extractions,
    )

    conditioned_extractions = tasks.apply_func_to_all_sheets(
        conditioning.data_conditioning_for_one_sheet,
        validated_extractions,
    )

    extractions_no_duplicates = conditioning.remove_duplicate_extractions(
        conditioned_extractions,
    )

    output.write_extracted_data(extractions_no_duplicates)
