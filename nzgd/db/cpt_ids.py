import multiprocessing as mp
from pathlib import Path

import pandas as pd
from natsort import index_natsorted, natsorted
from tqdm import tqdm

from nzgd import constants

extracted_files = natsorted(list(constants.CPT_TRACE_OUTPUT_DIR.glob("*.parquet")))


def extracted_cpt_trace_summary_for_file(file_path: str) -> pd.DataFrame:
    """Get the extraction information for a single parquet file.

    Parameters
    ----------
    file_path : str
        Path to the parquet file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['nzgd_id', 'file_name', 'sheet_name'] for this file.
    """
    file = Path(file_path)
    nzgd_id = int(file.stem)

    extracted_cpt_trace_per_nzgd_df = pd.read_parquet(file)

    file_names = extracted_cpt_trace_per_nzgd_df["file_name"].unique().tolist()
    sheet_names = extracted_cpt_trace_per_nzgd_df["sheet_name"].unique().tolist()

    rows = []
    for file_name in file_names:
        for sheet_name in sheet_names:
            rows.append(
                {"nzgd_id": nzgd_id, "file_name": file_name, "sheet_name": sheet_name}
            )

    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame(columns=["nzgd_id", "file_name", "sheet_name"])


def extracted_cpt_trace_summary():
    """Process all extracted CPT parquet files in parallel and return a summary DataFrame.

    Uses a multiprocessing pool with number of processes equal to available CPUs
    and collects per-file summaries into one DataFrame.
    """
    nproc = mp.cpu_count()
    extracted_data_summary_df = pd.DataFrame()

    with mp.Pool(processes=nproc) as pool:
        # map over file paths (string) for safe cross-process passing
        for result_df in tqdm(
            pool.imap(
                extracted_cpt_trace_summary_for_file, [str(p) for p in extracted_files]
            ),
            total=len(extracted_files),
        ):
            if not result_df.empty:
                extracted_data_summary_df = pd.concat(
                    [extracted_data_summary_df, result_df], ignore_index=True
                )

    # ensure nzgd_id is sorted in natural numeric order
    if not extracted_data_summary_df.empty:
        order = index_natsorted(extracted_data_summary_df["nzgd_id"].astype(str))
        extracted_data_summary_df = extracted_data_summary_df.loc[order].reset_index(
            drop=True
        )

    return extracted_data_summary_df


def assign():
    """Assign CPT IDs to extracted CPT data and supplemental values.

    Returns
    -------
    pd.DataFrame
        DataFrame with assigned CPT IDs and merged supplemental values.
    """

    print(
        "Assigning CPT IDs Part 1 of 2: summarizing extraction source keys "
        "(not deduplicating CPT trace content)..."
    )
    extracted_cpt_trace_summary_df = extracted_cpt_trace_summary()
    if not extracted_cpt_trace_summary_df.empty:
        order = index_natsorted(extracted_cpt_trace_summary_df["nzgd_id"].astype(str))
        extracted_cpt_trace_summary_df = extracted_cpt_trace_summary_df.loc[
            order
        ].reset_index(drop=True)

    extracted_cpt_trace_summary_df.loc[:, "nzgd_id_AND_filename_AND_sheetname"] = (
        extracted_cpt_trace_summary_df["nzgd_id"].astype(str)
        + "_AND_"
        + extracted_cpt_trace_summary_df["file_name"]
        + "_AND_"
        + extracted_cpt_trace_summary_df["sheet_name"]
    )

    extracted_cpt_trace_summary_df = extracted_cpt_trace_summary_df[
        ["nzgd_id_AND_filename_AND_sheetname", "nzgd_id"]
    ]

    extracted_supplemental_values_df = pd.read_csv(
        constants.SUPPLEMENTAL_VALUES_OUTPUT_DIR
        / constants.CPT_SUPPLEMENTAL_VALUES_FILENAME,
    )

    extracted_supplemental_values_df.loc[:, "nzgd_id_AND_filename_AND_sheetname"] = (
        extracted_supplemental_values_df["nzgd_id"].astype(str)
        + "_AND_"
        + extracted_supplemental_values_df["file_name"]
        + "_AND_"
        + extracted_supplemental_values_df["sheet_name"]
    )

    extracted_supplemental_values_df = extracted_supplemental_values_df.drop(
        columns=["file_name", "sheet_name"],
    )

    # Outer merge on 'filename_and_sheetname'
    merged_df = pd.merge(
        extracted_cpt_trace_summary_df,
        extracted_supplemental_values_df,
        on="nzgd_id_AND_filename_AND_sheetname",
        how="outer",
        suffixes=("_extracted", "_single_values"),
    )

    # Sort merged_df by 'nzgd_id_AND_filename_AND_sheetname' using natsort in reverse order
    merged_df = merged_df.loc[
        index_natsorted(merged_df["nzgd_id_AND_filename_AND_sheetname"])
    ].reset_index(drop=True)

    no_corresponding_cpt_extraction_indices = (
        merged_df["nzgd_id_single_values"].notna()
        & merged_df["nzgd_id_extracted"].isna()
    )

    # Collapse only duplicate rows that arise in the supplemental/key merge phase.
    # This does not perform content-level deduplication of CPT trace records.
    merged_df_single_values_no_corresponding_cpt_extraction = merged_df[
        no_corresponding_cpt_extraction_indices
    ].drop_duplicates(
        keep="first",
    )

    extracted_single_values_to_check = merged_df[
        ~no_corresponding_cpt_extraction_indices
    ]

    nzgd_ids_to_check_for_duplicates = extracted_single_values_to_check[
        "nzgd_id_extracted"
    ].unique()

    checked_rows = []

    print(
        "Assigning CPT IDs Part 2 of 2: Identifying unique supplemental value extractions..."
    )
    for nzgd_id in tqdm(nzgd_ids_to_check_for_duplicates):
        extracted_subset_df = extracted_single_values_to_check[
            extracted_single_values_to_check["nzgd_id_extracted"] == nzgd_id
        ]

        single_value_subset_df = extracted_subset_df[
            extracted_subset_df["nzgd_id_single_values"] == nzgd_id
        ]

        if len(single_value_subset_df) > len(extracted_subset_df):
            # there may be duplicated single values extractions that should be dropped

            single_value_subset_df = single_value_subset_df.drop_duplicates(
                keep="first",
            )
            checked_rows.append(single_value_subset_df)

        else:
            checked_rows.append(extracted_subset_df)

    checked_rows_df = pd.concat(checked_rows, ignore_index=True)

    deduplicated_single_values_df = pd.concat(
        [merged_df_single_values_no_corresponding_cpt_extraction, checked_rows_df],
        ignore_index=True,
    )

    # Sort merged_df by 'nzgd_id_AND_filename_AND_sheetname' using natsort in reverse order
    deduplicated_single_values_df = deduplicated_single_values_df.loc[
        index_natsorted(
            deduplicated_single_values_df["nzgd_id_AND_filename_AND_sheetname"],
        )
    ].reset_index(drop=True)

    nzgd_id_np_array = (
        deduplicated_single_values_df["nzgd_id_AND_filename_AND_sheetname"]
        .str.split("_AND_")
        .str[0]
        .astype(int)
        .to_numpy()
    )

    deduplicated_single_values_df.loc[:, "nzgd_id"] = nzgd_id_np_array

    deduplicated_single_values_df.loc[:, "cpt_id"] = (
        deduplicated_single_values_df.index.to_numpy() + 1
    )

    deduplicated_single_values_df = deduplicated_single_values_df[
        [
            "cpt_id",
            "nzgd_id",
            "termination_reason",
            "ground_water_level",
            "gwl_method",
            "tip_net_area_ratio",
            "predrill_depth",
            "nzgd_id_AND_filename_AND_sheetname",
            "nzgd_id_extracted",
            "nzgd_id_single_values",
        ]
    ]

    return deduplicated_single_values_df
