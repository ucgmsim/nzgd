"""Script to load NZGD data using chunked batch processing.

This script implements a chunked batch processing approach to avoid SQLite
concurrency issues. It processes records in batches using multiprocessing
for data extraction, then writes each batch to the database sequentially.
"""

import multiprocessing as mp

import natsort
from tqdm import tqdm

from nzgd import constants
from nzgd.extract.cpt import workflow

if __name__ == "__main__":
    # Collect all records to extract
    records_to_extract = []

    for investigation_type in constants.INVESTIGATION_TYPES:
        nzgd_source_data_dir = constants.NZGD_SOURCE_DATA_DIR / investigation_type
        records_to_extract.extend(
            natsort.natsorted(list(nzgd_source_data_dir.glob("*"))),
        )

    # id_to_do = "_24449"
    # records_to_extract2 = []
    # for x in records_to_extract:
    #     path_as_str = str(x)

    #     if id_to_do in path_as_str:
    #         records_to_extract2.append(x)

    # records_to_extract = records_to_extract2

    # db_path = constants.OUTPUT_DIRECTORY / "nzgd_cpt_and_scpt_data.db"

    # if Path(db_path).exists():
    #     db_path.unlink()

    # A small number of records have been removed from the NZGD after their
    # source files were downloaded.  These records were likely removed for a reason
    # such data quality or permission issues, so they are removed from the
    # list to extract.
    records_currently_in_nzgd = set(constants.NZGD_INDEX_DF["ID"].values)
    records_that_have_been_removed = set(records_to_extract) - records_currently_in_nzgd

    if len(records_that_have_been_removed) > 0:
        records_to_extract = [
            record_dir
            for record_dir in records_to_extract
            if record_dir.name not in records_that_have_been_removed
        ]

    # records_to_extract = records_to_extract[0:1]
    # print()

    NUM_WORKERS = 6
    results = []

    with mp.Pool(processes=NUM_WORKERS) as pool:
        results.extend(
            list(
                tqdm(
                    pool.imap(workflow.process_one_record, records_to_extract),
                    total=len(records_to_extract),
                ),
            ),
        )
