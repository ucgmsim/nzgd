"""Script to extract CPT data arrays from NZGD source files."""

import multiprocessing as mp
import os

import pandas as pd
from tqdm import tqdm

from nzgd import constants
from nzgd.extract.cpt import workflow
from natsort import natsorted

if __name__ == "__main__":
    nzgd_index_df = pd.read_csv(str(constants.INDEX_FILE_PATH))
    cpt_nzgd_ids = set(
        nzgd_index_df[nzgd_index_df["Type"] == "SCP"]["nzgd_id"].tolist()
    )

    downloaded_nzgd_paths = natsorted(list(constants.NZGD_SOURCE_DATA_DIR.glob("*")))

    cpt_paths = [
        path for path in downloaded_nzgd_paths if int(path.name) in cpt_nzgd_ids
    ]

    NUM_WORKERS = os.cpu_count()
    results = []

    with mp.Pool(processes=NUM_WORKERS) as pool:
        results.extend(
            list(
                tqdm(
                    pool.imap(workflow.process_one_record, cpt_paths),
                    total=len(cpt_paths),
                ),
            ),
        )
