"""A simple script to run Jake Faulkner's borehole data extraction code."""

import sqlite3

import pandas as pd

from nzgd import constants
from nzgd.extract.bh import ags_miner, miner

nzgd_index_df = pd.read_csv(constants.INDEX_FILE_PATH, low_memory=False) # low_memory=False prevents warnings about columns of mixed data types 

nzgd_index_df = nzgd_index_df[
    nzgd_index_df["TypeDisplay"].isin(constants.NZGD_TypeDisplay_VALUES_FOR_BOREHOLES)
]

pdf_files = []
ags_files = []

for nzgd_id in nzgd_index_df["nzgd_id"]:
    available_files = list((constants.NZGD_SOURCE_DATA_DIR / str(nzgd_id)).glob("*"))

    for f in available_files:
        if f.suffix.lower() == ".pdf":
            pdf_files.append((nzgd_id, f))
        elif f.suffix.lower() == ".ags":
            ags_files.append((nzgd_id, f))

ags_miner.mine_borehole_log(ags_files, constants.TEMP_SPT_AGS_DB_PATH)
miner.mine_borehole_log(pdf_files, constants.TEMP_SPT_PDF_DB_PATH)

