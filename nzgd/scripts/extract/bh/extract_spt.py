"""A simple script to run Jake Faulkner's borehole data extraction code."""

import sqlite3

import pandas as pd

from nzgd import constants
from nzgd.extract.bh import ags_miner, miner

# Jake applied a filtering as some incorrect extractions are not in his database.
# Jake's filtering criteria are not known, the extraction code and input data have
# not changed, so we only keep extractions that are also in his database.

# jake_conn = sqlite3.connect("/home/arr65/Downloads/jake_geodata.db")
# jake_sptreport_df = pd.read_sql_query("SELECT * FROM sptreport", jake_conn)
# jake_conn.close()
# jake_extracted_borehole_ids = jake_sptreport_df["borehole_id"].tolist()

nzgd_index_df = pd.read_csv(constants.INDEX_FILE_PATH, low_memory=False) # low_memory=False prevents warnings about columns of mixed data types 

# nzgd_index_df = nzgd_index_df[
#     nzgd_index_df["TypeDisplay"].isin(constants.NZGD_TypeDisplay_VALUES_FOR_BOREHOLES)
#     & nzgd_index_df["nzgd_id"].isin(jake_extracted_borehole_ids)
# ]

pdf_files = []
ags_files = []

for nzgd_id in nzgd_index_df["nzgd_id"]:
    available_files = list((constants.NZGD_SOURCE_DATA_DIR / str(nzgd_id)).glob("*"))

    for f in available_files:
        if f.suffix.lower() == ".pdf":
            pdf_files.append(f)
        elif f.suffix.lower() == ".ags":
            ags_files.append(f)

ags_miner.mine_borehole_log(ags_files, constants.TEMP_SPT_AGS_DB_PATH)
miner.mine_borehole_log(pdf_files, constants.TEMP_SPT_PDF_DB_PATH)

