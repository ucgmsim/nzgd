"""A simple script to run Jake Faulkner's borehole data extraction code."""

from pathlib import Path

from nzgd import constants

# from nzgd.extract.bh import miner
from nzgd.extract.bh import ags_miner

pdf_dir = Path(
    "/home/arr65/data/nzgd/downloads_and_metadata/borehole_files_by_type/pdf",
)

ags_dir = Path(
    "/home/arr65/data/nzgd/downloads_and_metadata/borehole_files_by_type/ags"
)

# miner.mine_borehole_log(pdf_dir, constants.OUTPUT_DB_PATH)
ags_miner.mine_borehole_log(ags_dir, constants.OUTPUT_DB_PATH)
