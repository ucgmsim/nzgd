"""Extract ground water levels from borehole AGS files
by searching for the WSTG (Water Strike - General) values.
See https://open-geotechnical.github.io/unofficial-ags4-data-dict/group/WSTG.html
"""

import enum

import natsort
import numpy as np
import pandas as pd
from tqdm import tqdm

from nzgd import constants
from nzgd.extract.cpt import info


class NewSectionStartFlag(enum.StrEnum):
    """Enum to indicate the type of AGS file being processed."""

    double_asterix = "**"
    double_quote_group = '"GROUP",'


def get_ground_water_levels(
    new_section_start_flag: NewSectionStartFlag,
    ags_content: str,
) -> list[float]:
    """Extract the ground water levels from the WSTG section of an AGS file.

    Parameters
    ----------
    new_section_start_flag : NewSectionStartFlag
        Enum value that indicates the type of AGS file being processed.
    ags_content : str
        The content of the AGS file as a string.

    Returns
    -------
    wstg_dpth_values : list
        List of ground water levels in the WSTG section of the AGS file.

    """
    sections = ags_content.split(new_section_start_flag)

    ### The WSTG section is the section that contains the ground water levels.
    ### Find the WSTG section.
    ### If WSTG is not recorded, the "HEADING" section can contain the string WSTG
    ### but there is no data. That case is caught below.
    wstg_section = None
    for section in sections:
        if "WSTG" in section:
            wstg_section = section
            break

    ### If the WSTG section is not found, return an empty list.
    if wstg_section is None:
        return []

    listed_lines = wstg_section.split("\n")
    split_lines = [line.split(",") for line in listed_lines]

    ## There are two types of AGS files that I refer to as NewSectionStartFlag.double_quote_group which seperates
    ## sections with '"GROUP",' and NewSectionStartFlag.double_asterix which seperates sections with '**'.
    ## For both types, the second line (index 1) contains the column names.

    depth_column_idx = None
    for column_idx, column_name in enumerate(split_lines[1]):
        if "WSTG_DPTH" in column_name:
            depth_column_idx = column_idx
            break

    ## Catching the case where "WSTG" is present in the "HEADER" section but there is no WSTG data.
    if depth_column_idx is None:
        return []

    wstg_dpth_values = []
    for line in split_lines[3:]:
        ## Type NewSectionStartFlag.double_asterix has the first data row at index 3,
        ## but type NewSectionStartFlag.double_quote_group has the first data row at index 4.
        ## To simplify the implementation, both are treated the same but the ValueError
        ## produced by trying to convert NewSectionStartFlag.double_quote_group metadata string on row index 3
        ## to a float is caught and ignored.

        ## Additionally, at the end of both sections, there are blank lines (with only one row cell) that will raise
        ## an IndexError when trying to access the depth column. This is also caught and ignored.

        try:
            wstg_dpth_values.append(float(line[depth_column_idx].strip('"')))
        except (IndexError, ValueError):
            continue

    return wstg_dpth_values


if __name__ == "__main__":
    ### Borehole records are identified from the NZGD index by TypeDisplay, then each
    ### record's source-file directory is searched for .ags files (same approach as
    ### extract_spt.py). The previous hard-coded "unorganised_raw_from_nzgd/borehole"
    ### directory no longer exists after the source data was reorganised by NZGD ID.
    ### Borehole .ags files whose NZGD ID is absent from the index are skipped by design
    ### (the index is the source of truth for which records to process).
    nzgd_index_df = pd.read_csv(constants.INDEX_FILE_PATH, low_memory=False)
    borehole_index_df = nzgd_index_df[
        nzgd_index_df["TypeDisplay"].isin(
            constants.NZGD_TypeDisplay_VALUES_FOR_BOREHOLES,
        )
    ]

    ags_files = []
    for nzgd_id in borehole_index_df["nzgd_id"]:
        for source_file in (constants.NZGD_SOURCE_DATA_DIR / str(nzgd_id)).glob("*"):
            if source_file.suffix.lower() == ".ags":
                ags_files.append(source_file)
    ags_files = natsort.natsorted(ags_files)

    unknown_file_type = []
    results = []

    for ags_file in tqdm(ags_files):
        ground_water_levels_for_file = []

        encoding = info.find_encoding(ags_file)

        with open(ags_file, encoding=encoding) as f:
            content = f.read()
        file_type = None

        if NewSectionStartFlag.double_asterix in content:
            ground_water_levels_for_file.extend(
                get_ground_water_levels(NewSectionStartFlag.double_asterix, content),
            )
            file_type = NewSectionStartFlag.double_asterix
        elif NewSectionStartFlag.double_quote_group in content:
            ground_water_levels_for_file.extend(
                get_ground_water_levels(NewSectionStartFlag.double_quote_group, content),
            )
            file_type = NewSectionStartFlag.double_quote_group
        else:
            unknown_file_type.append(ags_file)

        df_from_file = pd.DataFrame(
            {
                "file_name": ags_file.name,
                "type": file_type.value if file_type is not None else None,
                "median_water_strike_depth": None,
                "water_strike_depths": None,
            },
            index=[0],
        )

        if len(ground_water_levels_for_file) > 0:
            df_from_file.at[0, "median_water_strike_depth"] = np.median(
                ground_water_levels_for_file,
            )
            df_from_file.at[0, "water_strike_depths"] = ground_water_levels_for_file

        results.append(df_from_file)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_parquet(
        "/home/arr65/data/nzgd/extracted_single_values/borehole_extracted_ground_water_levels.parquet",
    )
