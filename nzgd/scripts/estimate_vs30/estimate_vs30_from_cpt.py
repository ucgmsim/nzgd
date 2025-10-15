import functools
import multiprocessing as mp
import sqlite3
import time

import natsort as natsort
import numpy as np
import pandas as pd
from tqdm import tqdm

import vs_calc
from nzgd import constants


def calc_vs30(
    cpt_id: int,
    cpt_vs_correlation: str,
    vs30_correlation: str,
) -> pd.DataFrame:
    """Calculate Vs30 from a given CPT file.

    Parameters
    ----------
    file_path : Path
        The path to the CPT file in Parquet format.
    cpt_vs_correlation : str
        The correlation method to use for CPT to Vs conversion.
    vs30_correlation : str
        The correlation method to use for Vs30 calculation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Vs30 calculation results and metadata.

    """

    # Select all distinct cpt_id values from the cptmeasurements table, ordered ascending
    # Use a parameterized query to safely select all measurements for the given cpt_id
    # and order the results by the `depth` column ascending.

    with sqlite3.connect(constants.OUTPUT_DB_PATH) as conn:
        cpt_df = pd.read_sql_query(
            "SELECT * FROM cptmeasurements WHERE cpt_id = ? ORDER BY depth ASC",
            conn,
            params=(cpt_id,),
        )

    # Filter the DataFrame to include only the rows with the same multiple measurements index
    cpt_df = cpt_df[cpt_df["fs"] > 0]

    try:
        # Raise an error if there are no valid measurements
        if cpt_df.size == 0:
            raise ValueError("No valid measurements in the CPT")

        # Create a CPT object from the DataFrame
        cpt = vs_calc.CPT(
            str(cpt_id),
            cpt_df["depth"].to_numpy(),
            cpt_df["qc"].to_numpy(),
            cpt_df["fs"].to_numpy(),
            cpt_df["u2"].to_numpy(),
        )

        # Create a VsProfile from the CPT object using the specified correlation
        cpt_vs_profile = vs_calc.VsProfile.from_cpt(cpt, cpt_vs_correlation)
        cpt_vs_profile.vs30_correlation = vs30_correlation

        # Calculate Vs30 and its standard deviation
        calc_vs30 = cpt_vs_profile.vs30
        vs30_std = cpt_vs_profile.vs30_sd
        error = np.nan

    except Exception as e:
        # Handle any exceptions by setting Vs30 and its standard deviation to NaN
        calc_vs30 = np.nan
        vs30_std = np.nan
        error = e

    # Create a DataFrame with the Vs30 calculation results and metadata
    return pd.DataFrame(
        {
            "cpt_id": cpt_id,
            "processing_error": error,
            "vs30": calc_vs30,
            "vs30_stddev": vs30_std,
            "vs_to_vs30_correlation_id": constants.VS_TO_VS30_CORRELATION_TO_ID[
                vs30_correlation
            ],
            "cpt_to_vs_correlation_id": constants.CPT_TO_VS_CORRELATION_TO_ID[
                cpt_vs_correlation
            ],
        },
        index=[0],
    )


if __name__ == "__main__":
    start_time = time.time()

    ## The code to calculate calc_vs30 from CPT data (vs_calc) produces tens of thousands of divide by zero and invalid
    ## value warnings that are suppressed.
    np.seterr(divide="ignore", invalid="ignore")

    with sqlite3.connect(constants.OUTPUT_DB_PATH) as conn:
        # Select all distinct cpt_id values from the cptmeasurements table, ordered ascending
        cpt_ids = (
            pd.read_sql_query(
                "SELECT DISTINCT cpt_id FROM cptmeasurements ORDER BY cpt_id ASC", conn
            )
            .to_numpy()
            .flatten()
            .tolist()
        )

    cpt_ids = cpt_ids[0:2]

    cpt_vs_correlations = [
        "andrus_2007_pleistocene",
        "andrus_2007_holocene",
        "andrus_2007_tertiary_age_cooper_marl",
        "robertson_2009",
        "hegazy_2006",
        "mcgann_2015",
        "mcgann_2018",
    ]

    cpt_vs_correlations = [
        "andrus_2007_pleistocene",
        # "mcgann_2015",
    ]

    # vs30_correlations = ["boore_2004", "boore_2011"]
    vs30_correlations = ["boore_2004"]

    # cpt_vs_correlations = ["andrus_2007_pleistocene", "mcgann_2018"]
    # vs30_correlations = ["boore_2004", "boore_2011"]
    # cpt_vs_correlations = ["andrus_2007_pleistocene"]
    # vs30_correlations = ["boore_2004"]

    results = []
    for cpt_vs_correlation in cpt_vs_correlations:
        for vs30_correlation in vs30_correlations:
            description_text = (
                f"Calculating Vs30 using {vs30_correlation} and {cpt_vs_correlation}"
            )
            print(description_text)

            calc_vs30_from_filename_partial = functools.partial(
                calc_vs30,
                cpt_vs_correlation=cpt_vs_correlation,
                vs30_correlation=vs30_correlation,
            )
            num_workers = 8
            with mp.Pool(processes=num_workers) as pool:
                results.extend(
                    list(
                        tqdm(
                            pool.imap(calc_vs30_from_filename_partial, cpt_ids),
                            total=len(cpt_ids),
                        ),
                    ),
                )

    vs30_df = pd.concat(results, ignore_index=True)
    vs30_df = vs30_df.dropna(subset=["vs30"]).reset_index(drop=True)

    conn = sqlite3.connect(constants.OUTPUT_DB_PATH)

    cpt_id_to_nzgd_id = pd.read_sql_query("SELECT cpt_id, nzgd_id FROM cptreport", conn)

    vs30_df = vs30_df.merge(cpt_id_to_nzgd_id, on="cpt_id", how="left")

    for_sqlite = []

    for _, row in vs30_df.iterrows():
        for_sqlite.append(
            (
                int(row["cpt_id"]),
                int(row["nzgd_id"]),
                int(row["cpt_to_vs_correlation_id"]),
                int(row["vs_to_vs30_correlation_id"]),
                float(row["vs30"]),
                float(row["vs30_stddev"]),
            )
        )

    cursor = conn.cursor()

    # executemany automatically assigns the vs30_id primary key so it is not specified
    cursor.executemany(
        """
        INSERT INTO cptvs30estimates (cpt_id, nzgd_id, cpt_to_vs_correlation_id, vs_to_vs30_correlation_id, vs30, vs30_stddev)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        for_sqlite,
    )
    conn.commit()
    conn.close()

    print()
