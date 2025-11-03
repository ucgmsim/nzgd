"""
Estimate Vs30 from the SPT data in the database and store results back in the database.
"""

import sqlite3

import numpy as np
import pandas as pd
from tqdm import tqdm

import vs_calc
from nzgd import constants


def get_finite_value_with_fallback(
    sptreport_df: pd.DataFrame, column_name: str, borehole_id: int, default_value: float
) -> float:
    """Get a finite value from a column with fallback logic.

    Parameters
    ----------
    sptreport_df : pd.DataFrame
        DataFrame containing SPT report data
    column_name : str
        Name of the column to extract value from (e.g., 'extracted_gwl_m', 'efficiency')
    borehole_id : int
        The specific borehole_id to try first
    default_value : float
        Default value to use if no finite values found

    Returns
    -------
    float
        A finite value following the fallback hierarchy
    """
    # Create a copy of the dataframe and convert column to numeric, coercing errors to NaN
    df_numeric = sptreport_df.copy()
    df_numeric[column_name] = pd.to_numeric(sptreport_df[column_name], errors="coerce")

    # First try: get value for specific spt_id
    sptreport_df_for_bh_id = df_numeric[df_numeric["spt_id"] == borehole_id]
    if not sptreport_df_for_bh_id.empty:
        value_for_bh_id = sptreport_df_for_bh_id[column_name].iloc[0]
        if pd.notna(value_for_bh_id) and np.isfinite(value_for_bh_id):
            return value_for_bh_id

    # Second try: find any finite value in the entire dataframe
    finite_mask = df_numeric[column_name].notna() & np.isfinite(df_numeric[column_name])
    if finite_mask.any():
        return df_numeric.loc[finite_mask, column_name].iloc[0]

    # Third try: use default constant
    return default_value


def make_surface_layer(soil_measurements_for_bh_id_df: pd.DataFrame) -> pd.DataFrame:
    """Add a layer extending to the surface if the first layer does not start at the surface.

    Parameters
    ----------
    soil_measurements_for_bh_id_df : pandas.DataFrame
        DataFrame containing soil layers.
    Returns
    -------
    pandas.DataFrame
        DataFrame containing soil layers with a surface layer added if the first layer does not start at the surface.
    """
    if (
        not soil_measurements_for_bh_id_df.empty
        and soil_measurements_for_bh_id_df["top_depth_m"].iloc[0] != 0
    ):
        surface_row = soil_measurements_for_bh_id_df.iloc[0].copy()
        surface_row["top_depth_m"] = 0.0
        soil_measurements_for_bh_id_df = pd.concat(
            [pd.DataFrame([surface_row]), soil_measurements_for_bh_id_df],
            ignore_index=True,
        )
        soil_measurements_for_bh_id_df = soil_measurements_for_bh_id_df.sort_values(
            "top_depth_m"
        ).reset_index(drop=True)
    return soil_measurements_for_bh_id_df


# Soil type name to enum mapping for SPT correlations
# Only includes the four main soil types; other types default to Clay
SOIL_TYPE_NAME_TO_ENUM = {
    "CLAY": vs_calc.constants.SoilType.Clay,
    "SILT": vs_calc.constants.SoilType.Silt,
    "SAND": vs_calc.constants.SoilType.Sand,
    "GRAVEL": vs_calc.constants.SoilType.Gravel,
}


def map_soil_types_to_measurement_depths(
    measurements_df: pd.DataFrame, layers_df: pd.DataFrame
) -> np.ndarray:
    """Map soil types from layers to measurement depths.

    For each SPT measurement depth, find the corresponding soil layer and return
    the appropriate SoilType enum value.

    Parameters
    ----------
    measurements_df : pd.DataFrame
        DataFrame with 'depth' column containing SPT measurement depths
    layers_df : pd.DataFrame
        DataFrame with columns: top_depth_m, bottom_depth_m, soil_type_name

    Returns
    -------
    np.ndarray
        Array of vs_calc.constants.SoilType enum values, same length as measurements_df

    Notes
    -----
    - If a measurement depth falls outside all layers, defaults to SoilType.Clay
    - If a soil type name is not recognized, defaults to SoilType.Clay
    - Uses the first matching layer if multiple layers contain the same depth
    """
    if measurements_df.empty or layers_df.empty:
        # Return default (all Clay) if no data
        return np.repeat(vs_calc.constants.SoilType.Clay, len(measurements_df))

    soil_types = []
    for depth in measurements_df["depth_m"]:
        # Find layer containing this depth
        layer_mask = (layers_df["top_depth_m"] <= depth) & (
            depth < layers_df["bottom_depth_m"]
        )
        matching_layers = layers_df[layer_mask]

        if not matching_layers.empty:
            soil_name = matching_layers.iloc[0]["soil_type_name"].upper()
            soil_type = SOIL_TYPE_NAME_TO_ENUM.get(
                soil_name,
                vs_calc.constants.SoilType.Clay,  # default if not found
            )
        else:
            # No matching layer - use default
            soil_type = vs_calc.constants.SoilType.Clay

        soil_types.append(soil_type)

    return np.array(soil_types)


soil_type_unit_weights_df = pd.read_csv(constants.SOIL_TYPE_UNIT_WEIGHTS_PATH)

spt_vs_correlations = vs_calc.spt_vs_correlations.SPT_CORRELATIONS
vs30_correlations = list(vs_calc.vs30_correlations.VS30_CORRELATIONS.keys())

hammer_types = [vs_calc.constants.HammerType.Auto]  # ,
#     vs_calc.constants.HammerType.Safety,
#     vs_calc.constants.HammerType.Standard,
# ]

assumed_borehole_diameter = constants.DEFAULT_BOREHOLE_DIAMETER_mm

conn = sqlite3.connect(constants.OUTPUT_DB_PATH)
nzgd_ids = (
    pd.read_sql_query(
        "SELECT DISTINCT nzgd_id FROM nzgdrecord WHERE type_id = ? ORDER BY nzgd_id ASC",
        conn,
        params=(2,),  # 2 is the id for boreholes
    )
    .to_numpy()
    .flatten()
    .tolist()
)

spt_vs30_data = []

progress_bar = tqdm(total=len(nzgd_ids))

for nzgd_id in nzgd_ids:
    progress_bar.update()

    sptreport_df = pd.read_sql_query(
        "SELECT * FROM sptreport WHERE nzgd_id = ? ORDER BY spt_id ASC",
        conn,
        params=(nzgd_id,),
    )

    # Get the SPT measurements for the spt_ids corresonding to this nzgd_id
    sptmeasurements_df = pd.read_sql_query(
        """
        SELECT m.* 
        FROM sptmeasurements m
        INNER JOIN sptreport r ON m.spt_id = r.spt_id
        WHERE r.nzgd_id = ?
        ORDER BY m.spt_id, m.depth_m ASC
        """,
        conn,
        params=(nzgd_id,),
    )

    # Get the soil measurements for the spt_ids corresonding to this nzgd_id
    soil_measurements_df = pd.read_sql_query(
        """
        SELECT 
            sm.soil_measurement_id,
            sm.spt_id as spt_id,
            sm.top_depth_m,
            st.id as soil_type_id,
            st.value as soil_type_name
        FROM soilmeasurements sm
        INNER JOIN sptreport r ON sm.spt_id = r.spt_id
        INNER JOIN soilmeasurementsoiltype smst ON sm.soil_measurement_id = smst.soil_measurement_id
        INNER JOIN soiltypes st ON smst.soil_type_id = st.id
        WHERE r.nzgd_id = ?
        ORDER BY sm.spt_id, sm.top_depth_m ASC
        """,
        conn,
        params=(nzgd_id,),
    )

    spt_ids_with_spt_data = sptreport_df["spt_id"].unique().tolist()
    bh_ids_with_soil_types = soil_measurements_df["spt_id"].unique().tolist()

    soil_counts_by_borehole = soil_measurements_df.groupby("spt_id").size()

    for spt_id in spt_ids_with_spt_data:
        # Get ground water level with fallbacks
        extracted_gwl_for_bh_id = get_finite_value_with_fallback(
            sptreport_df,
            "extracted_gwl_m",
            spt_id,
            constants.DEFAULT_GROUNDWATER_LEVEL_m,
        )

        # Get efficiency with fallbacks
        efficiency_for_bh_id = get_finite_value_with_fallback(
            sptreport_df, "efficiency", spt_id, constants.DEFAULT_SPT_EFFICIENCY_PERCENT
        )

        if len(spt_ids_with_spt_data) < len(bh_ids_with_soil_types):
            spt_id_for_soil_types = soil_counts_by_borehole.idxmax()
        else:
            spt_id_for_soil_types = spt_id

        soil_measurements_for_bh_id_df = soil_measurements_df[
            soil_measurements_df["spt_id"] == spt_id_for_soil_types
        ]
        soil_measurements_for_bh_id_df = make_surface_layer(
            soil_measurements_for_bh_id_df
        )

        # Process soil layer data if available
        has_soil_data = not soil_measurements_for_bh_id_df.empty

        if has_soil_data:
            # Build layers DataFrame: layer_thickness_m, unsaturated_unit_weight_kN/m3, saturated_unit_weight_kN/m3
            try:
                # Compute bottom_depth_m and thickness
                layers_base_df = soil_measurements_for_bh_id_df.copy()
                layers_base_df["bottom_depth_m"] = (
                    layers_base_df["top_depth_m"]
                    .shift(-1)
                    .fillna(
                        sptmeasurements_df["depth_m"].max()
                        + constants.BUFFER_BELOW_LOWEST_MEASUREMENT_DEPTH_m
                        + constants.SPT_DEPTH_OFFSET_m
                    )
                )
                layers_base_df["layer_thickness_m"] = (
                    layers_base_df["bottom_depth_m"] - layers_base_df["top_depth_m"]
                )

                # Prepare merge keys (lowercase for robust matching)
                layers_base_df["_soil_key"] = layers_base_df[
                    "soil_type_name"
                ].str.lower()
                unit_weights_df = soil_type_unit_weights_df.copy()
                unit_weights_df["_soil_key"] = unit_weights_df["soil_type"].str.lower()

                merged_df = layers_base_df.merge(
                    unit_weights_df[
                        [
                            "unsaturated_unit_weight_kN/m3",
                            "saturated_unit_weight_kN/m3",
                            "_soil_key",
                        ]
                    ],
                    on="_soil_key",
                    how="left",
                )

                if (
                    merged_df[
                        ["unsaturated_unit_weight_kN/m3", "saturated_unit_weight_kN/m3"]
                    ]
                    .isna()
                    .any()
                    .any()
                ):
                    missing = merged_df[
                        merged_df["unsaturated_unit_weight_kN/m3"].isna()
                        | merged_df["saturated_unit_weight_kN/m3"].isna()
                    ]["soil_type_name"].unique()
                    raise ValueError(f"Missing unit weights for soil types: {missing}")

                layers_df = merged_df[
                    [
                        "top_depth_m",
                        "bottom_depth_m",
                        "soil_type_name",
                        "layer_thickness_m",
                    ]
                ].copy()
                layers_df["unsaturated_unit_weight_kN/m3"] = merged_df[
                    "unsaturated_unit_weight_kN/m3"
                ]
                layers_df["saturated_unit_weight_kN/m3"] = merged_df[
                    "saturated_unit_weight_kN/m3"
                ]

            except ValueError as e:
                print(f"Skipping SPT {spt_id} soil data: {e}")
                has_soil_data = False
                layers_df = pd.DataFrame()
        else:
            layers_df = pd.DataFrame()  # Empty dataframe

        # Get SPT measurements for this specific spt_id
        measurements_df = sptmeasurements_df[
            sptmeasurements_df["spt_id"] == spt_id
        ].copy()

        # Skip if no measurements
        if measurements_df.empty:
            continue

        # Map soil types to measurement depths (will default to Clay if no soil data)
        soil_types_array = map_soil_types_to_measurement_depths(
            measurements_df, layers_df
        )

        # Process with each correlation combination
        for spt_vs_correlation_name in spt_vs_correlations:
            # Determine if this is a layered correlation
            is_layered = "layered" in spt_vs_correlation_name

            # For non-layered: run with and without soil info
            # For layered: only run with soil info (always use layers + soil types)
            use_soil_info_values = (
                [True] if is_layered else [True, False] if has_soil_data else [False]
            )

            for use_soil_info in use_soil_info_values:
                # Skip if trying to use soil info but don't have soil data
                if use_soil_info and not has_soil_data:
                    continue
                for vs30_correlation in vs30_correlations:
                    for hammer_type in hammer_types:
                        # Use DataFrame layers for layered correlations - only include required columns
                        if is_layered and has_soil_data:
                            layers = layers_df[
                                [
                                    "layer_thickness_m",
                                    "unsaturated_unit_weight_kN/m3",
                                    "saturated_unit_weight_kN/m3",
                                ]
                            ].copy()
                        else:
                            layers = None

                        # Create SPT object
                        spt = vs_calc.SPT(
                            name=str(spt_id),
                            depth=measurements_df["depth_m"].to_numpy(),
                            n=measurements_df["n"].to_numpy(),
                            hammer_type=hammer_type,
                            borehole_diameter=assumed_borehole_diameter,
                            layers=layers,
                            groundwater_level=extracted_gwl_for_bh_id,
                        )

                        # Set soil types if using soil info
                        if use_soil_info:
                            spt.soil_type = soil_types_array
                        # Otherwise, leave as default (all Clay)

                        # Set efficiency if available
                        if efficiency_for_bh_id is not None:
                            energy_ratio = efficiency_for_bh_id / 100
                            spt.energy_ratio = energy_ratio
                            used_efficiency = True
                        else:
                            used_efficiency = False

                        # Calculate Vs profile and Vs30
                        try:
                            spt_vs_profile = vs_calc.VsProfile.from_spt(
                                spt, spt_vs_correlation_name
                            )
                            spt_vs_profile.vs30_correlation = vs30_correlation
                            vs30 = spt_vs_profile.vs30
                            vs30_sd = spt_vs_profile.vs30_sd
                            error = np.nan
                        except Exception as e:
                            vs30 = np.nan
                            vs30_sd = np.nan
                            error = e

                        # Store results if successful
                        if not isinstance(error, Exception):
                            spt_vs30_data.append(
                                (
                                    spt_id,
                                    constants.SPT_TO_VS_CORRELATION_TO_ID[
                                        spt_vs_correlation_name
                                    ],
                                    constants.VS_TO_VS30_CORRELATION_TO_ID[
                                        vs30_correlation
                                    ],
                                    assumed_borehole_diameter,
                                    constants.HAMMER_TYPE_TO_ID[hammer_type.name],
                                    int(used_efficiency),
                                    int(use_soil_info),
                                    vs30,
                                    vs30_sd,
                                )
                            )

progress_bar.close()
conn.close()

# Insert the results into the database
print(f"Processed {len(spt_vs30_data)} Vs30 estimates")

if spt_vs30_data:
    db_conn = sqlite3.connect(constants.OUTPUT_DB_PATH)
    cursor = db_conn.cursor()
    # executemany automatically assigns the vs30_id primary key so it is not specified
    cursor.executemany(
        """
        INSERT INTO sptvs30estimates (spt_id, spt_to_vs_correlation_id, vs_to_vs30_correlation_id, assumed_borehole_diameter_mm, assumed_hammer_type_id, estimate_used_extracted_efficiency, estimate_used_extracted_layer_soil_types, vs30, vs30_stddev)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        spt_vs30_data,
    )
    db_conn.commit()
    db_conn.close()
    print(f"Inserted {len(spt_vs30_data)} Vs30 estimates into database")
