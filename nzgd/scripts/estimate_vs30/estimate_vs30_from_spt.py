"""
Estimate Vs30 from the SPT data in the database and store results back in the database.
"""

import sqlite3
from pathlib import Path

import natsort
import numpy as np
from tqdm import tqdm
import pandas as pd

import vs_calc
from nzgd import constants
from nzgd.db import retrieve


def get_unique_borehole_ids(db_path: Path) -> list[int]:
    """
    Get unique borehole IDs from the SPT report table.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.

    Returns
    -------
    list[int]
        Sorted list of unique borehole IDs.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute the query to get unique borehole_id values
    cursor.execute("SELECT DISTINCT borehole_id FROM sptreport")
    unique_borehole_ids = cursor.fetchall()

    # Close the connection
    conn.close()

    # Extract the borehole_id values from the result
    unique_borehole_ids = [row[0] for row in unique_borehole_ids]

    return natsort.natsorted(unique_borehole_ids)


def split_layers_at_groundwater(
    layers: list[dict], groundwater_level: float
) -> np.ndarray:
    """
    Split soil layers at the groundwater level, assigning appropriate unit weights.
    
    Layers above the groundwater level use unsaturated unit_weight, while layers
    below use saturated_unit_weight. If a layer is intersected by the groundwater
    level, it is split into two sub-layers.
    
    Parameters
    ----------
    layers : list[dict]
        List of layer dictionaries, each containing:
        - 'thickness' (float): Layer thickness in meters
        - 'unit_weight' (float): Unsaturated unit weight in kN/m³
        - 'saturated_unit_weight' (float): Saturated unit weight in kN/m³
    groundwater_level : float
        Depth to groundwater level from surface in meters.
        
    Returns
    -------
    np.ndarray
        Array of shape (n_layers, 2) where each row is [thickness, unit_weight].
        Layers are ordered from top to bottom.
        
    Notes
    -----
    Edge cases:
    - If groundwater_level <= 0: all layers use saturated_unit_weight
    - If groundwater_level is deeper than all layers: all layers use unit_weight
    - Zero-thickness layers are preserved (though unusual)
    
    Examples
    --------
    >>> layers = [
    ...     {"thickness": 2, "unit_weight": 17, "saturated_unit_weight": 19},
    ...     {"thickness": 3, "unit_weight": 16, "saturated_unit_weight": 18}
    ... ]
    >>> split_layers_at_groundwater(layers, groundwater_level=3.5)
    array([[2.0, 17.0],   # Layer 1: entirely above GWL
           [1.5, 16.0],   # Layer 2 upper: above GWL
           [1.5, 18.0]])  # Layer 2 lower: below GWL
    """
    if not layers:
        return np.array([]).reshape(0, 2)
    
    split_layers_list = []
    cumulative_depth = 0.0
    
    for layer in layers:
        thickness = layer["thickness"]
        unit_weight = layer["unit_weight"]
        saturated_unit_weight = layer["saturated_unit_weight"]
        
        layer_top = cumulative_depth
        layer_bottom = cumulative_depth + thickness
        
        # Case 1: Layer entirely above groundwater level
        if layer_bottom <= groundwater_level:
            split_layers_list.append([thickness, unit_weight])
        
        # Case 2: Layer entirely below groundwater level
        elif layer_top >= groundwater_level:
            split_layers_list.append([thickness, saturated_unit_weight])
        
        # Case 3: Groundwater level intersects this layer - split it
        else:
            # Upper part (above groundwater)
            thickness_above = groundwater_level - layer_top
            if thickness_above > 0:
                split_layers_list.append([thickness_above, unit_weight])
            
            # Lower part (below groundwater)
            thickness_below = layer_bottom - groundwater_level
            if thickness_below > 0:
                split_layers_list.append([thickness_below, saturated_unit_weight])
        
        cumulative_depth = layer_bottom
    
    return np.array(split_layers_list)

def get_finite_value_with_fallback(
    sptreport_df: pd.DataFrame,
    column_name: str,
    borehole_id: int,
    default_value: float
) -> float:
    """Get a finite value from a column with fallback logic.
    
    Parameters
    ----------
    sptreport_df : pd.DataFrame
        DataFrame containing SPT report data
    column_name : str
        Name of the column to extract value from (e.g., 'extracted_gwl', 'efficiency')
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
    df_numeric[column_name] = pd.to_numeric(sptreport_df[column_name], errors='coerce')
    
    # First try: get value for specific borehole_id
    sptreport_df_for_bh_id = df_numeric[df_numeric["borehole_id"] == borehole_id]

    value_for_bh_id = sptreport_df_for_bh_id[column_name].iloc[0]
    if pd.notna(value_for_bh_id) and np.isfinite(value_for_bh_id):
        return value_for_bh_id
    
    # Second try: find any finite value in the entire dataframe
    finite_mask = df_numeric[column_name].notna() & np.isfinite(df_numeric[column_name])
    if finite_mask.any():
        return sptreport_df_for_bh_id.loc[finite_mask, column_name].iloc[0]
    
    # Third try: use default constant
    return default_value


def split_layers_and_assign_unit_weights(
    soil_measurements_df: pd.DataFrame,
    groundwater_level: float,
    soil_type_unit_weights_df: pd.DataFrame
) -> pd.DataFrame:
    """Split soil layers at groundwater level and assign appropriate unit weights.
    
    This function processes soil measurement data to:
    1. Add bottom_depth for each layer
    2. Split layers that span the groundwater level
    3. Assign unit weights based on saturation state
    
    Parameters
    ----------
    soil_measurements_df : pd.DataFrame
        DataFrame with columns: measurement_id, borehole_id, top_depth, 
        soil_type_id, soil_type_name
    groundwater_level : float
        Depth to groundwater level from surface in meters
    soil_type_unit_weights_df : pd.DataFrame
        Reference dataframe with columns: soil_type, unsaturated_unit_weight_kN/m3,
        saturated_unit_weight_kN/m3
        
    Returns
    -------
    pd.DataFrame
        Enhanced dataframe with additional columns: bottom_depth, unit_weight, is_saturated
        
    Raises
    ------
    ValueError
        If a soil type is not found in soil_type_unit_weights_df
    """
    if soil_measurements_df.empty:
        return soil_measurements_df
    
    # Create a copy to avoid modifying the original
    df = soil_measurements_df.copy()
    
    # Add bottom_depth column (next layer's top_depth, or 100m for the last layer)
    df['bottom_depth'] = df['top_depth'].shift(-1).fillna(100.0)
    
    # Create a list to store split layers
    split_layers = []
    
    for idx, row in df.iterrows():
        top_depth = row['top_depth']
        bottom_depth = row['bottom_depth']
        
        # Check if groundwater level falls within this layer
        if top_depth < groundwater_level < bottom_depth:
            # Split into two layers
            # Layer 1: from top_depth to groundwater_level (unsaturated)
            layer_above = row.copy()
            layer_above['bottom_depth'] = groundwater_level
            split_layers.append(layer_above)
            
            # Layer 2: from groundwater_level to bottom_depth (saturated)
            layer_below = row.copy()
            layer_below['top_depth'] = groundwater_level
            layer_below['bottom_depth'] = bottom_depth
            split_layers.append(layer_below)
        else:
            # No split needed
            split_layers.append(row)
    
    # Convert back to DataFrame
    result_df = pd.DataFrame(split_layers).reset_index(drop=True)
    
    # Add unit weights based on saturation state
    result_df['is_saturated'] = result_df['top_depth'] >= groundwater_level
    
    # Create a lookup dictionary for unit weights (convert to lowercase for matching)
    unit_weight_lookup = {}
    for _, uw_row in soil_type_unit_weights_df.iterrows():
        soil_type = uw_row['soil_type'].lower()
        unit_weight_lookup[soil_type] = {
            'unsaturated': uw_row['unsaturated_unit_weight_kN/m3'],
            'saturated': uw_row['saturated_unit_weight_kN/m3']
        }
    
    # Assign unit weights
    unit_weights = []
    for _, row in result_df.iterrows():
        soil_type_lower = row['soil_type_name'].lower()
        
        if soil_type_lower not in unit_weight_lookup:
            raise ValueError(f"Soil type '{row['soil_type_name']}' not found in soil_type_unit_weights_df")
        
        if row['is_saturated']:
            unit_weight = unit_weight_lookup[soil_type_lower]['saturated']
        else:
            unit_weight = unit_weight_lookup[soil_type_lower]['unsaturated']
        
        unit_weights.append(unit_weight)
    
    result_df['unit_weight'] = unit_weights
    
    return result_df


def convert_layers_df_to_numpy(layers_df: pd.DataFrame) -> np.ndarray:
    """Convert layer DataFrame to numpy array format for refactored Jaehwi calculation.
    
    Parameters
    ----------
    layers_df : pd.DataFrame
        DataFrame with columns: top_depth, bottom_depth, unit_weight
        
    Returns
    -------
    np.ndarray
        Array of shape (n_layers, 2) with columns [thickness, unit_weight]
        Format expected by jaehwi_calculate_effective_stress_refactored()
    """
    if layers_df.empty:
        return np.array([])
    
    thickness = layers_df['bottom_depth'] - layers_df['top_depth']
    unit_weight = layers_df['unit_weight']
    
    return np.column_stack([thickness, unit_weight])


def convert_layers_df_to_dict_list(layers_df: pd.DataFrame) -> list[dict]:
    """Convert layer DataFrame to dict list format for original Jaehwi calculation.
    
    Parameters
    ----------
    layers_df : pd.DataFrame
        DataFrame with columns: top_depth, bottom_depth, unit_weight
        
    Returns
    -------
    list[dict]
        List of dictionaries with keys: thickness, unit_weight, saturated_unit_weight
        Format expected by jaehwi_calculate_effective_stress()
        
    Notes
    -----
    Since layers are already split at the groundwater level with correct weights,
    both unit_weight and saturated_unit_weight are set to the same value.
    """
    if layers_df.empty:
        return []
    
    layers_list = []
    for _, row in layers_df.iterrows():
        thickness = row['bottom_depth'] - row['top_depth']
        unit_weight = row['unit_weight']
        
        layers_list.append({
            'thickness': thickness,
            'unit_weight': unit_weight,
            'saturated_unit_weight': unit_weight  # Same as unit_weight since already split
        })
    
    return layers_list


# Soil type name to enum mapping for SPT correlations
# Only includes the four main soil types; other types default to Clay
SOIL_TYPE_NAME_TO_ENUM = {
    'CLAY': vs_calc.constants.SoilType.Clay,
    'SILT': vs_calc.constants.SoilType.Silt,
    'SAND': vs_calc.constants.SoilType.Sand,
    'GRAVEL': vs_calc.constants.SoilType.Gravel,
}


def map_soil_types_to_measurement_depths(
    measurements_df: pd.DataFrame,
    layers_df: pd.DataFrame
) -> np.ndarray:
    """Map soil types from layers to measurement depths.
    
    For each SPT measurement depth, find the corresponding soil layer and return
    the appropriate SoilType enum value.
    
    Parameters
    ----------
    measurements_df : pd.DataFrame
        DataFrame with 'depth' column containing SPT measurement depths
    layers_df : pd.DataFrame
        DataFrame with columns: top_depth, bottom_depth, soil_type_name
        
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
    for depth in measurements_df['depth']:
        # Find layer containing this depth
        layer_mask = (layers_df['top_depth'] <= depth) & (depth < layers_df['bottom_depth'])
        matching_layers = layers_df[layer_mask]
        
        if not matching_layers.empty:
            soil_name = matching_layers.iloc[0]['soil_type_name'].upper()
            soil_type = SOIL_TYPE_NAME_TO_ENUM.get(
                soil_name, 
                vs_calc.constants.SoilType.Clay  # default if not found
            )
        else:
            # No matching layer - use default
            soil_type = vs_calc.constants.SoilType.Clay
        
        soil_types.append(soil_type)
    
    return np.array(soil_types)


soil_type_unit_weights_df = pd.read_csv(constants.SOIL_TYPE_UNIT_WEIGHTS_PATH)

spt_vs_correlations = vs_calc.spt_vs_correlations.SPT_CORRELATIONS
vs30_correlations = list(vs_calc.vs30_correlations.VS30_CORRELATIONS.keys())

# hammer_types = [
#     vs_calc.constants.HammerType.Auto,
#     vs_calc.constants.HammerType.Safety,
#     vs_calc.constants.HammerType.Standard,
# ]

hammer_types = [vs_calc.constants.HammerType.Auto]#,
#     vs_calc.constants.HammerType.Safety,
#     vs_calc.constants.HammerType.Standard,
# ]

assumed_borehole_diameter = 150

conn = sqlite3.connect(constants.OUTPUT_DB_PATH)
nzgd_ids = pd.read_sql_query(
    "SELECT DISTINCT nzgd_id FROM nzgdrecord WHERE type_id = ? ORDER BY nzgd_id ASC",
    conn,
    params=(2,), # 2 is the id for boreholes
).to_numpy().flatten().tolist()

# a small subset of nzgd_ids for testing
nzgd_ids = nzgd_ids[:10]

spt_vs30_data = []

progress_bar = tqdm(total=len(nzgd_ids))

for nzgd_id in nzgd_ids:
    progress_bar.update()

    sptreport_df = pd.read_sql_query(
        "SELECT * FROM sptreport WHERE nzgd_id = ? ORDER BY borehole_id ASC",
        conn,
        params=(nzgd_id,),
    )

    # Get the SPT measurements for the borehole_ids corresonding to this nzgd_id
    sptmeasurements_df = pd.read_sql_query(
        """
        SELECT m.* 
        FROM sptmeasurements m
        INNER JOIN sptreport r ON m.borehole_id = r.borehole_id
        WHERE r.nzgd_id = ?
        ORDER BY m.borehole_id, m.depth ASC
        """,
        conn,
        params=(nzgd_id,)
    )

    # Get the soil measurements for the borehole_ids corresonding to this nzgd_id
    soil_measurements_df = pd.read_sql_query(
        """
        SELECT 
            sm.measurement_id,
            sm.report_id as borehole_id,
            sm.top_depth,
            st.id as soil_type_id,
            st.name as soil_type_name
        FROM soilmeasurements sm
        INNER JOIN sptreport r ON sm.report_id = r.borehole_id
        INNER JOIN soilmeasurementsoiltype smst ON sm.measurement_id = smst.soil_measurement_id
        INNER JOIN soiltypes st ON smst.soil_type_id = st.id
        WHERE r.nzgd_id = ?
        ORDER BY sm.report_id, sm.top_depth ASC
        """,
        conn,
        params=(nzgd_id,)
    )

    bh_ids_with_spt_data = sptreport_df["borehole_id"].unique().tolist()
    bh_ids_with_soil_types = soil_measurements_df["borehole_id"].unique().tolist()

    soil_counts_by_borehole = soil_measurements_df.groupby("borehole_id").size()    

    for bh_id in bh_ids_with_spt_data:
        
        # Get ground water level with fallbacks
        extracted_gwl_for_bh_id = get_finite_value_with_fallback(
            sptreport_df,
            'extracted_gwl',
            bh_id,
            constants.DEFAULT_GROUNDWATER_LEVEL
        )

        # Get efficiency with fallbacks
        efficiency_for_bh_id = get_finite_value_with_fallback(
            sptreport_df,
            'efficiency',
            bh_id,
            constants.DEFAULT_SPT_EFFICIENCY
        )

        if len(bh_ids_with_spt_data) < len(bh_ids_with_soil_types):
            bh_id_for_soil_types = soil_counts_by_borehole.idxmax()
        else:
            bh_id_for_soil_types = bh_id

        soil_measurements_for_bh_id_df = soil_measurements_df[soil_measurements_df["borehole_id"] == bh_id_for_soil_types]
        
        # Add a layer extending to the surface if the first layer is below the surface
        if not soil_measurements_for_bh_id_df.empty and soil_measurements_for_bh_id_df['top_depth'].iloc[0] != 0:
            # Get the first row to duplicate
            first_row = soil_measurements_for_bh_id_df.iloc[0].copy()
            
            # Create a new row with top_depth = 0
            surface_row = first_row.copy()
            surface_row['top_depth'] = 0.0
            
            # Add the surface row at the beginning
            soil_measurements_for_bh_id_df = pd.concat([
                pd.DataFrame([surface_row]),  # Convert single row to DataFrame
                soil_measurements_for_bh_id_df
            ], ignore_index=True)
            
            # Sort by top_depth to ensure proper ordering
            soil_measurements_for_bh_id_df = soil_measurements_for_bh_id_df.sort_values('top_depth').reset_index(drop=True)
        
        # Process soil layer data if available
        has_soil_data = not soil_measurements_for_bh_id_df.empty
        
        if has_soil_data:
            # Split layers at groundwater level and assign unit weights
            try:
                layers_df = split_layers_and_assign_unit_weights(
                    soil_measurements_for_bh_id_df,
                    extracted_gwl_for_bh_id,
                    soil_type_unit_weights_df
                )
            except ValueError as e:
                print(f"Skipping borehole {bh_id} soil data: {e}")
                has_soil_data = False
                layers_df = pd.DataFrame()  # Empty dataframe
        else:
            layers_df = pd.DataFrame()  # Empty dataframe
        
        # Convert to numpy format for layered correlations
        layers_numpy = convert_layers_df_to_numpy(layers_df) if has_soil_data else np.array([])
        
        # Convert to dict list format if needed for validation/testing
        # layers_dict_list = convert_layers_df_to_dict_list(layers_df)
        
        # Get SPT measurements for this specific borehole
        measurements_df = sptmeasurements_df[sptmeasurements_df["borehole_id"] == bh_id].copy()
        
        # Skip if no measurements
        if measurements_df.empty:
            continue
        
        # Map soil types to measurement depths (will default to Clay if no soil data)
        soil_types_array = map_soil_types_to_measurement_depths(measurements_df, layers_df)
        
        # Process with each correlation combination
        for spt_vs_correlation_name in spt_vs_correlations:
            # Determine if this is a layered correlation
            is_layered = "layered" in spt_vs_correlation_name
            
            # For non-layered: run with and without soil info
            # For layered: only run with soil info (always use layers + soil types)
            use_soil_info_values = [True] if is_layered else [True, False] if has_soil_data else [False]
            
            for use_soil_info in use_soil_info_values:
                # Skip if trying to use soil info but don't have soil data
                if use_soil_info and not has_soil_data:
                    continue
                for vs30_correlation in vs30_correlations:
                    for hammer_type in hammer_types:
                        # Use layers for layered correlations
                        layers = layers_numpy if is_layered else None
                        
                        # Create SPT object
                        spt = vs_calc.SPT(
                            name=str(bh_id),
                            depth=measurements_df["depth"].to_numpy(),
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
                            spt_vs_profile = vs_calc.VsProfile.from_spt(spt, spt_vs_correlation_name)
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
                                    bh_id,
                                    constants.SPT_TO_VS_CORRELATION_TO_ID[spt_vs_correlation_name],
                                    constants.VS_TO_VS30_CORRELATION_TO_ID[vs30_correlation],
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
        INSERT INTO sptvs30estimates (borehole_id, spt_to_vs_correlation_id, vs_to_vs30_correlation_id, assumed_borehole_diameter, assumed_hammer_type_id, estimate_used_extracted_efficiency, estimate_used_extracted_layer_soil_types, vs30, vs30_stddev)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        spt_vs30_data,
    )
    db_conn.commit()
    db_conn.close()
    print(f"Inserted {len(spt_vs30_data)} Vs30 estimates into database")
