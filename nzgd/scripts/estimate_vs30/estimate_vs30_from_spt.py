"""
Estimate Vs30 from the SPT data in the database and store results back in the database.
"""

import sqlite3
from pathlib import Path

import natsort
import numpy as np
from tqdm import tqdm

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


spt_vs_correlations = vs_calc.spt_vs_correlations.SPT_CORRELATIONS
vs30_correlations = list(vs_calc.vs30_correlations.VS30_CORRELATIONS.keys())

hammer_types = [
    vs_calc.constants.HammerType.Auto,
    vs_calc.constants.HammerType.Safety,
    vs_calc.constants.HammerType.Standard,
]

assumed_borehole_diameter = 150

# Example layer configuration for layer-based correlations
# TODO: This will be replaced with database-extracted layer data where:
#   - Layer thicknesses come from borehole/geological data
#   - Unit weights are determined based on soil type and depth relative to groundwater
#   - The split_layers_at_groundwater() function converts this to the simpler numpy array format
example_layers = [
    {"thickness": 2, "unit_weight": 17, "saturated_unit_weight": 18},
    {"thickness": 3, "unit_weight": 16, "saturated_unit_weight": 18}
]

borehole_ids = get_unique_borehole_ids(
    constants.OUTPUT_DB_PATH,
)

spt_vs30_data = []

progress_bar = tqdm(total=len(borehole_ids))

for borehole_id in borehole_ids:
    progress_bar.update()

    # retrieve.search_spt_reports() returns an iterator of retrieve.SPTReport objects.
    # When searching for a specific borehole_id, there will only be one object in the
    # iterator. To assign this object to a variable, we use a dummy for loop.
    for spt_search_result in retrieve.search_spt_reports(borehole_id=borehole_id):
        pass

    measurements_df = spt_search_result.measurements

    # Missing n values are empty strings in measurements_df so if any are found, skip this borehole
    if measurements_df["n_value"].apply(lambda x: isinstance(x, str)).any():
        # print(f"Skipping borehole {borehole_id} because of missing n values")
        continue

    # Also skip if any of the n values are None or nan
    if measurements_df["n_value"].apply(lambda x: isinstance(x, str)).any():
        # print(f"Skipping borehole {borehole_id} because of missing n values")
        continue

    soil_measurements_interval_tree = spt_search_result.soil_measurements

    ## Get soil types for each depth in measurements_df
    soil_type_list_enum = []
    for depth in measurements_df["depth"]:
        intervals = soil_measurements_interval_tree.at(depth)
        if intervals:
            soil_type_enum = list(intervals)[
                0
            ].data  # Take the first interval if multiple (unlikely)
            try:
                vs_soil_type = vs_calc.constants.SoilType[soil_type_enum.name.title()]
                soil_type_list_enum.append(vs_soil_type)
            except KeyError:
                continue  # Skip if not in vs_calc enum
        else:
            continue  # No soil type for this depth

    used_soil_info = len(soil_type_list_enum) == len(measurements_df)

    for spt_vs_correlation in spt_vs_correlations:
        for vs30_correlation in vs30_correlations:
            for hammer_type in hammer_types:
                # Use extracted groundwater level if available, otherwise use default
                groundwater_level = (
                    spt_search_result.extracted_gwl 
                    if spt_search_result.extracted_gwl is not None 
                    else constants.DEFAULT_GROUNDWATER_LEVEL
                )
                
                # Use layers for layered correlations - split at groundwater level
                if "layered" in spt_vs_correlation:
                    layers = split_layers_at_groundwater(example_layers, groundwater_level)
                else:
                    layers = None
                
                spt = vs_calc.SPT(
                    name=str(spt_search_result.borehole_id),
                    depth=measurements_df["depth"].to_numpy(),
                    n=measurements_df["n_value"].to_numpy(),
                    hammer_type=hammer_type,
                    borehole_diameter=assumed_borehole_diameter,
                    layers=layers,
                    groundwater_level=groundwater_level,
                )

                if used_soil_info:
                    spt.soil_type = np.array(soil_type_list_enum)

                used_efficiency = False
                efficiency = (spt_search_result.efficiency,)

                if isinstance(efficiency, tuple):
                    efficiency = efficiency[0]

                if efficiency is not None:
                    # Convert the efficiency percent to a ratio
                    energy_ratio = efficiency / 100

                    spt.energy_ratio = energy_ratio
                    used_efficiency = True

                try:
                    spt_vs_profile = vs_calc.VsProfile.from_spt(spt, spt_vs_correlation)
                    spt_vs_profile.vs30_correlation = vs30_correlation
                    vs30 = spt_vs_profile.vs30
                    vs30_sd = spt_vs_profile.vs30_sd
                    error = np.nan

                except Exception as e:
                    vs30 = np.nan
                    vs30_sd = np.nan
                    error = e

                if not isinstance(error, Exception):
                    spt_vs30_data.append(
                        (
                            borehole_id,
                            constants.SPT_TO_VS_CORRELATION_TO_ID[spt_vs_correlation],
                            constants.VS_TO_VS30_CORRELATION_TO_ID[vs30_correlation],
                            assumed_borehole_diameter,
                            constants.HAMMER_TYPE_TO_ID[hammer_type.name],
                            int(used_efficiency),
                            int(used_soil_info),
                            vs30,
                            vs30_sd,
                        )
                    )

progress_bar.close()

if spt_vs30_data:
    conn = sqlite3.connect(constants.OUTPUT_DB_PATH)
    cursor = conn.cursor()
    # executemany automatically assigns the vs30_id primary key so it is not specified
    cursor.executemany(
        """
        INSERT INTO sptvs30estimates (borehole_id, spt_to_vs_correlation_id, vs_to_vs30_correlation_id, assumed_borehole_diameter, assumed_hammer_type_id, estimate_used_extracted_efficiency, estimate_used_extracted_layer_soil_types, vs30, vs30_stddev)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        spt_vs30_data,
    )
    conn.commit()
    conn.close()
