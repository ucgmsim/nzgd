"""Sample model GeoTIFFs at NZTM points, mapping nodata/out-of-bounds to NaN.

Replaces the duplicated sampling code formerly in put_nzgd_metadata.py. The
legacy index baked nodata sentinels (e.g. -32767) into its model columns;
this module never lets a sentinel through.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from nzgd import constants

MODEL_COLUMNS = [
    "model_gwl_westerhoff_2018_m",
    "model_gwl_nlm_2025_m",
    "model_gwl_nlm_2025_stddev_m",
    "model_vs30_foster_2019_m_per_s",
    "model_vs30_stddev_foster_2019_ln",
]


def sample_band(raster_path: Path, xy_pairs: list, band: int = 1) -> list:
    """Sample one raster band at (easting, northing) points.

    Parameters
    ----------
    raster_path : Path
        GeoTIFF path (CRS must match the point coordinates, EPSG:2193 here).
    xy_pairs : list
        Sequence of (x, y) = (easting, northing) tuples with finite values.
    band : int
        1-based band index.

    Returns
    -------
    list
        One float per point; NaN for nodata, masked, or out-of-bounds samples.
    """
    values = []
    with rasterio.open(raster_path) as ds:
        nodata = ds.nodatavals[band - 1]
        for sample in ds.sample(xy_pairs, indexes=[band], masked=True):
            value = sample[0]
            if np.ma.is_masked(value):
                values.append(np.nan)
                continue
            value = float(value)
            if not np.isfinite(value) or (nodata is not None and value == nodata):
                values.append(np.nan)
            else:
                values.append(value)
    return values


def sample_model_columns(nztm_df: pd.DataFrame) -> pd.DataFrame:
    """Sample all five model columns for a frame with `nztm_x`/`nztm_y`.

    Rows with missing NZTM coordinates get NaN in every model column. Raster
    paths come from `nzgd.constants` (config.yaml).

    Returns
    -------
    pd.DataFrame
        The five MODEL_COLUMNS, aligned to `nztm_df.index`.
    """
    out = pd.DataFrame(index=nztm_df.index, columns=MODEL_COLUMNS, dtype=float)
    valid = nztm_df["nztm_x"].notna() & nztm_df["nztm_y"].notna()
    if not valid.any():
        return out
    xy = list(zip(nztm_df.loc[valid, "nztm_x"], nztm_df.loc[valid, "nztm_y"]))
    out.loc[valid, "model_gwl_westerhoff_2018_m"] = sample_band(
        constants.WESTERHOFF_2018_MODEL_PATH, xy, band=1
    )
    out.loc[valid, "model_gwl_nlm_2025_m"] = sample_band(constants.NLM_GWD_PATH, xy, band=1)
    out.loc[valid, "model_gwl_nlm_2025_stddev_m"] = sample_band(
        constants.NLM_GW_STD_PATH, xy, band=1
    )
    out.loc[valid, "model_vs30_foster_2019_m_per_s"] = sample_band(
        constants.FOSTER_2019_VS30_MODEL_PATH, xy, band=1
    )
    out.loc[valid, "model_vs30_stddev_foster_2019_ln"] = sample_band(
        constants.FOSTER_2019_VS30_MODEL_PATH, xy, band=2
    )
    return out
