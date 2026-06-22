"""Tests for GeoTIFF sampling with nodata handling."""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from nzgd.metadata.rasters import sample_band


def _write_tif(path, nodata=-32767.0):
    """A 10x10 EPSG:2193 raster covering x 1000-2000, y 5000-6000, value 7.0.

    Cell (0, 0) (top-left, i.e. x 1000-1100, y 5900-6000) is nodata.
    """
    data = np.full((10, 10), 7.0, dtype="float32")
    data[0, 0] = nodata
    transform = from_origin(1000.0, 6000.0, 100.0, 100.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:2193",
        transform=transform,
        nodata=nodata,
    ) as ds:
        ds.write(data, 1)


def test_sample_band_inside_nodata_and_outside(tmp_path):
    """In-bounds cells return values; nodata and out-of-bounds return NaN."""
    tif = tmp_path / "t.tif"
    _write_tif(tif)
    xy = [(1550.0, 5550.0), (1050.0, 5950.0), (999999.0, 999999.0)]
    values = sample_band(tif, xy, band=1)
    assert values[0] == 7.0
    assert np.isnan(values[1])
    assert np.isnan(values[2])
