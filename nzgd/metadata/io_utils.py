"""Atomic file-writing helpers for index artifacts."""

import os
from pathlib import Path

import pandas as pd


def atomic_write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to gzipped CSV atomically (temp file + rename).

    Parameters
    ----------
    df : pd.DataFrame
        Frame to write (index is not written).
    path : Path
        Destination path ending in .csv.gz.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    df.to_csv(tmp_path, index=False, compression="gzip")
    os.replace(tmp_path, path)
