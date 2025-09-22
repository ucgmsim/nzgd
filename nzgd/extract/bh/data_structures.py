"""Data structures used in borehole SPT extraction."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TextObject:
    """Represents a text object with positional and textual data.

    Attributes
    ----------
    y0 : float
        The lower y-coordinate of the text object.
    y1 : float
        The upper y-coordinate of the text object.
    x0 : float
        The left x-coordinate of the text object.
    x1 : float
        The right x-coordinate of the text object.
    text : str
        The textual content of the object.

    """

    y0: float
    y1: float
    x0: float
    x1: float
    text: str

    @property
    def yc(self) -> float:
        """float: The vertical centre coordinate."""
        return (self.y1 + self.y0) / 2

    @property
    def xc(self) -> float:
        """float: The horizontal centre coordinate."""
        return (self.x1 + self.x0) / 2


@dataclass
class SPTReport:
    """Represents an SPT report extracted from a borehole log."""

    borehole_id: int
    """The borehole ID number for this report."""

    nzgd_id: int
    """The NZGD D number for this report."""

    efficiency: float | None
    """The hammer efficiency ratio."""

    extracted_gwl: float | None
    """The extracted ground water level for the SPT (borehole) report."""

    source_file: Path
    """The path to the report."""

    spt_measurements: pd.DataFrame
    """The SPT record. A data frame with columns Depth, and N."""

    soil_measurements: pd.DataFrame
    """The SPT soil measurements. A dataframe with columns 'top_depth', and 'soil_types'"""
