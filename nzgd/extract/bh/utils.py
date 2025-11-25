"""Data structures used in borehole SPT extraction."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nzgd.constants import SOIL_TYPE_TO_ID


def extract_soil_report(description: str) -> set[str]:
    """Extract soil types mentioned in a description.

    Parameters
    ----------
    description : str
        The input text to search for soil types.

    Returns
    -------
    set[str]
        A set of identified soil types from the input.

    """
    soil_types = set(SOIL_TYPE_TO_ID.keys())

    return soil_types & {word.strip(",.;") for word in description.split()}


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
    
    density_measurements: pd.DataFrame
    """Depth-specific density measurements from GEOL table. A dataframe with columns 
    'top_depth_m', 'bottom_depth_m', 'density_description', 'density_index_min', 
    and 'density_index_max'. All rows have depth associations."""
    
    spt_density_indices: pd.DataFrame
    """General density index ranges from ADDL_CNDN table (no depth associations). 
    A dataframe with columns 'density_index_min' and 'density_index_max'."""