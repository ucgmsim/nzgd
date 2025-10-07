"""Classes and functions for retrieving SPT data from the UC NZGD SQLite database."""

import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Self

import pandas as pd
from intervaltree import Interval, IntervalTree

from nzgd.db import orm


class SoilTypesEnum(Enum):
    """Enum representing different types of soil."""

    SAND = "SAND"
    SILT = "SILT"
    CLAY = "CLAY"
    GRAVEL = "GRAVEL"
    BOULDERS = "BOULDERS"
    COBBLES = "COBBLES"


@dataclass
class NZGDRecord:
    """A data class representing a New Zealand Geotechnical Database (NZGD) record."""

    nzgd_id: int
    """int: The unique identifier for the NZGD record."""

    type_id: int
    """int: The foreign key referencing the investigation type (e.g., CPT, BH)."""

    latitude: float
    """float: The latitude coordinate of the investigation location."""

    longitude: float
    """float: The longitude coordinate of the investigation location."""

    model_vs30_foster_2019: float
    """float: The modelled Vs30 value from Foster et al. (2019), at this record's
    location."""

    model_vs30_stddev_foster_2019: float
    """float: The modelled Vs30 standard deviation from Foster et al. (2019), at this
    record's location."""

    model_gwl_westerhoff_2018: float
    """float: The modelled ground water level from Westerhoff et al. (2018), at this
    record's location."""

    original_investigation_name: str
    """str: The original reference for the record."""

    investigation_date: date
    """date: The date the investigation was conducted."""

    published_date: date
    """date: The date the record was published."""

    region_id: int
    """int: The foreign key referencing the region."""

    district_id: int
    """int: The foreign key referencing the district."""

    city_id: int
    """int: The foreign key referencing the city."""

    suburb_id: int
    """int: The foreign key referencing the suburb."""

    @classmethod
    def from_orm(cls, record: orm.NZGDRecord) -> Self:
        """Create an NZGDRecord instance from an ORM record.

        Parameters
        ----------
        record : orm.NZGDRecord
            The ORM record to convert.

        Returns
        -------
        NZGDRecord
            The corresponding NZGDRecord instance.

        """
        return cls(
            nzgd_id=record.nzgd_id,
            type_id=record.type_id,
            latitude=record.latitude,
            longitude=record.longitude,
            model_vs30_foster_2019=record.model_vs30_foster_2019,
            model_vs30_stddev_foster_2019=record.model_vs30_stddev_foster_2019,
            model_gwl_westerhoff_2018=record.model_gwl_westerhoff_2018,
            original_investigation_name=record.original_investigation_name,
            investigation_date=record.investigation_date,
            published_date=record.published_date,
            region_id=record.region_id,
            district_id=record.district_id,
            city_id=record.city_id,
            suburb_id=record.suburb_id,
        )


@dataclass
class SPTReport:
    """A data class representing a Standard Penetration Test (SPT) report."""

    borehole_id: int
    """int: The unique identifier for the borehole."""

    nzgd_id: int
    """int: The NZGD record ID associated with the report."""

    efficiency: float
    """float: The efficiency of the test."""

    extracted_gwl: float
    """float: The measured groundwater level."""

    gwl_residual: float
    """float: The residual (difference) between the extracted ground water level and
    the corresponding value from the Westerhoff et al. (2018) national groundwater
    level model."""

    source_file: str
    """str: The source file of the extracted data."""

    nzgd_record: NZGDRecord
    """NZGDRecord: The NZGD record associated with the report."""

    measurements: pd.DataFrame = field(default_factory=pd.DataFrame)
    """pd.DataFrame: A DataFrame containing SPT measurements."""

    soil_measurements: IntervalTree = field(default_factory=IntervalTree)
    """IntervalTree: An IntervalTree containing soil measurements."""

    @classmethod
    def from_orm(cls, report: orm.SPTReport) -> Self:
        """Create an SPTReport instance from an ORM report.

        Parameters
        ----------
        report : orm.SPTReport
            The ORM report to convert.

        Returns
        -------
        SPTReport
            The corresponding SPTReport instance.

        """
        # Create a DataFrame for SPT measurements
        measurements_data = [
            {"depth": m.depth, "n_value": m.n}
            for m in sorted(report.measurements, key=lambda x: x.depth)
        ]
        measurements_df = pd.DataFrame(measurements_data)

        # Create an IntervalTree for soil measurements
        soil_measurements_tree = IntervalTree()
        measurements = sorted(report.soil_measurements, key=lambda x: x.top_depth)
        if measurements:
            for s, next in itertools.pairwise(measurements):
                if s.top_depth == next.top_depth:
                    continue
                for soil_type in s.soil_types:
                    soil_measurements_tree.add(
                        Interval(
                            s.top_depth,
                            next.top_depth,
                            SoilTypesEnum[soil_type.soil_type_id.name],
                        ),
                    )
            bottom_measurement = measurements[-1]
            for soil_type in bottom_measurement.soil_types:
                soil_measurements_tree.add(
                    Interval(
                        bottom_measurement.top_depth,
                        100,
                        SoilTypesEnum[soil_type.soil_type_id.name],
                    ),
                )

        return cls(
            borehole_id=report.borehole_id,
            nzgd_id=report.nzgd_id,
            efficiency=report.efficiency,
            extracted_gwl=report.extracted_gwl,
            gwl_residual=report.gwl_residual,
            source_file=report.source_file,
            nzgd_record=NZGDRecord.from_orm(report.nzgd_id),
            measurements=measurements_df,
            soil_measurements=soil_measurements_tree,
        )


def search_spt_reports(
    borehole_id: int | None = None,
    min_efficiency: float | None = None,
    max_efficiency: float | None = None,
    nzgd_id: int | None = None,
    original_investigation_name: str | None = None,
    max_measurement_depth: float | None = None,
    min_measurement_depth: float | None = None,
    region: str | None = None,
    district: str | None = None,
    city: str | None = None,
    suburb: str | None = None,
) -> Iterator[SPTReport]:
    """Search for SPT reports based on the given filters.

    Parameters
    ----------
    borehole_id : Optional[int], optional
        The borehole ID to filter by.
    min_efficiency : Optional[float], optional
        The minimum efficiency to filter by.
    max_efficiency : Optional[float], optional
        The maximum efficiency to filter by.
    nzgd_id : Optional[int], optional
        The NZGD ID to filter by.
    original_investigation_name : Optional[str], optional
        The original reference to filter by.
    max_measurement_depth : Optional[float], optional
        The maximum measurement depth to filter by.
    min_measurement_depth : Optional[float], optional
        The minimum measurement depth to filter by.
    region : Optional[str], optional
        The region to filter by.

    district : Optional[str], optional
        The district to filter by.

    city : Optional[str], optional
        The city to filter by.

    suburb : Optional[str], optional
        The suburb to filter by.

    Returns
    -------
    list[SPTReport]
        A list of SPTReport instances that match the filter criteria.

    """
    return (
        SPTReport.from_orm(report)
        for report in orm.search_spt_reports(
            borehole_id=borehole_id,
            min_efficiency=min_efficiency,
            max_efficiency=max_efficiency,
            nzgd_id=nzgd_id,
            original_investigation_name=original_investigation_name,
            max_measurement_depth=max_measurement_depth,
            min_measurement_depth=min_measurement_depth,
            region=region,
            city=city,
            district=district,
            suburb=suburb,
        )
    )
