"""ORM Module for Geotechnical Data Management

This module defines an object-relational mapping (ORM) for managing geotechnical data
related to Standard Penetration Tests (SPT), Cone Penetration Tests (CPT), shear wave
velocity profiles, and soil measurements. It uses the `peewee` library to interact
with a SQLite database and provides models and query functions for geotechnical records.
"""

from peewee import (
    CompositeKey,
    DateField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)

from nzgd import constants

db = SqliteDatabase(
    constants.OUTPUT_DB_PATH,
)


class BaseModel(Model):
    """Base model for all database models."""

    class Meta:
        database = db


class Type(BaseModel):
    """Represents an investigation type in the database."""

    id = IntegerField(primary_key=True)
    """int: The unique identifier for the investigation type."""

    value = TextField()
    """str: The name of the investigation type."""


class Region(BaseModel):
    """Represents a region in the database."""

    id = IntegerField(primary_key=True)
    """int: The unique identifier for the region."""

    value = TextField()
    """str: The name of the region."""


class District(BaseModel):
    """Represents a district in the database."""

    id = IntegerField(primary_key=True)
    """int: The unique identifier for the district."""

    value = TextField()
    """str: The name of the district."""


class City(BaseModel):
    """Represents a city in the database."""

    id = IntegerField(primary_key=True)
    """int: The unique identifier for the city."""

    value = TextField()
    """str: The name of the city."""


class Suburb(BaseModel):
    """Represents a suburb in the database."""

    id = IntegerField(primary_key=True)
    """int: The unique identifier for the suburb."""

    value = TextField()
    """str: The name of the suburb."""


class CPTToVsCorrelation(BaseModel):
    """Represents a CPT to Vs correlation in the database."""

    id = IntegerField(primary_key=True)
    """int: The unique identifier for the CPT to Vs correlation."""

    value = TextField()
    """str: The CPT to Vs correlation"""


class SPTToVsCorrelation(BaseModel):
    """Represents an SPT to Vs correlation in the database."""

    id = IntegerField(primary_key=True)
    """int: The unique identifier for the SPT to Vs correlation."""

    value = TextField()
    """str: The SPT to Vs correlation"""


class VsToVs30Correlation(BaseModel):
    """Represents a Vs30 estimate in the database."""

    id = IntegerField(primary_key=True)
    """int: The unique identifier for the Vs30 estimate."""

    value = TextField()
    """str: The Vs30 estimate"""


class SPTToVs30HammerType(BaseModel):
    """Represents an SPT hammer type in the database."""

    id = IntegerField(primary_key=True)
    """int: The unique identifier for the hammer type."""

    value = TextField()
    """str: The hammer type for the Vs30 estimate."""


class TerminationReason(BaseModel):
    """Represents termination reasons for CPT tests in the database."""

    id = IntegerField(primary_key=True)
    """int: The unique identifier for the termination reason."""

    value = TextField()
    """str: The name of the termination reason."""


class CPTGroundWaterLevelMethod(BaseModel):
    """Represents the method for determining the ground water level."""

    id = IntegerField(primary_key=True)
    """int: The unique identifier for the ground water level method."""

    value = TextField()
    """str: The name of the ground water level method."""


class SoilTypes(BaseModel):
    """Represents soil types."""

    id = IntegerField(primary_key=True)
    """int: The unique identifier for the soil type."""

    name = TextField()
    """str: The name of the soil type."""


class NZGDRecord(BaseModel):
    """Represents a New Zealand Geotechnical Database (NZGD) record."""

    nzgd_id = IntegerField(primary_key=True)
    """int: The unique identifier for the NZGD record."""

    type_id = ForeignKeyField(Type, backref="type")
    """int: The foreign key referencing the investigation type (e.g., CPT, BH)."""

    latitude = FloatField()
    """float: The latitude coordinate of the investigation location."""

    longitude = FloatField()
    """float: The longitude coordinate of the investigation location."""

    model_vs30_foster_2019 = FloatField(null=True)
    """float: The modelled Vs30 value from Foster et al. (2019), at this record's
    location."""

    model_vs30_stddev_foster_2019 = FloatField(null=True)
    """float: The modelled Vs30 standard deviation from Foster et al. (2019), at this
    record's location."""

    model_gwl_westerhoff_2018 = FloatField(null=True)
    """float: The modelled ground water level from Westerhoff et al. (2018), at this
    record's location."""

    original_investigation_name = TextField(null=True)
    """str: The original reference for the record."""

    investigation_date = DateField(formats=["%Y-%m-%d"], null=True)
    """date: The date the investigation was conducted."""

    published_date = DateField(formats=["%Y-%m-%d"], null=True)
    """date: The date the record was published."""

    region_id = ForeignKeyField(Region, backref="region")
    """int: The foreign key referencing the region."""

    district_id = ForeignKeyField(District, backref="district")
    """int: The foreign key referencing the district."""

    city_id = ForeignKeyField(City, backref="city")
    """int: The foreign key referencing the city."""

    suburb_id = ForeignKeyField(Suburb, backref="suburb")
    """int: The foreign key referencing the suburb."""

    class Meta:
        indexes = (
            # (column, ), (boolean for is unique, )
            (("nzgd_id",), True),
            (("model_vs30_foster_2019",), False),
            (("model_gwl_westerhoff_2018",), False),
            (("region_id",), False),
            (("district_id",), False),
            (("city_id",), False),
            (("suburb_id",), False),
        )


class SPTReport(BaseModel):
    """Represents a Standard Penetration Test (SPT) report."""

    borehole_id = IntegerField(primary_key=True)
    """int: The unique identifier for the borehole."""

    nzgd_id = ForeignKeyField(NZGDRecord, backref="spt_reports")
    """int: The foreign key referencing the associated NZGD record."""

    efficiency = FloatField(null=True)
    """float: The efficiency of the test."""

    borehole_diameter = FloatField(null=True)
    """float: The diameter of the borehole."""

    extracted_gwl = FloatField(null=True)
    """float: The extracted ground water level for the SPT (borehole) report."""

    gwl_residual = FloatField(null=True)
    """float: The residual (difference) between the extracted ground water level and
    the corresponding value from the Westerhoff et al. (2018) national groundwater
    level model."""

    source_file = TextField()
    """str: The source file of the extracted data."""


class SoilMeasurements(BaseModel):
    """Represents soil measurements."""

    measurement_id = IntegerField(primary_key=True)
    """int: The unique identifier for the soil measurement."""

    report_id = ForeignKeyField(SPTReport, backref="soil_measurements")
    """int: The foreign key referencing the associated SPT report."""

    top_depth = FloatField()
    """float: The top depth of the soil layer."""


class SoilMeasurementSoilType(BaseModel):
    """Represents a junction table for soil measurements and soil types."""

    soil_measurement_id = ForeignKeyField(SoilMeasurements, backref="soil_types")
    """int: The foreign key referencing the soil measurement."""

    soil_type_id = ForeignKeyField(SoilTypes, backref="measurements")
    """int: The foreign key referencing the soil type."""

    class Meta:
        primary_key = CompositeKey("soil_measurement_id", "soil_type_id")


class SPTMeasurements(BaseModel):
    """Represents measurements for a Standard Penetration Test (SPT)."""

    measurement_id = IntegerField(primary_key=True)
    """int: The unique identifier for the measurement."""

    borehole_id = ForeignKeyField(SPTReport, backref="measurements")
    """int: The foreign key referencing the associated SPT report."""

    depth = FloatField()
    """float: The depth at which the measurement was taken."""

    n = IntegerField()
    """int: The N-value of the measurement."""


class CPTReport(BaseModel):
    """Represents a Cone Penetration Test (CPT) report."""

    cpt_id = IntegerField(primary_key=True)
    """int: The unique identifier for the CPT report."""

    nzgd_id = ForeignKeyField(NZGDRecord, backref="cpt_reports")
    """int: The foreign key referencing the associated NZGD record."""

    max_depth = FloatField(null=True)
    """float: The maximum depth of the CPT report"""

    min_depth = FloatField(null=True)
    """"float: The minimum depth of the CPT report"""

    gwl = FloatField(null=True)
    """float: The extracted ground water level for the CPT report."""

    gwl_method_id = ForeignKeyField(
        CPTGroundWaterLevelMethod,
        null=True,
    )
    """int: The foreign key referencing the ground water level method."""

    gwl_residual = FloatField(null=True)
    """float: The residual (difference) between the extracted ground water level and
    the corresponding value from the Westerhoff et al. (2018) national groundwater
    level model."""

    tip_net_area_ratio = FloatField(null=True)
    """float: The tip net area ratio for the CPT report."""

    termination_reason_id = ForeignKeyField(
        TerminationReason,
        backref="cpt_reports",
        null=True,
    )
    """int: The foreign key referencing the termination reason."""

    has_cpt_data = IntegerField()
    """int: A Boolean representing whether the CPT report has associated CPT data."""

    cpt_data_duplicate_of_cpt_id = IntegerField(null=True)
    """int: The cpt_id of the CPT report that this CPT data is a duplicate of."""

    did_explicit_unit_conversion = IntegerField(null=True)
    """int: A Boolean representing whether an explicit unit conversion was applied."""

    did_inferred_unit_conversion = IntegerField(null=True)
    """int: A Boolean representing whether an inferred unit conversion was applied."""

    source_file = TextField()
    """str: The source file of the extracted data."""

    class Meta:
        indexes = (
            # (column, ,), (boolean for is unique, ,), ...
            (("cpt_id",), True),
            (("nzgd_id",), False),
            (("termination_reason_id",), False),
            (("tip_net_area_ratio",), False),
        )


class CPTMeasurements(BaseModel):
    """Represents measurements for a Cone Penetration Test (CPT)."""

    measurement_id = IntegerField(primary_key=True)
    """int: The unique identifier for the measurement."""

    cpt_id = ForeignKeyField(CPTReport, backref="measurements")
    """int: The foreign key referencing the associated CPT report."""

    depth = FloatField(null=True)
    """float: The depth at which the measurement was taken."""

    qc = FloatField(null=True)
    """float: The cone resistance at the specified depth."""

    fs = FloatField(null=True)
    """float: The sleeve friction at the specified depth."""

    u2 = FloatField(null=True)
    """float: The pore water pressure at the specified depth."""


class CPTVs30Estimates(BaseModel):
    """Represents a Vs30 estimate calculated from a CPT record"""

    vs30_id = IntegerField(primary_key=True)
    """int: The unique identifier for the Vs30 estimate."""

    cpt_id = ForeignKeyField(CPTReport, backref="vs30_estimates")
    """int: The foreign key referencing the associated CPT report."""

    nzgd_id = ForeignKeyField(NZGDRecord, backref="cpt_vs30_estimates")
    """int: The foreign key referencing the associated NZGD record."""

    cpt_to_vs_correlation_id = ForeignKeyField(
        CPTToVsCorrelation,
        backref="cpt_to_vs_correlation",
    )
    """int: The foreign key referencing the CPT to Vs correlation."""

    vs_to_vs30_correlation_id = ForeignKeyField(
        VsToVs30Correlation,
        backref="vs_to_vs30_correlation",
    )
    """int: The foreign key referencing the Vs to Vs30 correlation."""

    vs30 = FloatField(null=True)
    """float: The calculated Vs30 value."""

    vs30_stddev = FloatField(null=True)
    """float: The calculated Vs30 standard deviation value."""

    class Meta:
        indexes = (
            # (column, ,), (boolean for is unique, ,), ...
            (("cpt_id",), False),
            (("nzgd_id",), False),
            (("cpt_to_vs_correlation_id",), False),
            (("vs_to_vs30_correlation_id",), False),
            (("vs30",), False),
        )


class SPTVs30Estimates(BaseModel):
    """Represents a Vs30 estimate calculated from a SPT record"""

    vs30_id = IntegerField(primary_key=True)
    """int: The unique identifier for the Vs30 estimate."""

    spt_id = ForeignKeyField(SPTReport, backref="vs30_estimates")
    """int: The foreign key referencing the associated SPT report."""

    spt_to_vs_correlation_id = ForeignKeyField(
        SPTToVsCorrelation,
        backref="spt_to_vs_correlation",
    )
    """int: The foreign key referencing the SPT to Vs correlation."""

    vs_to_vs30_correlation_id = ForeignKeyField(
        VsToVs30Correlation,
        backref="vs_to_vs30_correlation",
    )
    """int: The foreign key referencing the Vs to Vs30 correlation."""

    borehole_diameter = FloatField(null=True)
    """float: The diameter of the borehole."""

    hammer_type_id = ForeignKeyField(SPTToVs30HammerType, backref="hammer_type")
    """str: The ID of the hammer type used."""

    vs30_used_efficiency = IntegerField(null=True)
    """int: A Boolean representing whether information about the SPT efficiency was used for the Vs30 estimate
    0 = False, 1 = True."""

    vs30_used_soil_info = IntegerField(null=True)
    """int: A Boolean representing whether information about the soil type was used for the Vs30 estimate
    0 = False, 1 = True."""

    vs30 = FloatField(null=True)
    """float: The calculated Vs30 value."""

    vs30_stddev = FloatField(null=True)
    """float: The calculated Vs30 standard deviation value."""


# def search_spt_reports(
#     borehole_id: int | None = None,
#     min_efficiency: float | None = None,
#     max_efficiency: float | None = None,
#     min_diameter: float | None = None,
#     max_diameter: float | None = None,
#     nzgd_id: int | None = None,
#     original_investigation_name: str | None = None,
#     max_measurement_depth: float | None = None,
#     min_measurement_depth: float | None = None,
#     region: str | None = None,
#     district: str | None = None,
#     city: str | None = None,
#     suburb: str | None = None,
# ) -> Iterator[SPTReport]:
#     """Search for SPT (Standard Penetration Test) reports based on various filters.

#     Parameters
#     ----------
#     borehole_id : int, optional
#         Specific borehole ID to filter results.
#     min_efficiency : float, optional
#         Minimum efficiency value for the SPT report.
#     max_efficiency : float, optional
#         Maximum efficiency value for the SPT report.
#     min_diameter : float, optional
#         Minimum borehole diameter.
#     max_diameter : float, optional
#         Maximum borehole diameter.
#     nzgd_id : int, optional
#         ID of the associated NZGDRecord to filter results.
#     original_investigation_name : str, optional
#         Filter by a substring of the NZGDRecord's original reference.
#     max_measurement_depth : float, optional
#         Maximum depth of measurements in the SPT report.
#     min_measurement_depth : float, optional
#         Minimum depth of measurements in the SPT report.
#     region_name : str, optional
#         Name of the region to filter results.
#     district_name : str, optional
#         Name of the district to filter results.
#     city_name : str, optional
#         Name of the city to filter results.
#     suburb_name : str, optional
#         Name of the suburb to filter results.

#     Returns
#     -------
#     Iterator of SPTReport
#         A Iterator of `SPTReport` objects that match the specified criteria.

#     """
#     # Start with SPTReport and join related NZGDRecord
#     query = SPTReport.select(SPTReport).join(NZGDRecord, JOIN.INNER)

#     # Apply filters for SPTReport fields
#     if borehole_id is not None:
#         query = query.where(SPTReport.borehole_id == borehole_id)
#     if min_efficiency is not None:
#         query = query.where(SPTReport.efficiency >= min_efficiency)
#     if max_efficiency is not None:
#         query = query.where(SPTReport.efficiency <= max_efficiency)
#     if min_diameter is not None:
#         query = query.where(SPTReport.borehole_diameter >= min_diameter)
#     if max_diameter is not None:
#         query = query.where(SPTReport.borehole_diameter <= max_diameter)
#     if nzgd_id is not None:
#         query = query.where(NZGDRecord.nzgd_id == nzgd_id)
#     if original_investigation_name is not None:
#         query = query.where(NZGDRecord.original_investigation_name.contains(original_investigation_name))
#     if region is not None:
#         query = (
#             query.switch(NZGDRecord)
#             .join(Region, JOIN.INNER, on=(NZGDRecord.region_id == Region.region_id))
#             .where(Region.name == region)
#         )
#     if district is not None:
#         query = (
#             query.switch(NZGDRecord)
#             .join(
#                 District,
#                 JOIN.INNER,
#                 on=(NZGDRecord.district_id == District.district_id),
#             )
#             .where(District.name == district)
#         )
#     if city is not None:
#         query = (
#             query.switch(NZGDRecord)
#             .join(City, JOIN.INNER, on=(NZGDRecord.city_id == City.city_id))
#             .where(City.name == city)
#         )
#     if suburb is not None:
#         query = (
#             query.switch(NZGDRecord)
#             .join(Suburb, JOIN.INNER, on=(NZGDRecord.suburb_id == Suburb.suburb_id))
#             .where(Suburb.name == suburb)
#         )
#     # Apply filter for maximum measurement depth
#     if max_measurement_depth or min_measurement_depth:
#         query = (
#             query.switch(SPTReport)
#             .join(SPTMeasurements, JOIN.INNER)
#             .group_by(SPTReport.borehole_id)
#         )
#         if max_measurement_depth:
#             query = query.having(fn.MAX(SPTMeasurements.depth) <= max_measurement_depth)
#         if min_measurement_depth:
#             query = query.having(fn.MAX(SPTMeasurements.depth) >= min_measurement_depth)

#     # Execute query and return results
#     return iter(query)


# def search_cpt_reports(
#     cpt_id: int | None = None,
#     nzgd_id: int | None = None,
#     original_investigation_name: str | None = None,
#     region: str | None = None,
#     district: str | None = None,
#     city: str | None = None,
#     suburb: str | None = None,
#     max_measured_gwl: float | None = None,
#     min_measured_gwl: float | None = None,
#     max_measured_gwl_minus_model_gwl: float | None = None,
#     min_measured_gwl_minus_model_gwl: float | None = None,
#     max_model_gwl: float | None = None,
#     min_model_gwl: float | None = None,
#     max_model_vs30: float | None = None,
#     min_model_vs30: float | None = None,
#     max_model_vs30_stddev: float | None = None,
#     min_model_vs30_stddev: float | None = None,
#     max_tip_net_area_ratio: float | None = None,
#     min_tip_net_area_ratio: float | None = None,
#     max_deepest_depth: float | None = None,
#     min_deepest_depth: float | None = None,
#     max_shallowest_depth: float | None = None,
#     min_shallowest_depth: float | None = None,
#     termination_reason: str | None = None,
# ) -> Iterator[CPTReport]:
#     """Search for CPT (Cone Penetration Test) reports based on various filters.

#     Parameters
#     ----------
#     cpt_id : int, optional
#         Specific CPT report ID to filter results.
#     nzgd_id : int, optional
#         ID of the associated NZGDRecord to filter results.
#     original_investigation_name : str, optional
#         Filter by a substring of the NZGDRecord's original reference.
#     region : str, optional
#         Name of the region to filter results.
#     district_name : str, optional
#         Name of the district to filter results.
#     city_name : str, optional
#         Name of the city to filter results.
#     suburb : str, optional
#         Name of the suburb to filter results.
#     max_measured_gwl : float, optional
#         Maximum measured ground water level to filter results.
#     min_measured_gwl : float, optional
#         Minimum measured ground water level to filter results.
#     max_measured_gwl_minus_model_gwl : float, optional
#         Maximum difference between measured and modelled ground water level to filter results.
#     min_measured_gwl_minus_model_gwl : float, optional
#         Minimum difference between measured and modelled ground water level to filter results.
#     max_model_gwl : float, optional
#         Maximum modelled ground water level to filter results.
#     min_model_gwl : float, optional
#         Minimum modelled ground water level to filter results.
#     max_model_vs30 : float, optional
#         Maximum modelled Vs30 value to filter results.
#     min_model_vs30 : float, optional
#         Minimum modelled Vs30 value to filter results.
#     max_model_vs30_stddev : float, optional
#         Maximum modelled Vs30 standard deviation value to filter results.
#     min_model_vs30_stddev : float, optional
#         Minimum modelled Vs30 standard deviation value to filter results.
#     max_tip_net_area_ratio : float, optional
#         Maximum tip net area ratio to filter results.
#     min_tip_net_area_ratio : float, optional
#         Minimum tip net area ratio to filter results.
#     max_deepest_depth : float, optional
#         Maximum deepest depth to filter results.
#     min_deepest_depth : float, optional
#         Minimum deepest depth to filter results.
#     max_shallowest_depth : float, optional
#         Maximum shallowest depth to filter results.
#     min_shallowest_depth : float, optional
#         Minimum shallowest depth to filter results.
#     termination_reason : str, optional
#         The CPT's reason for termination.

#     Returns
#     -------
#     Iterator of CPTReport
#         A Iterator of `CPTReport` objects that match the specified criteria.

#     """
#     # Start with CPTReport and join related NZGDRecord
#     query = CPTReport.select(CPTReport).join(NZGDRecord, JOIN.INNER)

#     # Apply filters for CPTReport fields
#     if cpt_id is not None:
#         query = query.where(CPTReport.cpt_id == cpt_id)
#     if nzgd_id is not None:
#         query = query.where(NZGDRecord.nzgd_id == nzgd_id)
#     if original_investigation_name is not None:
#         query = query.where(NZGDRecord.original_investigation_name.contains(original_investigation_name))

#     # Apply filters for location-related fields
#     if region is not None:
#         query = (
#             query.switch(NZGDRecord)
#             .join(Region, JOIN.INNER, on=(NZGDRecord.region_id == Region.region_id))
#             .where(Region.name == region)
#         )
#     if district is not None:
#         query = (
#             query.switch(NZGDRecord)
#             .join(
#                 District,
#                 JOIN.INNER,
#                 on=(NZGDRecord.district_id == District.district_id),
#             )
#             .where(District.name == district)
#         )
#     if city is not None:
#         query = (
#             query.switch(NZGDRecord)
#             .join(City, JOIN.INNER, on=(NZGDRecord.city_id == City.city_id))
#             .where(City.name == city)
#         )
#     if suburb is not None:
#         query = (
#             query.switch(NZGDRecord)
#             .join(Suburb, JOIN.INNER, on=(NZGDRecord.suburb_id == Suburb.suburb_id))
#             .where(Suburb.name == suburb)
#         )

#     if max_deepest_depth:
#         query = query.where(CPTReport.deepest_depth <= max_deepest_depth)
#     if min_deepest_depth:
#         query = query.where(CPTReport.deepest_depth >= min_deepest_depth)
#     if max_shallowest_depth:
#         query = query.where(CPTReport.shallowest_depth <= max_shallowest_depth)
#     if min_shallowest_depth:
#         query = query.where(CPTReport.shallowest_depth >= min_shallowest_depth)

#     # Apply filter for measured ground water level (gwl)
#     if max_measured_gwl or min_measured_gwl:
#         if max_measured_gwl:
#             query = query.where(CPTReport.measured_gwl <= max_measured_gwl)
#         if min_measured_gwl:
#             query = query.where(CPTReport.measured_gwl >= min_measured_gwl)

#     # Apply filter for measured ground water level minus model ground water level
#     if max_measured_gwl_minus_model_gwl or min_measured_gwl_minus_model_gwl:
#         if max_measured_gwl_minus_model_gwl:
#             query = query.where(
#                 CPTReport.measured_gwl_minus_model_gwl
#                 <= max_measured_gwl_minus_model_gwl,
#             )
#         if min_measured_gwl_minus_model_gwl:
#             query = query.where(
#                 CPTReport.measured_gwl_minus_model_gwl
#                 >= min_measured_gwl_minus_model_gwl,
#             )

#     # Apply filter for model ground water level (model_gwl_westerhoff_2018)
#     if max_model_gwl or min_model_gwl:
#         if max_model_gwl:
#             query = query.where(NZGDRecord.model_gwl_westerhoff_2018 <= max_model_gwl)
#         if min_model_gwl:
#             query = query.where(NZGDRecord.model_gwl_westerhoff_2018 >= min_model_gwl)

#     # Apply filter for model Vs30 (model_vs30_foster_2019)
#     if max_model_vs30 or min_model_vs30:
#         if max_model_vs30:
#             query = query.where(NZGDRecord.model_vs30_foster_2019 <= max_model_vs30)
#         if min_model_vs30:
#             query = query.where(NZGDRecord.model_vs30_foster_2019 >= min_model_vs30)

#     # Apply filter for model Vs30 standard deviation (model_vs30_stddev_foster_2019)
#     if max_model_vs30_stddev or min_model_vs30_stddev:
#         if max_model_vs30_stddev:
#             query = query.where(
#                 NZGDRecord.model_vs30_stddev_foster_2019 <= max_model_vs30_stddev,
#             )
#         if min_model_vs30_stddev:
#             query = query.where(
#                 NZGDRecord.model_vs30_stddev_foster_2019 >= min_model_vs30_stddev,
#             )

#     # Apply filter for tip net area ratio (tip_net_area_ratio)
#     if max_tip_net_area_ratio or min_tip_net_area_ratio:
#         if max_tip_net_area_ratio:
#             query = query.where(CPTReport.tip_net_area_ratio <= max_tip_net_area_ratio)
#         if min_tip_net_area_ratio:
#             query = query.where(CPTReport.tip_net_area_ratio >= min_tip_net_area_ratio)

#     # Execute query and return results
#     return iter(query)


def initialize_db():
    with db:
        db.create_tables(
            [
                Type,
                Region,
                District,
                City,
                Suburb,
                CPTToVsCorrelation,
                SPTToVsCorrelation,
                VsToVs30Correlation,
                SPTToVs30HammerType,
                TerminationReason,
                CPTGroundWaterLevelMethod,
                SoilTypes,
                NZGDRecord,
                SPTReport,
                SoilMeasurements,
                SoilMeasurementSoilType,
                SPTMeasurements,
                CPTReport,
                CPTMeasurements,
                CPTVs30Estimates,
                SPTVs30Estimates,
            ],
        )
