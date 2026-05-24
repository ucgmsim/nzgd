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

    value = TextField()
    """str: The name of the soil type."""


class DensityDescriptions(BaseModel):
    """Represents normalized density description terms."""

    id = IntegerField(primary_key=True)
    """int: The unique identifier for the density description."""

    value = TextField()
    """str: The density description term (e.g., 'dense', 'medium dense', 'very dense')."""


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

    model_vs30_foster_2019_km_per_s = FloatField(null=True)
    """float: The modelled Vs30 value from Foster et al. (2019), at this record's
    location."""

    model_vs30_stddev_foster_2019_km_per_s = FloatField(null=True)
    """float: The modelled Vs30 standard deviation from Foster et al. (2019), at this
    record's location."""

    model_gwl_westerhoff_2018_m = FloatField(null=True)
    """float: The modelled ground water level from Westerhoff et al. (2018), at this
    record's location, in meters."""

    model_gwl_nlm_2025_m = FloatField(null=True)
    """float: The modelled ground water level from the National Liquefaction Model (NLM) 2025,
    at this record's location, in meters."""

    model_gwl_nlm_2025_stddev_m = FloatField(null=True)
    """float: The modelled ground water level standard deviation from the National
    Liquefaction Model (NLM) 2025, at this record's location, in meters."""

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
            (("model_vs30_foster_2019_km_per_s",), False),
            (("model_gwl_westerhoff_2018_m",), False),
            (("model_gwl_nlm_2025_m",), False),
            (("model_gwl_nlm_2025_stddev_m",), False),
            (("region_id",), False),
            (("district_id",), False),
            (("city_id",), False),
            (("suburb_id",), False),
        )


class SPTReport(BaseModel):
    """Represents a Standard Penetration Test (SPT) report."""

    spt_id = IntegerField(primary_key=True)
    """int: The unique identifier for the borehole."""

    nzgd_id = ForeignKeyField(NZGDRecord, backref="spt_reports")
    """int: The foreign key referencing the associated NZGD record."""

    efficiency = FloatField(null=True)
    """float: The efficiency of the test."""

    extracted_gwl_m = FloatField(null=True)
    """float: The extracted ground water level for the SPT (borehole) report."""

    borehole_diameter = FloatField(null=True)
    """float: The extracted borehole diameter."""

    casing_diameter = FloatField(null=True)
    """float: The extracted casing diameter."""

    source_file = TextField()
    """str: The source file of the extracted data."""


class SoilMeasurements(BaseModel):
    """Represents soil measurements."""

    soil_measurement_id = IntegerField(primary_key=True)
    """int: The unique identifier for the soil measurement."""

    spt_id = ForeignKeyField(SPTReport, backref="soil_measurements")
    """int: The foreign key referencing the associated SPT report."""

    top_depth_m = FloatField()
    """float: The top depth of the soil layer."""

    bottom_depth_m = FloatField(null=True)
    """float: The bottom depth of the soil layer (if available)."""


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

    spt_measurement_id = IntegerField(primary_key=True)
    """int: The unique identifier for the measurement."""

    spt_id = ForeignKeyField(SPTReport, backref="measurements")
    """int: The foreign key referencing the associated SPT report."""

    depth_m = FloatField(null=True)
    """float: The depth at which the measurement was taken."""

    ISPT_MAIN = IntegerField(null=True)
    """int: The number of blows extracted from a PDF file or the ISPT_MAIN field of an AGS file."""

    ISPT_NVAL = IntegerField(null=True)
    """int: The number of blows extracted from the ISPT_NVAL field of an AGS file."""

    ISPT_REP = IntegerField(null=True)
    """int: The number of blows extracted from the ISPT_REP field of an AGS file."""


class DensityMeasurements(BaseModel):
    """Represents depth-specific density measurements extracted from GEOL table.

    This table stores density information that is associated with specific depth
    intervals (from GEOL table). It contains descriptive terms tied to specific soil layers.
    """

    density_measurement_id = IntegerField(primary_key=True)
    """int: The unique identifier for the density measurement."""

    spt_id = ForeignKeyField(SPTReport, backref="density_measurements")
    """int: The foreign key referencing the associated SPT report."""

    top_depth_m = FloatField()
    """float: The top depth of the layer with this density measurement."""

    bottom_depth_m = FloatField(null=True)
    """float: The bottom depth of the layer with this density measurement.
    Null if only top depth is available."""

    density_keyword = TextField(null=True)
    """str: The extracted density keyword with modifiers (e.g., 'very dense', 'loose', 'well compacted')."""


class CPTReport(BaseModel):
    """Represents a Cone Penetration Test (CPT) report."""

    cpt_id = IntegerField(primary_key=True)
    """int: The unique identifier for the CPT report."""

    nzgd_id = ForeignKeyField(NZGDRecord, backref="cpt_reports")
    """int: The foreign key referencing the associated NZGD record."""

    max_depth_m = FloatField(null=True)
    """float: The maximum depth of the CPT report"""

    min_depth_m = FloatField(null=True)
    """"float: The minimum depth of the CPT report"""

    extracted_gwl_m = FloatField(null=True)
    """float: The extracted ground water level for the CPT report."""

    gwl_method_id = ForeignKeyField(
        CPTGroundWaterLevelMethod,
        null=True,
    )
    """int: The foreign key referencing the ground water level method."""

    tip_net_area_ratio = FloatField(null=True)
    """float: The tip net area ratio of the CPT."""

    predrill_depth_m = FloatField(null=True)
    """float: The pre-drill depth of the CPT."""

    termination_reason_id = ForeignKeyField(
        TerminationReason,
        backref="cpt_reports",
        null=True,
    )
    """int: The foreign key referencing the termination reason."""

    has_cpt_data = IntegerField()
    """int: A Boolean representing whether the CPT report has associated CPT data."""

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

    depth_m = FloatField(null=True)
    """float: The depth at which the measurement was taken."""

    qc_MPa = FloatField(null=True)
    """float: The cone resistance at the specified depth."""

    fs_MPa = FloatField(null=True)
    """float: The sleeve friction at the specified depth."""

    u2_MPa = FloatField(null=True)
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
    """int: The foreign key referencing the associated borehole_id."""

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

    assumed_borehole_diameter_mm = FloatField(null=True)
    """float: The assumed diameter of the borehole."""

    assumed_hammer_type_id = ForeignKeyField(SPTToVs30HammerType, backref="hammer_type")
    """str: The ID of the assumed hammer type used."""

    estimate_used_extracted_efficiency = IntegerField(null=True)
    """int: A Boolean representing whether extracted information about the SPT 
    efficiency was used in the Vs30 estimate. 0 = False, 1 = True."""

    estimate_used_extracted_layer_soil_types = IntegerField(null=True)
    """int: A Boolean representing whether extracted information about layer soil types
    was used in the Vs30 estimate. 0 = False, 1 = True."""

    vs30 = FloatField(null=True)
    """float: The calculated Vs30 value."""

    vs30_stddev = FloatField(null=True)
    """float: The calculated Vs30 standard deviation value."""


def initialize_db():
    """Initialize the database and create supporting tables if they do not exist."""
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
                DensityMeasurements,
                CPTReport,
                CPTMeasurements,
                CPTVs30Estimates,
                SPTVs30Estimates,
            ],
        )
