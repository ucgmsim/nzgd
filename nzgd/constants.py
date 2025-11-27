"""Constants and configuration for NZGD data extraction."""

import enum
import importlib.resources
from pathlib import Path

import numpy as np
import yaml

# Define the resource directory
RESOURCE_PATH = importlib.resources.files("nzgd") / "resources"

# Load all configurations from the YAML file
with (RESOURCE_PATH / "config.yaml").open() as f:
    CONFIG = yaml.safe_load(f)

INDEX_FILE_PATH = RESOURCE_PATH / CONFIG["nzgd_index_file_name"]

SOIL_TYPE_UNIT_WEIGHTS_PATH = RESOURCE_PATH / CONFIG["soil_type_unit_weights_filename"]

NZGD_TypeDisplay_VALUES_FOR_BOREHOLES = CONFIG["nzgd_TypeDisplay_values_for_boreholes"]
NZGD_TypeDisplay_VALUES_FOR_CPTS = CONFIG["nzgd_TypeDisplay_values_for_CPTs"]

CPT_TRACE_OUTPUT_DIR = Path(CONFIG["base_output_dir"]) / Path(
    CONFIG["subdirectories"]["cpt_trace"]
)

FAILED_CPT_TRACE_OUTPUT_DIR = Path(CONFIG["base_output_dir"]) / Path(
    CONFIG["subdirectories"]["failed_cpt_trace"]
)

SUPPLEMENTAL_VALUES_OUTPUT_DIR = Path(CONFIG["base_output_dir"]) / Path(
    CONFIG["subdirectories"]["supplemental_values"]
)

ALL_POTENTIAL_CPT_SUPPLEMENTAL_VALUES_FILENAME = CONFIG[
    "all_potential_cpt_supplemental_values_filename"
]
CPT_SUPPLEMENTAL_VALUES_FILENAME = CONFIG["cpt_supplemental_values_filename"]

NZGD_SOURCE_DATA_DIR = Path(
    CONFIG["nzgd_source_data_dir"],
)

OUTPUT_DB_PATH = Path(CONFIG["output_db_path"])

TEMP_SPT_AGS_DB_PATH = Path(CONFIG["temp_spt_ags_db"])
TEMP_SPT_PDF_DB_PATH = Path(CONFIG["temp_spt_pdf_db"])

SPT_AGS_LOG_FILE_PATH = Path(CONFIG["spt_ags_log_file_path"])
SPT_AGS_EXTRACTION_WARNINGS_DIR = Path(CONFIG["spt_ags_extraction_warnings_dir"])

# Paths to groundwater level model GeoTIFF files
WESTERHOFF_2018_MODEL_PATH = Path(CONFIG["westerhoff_2018_model_path"])
NLM_GWD_PATH = Path(CONFIG["nlm_gwd_path"])
NLM_GW_STD_PATH = Path(CONFIG["nlm_gw_std_path"])

# Path to Foster et al. (2019) Vs30 model GeoTIFF file
FOSTER_2019_VS30_MODEL_PATH = Path(CONFIG["foster_2019_vs30_model_path"])

# Get relevant file extensions from the configuration file
FILE_EXTENSIONS = CONFIG["file_extensions"]

# Get the column descriptions from the configuration file
COLUMN_DESCRIPTIONS = CONFIG["cpt_column_name_descriptions"]

# Get the column names from the configuration file
COLUMN_CPT_DATA_TYPES = np.array(list(COLUMN_DESCRIPTIONS.keys())[:4])

# Load known missing value placeholders
KNOWN_MISSING_VALUE_PLACEHOLDERS = np.array(CONFIG["known_missing_value_placeholders"])

# Get known special cases from the configuration file
KNOWN_SPECIAL_CASES = CONFIG["known_special_cases"]

# Get search patterns from the configuration file
SEARCH_PATTERNS_CONFIG = CONFIG["search_patterns"]

# Get the list of strings in column names to avoid if possible
AVOID_COLUMN_NAMES_CONTAINING_IF_POSSIBLE = CONFIG[
    "avoid_columns_containing_if_possible"
]

MIN_NUM_DATA_POINTS_PER_COLUMN = CONFIG["min_num_data_points_per_column"]
MIN_NUMERICAL_SURPLUS_PER_COLUMN = CONFIG["min_numerical_surplus_per_column"]
MIN_NUMERICAL_SURPLUS_PER_ROW = CONFIG["min_numerical_surplus_per_row"]

REQUIRED_NUMBER_OF_COLUMNS = 4

INFER_WRONG_UNITS_THRESHOLDS = CONFIG[
    "infer_wrong_units_thresholds"
]  # Dictionary with keys as column names and values as thresholds


class ExcelEngine(enum.StrEnum):
    """Enumeration of the investigation types for which data can be extracted.

    Attributes
    ----------
    xlrd : str
        Represents the 'xlrd' engine for reading Excel files.
    openpyxl : str
        Represents the 'openpyxl' engine for reading Excel files.

    """

    xlrd = "xlrd"
    openpyxl = "openpyxl"


class NumOrText(enum.StrEnum):
    """Enumerations to describe or numeric or text data.

    Attributes
    ----------
    NUMERIC : enum.auto()
        Represents numeric cells.
    TEXT : enum.auto()
        Represents text cells.

    """

    NUMERIC = enum.auto()
    TEXT = enum.auto()


class QuantityToExtract(enum.StrEnum):
    """Enumerations to describe the quantity to extract.

    Attributes
    ----------
    ground_water_level : enum.StrEnum
        StrEnum representing ground water level.
    tip_net_area_ratio : enum.StrEnum
        StrEnum representing tip net area ratio.
    predrill_depth : enum.StrEnum
        StrEnum representing predrill depth.
    termination_reason : enum.StrEnum
        StrEnum representing termination reason.
    """

    ground_water_level = "ground_water_level"
    tip_net_area_ratio = "tip_net_area_ratio"
    predrill_depth = "predrill_depth"
    termination_reason = "termination_reason"


ENCODINGS_TO_TRY = CONFIG["encodings_to_try"]

# Key words for termination reason, ground water level, and tip net area ratio

MAX_NUM_BLANK_SPACES_TO_TRY_SKIPPING = 4


class TerminationReason(enum.StrEnum):
    """Represent the reason for termination of a CPT record."""

    TARGET_DEPTH = "target_depth"
    REFUSAL = "refusal"
    MISSING_VALUE = "missing_value"


class GwlMethod(enum.StrEnum):
    """Represent the reason for termination of a CPT record."""

    ESTIMATED = "estimated"
    ASSUMED = "assumed"
    DERIVED = "derived"
    DIPPER = "dipper"
    MEASURED = "measured"
    COLLAPSED = "collapsed"


search_target__from_one_cell__target_depth_reached = [
    "target depth reached",
    "target depth reach",  # noticed this case for CPT_125456
    "reached target depth",
    "target reached",
    "reached_target",
    "reason for ending test:;;;target depth",
]

search_target__from_one_cell__refused = [
    "not",  # e.g., "did not reach target depth"
    "prevent",  # e.g., "prevented from reaching target depth"
    "unable",  # e.g., "unable to reach target depth"
    "failed",  # e.g., "failed to reach target depth"
    "before",
]


search_target__assuming_is_value__field_names__target_depth_reached = [
    "remark",
    "remarks",
    "comment",
    "reason",
    "refusal factor",
    "termination",
    "resusal factor",  # noticed this case for CPT_54820
    "factor",
]

# Regex patterns to match target depth values indicating target depth was reached
# \d+\.?\d* - matches digits with optional decimal point and decimal digits
# c?m - matches optional "c" followed by "m" (for both meters and centimeters)
# Examples: "10 m", "10m", "10cm", "15.5 cm", "20m.", etc.
search_target__is_field_name__values_for_target_depth_reached = [
    r"\d+\.?\d* c?m",  # Number with space before unit (e.g., "10 m", "15.5 cm")
    r"\d+\.?\d*c?m",  # Number without space before unit (e.g., "10m", "15cm")
    r"\d+\.?\d*c?m\.",  # Number with unit followed by period (e.g., "10m.", "15cm.")
    r"\d+\.?\d*",  # Just a number without unit (e.g., "10", "15.5")
    r"yes",  # Literal "yes" indicating target depth reached
]

search_target__is_field_name__values_for_refused = [
    "no",
]

search_termination__from_one_cell__refused = ["termination critera: max thrust"]

search_termination__is_field_name__values_for_target_depth_reached = (
    search_target__from_one_cell__target_depth_reached
)

search_termination__is_field_name__values_for_refused = [
    "refuse",
    "refusal",
    "limit of reaction force",
    "inclination (tilt) limit exceeded",
    "high",
    "buckling",
    "danger",
    "stop",  # "stopped before reaching target depth"
    "lifting",  # From "TRUCK LIFTING" in CPT_204197
    "max",
    "bending",  # From "RODS BENDING" in CPT_158858
    "obstacle",  # From "OBSTACLE" in CPT_213090
    "high qc",  # E.g., CPT_189393_Raw01.CSV
    "effective refusal.",
    "danger of buckling rods",  # E.g., SCPT_172267
]

header_search_false_positives = [
    ["drilled", "target", "start", "end", "water", "max", "initial", "final"],
    [
        "_id",
        "saturation",
        "method",
        "zero",
        "before",
        "after",
        "reducer",
        "pressure",
        "initial",
        "final",
    ],
    ["reducer", "zero", "before", "after", "initial", "final"],
    ["depth", "before", "after", "initial", "final"],
]

search_refusal__is_field_name__values_for_target_depth_reached = (
    search_target__is_field_name__values_for_refused
)

search_refusal__is_field_name__values_for_refused = (
    search_target__is_field_name__values_for_target_depth_reached
)

# Noticed this misspelling case for CPT_94621
search_reason__from_one_cell__values_for_target_depth_reached = [
    *search_target__from_one_cell__target_depth_reached,
    "REASON FOR ENDING TEST:;;;TRAGET DEPTH",
]

search_reason__from_one_cell__values_refused = [
    *search_termination__is_field_name__values_for_refused,
    "termination critera: max thrust",
    "REASON FOR ENDING TEST:;;;GRAVELS",
]

search_reason__is_field_name__values_for_target_depth_reached = (
    search_target__from_one_cell__target_depth_reached
)

search_reason__is_field_name__values_for_refused = (
    search_termination__is_field_name__values_for_refused
)


search_comment__from_one_cell__values_for_target_depth_reached = [
    *search_target__from_one_cell__target_depth_reached,
    "target",
]


# Regex pattern to match decimal numbers including scientific notation
# -? - optionally matches a minus sign (for negative numbers)
# \d+ - matches one or more digits (integer part)
# (?:\.\d+)? - optional non-capturing group for decimal part:
#   \. - matches literal decimal point
#   \d+ - matches one or more digits (ensures digits after decimal point)
# (?:[eE][+-]?\d+)? - optional non-capturing group for scientific notation:
#   [eE] - matches 'e' or 'E'
#   [+-]? - optionally matches '+' or '-' sign
#   \d+ - matches one or more digits for the exponent
# Examples: 123, -0.80, 1.40, 1.4E+0000, -1.4e-03, 123E5
# Note: This pattern ensures that if a decimal point exists, digits must follow it
NUMERICAL_VALUES_REGEX = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

MAX_ALLOWED_GWL = CONFIG["allowed_bounds_for_extracted_values"]["max_allowed_gwl"]
MIN_ALLOWED_GWL = CONFIG["allowed_bounds_for_extracted_values"]["min_allowed_gwl"]

MAX_ALLOWED_PREDRILL_DEPTH = CONFIG["allowed_bounds_for_extracted_values"][
    "max_allowed_predrill_depth"
]
MIN_ALLOWED_PREDRILL_DEPTH = CONFIG["allowed_bounds_for_extracted_values"][
    "min_allowed_predrill_depth"
]

MAX_ALLOWED_TIP_NET_AREA_RATIO = CONFIG["allowed_bounds_for_extracted_values"][
    "max_allowed_tip_net_area_ratio"
]
MIN_ALLOWED_TIP_NET_AREA_RATIO = CONFIG["allowed_bounds_for_extracted_values"][
    "min_allowed_tip_net_area_ratio"
]

gwl__not_measured_terms = [
    "no",
    "not",
    "collapse",
    "assume",
    "estimate",
    "unavailable",
    "unable",
]

search_gwl__incl_not_measured = [
    NUMERICAL_VALUES_REGEX,
    *gwl__not_measured_terms,
]


term_dict = {
    "target": {
        "assuming_cell_is_standalone": search_target__from_one_cell__target_depth_reached
        + search_target__from_one_cell__refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": search_target__assuming_is_value__field_names__target_depth_reached
        + [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_target__is_field_name__values_for_target_depth_reached
        + search_target__is_field_name__values_for_refused,
    },
    "termination": {
        "assuming_cell_is_standalone": [] + search_termination__from_one_cell__refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [] + [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_termination__is_field_name__values_for_target_depth_reached
        + search_termination__is_field_name__values_for_refused,
    },
    "refusal": {
        "assuming_cell_is_standalone": [],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_refusal__is_field_name__values_for_target_depth_reached
        + search_refusal__is_field_name__values_for_refused,
    },
    # Search terms reason, comment, and remark often have the same key words
    "reason": {
        "assuming_cell_is_standalone": search_reason__from_one_cell__values_for_target_depth_reached
        + search_reason__from_one_cell__values_refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_target_depth_reached
        + search_reason__is_field_name__values_for_refused,
    },
    "reasons": {
        "assuming_cell_is_standalone": search_reason__from_one_cell__values_for_target_depth_reached
        + search_reason__from_one_cell__values_refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_target_depth_reached
        + search_reason__is_field_name__values_for_refused,
    },
    "comment": {
        "assuming_cell_is_standalone": search_comment__from_one_cell__values_for_target_depth_reached
        + search_reason__from_one_cell__values_refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_target_depth_reached
        + search_reason__is_field_name__values_for_refused,
    },
    "comments": {
        "assuming_cell_is_standalone": search_comment__from_one_cell__values_for_target_depth_reached
        + search_reason__from_one_cell__values_refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_target_depth_reached
        + search_reason__is_field_name__values_for_refused,
    },
    "remark": {
        "assuming_cell_is_standalone": search_comment__from_one_cell__values_for_target_depth_reached
        + search_reason__from_one_cell__values_refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_target_depth_reached
        + search_reason__is_field_name__values_for_refused,
    },
    "remarks": {
        "assuming_cell_is_standalone": search_comment__from_one_cell__values_for_target_depth_reached
        + search_reason__from_one_cell__values_refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_target_depth_reached
        + search_reason__is_field_name__values_for_refused,
    },
    "gwl": {
        "assuming_cell_is_standalone": search_gwl__incl_not_measured,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_gwl__incl_not_measured,
    },
    "swl": {  # An uncommon variation of gwl, likely indicating "static water level"
        "assuming_cell_is_standalone": search_gwl__incl_not_measured,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_gwl__incl_not_measured,
    },
    "ground water": {
        "assuming_cell_is_standalone": search_gwl__incl_not_measured,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_gwl__incl_not_measured,
    },
    "waterlevel": {
        "assuming_cell_is_standalone": search_gwl__incl_not_measured,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_gwl__incl_not_measured,
    },
    "water level": {
        "assuming_cell_is_standalone": search_gwl__incl_not_measured,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_gwl__incl_not_measured,
    },
    "groundwater level": {
        "assuming_cell_is_standalone": search_gwl__incl_not_measured,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_gwl__incl_not_measured,
    },
    "water table": {
        "assuming_cell_is_standalone": search_gwl__incl_not_measured,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_gwl__incl_not_measured,
    },
    "water depth": {
        "assuming_cell_is_standalone": search_gwl__incl_not_measured,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_gwl__incl_not_measured,
    },
    "area ratio": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX],
    },
    "arearatio": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX],
    },
    "conearearatio": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX],
    },
    "conetipnetarearatio": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX],
    },
    "cone tip net area ratio": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX],
    },
    "net surface area quotient of cone tip": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX],
    },
    "a factor": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX],
    },
    "alpha factor": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX],
    },
    "alphafactor": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX],
    },
    "predrill": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX],
    },
    "pre-drill": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX],
    },
    "predrilled": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX],
    },
}


target_depth_keywords_dict = {
    "target": {
        "assuming_cell_is_standalone": search_target__from_one_cell__target_depth_reached,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": search_target__assuming_is_value__field_names__target_depth_reached,
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_target__is_field_name__values_for_target_depth_reached,
    },
    "termination": {
        "assuming_cell_is_standalone": [],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_termination__is_field_name__values_for_target_depth_reached,
    },
    "refusal": {
        "assuming_cell_is_standalone": [],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_refusal__is_field_name__values_for_target_depth_reached,
    },
    # Search terms reason, comment, and remark often have the same key words
    "reason": {
        "assuming_cell_is_standalone": search_reason__from_one_cell__values_for_target_depth_reached,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_target_depth_reached,
    },
    "reasons": {
        "assuming_cell_is_standalone": search_reason__from_one_cell__values_for_target_depth_reached,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_target_depth_reached,
    },
    "comment": {
        "assuming_cell_is_standalone": search_comment__from_one_cell__values_for_target_depth_reached,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_target_depth_reached,
    },
    "comments": {
        "assuming_cell_is_standalone": search_comment__from_one_cell__values_for_target_depth_reached,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_target_depth_reached,
    },
    "remark": {
        "assuming_cell_is_standalone": search_comment__from_one_cell__values_for_target_depth_reached,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_target_depth_reached,
    },
    "remarks": {
        "assuming_cell_is_standalone": search_comment__from_one_cell__values_for_target_depth_reached,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_target_depth_reached,
    },
}

refused_keywords_dict = {
    "target": {
        "assuming_cell_is_standalone": search_target__from_one_cell__refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_target__is_field_name__values_for_refused,
    },
    "termination": {
        "assuming_cell_is_standalone": search_termination__from_one_cell__refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_termination__is_field_name__values_for_refused,
    },
    "refusal": {
        "assuming_cell_is_standalone": [],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_refusal__is_field_name__values_for_refused,
    },
    # Search terms reason, comment, and remark often have the same key words
    "reason": {
        "assuming_cell_is_standalone": search_reason__from_one_cell__values_refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_refused,
    },
    "reasons": {
        "assuming_cell_is_standalone": search_reason__from_one_cell__values_refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_refused,
    },
    "comment": {
        "assuming_cell_is_standalone": search_reason__from_one_cell__values_refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_refused,
    },
    "comments": {
        "assuming_cell_is_standalone": search_reason__from_one_cell__values_refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_refused,
    },
    "remark": {
        "assuming_cell_is_standalone": search_reason__from_one_cell__values_refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_refused,
    },
    "remarks": {
        "assuming_cell_is_standalone": search_reason__from_one_cell__values_refused,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_reason__is_field_name__values_for_refused,
    },
}


quantity_to_search_term = {
    QuantityToExtract.termination_reason: [
        "target",
        "termination",
        "refusal",
        "reason",
        "reasons",
        "comment",
        "comments",
        "remark",
        "remarks",
    ],
    QuantityToExtract.ground_water_level: [
        "gwl",
        "swl",
        "ground water",
        "waterlevel",
        "water level",
        "water table",
        "water depth",
        "groundwater level",
    ],
    QuantityToExtract.tip_net_area_ratio: [
        "alpha factor",
        "area ratio",
        "arearatio",
        "conearearatio",
        "conetipnetarearatio",
        "cone tip net area ratio",
        "net surface area quotient of cone tip",
        "a factor",
        "alpha factor",
        "alphafactor",
    ],
    QuantityToExtract.predrill_depth: [
        "predrill",
        "pre-drill",
        "pre-drill (m)",
        "predrilled",
    ],
}

# Values to SQLite ID mappings
SOIL_TYPE_TO_ID = {
    "ASH": 1,
    "ASPHALT": 2,
    "BASALT": 3,
    "BOULDERS": 4,
    "CLAY": 5,
    "COBBLES": 6,
    "CONCRETE": 7,
    "GRAVEL": 8,
    "PEAT": 9,
    "SAND": 10,
    "SANDSTONE": 11,
    "SILT": 12,
    "SILTSTONE": 13,
    "TOPSOIL": 14,
}

CPT_TERMINATION_REASON_TO_ID = {
    "refusal": 1,
    "target_depth": 2,
}

GROUND_WATER_LEVEL_METHOD_TO_ID = {
    "assumed": 1,
    "collapsed": 2,
    "derived": 3,
    "dipper": 4,
    "estimated": 5,
    "measured": 6,
}

CPT_TO_VS_CORRELATION_TO_ID = {
    "andrus_2007_holocene": 1,
    "andrus_2007_pleistocene": 2,
    "andrus_2007_tertiary_age_cooper_marl": 3,
    "hegazy_2006": 4,
    "mcgann_2015": 5,
    "mcgann_2018": 6,
    "robertson_2009": 7,
}

SPT_TO_VS_CORRELATION_TO_ID = {
    "brandenberg_2010": 1,
    "kwak_2015": 2,
}

# Correlation mappings for CPT and Vs30 correlations
VS_TO_VS30_CORRELATION_TO_ID = {
    "boore_2004": 1,
    "boore_2011": 2,
}

TYPE_TO_ID = {
    "CPT": 1,
    "BH": 2,
}

HAMMER_TYPE_TO_ID = {
    "Auto": 1,
    "Safety": 2,
    "Standard": 3,
}

# Default parameters for SPT calculations (loaded from config.yaml)
DEFAULT_GROUNDWATER_LEVEL_m = CONFIG["default_groundwater_level_m"]
DEFAULT_SPT_EFFICIENCY_PERCENT = CONFIG["default_spt_efficiency_percent"]
DEFAULT_BOREHOLE_DIAMETER_mm = CONFIG["default_borehole_diameter_mm"]
BUFFER_BELOW_LOWEST_MEASUREMENT_DEPTH_m = CONFIG[
    "buffer_below_lowest_measurement_depth_m"
]

# Density-related keywords found in AGS files (loaded from config.yaml)
DENSITY_RELATED_KEYWORDS = CONFIG["density_related_keywords"]

# Density keyword modifiers (ordered from most specific to least specific)
# Each tuple contains (regex_pattern, display_text)
# These modifiers can appear before density keywords to create compound phrases
DENSITY_MODIFIERS = [
    # Multi-word intensity modifiers (most specific)
    (r"very\s+high", "very high"),
    (r"very\s+low", "very low"),
    (r"very\s+well", "very well"),
    (r"very\s+poorly", "very poorly"),
    (r"medium\s+to\s+high", "medium to high"),
    (r"medium\s+to\s+low", "medium to low"),
    (r"medium\s+high", "medium high"),
    (r"loose\s+to\s+medium", "loose to medium"),
    (r"loose\s+to", "loose to"),
    (r"dense\s+to\s+very", "dense to very"),
    # Single-word intensity modifiers
    (r"very", "very"),
    (r"extremely", "extremely"),
    (r"highly", "highly"),
    (r"moderately", "moderately"),
    (r"slightly", "slightly"),
    (r"somewhat", "somewhat"),
    (r"quite", "quite"),
    (r"rather", "rather"),
    (r"fairly", "fairly"),
    (r"high", "high"),
    (r"medium", "medium"),
    (r"low", "low"),
    # Compaction/consolidation quality modifiers
    (r"well", "well"),
    (r"poorly", "poorly"),
    (r"strongly", "strongly"),
    (r"lightly", "lightly"),
    (r"heavily", "heavily"),
    (r"fully", "fully"),
    (r"partially", "partially"),
    (r"completely", "completely"),
    (r"incompletely", "incompletely"),
    (r"properly", "properly"),
    (r"improperly", "improperly"),
    (r"over", "over"),
    (r"under", "under"),
    # Packing/spacing modifiers
    (r"tightly", "tightly"),
    (r"loosely", "loosely"),
    (r"densely", "densely"),
    (r"sparsely", "sparsely"),
    (r"closely", "closely"),
    # Consistency/stiffness modifiers
    (r"firmly", "firmly"),
    (r"stiffly", "stiffly"),
    (r"softly", "softly"),
]

WATER_UNIT_WEIGHT_kN_m3 = 9.81

## The following constant was found to be added to all SPT depths in vs_calc with the following explanation:
# Spt testing driven a pile 18 inches into the ground in 3 incremental steps. the
# number of blows is ignored and only consider the total of the second and third increments. We interests
# in the vertical effective stress after second increments hence add 12 inches(0.3 m) on top of the start
# depth given
SPT_DEPTH_OFFSET_m = 0.3048

# Identify ",", ";", and "." not preceded or followed by a digit as phrase boundaries.
# Regex explanation for r"(?<!\d)\.(?!\d)":
#   (?<!\d) - negative lookbehind: matches if NOT preceded by a digit
#   \.      - matches a literal period (escaped because . is special in regex)
#   (?!\d)  - negative lookahead: matches if NOT followed by a digit
# This matches periods that are NOT part of decimal numbers (e.g., "3.14" won't match, but "a.b" will)
PHRASE_BOUNDARY_CHARACTERS = [",", ";", r"\|", r"(?<!\d)\.(?!\d)"]

LIKELY_PRECEDING_NUMERICAL_VALUES = [":", "="]
