"""Constants and configuration for NZGD data extraction."""

import enum
import importlib.resources
from pathlib import Path

import numpy as np
import yaml

# Define the resource directory
RESOURCE_PATH = importlib.resources.files("nzgd") / "resources"

INDEX_FILE_PATH = RESOURCE_PATH / "nzgd_metadata_from_coordinates_22_august_2025.csv"

# Load all configurations from the YAML file
with (RESOURCE_PATH / "config.yaml").open() as f:
    CONFIG = yaml.safe_load(f)

CPT_TRACE_OUTPUT_DIR = Path(CONFIG["base_output_dir"]) / Path(
    CONFIG["subdirectories"]["cpt_trace"]
)

FAILED_CPT_TRACE_OUTPUT_DIR = Path(CONFIG["base_output_dir"]) / Path(
    CONFIG["subdirectories"]["failed_cpt_trace"]
)


SUPPLEMENTAL_VALUES_OUTPUT_DIR = Path(CONFIG["base_output_dir"]) / Path(
    CONFIG["subdirectories"]["supplemental_values"]
)

NZGD_SOURCE_DATA_DIR = Path(
    CONFIG["nzgd_source_data_dir"],
)

OUTPUT_DB_PATH = CONFIG["output_db_path"]

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
    """Enumerations to describe or numeric or text data.

    Attributes
    ----------
    ground_water_level : enum.auto()
        Represents numeric cells.
    tip_net_area_ratio : enum.auto()
        Represents text cells.

    """

    ground_water_level = "ground_water_level"
    tip_net_area_ratio = "tip_net_area_ratio"


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

MAX_ALLOWED_GWL = 40
MIN_ALLOWED_GWL = 0

MAX_ALLOWED_TNAR = 1.0
MIN_ALLOWED_TNAR = 0.2

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
    "alpha factor": {
        "assuming_cell_is_standalone": [NUMERICAL_VALUES_REGEX],
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": [NUMERICAL_VALUES_REGEX],
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
    "waterlevel": {
        "assuming_cell_is_standalone": search_gwl__incl_not_measured,
        "assuming_cell_is_a_value_in_need_of_field_name_to_confirm": [],
        "assuming_cell_is_a_field_name_in_need_of_a_value": search_gwl__incl_not_measured,
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
    "pre-drill (m)": {
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


# Values to SQLite ID mappings
SOIL_TYPE_TO_ID = {
    "BOULDERS": 1,
    "CLAY": 2,
    "COBBLES": 3,
    "GRAVEL": 4,
    "SAND": 5,
    "SILT": 6,
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
