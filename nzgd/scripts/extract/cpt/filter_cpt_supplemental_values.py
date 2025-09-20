"""Filter supplemental values extracted from CPT investigations.

These supplemental values are ground water level, tip net area ratio, and the reason
for terminating the CPT investigation.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from nzgd_data_extraction import constants


@dataclass
class ExtractedSingleValues:
    nzgd_id: int | None = None
    file_name: str | None = None
    sheet_name: str | None = None
    termination_reason: str | None = None
    ground_water_level: float | None = None
    ground_water_level_note: str | None = None
    tip_net_area_ratio: float | None = None


class ExtractedSingleValuesDict(dict):
    def __missing__(self, key):
        # Key must be a tuple of (nzgd_id, file_name, sheet_name)
        nzgd_id, file_name, sheet_name = key
        self[key] = ExtractedSingleValues(
            nzgd_id=nzgd_id,
            file_name=file_name,
            sheet_name=sheet_name,
        )
        return self[key]


def extract_numerical_value(
    s: str,
    check_for_cm: bool = False,
) -> float | None:
    """Extract numerical value from a string, optionally checking for cm.

    Parameters
    ----------
    s : str
        The input string from which to extract the numerical value.
    check_for_cm : bool
        Whether to check for "cm" units and convert to meters.

    Returns
    -------
    float | None
        Returns the numerical value, optionally divided by 100 if in cm, or None if no
        value is found.

    """
    match = re.search(constants.NUMERICAL_VALUES_REGEX, s)

    if not match:
        return None

    # Get the numerical value
    value = np.abs(float(match.group()))

    if check_for_cm:
        # Check if the value is followed by "cm" (with optional spaces)
        end_pos = match.end()
        remaining_text = s[end_pos:].strip()

        if remaining_text.lower().startswith("cm"):
            # Convert from cm to m
            return value / 100.0
        # return the value without conversion
    return value


def extract_closest_digits(s: str, idx: int) -> str:
    """Find all numerical values in the string and return the one closest to the given index.

    Returns an empty string if no digits are found.
    """
    # Find all numerical matches in the string
    matches = list(re.finditer(r"\d+\.?\d*", s))

    if not matches:
        return ""

    # If only one match, return it
    if len(matches) == 1:
        return matches[0].group()

    # Find the match with the closest distance to the index
    closest_match = None
    min_distance = float("inf")

    for match in matches:
        # Calculate distance from keyword index to start of numerical value
        distance = abs(match.start() - idx)
        if distance < min_distance:
            min_distance = distance
            closest_match = match

    return closest_match.group() if closest_match else ""


def gwl_method_flag(field_label: str) -> str | None:
    # catch "collapse" and "collapsed" e.g., CPT_189393_Raw01.CSV
    # and an observed misspelling
    if ("collapse" in field_label) or ("collapase" in field_label):
        return constants.GwlMethod.COLLAPSED
    # catch "estimate" and "estimated"
    if "estimate" in field_label:
        return constants.GwlMethod.ESTIMATED
    # catch "assume" and "assumed"
    if "assume" in field_label:
        return constants.GwlMethod.ASSUMED
    # catch "derive", "derived"
    if "derive" in field_label and "not" not in field_label:
        return constants.GwlMethod.DERIVED
    # catch "dip", "dipper", "dipped"
    if "dip" in field_label:
        return constants.GwlMethod.DIPPER
    # catch "measure" and "measured"
    if "measure" in field_label and "not" not in field_label:
        return constants.GwlMethod.MEASURED

    return None


def extract_gwl_or_tnar(
    extracted_values: ExtractedSingleValues,
    possible_df: pd.DataFrame,
    quantity_to_extract: constants.QuantityToExtract,
) -> ExtractedSingleValues:
    # Filter out rows that do not include a numerical value
    possible_df = possible_df[
        possible_df["value"].str.contains(constants.NUMERICAL_VALUES_REGEX, na=False)
    ]
    possible_df = possible_df.reset_index(drop=True)

    extracted_values_df = possible_df.copy()

    check_for_cm = False
    max_allowed_value = constants.MAX_ALLOWED_TNAR
    min_allowed_value = constants.MIN_ALLOWED_TNAR

    extracted_values_df[quantity_to_extract] = pd.NA
    if quantity_to_extract == constants.QuantityToExtract.ground_water_level:
        extracted_values_df["ground_water_level_note"] = pd.NA
        check_for_cm = True
        max_allowed_value = constants.MAX_ALLOWED_GWL
        min_allowed_value = constants.MIN_ALLOWED_GWL

    # Loop through all rows that include numerical values to find valid options
    for row_idx, row in possible_df.iterrows():
        # If the value is a string with multiple quantities, extract the number that is
        # closest to the keyword
        cell_contents = row["value"].lower()
        if any(
            quantity_to_search_term in cell_contents
            for quantity_to_search_term in quantity_to_search_term[quantity_to_extract]
        ):
            # Find which term is in the value
            for search_term in quantity_to_search_term[quantity_to_extract]:
                if search_term in cell_contents:
                    idx_in_str = cell_contents.find(search_term)
                    break

            closest_digit_str = extract_closest_digits(cell_contents, idx_in_str)
            if closest_digit_str:
                value = extract_numerical_value(
                    cell_contents[cell_contents.find(closest_digit_str) :],
                    check_for_cm=check_for_cm,
                )
                if value is not None:
                    extracted_values_df.loc[row_idx, quantity_to_extract] = value

                    if (
                        quantity_to_extract
                        == constants.QuantityToExtract.ground_water_level
                    ):
                        extracted_values_df.loc[row_idx, "ground_water_level_note"] = (
                            gwl_method_flag(cell_contents)
                        )
                    continue

        # The cell_contents does not contain keywords, so just extract the only value
        # numerical value
        extracted_values_df.loc[row_idx, quantity_to_extract] = extract_numerical_value(
            cell_contents,
            check_for_cm=check_for_cm,
        )

        if quantity_to_extract == constants.QuantityToExtract.ground_water_level:
            # check both field label and cell contents for a ground water level note
            gwl_note = gwl_method_flag(row["field_label"].lower())
            if gwl_note is None:
                gwl_note = gwl_method_flag(cell_contents)
            extracted_values_df.loc[row_idx, "ground_water_level_note"] = gwl_note

    # Now extracted_values_df has floats or pd.NA, so it can be filtered to only keep
    # rows where MIN_ALLOWED <= numerical_value <= MAX_ALLOWED
    extracted_values_df = extracted_values_df[
        (extracted_values_df[quantity_to_extract] >= min_allowed_value)
        & (extracted_values_df[quantity_to_extract] <= max_allowed_value)
    ]

    # If no rows remain after filtering, return unchanged extracted_values
    if len(extracted_values_df) == 0:
        return extracted_values

    # If there is more than one option, preferentially keep the ones that were
    # extracted with either no assumed orientation (for stand alone cells) or with
    # the assumed orientation equal to the likely orientation.
    # But the row with assumed orientation not equal to the likely orientation will
    # be used if there are no other valid options.

    selected_bool_mask = (
        extracted_values_df["likely_orientation"]
        == extracted_values_df["assumed_orientation"]
    ) | (extracted_values_df["assumed_orientation"].isna())

    if not any(selected_bool_mask):
        selected_bool_mask = ~selected_bool_mask

    extracted_values_df = extracted_values_df[selected_bool_mask]

    # If there is multiple options, all are valid, so just return the first one

    if quantity_to_extract == constants.QuantityToExtract.ground_water_level:
        extracted_values.ground_water_level = extracted_values_df[
            quantity_to_extract
        ].iloc[0]

        extracted_values.ground_water_level_note = extracted_values_df[
            "ground_water_level_note"
        ].iloc[0]

    if quantity_to_extract == constants.QuantityToExtract.tip_net_area_ratio:
        extracted_values.tip_net_area_ratio = extracted_values_df[
            quantity_to_extract
        ].iloc[0]

    return extracted_values


def extract_termination_reason(
    extracted_values: ExtractedSingleValues,
    possible_df: pd.DataFrame,
) -> ExtractedSingleValues:
    termination_reasons = []

    for _, row in possible_df.iterrows():
        # If the search assumption is `assuming_cell_is_a_value_in_need_of_field_name_to_confirm`
        # we we need to check if the `field_label` is in the refused or target depth
        # dictionaries. For the other search assumptions, we need to check if the found
        # value is in those dictionaries.
        if (
            row["search_assumption"]
            == "assuming_cell_is_a_value_in_need_of_field_name_to_confirm"
        ):
            if (
                row["field_label"].lower()
                in constants.refused_keywords_dict[row["search_term"]][
                    row["search_assumption"]
                ]
            ):
                termination_reasons.append(constants.TerminationReason.REFUSAL)

            if (
                row["field_label"].lower()
                in constants.target_depth_keywords_dict[row["search_term"]][
                    row["search_assumption"]
                ]
            ):
                termination_reasons.append(constants.TerminationReason.TARGET_DEPTH)
        elif (
            row["search_assumption"]
            == "assuming_cell_is_a_field_name_in_need_of_a_value"
        ):
            if (
                row["value"].lower()
                in constants.refused_keywords_dict[row["search_term"]][
                    row["search_assumption"]
                ]
            ) or "refus" in row[
                "value"
            ].lower():  # refus to catch refusal, refuses, refused
                termination_reasons.append(constants.TerminationReason.REFUSAL)

            if (
                (
                    row["value"].lower()
                    in constants.target_depth_keywords_dict[row["search_term"]][
                        row["search_assumption"]
                    ]
                )
                or "target" in row["value"].lower()
                or "reached" in row["value"].lower()
            ):
                termination_reasons.append(constants.TerminationReason.TARGET_DEPTH)

        # The only remaining case is if it is assumed to be a standalone cell
        else:
            if (
                row["value"].lower()
                in constants.refused_keywords_dict[row["search_term"]][
                    row["search_assumption"]
                ]
            ):
                termination_reasons.append(constants.TerminationReason.REFUSAL)

            if (
                row["value"].lower()
                in constants.target_depth_keywords_dict[row["search_term"]][
                    row["search_assumption"]
                ]
            ):
                termination_reasons.append(constants.TerminationReason.TARGET_DEPTH)

    if constants.TerminationReason.REFUSAL in termination_reasons:
        extracted_values.termination_reason = constants.TerminationReason.REFUSAL
        return extracted_values

    if constants.TerminationReason.TARGET_DEPTH in termination_reasons:
        extracted_values.termination_reason = constants.TerminationReason.TARGET_DEPTH
        return extracted_values

    return extracted_values


quantity_to_search_term = {
    "termination_reason": [
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
    "ground_water_level": [
        "gwl",
        "swl",
        "ground water",
        "waterlevel",
        "water level",
        "water table",
        "water depth",
        "groundwater level",
    ],
    "tip_net_area_ratio": [
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
}

investigation_type = "cpt"
# investigation_type = "scpt"

working_dir = Path(
    "/home/arr65/data/nzgd/extracted_single_values_V3/all_possible_values",
)

all_options_df = pd.read_csv(
    working_dir / f"{investigation_type}_v9.csv",
)

# Create reverse mapping from search term to quantity description
search_term_to_quantity = {}
for quantity_desc, search_terms in quantity_to_search_term.items():
    for search_term in search_terms:
        search_term_to_quantity[search_term] = quantity_desc

# Add quantity_description column based on search_term
all_options_df["quantity_description"] = all_options_df["search_term"].map(
    search_term_to_quantity,
)

unique_nzgd_id = all_options_df["nzgd_id"].unique()

# Initialize the dictionary to store extracted values
extracted_values_dict = ExtractedSingleValuesDict()
# Dictionary to track cases with multiple possible values
# Key: (nzgd_id, file_name, sheet_name, quantity_description), Value: count
multiple_values_count = {}
for nzgd_id in tqdm(unique_nzgd_id):
    per_nzgd_id = all_options_df[all_options_df["nzgd_id"] == nzgd_id]

    unique_file_names = per_nzgd_id["file_name"].unique()

    for file_name in unique_file_names:
        per_file_name = per_nzgd_id[per_nzgd_id["file_name"] == file_name]

        unique_sheet_names = per_file_name["sheet_name"].unique()

        for sheet_name in unique_sheet_names:
            per_sheet = per_file_name[per_file_name["sheet_name"] == sheet_name]

            # Get the ExtractedSingleValues instance for this combination
            key = (nzgd_id, file_name, sheet_name)
            extracted_values = extracted_values_dict[key]

            unique_quantity_descriptions = per_sheet["quantity_description"].unique()

            for quantity_description in unique_quantity_descriptions:
                if quantity_description is None:
                    continue

                per_quantity_description = per_sheet[
                    per_sheet["quantity_description"] == quantity_description
                ].copy()

                # There are no valid options, so just continue to the next check
                if len(per_quantity_description) == 0:
                    continue

                per_quantity_description["value"] = per_quantity_description[
                    "value"
                ].astype(str)

                if quantity_description == "termination_reason":
                    extracted_values = extract_termination_reason(
                        extracted_values,
                        per_quantity_description,
                    )

                if quantity_description == "ground_water_level":
                    extracted_values = extract_gwl_or_tnar(
                        extracted_values,
                        per_quantity_description,
                        constants.QuantityToExtract.ground_water_level,
                    )

                if quantity_description == "tip_net_area_ratio":
                    extracted_values = extract_gwl_or_tnar(
                        extracted_values,
                        per_quantity_description,
                        constants.QuantityToExtract.tip_net_area_ratio,
                    )

            extracted_values_dict[key] = extracted_values

# Prepare data for DataFrame
df_data = []
for key, values in extracted_values_dict.items():
    nzgd_id, file_name, sheet_name = key
    row_data = {
        "nzgd_id": nzgd_id,
        "file_name": file_name,
        "sheet_name": sheet_name,
        "termination_reason": values.termination_reason,
        "ground_water_level": values.ground_water_level,
        "gwl_method": values.ground_water_level_note,
        "tip_net_area_ratio": values.tip_net_area_ratio,
    }
    df_data.append(row_data)

# Create DataFrame
extracted_df = pd.DataFrame(df_data)

extracted_df.to_csv(
    working_dir.parent / f"{investigation_type}_v9.csv",
    index=False,
)
