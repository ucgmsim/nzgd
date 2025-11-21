"""Filter supplemental values extracted from CPT investigations.

These supplemental values are ground water level, tip net area ratio, and the reason
for terminating the CPT investigation.
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from nzgd import constants


@dataclass
class ExtractedSingleValues:
    """Container for extracted supplemental values from a single CPT investigation.

    This dataclass holds the extracted values for various supplemental parameters
    associated with a specific CPT investigation identified by nzgd_id, filename,
    and sheet_name.

    Attributes
    ----------
    nzgd_id : int | None
        The unique identifier for the CPT investigation.
    filename : str | None
        The name of the file from which the values were extracted.
    sheet_name : str | None
        The name of the sheet within the file.
    termination_reason : str | None
        The reason for terminating the CPT investigation (e.g., 'refusal', 'target_depth').
    ground_water_level : float | None
        The extracted ground water level value in meters.
    ground_water_level_note : str | None
        Additional notes about the ground water level method (e.g., 'measured', 'estimated').
    tip_net_area_ratio : float | None
        The tip net area ratio of the CPT cone.
    predrill_depth : float | None
        The predrill depth in meters.
    """

    nzgd_id: int | None = None
    filename: str | None = None
    sheet_name: str | None = None
    termination_reason: str | None = None
    ground_water_level: float | None = None
    ground_water_level_note: str | None = None
    tip_net_area_ratio: float | None = None
    predrill_depth: float | None = None


class ExtractedSingleValuesDict(dict):
    """A dictionary subclass that automatically creates ExtractedSingleValues instances.

    This dictionary is keyed by tuples of (nzgd_id, filename, sheet_name). When a key
    is accessed that doesn't exist, it automatically creates a new ExtractedSingleValues
    instance with the key components as the initial values for nzgd_id, filename, and
    sheet_name.

    This allows for lazy initialization of extracted values for each unique CPT
    investigation file/sheet combination.
    """

    def __missing__(self, key: tuple[int, str, str]) -> ExtractedSingleValues:
        """Create and return a new ExtractedSingleValues instance for the given key.

        Parameters
        ----------
        key : tuple[int, str, str]
            A tuple containing (nzgd_id, filename, sheet_name).

        Returns
        -------
        ExtractedSingleValues
            A new instance initialized with the key components.
        """
        # Key must be a tuple of (nzgd_id, filename, sheet_name)
        nzgd_id, filename, sheet_name = key
        self[key] = ExtractedSingleValues(
            nzgd_id=nzgd_id,
            filename=filename,
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


def extract_associated_value_from_str(s: str, idx: int) -> str:
    """Find all numerical values in the string and return the one closest to the given index.

    Returns an empty string if no digits are found.
    """
    # Find all numerical matches in the string
    matches = list(re.finditer(constants.NUMERICAL_VALUES_REGEX, s))

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
    """Determine the ground water level method flag based on the field label.

    This function analyzes the field label string to identify keywords that indicate
    the method used to determine the ground water level, such as measured, estimated,
    assumed, etc.

    Parameters
    ----------
    field_label : str
        The field label string to analyze for method keywords.

    Returns
    -------
    str | None
        The corresponding GwlMethod constant if a keyword is found, otherwise None.
    """
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


def extract_numerical_quantity(
    extracted_values: ExtractedSingleValues,
    possible_values_df: pd.DataFrame,
    quantity_to_extract: constants.QuantityToExtract,
) -> ExtractedSingleValues:
    """Extract a numerical quantity from possible values DataFrame.

    Parameters
    ----------
    extracted_values : ExtractedSingleValues
        The current extracted values to be updated.
    possible_values_df : pd.DataFrame
        DataFrame containing possible extracted values for the specified quantity.
    quantity_to_extract : constants.QuantityToExtract
        The specific quantity to extract (e.g., ground water level, tip net area ratio).

    Returns
    -------
    ExtractedSingleValues
        Updated extracted values with the specified quantity if found.
    """

    # Filter out rows that do not include a numerical value
    possible_values_df = possible_values_df[
        possible_values_df["value"].str.contains(
            constants.NUMERICAL_VALUES_REGEX, na=False, regex=True
        )
    ].reset_index(drop=True)

    extracted_values_df = possible_values_df.copy()

    # Initialize a column to store the extracted numerical value
    extracted_values_df[quantity_to_extract] = pd.NA

    check_for_cm = True
    if quantity_to_extract == constants.QuantityToExtract.ground_water_level:
        extracted_values_df["ground_water_level_note"] = pd.NA
        max_allowed_value = constants.MAX_ALLOWED_GWL
        min_allowed_value = constants.MIN_ALLOWED_GWL

    elif quantity_to_extract == constants.QuantityToExtract.predrill_depth:
        max_allowed_value = constants.MAX_ALLOWED_PREDRILL_DEPTH
        min_allowed_value = constants.MIN_ALLOWED_PREDRILL_DEPTH

    else:  # tip_net_area_ratio
        check_for_cm = False
        max_allowed_value = constants.MAX_ALLOWED_TIP_NET_AREA_RATIO
        min_allowed_value = constants.MIN_ALLOWED_TIP_NET_AREA_RATIO

    search_terms_for_all_quantities = [
        term
        for term_list in constants.quantity_to_search_term.values()
        for term in term_list
    ]

    # Examine each row of possible values in detail
    for row_idx, row in possible_values_df.iterrows():
        cell_contents = row["value"].lower()
        # If the cell contents (containing possible values) is a string that
        # includes at least one key word, we extract the numerical value that
        # is most likely associated with the quantity.

        indices_of_quantity_search_terms_in_cell_contents = [
            cell_contents.find(search_term)
            for search_term in constants.quantity_to_search_term[quantity_to_extract]
        ]

        quantity_search_term_is_present_bool = (
            np.array(indices_of_quantity_search_terms_in_cell_contents) >= 0
        )

        if any(quantity_search_term_is_present_bool):
            # Get the search term that is present in the cell contents.
            # np.array(constants.quantity_to_search_term[quantity_to_extract]) is an array so get the first element with [0]
            # If multiple search terms are present, any will do, so just use the first element [0]

            # found_quantity_search_term = str(
            #     np.array(constants.quantity_to_search_term[quantity_to_extract])[
            #         quantity_search_term_is_present_bool
            #     ][0]
            # )

            found_quantity_search_term_idx = np.array(
                indices_of_quantity_search_terms_in_cell_contents
            )[quantity_search_term_is_present_bool][0]

            # Find the minimum index of any present search term for any quantity in the cell contents
            # to inform if numerical values are before or after search terms.

            min_idx_of_any_present_quantity_search_term_in_cell_contents = min(
                [
                    cell_contents.find(search_term)
                    for search_term in search_terms_for_all_quantities
                    if cell_contents.find(search_term) != -1
                ]
            )

            # Get index of first numerical value. If no numerical value is found, the value is None.
            first_num_match = re.search(constants.NUMERICAL_VALUES_REGEX, cell_contents)
            first_num_idx = first_num_match.start() if first_num_match else None

            if first_num_idx is None:
                values_before_search_term = False
            else:
                values_before_search_term = (
                    first_num_idx
                    < min_idx_of_any_present_quantity_search_term_in_cell_contents
                )

            if values_before_search_term:
                # We need to get the last numerical value found in the slice of the cell contents from 0 to the found quantity search term index
                slice_idxs = (0, found_quantity_search_term_idx)

                # Get the last match in matches from this slice
                needed_idx_for_matches = -1
            else:
                # We need to get the first numerical value found in the slice of the cell centents from the found quantity search term index to the end of the cell contents
                slice_idxs = (found_quantity_search_term_idx, len(cell_contents))

                # Get the first match in matches from this slice
                needed_idx_for_matches = 0

            matches = [
                m
                for m in re.finditer(
                    constants.NUMERICAL_VALUES_REGEX,
                    cell_contents[slice_idxs[0] : slice_idxs[1]],
                )
            ]

            # If no matches are found, just continue to the next row
            if len(matches) == 0:
                continue

            # Get the numerical value that is closest to the found quantity search term
            closest_numerical_value_str = matches[needed_idx_for_matches].group()

            if closest_numerical_value_str:
                extracted_values_df.at[row_idx, str(quantity_to_extract)] = (  # type: ignore  # Pylance is overly strict about pandas indexing types
                    extract_numerical_value(
                        cell_contents[
                            cell_contents.find(closest_numerical_value_str) :
                        ],
                        check_for_cm=check_for_cm,
                    )
                )
                # If extracting ground water level, also check for a method note
                if (
                    quantity_to_extract
                    == constants.QuantityToExtract.ground_water_level
                ):
                    extracted_values_df.at[row_idx, "ground_water_level_note"] = (  # type: ignore  # Pylance is overly strict about pandas indexing types
                        gwl_method_flag(cell_contents)
                    )

        # The cell_contents do not contain keywords, so just extract the numerical value
        else:
            extracted_values_df.at[row_idx, str(quantity_to_extract)] = (  # type: ignore  # Pylance is overly strict about pandas indexing types
                extract_numerical_value(
                    cell_contents,
                    check_for_cm=check_for_cm,
                )
            )

            # If extracting ground water level, also check for a method note
            if quantity_to_extract == constants.QuantityToExtract.ground_water_level:
                # check both field label and cell contents for a ground water level note
                gwl_note = gwl_method_flag(row["field_label"].lower())
                if gwl_note is None:
                    gwl_note = gwl_method_flag(cell_contents)
                extracted_values_df.at[row_idx, "ground_water_level_note"] = gwl_note  # type: ignore  # Pylance is overly strict about pandas indexing types

    # Now extracted_values_df[quantity_to_extract] has floats or pd.NA, so it can be
    # filtered to only keep rows where MIN_ALLOWED <= numerical_value <= MAX_ALLOWED
    extracted_values_df = extracted_values_df[
        (extracted_values_df[quantity_to_extract] >= min_allowed_value)
        & (extracted_values_df[quantity_to_extract] <= max_allowed_value)
    ]

    # If no rows remain after filtering, return the unchanged extracted_values object
    if len(extracted_values_df) == 0:
        return extracted_values

    # If there several valid options, preferentially use the ones that were
    # extracted with either no assumed orientation (for stand alone cells) or with
    # the assumed orientation equal to the likely orientation.
    # Rows with assumed orientation not equal to the likely orientation will only be
    # used if there are no other valid options.
    selected_bool_mask = (
        extracted_values_df["likely_orientation"]
        == extracted_values_df["assumed_orientation"]
    ) | (extracted_values_df["assumed_orientation"].isna())

    # If no preferred rows exist (all False in selected_bool_mask), fall back to
    # considering all rows by flipping the values in selected_bool_mask from False to True
    if not any(selected_bool_mask):
        selected_bool_mask = ~selected_bool_mask

    extracted_values_df = extracted_values_df[selected_bool_mask]

    # If there are multiple remaining rows in extracted_values_df, all options should
    # be valid so just select the first one. The selected value will be assigned to the
    # attribute of the extracted_values container that corresponds to the quantity
    # being extracted, so we check for each possible quantity.

    if quantity_to_extract == constants.QuantityToExtract.ground_water_level:
        # for ground water level, also extract the method note
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

    if quantity_to_extract == constants.QuantityToExtract.predrill_depth:
        extracted_values.predrill_depth = extracted_values_df[quantity_to_extract].iloc[
            0
        ]

    return extracted_values


def extract_termination_reason(
    extracted_values: ExtractedSingleValues,
    possible_df: pd.DataFrame,
) -> ExtractedSingleValues:
    """
    Extract the termination reason from previously extracted possible values.

    Parameters
    ----------
    extracted_values : ExtractedSingleValues
        The current extracted values to be updated.
    possible_df : pd.DataFrame
        DataFrame containing possible extracted values for termination reason.

    Returns
    -------
    ExtractedSingleValues
        Updated extracted values with termination reason if found.

    """
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


all_options_df = pd.read_csv(
    constants.SUPPLEMENTAL_VALUES_OUTPUT_DIR
    / constants.ALL_POTENTIAL_CPT_SUPPLEMENTAL_VALUES_FILENAME,
)

# Create reverse mapping from search term to quantity description
search_term_to_quantity = {}
for quantity_desc, search_terms in constants.quantity_to_search_term.items():
    for search_term in search_terms:
        search_term_to_quantity[search_term] = quantity_desc

# Add quantity column based on search_term
all_options_df["quantity"] = all_options_df["search_term"].map(
    search_term_to_quantity,
)

unique_nzgd_id = all_options_df["nzgd_id"].unique()

# Initialize the dictionary to store extracted values
extracted_values_dict = ExtractedSingleValuesDict()
# Dictionary to track cases with multiple possible values
# Key: (nzgd_id, filename, sheet_name, quantity), Value: count
multiple_values_count = {}
for nzgd_id in tqdm(unique_nzgd_id):
    per_nzgd_id_df = all_options_df[all_options_df["nzgd_id"] == nzgd_id]

    unique_file_names = per_nzgd_id_df["file_name"].unique()

    for filename in unique_file_names:
        per_filename_nzgd_id_df = per_nzgd_id_df[
            per_nzgd_id_df["file_name"] == filename
        ]

        unique_sheet_names = per_filename_nzgd_id_df["sheet_name"].unique()

        for sheet_name in unique_sheet_names:
            per_sheet_filename_id_df = per_filename_nzgd_id_df[
                per_filename_nzgd_id_df["sheet_name"] == sheet_name
            ]

            # Get the ExtractedSingleValues instance for this combination
            key = (nzgd_id, filename, sheet_name)
            extracted_values_from_sheet_filename_id = extracted_values_dict[key]

            unique_quantities = per_sheet_filename_id_df["quantity"].unique()

            # For each unique (nzgd_id, filename, sheet_name) combination, select a
            # possible extracted value for each quantity.
            # The same ExtractedSingleValues instance is updated for each quantity.
            for quantity in unique_quantities:
                if quantity is None:
                    continue

                per_quantity_sheet_filename_id_df = per_sheet_filename_id_df[
                    per_sheet_filename_id_df["quantity"] == quantity
                ].copy()

                # There are no valid options, so just continue to the next check
                if len(per_quantity_sheet_filename_id_df) == 0:
                    continue
                per_quantity_sheet_filename_id_df["value"] = (
                    per_quantity_sheet_filename_id_df["value"].astype(str)
                )

                if quantity == constants.QuantityToExtract.termination_reason:
                    extracted_values_from_sheet_filename_id = (
                        extract_termination_reason(
                            extracted_values_from_sheet_filename_id,
                            per_quantity_sheet_filename_id_df,
                        )
                    )

                else:
                    extracted_values_from_sheet_filename_id = (
                        extract_numerical_quantity(
                            extracted_values_from_sheet_filename_id,
                            per_quantity_sheet_filename_id_df,
                            quantity,
                        )
                    )

            extracted_values_dict[key] = extracted_values_from_sheet_filename_id

# Prepare data for DataFrame
df_data = []
for key, values in extracted_values_dict.items():
    nzgd_id, file_name, sheet_name = key
    row_data = {
        "nzgd_id": nzgd_id,
        "file_name": filename,
        "sheet_name": sheet_name,
        "termination_reason": values.termination_reason,
        "ground_water_level": values.ground_water_level,
        "gwl_method": values.ground_water_level_note,
        "tip_net_area_ratio": values.tip_net_area_ratio,
        "predrill_depth": values.predrill_depth,
    }
    df_data.append(row_data)

# Create DataFrame
extracted_df = pd.DataFrame(df_data)

extracted_df.to_csv(
    constants.SUPPLEMENTAL_VALUES_OUTPUT_DIR
    / constants.CPT_SUPPLEMENTAL_VALUES_FILENAME,
    index=False,
)
