import pandas as pd

from nzgd import constants
from nzgd.scripts.extract.cpt.filter_potential_cpt_supplemental_values import (
    ExtractedSingleValues,
    extract_numerical_quantity,
    extract_numerical_value,
)


def test_gwl_negative_sentinels_dropped():
    for sentinel in ("-30.00", "-60", "-100.0"):
        assert extract_numerical_value(sentinel, check_for_cm=True, is_gwl=True) is None


def test_gwl_real_negative_below_ground_kept():
    # below-ground sign convention: -1.2 m -> 1.2 m (NOT a sentinel)
    assert extract_numerical_value("-1.2", check_for_cm=True, is_gwl=True) == 1.2


def test_non_gwl_unaffected_by_sentinel_rule():
    # predrill/other: -30 still becomes 30 (no GWL sentinel handling)
    assert extract_numerical_value("-30.00", check_for_cm=True, is_gwl=False) == 30.0


def test_predrill_nil_becomes_zero():
    # A predrill candidate whose value is the literal "Nil" must yield predrill_depth == 0.0.
    # "Nil" means "no pre-drilling was performed" — equivalent to 0 m.
    df = pd.DataFrame(
        [
            {
                "nzgd_id": 1,
                "file_name": "a.xls",
                "sheet_name": "s",
                "likely_orientation": "columns",
                "search_term": "predrill",
                "search_assumption": "assuming_cell_is_a_field_name_in_need_of_a_value",
                "assumed_orientation": "columns",
                "field_label": "Pre-Drill:",
                "value": "Nil",
            }
        ]
    )
    result = extract_numerical_quantity(
        ExtractedSingleValues(),
        df,
        constants.QuantityToExtract.predrill_depth,
    )
    assert result.predrill_depth == 0.0
