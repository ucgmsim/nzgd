from nzgd.scripts.extract.cpt.filter_potential_cpt_supplemental_values import (
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
