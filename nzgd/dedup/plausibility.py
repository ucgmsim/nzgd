"""Per-field plausibility check shared by all dedup passes."""

from typing import Any

from nzgd import constants


def is_useful_value(value: Any, table: str, column: str) -> bool:
    """Return True if `value` is non-NULL and (if a range is configured) within range.

    `table` is 'nzgdrecord', 'cptreport', or 'sptreport'. `column` is a column
    name within that table.

    A `None` value is never useful. A numeric value is checked against the
    configured plausibility range (inclusive) for `(table, column)`; if no
    range is configured, the value is treated as useful. Non-numeric values
    (text fields, dates) bypass the range check entirely — only the non-NULL
    check applies.
    """
    if value is None:
        return False
    ranges = constants.DEDUP_CONFIG.get("field_plausibility_ranges", {}).get(table, {})
    if column not in ranges:
        return True
    lo, hi = ranges[column]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    return lo <= v <= hi
