"""Post-dedup within-record supplemental-value consolidation.

After the dedup passes, one nzgd_id can have several surviving cptreport rows
(format-sibling files that landed in different trace clusters). A supplemental
value extracted from one sibling never reaches the others. This step fills each
record's surviving rows with the record's best-available value per supplemental
column — faithful to the sources: prefer the single in-range value, else the
single recorded value (e.g. a literal 0), else leave NULL. Small-spread valid
conflicts are resolved by a corroboration-first selector; large-spread (likely
artifact) conflicts are skipped and reported. Only NULL/non-useful cells are
filled; a valid value is never overridden.
"""

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone

from nzgd import constants
from nzgd.dedup.data_types import TableConfig
from nzgd.dedup.plausibility import is_useful_value

# Supplemental columns consolidated per record (CPT). gwl_method_id is filled
# from the same source row as extracted_gwl_m, not consolidated independently.
_SUPPLEMENTAL_COLUMNS = (
    "extracted_gwl_m",
    "tip_net_area_ratio",
    "predrill_depth_m",
    "termination_reason_id",
)
_GWL_METHOD_COLUMN = "gwl_method_id"


def _decimal_places(value: float) -> int:
    """Decimal places of `value` after rounding to 3 dp (guards against float noise)."""
    text = f"{round(float(value), 3):.3f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text and text.split(".")[1] else 0


def select_value(candidates: list[tuple[float, int]]) -> float:
    """Pick a consolidated value from (value, cpt_id) candidates of small spread.

    Corroboration first (the value on the most rows), then most decimal places
    (rounded to 3 dp), then smallest cpt_id. Only ever returns a value present in
    `candidates`; unbiased; deterministic.
    """
    counts = Counter(v for v, _ in candidates)
    best = max(counts.values())
    tied = [v for v in counts if counts[v] == best]
    if len(tied) == 1:
        return tied[0]

    def smallest_cpt(v: float) -> int:
        return min(cid for vv, cid in candidates if vv == v)

    return sorted(tied, key=lambda v: (-_decimal_places(v), smallest_cpt(v)))[0]
