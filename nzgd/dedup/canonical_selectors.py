"""Pluggable canonical-selection rules for within-record consolidation."""

from dataclasses import dataclass
from typing import Callable, Sequence

from nzgd.dedup.data_types import TableConfig


@dataclass(frozen=True)
class ClusterRow:
    """Compact summary of one cptreport/sptreport row for selector input."""

    report_id: int                  # cpt_id or spt_id
    has_data: bool                  # has_cpt_data=1 for CPT; measurement_row_count > 0 for SPT
    measurement_row_count: int
    metadata_non_null_count: int    # non-NULL fields in cptreport/sptreport metadata


CanonicalSelector = Callable[[Sequence[ClusterRow], TableConfig], int]


def default_within_record_canonical(
    cluster_rows: Sequence[ClusterRow],
    table_cfg: TableConfig,
) -> int:
    """v1 default: prefer rows with has_data=True; tiebreaker smallest report_id."""
    candidates = [r for r in cluster_rows if r.has_data] or list(cluster_rows)
    return min(candidates, key=lambda r: r.report_id).report_id
