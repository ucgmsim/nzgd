"""Data types shared across dedup passes and the executor."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TableConfig:
    """Per-record-type table names so dedup code is parameterised over CPT vs SPT."""

    record_type: str  # 'CPT' or 'BH'
    report_table: str  # 'cptreport' or 'sptreport'
    measurement_table: str  # 'cptmeasurements' or 'sptmeasurements'
    report_id_column: str  # 'cpt_id' or 'spt_id'
    measurement_value_columns: tuple[str, ...]
    # Tables that must be deleted in this order before deleting the report row
    # itself. Each entry is (table_name, fk_column). Innermost dependents first.
    dependent_tables: tuple[tuple[str, str], ...]


CPT_TABLE_CONFIG = TableConfig(
    record_type="CPT",
    report_table="cptreport",
    measurement_table="cptmeasurements",
    report_id_column="cpt_id",
    measurement_value_columns=("depth_m", "qc_MPa", "fs_MPa", "u2_MPa"),
    dependent_tables=(
        ("cptvs30estimates", "cpt_id"),
        ("cptmeasurements", "cpt_id"),
    ),
)

SPT_TABLE_CONFIG = TableConfig(
    record_type="BH",
    report_table="sptreport",
    measurement_table="sptmeasurements",
    report_id_column="spt_id",
    measurement_value_columns=("depth_m", "ISPT_MAIN", "ISPT_NVAL", "ISPT_REP"),
    dependent_tables=(
        # soilmeasurementsoiltype joins by soil_measurement_id; the executor
        # handles its cascade explicitly via a subquery rather than a flat list.
        ("soilmeasurements", "spt_id"),
        ("densitymeasurements", "spt_id"),
        ("sptvs30estimates", "spt_id"),
        ("sptmeasurements", "spt_id"),
    ),
)


@dataclass(frozen=True)
class ReportPairMatch:
    """One (canonical_report_id, merged_report_id) pair identified by a pass."""

    canonical_report_id: int
    merged_report_id: int
    # Free-form metrics: {"hash": "<hex>"} for hash matches; full feature vector for fuzzy.
    metrics: dict[str, Any]


@dataclass(frozen=True)
class MergePlanEntry:
    """One (canonical, merged) pair within a cluster. A 3-way cluster produces 2 entries."""

    cluster_id: int
    canonical_nzgd_id: int
    merged_nzgd_id: int
    record_type: str
    match_pass: str  # 'hash' or 'fuzzy'
    matched_pairs: list[ReportPairMatch]  # reports to delete from merged record
    unique_merged_report_ids: list[int] = field(default_factory=list)  # reports to re-parent


@dataclass(frozen=True)
class QualityRejectEntry:
    """One CPT report discarded by the constant-column quality filter."""

    record_type: str
    nzgd_id: int
    report_id: int
    reason: str                          # 'constant_column'
    constant_columns: dict[str, float]   # {column_name: constant_value}
    n_rows: int
