"""Trace-comparison helpers shared between the dedup passes."""

import math
import sqlite3
from collections import defaultdict

import numpy as np

from nzgd.dedup.data_types import TableConfig


def coerce_to_float(v) -> float:
    """Coerce a SQLite cell to float; non-numeric strings become NaN."""
    if v is None:
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def load_traces(
    conn: sqlite3.Connection, nzgd_id: int, table_cfg: TableConfig
) -> dict[int, np.ndarray]:
    """Return `{report_id: ndarray of shape (n_rows, len(value_columns))}` for one nzgd_id.

    Columns appear in the order given by `table_cfg.measurement_value_columns`;
    the first must be `depth_m` so column 0 of each array is depth. Non-numeric
    measurement values (e.g., SPT `ISPT_REP` blow-count strings) become NaN.
    """
    value_cols_with_m = ", ".join(f"m.{c}" for c in table_cfg.measurement_value_columns)
    cur = conn.cursor()
    cur.execute(
        f"SELECT r.{table_cfg.report_id_column}, {value_cols_with_m} "
        f"FROM {table_cfg.measurement_table} m "
        f"JOIN {table_cfg.report_table} r "
        f"ON r.{table_cfg.report_id_column} = m.{table_cfg.report_id_column} "
        f"WHERE r.nzgd_id = ? "
        f"ORDER BY r.{table_cfg.report_id_column}, m.depth_m",
        (nzgd_id,),
    )
    rows_by_report: dict[int, list[tuple]] = defaultdict(list)
    for row in cur.fetchall():
        rid = row[0]
        rows_by_report[rid].append(tuple(coerce_to_float(v) for v in row[1:]))
    return {rid: np.array(rows, dtype=float) for rid, rows in rows_by_report.items()}


def trace_score(a: np.ndarray, b: np.ndarray, step: float) -> float:
    """Aligned normalised-RMSE sum across non-depth channels.

    Returns inf if either trace has fewer than 2 points or their depth ranges
    do not overlap. Each channel's RMSE is divided by the mean absolute value
    across both traces (floored at 1e-6) before being summed across channels,
    so the score is comparable across qc/fs/u2 which have different magnitudes.
    """
    if a.shape[0] < 2 or b.shape[0] < 2:
        return math.inf
    lo = max(a[:, 0].min(), b[:, 0].min())
    hi = min(a[:, 0].max(), b[:, 0].max())
    if hi <= lo:
        return math.inf
    grid = np.arange(lo, hi + step / 2, step)
    if grid.size < 2:
        return math.inf
    total = 0.0
    for ch in range(1, a.shape[1]):
        ai = np.interp(grid, a[:, 0], a[:, ch])
        bi = np.interp(grid, b[:, 0], b[:, ch])
        denom = max((np.abs(ai).mean() + np.abs(bi).mean()) / 2.0, 1e-6)
        rmse = float(np.sqrt(np.mean((ai - bi) ** 2)))
        total += rmse / denom
    return total


def best_trace_score(
    traces_a: dict[int, np.ndarray],
    traces_b: dict[int, np.ndarray],
    step: float,
) -> tuple[float, tuple[int, int] | None]:
    """Best (lowest) trace_score over all cross-record report pairs, plus the winning pair."""
    best_score = math.inf
    best_pair: tuple[int, int] | None = None
    for ra, ta in traces_a.items():
        for rb, tb in traces_b.items():
            s = trace_score(ta, tb, step)
            if s < best_score:
                best_score = s
                best_pair = (ra, rb)
    return best_score, best_pair
