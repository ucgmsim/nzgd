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

    Returns inf if either trace has fewer than 2 points, their depth ranges do
    not overlap, or no channel has enough finite points to score. Each channel
    is independently masked: per-channel NaN values are filtered out before
    interpolation, and grid points that fall inside a NaN gap in either source
    are excluded from the RMSE so np.interp's straight-line bridge across a
    gap does not invent a fake discrepancy. Each channel's RMSE is divided by
    the mean absolute value across both traces (floored at 1e-6) before being
    summed across channels, so the score is comparable across qc/fs/u2 which
    have different magnitudes.
    """
    if a.shape[0] < 2 or b.shape[0] < 2:
        return math.inf
    a_depth_finite = a[np.isfinite(a[:, 0])]
    b_depth_finite = b[np.isfinite(b[:, 0])]
    if a_depth_finite.shape[0] < 2 or b_depth_finite.shape[0] < 2:
        return math.inf
    lo = max(a_depth_finite[:, 0].min(), b_depth_finite[:, 0].min())
    hi = min(a_depth_finite[:, 0].max(), b_depth_finite[:, 0].max())
    if hi <= lo:
        return math.inf
    grid = np.arange(lo, hi + step / 2, step)
    if grid.size < 2:
        return math.inf
    a_dfinite_mask = np.isfinite(a[:, 0])
    b_dfinite_mask = np.isfinite(b[:, 0])
    a_dfinite = a[a_dfinite_mask, 0]
    b_dfinite = b[b_dfinite_mask, 0]
    total = 0.0
    n_channels_used = 0
    for ch in range(1, a.shape[1]):
        fa = a_dfinite_mask & np.isfinite(a[:, ch])
        fb = b_dfinite_mask & np.isfinite(b[:, ch])
        if fa.sum() < 2 or fb.sum() < 2:
            continue
        ai = np.interp(grid, a[fa, 0], a[fa, ch])
        bi = np.interp(grid, b[fb, 0], b[fb, ch])
        # Validity indicators: 1.0 where the channel was finite in source, 0.0
        # where it was NaN. Interp on these tells us how much of the value at
        # each grid point came from a real measurement (vs. bridging a NaN gap).
        ind_a = np.interp(grid, a_dfinite, fa[a_dfinite_mask].astype(float))
        ind_b = np.interp(grid, b_dfinite, fb[b_dfinite_mask].astype(float))
        valid = (ind_a >= 0.999) & (ind_b >= 0.999)
        if valid.sum() < 2:
            continue
        denom = max(
            (np.abs(ai[valid]).mean() + np.abs(bi[valid]).mean()) / 2.0, 1e-6,
        )
        rmse = float(np.sqrt(np.mean((ai[valid] - bi[valid]) ** 2)))
        total += rmse / denom
        n_channels_used += 1
    if n_channels_used == 0:
        return math.inf
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
