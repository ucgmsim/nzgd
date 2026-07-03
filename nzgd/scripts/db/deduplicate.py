r"""CLI entry point for cross-record CPT/SPT deduplication.

Copies a source NZGD SQLite DB to a target path, then applies the deduplication
passes to the copy: within-record consolidation (Pass 0), bit-exact hash
matching (Pass 1), and metadata-blocked fuzzy matching (Pass 2), followed by
supplemental consolidation. The deduped DB and CSV reports are written to the
target's directory. The source DB is never modified.

Run as a module from the repository root, using the project virtualenv.

Examples
--------
Deduplicate a database, writing the result to ``<source>_deduped.db`` next to
the source::

    python -m nzgd.scripts.db.deduplicate \
        --source /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p6p0_20260403.db

Write the deduped DB to an explicit path::

    python -m nzgd.scripts.db.deduplicate \
        --source /path/to/source.db \
        --target /path/to/output_deduped.db

Deduplicate only CPT records, skipping SPT/borehole records::

    python -m nzgd.scripts.db.deduplicate \
        --source /path/to/source.db --skip-spt
"""

import csv
import json
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import typer

from nzgd import constants
from nzgd.dedup.data_types import CPT_TABLE_CONFIG, SPT_TABLE_CONFIG
from nzgd.dedup.executor import apply_merge_plan
from nzgd.dedup.pass0_within_record import (
    apply_within_record_consolidation_plan,
    generate_within_record_consolidation_plan,
)
from nzgd.dedup.pass1_hash import generate_hash_merge_plan
from nzgd.dedup.pass2_fuzzy import generate_fuzzy_merge_plan
from nzgd.dedup.quality_filter import (
    apply_quality_filter,
    find_constant_column_reports,
)
from nzgd.dedup.reports import (
    write_calibration_report,
    write_dedup_report,
    write_failures_report,
    write_quality_filter_report,
    write_supplemental_consolidation_report,
)
from nzgd.dedup.schema import apply_dedup_schema
from nzgd.dedup.supplemental_consolidation import (
    consolidate_within_record_supplemental,
)
from nzgd.dedup.verify import find_spt_format_orphans

app = typer.Typer(help=__doc__)


def _script_version() -> str:
    """Return the current git commit hash for provenance, or ``"unknown"``.

    Returns
    -------
    str
        The full HEAD commit hash of the repository containing this script, or
        ``"unknown"`` if git is unavailable or the command fails.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
            cwd=Path(__file__).resolve().parents[3],
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError:
        pass
    return "unknown"


@app.command()
def main(
    source: Path = typer.Option(..., "--source", help="Source SQLite DB (read-only)."),
    target: Path = typer.Option(None, "--target", help="Target deduped DB path. Defaults to '<source>_deduped.db'."),
    skip_cpt: bool = typer.Option(False, "--skip-cpt", help="Skip CPT deduplication."),
    skip_spt: bool = typer.Option(False, "--skip-spt", help="Skip SPT deduplication."),
    skip_quality_filter: bool = typer.Option(
        False, "--skip-quality-filter", help="Skip the constant-column quality filter."
    ),
) -> None:
    """Run the dedup pipeline against ``source``, producing a deduped DB at ``target``.

    Copies ``source`` to ``target`` and applies, to the copy, the constant-column
    quality filter followed by the deduplication passes (quality filter,
    within-record consolidation, hash matching, fuzzy matching, then supplemental
    consolidation), writing the deduped DB and CSV reports to the target's
    directory. A read-only post-run check confirms no cross-record SPT merge
    dropped the only copy of an AGS/PDF format. The source DB is never modified.

    Parameters
    ----------
    source
        Path to the source NZGD SQLite DB. Opened read-only; never modified.
    target
        Path for the deduped output DB. Defaults to ``<source>_deduped.db``
        next to the source. Refuses to overwrite an existing file.
    skip_cpt
        If True, skip deduplication of CPT records.
    skip_spt
        If True, skip deduplication of SPT (borehole) records.
    skip_quality_filter
        If True, skip the constant-column quality filter for all record types.
    """
    if target is None:
        suffix = constants.DEDUP_CONFIG["output"]["deduped_db_suffix"]
        target = source.with_name(source.stem + suffix + ".db")
    if target.exists():
        typer.echo(f"Target {target} already exists; refusing to overwrite. Delete it and rerun.", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Copying {source} → {target} ...")
    shutil.copyfile(source, target)

    conn = sqlite3.connect(target)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_dedup_schema(conn)

    config_snapshot = json.dumps(constants.DEDUP_CONFIG)
    started = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO dedup_run (started_at, source_db_path, script_version, config_snapshot_json) "
        "VALUES (?, ?, ?, ?)",
        (started, str(source), _script_version(), config_snapshot),
    )
    run_id = cur.lastrowid
    conn.commit()

    out_dir = target.parent
    fuzzy_thresholds = {
        **constants.DEDUP_CONFIG["fuzzy_pass"],
        "random_pair_sample_size": constants.DEDUP_CONFIG["calibration"]["random_pair_sample_size"],
    }

    all_failures: list[dict] = []
    total_clusters = 0
    total_records = 0

    within_enabled = set(
        constants.DEDUP_CONFIG.get("within_record", {}).get("enabled_record_types", ["CPT", "BH"])
    )
    qf_cfg = constants.DEDUP_CONFIG.get("quality_filter", {})
    qf_enabled = set(qf_cfg.get("enabled_record_types", []))
    qf_columns = qf_cfg.get("constant_columns", {})
    qf_min_rows = qf_cfg.get("min_non_null_rows", 3)
    for cfg, skip in ((CPT_TABLE_CONFIG, skip_cpt), (SPT_TABLE_CONFIG, skip_spt)):
        if skip:
            typer.echo(f"Skipping {cfg.record_type} per CLI flag.")
            continue
        if not skip_quality_filter and cfg.record_type in qf_enabled:
            typer.echo(f"[{cfg.record_type}] Quality filter: discarding constant-column reports ...")
            qf_entries = find_constant_column_reports(
                conn, cfg, qf_columns.get(cfg.record_type, []), qf_min_rows
            )
            n_qf = apply_quality_filter(conn, qf_entries, run_id, cfg, failures=all_failures)
            typer.echo(f"[{cfg.record_type}] Quality filter: discarded {n_qf} reports.")
        if cfg.record_type in within_enabled:
            typer.echo(f"[{cfg.record_type}] Pass 0: within-record consolidation ...")
            pass0_thresholds = {
                "trace_score_max": constants.DEDUP_CONFIG["fuzzy_pass"]["trace_score_max"],
                "trace_resample_step_m": constants.DEDUP_CONFIG["fuzzy_pass"]["trace_resample_step_m"],
            }
            within_plan = generate_within_record_consolidation_plan(conn, cfg, pass0_thresholds)
            c0, r0 = apply_within_record_consolidation_plan(
                conn, within_plan, run_id, cfg, failures=all_failures,
            )
            typer.echo(f"[{cfg.record_type}] Pass 0: absorbed {r0} rows across {c0} clusters.")
        else:
            typer.echo(f"[{cfg.record_type}] Pass 0: skipped (not in within_record.enabled_record_types).")
            c0, r0 = 0, 0
        typer.echo(f"[{cfg.record_type}] Pass 1: hash ...")
        hash_plan = generate_hash_merge_plan(conn, cfg)
        c1, r1 = apply_merge_plan(conn, hash_plan, run_id, cfg, failures=all_failures)
        typer.echo(f"[{cfg.record_type}] Pass 1: merged {r1} records across {c1} clusters.")

        typer.echo(f"[{cfg.record_type}] Pass 2: fuzzy ...")
        calibration: dict = {}
        fuzzy_plan = generate_fuzzy_merge_plan(conn, cfg, fuzzy_thresholds, calibration_collector=calibration)
        c2, r2 = apply_merge_plan(conn, fuzzy_plan, run_id, cfg, failures=all_failures)
        typer.echo(f"[{cfg.record_type}] Pass 2: merged {r2} records across {c2} clusters.")

        if cfg.record_type in within_enabled:
            typer.echo(f"[{cfg.record_type}] Supplemental consolidation ...")
            supp_records, supp_cells = consolidate_within_record_supplemental(
                conn, cfg, run_id, failures=all_failures
            )
            typer.echo(
                f"[{cfg.record_type}] Supplemental consolidation: filled {supp_cells} "
                f"cells across {supp_records} records."
            )

        # Write a per-record-type calibration file when there's content
        if calibration.get("positive") or calibration.get("negative"):
            cal_path = out_dir / f"{cfg.record_type.lower()}_{constants.DEDUP_CONFIG['output']['calibration_report_filename']}"
            write_calibration_report(calibration.get("positive", []), calibration.get("negative", []), cal_path)

        total_clusters += c0 + c1 + c2
        total_records += r0 + r1 + r2

    finished = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE dedup_run SET finished_at = ?, n_clusters_merged = ?, n_records_merged = ? WHERE run_id = ?",
        (finished, total_clusters, total_records, run_id),
    )
    conn.commit()

    report_path = out_dir / constants.DEDUP_CONFIG["output"]["report_filename"]
    write_dedup_report(conn, run_id, report_path)

    supp_report_path = out_dir / constants.DEDUP_CONFIG["output"]["supplemental_consolidation_report_filename"]
    write_supplemental_consolidation_report(conn, run_id, supp_report_path)
    typer.echo(f"Supplemental consolidation report at {supp_report_path}.")

    qf_report_path = out_dir / constants.DEDUP_CONFIG["output"]["quality_filter_report_filename"]
    write_quality_filter_report(conn, run_id, qf_report_path)
    typer.echo(f"Quality filter report at {qf_report_path}.")

    if all_failures:
        failures_path = out_dir / constants.DEDUP_CONFIG["output"]["failures_filename"]
        write_failures_report(all_failures, failures_path)
        typer.echo(f"{len(all_failures)} cluster(s) failed; see {failures_path}")

    # Post-run verification (read-only): confirm no cross-record SPT merge dropped the
    # only copy of an AGS/PDF format. Deleted reports survive only in `source`, so open
    # it read-only to recover their formats.
    source_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        spt_format_orphans = find_spt_format_orphans(conn, source_conn)
    finally:
        source_conn.close()
    n_spt_cross_record = conn.execute(
        "SELECT COUNT(*) FROM dedup_audit WHERE record_type = 'BH' "
        "AND match_pass IN ('hash', 'fuzzy')"
    ).fetchone()[0]
    if spt_format_orphans:
        orphans_path = out_dir / "dedup_spt_format_orphans.csv"
        with open(orphans_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(spt_format_orphans[0].keys()))
            writer.writeheader()
            writer.writerows(spt_format_orphans)
        typer.echo(
            f"VERIFY FAIL: {len(spt_format_orphans)} cross-record SPT merge(s) dropped the "
            f"only copy of an AGS/PDF format; see {orphans_path}.",
            err=True,
        )
    else:
        typer.echo(
            f"VERIFY OK: {n_spt_cross_record} cross-record SPT merges checked; "
            f"no orphaned AGS/PDF formats."
        )

    typer.echo(f"Done. Deduped DB at {target}. Report at {report_path}.")
    conn.close()


if __name__ == "__main__":
    app()
