"""Checkpoint central Vs30 estimates, then explicitly publish validated results.

Uncertainty is deliberately NULL; see VsViewer/docs/vs30_uncertainty_followup.md
in the Vs30 repository. Source measurements are always read-only during runs.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import multiprocessing
import os
import sqlite3
import sys
import time
import warnings
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

import vs_calc
from vs_calc.CPT import ExceededMaxIterations
from vs_calc.scripts import validate_spt_database

SOURCE = None
UNIT_WEIGHTS = None
MINIMUM_DEPTH = {"boore_2004": 10, "boore_2011": 5}
LOOKUP_TABLES = (
    "cpttovscorrelation",
    "spttovscorrelation",
    "vstovs30correlation",
    "spttovs30hammertype",
)


def read_only(path: Path) -> sqlite3.Connection:
    """Open an existing SQLite file without permission to change it."""
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=60)


def source_identity(database: Path) -> dict:
    """Identify the exact input file; refuse changed inputs when resuming."""
    stat = database.stat()
    return {
        "path": str(database.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def read_lookups(conn: sqlite3.Connection) -> dict:
    """Read actual database IDs rather than relying on extraction configuration."""
    return {
        table: {
            value: key for key, value in conn.execute(f"SELECT id, value FROM {table}")
        }
        for table in LOOKUP_TABLES
    }


def initialize_worker(database: str, weights: str) -> None:
    """Give each process its own read-only source connection."""
    global SOURCE, UNIT_WEIGHTS
    SOURCE = read_only(Path(database))
    SOURCE.row_factory = sqlite3.Row
    UNIT_WEIGHTS = pd.read_csv(weights).set_index("soil_type")


def finite_metadata(value: object, default: float) -> float:
    """Use defaults only for absent or non-numeric metadata."""
    return validate_spt_database.numeric_value(value, default)


def prepare_cpt(report: dict) -> tuple[dict, dict, dict]:
    """Prepare separate input groups so missing u2 does not exclude McGann."""
    data = pd.read_sql_query(
        "SELECT depth_m, qc_MPa, fs_MPa, u2_MPa FROM cptmeasurements "
        "WHERE cpt_id = ? ORDER BY depth_m, measurement_id",
        SOURCE,
        params=(report["cpt_id"],),
    ).apply(pd.to_numeric, errors="coerce")
    groundwater = finite_metadata(report["extracted_gwl_m"], 1.0)
    area = finite_metadata(report["tip_net_area_ratio"], 0.8)
    if groundwater < 0 or not 0 < area <= 1:
        raise ValueError("invalid CPT groundwater or net area ratio")
    assumptions = {
        "measurement_count": len(data),
        "groundwater_m": groundwater,
        "groundwater_assumed": not np.isfinite(
            finite_metadata(report["extracted_gwl_m"], np.nan)
        ),
        "groundwater_zero": groundwater == 0,
        "net_area_ratio": area,
        "net_area_ratio_assumed": not np.isfinite(
            finite_metadata(report["tip_net_area_ratio"], np.nan)
        ),
        "groups": {},
    }
    groups, errors = {}, {}
    for group, columns in {
        "qc_fs": ["depth_m", "qc_MPa", "fs_MPa"],
        "qt": ["depth_m", "qc_MPa", "u2_MPa"],
        "normalized": ["depth_m", "qc_MPa", "fs_MPa", "u2_MPa"],
    }.items():
        valid = (
            np.isfinite(data[columns]).all(axis=1)
            & (data.depth_m > 0)
            & (data.qc_MPa > 0)
        )
        if "fs_MPa" in columns:
            valid &= data.fs_MPa > 0
        selected = data.loc[valid, columns].drop_duplicates()
        assumptions["groups"][group] = {
            "invalid_measurements_removed": int((~valid).sum()),
            "duplicate_measurements_removed": int(valid.sum() - len(selected)),
            "usable_measurements": len(selected),
        }
        if selected.depth_m.duplicated().any():
            errors[group] = "conflicting measurements at the same depth"
            continue
        if len(selected) < 2 or selected.depth_m.max() < 5:
            errors[group] = "fewer than two usable samples or depth below 5 m"
            continue
        assumptions["groups"][group].update(
            {
                "minimum_depth_m": float(selected.depth_m.min()),
                "maximum_depth_m": float(selected.depth_m.max()),
                "largest_sample_gap_m": float(selected.depth_m.diff().max()),
            }
        )
        cpt = vs_calc.CPT(
            str(report["cpt_id"]),
            selected.depth_m.to_numpy().copy(),
            selected.qc_MPa.to_numpy().copy(),
            selected.fs_MPa.to_numpy().copy()
            if "fs_MPa" in columns
            else np.zeros(len(selected)),
            selected.u2_MPa.to_numpy().copy()
            if "u2_MPa" in columns
            else np.zeros(len(selected)),
            ground_water_level=groundwater,
            net_area_ratio=area,
        )
        if group != "qc_fs":
            # qc and the pore-pressure correction can cancel exactly in decimal
            # source data yet leave a tiny positive float. Such a value is zero
            # to input precision, not a valid resistance for a power-law model.
            roundoff = (
                8 * np.finfo(float).eps * (np.abs(cpt.Qc) + np.abs(cpt.u * (1 - area)))
            )
            if (cpt.qt <= roundoff).any():
                errors[group] = (
                    "non-positive or numerically zero corrected tip resistance"
                )
                continue
        if group == "normalized":
            try:
                (cpt._Qtn, cpt._effStress, cpt._Ic, cpt._n, cpt._totalStress) = (
                    cpt.calc_cpt_params(max_iterations=1000)
                )
            except (ExceededMaxIterations, ValueError, ArithmeticError) as exc:
                errors[group] = f"CPT normalization: {type(exc).__name__}: {exc}"
                continue
        groups[group] = cpt
    return groups, errors, assumptions


def prepare_spt(report: dict) -> tuple[vs_calc.SPT, dict]:
    """Use validated nearest-layer logs; default to clay only if no log is present."""
    has_layers = SOURCE.execute(
        "SELECT 1 FROM soilmeasurements WHERE spt_id=? LIMIT 1", (report["spt_id"],)
    ).fetchone()
    if has_layers:
        spt, _, assumptions = validate_spt_database.load_record(
            SOURCE, report, UNIT_WEIGHTS
        )
        assumptions["used_layer_soil_types"] = True
        assumptions["non_core_soils_mapped_to_clay"] = [
            name
            for name in assumptions["soil_types"]
            if name not in ("CLAY", "SILT", "SAND", "GRAVEL")
        ]
    else:
        data = pd.read_sql_query(
            "SELECT depth_m, ISPT_MAIN, ISPT_NVAL FROM sptmeasurements "
            "WHERE spt_id=? ORDER BY depth_m, spt_measurement_id",
            SOURCE,
            params=(report["spt_id"],),
        ).apply(pd.to_numeric, errors="coerce")
        data["n"] = data.ISPT_NVAL.where(np.isfinite(data.ISPT_NVAL), data.ISPT_MAIN)
        valid = (
            np.isfinite(data[["depth_m", "n"]]).all(axis=1)
            & (data.depth_m >= 0)
            & (data.n >= 0)
        )
        selected = data.loc[valid, ["depth_m", "n"]].drop_duplicates()
        if selected.depth_m.duplicated().any():
            raise ValueError("conflicting N values at the same depth")
        positive = selected[selected.n > 0]
        if len(positive) < 2 or positive.depth_m.max() < 5:
            raise ValueError(
                "fewer than two positive N values or insufficient depth for Vs30"
            )
        groundwater = finite_metadata(report["extracted_gwl_m"], 2.0)
        efficiency = finite_metadata(report["efficiency"], 75.0)
        diameter = finite_metadata(report["borehole_diameter"], 150.0)
        if groundwater < 0 or efficiency <= 0 or diameter <= 0:
            raise ValueError("invalid SPT groundwater, efficiency or diameter")
        spt = vs_calc.SPT(
            str(report["spt_id"]),
            selected.depth_m.to_numpy(),
            selected.n.to_numpy(),
            energy_ratio=efficiency,
            borehole_diameter=diameter,
            groundwater_level=groundwater,
        )
        assumptions = {
            "measurement_count": len(selected),
            "groundwater_m": groundwater,
            "groundwater_assumed": not np.isfinite(
                finite_metadata(report["extracted_gwl_m"], np.nan)
            ),
            "energy_ratio_percent": efficiency,
            "energy_ratio_assumed": not np.isfinite(
                finite_metadata(report["efficiency"], np.nan)
            ),
            "borehole_diameter_mm": diameter,
            "diameter_assumed": not np.isfinite(
                finite_metadata(report["borehole_diameter"], np.nan)
            ),
            "used_layer_soil_types": False,
            "soil_fallback": "no logged layers: default clay",
            "invalid_measurements_removed": int((~valid).sum()),
            "duplicate_measurements_removed": int(valid.sum() - len(selected)),
            "zero_n_count": int((selected.n == 0).sum()),
        }
    assumptions["groundwater_zero"] = assumptions["groundwater_m"] == 0
    assumptions["hammer_type"] = "Auto"
    assumptions["minimum_depth_m"] = float(spt.depth.min())
    assumptions["maximum_depth_m"] = float(spt.depth.max())
    assumptions["largest_sample_gap_m"] = float(np.diff(spt.depth).max())
    return spt, assumptions


def central_estimates(name: str, vs: np.ndarray, depth: np.ndarray) -> list[tuple]:
    """Evaluate each eligible existing central relation; never return its sigma."""
    vs, depth = np.asarray(vs).reshape(-1), np.asarray(depth).reshape(-1)
    if (
        len(vs) < 2
        or vs.shape != depth.shape
        or not np.isfinite(vs).all()
        or (vs <= 0).any()
        or not np.isfinite(depth).all()
        or (np.diff(depth) <= 0).any()
    ):
        raise ValueError(
            "Vs profile must have >=2 finite positive velocities at increasing depths"
        )
    rows = []
    for method, minimum in MINIMUM_DEPTH.items():
        if depth.max() < minimum:
            rows.append(
                (
                    method,
                    "ineligible",
                    f"usable depth {depth.max():g} m below {minimum} m",
                    None,
                )
            )
            continue
        if not (depth <= min(int(depth.max()), 30)).any():
            rows.append(
                (method, "ineligible", "no measurement within integration depth", None)
            )
            continue
        try:
            # VsProfile and several input correlations mutate arrays. Each conversion
            # gets fresh copies; uncertainty placeholders do not affect its mean.
            profile = vs_calc.VsProfile(
                name,
                vs.copy(),
                np.zeros_like(vs),
                depth.copy(),
                vs30_correlation=method,
            )
            value = float(profile.vs30)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("non-finite or non-positive Vs30")
            rows.append(
                (
                    method,
                    "ok",
                    "direct 30 m integration" if profile.max_depth == 30 else "",
                    value,
                )
            )
        except (ValueError, IndexError, KeyError, ArithmeticError) as exc:
            rows.append((method, "error", f"{type(exc).__name__}: {exc}", None))
    return rows


def process_report(task: tuple[str, int]) -> tuple:
    """Return all combination outcomes and assumptions for one source report."""
    started = time.monotonic()
    kind, report_id = task
    report = dict(
        SOURCE.execute(
            f"SELECT * FROM {kind}report WHERE {kind}_id=?", (report_id,)
        ).fetchone()
    )
    correlations = (
        vs_calc.CPT_CORRELATIONS if kind == "cpt" else vs_calc.SPT_CORRELATIONS
    )
    outcomes, assumptions = [], {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        try:
            if kind == "cpt":
                groups, group_errors, assumptions = prepare_cpt(report)
            else:
                spt, assumptions = prepare_spt(report)
            for name, correlation in correlations.items():
                try:
                    if kind == "cpt":
                        group = (
                            "qc_fs"
                            if name.startswith("mcgann")
                            else (
                                "qt"
                                if name == "andrus_2007_tertiary_age_cooper_marl"
                                else "normalized"
                            )
                        )
                        if group in group_errors:
                            raise ValueError(group_errors[group])
                        cpt = copy.deepcopy(groups[group])
                        vs, _ = correlation(cpt)
                        depth = cpt.depth
                    else:
                        vs, _, depth, _ = correlation(spt)
                    outcomes.extend(
                        (name, *row)
                        for row in central_estimates(str(report_id), vs, depth)
                    )
                except (ValueError, IndexError, KeyError, ArithmeticError) as exc:
                    outcomes.extend(
                        (name, method, "error", f"{type(exc).__name__}: {exc}", None)
                        for method in MINIMUM_DEPTH
                    )
        except (ValueError, IndexError, KeyError, ArithmeticError) as exc:
            assumptions["input_error"] = f"{type(exc).__name__}: {exc}"
            outcomes = [
                (name, method, "ineligible", assumptions["input_error"], None)
                for name in correlations
                for method in MINIMUM_DEPTH
            ]
    assumptions["runtime_warning_count"] = len(caught)
    assumptions["runtime_warning_examples"] = sorted(
        {str(item.message) for item in caught}
    )[:10]
    return (
        kind,
        report_id,
        report["nzgd_id"],
        assumptions,
        outcomes,
        time.monotonic() - started,
    )


def manifest(database: Path, weights: Path) -> dict:
    """Fingerprint scientific code, resource data, source file and ID mappings."""
    code_root = Path(vs_calc.__file__).resolve().parent
    paths = sorted(code_root.rglob("*.py")) + [
        Path(__file__).resolve(),
        weights.resolve(),
    ]
    with closing(read_only(database)) as conn:
        lookups = read_lookups(conn)
    for table, expected in (
        ("cpttovscorrelation", vs_calc.CPT_CORRELATIONS),
        ("spttovscorrelation", vs_calc.SPT_CORRELATIONS),
        ("vstovs30correlation", MINIMUM_DEPTH),
        ("spttovs30hammertype", ["Auto"]),
    ):
        if set(expected) - set(lookups[table]):
            raise ValueError(f"Missing correlation/hammer names in {table}")
    return {
        "source": source_identity(database),
        "lookups": lookups,
        "code_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "uncertainty_policy": "NULL: not reliably evaluated",
        "policy_version": 2,
    }


def open_stage(run_dir: Path) -> sqlite3.Connection:
    """Create the checkpoint schema; a report and its outcomes commit together."""
    conn = sqlite3.connect(run_dir / "estimates.sqlite", timeout=60)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS records (
            kind TEXT NOT NULL, report_id INTEGER NOT NULL, nzgd_id INTEGER NOT NULL,
            assumptions TEXT NOT NULL, seconds REAL NOT NULL,
            PRIMARY KEY (kind, report_id));
        CREATE TABLE IF NOT EXISTS results (
            kind TEXT NOT NULL, report_id INTEGER NOT NULL, vs_method TEXT NOT NULL,
            vs30_method TEXT NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL,
            vs30 REAL,
            PRIMARY KEY (kind, report_id, vs_method, vs30_method),
            FOREIGN KEY (kind, report_id) REFERENCES records(kind, report_id),
            CHECK ((status='ok' AND vs30>0 AND vs30 IS NOT NULL) OR
                   (status!='ok' AND vs30 IS NULL)));
        CREATE TABLE IF NOT EXISTS invocations (
            started_utc TEXT NOT NULL, workers INTEGER NOT NULL,
            processed INTEGER NOT NULL, elapsed_seconds REAL NOT NULL);
    """)
    return conn


def report_status(conn: sqlite3.Connection) -> dict:
    """Summarise checkpoints, successes and reasons without loading all results."""
    return {
        "reports": dict(
            conn.execute("SELECT kind, count(*) FROM records GROUP BY kind")
        ),
        "outcomes": [
            dict(zip(("kind", "status", "count"), row))
            for row in conn.execute(
                "SELECT kind,status,count(*) FROM results GROUP BY kind,status"
            )
        ],
        "estimates": [
            dict(
                zip(
                    ("kind", "vs_method", "vs30_method", "count", "minimum", "maximum"),
                    row,
                )
            )
            for row in conn.execute(
                "SELECT kind,vs_method,vs30_method,count(*),min(vs30),max(vs30) "
                "FROM results WHERE status='ok' GROUP BY kind,vs_method,vs30_method"
            )
        ],
        "top_exclusions": [
            dict(zip(("kind", "reason", "combinations"), row))
            for row in conn.execute(
                "SELECT kind,reason,count(*) FROM results WHERE status!='ok' "
                "GROUP BY kind,reason ORDER BY count(*) DESC LIMIT 20"
            )
        ],
        "completed_invocations": [
            list(row) for row in conn.execute("SELECT * FROM invocations")
        ],
    }


def run_batch(args: argparse.Namespace, stage: sqlite3.Connection) -> None:
    """Run pending reports with one coordinator writing durable checkpoints."""
    started = time.monotonic()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    current = manifest(args.database, args.unit_weights)
    existing = stage.execute(
        "SELECT value FROM metadata WHERE key='manifest'"
    ).fetchone()
    if existing and json.loads(existing[0]) != current:
        raise ValueError("Source, code or environment changed; use a new run directory")
    if not existing:
        stage.execute(
            "INSERT INTO metadata VALUES ('manifest',?)",
            (json.dumps(current, sort_keys=True),),
        )
        stage.commit()
    tasks = []
    with closing(read_only(args.database)) as source:
        for kind in ["cpt", "spt"] if args.kind == "both" else [args.kind]:
            completed = {
                row[0]
                for row in stage.execute(
                    "SELECT report_id FROM records WHERE kind=?", (kind,)
                )
            }
            pending = [
                row[0]
                for row in source.execute(
                    f"SELECT r.{kind}_id FROM {kind}report r WHERE EXISTS "
                    f"(SELECT 1 FROM {kind}measurements m WHERE m.{kind}_id=r.{kind}_id) ORDER BY r.{kind}_id"
                )
                if row[0] not in completed
            ]
            if args.limit and len(pending) > args.limit:
                pending = [
                    pending[i]
                    for i in np.linspace(0, len(pending) - 1, args.limit, dtype=int)
                ]
            tasks.extend((kind, report_id) for report_id in pending)
    print(
        json.dumps(
            {
                "pending_reports": len(tasks),
                "workers": args.workers,
                "source": current["source"],
            }
        ),
        flush=True,
    )
    if not tasks:
        print(json.dumps(report_status(stage), indent=2), flush=True)
        return
    processed = successful = 0
    last_progress = time.monotonic()
    # A spawned process cannot accidentally inherit the coordinator's SQLite writer.
    with multiprocessing.get_context("spawn").Pool(
        args.workers,
        initialize_worker,
        (str(args.database), str(args.unit_weights)),
    ) as pool:
        for (
            kind,
            report_id,
            nzgd_id,
            assumptions,
            outcomes,
            seconds,
        ) in pool.imap_unordered(process_report, tasks, chunksize=1):
            stage.execute(
                "INSERT INTO records VALUES (?,?,?,?,?)",
                (
                    kind,
                    report_id,
                    nzgd_id,
                    json.dumps(assumptions, allow_nan=False, sort_keys=True),
                    seconds,
                ),
            )
            stage.executemany(
                "INSERT INTO results VALUES (?,?,?,?,?,?,?)",
                ((kind, report_id, *row) for row in outcomes),
            )
            processed += 1
            successful += sum(row[2] == "ok" for row in outcomes)
            if processed % 25 == 0:
                stage.commit()
            if time.monotonic() - last_progress >= 20 or processed == len(tasks):
                stage.commit()
                elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "pending_at_start": len(tasks),
                            "estimates": successful,
                            "elapsed_seconds": round(elapsed, 1),
                            "reports_per_second": round(processed / elapsed, 2),
                            "last_report": [kind, report_id],
                        }
                    ),
                    flush=True,
                )
                last_progress = time.monotonic()
    if source_identity(args.database) != current["source"]:
        raise ValueError("Source changed while processing; do not publish this run")
    stage.execute(
        "INSERT INTO invocations VALUES (?,?,?,?)",
        (started_utc, args.workers, processed, time.monotonic() - started),
    )
    stage.commit()
    print(json.dumps(report_status(stage), indent=2), flush=True)


def publish(args: argparse.Namespace, stage: sqlite3.Connection) -> None:
    """Back up and transactionally insert only absent, validated estimate rows."""
    invalidated = stage.execute(
        "SELECT value FROM metadata WHERE key='invalidated'"
    ).fetchone()
    if invalidated:
        raise ValueError(f"This run was invalidated: {invalidated[0]}")
    stored = stage.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
    if not stored:
        raise ValueError("No calculated run to publish")
    original = json.loads(stored[0])
    if str(args.database.resolve()) != original["source"]["path"]:
        raise ValueError("Target database differs from the run source")
    published = stage.execute(
        "SELECT value FROM metadata WHERE key='published'"
    ).fetchone()
    if not published and source_identity(args.database) != original["source"]:
        raise ValueError("Source changed since calculation; refusing publication")
    if stage.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise ValueError("Checkpoint database failed quick_check")
    if stage.execute("PRAGMA foreign_key_check").fetchone():
        raise ValueError("Checkpoint database has broken references")
    if stage.execute(
        "SELECT 1 FROM results WHERE status='ok' AND "
        "(vs30 IS NULL OR vs30<=0 OR vs30>1.7976931348623157e308) LIMIT 1"
    ).fetchone():
        raise ValueError("Invalid staged central estimate")
    expected = dict(
        stage.execute(
            "SELECT kind,count(*) FROM results WHERE status='ok' GROUP BY kind"
        )
    )
    if not sum(expected.values()):
        raise ValueError("No successful estimates to publish")
    if not published:
        if args.backup.resolve() in (
            args.database.resolve(),
            (args.run_dir / "estimates.sqlite").resolve(),
        ):
            raise ValueError("Backup must be a distinct new file")
        # Reserve the path exclusively: never overwrite an existing backup.
        descriptor = os.open(args.backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        print(f"Creating SQLite backup: {args.backup}", flush=True)
        with (
            closing(read_only(args.database)) as source,
            closing(sqlite3.connect(args.backup)) as backup,
        ):
            source.backup(backup, pages=8192)
            if backup.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("Backup failed quick_check")
        if source_identity(args.database) != original["source"]:
            raise ValueError("Source changed during backup; refusing publication")
    with closing(
        sqlite3.connect(
            args.database.resolve().as_uri() + "?mode=rw", uri=True, timeout=60
        )
    ) as target:
        target.execute("PRAGMA foreign_keys=ON")
        target.execute(
            "ATTACH DATABASE ? AS staged",
            ((args.run_dir / "estimates.sqlite").resolve().as_uri() + "?mode=ro",),
        )
        target.execute("BEGIN IMMEDIATE")
        try:
            if not published and source_identity(args.database) != original["source"]:
                raise ValueError("Source changed before the import transaction")
            if read_lookups(target) != original["lookups"]:
                raise ValueError("Lookup IDs changed")
            inserted = {}
            for kind in ("cpt", "spt"):
                joins = (
                    "FROM staged.results r JOIN staged.records p USING (kind,report_id) "
                    f"JOIN {kind}tovscorrelation v ON v.value=r.vs_method "
                    "JOIN vstovs30correlation b ON b.value=r.vs30_method "
                )
                match = (
                    f"e.{kind}_id=r.report_id AND e.{kind}_to_vs_correlation_id=v.id "
                    "AND e.vs_to_vs30_correlation_id=b.id"
                )
                where = f"r.kind='{kind}' AND r.status='ok'"
                extra_conflict = (
                    "e.nzgd_id IS NOT p.nzgd_id"
                    if kind == "cpt"
                    else (
                        "e.assumed_borehole_diameter_mm IS NOT json_extract(p.assumptions,'$.borehole_diameter_mm') OR "
                        f"e.assumed_hammer_type_id IS NOT {original['lookups']['spttovs30hammertype']['Auto']} OR "
                        "e.estimate_used_extracted_efficiency IS NOT (1-json_extract(p.assumptions,'$.energy_ratio_assumed')) OR "
                        "e.estimate_used_extracted_layer_soil_types IS NOT json_extract(p.assumptions,'$.used_layer_soil_types')"
                    )
                )
                conflict = target.execute(
                    "SELECT r.report_id "
                    + joins
                    + f"JOIN {kind}vs30estimates e ON {match} "
                    f"WHERE {where} AND (e.vs30 IS NOT r.vs30 OR e.vs30_stddev IS NOT NULL OR {extra_conflict}) LIMIT 1"
                ).fetchone()
                if conflict:
                    raise ValueError(
                        f"Existing differing {kind} estimate for report {conflict[0]}; no overwrite allowed"
                    )
                if target.execute(
                    "SELECT count(*) " + joins + f"WHERE {where}"
                ).fetchone()[0] != expected.get(kind, 0):
                    raise ValueError(f"Unmapped staged {kind} correlation")
                if kind == "cpt":
                    columns = "cpt_id,nzgd_id,cpt_to_vs_correlation_id,vs_to_vs30_correlation_id,vs30,vs30_stddev"
                    values = "r.report_id,p.nzgd_id,v.id,b.id,r.vs30,NULL "
                else:
                    columns = (
                        "spt_id,spt_to_vs_correlation_id,vs_to_vs30_correlation_id,assumed_borehole_diameter_mm,"
                        "assumed_hammer_type_id,estimate_used_extracted_efficiency,estimate_used_extracted_layer_soil_types,"
                        "vs30,vs30_stddev"
                    )
                    values = (
                        "r.report_id,v.id,b.id,json_extract(p.assumptions,'$.borehole_diameter_mm'),"
                        f"{original['lookups']['spttovs30hammertype']['Auto']},"
                        "1-json_extract(p.assumptions,'$.energy_ratio_assumed'),"
                        "json_extract(p.assumptions,'$.used_layer_soil_types'),r.vs30,NULL "
                    )
                target.execute(
                    f"INSERT INTO {kind}vs30estimates ({columns}) SELECT "
                    + values
                    + joins
                    + f"WHERE {where} AND NOT EXISTS (SELECT 1 FROM {kind}vs30estimates e WHERE {match})"
                )
                inserted[kind] = target.execute("SELECT changes()").fetchone()[0]
                if target.execute(
                    f"PRAGMA foreign_key_check('{kind}vs30estimates')"
                ).fetchone():
                    raise ValueError(f"Foreign key failure in {kind} estimates")
                if target.execute(
                    f"SELECT 1 FROM {kind}vs30estimates GROUP BY {kind}_id,"
                    f"{kind}_to_vs_correlation_id,vs_to_vs30_correlation_id HAVING count(*)>1 LIMIT 1"
                ).fetchone():
                    raise ValueError(f"Duplicate natural keys in {kind} estimates")
            target.commit()
        except BaseException:
            target.rollback()
            raise
    publication = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inserted": inserted,
        "backup": str(args.backup.resolve()),
        "expected_staged": expected,
    }
    stage.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('published',?)",
        (json.dumps(publication),),
    )
    stage.commit()
    print(json.dumps(publication, indent=2), flush=True)


def main(default_kind: str = "both") -> None:
    """Run, inspect or explicitly publish a checkpointed batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="calculate central estimates without modifying the source"
    )
    run.add_argument("database", type=Path)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--kind", choices=("cpt", "spt", "both"), default=default_kind)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument(
        "--limit",
        type=int,
        default=0,
        help="pilot: cap pending reports per kind, evenly spread over IDs",
    )
    run.add_argument(
        "--unit-weights",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "resources"
        / "soil_type_unit_weights.csv",
    )
    status = commands.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    write = commands.add_parser(
        "publish", help="backup, validate and insert central estimates; never overwrite"
    )
    write.add_argument("database", type=Path)
    write.add_argument("--run-dir", type=Path, required=True)
    write.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "status":
        with closing(read_only(args.run_dir / "estimates.sqlite")) as stage:
            print(json.dumps(report_status(stage), indent=2))
        return
    if args.command == "run" and (args.workers < 1 or args.limit < 0):
        parser.error("workers must be positive and limit non-negative")
    if args.command == "publish" and not (args.run_dir / "estimates.sqlite").is_file():
        parser.error("publish requires an existing checkpoint database")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with closing(open_stage(args.run_dir)) as stage:
            if args.command == "run":
                run_batch(args, stage)
            else:
                publish(args, stage)


if __name__ == "__main__":
    main()
