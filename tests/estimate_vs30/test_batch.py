"""Small SQLite fixtures exercise the runner without the production database."""

import argparse
import importlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

import vs_calc
from nzgd.scripts.estimate_vs30 import batch


@pytest.fixture
def source(tmp_path: Path):
    database = tmp_path / "source.db"
    with sqlite3.connect(database) as conn:
        conn.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE nzgdrecord (nzgd_id INTEGER PRIMARY KEY);
            INSERT INTO nzgdrecord VALUES (100),(200),(300);
            CREATE TABLE cpttovscorrelation (id INTEGER PRIMARY KEY,value TEXT);
            CREATE TABLE spttovscorrelation (id INTEGER PRIMARY KEY,value TEXT);
            CREATE TABLE vstovs30correlation (id INTEGER PRIMARY KEY,value TEXT);
            CREATE TABLE spttovs30hammertype (id INTEGER PRIMARY KEY,value TEXT);
            CREATE TABLE cptreport (cpt_id INTEGER PRIMARY KEY,nzgd_id INTEGER,
                extracted_gwl_m REAL,tip_net_area_ratio REAL);
            CREATE TABLE cptmeasurements (measurement_id INTEGER PRIMARY KEY,cpt_id INTEGER,
                depth_m REAL,qc_MPa REAL,fs_MPa REAL,u2_MPa REAL);
            CREATE TABLE sptreport (spt_id INTEGER PRIMARY KEY,nzgd_id INTEGER,
                extracted_gwl_m REAL,efficiency REAL,borehole_diameter REAL);
            CREATE TABLE sptmeasurements (spt_measurement_id INTEGER PRIMARY KEY,
                spt_id INTEGER,depth_m REAL,ISPT_MAIN REAL,ISPT_NVAL REAL);
            CREATE TABLE soilmeasurements (soil_measurement_id INTEGER PRIMARY KEY,
                spt_id INTEGER,top_depth_m REAL,bottom_depth_m REAL);
            CREATE TABLE soiltypes (id INTEGER PRIMARY KEY,value TEXT);
            CREATE TABLE soilmeasurementsoiltype (soil_measurement_id INTEGER,soil_type_id INTEGER);
            CREATE TABLE cptvs30estimates (vs30_id INTEGER PRIMARY KEY,
                cpt_id INTEGER REFERENCES cptreport(cpt_id),nzgd_id INTEGER REFERENCES nzgdrecord(nzgd_id),
                cpt_to_vs_correlation_id INTEGER REFERENCES cpttovscorrelation(id),
                vs_to_vs30_correlation_id INTEGER REFERENCES vstovs30correlation(id),
                vs30 REAL,vs30_stddev REAL);
            CREATE TABLE sptvs30estimates (vs30_id INTEGER PRIMARY KEY,
                spt_id INTEGER REFERENCES sptreport(spt_id),
                spt_to_vs_correlation_id INTEGER REFERENCES spttovscorrelation(id),
                vs_to_vs30_correlation_id INTEGER REFERENCES vstovs30correlation(id),
                assumed_borehole_diameter_mm REAL,assumed_hammer_type_id INTEGER REFERENCES spttovs30hammertype(id),
                estimate_used_extracted_efficiency INTEGER,estimate_used_extracted_layer_soil_types INTEGER,
                vs30 REAL,vs30_stddev REAL);
            INSERT INTO cptreport VALUES (1,100,NULL,NULL),(2,200,0,0.8);
            INSERT INTO sptreport VALUES (1,100,NULL,NULL,NULL),(2,200,2,75,150),(3,300,2,75,150);
            INSERT INTO soiltypes VALUES (1,'SAND'),(2,'CLAY');
            INSERT INTO soilmeasurements VALUES (1,2,0,3),(2,2,5,15),(3,3,0,7),(4,3,5,15);
            INSERT INTO soilmeasurementsoiltype VALUES (1,1),(2,2),(3,1),(4,2);
        """)
        for table, values in (
            ("cpttovscorrelation", vs_calc.CPT_CORRELATIONS),
            ("spttovscorrelation", vs_calc.SPT_CORRELATIONS),
            ("vstovs30correlation", batch.MINIMUM_DEPTH),
            ("spttovs30hammertype", ["Auto"]),
        ):
            conn.executemany(f"INSERT INTO {table} VALUES (?,?)", enumerate(values, 1))
        conn.executemany(
            "INSERT INTO cptmeasurements (cpt_id,depth_m,qc_MPa,fs_MPa,u2_MPa) VALUES (?,?,?,?,?)",
            [
                (report_id, depth, 5.0, 0.06, None if report_id == 2 else 0.1)
                for report_id in (1, 2)
                for depth in np.arange(0.5, 12.1, 0.5)
            ],
        )
        conn.executemany(
            "INSERT INTO sptmeasurements (spt_id,depth_m,ISPT_MAIN,ISPT_NVAL) VALUES (?,?,?,?)",
            [
                (report_id, depth, 10, None)
                for report_id in (1, 2, 3)
                for depth in (2, 6, 12)
            ],
        )
    return database


@pytest.fixture
def args(source: Path, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return argparse.Namespace(
        database=source,
        run_dir=run_dir,
        workers=2,
        kind="both",
        limit=0,
        unit_weights=Path(batch.__file__).resolve().parents[2]
        / "resources"
        / "soil_type_unit_weights.csv",
        backup=tmp_path / "backup.db",
    )


@pytest.mark.parametrize(
    "depth,methods", [(4.9, 0), (5, 1), (9.9, 1), (10, 2), (29, 2), (30, 2), (35, 2)]
)
def test_depth_eligibility_and_no_input_mutation(depth: float, methods: int):
    depths = np.linspace(0.5, depth, 30)
    velocities = np.linspace(150, 250, 30)
    original = depths.copy(), velocities.copy()
    rows = batch.central_estimates("test", velocities, depths)
    assert sum(row[1] == "ok" for row in rows) == methods
    np.testing.assert_array_equal(depths, original[0])
    np.testing.assert_array_equal(velocities, original[1])
    if depth >= 30:
        assert rows[0][3] == rows[1][3]


def test_invalid_velocity_is_not_silently_dropped():
    with pytest.raises(ValueError, match="finite positive"):
        batch.central_estimates(
            "bad", np.array([100, np.nan, 200]), np.array([1, 5, 12])
        )


@pytest.mark.parametrize("qc,u2", [(0.04, -0.2), (0.02, -0.1), (0.5, -2.5)])
def test_roundoff_cancellation_is_not_a_positive_resistance(
    args: argparse.Namespace, qc: float, u2: float
):
    with sqlite3.connect(args.database) as target:
        target.execute(
            "UPDATE cptmeasurements SET qc_MPa=?,u2_MPa=? WHERE measurement_id=1",
            (qc, u2),
        )
    batch.initialize_worker(str(args.database), str(args.unit_weights))
    try:
        result = batch.process_report(("cpt", 1))
        successes = [row for row in result[4] if row[2] == "ok"]
        assert len(successes) == 4
        assert all(row[0].startswith("mcgann") for row in successes)
        assert any("numerically zero" in row[3] for row in result[4])
    finally:
        batch.SOURCE.close()


def test_invalidated_run_cannot_be_published(args: argparse.Namespace):
    with batch.open_stage(args.run_dir) as stage:
        stage.execute(
            "INSERT INTO metadata VALUES ('invalidated','failed quality control')"
        )
        stage.commit()
        with pytest.raises(ValueError, match="invalidated"):
            batch.publish(args, stage)
    assert not args.backup.exists()


def test_per_correlation_requirements_and_layer_gap(args: argparse.Namespace):
    batch.initialize_worker(str(args.database), str(args.unit_weights))
    try:
        cpt = batch.process_report(("cpt", 1))
        assert sum(row[2] == "ok" for row in cpt[4]) == 14
        missing_u2 = batch.process_report(("cpt", 2))
        assert sum(row[2] == "ok" for row in missing_u2[4]) == 4
        assert {row[0] for row in missing_u2[4] if row[2] == "ok"} == {
            "mcgann_2015",
            "mcgann_2018",
        }
        fallback = batch.process_report(("spt", 1))
        layered = batch.process_report(("spt", 2))
        assert len([row for row in fallback[4] if row[2] == "ok"]) == 4
        assert fallback[3]["used_layer_soil_types"] is False
        assert fallback[3]["energy_ratio_percent"] == 75
        assert layered[3]["used_layer_soil_types"] is True
        assert layered[3]["interval_gaps_filled"] == 1
        conflicting = batch.process_report(("spt", 3))
        assert all(row[2] == "ineligible" for row in conflicting[4])
        assert "overlapping" in conflicting[3]["input_error"]
    finally:
        batch.SOURCE.close()


def test_resume_backup_publication_and_idempotence(args: argparse.Namespace):
    before = batch.source_identity(args.database)
    with batch.open_stage(args.run_dir) as stage:
        batch.run_batch(args, stage)
        assert batch.source_identity(args.database) == before
        assert stage.execute("SELECT count(*) FROM records").fetchone()[0] == 5
        assert stage.execute("SELECT count(*) FROM results").fetchone()[0] == 40
        assert (
            stage.execute("SELECT count(*) FROM results WHERE status='ok'").fetchone()[
                0
            ]
            == 26
        )
        batch.run_batch(args, stage)
        assert stage.execute("SELECT count(*) FROM records").fetchone()[0] == 5
        batch.publish(args, stage)
        with (
            sqlite3.connect(args.database) as target,
            sqlite3.connect(args.backup) as backup,
        ):
            assert (
                target.execute("SELECT count(*) FROM cptvs30estimates").fetchone()[0]
                == 18
            )
            assert (
                target.execute("SELECT count(*) FROM sptvs30estimates").fetchone()[0]
                == 8
            )
            assert (
                target.execute(
                    "SELECT count(vs30_stddev) FROM cptvs30estimates"
                ).fetchone()[0]
                == 0
            )
            assert (
                target.execute(
                    "SELECT count(vs30_stddev) FROM sptvs30estimates"
                ).fetchone()[0]
                == 0
            )
            assert (
                backup.execute("SELECT count(*) FROM cptvs30estimates").fetchone()[0]
                == 0
            )
            assert (
                backup.execute("SELECT count(*) FROM sptvs30estimates").fetchone()[0]
                == 0
            )
            assert target.execute("PRAGMA foreign_key_check").fetchall() == []
            assert target.execute(
                "SELECT DISTINCT spt_id,assumed_borehole_diameter_mm,assumed_hammer_type_id,"
                "estimate_used_extracted_efficiency,estimate_used_extracted_layer_soil_types "
                "FROM sptvs30estimates ORDER BY spt_id"
            ).fetchall() == [(1, 150.0, 1, 0, 0), (2, 150.0, 1, 1, 1)]
        batch.publish(args, stage)
        published = json.loads(
            stage.execute(
                "SELECT value FROM metadata WHERE key='published'"
            ).fetchone()[0]
        )
        assert published["inserted"] == {"cpt": 0, "spt": 0}


def test_changed_source_prevents_resume_and_publication(args: argparse.Namespace):
    with batch.open_stage(args.run_dir) as stage:
        batch.run_batch(args, stage)
        with sqlite3.connect(args.database) as target:
            target.execute("UPDATE cptmeasurements SET qc_MPa=6 WHERE measurement_id=1")
        with pytest.raises(ValueError, match="changed"):
            batch.run_batch(args, stage)
        with pytest.raises(ValueError, match="changed"):
            batch.publish(args, stage)
        assert not args.backup.exists()


def test_existing_differing_estimates_are_never_overwritten(args: argparse.Namespace):
    with sqlite3.connect(args.database) as target:
        target.execute("INSERT INTO cptvs30estimates VALUES (1,1,100,1,1,999,NULL)")
    with batch.open_stage(args.run_dir) as stage:
        batch.run_batch(args, stage)
        with pytest.raises(ValueError, match="no overwrite"):
            batch.publish(args, stage)
    with sqlite3.connect(args.database) as target:
        assert target.execute("SELECT vs30 FROM cptvs30estimates").fetchall() == [
            (999.0,)
        ]
        assert (
            target.execute("SELECT count(*) FROM sptvs30estimates").fetchone()[0] == 0
        )


def test_original_scripts_are_import_safe(monkeypatch: pytest.MonkeyPatch):
    def no_database(*args: object, **kwargs: object):
        raise AssertionError("Import must not open a database")

    monkeypatch.setattr(sqlite3, "connect", no_database)
    for kind in ("cpt", "spt"):
        importlib.import_module(f"nzgd.scripts.estimate_vs30.estimate_vs30_from_{kind}")


def test_spt_conflict_rolls_back_cpt_inserts_too(args: argparse.Namespace):
    with sqlite3.connect(args.database) as target:
        target.execute(
            "INSERT INTO sptvs30estimates VALUES (1,1,1,1,150,1,0,0,999,NULL)"
        )
    with batch.open_stage(args.run_dir) as stage:
        batch.run_batch(args, stage)
        with pytest.raises(ValueError, match="no overwrite"):
            batch.publish(args, stage)
    with sqlite3.connect(args.database) as target:
        assert (
            target.execute("SELECT count(*) FROM cptvs30estimates").fetchone()[0] == 0
        )
        assert target.execute("SELECT vs30 FROM sptvs30estimates").fetchall() == [
            (999.0,)
        ]


def test_existing_backup_is_never_overwritten(args: argparse.Namespace):
    with sqlite3.connect(args.backup) as backup:
        backup.execute("CREATE TABLE precious_data (value INTEGER)")
        backup.execute("INSERT INTO precious_data VALUES (2026)")
    before = batch.source_identity(args.backup)
    with batch.open_stage(args.run_dir) as stage:
        batch.run_batch(args, stage)
        with pytest.raises(FileExistsError):
            batch.publish(args, stage)
    assert batch.source_identity(args.backup) == before
    with sqlite3.connect(args.database) as target:
        assert (
            target.execute("SELECT count(*) FROM cptvs30estimates").fetchone()[0] == 0
        )
