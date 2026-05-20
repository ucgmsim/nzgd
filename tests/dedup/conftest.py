"""Synthetic-DB fixtures for dedup integration tests."""

import sqlite3
from pathlib import Path

import pytest


_FULL_SCHEMA_SQL = """
CREATE TABLE type (id INTEGER PRIMARY KEY, value TEXT);
INSERT INTO type VALUES (1, 'CPT'), (2, 'BH');

CREATE TABLE nzgdrecord (
    nzgd_id INTEGER PRIMARY KEY,
    type_id INTEGER NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    model_vs30_foster_2019_km_per_s REAL,
    model_vs30_stddev_foster_2019_km_per_s REAL,
    model_gwl_westerhoff_2018_m REAL,
    model_gwl_nlm_2025_m REAL,
    model_gwl_nlm_2025_stddev_m REAL,
    original_investigation_name TEXT,
    investigation_date TEXT,
    published_date TEXT,
    region_id INTEGER NOT NULL DEFAULT 0,
    district_id INTEGER NOT NULL DEFAULT 0,
    city_id INTEGER NOT NULL DEFAULT 0,
    suburb_id INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE cptreport (
    cpt_id INTEGER PRIMARY KEY,
    nzgd_id INTEGER NOT NULL,
    max_depth_m REAL,
    min_depth_m REAL,
    extracted_gwl_m REAL,
    gwl_method_id INTEGER,
    tip_net_area_ratio REAL,
    predrill_depth_m REAL,
    termination_reason_id INTEGER,
    has_cpt_data INTEGER NOT NULL DEFAULT 1,
    cpt_data_duplicate_of_cpt_id INTEGER,
    did_explicit_unit_conversion INTEGER,
    did_inferred_unit_conversion INTEGER,
    source_file TEXT NOT NULL DEFAULT 'synthetic.xls'
);

CREATE TABLE cptmeasurements (
    measurement_id INTEGER PRIMARY KEY,
    cpt_id INTEGER NOT NULL,
    depth_m REAL,
    qc_MPa REAL,
    fs_MPa REAL,
    u2_MPa REAL
);

CREATE TABLE cptvs30estimates (
    vs30_id INTEGER PRIMARY KEY,
    cpt_id INTEGER,
    nzgd_id INTEGER,
    cpt_to_vs_correlation_id INTEGER,
    vs_to_vs30_correlation_id INTEGER,
    vs30 REAL,
    vs30_stddev REAL
);

CREATE TABLE sptreport (
    spt_id INTEGER PRIMARY KEY,
    nzgd_id INTEGER NOT NULL,
    efficiency REAL,
    extracted_gwl_m REAL,
    borehole_diameter REAL,
    casing_diameter REAL,
    source_file TEXT NOT NULL DEFAULT 'synthetic.ags'
);

CREATE TABLE sptmeasurements (
    spt_measurement_id INTEGER PRIMARY KEY,
    spt_id INTEGER NOT NULL,
    depth_m REAL,
    ISPT_MAIN INTEGER,
    ISPT_NVAL INTEGER,
    ISPT_REP INTEGER
);

CREATE TABLE soilmeasurements (
    soil_measurement_id INTEGER PRIMARY KEY,
    spt_id INTEGER NOT NULL,
    top_depth_m REAL,
    bottom_depth_m REAL
);

CREATE TABLE densitymeasurements (
    density_measurement_id INTEGER PRIMARY KEY,
    spt_id INTEGER NOT NULL,
    top_depth_m REAL,
    bottom_depth_m REAL,
    density_keyword TEXT
);

CREATE TABLE soilmeasurementsoiltype (
    soil_measurement_id INTEGER NOT NULL,
    soil_type_id INTEGER NOT NULL,
    PRIMARY KEY (soil_measurement_id, soil_type_id)
);

CREATE TABLE sptvs30estimates (
    vs30_id INTEGER PRIMARY KEY,
    spt_id INTEGER,
    spt_to_vs_correlation_id INTEGER,
    vs_to_vs30_correlation_id INTEGER,
    assumed_borehole_diameter_mm REAL,
    assumed_hammer_type_id INTEGER,
    estimate_used_extracted_efficiency INTEGER,
    estimate_used_extracted_layer_soil_types INTEGER,
    vs30 REAL,
    vs30_stddev REAL
);

CREATE INDEX cptmeasurements_cpt_id ON cptmeasurements(cpt_id);
CREATE INDEX cptreport_nzgd_id ON cptreport(nzgd_id);
CREATE INDEX sptmeasurements_spt_id ON sptmeasurements(spt_id);
CREATE INDEX sptreport_nzgd_id ON sptreport(nzgd_id);
CREATE INDEX soilmeasurements_spt_id ON soilmeasurements(spt_id);
CREATE INDEX densitymeasurements_spt_id ON densitymeasurements(spt_id);
CREATE INDEX soilmeasurementsoiltype_soil_measurement_id ON soilmeasurementsoiltype(soil_measurement_id);
"""


def _make_fresh_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_FULL_SCHEMA_SQL)
    return conn


def add_cpt_record(
    conn: sqlite3.Connection,
    nzgd_id: int,
    lat: float = -41.0,
    lon: float = 174.0,
    investigation_name: str | None = None,
    investigation_date: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO nzgdrecord (nzgd_id, type_id, latitude, longitude, "
        "original_investigation_name, investigation_date) VALUES (?,1,?,?,?,?)",
        (nzgd_id, lat, lon, investigation_name, investigation_date),
    )


def add_bh_record(
    conn: sqlite3.Connection,
    nzgd_id: int,
    lat: float = -41.0,
    lon: float = 174.0,
    investigation_name: str | None = None,
    investigation_date: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO nzgdrecord (nzgd_id, type_id, latitude, longitude, "
        "original_investigation_name, investigation_date) VALUES (?,2,?,?,?,?)",
        (nzgd_id, lat, lon, investigation_name, investigation_date),
    )


def add_cpt_report(
    conn: sqlite3.Connection,
    cpt_id: int,
    nzgd_id: int,
    trace: list[tuple[float, float, float, float]],
    source_file: str = "synthetic.xls",
) -> None:
    conn.execute(
        "INSERT INTO cptreport (cpt_id, nzgd_id, max_depth_m, min_depth_m, has_cpt_data, source_file) "
        "VALUES (?,?,?,?,1,?)",
        (cpt_id, nzgd_id,
         max(r[0] for r in trace) if trace else None,
         min(r[0] for r in trace) if trace else None,
         source_file),
    )
    conn.executemany(
        "INSERT INTO cptmeasurements (cpt_id, depth_m, qc_MPa, fs_MPa, u2_MPa) VALUES (?,?,?,?,?)",
        [(cpt_id, *row) for row in trace],
    )


def add_spt_report(
    conn: sqlite3.Connection,
    spt_id: int,
    nzgd_id: int,
    trace: list[tuple[float, int, int, int]],
    source_file: str = "synthetic.ags",
) -> None:
    conn.execute(
        "INSERT INTO sptreport (spt_id, nzgd_id, source_file) VALUES (?,?,?)",
        (spt_id, nzgd_id, source_file),
    )
    conn.executemany(
        "INSERT INTO sptmeasurements (spt_id, depth_m, ISPT_MAIN, ISPT_NVAL, ISPT_REP) VALUES (?,?,?,?,?)",
        [(spt_id, *row) for row in trace],
    )


@pytest.fixture
def fresh_db(tmp_path: Path) -> sqlite3.Connection:
    """A blank schema-loaded DB. Each test populates it as needed."""
    db_path = tmp_path / "fresh.db"
    conn = _make_fresh_db(db_path)
    yield conn
    conn.close()
