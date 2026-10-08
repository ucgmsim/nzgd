# nzgd

Tools to extract, clean, deduplicate and analyse geotechnical data from the
investigation source files in the
[New Zealand Geotechnical Database (NZGD)](https://www.nzgd.org.nz/).

NZGD hosts tens of thousands of Cone Penetration Tests (CPTs) and borehole logs
uploaded by many different organisations, in many inconsistent formats (XLS,
XLSX, CSV, TXT, AGS and PDF). This package turns those files into a single,
standardised SQLite database of CPT and SPT measurements with location and
model metadata, removes duplicate uploads, and estimates Vs30 from the
measurements.

## Capabilities

| Area | Package / scripts | What it does |
|------|-------------------|--------------|
| NZGD index | `nzgd/metadata`, `nzgd/scripts/metadata` | Rebuilds `nzgd_index.csv.gz` from the NZGD API investigation catalog: classifies each record's region/district/city/suburb (LINZ shapefiles), computes NZTM coordinates, and samples groundwater and Vs30 model rasters at each location. |
| CPT trace extraction | `nzgd/extract/cpt`, `nzgd/scripts/extract/cpt` | Extracts depth, qc, fs and u2 from messy CPT spreadsheets and AGS files. Detects header rows, scores candidate column assignments using the physics of the data, removes text and placeholder values, and converts units (from headers, or inferred from magnitudes). Writes one parquet file per record. |
| CPT supplemental values | `nzgd/scripts/extract/cpt` | Keyword search for groundwater level (and how it was measured), tip net area ratio, predrill depth and termination reason, then filters the candidates to one value per sheet. |
| Borehole / SPT extraction | `nzgd/extract/bh`, `nzgd/scripts/extract/bh` | Mines borehole logs (AGS and PDF) for SPT N-values, depths, soil types, density descriptions, hammer efficiency and borehole/casing diameter. |
| Database | `nzgd/db`, `nzgd/scripts/db` | SQLite schema (peewee ORM) with lookup, record, CPT, SPT and Vs30 estimate tables, plus scripts that load the extracted data and NZGD metadata into it. |
| Deduplication | `nzgd/dedup`, `nzgd/scripts/db/deduplicate.py` | Writes a cleaned copy of the database: discards CPT reports with a constant measurement column, merges duplicate reports within a record (Pass 0), across records by exact trace hash (Pass 1) and by fuzzy matching on location, date, name and trace similarity (Pass 2), and consolidates supplemental values. Every change is recorded in audit tables and CSV reports. The source database is never modified. |
| Metadata summary | `nzgd/scripts/metadata/make_metadata_summary_csv.py` | One-row-per-record CSV summary of the deduplicated database, including tombstones that point merged records to their canonical record. |
| Vs30 estimation | `nzgd/scripts/estimate_vs30` | Resumable, checkpointed batch calculation of Vs30 from CPT and SPT traces using several published correlations (via `vs_calc`), with a separate, explicit step to publish the results into the database. |

## End-to-end workflow

The NZGD source files are downloaded by the separate `api_nzgd` repository,
which also produces the catalogs stored in `nzgd/resources/nzgd_catalogs_from_api/`.
After that, a database release is built roughly as follows:

```bash
# 1. Rebuild the NZGD index from the API catalog
python nzgd/scripts/metadata/build_nzgd_index.py

# 2. Extract CPT traces and supplemental values
python -m nzgd.scripts.extract.cpt.extract_cpt_trace_arrays
python -m nzgd.scripts.extract.cpt.extract_all_potential_cpt_supplemental_values
python -m nzgd.scripts.extract.cpt.filter_potential_cpt_supplemental_values

# 3. Create the (empty) databases, then extract SPT data from borehole logs
python -m nzgd.scripts.db.create_empty_db_and_fill_support_tables
python -m nzgd.scripts.extract.bh.extract_spt

# 4. Load everything into the main database
python -m nzgd.scripts.db.put_cpts_in_db
python -m nzgd.scripts.db.put_spt_in_main_db
python -m nzgd.scripts.db.put_nzgd_metadata

# 5. Deduplicate (writes <source>_deduped.db and audit CSVs)
python -m nzgd.scripts.db.deduplicate --source <uc_nzgd_vX.db>

# 6. Summarise the deduplicated database
python -m nzgd.scripts.metadata.make_metadata_summary_csv

# 7. Estimate Vs30, then publish the results into the database
python -m nzgd.scripts.estimate_vs30.batch run <db> --run-dir <dir>
python -m nzgd.scripts.estimate_vs30.batch publish <db> --run-dir <dir> --backup <backup.db>
```

Most scripts take no arguments. Input and output paths, database versions and
all extraction thresholds come from `nzgd/resources/config.yaml`.

### Outputs

- Per-record CPT trace parquet files, plus parquet files describing failed extractions
- CSV files of CPT supplemental values
- The SQLite database `uc_nzgd_v<version>_<date>.db` and its deduplicated copy
  `..._deduped.db` (the main deliverable), with dedup audit tables and CSV reports
- A metadata summary CSV
- Vs30 estimates in the `cptvs30estimates` and `sptvs30estimates` tables

## Repository layout

```
nzgd/
  constants.py        Loads config.yaml; enums and module-level constants
  extract/cpt/        CPT trace extraction pipeline
  extract/bh/         Borehole log (AGS / PDF) SPT miners
  db/                 Database ORM schema and CPT id assignment
  dedup/              Quality filter and deduplication passes
  metadata/           NZGD index build (location, rasters, atomic writes)
  resources/          config.yaml, NZGD API catalogs, nzgd_index.csv.gz,
                      location sidecar, soil unit weights
  scripts/            Entry points (extract, db, metadata, estimate_vs30);
                      scripts/temp/ holds ad-hoc analysis, not part of the pipeline
tests/                pytest tests (dedup, metadata, Vs30 batch, extraction filters, AGS miner)
docs/                 Investigation notes, design specs and implementation plans
KNOWN_ISSUES.md       Known issues and technical debt
```

## Installation

Requires Python 3.11 or later.

```bash
pip install -e .
```

Dependencies are listed in `requirements.txt`. Vs30 estimation also needs
`vs_calc`, and the index build needs `qcore` for coordinate transforms; neither
is in `requirements.txt` yet.

## Tests

```bash
python -m pytest tests
```

The Vs30 batch tests need `vs_calc` to be installed.

## Known issues

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md). In particular, the borehole
groundwater-level AGS extraction script still depends on the retired
`nzgd_data_extraction` package.
