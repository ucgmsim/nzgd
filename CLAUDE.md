# CLAUDE.md — NZGD Project Guide

## Project Overview

The `nzgd` package extracts geotechnical data from investigation source files in the **New Zealand Geotechnical Database (NZGD)**. The primary focus is extracting Cone Penetration Test (CPT) trace data (depth, qc, fs, u2) from diverse spreadsheet formats (XLS, XLSX, CSV, TXT, AGS) into standardised parquet output files.

## Python Environment

- **Virtual environment**: `/home/arr65/venvs/dev_nzgd_venv/bin/activate`
- **Python binary**: `/home/arr65/venvs/dev_nzgd_venv/bin/python`
- **Run scripts**: `/home/arr65/venvs/dev_nzgd_venv/bin/python -m nzgd.scripts.extract.cpt.extract_cpt_trace_arrays`
- The project `.venv` at `/home/arr65/src/nzgd/.venv/` does NOT have the required dependencies installed. Always use `dev_nzgd_venv`.

## Key Directories

- **Source code**: `nzgd/` (the package)
- **CPT extraction pipeline**: `nzgd/extract/cpt/` — the most developed module
- **Borehole extraction**: `nzgd/extract/bh/`
- **Scripts**: `nzgd/scripts/extract/cpt/` and `nzgd/scripts/db/`
- **Config**: `nzgd/resources/config.yaml` — thresholds, search patterns, file paths
- **Constants**: `nzgd/constants.py` — loads config.yaml and defines enums/module-level constants
- **NZGD source data** (read-only, never modify): `/home/arr65/data/nzgd/downloads/nzgd_source_files/`
- **Extraction output**: `/home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/`

## CPT Data Fundamentals

CPT measurements record four parameters at each depth increment (~1 cm):

| Parameter | Symbol | Typical MPa range | Description |
|-----------|--------|-------------------|-------------|
| Depth     | —      | 0–50 m            | Monotonically increasing, non-negative |
| Cone resistance | qc | 0–100 MPa      | Always positive |
| Sleeve friction | fs | 0–2 MPa         | Should be mostly positive; small negatives from measurement uncertainty |
| Pore pressure   | u2 | -0.1–5 MPa      | Can have significant negative values (only parameter that should) |

Source files often store fs and u2 in **kPa** (or kN/m²) rather than MPa. The pipeline infers and converts units.

## CPT Extraction Pipeline (`nzgd/extract/cpt/`)

The pipeline runs via `workflow.process_one_record(record_dir)`:

1. **extraction.py** — Loads all files for a record, calls `tasks.load_excel_sheet()` or `tasks.safe_load_csv_or_txt()` per sheet/file
2. **validation.py** `identify_missing_columns_in_sheet()` — Checks all 4 required columns are found
3. **conditioning.py** `remove_non_numerical_data_for_one_sheet()` — Strips text rows, coerces to numeric
4. **select_columns.py** `select_columns_for_one_sheet()` — Validates and selects the 4 required columns
5. **validation.py** `validate_initial_extraction_of_sheet()` — Validates data ranges
6. **conditioning.py** `data_conditioning_for_one_sheet()` — Unit conversions (explicit from headers, then inferred from magnitudes), placeholder removal
7. **conditioning.py** `remove_duplicate_extractions()` — Deduplicates across sheets/files
8. **output.py** `write_extracted_data()` — Writes parquet files

### Header Detection (search.py)

Source spreadsheets have messy, inconsistent headers. The pipeline must identify which row(s) contain column headers and which columns correspond to depth/qc/fs/u2.

- **`find_all_candidate_header_rows()`** — Finds text-heavy rows near the data block by searching backwards from the first contiguous run of numerical rows
- **`find_best_header_combination()`** — Tries all subsets of candidate header rows (up to triples), concatenates each combination, searches for CPT column names, and **scores** each combination using `score_column_assignment()` based on data magnitude physics (depth monotonicity, u2 most negative, fs few negatives, column uniqueness)
- **`find_row_indices_of_header_lines()`** — Older/fallback header detection used when scoring finds nothing

### Unit Conversion (tasks.py)

Two stages in `conditioning.data_conditioning_for_one_sheet()`:

1. **Explicit** (`explicit_unit_conversions_in_sheet()`): Reads unit text from column headers (e.g., "kPa", "cm")
2. **Inferred** (`infer_unit_conversions_for_sheet()`): Two complementary checks:
   - *Percentage-based*: If >75% of values exceed a per-parameter threshold (qc>100, fs>10, |u|>5), convert kPa→MPa
   - *Max-value*: If `max(qc)>500`, `max(fs)>5`, or `max(|u|)>10`, values are physically implausible in MPa → convert. This catches cases like u2 with wide distributions where the percentage check fails.

## Key Data Structures (`data_structures.py`)

- **`SheetExtractionResult`** — Wraps either a successful `ExtractedDataAndColInfo` or a failure DataFrame
- **`ExtractedDataAndColInfo`** — Contains `data_df` (the extracted DataFrame) and `col_info` (an `AllCPTColsSearchResults`)
- **`AllCPTColsSearchResults`** — 4 lists of `SearchForColResults`, one per parameter (depth, qc, fs, u)
- **`SearchForColResults`** — `(col_index_in_line, search_term, matched_string)` — maps a parameter to a column index

## Testing a Specific NZGD ID

Edit `nzgd/scripts/extract/cpt/extract_cpt_trace_arrays.py` to set `cpt_nzgd_ids = [YOUR_ID]`, then run:
```bash
/home/arr65/venvs/dev_nzgd_venv/bin/python -m nzgd.scripts.extract.cpt.extract_cpt_trace_arrays
```

Output goes to:
- Success: `extracted_cpt_trace_per_record/{id}.parquet`
- Failure: `failed_cpt_trace_extractions_per_record/{id}.parquet`

## Config Reference (`config.yaml`)

Key sections:
- `search_patterns` — Character and substring patterns for identifying depth/qc/fs/u columns
- `infer_wrong_units_thresholds` — Percentage thresholds and max-value plausibility thresholds for unit conversion
- `column_data_validation` — Scoring thresholds for header combination selection (depth monotonicity, negative fraction limits)
- `known_missing_value_placeholders` — Sentinel values to replace with NaN before processing
- `known_false_positive_column_names` — Column names that match search patterns but aren't the target parameter
- `known_special_cases` — NZGD IDs with known corrupt/unusual files

## Style & Conventions

- Linter: ruff (numpy docstring convention)
- Type hints used throughout
- `constants.py` loads all config at import time as module-level constants
- Error types defined in `nzgd/extract/cpt/errors.py`
- **Imports**: Prefer importing entire modules and referencing with dot notation (e.g., `from nzgd import constants` then `constants.VALUE`, or `from nzgd.extract.cpt import search` then `search.func()`). Exceptions: `from pathlib import Path`, `from tqdm import tqdm`, `import numpy as np`, `import pandas as pd`.
