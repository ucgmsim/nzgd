"""One-time: prove the regenerated index loses nothing vs the legacy index.

Run before deleting any legacy artifact. Non-zero exit on any failure.
See docs/superpowers/specs/2026-06-11-regenerated-nzgd-index-design.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nzgd import constants
from nzgd.metadata.build import RAW_CATALOG_COLUMNS, nan_aware_neq
from nzgd.metadata.location import LOCATION_COLUMNS

LEGACY_INDEX_PATH = (
    constants.RESOURCE_PATH / "nzgd_metadata_from_coordinates_22_august_2025.csv"
)
NEW_INDEX_PATH = constants.RESOURCE_PATH / "nzgd_index.csv.gz"
ARCHIVE_DIR = Path(
    "/home/arr65/data/nzgd/downloads/nzgd_source_files_of_overwritten_nzgd_ids"
)
# IDs whose raw-column divergence is understood and archived (see spec).
KNOWN_DIVERGENT_IDS = {16, 229775, 229822}
# IDs whose location was corrected at seed time (229630 + 229545; see seed script).
SEED_CORRECTED_IDS = {229630, 229545}
# Legacy model column -> new model column (renamed; values sampled fresh).
MODEL_RENAME_MAP = {
    "model_gwl_westerhoff_2018": "model_gwl_westerhoff_2018_m",
    "model_vs30_foster_2019": "model_vs30_foster_2019_m_per_s",
    "model_vs30_std_foster_2019": "model_vs30_stddev_foster_2019_ln",
}


def main() -> int:
    """Run all no-loss checks; return a process exit code."""
    legacy = pd.read_csv(LEGACY_INDEX_PATH, low_memory=False)
    new = pd.read_csv(NEW_INDEX_PATH, low_memory=False)
    failures: list[str] = []

    # 1. Legacy IDs must all survive.
    missing = sorted(set(legacy["nzgd_id"]) - set(new["nzgd_id"]))
    if missing:
        failures.append(f"{len(missing)} legacy IDs missing: {missing[:20]}")
    added = sorted(set(new["nzgd_id"]) - set(legacy["nzgd_id"]))
    print(f"IDs added vs legacy ({len(added)}): {added[:50]}")

    # 2. Raw catalog columns: divergence only on known, archived IDs.
    merged = legacy.merge(new, on="nzgd_id", suffixes=("_old", "_new"), how="inner")
    divergent: set = set()
    for col in RAW_CATALOG_COLUMNS:
        if col == "nzgd_id":
            continue
        mask = nan_aware_neq(merged[f"{col}_old"], merged[f"{col}_new"])
        divergent |= set(merged.loc[mask, "nzgd_id"])
    unexpected = divergent - KNOWN_DIVERGENT_IDS
    if unexpected:
        failures.append(
            "unexpected divergent IDs (review upstream change, archive the old "
            f"row if it is a reassignment, then extend KNOWN_DIVERGENT_IDS): "
            f"{sorted(unexpected)[:20]}"
        )
    print(f"Divergent IDs on raw columns: {sorted(divergent)}")
    show_cols = [
        "nzgd_id", "State", "Type", "TypeDisplay", "InvestigationId",
        "Latitude", "Longitude", "CreatedOn", "LastModifiedOn",
    ]
    for nzgd_id in sorted(divergent):
        print(f"--- {nzgd_id} legacy ---")
        print(legacy.loc[legacy["nzgd_id"] == nzgd_id, show_cols].to_string(index=False))
        print(f"--- {nzgd_id} new ---")
        print(new.loc[new["nzgd_id"] == nzgd_id, show_cols].to_string(index=False))

    # 3. Each divergent ID must have an archived metadata row.
    for nzgd_id in sorted(divergent):
        archive = ARCHIVE_DIR / str(nzgd_id)
        has_row = archive.is_dir() and any(
            "metadata_row" in p.name for p in archive.iterdir()
        )
        if not has_row:
            failures.append(f"no archived metadata row for divergent ID {nzgd_id}")

    # 4. Location columns: identical except documented seed-time corrections.
    loc_old = legacy[["nzgd_id", *LOCATION_COLUMNS]].set_index("nzgd_id")
    loc_new = new[["nzgd_id", *LOCATION_COLUMNS]].set_index("nzgd_id").loc[loc_old.index]
    loc_diff_ids = set(loc_old.index[(loc_old != loc_new).any(axis=1)])
    if loc_diff_ids - SEED_CORRECTED_IDS:
        failures.append(
            f"location changed beyond documented corrections: "
            f"{sorted(loc_diff_ids - SEED_CORRECTED_IDS)[:20]}"
        )
    print(f"Location diffs (expected only {sorted(SEED_CORRECTED_IDS)}): {sorted(loc_diff_ids)}")

    # 5. NZTM coordinates: allclose where both sides have values, excluding
    #    reassigned IDs whose upstream lat/lon legitimately changed (e.g. 229775,
    #    229822 now carry 0,0 placeholders). Those are already accounted for as
    #    KNOWN_DIVERGENT_IDS in check 2, so a coordinate change there is expected.
    nztm = legacy[["nzgd_id", "nztm_y", "nztm_x"]].merge(
        new[["nzgd_id", "nztm_y", "nztm_x"]], on="nzgd_id", suffixes=("_old", "_new")
    )
    nztm = nztm[~nztm["nzgd_id"].isin(KNOWN_DIVERGENT_IDS)]
    both = nztm.drop(columns="nzgd_id").notna().all(axis=1)
    nztm_ok = bool(
        np.allclose(
            nztm.loc[both, ["nztm_y_old", "nztm_x_old"]].values,
            nztm.loc[both, ["nztm_y_new", "nztm_x_new"]].values,
        )
    )
    if not nztm_ok:
        failures.append("nztm values not allclose to legacy")
    print(f"nztm allclose over {int(both.sum()):,} rows (excl. reassigned IDs): {nztm_ok}")

    # 6. Model columns: reported, not asserted (fresh sampling, corrected names;
    #    legacy baked nodata sentinels like -32767 which are now NaN).
    for old_col, new_col in MODEL_RENAME_MAP.items():
        pair = legacy[["nzgd_id", old_col]].merge(new[["nzgd_id", new_col]], on="nzgd_id")
        both_present = pair[old_col].notna() & pair[new_col].notna()
        delta = (pair.loc[both_present, new_col] - pair.loc[both_present, old_col]).abs()
        print(
            f"{old_col} -> {new_col}: n_both={int(both_present.sum()):,} "
            f"max|delta|={delta.max():.6g} legacy_nan={int(pair[old_col].isna().sum())} "
            f"new_nan={int(pair[new_col].isna().sum())}"
        )

    if failures:
        print("\nVERIFICATION FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll no-loss checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
