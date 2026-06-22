"""One-time: seed nzgd_id_to_location.csv.gz from the legacy 22-Aug-2025 index.

Preserves the manually assembled location classifications as a first-class,
git-tracked artifact. Reports rows whose coordinates fall outside the NZ
bounding box yet carry a real classification, for case-by-case review.
"""

import sys

import pandas as pd

from nzgd import constants
from nzgd.metadata.io_utils import atomic_write_csv_gz
from nzgd.metadata.location import LOCATION_COLUMNS, UNCLASSIFIED, coords_outside_nz

LEGACY_INDEX_PATH = (
    constants.RESOURCE_PATH / "nzgd_metadata_from_coordinates_22_august_2025.csv"
)
SIDECAR_PATH = constants.RESOURCE_PATH / "nzgd_id_to_location.csv.gz"

# Upstream NZGD test/garbage records at impossible coordinates that the legacy
# index nonetheless gave a fabricated NZ classification. Reviewed case-by-case
# (see docs/superpowers/specs/2026-06-11-regenerated-nzgd-index-design.md):
#   229630 - test record at 12.123/34.123 (Sudan), classified
#            Canterbury/Christchurch_City/Christchurch/Casebrook.
#   229545 - CPT test record "ISAB-124bbbf444" at 0.0/0.0 (null island),
#            classified Canterbury/Selwyn_District/Dunsandel; bulk-edited
#            2025-06-24 alongside the other upstream test records.
# Note: 8753 (a real 2012 Automatic Ballast Sampler record with corrupted
# coordinates 33.0/170.0 but a plausible Christchurch/Merivale classification)
# is deliberately NOT corrected here - its coordinates are bad but its
# classification is likely legitimate, so the value is preserved.
KNOWN_CORRECTIONS = {229630: UNCLASSIFIED, 229545: UNCLASSIFIED}


def main() -> int:
    """Seed the sidecar; return a process exit code."""
    legacy = pd.read_csv(LEGACY_INDEX_PATH, low_memory=False)
    sidecar = legacy[["nzgd_id", *LOCATION_COLUMNS]].copy()

    outside = coords_outside_nz(legacy)
    classified = (sidecar[LOCATION_COLUMNS] != UNCLASSIFIED).any(axis=1)
    flagged = legacy.loc[
        outside & classified, ["nzgd_id", "Latitude", "Longitude", *LOCATION_COLUMNS]
    ]
    if len(flagged):
        print("Out-of-NZ coordinates but carrying a classification (review case-by-case):")
        print(flagged.to_string(index=False))

    for nzgd_id, value in KNOWN_CORRECTIONS.items():
        sidecar.loc[sidecar["nzgd_id"] == nzgd_id, LOCATION_COLUMNS] = value
        print(f"Applied documented correction: {nzgd_id} -> {value}")

    sidecar = sidecar.sort_values("nzgd_id").reset_index(drop=True)
    atomic_write_csv_gz(sidecar, SIDECAR_PATH)
    print(
        f"Wrote {SIDECAR_PATH} with {len(sidecar):,} rows "
        f"({int(classified.sum()):,} carrying a classification)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
