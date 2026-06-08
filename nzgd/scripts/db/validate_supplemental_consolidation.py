"""Validate within-record supplemental consolidation against Maxim's backfill.

Runs consolidation on a working COPY of the no-fill deduped DB, then categorises
every per-(cpt_id, field) difference vs the Maxim-filled deduped DB:
match / preserved-0 / intended-improvement / intended-difference (Maxim 0 from
Nil -> our NULL) / genuine-residual-gap / conflict. Read-only on the inputs.
"""

import shutil
import sqlite3
from collections import Counter
from pathlib import Path

from nzgd.dedup.data_types import CPT_TABLE_CONFIG
from nzgd.dedup.schema import apply_dedup_schema
from nzgd.dedup.supplemental_consolidation import consolidate_within_record_supplemental

DATA = Path("/home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data")
NOFILL = DATA / "uc_nzgd_v0p7p0_20260528_deduped_NO_FILL_WITH_MAXIM_VALUES.db"
MAXIM = DATA / "uc_nzgd_v0p7p0_20260528_deduped.db"
WORK = DATA / "supplemental_value_analysis" / "consolidation_validation_work.db"
FIELDS = ("predrill_depth_m", "extracted_gwl_m", "tip_net_area_ratio")


def main() -> None:
    shutil.copyfile(NOFILL, WORK)
    conn = sqlite3.connect(WORK)
    conn.execute("PRAGMA foreign_keys = ON")
    apply_dedup_schema(conn)
    cur = conn.execute(
        "INSERT INTO dedup_run (started_at, source_db_path, script_version, config_snapshot_json) "
        "VALUES ('validate', ?, 'validate', '{}')",
        (str(NOFILL),),
    )
    run_id = cur.lastrowid
    conn.commit()
    recs, cells = consolidate_within_record_supplemental(conn, CPT_TABLE_CONFIG, run_id)
    print(f"consolidation: filled {cells} cells across {recs} records")

    ours = {}
    for r in conn.execute(f"SELECT cpt_id, {', '.join(FIELDS)} FROM cptreport"):
        for i, f in enumerate(FIELDS):
            ours[(r[0], f)] = r[i + 1]
    conn.close()

    mx = sqlite3.connect(f"file:{MAXIM}?mode=ro", uri=True)
    cats: dict[str, Counter] = {f: Counter() for f in FIELDS}
    gaps = []
    for r in mx.execute(f"SELECT cpt_id, {', '.join(FIELDS)} FROM cptreport"):
        cpt_id = r[0]
        for i, f in enumerate(FIELDS):
            maxim_v, our_v = r[i + 1], ours.get((cpt_id, f))
            if maxim_v is None:
                continue
            if our_v is not None and abs(float(our_v) - float(maxim_v)) <= 0.011:
                cats[f]["match"] += 1
            elif maxim_v == 0 and our_v is None:
                cats[f]["intended_difference (Maxim 0 -> our NULL)"] += 1
            elif our_v is not None and float(our_v) > 0 and float(maxim_v) == 0:
                cats[f]["intended_improvement (Maxim 0 -> our positive)"] += 1
            elif our_v is None:
                cats[f]["genuine_residual_gap"] += 1
                if len(gaps) < 50:
                    gaps.append((cpt_id, f, maxim_v))
            else:
                cats[f]["other_difference"] += 1
    mx.close()
    for f in FIELDS:
        print(f"\n{f}:")
        for k, n in sorted(cats[f].items()):
            print(f"  {k}: {n}")
    print(f"\nSample genuine residual gaps (investigate): {gaps[:20]}")
    print(f"\nWork DB left at: {WORK}")


if __name__ == "__main__":
    main()
