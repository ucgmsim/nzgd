"""Post-dedup verification: confirm no cross-record SPT merge orphaned a format.

Read-only safety net for the SPT AGS/PDF-preservation policy. Cross-record SPT
merges (the hash and fuzzy passes) delete the duplicated report on the merged
side. This check confirms that, for every deleted SPT report, a report of the
same source-file format (`.ags` / `.pdf`) still survives under the final
surviving record — i.e. the merge never dropped the only copy of a format for a
borehole. CPT is intentionally excluded: it merges aggressively by design, so a
dropped format there is expected, not a finding.
"""

import json
import sqlite3


def _spt_format(source_file: str | None) -> str:
    """Classify an SPT source file as 'ags', 'pdf', or 'other' by extension."""
    if not source_file:
        return "other"
    lowered = source_file.lower()
    if lowered.endswith(".ags"):
        return "ags"
    if lowered.endswith(".pdf"):
        return "pdf"
    return "other"


def find_spt_format_orphans(
    deduped_conn: sqlite3.Connection, source_conn: sqlite3.Connection
) -> list[dict]:
    """Return one dict per cross-record SPT merge that orphaned a source-file format.

    An empty list means the run preserved every AGS/PDF format across cross-record
    SPT merges. `source_conn` is the original (pre-dedup) DB, needed because the
    deleted reports no longer exist in the deduped DB.
    """
    src_format = {
        spt_id: _spt_format(source_file)
        for spt_id, source_file in source_conn.execute(
            "SELECT spt_id, source_file FROM sptreport"
        )
    }

    survivor_formats: dict[int, set[str]] = {}

    def final_survivor(nzgd_id: int) -> int:
        """Follow merged_into_nzgd_id to the active record this nzgd_id resolves to."""
        seen: set[int] = set()
        while nzgd_id not in seen:
            seen.add(nzgd_id)
            row = deduped_conn.execute(
                "SELECT merged_into_nzgd_id FROM nzgdrecord WHERE nzgd_id = ?", (nzgd_id,)
            ).fetchone()
            if row is None or row[0] is None:
                break
            nzgd_id = row[0]
        return nzgd_id

    def formats_under(nzgd_id: int) -> set[str]:
        if nzgd_id not in survivor_formats:
            survivor_formats[nzgd_id] = {
                _spt_format(source_file)
                for (source_file,) in deduped_conn.execute(
                    "SELECT source_file FROM sptreport WHERE nzgd_id = ?", (nzgd_id,)
                )
            }
        return survivor_formats[nzgd_id]

    orphans: list[dict] = []
    audit_rows = deduped_conn.execute(
        "SELECT cluster_id, canonical_nzgd_id, merged_nzgd_id, match_pass, report_pairs_json "
        "FROM dedup_audit WHERE record_type = 'BH' AND match_pass IN ('hash', 'fuzzy')"
    ).fetchall()
    for cluster_id, canonical_nzgd_id, merged_nzgd_id, match_pass, pairs_json in audit_rows:
        survivor = final_survivor(canonical_nzgd_id)
        kept = formats_under(survivor)
        for pair in json.loads(pairs_json):
            deleted_id = pair["merged_report_id"]
            fmt = src_format.get(deleted_id, "unknown")
            if fmt not in kept:
                orphans.append(
                    {
                        "cluster_id": cluster_id,
                        "match_pass": match_pass,
                        "canonical_nzgd_id": canonical_nzgd_id,
                        "final_survivor_nzgd_id": survivor,
                        "merged_nzgd_id": merged_nzgd_id,
                        "deleted_report_id": deleted_id,
                        "orphaned_format": fmt,
                        "formats_kept_by_survivor": ",".join(sorted(kept)) or "(none)",
                    }
                )
    return orphans
