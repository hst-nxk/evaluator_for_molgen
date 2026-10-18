"""Target types behind ChEMBL Ki/Kd/IC50/EC50 activities (read-only).

Purpose: tell whether 'target_has_no_protein_component' comes from targets that are not
proteins, or from a missing join.

    python scripts\\check_chembl_target_types.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(r"E:\eval_training\work\chembl_37\chembl_37_sqlite\chembl_37.db")
ENDPOINTS = ("Ki", "Kd", "IC50", "EC50")
SINGLE_PROTEIN = "SINGLE PROTEIN"
NULL_LABEL = "<NULL>"
PLACEHOLDERS = ", ".join("?" for _ in ENDPOINTS)

SELECTED_ACTIVITIES = f"""
FROM activities AS act
LEFT JOIN assays AS a ON a.assay_id = act.assay_id
LEFT JOIN target_dictionary AS td ON td.tid = a.tid
WHERE act.standard_type IN ({PLACEHOLDERS})
"""

TARGET_TYPE_DISTRIBUTION = f"""
SELECT td.target_type, COUNT(*) AS activities
{SELECTED_ACTIVITIES}
GROUP BY td.target_type ORDER BY activities DESC
"""

JOIN_GAPS = f"""
SELECT
    SUM(a.assay_id IS NULL),
    SUM(a.assay_id IS NOT NULL AND a.tid IS NULL),
    SUM(a.tid IS NOT NULL AND td.tid IS NULL),
    COUNT(*)
{SELECTED_ACTIVITIES}
"""

SINGLE_PROTEIN_COMPONENTS = f"""
SELECT
    EXISTS (SELECT 1 FROM target_components AS tc WHERE tc.tid = td.tid) AS has_component,
    COUNT(*) AS activities
{SELECTED_ACTIVITIES}
AND td.target_type = ?
GROUP BY has_component
"""


def open_readonly(path: Path) -> sqlite3.Connection:
    """Open the ChEMBL SQLite file read-only; fail clearly if it is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"ChEMBL SQLite not found at {path}")
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA temp_store = MEMORY")
    return connection


def main() -> int:
    connection = open_readonly(DB_PATH)
    try:
        print(f"Reading {DB_PATH}")
        print(f"Activities with standard_type in {ENDPOINTS}\n")

        print("target_dictionary.target_type distribution (activities):")
        for target_type, activities in connection.execute(TARGET_TYPE_DISTRIBUTION, ENDPOINTS):
            print(f"  {target_type or NULL_LABEL:<35} {activities:,}")

        no_assay, no_tid, tid_not_in_targets, total = connection.execute(
            JOIN_GAPS, ENDPOINTS).fetchone()
        print(f"\nJoin gaps (of {total:,} activities):")
        print(f"  activities.assay_id not found in assays:        {no_assay or 0:,}")
        print(f"  assays.tid is NULL:                             {no_tid or 0:,}")
        print(f"  assays.tid not found in target_dictionary:      {tid_not_in_targets or 0:,}")

        counts = dict(connection.execute(SINGLE_PROTEIN_COMPONENTS,
                                         (*ENDPOINTS, SINGLE_PROTEIN)).fetchall())
        print(f"\ntarget_type = '{SINGLE_PROTEIN}':")
        print(f"  with target_components:    {counts.get(1, 0):,} activities")
        print(f"  without target_components: {counts.get(0, 0):,} activities")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
