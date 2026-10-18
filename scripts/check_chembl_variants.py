"""Measure the scale of variant/component questions in ChEMBL 37 before the full run.

Read-only: opens the SQLite file in URI mode=ro and writes nothing.

    python scripts\\check_chembl_variants.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(r"E:\eval_training\work\chembl_37\chembl_37_sqlite\chembl_37.db")
ENDPOINTS = ("Ki", "Kd", "IC50", "EC50")
TOP_N = 10
PLACEHOLDERS = ", ".join("?" for _ in ENDPOINTS)

ACTIVITIES_BY_ENDPOINT = f"""
SELECT standard_type, COUNT(*) FROM activities
WHERE standard_type IN ({PLACEHOLDERS}) GROUP BY standard_type
"""

VARIANT_ACTIVITIES_BY_ENDPOINT = f"""
SELECT act.standard_type, COUNT(*) FROM activities AS act
JOIN assays AS a ON a.assay_id = act.assay_id
WHERE act.standard_type IN ({PLACEHOLDERS}) AND a.variant_id IS NOT NULL
GROUP BY act.standard_type
"""

VARIANT_SEQUENCE_PRESENCE = """
SELECT
    SUM(sequence IS NOT NULL AND TRIM(sequence) <> ''),
    SUM(sequence IS NULL OR TRIM(sequence) = ''),
    COUNT(*)
FROM variant_sequences
"""

COMPONENTS_PER_TARGET = """
SELECT component_count, COUNT(*) AS targets FROM (
    SELECT td.tid, COUNT(tc.component_id) AS component_count
    FROM target_dictionary AS td
    LEFT JOIN target_components AS tc ON tc.tid = td.tid
    GROUP BY td.tid
) GROUP BY component_count ORDER BY targets DESC LIMIT ?
"""


def open_readonly(path: Path) -> sqlite3.Connection:
    """Open the ChEMBL SQLite file read-only; fail clearly if it is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"ChEMBL SQLite not found at {path}")
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA temp_store = MEMORY")
    return connection


def print_by_endpoint(title: str, rows: list[tuple[str, int]]) -> None:
    """Print one count per endpoint, including endpoints with zero rows."""
    counts = dict(rows)
    print(title)
    for endpoint in ENDPOINTS:
        print(f"  {endpoint}: {counts.get(endpoint, 0):,}")


def main() -> int:
    connection = open_readonly(DB_PATH)
    try:
        print(f"Reading {DB_PATH}\n")
        print_by_endpoint("Activities by endpoint (standard_type):",
                          connection.execute(ACTIVITIES_BY_ENDPOINT, ENDPOINTS).fetchall())
        print_by_endpoint("\n...of which assays.variant_id IS NOT NULL:",
                          connection.execute(VARIANT_ACTIVITIES_BY_ENDPOINT, ENDPOINTS).fetchall())

        with_sequence, without_sequence, total = connection.execute(
            VARIANT_SEQUENCE_PRESENCE).fetchone()
        print(f"\nvariant_sequences: {total:,} rows; with sequence {with_sequence or 0:,}; "
              f"missing sequence {without_sequence or 0:,}")

        print(f"\nComponents per target (top {TOP_N} by number of targets):")
        for component_count, targets in connection.execute(COMPONENTS_PER_TARGET, (TOP_N,)):
            print(f"  {component_count} component(s): {targets:,} targets")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
