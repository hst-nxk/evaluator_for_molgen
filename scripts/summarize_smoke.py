"""Summarize the newest smoke run in runs/observations/smoke_*/ (read-only).

    python scripts\\summarize_smoke.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

RUNS_DIR = Path(r"E:\eval_training\runs\observations")
SMOKE_PATTERN = "smoke_*"
STATUS_COLUMNS = ("protein_status", "assay_status", "structure_status", "activity_status")
SAMPLE_ROWS = 3
SUMMARY_COLUMNS = [
    "source_database", *STATUS_COLUMNS,
    *(column.replace("_status", "_reason") for column in STATUS_COLUMNS),
    "activity_censored",
]
SAMPLE_COLUMNS = [
    "observation_id", "protein_accession", "protein_sequence", "assay_native_id", "endpoint",
    "activity_value_source", "activity_unit_source", "activity_qualifier_source",
    "activity_qualifier", "activity_p_activity",
]


def latest_smoke_dir() -> Path:
    """Newest smoke_<YYYYmmdd_HHMMSS> directory (the timestamp sorts by name)."""
    candidates = sorted(path for path in RUNS_DIR.glob(SMOKE_PATTERN) if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No {SMOKE_PATTERN} directory in {RUNS_DIR}; run --mode smoke")
    return candidates[-1]


def print_summary(path: Path) -> None:
    """Row count, status/reason distribution and censored count for one file."""
    rows = pq.read_table(path, columns=SUMMARY_COLUMNS).to_pylist()
    sources = sorted({row["source_database"] for row in rows})
    print(f"\n=== {path.name}  source={', '.join(sources)}  rows={len(rows):,}")
    for status_column in STATUS_COLUMNS:
        reason_column = status_column.replace("_status", "_reason")
        counts = Counter((row[status_column], row[reason_column]) for row in rows)
        print(f"  {status_column}:")
        for (status, reason), count in counts.most_common():
            print(f"    {status:<10} {reason or '-':<70} {count:,}")
    censored = sum(1 for row in rows if row["activity_censored"] is True)
    print(f"  censored rows: {censored:,}")


def print_samples(path: Path) -> None:
    """The first rows of one file, with the requested fields."""
    rows = pq.read_table(path, columns=SAMPLE_COLUMNS).slice(0, SAMPLE_ROWS).to_pylist()
    print(f"  sample rows ({len(rows)}):")
    for row in rows:
        sequence = row["protein_sequence"]
        qualifier = row["activity_qualifier_source"] or row["activity_qualifier"]
        print(f"    {row['observation_id']}: accession={row['protein_accession']} "
              f"seq_len={len(sequence) if sequence else None} "
              f"assay={row['assay_native_id']} endpoint={row['endpoint']} "
              f"value={row['activity_value_source']!r} unit={row['activity_unit_source']} "
              f"qualifier={qualifier} pActivity={row['activity_p_activity']}")


def main() -> int:
    smoke_dir = latest_smoke_dir()
    paths = sorted(smoke_dir.glob("*.parquet"))
    print(f"Smoke run: {smoke_dir} ({len(paths)} parquet files)")
    if len(paths) != 2:
        print(f"WARNING: expected 2 parquet files, found {len(paths)}")
    for path in paths:
        print_summary(path)
        print_samples(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
