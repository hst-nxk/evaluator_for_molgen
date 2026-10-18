"""Measure how many sequence lengths (constructs) each SwissProt accession has in BindingDB_All.

Read-only: streams the TSV straight from the ZIP, never extracts or writes anything.

    python scripts\\check_bindingdb_constructs.py
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ALL_ZIP = Path(r"E:\eval_training\raw\bindingdb202610\BindingDB_All_202610_tsv.zip")
ENCODING = "utf-8"
ACCESSION_COLUMN = "UniProt (SwissProt) Primary ID of Target Chain 1"
SEQUENCE_COLUMN = "BindingDB Target Chain Sequence 1"
FOCUS_ACCESSION = "P03367"
TOP_N = 10
PROGRESS_EVERY = 500_000
CSV_FIELD_SIZE_LIMIT = 2_147_483_647


def find_tsv_member(archive: zipfile.ZipFile) -> str:
    """The single .tsv member of the archive."""
    members = [name for name in archive.namelist() if name.lower().endswith(".tsv")]
    if len(members) != 1:
        raise ValueError(f"{ALL_ZIP}: expected exactly one .tsv member, found {members}")
    return members[0]


def column_index(header: list[str], name: str) -> int:
    """Index of a column by exact header name; fail if missing or duplicated."""
    positions = [index for index, column in enumerate(header) if column == name]
    if len(positions) != 1:
        raise ValueError(f"Column {name!r} found {len(positions)} times in header")
    return positions[0]


def cell(fields: list[str], index: int) -> str:
    """Stripped cell value, or '' when the row is shorter than the header."""
    return fields[index].strip() if index < len(fields) else ""


def collect_lengths() -> tuple[int, int, dict[str, Counter]]:
    """Stream the TSV; return (data rows, rows without accession or sequence, lengths per accession)."""
    lengths: dict[str, Counter] = defaultdict(Counter)
    total_rows = rows_missing = 0
    with zipfile.ZipFile(ALL_ZIP) as archive, archive.open(find_tsv_member(archive)) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding=ENCODING, newline=""),
                            delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        accession_index = column_index(header, ACCESSION_COLUMN)
        sequence_index = column_index(header, SEQUENCE_COLUMN)
        for fields in reader:
            total_rows += 1
            accession = cell(fields, accession_index)
            sequence = "".join(cell(fields, sequence_index).split())
            if accession and sequence:
                lengths[accession][len(sequence)] += 1
            else:
                rows_missing += 1
            if total_rows % PROGRESS_EVERY == 0:
                print(f"  ... {total_rows:,} rows read", flush=True)
    return total_rows, rows_missing, lengths


def main() -> int:
    csv.field_size_limit(CSV_FIELD_SIZE_LIMIT)
    print(f"Reading {ALL_ZIP}")
    total_rows, rows_missing, lengths = collect_lengths()

    print(f"\nTotal data rows: {total_rows:,}")
    print(f"Rows without chain-1 SwissProt ID or chain-1 sequence: {rows_missing:,}")

    focus = lengths.get(FOCUS_ACCESSION, Counter())
    print(f"\n{FOCUS_ACCESSION}: {len(focus)} distinct sequence lengths (length: rows)")
    for length, rows in sorted(focus.items()):
        print(f"  {length} aa: {rows:,} rows")

    multi = [accession for accession, counter in lengths.items() if len(counter) > 1]
    print(f"\nAccessions with more than one sequence length: {len(multi):,} / {len(lengths):,}")

    spreads = sorted(((max(counter) - min(counter), accession)
                      for accession, counter in lengths.items()), reverse=True)[:TOP_N]
    print(f"\nTop {TOP_N} accessions by length spread (max - min):")
    for spread, accession in spreads:
        listed = ", ".join(f"{length}({rows:,})"
                           for length, rows in sorted(lengths[accession].items()))
        print(f"  {accession}: spread {spread} aa; lengths(rows): {listed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
