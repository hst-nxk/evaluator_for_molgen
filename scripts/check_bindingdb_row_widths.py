"""Kiểm tra chỉ đọc: dòng TSV BindingDB nào thiếu cột mà Stage 1 đang dùng.

Chạy từ thư mục gốc của repo:
    python scripts\\check_bindingdb_row_widths.py
    python scripts\\check_bindingdb_row_widths.py <thư mục gốc khác>

Đọc 3 file ZIP trong raw/bindingdb202610/ đúng cách Stage 1 đọc (UTF-8, tab, QUOTE_NONE),
không giải nén, không ghi file nào. Với mỗi file in ra:
  - số cột của header và chỉ số lớn nhất của cột Stage 1 dùng;
  - phân bố độ rộng dòng (số trường mỗi dòng);
  - số dòng ngắn hơn header (có thể bình thường: BindingDB có thể bỏ ô trống cuối dòng);
  - số dòng không chạm tới cột lớn nhất Stage 1 dùng (đáng nghi: Stage 1 đọc ô đó là rỗng);
  - riêng file All: số dòng có Reactant_set_id không phải số nguyên (dấu hiệu dòng bị
    tách đôi bởi ký tự xuống dòng nằm trong một ô, hoặc cột bị lệch).
Vì sao cần: check_row_width của Stage 1 chỉ bắt dòng DÀI hơn header. Dòng ngắn hơn đi
qua im lặng, và các ô thiếu được đọc là rỗng.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

EXAMPLES = 5
TOP_WIDTHS = 15
PROGRESS_EVERY = 500_000


def tsv_member(zip_path: Path) -> str:
    """Tên file .tsv duy nhất trong ZIP (giống Stage 1: phải có đúng một)."""
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".tsv")]
    if len(members) != 1:
        raise ValueError(f"{zip_path}: cần đúng một file .tsv, thấy {members}")
    return members[0]


def wanted_indexes(header: list[str], names: list[str], label: str) -> dict[str, int]:
    """Chỉ số của từng cột theo tên; dừng nếu một tên không xuất hiện đúng một lần."""
    indexes = {}
    for name in names:
        positions = [index for index, column in enumerate(header) if column == name]
        if len(positions) != 1:
            raise ValueError(f"{label}: cột {name!r} xuất hiện {len(positions)} lần")
        indexes[name] = positions[0]
    return indexes


def scan(zip_path: Path, column_names: list[str], encoding: str, id_column: str | None) -> None:
    """Đọc một file và in thống kê độ rộng dòng."""
    started = time.monotonic()
    widths: Counter = Counter()
    short_of_header = 0
    short_of_used: list[int] = []
    short_of_used_count = 0
    bad_ids: list[tuple[int, str]] = []
    bad_id_count = 0
    with zipfile.ZipFile(zip_path) as archive, archive.open(tsv_member(zip_path)) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding=encoding, newline=""),
                            delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        indexes = wanted_indexes(header, column_names, zip_path.name)
        max_used = max(indexes.values())
        id_index = indexes[id_column] if id_column else None
        rows = 0
        for line_number, fields in enumerate(reader, start=2):
            rows += 1
            width = len(fields)
            widths[width] += 1
            if width < len(header):
                short_of_header += 1
            if width <= max_used:
                short_of_used_count += 1
                if len(short_of_used) < EXAMPLES:
                    short_of_used.append(line_number)
            if id_index is not None:
                value = fields[id_index].strip() if id_index < width else ""
                if not value.isdigit():
                    bad_id_count += 1
                    if len(bad_ids) < EXAMPLES:
                        bad_ids.append((line_number, value[:40]))
            if rows % PROGRESS_EVERY == 0:
                print(f"  ... {rows:,} dòng, {time.monotonic() - started:.0f} giây",
                      file=sys.stderr)

    print(f"\n== {zip_path.name}")
    print(f"  dòng dữ liệu (không kể header): {rows:,}")
    print(f"  số cột header: {len(header)}; chỉ số lớn nhất Stage 1 dùng: {max_used} "
          f"(cột {header[max_used]!r})")
    print(f"  dòng ngắn hơn header:           {short_of_header:,}")
    print(f"  dòng không chạm cột Stage 1 dùng: {short_of_used_count:,}   "
          f"ví dụ số dòng: {short_of_used}")
    if id_column:
        print(f"  Reactant_set_id không phải số:  {bad_id_count:,}   ví dụ: {bad_ids}")
    print(f"  phân bố độ rộng dòng (top {TOP_WIDTHS}): số trường -> số dòng")
    for width, count in sorted(widths.most_common(TOP_WIDTHS)):
        marker = "  <- không chạm cột Stage 1 dùng" if width <= max_used else ""
        print(f"    {width:>5} -> {count:>12,}{marker}")
    print(f"  thời gian: {time.monotonic() - started:.0f} giây")


def main() -> int:
    """Điểm vào: đọc config của repo, quét 3 file ZIP BindingDB."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Mặc định là thư mục gốc repo (thư mục cha của scripts/); có thể truyền đường dẫn khác.
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    settings = config["bindingdb"]
    csv.field_size_limit(settings["csv_field_size_limit"])
    raw_dir = root / config["paths"]["raw_bindingdb_dir"]
    all_columns = list(settings["all_columns"].values()) + [
        spec["column"] for spec in settings["endpoint_columns"].values()]
    scan(raw_dir / settings["all_zip"], all_columns, settings["encoding"],
         settings["all_columns"]["reactant_set_id"])
    scan(raw_dir / settings["rsid_eaids_zip"], list(settings["rsid_eaids_columns"].values()),
         settings["encoding"], None)
    scan(raw_dir / settings["assays_zip"], list(settings["assays_columns"].values()),
         settings["encoding"], None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
