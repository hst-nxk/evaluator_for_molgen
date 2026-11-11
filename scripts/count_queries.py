"""Đếm query, protein và mức mất cân bằng protein của một bộ dữ liệu evaluator.

    python scripts/count_queries.py
    python scripts/count_queries.py --key protein_hash,smiles_a
    python scripts/count_queries.py --dataset data/dataset_v3 --json

Script CHỈ ĐỌC: không ghi gì vào thư mục dữ liệu.

VÌ SAO CÓ FILE NÀY: "query" không phải một cột có sẵn trong dữ liệu, nó là một KHÓA GOM
NHÓM do người đếm tự chọn, nên hai người đếm hai kiểu sẽ ra hai con số khác nhau mà vẫn
đúng cả hai. Ngày 07/11/2026 Tùng báo train có 669.035 query còn ở đây đếm được 636.118;
dữ liệu giống nhau (protein lớn nhất đều 169.073 dòng) nên chênh lệch nằm ở khóa gom.
Vì vậy khóa gom là THAM SỐ dòng lệnh chứ không chôn cứng trong code, và mọi con số in ra
đều kèm khóa đã dùng.

Mặc định dùng (protein_hash, smiles_a, endpoint): một query là "cho protein này, phân tử
mốc này, loại điểm này, hãy xếp hạng các phân tử thay thế".

SỬA 10/11/2026 — ĐÍNH CHÍNH. Trước đây docstring này ghi rằng khóa mặc định "là đúng đơn
vị mà các chỉ số xếp hạng đang chấm trên". **Câu đó sai.** `train.evaluate` gom query SAU
khi gọi `metrics.expand_directions`, và hàm đó nhân đôi dữ liệu theo hai chiều, nên cột
phân tử tham chiếu lấy từ **cả hai phía** của cặp. Số query mà chỉ số xếp hạng thật sự
chấm trên, đo ngày 10/11/2026 trên `data/dataset_v4`:

| khóa | train | validation |
|---|---:|---:|
| `(protein_hash, smiles_a, endpoint)` — mặc định ở đây | 636.118 | 88.781 |
| cả hai phía — **đơn vị metric thật chấm** | **783.736** | **113.835** |

Con số đưa vào paper phải là 783.736 / 113.835. Khóa mặc định của script này vẫn giữ
nguyên vì nó là con số đã dùng trong mọi báo cáo trước, nhưng phải gọi đúng tên nó: đó là
số phân tử mốc PHÍA A, không phải số query được chấm. Và 669.035 của Tùng (07/11/2026)
không khớp khóa nào trong hai khóa trên, nên vẫn cần cậu ấy đưa khóa gom của mình.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "dataset_v4"
EVALUATOR_FILE = "evaluator_training.parquet"
DEFAULT_KEY = ("protein_hash", "smiles_a", "endpoint")
PARTITIONS = ("train", "validation", "test")
PROTEIN_COLUMN = "protein_hash"
# Các mức dùng trong bảng mất cân bằng; phần trăm protein ĐÔNG query nhất.
TOP_FRACTIONS = (0.01, 0.05, 0.50)


def parse_key(raw: str) -> list[str]:
    """Khóa gom nhóm từ chuỗi 'cot1,cot2'; rỗng thì báo lỗi chứ không đoán."""
    columns = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not columns:
        raise ValueError("--key cần ít nhất một tên cột")
    return columns


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Đọc đúng các cột cần từ parquet; thiếu cột nào thì báo tên cột đó."""
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = [name for name in columns if name not in available]
    if missing:
        raise ValueError(f"file {path.name} không có cột {missing}; "
                         f"cột đang có: {sorted(available)}")
    return pq.read_table(path, columns=columns).to_pandas()


def concentration(queries_per_protein: pd.Series) -> dict[str, dict[str, float]]:
    """Bao nhiêu phần trăm query rơi vào nhóm protein đông nhất, theo từng mức.

    Nhận: số query của mỗi protein (một dòng một protein).
    Trả về: {'top_1pct': {'proteins': k, 'queries': n, 'share': tỉ lệ}, ...}.
    """
    ordered = queries_per_protein.sort_values(ascending=False)
    total = int(ordered.sum())
    report: dict[str, dict[str, float]] = {}
    for fraction in TOP_FRACTIONS:
        count = max(1, int(round(len(ordered) * fraction)))
        taken = int(ordered.iloc[:count].sum())
        report[f"top_{fraction:.0%}".replace("%", "pct")] = {
            "proteins": count, "queries": taken,
            "share": 0.0 if total == 0 else taken / total}
    return report


def spread(rows_per_protein: pd.Series) -> dict[str, float]:
    """Phân phối số dòng trên mỗi protein: lớn nhất, trung vị, đuôi dưới."""
    values = rows_per_protein.to_numpy()
    return {"proteins": int(len(values)),
            "largest_protein_rows": int(values.max()),
            "median_rows": float(np.median(values)),
            "p10_rows": float(np.percentile(values, 10)),
            "proteins_with_one_row": int((values == 1).sum()),
            "share_with_one_row": float((values == 1).mean())}


def measure(frame: pd.DataFrame, key: list[str]) -> dict[str, Any]:
    """Mọi con số của MỘT phân vùng.

    Nhận: bảng của riêng phân vùng đó; khóa gom nhóm.
    Trả về: khối thống kê (dòng, query, protein, tập trung, phân phối).
    """
    query_codes = frame[key].astype(str).agg("\x1f".join, axis=1)
    queries = int(query_codes.nunique())
    rows = int(len(frame))
    per_protein = (pd.DataFrame({"protein": frame[PROTEIN_COLUMN], "query": query_codes})
                   .groupby("protein")["query"].nunique())
    return {"rows": rows, "queries": queries,
            "proteins": int(frame[PROTEIN_COLUMN].nunique()),
            "rows_per_query": 0.0 if queries == 0 else rows / queries,
            "concentration": concentration(per_protein),
            "spread": spread(frame.groupby(PROTEIN_COLUMN).size())}


def report(dataset: Path, key: list[str]) -> dict[str, Any]:
    """Thống kê cho cả ba phân vùng của một bộ dữ liệu."""
    path = dataset / EVALUATOR_FILE
    if not path.is_file():
        raise FileNotFoundError(f"không thấy {path}")
    columns = sorted({*key, PROTEIN_COLUMN, "partition"})
    frame = read_columns(path, columns)
    partitions = {}
    for name in PARTITIONS:
        block = frame[frame["partition"] == name]
        if len(block):
            partitions[name] = measure(block, key)
    return {"dataset": str(dataset), "file": EVALUATOR_FILE, "query_key": key,
            "total_rows": int(len(frame)), "partitions": partitions}


def render(result: dict[str, Any]) -> str:
    """Bảng chữ cho người đọc; mọi số đều kèm khóa gom đã dùng."""
    lines = [f"dữ liệu: {result['dataset']}/{result['file']}",
             f"khóa gom query: {' + '.join(result['query_key'])}",
             "",
             f"{'phân vùng':<12}{'dòng':>12}{'query':>11}{'protein':>9}"
             f"{'dòng/query':>12}"]
    lines.append("-" * 56)
    for name, block in result["partitions"].items():
        lines.append(f"{name:<12}{block['rows']:>12,}{block['queries']:>11,}"
                     f"{block['proteins']:>9,}{block['rows_per_query']:>12.2f}")
    lines.extend(["", "tập trung protein (phần trăm QUERY thuộc nhóm protein đông nhất)",
                  f"{'phân vùng':<12}{'top 1%':>9}{'top 5%':>9}{'top 50%':>9}"])
    lines.append("-" * 39)
    for name, block in result["partitions"].items():
        shares = [f"{block['concentration'][f'top_{int(f * 100)}pct']['share']:.0%}"
                  for f in TOP_FRACTIONS]
        lines.append(f"{name:<12}{shares[0]:>9}{shares[1]:>9}{shares[2]:>9}")
    lines.extend(["", "phân phối DÒNG trên mỗi protein",
                  f"{'phân vùng':<12}{'lớn nhất':>12}{'trung vị':>10}"
                  f"{'chỉ 1 dòng':>12}"])
    lines.append("-" * 46)
    for name, block in result["partitions"].items():
        item = block["spread"]
        lines.append(f"{name:<12}{item['largest_protein_rows']:>12,}"
                     f"{item['median_rows']:>10,.0f}"
                     f"{item['proteins_with_one_row']:>7,} "
                     f"({item['share_with_one_row']:.1%})")
    return "\n".join(lines)


def main() -> int:
    """Chạy đếm và in kết quả; trả về mã thoát (0 = xong)."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"thư mục bộ dữ liệu (mặc định {DEFAULT_DATASET.name})")
    parser.add_argument("--key", default=",".join(DEFAULT_KEY),
                        help="khóa gom nhóm một query, các cột cách nhau bằng dấu phẩy "
                             f"(mặc định {','.join(DEFAULT_KEY)})")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="in JSON thay cho bảng chữ")
    args = parser.parse_args()
    # Console Windows mặc định cp1252, không in được tiếng Việt khi hứng qua pipe.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        key = parse_key(args.key)
        result = report(args.dataset, key)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.as_json
          else render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
