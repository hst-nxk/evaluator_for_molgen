"""Việc 13 họp 09/11/2026: dựng cách chia NGƯỢC — trùng protein được, trùng cặp thì không.

    python scripts/build_pair_split.py --out work/splits/pair_split_1411.parquet

Cách chia hiện tại của dự án ép protein của validation chưa từng xuất hiện trong train
(kịch bản 1: protein hoàn toàn mới, zero-shot). Biên bản yêu cầu thử điều kiện ngược lại:

    "Cho phép TRÙNG protein P, nhưng cặp (A, B) ở validation KHÔNG được trùng train — đo
     khả năng tổng quát hóa trên cấu trúc hóa học."

Đó đúng là kịch bản 2 của việc 15 ("protein đã có dữ liệu lịch sử, dự đoán cho cặp chất
(A, B) mới"), nên một lần chạy trả lời cả hai việc.

THIẾT KẾ, và vì sao từng lựa chọn:

* Tập ứng viên là ĐÚNG các dòng của tập con 100k đang dùng (`--subset`), không phải toàn
  bộ 6,6 triệu dòng. Nhờ vậy protein, quy mô và thời gian chạy khớp với các thí nghiệm
  100k đã có, và khác biệt duy nhất giữa hai bên là CÁCH CHIA. Đổi cả quy mô lẫn cách chia
  cùng lúc thì không quy được chênh lệch cho cái nào.
* Đơn vị chia là CẶP PHÂN TỬ KHÔNG KỂ THỨ TỰ: {A, B} và {B, A} là một. Dataset lưu cặp
  theo một chiều, nhưng mô hình được đánh giá cả hai chiều (metrics.expand_both_directions),
  nên tách hai chiều về hai bên là rò rỉ thẳng.
* Mặc định cặp KHÔNG kèm endpoint: cùng {A, B} đo bằng Ki và bằng IC50 vẫn là một cặp hóa
  học, để chung một bên. `--pair-key molecules_endpoint` nới điều kiện đó ra nếu muốn đo
  riêng.
* Dòng validation nào có protein KHÔNG xuất hiện trong train thì bị loại (tắt bằng
  `--keep-unseen-proteins`). Không loại thì tập validation lẫn cả ca zero-shot, và đó lại
  chính là kịch bản 1 — hai kịch bản trộn vào nhau thì số đo không nói được điều gì.
* Cặp không được liệt kê trong bảng ra sẽ được Stage 3 đẩy sang 'test', tức loại hẳn,
  chứ không bị bỏ lại theo cách chia cũ (xem data.apply_pair_partition_override).

Script CHỈ ĐỌC dataset; nó chỉ ghi một file parquet ba cột (pair_id, partition,
unseen_pair) tại --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from molopt.evaluator import data as data_module  # noqa: E402

NEEDED = ["pair_id", "protein_hash", "endpoint", "smiles_a", "smiles_b"]


def read_candidates(dataset: Path, subset: Path | None) -> pa.Table:
    """Các dòng ứng viên: cả dataset, hoặc chỉ những dòng có pair_id trong bảng subset."""
    keep = None
    if subset is not None:
        if not subset.is_file():
            raise FileNotFoundError(f"chưa có bảng subset {subset}")
        keep = data_module._flat(pq.read_table(subset, columns=["pair_id"]).column("pair_id"))
    handle = pq.ParquetFile(dataset)
    chunks = []
    for batch in handle.iter_batches(batch_size=200_000, columns=NEEDED):
        table = pa.Table.from_batches([batch])
        if keep is not None:
            table = table.filter(pc.is_in(table.column("pair_id"), value_set=keep))
        if table.num_rows:
            chunks.append(table)
    if not chunks:
        raise ValueError("không còn dòng ứng viên nào")
    table = pa.concat_tables(chunks)
    if keep is not None and table.num_rows != len(keep):
        raise ValueError(f"bảng subset có {len(keep):,} pair_id nhưng khớp "
                         f"{table.num_rows:,} dòng của dataset; subset dựng từ lần "
                         f"export khác")
    return table


def pair_group_codes(table: pa.Table, with_endpoint: bool) -> tuple[np.ndarray, int]:
    """Mã nhóm cho từng dòng theo cặp phân tử KHÔNG kể thứ tự (tùy chọn kèm endpoint)."""
    first, second = table.column("smiles_a"), table.column("smiles_b")
    forward = pc.less_equal(first, second)
    low = pc.if_else(forward, first, second)
    high = pc.if_else(forward, second, first)
    key = pc.binary_join_element_wise(low, high, "\u0001")
    if with_endpoint:
        key = pc.binary_join_element_wise(key, table.column("endpoint"), "\u0001")
    encoded = pc.dictionary_encode(data_module._flat(key))
    return np.asarray(encoded.indices), len(encoded.dictionary)


def split_groups(codes: np.ndarray, total_groups: int, fraction: float,
                 seed: int) -> np.ndarray:
    """Chọn các NHÓM CẶP cho validation tới khi đủ ~fraction số dòng. Trả mảng bool theo dòng.

    Chia theo nhóm chứ không theo dòng: hai dòng cùng một cặp {A, B} (khác protein, khác
    endpoint) phải nằm cùng một bên, nếu không thì cặp đó vừa ở train vừa ở validation.
    """
    sizes = np.bincount(codes, minlength=total_groups)
    order = np.random.default_rng(seed).permutation(total_groups)
    target = fraction * len(codes)
    taken = np.cumsum(sizes[order])
    count = int(np.searchsorted(taken, target, side="left")) + 1
    count = max(1, min(count, total_groups - 1))
    chosen = np.zeros(total_groups, dtype=bool)
    chosen[order[:count]] = True
    return chosen[codes]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="",
                        help="mặc định stage3.inputs.evaluator của config.json")
    parser.add_argument("--subset", default="work/splits/train_subset_c200_r500.parquet",
                        help="chỉ lấy các dòng có pair_id trong bảng này; '' = cả dataset")
    parser.add_argument("--fraction", type=float, default=0.25,
                        help="tỉ lệ dòng mong muốn cho validation (mặc định 0,25)")
    parser.add_argument("--pair-key", default="molecules",
                        choices=("molecules", "molecules_endpoint"),
                        help="đơn vị 'một cặp'; mặc định không kể endpoint")
    parser.add_argument("--keep-unseen-proteins", action="store_true",
                        help="GIỮ dòng validation có protein chưa thấy trong train "
                             "(mặc định loại, để tập này đúng là kịch bản 2)")
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--out", required=True, help="file parquet sẽ ghi")
    args = parser.parse_args()

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    dataset = Path(args.dataset) if args.dataset else ROOT / config["stage3"]["inputs"]["evaluator"]
    subset = Path(args.subset) if args.subset else None
    table = read_candidates(dataset, subset)
    codes, groups = pair_group_codes(table, args.pair_key == "molecules_endpoint")
    is_validation = split_groups(codes, groups, args.fraction, args.seed)

    proteins = np.asarray(table.column("protein_hash").to_pylist())
    train_proteins = set(proteins[~is_validation].tolist())
    dropped = 0
    if not args.keep_unseen_proteins:
        orphan = is_validation & np.array([p not in train_proteins for p in proteins])
        dropped = int(orphan.sum())
        is_validation = is_validation & ~orphan
        kept = ~orphan
    else:
        kept = np.ones(len(codes), dtype=bool)

    pair_ids = np.asarray(table.column("pair_id").to_pylist())
    partition = np.where(is_validation, data_module.VALIDATION, data_module.TRAIN)
    out = pa.table({
        "pair_id": pa.array(pair_ids[kept]),
        "partition": pa.array(partition[kept]),
        # Theo đúng cách dựng: mọi cặp validation đều chưa từng ở train, nên lát `strict`
        # của Stage 3 trùng lát `all` trên tập này. Ghi ra tường minh để lát strict không
        # còn đọc cột unseen_pair của cách chia CŨ.
        "unseen_pair": pa.array(is_validation[kept]),
    })

    validation_codes = set(codes[is_validation].tolist())
    train_codes = set(codes[kept & ~is_validation].tolist())
    shared = validation_codes & train_codes
    if shared:
        raise AssertionError(f"{len(shared)} cặp nằm ở CẢ train lẫn validation — hỏng")
    validation_proteins = set(proteins[is_validation].tolist())
    train_proteins = set(proteins[kept & ~is_validation].tolist())
    both = validation_proteins & train_proteins

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, target)

    rows_train = int((kept & ~is_validation).sum())
    rows_validation = int(is_validation.sum())
    print(f"nguồn            : {dataset}")
    print(f"tập ứng viên     : {table.num_rows:,} dòng, {groups:,} cặp phân biệt"
          + (f" (theo {subset})" if subset else " (cả dataset)"))
    print(f"train            : {rows_train:,} dòng | {len(train_codes):,} cặp | "
          f"{len(train_proteins):,} protein")
    print(f"validation       : {rows_validation:,} dòng | {len(validation_codes):,} cặp | "
          f"{len(validation_proteins):,} protein")
    print(f"cặp trùng hai bên: {len(shared)}  (phải bằng 0 — đây là điều kiện của việc 13)")
    print(f"protein trùng    : {len(both):,} = {len(both) / max(1, len(validation_proteins)):.1%} "
          f"số protein của validation  (CÀNG CAO CÀNG ĐÚNG ý việc 13)")
    print(f"dòng bị loại     : {dropped:,} (validation có protein chưa thấy trong train)")
    print(f"\nđã ghi {target}")
    print("\nDùng: --set inputs.partition_override_pairs=" + str(target).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
