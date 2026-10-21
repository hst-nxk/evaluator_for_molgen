"""Stage 2b, bước 3: Δp trong từng assay.

Δp = p(B) − p(A), dùng giá trị đại diện của bước 1, tính riêng trong từng assay, giữ dấu.
Chiều của cặp cố định: A là phân tử có khóa định danh nhỏ hơn theo thứ tự chuỗi. Vì sao
cố định theo khóa: cùng một cặp ở mọi assay luôn có cùng chiều, nên bước 4 gộp được mà
không phải đảo dấu, và không bao giờ lưu cả A→B lẫn B→A.

pair_id: SHA-256 của 'endpoint|protein|A|B', lấy pair_id_length ký tự hex đầu. Ổn định
giữa các lần chạy và là khóa nối phần training với phần evidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pyarrow as pa

from molopt.dataset.mmp import Pair

ASSAY_DELTA_SCHEMA = pa.schema([
    ("pair_id", pa.string()),
    ("endpoint", pa.string()),
    ("protein_hash", pa.string()),
    ("assay_key", pa.string()),
    ("source", pa.string()),
    ("molecule_a_key", pa.string()),
    ("molecule_b_key", pa.string()),
    ("smiles_a", pa.string()),
    ("smiles_b", pa.string()),
    ("constant_smiles", pa.string()),
    ("variable_a", pa.string()),
    ("variable_b", pa.string()),
    ("p_a", pa.float64()),
    ("p_b", pa.float64()),
    ("delta_p", pa.float64()),
    ("n_merged_a", pa.int64()),
    ("n_merged_b", pa.int64()),
    ("independent", pa.bool_()),
    ("multi_source_id", pa.bool_()),
    ("observation_ids_a", pa.list_(pa.string())),
    ("observation_ids_b", pa.list_(pa.string())),
])

LOST_PAIR_SCHEMA = pa.schema([
    ("endpoint", pa.string()),
    ("protein_hash", pa.string()),
    ("assay_key", pa.string()),
    ("molecule_a_key", pa.string()),
    ("molecule_b_key", pa.string()),
])


def pair_id(endpoint: str, protein_hash: str, molecule_a: str, molecule_b: str,
            length: int) -> str:
    """ID ổn định của bản ghi (A, B, protein, endpoint)."""
    text = f"{endpoint}|{protein_hash}|{molecule_a}|{molecule_b}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def orient(key_x: str, key_y: str) -> bool:
    """True khi phải đổi chỗ (x, y) để A là khóa nhỏ hơn."""
    return key_x > key_y


def assay_deltas(groups: pa.Table, pairs: Sequence[Pair], endpoint: str,
                 id_length: int) -> pa.Table:
    """Bảng Δp trong assay cho các cặp tìm được ở bước 2.

    Nhận: bảng bước 1 của một endpoint (đã thêm cột smiles); các cặp với row_x, row_y là
    chỉ số dòng của bảng đó (hai dòng cùng assay và protein); endpoint; độ dài pair_id.
    Trả về: bảng theo ASSAY_DELTA_SCHEMA. 'independent' = cả hai phía có ít nhất một phép
    đo không mang cờ potential_duplicate. 'multi_source_id' = ít nhất một phía là nhóm có
    nhiều molecule_source_id (cờ của bước 1, chỉ để báo cáo và tra lại, không lọc).
    """
    if not pairs:
        return ASSAY_DELTA_SCHEMA.empty_table()
    keys = groups["molecule_key"].to_pylist()
    rows_a, rows_b, variables_a, variables_b = [], [], [], []
    for pair in pairs:
        swap = orient(keys[pair.row_x], keys[pair.row_y])
        row_a, row_b = (pair.row_y, pair.row_x) if swap else (pair.row_x, pair.row_y)
        variable_a, variable_b = ((pair.variable_y, pair.variable_x) if swap
                                  else (pair.variable_x, pair.variable_y))
        rows_a.append(row_a)
        rows_b.append(row_b)
        variables_a.append(variable_a)
        variables_b.append(variable_b)
    take_a, take_b = pa.array(rows_a, pa.int64()), pa.array(rows_b, pa.int64())
    side_a, side_b = groups.take(take_a), groups.take(take_b)
    protein = side_a["protein_hash"].to_pylist()
    key_a, key_b = side_a["molecule_key"].to_pylist(), side_b["molecule_key"].to_pylist()
    p_a = np.asarray(side_a["p_mean"].to_numpy(zero_copy_only=False), dtype=np.float64)
    p_b = np.asarray(side_b["p_mean"].to_numpy(zero_copy_only=False), dtype=np.float64)
    independent_a = np.asarray(side_a["n_independent"].to_numpy(zero_copy_only=False)) > 0
    independent_b = np.asarray(side_b["n_independent"].to_numpy(zero_copy_only=False)) > 0
    multi_a = np.asarray(side_a["multi_source_id"].to_numpy(zero_copy_only=False), dtype=bool)
    multi_b = np.asarray(side_b["multi_source_id"].to_numpy(zero_copy_only=False), dtype=bool)
    ids = [pair_id(endpoint, protein[i], key_a[i], key_b[i], id_length)
           for i in range(len(pairs))]
    return pa.table({
        "pair_id": pa.array(ids, pa.string()),
        "endpoint": pa.array([endpoint] * len(pairs), pa.string()),
        "protein_hash": side_a["protein_hash"],
        "assay_key": side_a["assay_key"],
        "source": side_a["source"],
        "molecule_a_key": side_a["molecule_key"],
        "molecule_b_key": side_b["molecule_key"],
        "smiles_a": side_a["smiles"],
        "smiles_b": side_b["smiles"],
        "constant_smiles": pa.array([pair.constant for pair in pairs], pa.string()),
        "variable_a": pa.array(variables_a, pa.string()),
        "variable_b": pa.array(variables_b, pa.string()),
        "p_a": pa.array(p_a, pa.float64()),
        "p_b": pa.array(p_b, pa.float64()),
        "delta_p": pa.array(p_b - p_a, pa.float64()),
        "n_merged_a": side_a["n_merged"],
        "n_merged_b": side_b["n_merged"],
        "independent": pa.array(independent_a & independent_b, pa.bool_()),
        "multi_source_id": pa.array(multi_a | multi_b, pa.bool_()),
        "observation_ids_a": side_a["observation_ids"],
        "observation_ids_b": side_b["observation_ids"],
    }, schema=ASSAY_DELTA_SCHEMA)


def lost_pairs_table(groups: pa.Table, lost: Sequence[tuple[int, int]],
                     endpoint: str) -> pa.Table:
    """Cặp bị loại vì quy tắc |r| (một bên có cờ, bên kia không), chiều A < B theo khóa."""
    if not lost:
        return LOST_PAIR_SCHEMA.empty_table()
    keys = groups["molecule_key"].to_pylist()
    rows_a = [row_y if orient(keys[row_x], keys[row_y]) else row_x for row_x, row_y in lost]
    rows_b = [row_x if orient(keys[row_x], keys[row_y]) else row_y for row_x, row_y in lost]
    side_a = groups.take(pa.array(rows_a, pa.int64()))
    side_b = groups.take(pa.array(rows_b, pa.int64()))
    return pa.table({
        "endpoint": pa.array([endpoint] * len(lost), pa.string()),
        "protein_hash": side_a["protein_hash"],
        "assay_key": side_a["assay_key"],
        "molecule_a_key": side_a["molecule_key"],
        "molecule_b_key": side_b["molecule_key"],
    }, schema=LOST_PAIR_SCHEMA)
