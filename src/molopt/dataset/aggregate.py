"""Stage 2b, bước 1 và bước 4: gộp replicate trong assay, gộp Δp giữa các assay.

Bước 1 — mỗi nhóm (nguồn, assay, protein, endpoint, phân tử) thành một giá trị đại diện:
    các pActivity lệch nhau ≤ merge_tolerance được coi là một phép đo và gộp làm một,
    rồi lấy mean của các phép đo sau gộp.
Bước 4 — mỗi bộ (A, B, protein, endpoint) thành một bản ghi:
    mean, median, STD, min, max của các Δp (mỗi assay một Δp), độ phân tán và cờ QC.

Mọi hàm ở đây là logic thuần trên bảng Arrow / mảng numpy, không đọc hay ghi file.
Endpoint không nằm trong khóa nhóm vì pipeline luôn xử lý từng endpoint riêng.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# Khóa nhóm của bước 1 (endpoint cố định trong một lần gọi). assay_key đã chứa nguồn.
REPLICATE_KEYS = ["assay_key", "protein_hash", "molecule_key"]
# Khóa bản ghi của bước 4 (endpoint cố định trong một lần gọi).
RECORD_KEYS = ["protein_hash", "molecule_a_key", "molecule_b_key"]

QC_PASS = "pass"
QC_SINGLE = "single_assay"
QC_QUARANTINE = "quarantine"
QC_STATUSES = (QC_PASS, QC_SINGLE, QC_QUARANTINE)

# Cột lấy nguyên từ dòng đầu của mỗi bản ghi ở bước 4: mọi assay của cùng một cặp (A, B)
# cho cùng giá trị, vì SMILES đại diện và cách cắt chỉ phụ thuộc A và B (xem mmp.py).
RECORD_CARRIED = ["pair_id", "smiles_a", "smiles_b", "constant_smiles", "variable_a",
                  "variable_b"]


# --------------------------------------------------------------------------- tiện ích chung


def group_bounds(table: pa.Table, keys: Sequence[str]) -> np.ndarray:
    """Ranh giới các nhóm liên tiếp của một bảng ĐÃ SẮP XẾP theo `keys`.

    Nhận: bảng đã sort theo keys; danh sách cột khóa.
    Trả về: mảng int64 [s0, s1, ..., n]: nhóm i là các dòng s_i .. s_{i+1} - 1.
    Bảng rỗng -> [0]. Null trong khóa được coi như một giá trị (null == null).
    """
    rows = table.num_rows
    if rows == 0:
        return np.zeros(1, dtype=np.int64)
    changed = np.zeros(rows - 1, dtype=bool)
    for key in keys:
        column = table[key]
        later, earlier = column.slice(1), column.slice(0, rows - 1)
        differs = pc.fill_null(pc.not_equal(later, earlier), True)
        # Hai null liên tiếp: not_equal cho null -> đã thành True ở trên; sửa lại thành False.
        both_null = pc.and_(pc.is_null(later), pc.is_null(earlier))
        differs = pc.and_(differs, pc.invert(both_null))
        changed |= np.asarray(differs.to_numpy(zero_copy_only=False), dtype=bool)
    starts = np.flatnonzero(changed) + 1
    return np.concatenate([[0], starts, [rows]]).astype(np.int64)


def nullable(values: np.ndarray) -> pa.Array:
    """Mảng float numpy -> mảng Arrow, NaN thành null."""
    return pa.array(values, type=pa.float64(), from_pandas=True)


def list_column(bounds: np.ndarray, values: pa.ChunkedArray | pa.Array) -> pa.ListArray:
    """Cột danh sách: nhóm i chứa values[bounds[i]:bounds[i+1]]."""
    flat = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    return pa.ListArray.from_arrays(pa.array(bounds, type=pa.int64()).cast(pa.int32()), flat)


def integer_codes(values: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """Mã nguyên của từng giá trị: giá trị bằng nhau có mã bằng nhau, null -> −1.

    Dùng để đếm số giá trị khác nhau trong mỗi nhóm bằng numpy (nhanh hơn so chuỗi).
    """
    flat = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    if flat.null_count == len(flat):
        return np.full(len(flat), -1, dtype=np.int64)
    encoded = flat.dictionary_encode()
    return np.asarray(pc.fill_null(encoded.indices, -1).to_numpy(zero_copy_only=False),
                      dtype=np.int64)


# --------------------------------------------------------------------------- bước 1


def merge_values(values: np.ndarray, independent: np.ndarray, tolerance: float,
                 slack: float) -> tuple[np.ndarray, np.ndarray]:
    """Gộp các giá trị gần nhau của MỘT nhóm thành các phép đo.

    Nhận: pActivity đã sắp tăng dần; cờ "độc lập" của từng dòng (không mang cờ
    potential_duplicate); ngưỡng gộp; sai số dấu phẩy động cho phép.
    Trả về: (giá trị mỗi phép đo sau gộp = mean các dòng trong cụm, cụm có ít nhất một
    dòng độc lập không).
    Cách gộp: cụm bắt đầu ở giá trị nhỏ nhất chưa xếp, nhận mọi giá trị ≤ giá trị đầu cụm
    + tolerance. Vì sao neo vào giá trị đầu cụm chứ không nối chuỗi (6,000; 6,008; 6,016):
    nối chuỗi cho cụm rộng tùy ý, còn neo bảo đảm mọi giá trị trong một cụm lệch nhau
    ≤ tolerance, đúng định nghĩa "lệch nhau ≤ 0,01 thì là một phép đo".
    Ví dụ: [6,000; 6,005; 6,02], tolerance 0,01 -> ([6,0025; 6,02], ...).
    """
    means: list[float] = []
    flags: list[bool] = []
    start = 0
    size = values.size
    for index in range(1, size + 1):
        if index == size or values[index] - values[start] > tolerance + slack:
            means.append(float(values[start:index].mean()))
            flags.append(bool(independent[start:index].any()))
            start = index
    return np.asarray(means, dtype=np.float64), np.asarray(flags, dtype=bool)


def collapse_replicates(rows: pa.Table, tolerance: float, slack: float,
                        ddof: int) -> pa.Table:
    """Bước 1: mỗi nhóm (assay, protein, phân tử) của một endpoint thành một dòng.

    Nhận: bảng có assay_key, source, protein_hash, molecule_key, observation_id,
    p_activity, independent (bool), source_id (ID phân tử theo nguồn); ngưỡng gộp; sai
    số; ddof của STD.
    Trả về: bảng assay_key, source, protein_hash, molecule_key, p_mean, p_std, n_merged,
    n_independent, n_raw, n_source_ids, multi_source_id, observation_ids.
    - p_mean: mean các phép đo sau gộp (mọi phép đo, kể cả phép đo chỉ gồm dòng có cờ
      potential_duplicate: dòng có cờ được giữ);
    - p_std: STD các phép đo sau gộp, null khi chỉ có một phép đo;
    - n_merged: số phép đo sau gộp; n_independent: số phép đo sau gộp có ít nhất một dòng
      không mang cờ potential_duplicate (dòng có cờ không được tính thêm); n_raw: số dòng;
    - n_source_ids: số molecule_source_id khác nhau trong nhóm (null tính là một giá trị);
      multi_source_id: n_source_ids > 1, tức cùng InChIKey nhưng nguồn ghi nhiều ID phân
      tử khác nhau. Nhóm như vậy được GIỮ và chỉ gắn cờ (quyết định 2, mục 5.6 của audit);
    - observation_ids: mọi dòng góp vào, theo thứ tự pActivity tăng dần.
    """
    order = [(key, "ascending") for key in REPLICATE_KEYS]
    ordered = rows.sort_by(order + [("p_activity", "ascending"),
                                    ("observation_id", "ascending")])
    bounds = group_bounds(ordered, REPLICATE_KEYS)
    starts, sizes = bounds[:-1], np.diff(bounds)
    values = np.asarray(ordered["p_activity"].to_numpy(zero_copy_only=False), dtype=np.float64)
    independent = np.asarray(ordered["independent"].to_numpy(zero_copy_only=False), dtype=bool)
    source_codes = integer_codes(ordered["source_id"])

    # Nhóm một dòng (đa số): tính thẳng bằng numpy.
    p_mean = values[starts].copy()
    p_std = np.full(starts.size, np.nan)
    n_merged = np.ones(starts.size, dtype=np.int64)
    n_independent = independent[starts].astype(np.int64)
    n_source_ids = np.ones(starts.size, dtype=np.int64)
    # Nhóm nhiều dòng: gộp theo merge_values.
    for group in np.flatnonzero(sizes > 1):
        first, last = bounds[group], bounds[group + 1]
        merged, flags = merge_values(values[first:last], independent[first:last],
                                     tolerance, slack)
        p_mean[group] = merged.mean()
        n_merged[group] = merged.size
        n_independent[group] = int(flags.sum())
        n_source_ids[group] = np.unique(source_codes[first:last]).size
        if merged.size > ddof:
            p_std[group] = float(np.std(merged, ddof=ddof))

    take = pa.array(starts)
    return pa.table({
        "assay_key": ordered["assay_key"].take(take),
        "source": ordered["source"].take(take),
        "protein_hash": ordered["protein_hash"].take(take),
        "molecule_key": ordered["molecule_key"].take(take),
        "p_mean": pa.array(p_mean, type=pa.float64()),
        "p_std": nullable(p_std),
        "n_merged": pa.array(n_merged, type=pa.int64()),
        "n_independent": pa.array(n_independent, type=pa.int64()),
        "n_raw": pa.array(sizes, type=pa.int64()),
        "n_source_ids": pa.array(n_source_ids, type=pa.int64()),
        "multi_source_id": pa.array(n_source_ids > 1, type=pa.bool_()),
        "observation_ids": list_column(bounds, ordered["observation_id"]),
    })


# --------------------------------------------------------------------------- bước 4


def dispersion(sizes: np.ndarray, std: np.ndarray, minimum: np.ndarray,
               maximum: np.ndarray) -> np.ndarray:
    """Độ phân tán của các Δp mỗi bản ghi (NaN khi chỉ có 1 assay).

    - ≥ 3 assay: STD;
    - đúng 2 assay: |Δ_X − Δ_Y| = max − min (theo đề bài; lớn hơn STD ddof = 1 của hai
      giá trị đúng √2 lần, tức chặt hơn khi so cùng ngưỡng);
    - 1 assay: không tính được.
    """
    result = np.where(sizes >= 3, std, maximum - minimum)
    return np.where(sizes >= 2, result, np.nan)


def qc_status(sizes: np.ndarray, spread: np.ndarray, threshold: float) -> np.ndarray:
    """Cờ QC của từng bản ghi: single_assay, quarantine (phân tán > ngưỡng) hoặc pass."""
    status = np.full(sizes.size, QC_PASS, dtype=object)
    status[sizes == 1] = QC_SINGLE
    with np.errstate(invalid="ignore"):
        status[(sizes >= 2) & (spread > threshold)] = QC_QUARANTINE
    return status


def group_median(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Median chính xác của từng nhóm liên tiếp (giá trị trong nhóm chưa cần sắp)."""
    starts, sizes = bounds[:-1], np.diff(bounds)
    labels = np.repeat(np.arange(starts.size), sizes)
    ordered = values[np.lexsort((values, labels))]
    low = ordered[starts + (sizes - 1) // 2]
    high = ordered[starts + sizes // 2]
    return (low + high) / 2.0


def aggregate_deltas(deltas: pa.Table, threshold: float, ddof: int) -> pa.Table:
    """Bước 4: gộp Δp của mọi assay cho từng (A, B, protein) của một endpoint.

    Nhận: bảng Δp trong assay (một dòng mỗi (assay, A, B), đã có pair_id, delta_p,
    assay_key, independent và các cột trong RECORD_CARRIED); ngưỡng phân tán của endpoint;
    ddof của STD.
    Trả về: một dòng mỗi bản ghi: khóa, các cột mang theo, n_assays, n_independent_assays,
    delta_p_mean, delta_p_median, delta_p_std, delta_p_min, delta_p_max, dispersion,
    qc_status, delta_p_values, assay_keys, assay_independent (ba cột danh sách theo thứ tự
    assay_key).
    Không trộn endpoint (mỗi lần gọi một endpoint) hay protein (protein_hash trong khóa).
    Mỗi assay góp đúng một Δp vì bảng đầu vào có một dòng mỗi (assay, A, B); verify kiểm
    tra lại điều này.
    """
    order = [(key, "ascending") for key in RECORD_KEYS] + [("assay_key", "ascending")]
    ordered = deltas.sort_by(order)
    bounds = group_bounds(ordered, RECORD_KEYS)
    starts, sizes = bounds[:-1], np.diff(bounds)
    if starts.size == 0:
        return empty_records()
    values = np.asarray(ordered["delta_p"].to_numpy(zero_copy_only=False), dtype=np.float64)
    independent = np.asarray(ordered["independent"].to_numpy(zero_copy_only=False), dtype=bool)

    mean = np.add.reduceat(values, starts) / sizes
    minimum = np.minimum.reduceat(values, starts)
    maximum = np.maximum.reduceat(values, starts)
    squares = np.add.reduceat((values - np.repeat(mean, sizes)) ** 2, starts)
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.where(sizes > ddof, np.sqrt(squares / np.maximum(sizes - ddof, 1)), np.nan)
    spread = dispersion(sizes, std, minimum, maximum)

    take = pa.array(starts)
    columns = {key: ordered[key].take(take) for key in RECORD_KEYS + RECORD_CARRIED}
    columns.update({
        "n_assays": pa.array(sizes, type=pa.int64()),
        "n_independent_assays": pa.array(np.add.reduceat(independent.astype(np.int64), starts),
                                         type=pa.int64()),
        "delta_p_mean": pa.array(mean, type=pa.float64()),
        "delta_p_median": pa.array(group_median(values, bounds), type=pa.float64()),
        "delta_p_std": nullable(std),
        "delta_p_min": pa.array(minimum, type=pa.float64()),
        "delta_p_max": pa.array(maximum, type=pa.float64()),
        "dispersion": nullable(spread),
        "qc_status": pa.array(qc_status(sizes, spread, threshold).tolist(), type=pa.string()),
        "delta_p_values": list_column(bounds, ordered["delta_p"]),
        "assay_keys": list_column(bounds, ordered["assay_key"]),
        "assay_independent": list_column(bounds, ordered["independent"]),
    })
    return pa.table(columns)


RECORD_SCHEMA = pa.schema(
    [(key, pa.string()) for key in RECORD_KEYS + RECORD_CARRIED]
    + [("n_assays", pa.int64()), ("n_independent_assays", pa.int64()),
       ("delta_p_mean", pa.float64()), ("delta_p_median", pa.float64()),
       ("delta_p_std", pa.float64()), ("delta_p_min", pa.float64()),
       ("delta_p_max", pa.float64()), ("dispersion", pa.float64()),
       ("qc_status", pa.string()), ("delta_p_values", pa.list_(pa.float64())),
       ("assay_keys", pa.list_(pa.string())), ("assay_independent", pa.list_(pa.bool_()))])


def empty_records() -> pa.Table:
    """Bảng bản ghi rỗng đúng schema."""
    return RECORD_SCHEMA.empty_table()
