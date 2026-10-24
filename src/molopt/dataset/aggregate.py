"""Stage 2b, bước 1 và bước 4: gộp replicate trong assay, gộp Δp giữa các assay.

Bước 1 — mỗi nhóm (nguồn, assay, protein, endpoint, phân tử) thành một giá trị đại diện:
    các pActivity lệch nhau ≤ merge_tolerance được coi là một phép đo và gộp làm một,
    rồi lấy mean của các phép đo sau gộp.
Bước 4 — mỗi bộ (A, B, protein, endpoint) thành một bản ghi: các Δp (mỗi assay một Δp)
    lệch nhau ≤ ngưỡng gộp được coi là MỘT giá trị (bản sao giữa các assay), rồi tính mean,
    median, STD, min, max, độ phân tán và cờ QC trên các giá trị khác nhau; kèm số tài liệu
    độc lập của bản ghi.

Mọi hàm ở đây là logic thuần trên bảng Arrow / mảng numpy, không đọc hay ghi file.
Endpoint không nằm trong khóa nhóm vì pipeline luôn xử lý từng endpoint riêng.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
QC_IDENTICAL = "identical_values"
QC_STATUSES = (QC_PASS, QC_SINGLE, QC_IDENTICAL, QC_QUARANTINE)

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
    """Độ phân tán của các giá trị Δp KHÁC NHAU của mỗi bản ghi (NaN khi chỉ có 1).

    Nhận: số giá trị khác nhau (sau khi gộp bản sao), STD, min, max của chúng.
    - ≥ 3 giá trị: STD;
    - đúng 2 giá trị: |Δ_X − Δ_Y| = max − min (theo đề bài; lớn hơn STD ddof = 1 của hai
      giá trị đúng √2 lần, tức chặt hơn khi so cùng ngưỡng);
    - 1 giá trị: không tính được.
    """
    result = np.where(sizes >= 3, std, maximum - minimum)
    return np.where(sizes >= 2, result, np.nan)


def qc_status(n_assays: np.ndarray, n_distinct: np.ndarray, spread: np.ndarray,
              threshold: float) -> np.ndarray:
    """Cờ QC của từng bản ghi.

    - single_assay: chỉ 1 assay;
    - identical_values: ≥ 2 assay nhưng mọi Δp trùng nhau (lệch ≤ ngưỡng gộp), tức chỉ có
      MỘT giá trị — đo 24/10/2026, phần lớn trường hợp này là số liệu chép lại giữa các
      tài liệu (docs/AUDIT_20261024.md mục 4), nên không được coi là "được xác nhận";
    - quarantine: ≥ 2 giá trị khác nhau và độ phân tán > ngưỡng;
    - pass: ≥ 2 giá trị khác nhau và độ phân tán ≤ ngưỡng.
    """
    status = np.full(n_assays.size, QC_PASS, dtype=object)
    status[(n_assays >= 2) & (n_distinct == 1)] = QC_IDENTICAL
    status[n_assays == 1] = QC_SINGLE
    with np.errstate(invalid="ignore"):
        status[(n_distinct >= 2) & (spread > threshold)] = QC_QUARANTINE
    return status


def collapse_values(values: np.ndarray, tolerance: float, slack: float) -> np.ndarray:
    """Các giá trị Δp khác nhau của MỘT bản ghi: giá trị lệch ≤ tolerance gộp làm một.

    Nhận: Δp của các assay (thứ tự bất kỳ); ngưỡng gộp; sai số dấu phẩy động.
    Trả về: mean của từng cụm, tăng dần.
    Dùng đúng quy tắc neo của bước 1 (merge_values, Quyết định 1 của Stage 2a): một giá trị
    bị chép sang assay khác không được tính thành bằng chứng thứ hai, và không kéo mean về
    phía nó.
    """
    ordered = np.sort(values)
    merged, _ = merge_values(ordered, np.zeros(ordered.size, dtype=bool), tolerance, slack)
    return merged


def summarize_values(values: np.ndarray, tolerance: float, slack: float,
                     ddof: int) -> tuple[float, float, float, float, float, float, int]:
    """(mean, median, STD, min, max, phân tán, số giá trị khác nhau) của một bản ghi.

    Mọi thống kê tính trên các giá trị đã gộp bản sao (collapse_values). STD là NaN khi số
    giá trị khác nhau ≤ ddof. Dùng chung cho bước 4 và cho verify.
    """
    distinct = collapse_values(values, tolerance, slack)
    size = distinct.size
    std = float(np.std(distinct, ddof=ddof)) if size > ddof else float("nan")
    minimum, maximum = float(distinct[0]), float(distinct[-1])
    spread = float(dispersion(np.array([size]), np.array([std]), np.array([minimum]),
                              np.array([maximum]))[0])
    return (float(distinct.mean()), float(np.median(distinct)), std, minimum, maximum,
            spread, size)


def document_count(assay_keys: Sequence[str],
                   documents: Mapping[str, frozenset[str]]) -> int:
    """Số tài liệu độc lập của một bản ghi.

    Nhận: assay_key của các assay đóng góp; assay_key -> tập định danh tài liệu (PMID, DOI,
    patent đã chuẩn hóa, hoặc mã tài liệu của nguồn khi không có ba thứ đó).
    Trả về: số thành phần liên thông khi nối hai assay có chung ít nhất một định danh — vì
    cùng một bài có thể mang PMID ở nguồn này và chỉ DOI ở nguồn kia.
    assay_key thiếu trong bảng tài liệu là lỗi: mọi assay của dataset đều đến từ đầu vào.
    """
    sets = []
    for key in assay_keys:
        if key not in documents:
            raise KeyError(f"assay {key!r} không có trong bảng tài liệu")
        sets.append(documents[key])
    parent = list(range(len(sets)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(len(sets)):
        for second in range(first + 1, len(sets)):
            if sets[first] & sets[second]:
                parent[root(first)] = root(second)
    return len({root(index) for index in range(len(sets))})


def group_max(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Max bỏ NaN của từng nhóm liên tiếp; nhóm toàn NaN -> NaN."""
    starts = bounds[:-1]
    filled = np.where(np.isnan(values), -np.inf, values)
    result = np.maximum.reduceat(filled, starts) if starts.size else np.empty(0)
    return np.where(np.isinf(result), np.nan, result)


def aggregate_deltas(deltas: pa.Table, threshold: float, ddof: int, tolerance: float,
                     slack: float, documents: Mapping[str, frozenset[str]]) -> pa.Table:
    """Bước 4: gộp Δp của mọi assay cho từng (A, B, protein) của một endpoint.

    Nhận: bảng Δp trong assay (một dòng mỗi (assay, A, B), có pair_id, delta_p, p_a,
    p_std_a, p_std_b, assay_key, independent và các cột trong RECORD_CARRIED); ngưỡng phân
    tán của endpoint; ddof của STD; ngưỡng gộp bản sao và sai số; bảng tài liệu của assay.
    Trả về: một dòng mỗi bản ghi: khóa, các cột mang theo, n_assays, n_independent_assays,
    n_distinct_values, n_documents, delta_p_mean / median / std / min / max và dispersion
    (tính trên các giá trị ĐÃ GỘP bản sao), qc_status, p_a_range (max − min của p(A) giữa
    các assay: Δp trùng mà p(A) lệch đúng 3 hay 6 log là dấu hiệu bản sao sai đơn vị),
    max_replicate_std (STD replicate lớn nhất trong cùng assay của hai phía, null nếu không
    có), và ba cột danh sách nguyên văn theo thứ tự assay_key.
    Không trộn endpoint (mỗi lần gọi một endpoint) hay protein (protein_hash trong khóa).
    """
    order = [(key, "ascending") for key in RECORD_KEYS] + [("assay_key", "ascending")]
    ordered = deltas.sort_by(order)
    bounds = group_bounds(ordered, RECORD_KEYS)
    starts, sizes = bounds[:-1], np.diff(bounds)
    if starts.size == 0:
        return empty_records()
    values = np.asarray(ordered["delta_p"].to_numpy(zero_copy_only=False), dtype=np.float64)
    independent = np.asarray(ordered["independent"].to_numpy(zero_copy_only=False), dtype=bool)
    p_a = np.asarray(ordered["p_a"].to_numpy(zero_copy_only=False), dtype=np.float64)
    replicate_std = np.fmax(
        np.asarray(ordered["p_std_a"].to_numpy(zero_copy_only=False), dtype=np.float64),
        np.asarray(ordered["p_std_b"].to_numpy(zero_copy_only=False), dtype=np.float64))
    assay_keys = ordered["assay_key"].to_pylist()

    # Bản ghi một assay (đa số): tính thẳng bằng numpy.
    mean = values[starts].copy()
    median = mean.copy()
    minimum, maximum = mean.copy(), mean.copy()
    std = np.full(starts.size, np.nan)
    spread = np.full(starts.size, np.nan)
    n_distinct = np.ones(starts.size, dtype=np.int64)
    n_documents = np.ones(starts.size, dtype=np.int64)
    missing = set(assay_keys).difference(documents)
    if missing:
        raise KeyError(f"assay không có trong bảng tài liệu: {sorted(missing)[:3]}")
    # Bản ghi nhiều assay: gộp bản sao rồi mới tính thống kê.
    for group in np.flatnonzero(sizes > 1):
        first, last = int(bounds[group]), int(bounds[group + 1])
        (mean[group], median[group], std[group], minimum[group], maximum[group],
         spread[group], n_distinct[group]) = summarize_values(values[first:last], tolerance,
                                                              slack, ddof)
        n_documents[group] = document_count(assay_keys[first:last], documents)
    p_a_range = np.maximum.reduceat(p_a, starts) - np.minimum.reduceat(p_a, starts)

    take = pa.array(starts)
    columns = {key: ordered[key].take(take) for key in RECORD_KEYS + RECORD_CARRIED}
    columns.update({
        "n_assays": pa.array(sizes, type=pa.int64()),
        "n_independent_assays": pa.array(np.add.reduceat(independent.astype(np.int64), starts),
                                         type=pa.int64()),
        "n_distinct_values": pa.array(n_distinct, type=pa.int64()),
        "n_documents": pa.array(n_documents, type=pa.int64()),
        "delta_p_mean": pa.array(mean, type=pa.float64()),
        "delta_p_median": pa.array(median, type=pa.float64()),
        "delta_p_std": nullable(std),
        "delta_p_min": pa.array(minimum, type=pa.float64()),
        "delta_p_max": pa.array(maximum, type=pa.float64()),
        "dispersion": nullable(spread),
        "qc_status": pa.array(qc_status(sizes, n_distinct, spread, threshold).tolist(),
                              type=pa.string()),
        "p_a_range": pa.array(p_a_range, type=pa.float64()),
        "max_replicate_std": nullable(group_max(replicate_std, bounds)),
        "delta_p_values": list_column(bounds, ordered["delta_p"]),
        "assay_keys": list_column(bounds, ordered["assay_key"]),
        "assay_independent": list_column(bounds, ordered["independent"]),
    })
    return pa.table(columns).select(RECORD_SCHEMA.names)


RECORD_SCHEMA = pa.schema(
    [(key, pa.string()) for key in RECORD_KEYS + RECORD_CARRIED]
    + [("n_assays", pa.int64()), ("n_independent_assays", pa.int64()),
       ("n_distinct_values", pa.int64()), ("n_documents", pa.int64()),
       ("delta_p_mean", pa.float64()), ("delta_p_median", pa.float64()),
       ("delta_p_std", pa.float64()), ("delta_p_min", pa.float64()),
       ("delta_p_max", pa.float64()), ("dispersion", pa.float64()),
       ("qc_status", pa.string()), ("p_a_range", pa.float64()),
       ("max_replicate_std", pa.float64()), ("delta_p_values", pa.list_(pa.float64())),
       ("assay_keys", pa.list_(pa.string())), ("assay_independent", pa.list_(pa.bool_()))])


def empty_records() -> pa.Table:
    """Bảng bản ghi rỗng đúng schema."""
    return RECORD_SCHEMA.empty_table()
