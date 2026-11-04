"""Stage 3: phân tích ca sai của một checkpoint (việc 6, họp 02/11/2026).

Câu hỏi cần trả lời: mô hình sai ở ĐÂU. Train chỉ đạt khoảng 0,75 đúng chiều và valid
0,56, nên trước khi đổi kiến trúc cần biết lỗi dồn vào nhóm nào:

    độ lớn nhãn   : |Δp| nhỏ nằm trong sai số thí nghiệm, sai ở đó là chuyện bình thường
    độ nhiễu nhãn : ít assay, ít tài liệu, độ lệch lặp lại lớn, trộn loại assay
    độ hiếm       : protein, phân tử, phép sửa ít gặp trong partition
    cỡ query      : query ít ứng viên có tỉ lệ đúng chiều dao động mạnh
    độ co dự đoán : mô hình đoán quanh 0 (underfit) hay đoán đủ rộng

Mọi hàm ở đây làm việc trên mảng numpy, không import torch, nên test được trên CPU. Việc
nạp checkpoint và chạy dự đoán nằm ở train.run_errors.

Hai cách đếm "đúng" được dùng, cố ý tách rời:
    đúng dấu (theo dòng) : sign(dự đoán) = sign(Δp thật), CHỈ trên dòng có |Δp thật| lớn
                           hơn ngưỡng (metrics.direction_tolerance); rẻ, chia nhóm được
                           theo bất kỳ cột nào của dòng.
    đúng chiều (theo query): đúng như chỉ số chính của dự án (metrics.ranking_metrics),
                           tính lại cho từng query để chia nhóm theo query và protein.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import metrics

MISSING_LABEL = "thiếu"


# --------------------------------------------------------------------------- tần suất


def code_frequency(codes: np.ndarray) -> np.ndarray:
    """Với mỗi dòng: mã của dòng đó xuất hiện bao nhiêu lần trong cả mảng."""
    codes = np.asarray(codes, dtype=np.int64)
    if codes.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.bincount(codes)[codes]


def molecule_frequency(molecule_a: np.ndarray, molecule_b: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Số cặp mà phân tử A (và B) của mỗi dòng tham gia, tính cả hai phía của cặp."""
    left = np.asarray(molecule_a, dtype=np.int64)
    right = np.asarray(molecule_b, dtype=np.int64)
    if left.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    counts = np.bincount(np.concatenate([left, right]))
    return counts[left], counts[right]


def edit_codes(fragment_a: np.ndarray, fragment_b: np.ndarray) -> np.ndarray:
    """Mã của phép sửa KHÔNG kể chiều: X->Y và Y->X là cùng một phép sửa.

    Vì sao không kể chiều: dataset lưu cặp theo khóa A < B, nên chiều trong bảng là chiều
    của khóa, không phải chiều hóa học (xem metrics.expand_directions).
    """
    left = np.asarray(fragment_a, dtype=np.int64)
    right = np.asarray(fragment_b, dtype=np.int64)
    pairs = np.stack([np.minimum(left, right), np.maximum(left, right)], axis=1)
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    _, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return inverse.astype(np.int64).ravel()


# --------------------------------------------------------------------------- chia nhóm


def quantile_groups(values: np.ndarray, bins: int) -> tuple[np.ndarray, list[str]]:
    """Chia một cột số theo phân vị. Trả về (mã nhóm từng dòng, tên nhóm).

    Biên trùng nhau được gộp (cột số nguyên như n_assays phần lớn bằng 1 thì ra ít nhóm
    hơn `bins`), nên không có nhóm rỗng. Giá trị thiếu (NaN) vào nhóm riêng 'thiếu' ở cuối.
    Nhóm i là [biên i, biên i+1), nhóm cuối lấy cả biên trên.
    """
    if bins < 1:
        raise ValueError(f"quantile_bins phải >= 1, nhận {bins}")
    numbers = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(numbers)
    codes = np.full(numbers.size, -1, dtype=np.int64)
    names: list[str] = []
    if finite.any():
        # method="lower": biên luôn là một giá trị có thật trong dữ liệu. Nội suy (mặc định)
        # cho biên kiểu 1,0000000000000004 vì sai số làm tròn của linspace, sinh nhóm rỗng.
        edges = np.unique(np.quantile(numbers[finite], np.linspace(0.0, 1.0, bins + 1),
                                      method="lower"))
        if edges.size == 1:
            codes[finite] = 0
            names.append(f"= {edges[0]:.4g}")
        else:
            codes[finite] = np.searchsorted(edges[1:-1], numbers[finite], side="right")
            for index in range(edges.size - 1):
                closing = "]" if index == edges.size - 2 else ")"
                names.append(f"[{edges[index]:.4g}, {edges[index + 1]:.4g}{closing}")
    if (~finite).any():
        codes[~finite] = len(names)
        names.append(MISSING_LABEL)
    return codes, names


def category_groups(values: Sequence[Any]) -> tuple[np.ndarray, list[str]]:
    """Chia theo giá trị rời rạc (bool, chuỗi). None vào nhóm 'thiếu'."""
    labels = [MISSING_LABEL if value is None else str(value) for value in values]
    names = sorted(set(labels))
    position = {name: index for index, name in enumerate(names)}
    return np.array([position[label] for label in labels], dtype=np.int64), names


def magnitude_groups(truth: np.ndarray, tolerance: float,
                     bins: int) -> tuple[np.ndarray, list[str]]:
    """Chia theo |Δp thật|: một nhóm 'trong sai số' (≤ tolerance), phần còn lại theo phân vị.

    Tách riêng nhóm ≤ tolerance vì ở đó dấu của nhãn không có nghĩa (sai số thí nghiệm,
    xem metrics.change_class_metrics), nên không tính đúng dấu cho nhóm này.
    """
    size = np.abs(np.asarray(truth, dtype=np.float64))
    inside = size <= tolerance
    codes = np.zeros(size.size, dtype=np.int64)
    names = [f"|Δp| <= {tolerance:g} (trong sai số)"]
    if (~inside).any():
        outer, outer_names = quantile_groups(size[~inside], bins)
        codes[~inside] = outer + 1
        names += [f"|Δp| {name}" for name in outer_names]
    return codes, names


# --------------------------------------------------------------------------- chỉ số


def sign_agreement(truth: np.ndarray, prediction: np.ndarray,
                   tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """(dòng được tính, dòng đúng dấu). Dòng có |Δp thật| ≤ tolerance không được tính."""
    counted = np.abs(truth) > tolerance
    correct = counted & (np.sign(prediction) == np.sign(truth))
    return counted, correct


def spread_report(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    """Mô hình đoán rộng hay hẹp so với nhãn.

    prediction_std / truth_std nhỏ hơn hẳn 1 và slope nhỏ hơn hẳn 1 nghĩa là dự đoán bị
    co về 0: mô hình chưa học đủ (underfit), không phải overfit.
    slope là hệ số của hồi quy tuyến tính dự đoán theo nhãn thật.
    """
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.size < 2:
        return {"truth_std": None, "prediction_std": None, "std_ratio": None,
                "slope": None, "bias": None}
    truth_std = float(truth.std())
    prediction_std = float(prediction.std())
    variance = float(truth.var())
    covariance = float(np.mean((truth - truth.mean()) * (prediction - prediction.mean())))
    return {
        "truth_std": truth_std,
        "prediction_std": prediction_std,
        "std_ratio": prediction_std / truth_std if truth_std > 0.0 else None,
        "slope": covariance / variance if variance > 0.0 else None,
        "bias": float(np.mean(prediction - truth)),
    }


def group_table(codes: np.ndarray, names: Sequence[str], truth: np.ndarray,
                prediction: np.ndarray, tolerance: float) -> list[dict[str, Any]]:
    """Chỉ số theo từng nhóm dòng: số dòng, tỉ lệ, MAE, bias, đúng dấu, Spearman."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    counted, correct = sign_agreement(truth, prediction, tolerance)
    total = max(int(truth.size), 1)
    table = []
    for index, name in enumerate(names):
        rows = np.flatnonzero(codes == index)
        if rows.size == 0:
            continue
        sign_rows = int(counted[rows].sum())
        error = prediction[rows] - truth[rows]
        table.append({
            "group": name,
            "rows": int(rows.size),
            "share": rows.size / total,
            "mae": float(np.abs(error).mean()),
            "bias": float(error.mean()),
            "sign_rows": sign_rows,
            "sign_accuracy": (float(correct[rows].sum()) / sign_rows) if sign_rows else None,
            "spearman": metrics.spearman(truth[rows], prediction[rows]),
        })
    return table


# --------------------------------------------------------------------------- query


def query_table(keys: Mapping[str, np.ndarray], truth: np.ndarray, prediction: np.ndarray,
                tolerance: float, min_candidates: int,
                expand_both_directions: bool) -> dict[str, np.ndarray]:
    """Chỉ số ranking cho TỪNG query, cùng định nghĩa với metrics.ranking_metrics.

    Trả về các cột cùng độ dài (một dòng một query có ít nhất min_candidates ứng viên):
    reference, protein, endpoint (mã), candidates, agree, counted (số cặp ứng viên đúng
    chiều / được tính), spearman (NaN khi mọi nhãn bằng nhau), top_1 (0/1).
    Trung bình agree/counted trên các query có counted > 0 đúng bằng
    pairwise_accuracy_macro của metrics.ranking_metrics.
    """
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if expand_both_directions:
        reference, _, truth, prediction = metrics.expand_directions(
            keys["molecule_a"], keys["molecule_b"], truth, prediction)
        protein = np.concatenate([keys["protein"]] * 2)
        endpoint = np.concatenate([keys["endpoint"]] * 2)
    else:
        reference, protein, endpoint = keys["molecule_a"], keys["protein"], keys["endpoint"]
    groups = metrics.query_groups([reference, protein, endpoint])
    order, bounds = metrics.group_bounds(groups)
    columns: dict[str, list[float]] = {name: [] for name in (
        "reference", "protein", "endpoint", "candidates", "agree", "counted", "spearman",
        "top_1")}
    for index in range(bounds.size - 1):
        rows = order[bounds[index]:bounds[index + 1]]
        if rows.size < min_candidates:
            continue
        group_truth = truth[rows]
        group_prediction = prediction[rows]
        agree, counted = metrics.pairwise_direction(group_truth, group_prediction, tolerance)
        value = metrics.spearman(group_truth, group_prediction)
        first = rows[0]
        columns["reference"].append(reference[first])
        columns["protein"].append(protein[first])
        columns["endpoint"].append(endpoint[first])
        columns["candidates"].append(rows.size)
        columns["agree"].append(agree)
        columns["counted"].append(counted)
        columns["spearman"].append(np.nan if value is None else value)
        columns["top_1"].append(float(metrics.top_k_hit(group_truth, group_prediction, 1)))
    integer = ("reference", "protein", "endpoint", "candidates", "agree", "counted")
    return {name: np.asarray(values, dtype=np.int64 if name in integer else np.float64)
            for name, values in columns.items()}


def query_group_table(codes: np.ndarray, names: Sequence[str],
                      queries: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Đúng chiều theo nhóm query: macro (mỗi query một phiếu), micro, top-1."""
    agree = queries["agree"].astype(np.float64)
    counted = queries["counted"].astype(np.float64)
    table = []
    for index, name in enumerate(names):
        rows = np.flatnonzero(codes == index)
        if rows.size == 0:
            continue
        used = rows[counted[rows] > 0]
        table.append({
            "group": name,
            "queries": int(rows.size),
            "pairwise_accuracy_macro": (float(np.mean(agree[used] / counted[used]))
                                        if used.size else None),
            "pairwise_accuracy_micro": (float(agree[rows].sum() / counted[rows].sum())
                                        if counted[rows].sum() else None),
            "top_1": float(queries["top_1"][rows].mean()),
        })
    return table


def worst_proteins(queries: Mapping[str, np.ndarray], min_queries: int,
                   count: int) -> list[dict[str, Any]]:
    """Các protein có đúng chiều (macro theo query) thấp nhất, chỉ xét protein đủ query.

    Trả về mã protein; train.run_errors đổi mã ra protein_hash.
    """
    usable = queries["counted"] > 0
    protein = queries["protein"][usable]
    ratio = queries["agree"][usable] / queries["counted"][usable]
    if protein.size == 0:
        return []
    codes, inverse, sizes = np.unique(protein, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=ratio) / sizes
    keep = np.flatnonzero(sizes >= min_queries)
    keep = keep[np.argsort(means[keep], kind="stable")][:count]
    return [{"protein": int(codes[i]), "queries": int(sizes[i]),
             "pairwise_accuracy_macro": float(means[i])} for i in keep]


def worst_edits(edits: np.ndarray, fragment_a: np.ndarray, fragment_b: np.ndarray,
                truth: np.ndarray, prediction: np.ndarray, tolerance: float,
                min_rows: int, count: int) -> list[dict[str, Any]]:
    """Các phép sửa có tỉ lệ đúng dấu thấp nhất, chỉ xét phép sửa đủ dòng được tính.

    Trả về mã nhóm thế của một dòng đại diện; train.run_errors đổi mã ra SMILES.
    """
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    counted, correct = sign_agreement(truth, prediction, tolerance)
    if edits.size == 0:
        return []
    size = int(edits.max()) + 1
    used = np.bincount(edits, weights=counted, minlength=size)
    hits = np.bincount(edits, weights=correct, minlength=size)
    rows = np.bincount(edits, minlength=size)
    mae = np.bincount(edits, weights=np.abs(prediction - truth), minlength=size) / np.maximum(
        rows, 1)
    # Ít nhất một dòng được tính, để tỉ lệ đúng dấu luôn xác định.
    keep = np.flatnonzero(used >= max(min_rows, 1))
    accuracy = hits[keep] / used[keep]
    keep = keep[np.argsort(accuracy, kind="stable")][:count]
    first = np.full(size, -1, dtype=np.int64)
    present, position = np.unique(edits, return_index=True)
    first[present] = position
    return [{"edit": int(code), "fragment_a": int(fragment_a[first[code]]),
             "fragment_b": int(fragment_b[first[code]]), "rows": int(rows[code]),
             "sign_rows": int(used[code]), "sign_accuracy": float(hits[code] / used[code]),
             "mae": float(mae[code])} for code in keep]


# --------------------------------------------------------------------------- in báo cáo


def format_group_table(title: str, table: Sequence[Mapping[str, Any]]) -> list[str]:
    """Bảng chữ cho log. Ô None in là '-'."""
    def show(value: Any, digits: int = 4) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return f"{value:,}" if isinstance(value, int) else str(value)

    if not table:
        return [f"{title}: không có nhóm nào"]
    columns = [key for key in table[0] if key != "group"]
    lines = [title]
    width = max(len(str(row["group"])) for row in table)
    lines.append("  " + "nhóm".ljust(width) + " | " + " | ".join(columns))
    for row in table:
        lines.append("  " + str(row["group"]).ljust(width) + " | "
                     + " | ".join(show(row[key]) for key in columns))
    return lines
