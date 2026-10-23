"""Stage 3: chỉ số regression và chỉ số ranking trong từng query.

Hai nhóm chỉ số, cố ý tách rời:

Regression — trên từng dòng, không cần nhóm: MAE, RMSE, R², Spearman toàn cục.
Ranking    — CHỈ trong cùng một query. Một query là (phân tử tham chiếu, protein,
             endpoint) với nhiều ứng viên. Không bao giờ xếp chung ứng viên của hai
             query khác nhau vào một bảng.

Mọi hàm ở đây làm việc trên mảng numpy và không import torch, nên test được trên máy
chưa cài thư viện học sâu. Không dùng scipy: hệ số Spearman được tính tại chỗ (scipy
không nằm trong requirements của dự án).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

# Giá trị trả về khi một chỉ số không xác định được (ví dụ Spearman của một query mà mọi
# nhãn bằng nhau). Dùng None chứ không dùng 0.0 để không bị cộng vào trung bình như thể
# là một kết quả thật.
UNDEFINED = None


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Hạng của từng phần tử, giá trị bằng nhau nhận hạng trung bình.

    Nhận: mảng 1 chiều.
    Trả về: mảng float64 cùng độ dài, hạng bắt đầu từ 1.
    Vì sao cần: Spearman trên dữ liệu có nhiều giá trị trùng (Δp hay trùng nhau sau khi
    làm tròn) sẽ sai nếu phá thế hòa tuỳ ý; hạng trung bình là định nghĩa chuẩn.
    """
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            # Cả khối [start, index) cùng giá trị -> cùng hạng trung bình.
            ranks[order[start:index]] = 0.5 * (start + index - 1) + 1.0
            start = index
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    """Hệ số tương quan Pearson; None nếu một phía không có phương sai."""
    if left.size < 2:
        return UNDEFINED
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = math.sqrt(float(left_centered @ left_centered)
                            * float(right_centered @ right_centered))
    if denominator == 0.0:
        return UNDEFINED
    return float(left_centered @ right_centered) / denominator


def spearman(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    """Spearman = Pearson trên hạng; None nếu một phía toàn giá trị bằng nhau."""
    if truth.size < 2:
        return UNDEFINED
    return pearson(average_ranks(truth), average_ranks(prediction))


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    """MAE, RMSE, R², Spearman và số dòng.

    Nhận: nhãn thật và dự đoán, cùng độ dài.
    Trả về: dict chỉ số; R² và Spearman là None khi nhãn không có phương sai.
    R² tính quanh trung bình của nhãn thật (định nghĩa thông thường), nên có thể âm khi
    mô hình tệ hơn việc đoán bằng trung bình.
    """
    if truth.shape != prediction.shape:
        raise ValueError(f"lệch độ dài: {truth.shape} và {prediction.shape}")
    if truth.size == 0:
        return {"n": 0, "mae": UNDEFINED, "rmse": UNDEFINED, "r2": UNDEFINED,
                "spearman": UNDEFINED}
    error = prediction - truth
    residual = float(error @ error)
    centered = truth - truth.mean()
    total = float(centered @ centered)
    return {
        "n": int(truth.size),
        "mae": float(np.abs(error).mean()),
        "rmse": math.sqrt(residual / truth.size),
        "r2": (1.0 - residual / total) if total > 0.0 else UNDEFINED,
        "spearman": spearman(truth, prediction),
    }


def expand_directions(reference: np.ndarray, candidate: np.ndarray, truth: np.ndarray,
                      prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                       np.ndarray]:
    """Mỗi cặp (A, B, Δp) thành hai hướng: A->B giữ dấu, B->A đổi dấu.

    Nhận: mã phân tử A, mã phân tử B, Δp thật, Δp dự đoán.
    Trả về: (mã phân tử tham chiếu, mã ứng viên, Δp thật, Δp dự đoán) đã nhân đôi.
    Vì sao cần: dataset lưu cặp theo chiều khóa A < B (xem Stage 2b), nên nếu chỉ nhóm
    theo phía A thì một query bị cắt vụn theo thứ tự khóa phân tử chứ không theo hóa học.
    Với một phân tử tham chiếu cố định, xếp hạng ứng viên theo Δp đúng bằng xếp hạng theo
    hoạt tính của ứng viên, nên đổi dấu là phép biến đổi hợp lệ, không phải bịa thêm dữ
    liệu.
    """
    return (np.concatenate([reference, candidate]),
            np.concatenate([candidate, reference]),
            np.concatenate([truth, -truth]),
            np.concatenate([prediction, -prediction]))


def query_groups(columns: Sequence[np.ndarray]) -> np.ndarray:
    """Mã nhóm (query) cho từng dòng, từ các cột khóa đã mã hóa thành số nguyên.

    Nhận: danh sách mảng số nguyên cùng độ dài (ví dụ: phân tử tham chiếu, protein,
    endpoint).
    Trả về: mảng int64, cùng mã nghĩa là cùng query.
    """
    if not columns:
        raise ValueError("cần ít nhất một cột khóa")
    stacked = np.stack([np.asarray(column, dtype=np.int64) for column in columns], axis=1)
    _, inverse = np.unique(stacked, axis=0, return_inverse=True)
    return inverse.astype(np.int64).ravel()


def group_bounds(group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sắp xếp theo mã nhóm; trả về (thứ tự dòng, biên của từng nhóm).

    Biên có dạng [0, ..., n] nên nhóm thứ i là order[bounds[i]:bounds[i + 1]].
    """
    order = np.argsort(group_ids, kind="stable")
    sorted_ids = group_ids[order]
    changes = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
    bounds = np.concatenate([[0], changes, [group_ids.size]])
    return order, bounds


def pairwise_direction(truth: np.ndarray, prediction: np.ndarray,
                       tolerance: float) -> tuple[int, int]:
    """Số cặp ứng viên đoán đúng chiều, và số cặp được tính.

    Nhận: nhãn và dự đoán của các ứng viên TRONG MỘT query; ngưỡng coi hai nhãn là bằng.
    Trả về: (số cặp đúng chiều, số cặp được tính).
    Cặp có hai nhãn lệch nhau không quá tolerance bị loại khỏi mẫu số: không có chiều
    đúng để so. Cặp mà dự đoán bằng nhau tính là sai, vì mô hình không chọn được.
    """
    difference_truth = truth[:, None] - truth[None, :]
    difference_prediction = prediction[:, None] - prediction[None, :]
    upper = np.triu(np.ones_like(difference_truth, dtype=bool), k=1)
    counted = upper & (np.abs(difference_truth) > tolerance)
    total = int(counted.sum())
    if total == 0:
        return 0, 0
    agree = np.sign(difference_truth) == np.sign(difference_prediction)
    return int((counted & agree).sum()), total


def top_k_hit(truth: np.ndarray, prediction: np.ndarray, k: int) -> bool:
    """Ứng viên tốt nhất theo nhãn có nằm trong k ứng viên tốt nhất theo dự đoán không.

    Nhãn trùng ở đỉnh: tính là trúng nếu BẤT KỲ ứng viên đạt nhãn lớn nhất nằm trong
    top-k dự đoán.
    """
    best = np.flatnonzero(truth >= truth.max())
    chosen = np.argsort(-prediction, kind="stable")[:k]
    return bool(np.intersect1d(best, chosen).size > 0)


def ranking_metrics(group_ids: np.ndarray, truth: np.ndarray, prediction: np.ndarray,
                    top_k: Sequence[int], min_candidates: int,
                    tolerance: float = 0.0) -> dict[str, object]:
    """Chỉ số ranking trong từng query, tổng hợp theo cả macro và micro.

    Nhận: mã query của từng dòng, nhãn, dự đoán, danh sách k, số ứng viên tối thiểu để
    một query được xếp hạng, ngưỡng coi hai nhãn là bằng nhau.
    Trả về: dict gồm
        queries_total / queries_ranked / queries_skipped — mẫu số được ghi rõ;
        spearman_macro, spearman_median — trung bình và trung vị Spearman THEO QUERY;
        spearman_undefined — số query mọi nhãn bằng nhau nên không tính được;
        pairwise_accuracy_macro — trung bình theo query của tỷ lệ đúng chiều;
        pairwise_accuracy_micro — gộp mọi cặp ứng viên của mọi query rồi mới chia;
        pairs_counted — mẫu số của micro;
        top_{k} và top_{k}_queries — tỷ lệ trúng và mẫu số riêng của từng k.

    Vì sao báo cáo cả macro và micro: query dài có nhiều cặp nên chi phối micro, còn
    macro cho mỗi query một phiếu như nhau. Hai con số trả lời hai câu hỏi khác nhau và
    buổi họp chưa chốt, nên không chọn một cái.
    Vì sao KHÔNG có spearman_micro: Spearman chỉ có nghĩa trong một query; gộp hạng của
    các query khác nhau vào một bảng chính là điều phải tránh. Với top-k, macro và micro
    trùng nhau (chỉ số 0/1 trên mỗi query) nên chỉ báo cáo một con số kèm mẫu số.
    """
    if not (truth.shape == prediction.shape == group_ids.shape):
        raise ValueError("group_ids, truth và prediction phải cùng độ dài")
    if min_candidates < 2:
        raise ValueError(f"min_candidates phải từ 2 trở lên, nhận {min_candidates}")
    order, bounds = group_bounds(group_ids)
    spearman_values: list[float] = []
    accuracy_values: list[float] = []
    agree_total = 0
    pairs_total = 0
    undefined = 0
    skipped = 0
    hits = {int(k): 0 for k in top_k}
    denominators = {int(k): 0 for k in top_k}
    for index in range(bounds.size - 1):
        rows = order[bounds[index]:bounds[index + 1]]
        if rows.size < min_candidates:
            skipped += 1
            continue
        group_truth = truth[rows]
        group_prediction = prediction[rows]
        value = spearman(group_truth, group_prediction)
        if value is None:
            undefined += 1
        else:
            spearman_values.append(value)
        agree, counted = pairwise_direction(group_truth, group_prediction, tolerance)
        if counted:
            agree_total += agree
            pairs_total += counted
            accuracy_values.append(agree / counted)
        for k in hits:
            # top-k chỉ có nghĩa khi query có ít nhất k ứng viên; mẫu số riêng từng k.
            if rows.size >= k:
                denominators[k] += 1
                if top_k_hit(group_truth, group_prediction, k):
                    hits[k] += 1
    ranked = int(bounds.size - 1 - skipped)
    result: dict[str, object] = {
        "queries_total": int(bounds.size - 1),
        "queries_ranked": ranked,
        "queries_skipped_too_few_candidates": skipped,
        "spearman_macro": float(np.mean(spearman_values)) if spearman_values else UNDEFINED,
        "spearman_median": float(np.median(spearman_values)) if spearman_values else UNDEFINED,
        "spearman_queries": len(spearman_values),
        "spearman_undefined": undefined,
        "pairwise_accuracy_macro": (float(np.mean(accuracy_values)) if accuracy_values
                                    else UNDEFINED),
        "pairwise_accuracy_micro": (agree_total / pairs_total) if pairs_total else UNDEFINED,
        "pairs_counted": pairs_total,
    }
    for k in sorted(hits):
        result[f"top_{k}"] = (hits[k] / denominators[k]) if denominators[k] else UNDEFINED
        result[f"top_{k}_queries"] = denominators[k]
    return result


def metrics_by_endpoint(endpoints: np.ndarray, truth: np.ndarray, prediction: np.ndarray,
                        names: Sequence[str]) -> dict[str, dict[str, float | None]]:
    """Chỉ số regression tách riêng từng endpoint.

    Nhận: mã endpoint của từng dòng, nhãn, dự đoán, danh sách tên endpoint theo mã.
    Trả về: {tên endpoint: chỉ số regression}; endpoint không có dòng nào vẫn xuất hiện
    với n = 0, để bảng báo cáo không bị thiếu cột giữa các lần chạy.
    """
    result = {}
    for code, name in enumerate(names):
        mask = endpoints == code
        result[name] = regression_metrics(truth[mask], prediction[mask])
    return result


def asymmetry_metrics(forward: np.ndarray, backward: np.ndarray, identity: np.ndarray,
                      label_spread: float) -> dict[str, float | None]:
    """Mức lệch khỏi hai tính chất mà kiến trúc `direct` KHÔNG bảo đảm.

    Nhận:
        forward      : f(A, B) trên một mẫu dòng validation
        backward     : f(B, A) trên đúng các dòng đó, đã đổi chỗ hai phía
        identity     : f(A, A) trên đúng các dòng đó, phía B thay bằng chính phía A
        label_spread : độ lệch chuẩn của nhãn thật trên mẫu đó, dùng làm thước đo
    Trả về: các số đo, kèm hai tỉ lệ đã chuẩn hóa.

    Phần đối xứng của dự đoán là r = (f(A,B) + f(B,A)) / 2. Mô hình đối xứng hoàn hảo có
    r = 0 với mọi cặp; `twin` cho đúng 0 do xây dựng, `direct` thì không. Một con số tuyệt
    đối không đọc được nếu không có thước đo, nên hàm trả về cả tỉ lệ so với biên độ dự
    đoán và so với độ lệch chuẩn của nhãn: lệch 0,1 log là nhỏ khi nhãn trải 1,5 log và
    lớn khi nhãn trải 0,2 log.
    """
    if not (forward.size == backward.size == identity.size):
        raise ValueError("ba mảng phải cùng số dòng")
    if forward.size == 0:
        return {"n": 0, "antisymmetry_mean_abs": UNDEFINED, "antisymmetry_p90": UNDEFINED,
                "antisymmetry_max": UNDEFINED, "antisymmetry_over_prediction": UNDEFINED,
                "antisymmetry_over_label_spread": UNDEFINED, "identity_mean_abs": UNDEFINED,
                "identity_p90": UNDEFINED, "identity_max": UNDEFINED,
                "identity_over_label_spread": UNDEFINED, "prediction_mean_abs": UNDEFINED,
                "label_spread": UNDEFINED}
    residual = np.abs((forward + backward) / 2.0)
    scale = float(np.mean(np.abs(forward)))
    identity_abs = np.abs(identity)
    spread = float(label_spread)
    return {
        "n": int(forward.size),
        "antisymmetry_mean_abs": float(np.mean(residual)),
        "antisymmetry_p90": float(np.percentile(residual, 90)),
        "antisymmetry_max": float(np.max(residual)),
        "antisymmetry_over_prediction": float(np.mean(residual) / scale) if scale else UNDEFINED,
        "antisymmetry_over_label_spread": float(np.mean(residual) / spread) if spread
        else UNDEFINED,
        "identity_mean_abs": float(np.mean(identity_abs)),
        "identity_p90": float(np.percentile(identity_abs, 90)),
        "identity_max": float(np.max(identity_abs)),
        "identity_over_label_spread": float(np.mean(identity_abs) / spread) if spread
        else UNDEFINED,
        "prediction_mean_abs": scale,
        "label_spread": spread,
    }


def format_asymmetry(label: str, scores: Mapping[str, float | None]) -> list[str]:
    """Một dòng log cho số đo bất đối xứng; None in thành '-'."""

    def show(value: object) -> str:
        if value is None:
            return "-"
        return f"{value:.4f}" if isinstance(value, float) else f"{value:,}"

    return [f"{label}: n = {show(scores.get('n'))}"
            f" | |f(A,B)+f(B,A)|/2 trung bình {show(scores.get('antisymmetry_mean_abs'))}"
            f" (= {show(scores.get('antisymmetry_over_label_spread'))} lần sd nhãn)"
            f" | |f(A,A)| trung bình {show(scores.get('identity_mean_abs'))}"
            f" (= {show(scores.get('identity_over_label_spread'))} lần sd nhãn)"]


def format_metrics(label: str, regression: Mapping[str, float | None],
                   ranking: Mapping[str, object] | None = None) -> list[str]:
    """Các dòng văn bản gọn để ghi log; None in thành '-' chứ không in thành 0."""

    def show(value: object, digits: int = 4) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return f"{value:,}"

    lines = [f"{label}: n = {show(regression.get('n'))}"
             f" | MAE {show(regression.get('mae'))}"
             f" | RMSE {show(regression.get('rmse'))}"
             f" | R2 {show(regression.get('r2'))}"
             f" | Spearman {show(regression.get('spearman'))}"]
    if ranking is not None:
        lines.append(f"    ranking: query xếp hạng được {show(ranking.get('queries_ranked'))}"
                     f"/{show(ranking.get('queries_total'))}"
                     f" | Spearman macro {show(ranking.get('spearman_macro'))}"
                     f" | đúng chiều macro {show(ranking.get('pairwise_accuracy_macro'))}"
                     f" micro {show(ranking.get('pairwise_accuracy_micro'))}")
        tops = [f"top-{k} {show(ranking.get(f'top_{k}'))}"
                f" (trên {show(ranking.get(f'top_{k}_queries'))} query)"
                for k in sorted(int(name.split('_')[1]) for name in ranking
                                if name.startswith('top_') and name.count('_') == 1)]
        if tops:
            lines.append("    " + " | ".join(tops))
    return lines
