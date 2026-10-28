"""Chỉ số của Stage 3: regression và ranking trong từng query (không cần torch)."""

from __future__ import annotations

import numpy as np
import pytest

from molopt.evaluator import metrics


def test_average_ranks_gives_tied_values_the_same_rank() -> None:
    ranks = metrics.average_ranks(np.array([10.0, 20.0, 20.0, 30.0]))
    assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_is_one_for_a_perfectly_ordered_prediction() -> None:
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.spearman(truth, np.array([0.1, 0.2, 0.9, 1.5])) == pytest.approx(1.0)


def test_spearman_is_minus_one_for_a_reversed_prediction() -> None:
    truth = np.array([1.0, 2.0, 3.0])
    assert metrics.spearman(truth, np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)


def test_spearman_is_undefined_when_one_side_has_no_spread() -> None:
    # Mọi nhãn bằng nhau: không có thứ tự thật để so, nên None chứ không phải 0.0.
    assert metrics.spearman(np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0])) is None


def test_regression_metrics_on_a_hand_computed_example() -> None:
    truth = np.array([1.0, 2.0, 3.0])
    prediction = np.array([1.5, 2.0, 2.0])
    result = metrics.regression_metrics(truth, prediction)
    assert result["n"] == 3
    assert result["mae"] == pytest.approx((0.5 + 0.0 + 1.0) / 3)
    assert result["rmse"] == pytest.approx(np.sqrt((0.25 + 0.0 + 1.0) / 3))
    # SS_tot = 2.0 quanh trung bình 2.0; SS_res = 1.25.
    assert result["r2"] == pytest.approx(1.0 - 1.25 / 2.0)


def test_regression_metrics_reports_r2_as_undefined_without_label_spread() -> None:
    result = metrics.regression_metrics(np.array([5.0, 5.0]), np.array([4.0, 6.0]))
    assert result["r2"] is None and result["n"] == 2


def test_regression_metrics_on_an_empty_slice() -> None:
    result = metrics.regression_metrics(np.zeros(0), np.zeros(0))
    assert result["n"] == 0 and result["mae"] is None


def test_expand_directions_doubles_the_rows_and_flips_the_sign() -> None:
    reference, candidate, truth, prediction = metrics.expand_directions(
        np.array([0]), np.array([1]), np.array([1.5]), np.array([1.2]))
    assert reference.tolist() == [0, 1]
    assert candidate.tolist() == [1, 0]
    assert truth.tolist() == [1.5, -1.5]
    assert prediction.tolist() == [1.2, -1.2]


def test_query_groups_gives_the_same_code_to_identical_keys() -> None:
    groups = metrics.query_groups([np.array([7, 7, 8]), np.array([1, 1, 1])])
    assert groups[0] == groups[1] and groups[0] != groups[2]


def test_pairwise_direction_skips_pairs_whose_labels_are_equal() -> None:
    truth = np.array([1.0, 1.0, 2.0])
    prediction = np.array([0.0, 5.0, 9.0])
    agree, counted = metrics.pairwise_direction(truth, prediction, tolerance=0.0)
    # Cặp (0,1) có nhãn bằng nhau nên bị loại; hai cặp còn lại đều đúng chiều.
    assert (agree, counted) == (2, 2)


def test_pairwise_direction_counts_a_tied_prediction_as_wrong() -> None:
    agree, counted = metrics.pairwise_direction(np.array([1.0, 2.0]), np.array([3.0, 3.0]),
                                                tolerance=0.0)
    assert (agree, counted) == (0, 1)


def test_top_k_hit_accepts_any_candidate_sharing_the_best_label() -> None:
    truth = np.array([3.0, 3.0, 1.0])
    prediction = np.array([0.0, 9.0, 1.0])
    assert metrics.top_k_hit(truth, prediction, k=1) is True


def test_top_k_hit_is_false_when_the_best_candidate_is_ranked_below_k() -> None:
    assert metrics.top_k_hit(np.array([5.0, 1.0]), np.array([1.0, 5.0]), k=1) is False


def ranking_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ba query: một xếp hạng đúng hoàn toàn, một sai hoàn toàn, một chỉ có 1 ứng viên."""
    groups = np.array([0, 0, 0, 1, 1, 2])
    truth = np.array([1.0, 2.0, 3.0, 5.0, 1.0, 9.0])
    prediction = np.array([1.0, 2.0, 3.0, 1.0, 5.0, 9.0])
    return groups, truth, prediction


def test_ranking_metrics_separates_macro_from_micro() -> None:
    groups, truth, prediction = ranking_example()
    result = metrics.ranking_metrics(groups, truth, prediction, top_k=[1, 3],
                                     min_candidates=2)
    assert result["queries_total"] == 3
    assert result["queries_ranked"] == 2
    assert result["queries_skipped_too_few_candidates"] == 1
    # macro cho mỗi query một phiếu: (1,0 + 0,0) / 2; micro gộp cặp: 3 đúng trên 4 cặp.
    assert result["pairwise_accuracy_macro"] == pytest.approx(0.5)
    assert result["pairwise_accuracy_micro"] == pytest.approx(0.75)
    assert result["pairs_counted"] == 4
    assert result["spearman_macro"] == pytest.approx(0.0)


def test_ranking_metrics_gives_each_k_its_own_denominator() -> None:
    groups, truth, prediction = ranking_example()
    result = metrics.ranking_metrics(groups, truth, prediction, top_k=[1, 3],
                                     min_candidates=2)
    # top-1 tính trên cả hai query xếp hạng được; top-3 chỉ trên query có >= 3 ứng viên.
    assert (result["top_1"], result["top_1_queries"]) == (pytest.approx(0.5), 2)
    assert (result["top_3"], result["top_3_queries"]) == (pytest.approx(1.0), 1)


def test_ranking_metrics_counts_queries_whose_labels_are_all_equal() -> None:
    groups = np.array([0, 0])
    result = metrics.ranking_metrics(groups, np.array([4.0, 4.0]), np.array([1.0, 2.0]),
                                     top_k=[1], min_candidates=2)
    assert result["spearman_undefined"] == 1
    assert result["spearman_macro"] is None
    # Không có cặp nào có chiều thật, nên micro cũng không xác định.
    assert result["pairwise_accuracy_micro"] is None


def test_ranking_metrics_refuses_a_min_candidates_below_two() -> None:
    with pytest.raises(ValueError, match="min_candidates"):
        metrics.ranking_metrics(np.array([0]), np.array([1.0]), np.array([1.0]),
                                top_k=[1], min_candidates=1)


def test_ranking_metrics_refuses_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="cùng độ dài"):
        metrics.ranking_metrics(np.array([0, 0]), np.array([1.0]), np.array([1.0]),
                                top_k=[1], min_candidates=2)


def test_metrics_by_endpoint_keeps_an_entry_for_an_empty_endpoint() -> None:
    endpoints = np.array([0, 0, 1])
    truth = np.array([1.0, 2.0, 3.0])
    result = metrics.metrics_by_endpoint(endpoints, truth, truth, ["Ki", "Kd", "IC50"])
    assert result["Ki"]["n"] == 2 and result["Kd"]["n"] == 1
    assert result["IC50"]["n"] == 0 and result["IC50"]["mae"] is None


def test_format_metrics_prints_a_dash_for_an_undefined_value() -> None:
    lines = metrics.format_metrics("thử", {"n": 2, "mae": 0.5, "rmse": 0.5, "r2": None,
                                           "spearman": None})
    assert "R2 -" in lines[0] and "Spearman -" in lines[0]


def test_change_class_puts_the_threshold_itself_in_neutral() -> None:
    values = np.array([-0.31, -0.3, 0.0, 0.3, 0.31])
    assert metrics.change_class(values, 0.3).tolist() == [0, 1, 1, 1, 2]


def test_change_class_metrics_on_a_hand_computed_example() -> None:
    # Thật: kém, ngang, tốt, tốt. Dự đoán: kém, kém, ngang, tốt.
    truth = np.array([-1.0, 0.1, 0.5, 2.0])
    prediction = np.array([-0.5, -0.5, 0.2, 1.0])
    result = metrics.change_class_metrics(truth, prediction, 0.3)
    assert result["confusion"] == [[1, 0, 0], [1, 0, 0], [0, 1, 1]]
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["worse_f1"] == pytest.approx(2 / 3)
    assert result["neutral_f1"] == pytest.approx(0.0)
    assert result["better_f1"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(4 / 9)
    assert result["better_recall"] == pytest.approx(0.5)
    assert (result["worse_rows"], result["neutral_rows"], result["better_rows"]) == (1, 1, 2)


def test_change_class_metrics_leaves_an_absent_class_out_of_macro_f1() -> None:
    # Không có dòng "kém" nào, cả thật lẫn dự đoán: F1 của lớp đó là None.
    result = metrics.change_class_metrics(np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.3)
    assert result["worse_f1"] is None
    assert result["macro_f1"] == pytest.approx(1.0)


def test_change_class_metrics_on_an_empty_slice() -> None:
    result = metrics.change_class_metrics(np.array([]), np.array([]), 0.3)
    assert result["n"] == 0 and result["accuracy"] is None and result["macro_f1"] is None


def test_format_metrics_prints_the_three_class_line() -> None:
    classes = metrics.change_class_metrics(np.array([-1.0, 1.0]), np.array([-1.0, 1.0]), 0.3)
    lines = metrics.format_metrics("thử", {"n": 2, "mae": 0.0, "rmse": 0.0, "r2": 1.0,
                                           "spearman": 1.0}, classes=classes)
    assert "ba lớp (±0.30)" in lines[-1] and "macro-F1 1.0000" in lines[-1]


def test_asymmetry_metrics_is_zero_for_a_perfectly_antisymmetric_model() -> None:
    # Đây chính là điều kiến trúc twin bảo đảm, nên số đo phải ra đúng 0.
    forward = np.array([1.0, -2.0, 0.5])
    scores = metrics.asymmetry_metrics(forward, -forward, np.zeros(3), 1.0)
    assert scores["n"] == 3
    assert scores["antisymmetry_mean_abs"] == 0.0
    assert scores["antisymmetry_max"] == 0.0
    assert scores["identity_mean_abs"] == 0.0


def test_asymmetry_metrics_measures_the_symmetric_part_of_the_prediction() -> None:
    forward = np.array([1.0, 1.0])
    backward = np.array([1.0, -1.0])
    # r = (1+1)/2 = 1 cho dòng đầu, (1−1)/2 = 0 cho dòng sau -> trung bình 0,5.
    scores = metrics.asymmetry_metrics(forward, backward, np.zeros(2), 2.0)
    assert scores["antisymmetry_mean_abs"] == 0.5
    assert scores["antisymmetry_max"] == 1.0
    assert scores["prediction_mean_abs"] == 1.0
    assert scores["antisymmetry_over_prediction"] == 0.5
    assert scores["antisymmetry_over_label_spread"] == 0.25


def test_asymmetry_metrics_reports_the_identity_error_separately() -> None:
    scores = metrics.asymmetry_metrics(np.array([1.0, 1.0]), np.array([-1.0, -1.0]),
                                       np.array([0.2, -0.8]), 2.0)
    assert scores["identity_mean_abs"] == 0.5
    assert scores["identity_max"] == 0.8
    assert scores["identity_over_label_spread"] == 0.25


def test_asymmetry_metrics_leaves_the_ratios_undefined_without_a_scale() -> None:
    scores = metrics.asymmetry_metrics(np.zeros(2), np.zeros(2), np.zeros(2), 0.0)
    assert scores["antisymmetry_over_prediction"] is None
    assert scores["antisymmetry_over_label_spread"] is None
    assert scores["identity_over_label_spread"] is None


def test_asymmetry_metrics_on_an_empty_sample() -> None:
    scores = metrics.asymmetry_metrics(np.zeros(0), np.zeros(0), np.zeros(0), 1.0)
    assert scores["n"] == 0
    assert scores["antisymmetry_mean_abs"] is None
    assert scores["identity_mean_abs"] is None


def test_asymmetry_metrics_refuses_arrays_of_different_length() -> None:
    with pytest.raises(ValueError, match="cùng số dòng"):
        metrics.asymmetry_metrics(np.zeros(2), np.zeros(3), np.zeros(2), 1.0)


def test_format_asymmetry_prints_a_dash_for_an_undefined_value() -> None:
    lines = metrics.format_asymmetry("thu", metrics.asymmetry_metrics(
        np.zeros(0), np.zeros(0), np.zeros(0), 1.0))
    assert "n = 0" in lines[0]
    assert "-" in lines[0]

