"""Baseline tầm thường của Stage 3, và lời hứa train.py nạp được khi chưa cài torch."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest

from molopt.evaluator import train
from molopt.observations import normalize

ENDPOINTS = 4


def test_train_module_imports_torch_only_inside_functions() -> None:
    """train.py phải nạp được trên máy chưa cài torch.

    `--mode baseline` và `--mode prepare --skip-protein` không cần torch, nên module này
    không được import torch — và cũng không được import `models`, vì models.py có import
    torch ở đầu file — ở cấp module. Kiểm tra bằng AST thay vì bằng sys.modules: khi máy
    CÓ torch, một file test khác đã nạp nó trước rồi nên sys.modules không nói được gì.
    """
    tree = ast.parse(Path(train.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
            if node.module:
                imported.add(node.module.split(".")[0])
    assert "torch" not in imported
    assert "models" not in imported


def test_canonical_sequence_upper_cases_and_drops_whitespace() -> None:
    # Tokenizer ESM2 biến mỗi chữ thường thành <unk>; chuỗi BindingDB viết thường phải
    # được viết hoa trước, đúng như hash của Stage 1.
    assert train.canonical_sequence("mkt ay\nlq") == "MKTAYLQ"
    assert train.canonical_sequence("MKTAY") == "MKTAY"


def test_canonical_sequence_keeps_the_stage1_identity() -> None:
    raw = "mktay lq"
    cleaned = train.canonical_sequence(raw)
    assert normalize.sequence_hash(raw) == hashlib.sha256(cleaned.encode("ascii")).hexdigest()


def test_endpoint_means_averages_within_each_endpoint() -> None:
    endpoints = np.array([0, 0, 1, 2])
    targets = np.array([1.0, 3.0, -2.0, 0.5])
    means, seen = train.endpoint_means(endpoints, targets, ENDPOINTS)
    assert means.tolist() == [2.0, -2.0, 0.5, 0.0]
    assert seen.tolist() == [2, 1, 1, 0]


def test_endpoint_means_reports_zero_for_an_endpoint_with_no_rows() -> None:
    means, seen = train.endpoint_means(np.array([1]), np.array([4.0]), ENDPOINTS)
    # Endpoint 3 không có dòng train nào: baseline đoán 0, và seen nói rõ là 0 dòng.
    assert means[3] == 0.0
    assert seen[3] == 0


def test_baseline_zero_always_predicts_zero() -> None:
    means = np.array([2.0, -2.0, 0.5, 0.0])
    prediction = train.baseline_prediction("zero", np.array([0, 1, 2, 3]), means)
    assert prediction.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_baseline_endpoint_mean_looks_up_the_endpoint_of_each_row() -> None:
    means = np.array([2.0, -2.0, 0.5, 0.0])
    prediction = train.baseline_prediction("endpoint_mean", np.array([1, 0, 0]), means)
    assert prediction.tolist() == [-2.0, 2.0, 2.0]


def test_baseline_prediction_refuses_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="baseline lạ"):
        train.baseline_prediction("median", np.array([0]), np.zeros(ENDPOINTS))


POOL = np.array([1.0, -2.0, 0.5, 3.0])


def test_baseline_random_is_deterministic_for_a_fixed_seed() -> None:
    first = train.baseline_prediction("random", np.array([0, 1, 2]), np.zeros(ENDPOINTS),
                                      POOL, seed=11)
    second = train.baseline_prediction("random", np.array([0, 1, 2]), np.zeros(ENDPOINTS),
                                       POOL, seed=11)
    assert first.tolist() == second.tolist()


def test_baseline_random_changes_with_the_seed() -> None:
    endpoints = np.arange(50) % ENDPOINTS
    first = train.baseline_prediction("random", endpoints, np.zeros(ENDPOINTS), POOL, seed=1)
    second = train.baseline_prediction("random", endpoints, np.zeros(ENDPOINTS), POOL, seed=2)
    assert first.tolist() != second.tolist()


def test_baseline_random_only_returns_values_from_the_train_pool() -> None:
    drawn = train.baseline_prediction("random", np.arange(100) % ENDPOINTS,
                                      np.zeros(ENDPOINTS), POOL, seed=3)
    assert drawn.size == 100
    assert set(drawn.tolist()) <= set(POOL.tolist())


def test_baseline_random_ignores_the_endpoint_of_the_row() -> None:
    # Đây chính là tính chất làm nó thành mốc 50 %: dự đoán độc lập với dòng đang xét.
    first = train.baseline_prediction("random", np.array([0, 1, 2]), np.zeros(ENDPOINTS),
                                      POOL, seed=7)
    second = train.baseline_prediction("random", np.array([3, 3, 3]), np.zeros(ENDPOINTS),
                                       POOL, seed=7)
    assert first.tolist() == second.tolist()


def test_baseline_random_refuses_a_missing_pool() -> None:
    with pytest.raises(ValueError, match="phân bố nhãn train"):
        train.baseline_prediction("random", np.array([0]), np.zeros(ENDPOINTS))



def test_limit_gpu_memory_is_a_no_op_without_a_limit_or_without_gpu() -> None:
    # Thiếu khóa, khóa null, hay chạy trên CPU thì không đặt trần (và không cần torch).
    assert train.limit_gpu_memory({}, "cuda") is None
    assert train.limit_gpu_memory({"gpu_memory_limit_mib": None}, "cuda") is None
    assert train.limit_gpu_memory({"gpu_memory_limit_mib": 9216}, "cpu") is None


def test_peak_gpu_memory_is_none_on_cpu() -> None:
    assert train.peak_gpu_memory_mib("cpu") is None
