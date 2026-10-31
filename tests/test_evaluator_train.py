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


# --------------------------------------------------- đo validation giữa epoch (eval_every)

torch = pytest.importorskip("torch", reason="train_one_epoch cần torch")


class FakeArrays:
    """Đủ cho train_one_epoch: nó chỉ cần len() để chia batch."""

    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count


class FakeModel(torch.nn.Module):
    """Mô hình bé nhất chạy được một bước train thật."""

    uses_edit_branch = False

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, ligand_a, ligand_b, fragment_a, fragment_b, protein,
                endpoint):  # noqa: ANN001, D102
        return self.layer(ligand_a).squeeze(-1)


def fake_batch(arrays, rows, caches, protein_source, device, uses_edit):  # noqa: ANN001
    """Thay batch_tensors: trả về tensor đúng hình, không đụng tới cache thật."""
    size = len(rows)
    return {"ligand_a": torch.ones(size, 2), "ligand_b": None, "fragment_a": None,
            "fragment_b": None, "protein": None, "endpoint": None,
            "target": torch.zeros(size)}


def run_epoch(monkeypatch: pytest.MonkeyPatch, rows: int, batch_size: int,
              eval_every: int, on_eval=None) -> tuple[float, bool, int]:
    """Chạy một epoch giả lập; trả về đúng thứ train_one_epoch trả về."""
    monkeypatch.setattr(train, "batch_tensors", fake_batch)
    model = FakeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return train.train_one_epoch(model, FakeArrays(rows), None, None, "cpu", optimizer,
                                 torch.nn.MSELoss(), batch_size, 1.0,
                                 np.random.default_rng(0), None, eval_every, on_eval)


def test_one_epoch_reports_the_number_of_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    # 100 dòng, batch 10 -> đúng 10 bước; batch cuối ngắn vẫn tính là một bước.
    _, stopped, steps = run_epoch(monkeypatch, 95, 10, 0)
    assert (stopped, steps) == (False, 10)


def test_eval_every_zero_never_calls_the_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """eval_every = 0 phải giữ nguyên hành vi cũ: chỉ đo ở cuối epoch."""
    seen: list[int] = []
    run_epoch(monkeypatch, 100, 10, 0, lambda step: seen.append(step) or False)
    assert seen == []


def test_hook_fires_on_every_eval_every_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    _, stopped, steps = run_epoch(monkeypatch, 100, 10, 3,
                                  lambda step: seen.append(step) or False)
    assert seen == [3, 6, 9]
    assert (stopped, steps) == (False, 10)


def test_hook_returning_true_stops_inside_the_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dừng sớm phải cắt ngang epoch, không chạy nốt các bước còn lại."""
    seen: list[int] = []

    def stop_at_second(step: int) -> bool:
        seen.append(step)
        return len(seen) == 2

    _, stopped, steps = run_epoch(monkeypatch, 100, 10, 3, stop_at_second)
    assert seen == [3, 6]
    assert (stopped, steps) == (True, 6)


def test_model_goes_back_to_train_mode_after_the_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hook gọi predict() nên model bị chuyển sang eval; phải bật lại, nếu không các
    bước sau của epoch sẽ chạy sai (dropout/batchnorm tắt)."""
    modes: list[bool] = []
    monkeypatch.setattr(train, "batch_tensors", fake_batch)
    model = FakeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def hook(step: int) -> bool:
        model.eval()           # giống hệt điều predict() làm
        modes.append(model.training)
        return False

    train.train_one_epoch(model, FakeArrays(100), None, None, "cpu", optimizer,
                          torch.nn.MSELoss(), 10, 1.0, np.random.default_rng(0), None,
                          3, hook)
    assert modes == [False, False, False]
    assert model.training is True


# --------------------------------------------------- ghi đè tham số train từ dòng lệnh

import run_evaluator  # noqa: E402


def train_settings() -> dict:
    return {"learning_rate": 0.001, "weight_decay": 0.01, "epochs": 10, "loss": "huber",
            "huber_delta": 1.0, "device": "auto", "sharpness": {"enabled": False}}


def test_override_keeps_the_type_of_the_value_in_config() -> None:
    settings = train_settings()
    applied = run_evaluator.apply_overrides(settings, ["huber_delta=0.3", "epochs=3",
                                                       "loss=mse"])
    assert applied == {"huber_delta": 0.3, "epochs": 3, "loss": "mse"}
    assert settings["huber_delta"] == 0.3
    assert settings["epochs"] == 3 and isinstance(settings["epochs"], int)
    assert settings["loss"] == "mse"


def test_override_rejects_a_key_that_is_not_in_the_config() -> None:
    """Gõ sai tên khóa phải báo lỗi: nếu bỏ qua âm thầm thì cả một đợt quét tham số
    sẽ chạy với giá trị cũ mà không ai biết."""
    with pytest.raises(ValueError, match="weigth_decay"):
        run_evaluator.apply_overrides(train_settings(), ["weigth_decay=0.1"])


def test_override_rejects_writing_over_a_whole_block() -> None:
    with pytest.raises(ValueError, match="sharpness"):
        run_evaluator.apply_overrides(train_settings(), ["sharpness=1"])


def test_override_reaches_inside_a_block_with_a_dot() -> None:
    """finetune_overrides.batch_size là nút VRAM quan trọng nhất của cấu hình finetune,
    phải chỉnh được từ dòng lệnh mà không sửa config."""
    settings = train_settings()
    settings["finetune_overrides"] = {"batch_size": 64, "max_residues": 512}
    applied = run_evaluator.apply_overrides(settings, ["finetune_overrides.batch_size=16"])
    assert applied == {"finetune_overrides.batch_size": 16}
    assert settings["finetune_overrides"] == {"batch_size": 16, "max_residues": 512}


def test_override_rejects_an_unknown_key_inside_a_block() -> None:
    settings = train_settings()
    settings["finetune_overrides"] = {"batch_size": 64}
    with pytest.raises(ValueError, match="finetune_overrides.batch_sze"):
        run_evaluator.apply_overrides(settings, ["finetune_overrides.batch_sze=16"])


def test_override_rejects_an_unknown_block() -> None:
    with pytest.raises(ValueError, match="khong_co"):
        run_evaluator.apply_overrides(train_settings(), ["khong_co.batch_size=16"])


def test_override_needs_an_equals_sign() -> None:
    with pytest.raises(ValueError, match="khóa=giá trị"):
        run_evaluator.apply_overrides(train_settings(), ["huber_delta 0.3"])


def test_override_of_nothing_changes_nothing() -> None:
    settings = train_settings()
    assert run_evaluator.apply_overrides(settings, []) == {}
    assert settings == train_settings()


# ------------------------------------------- batch của finetune (VRAM) vs batch frozen

BATCH_SETTINGS = {"batch_size": 512, "eval_batch_size": 4096,
                  "finetune_overrides": {"batch_size": 64, "eval_batch_size": 128}}


def test_frozen_keeps_the_configured_batch_sizes() -> None:
    assert train.effective_batch_sizes(BATCH_SETTINGS, "frozen", False) == (512, 4096)


def test_finetune_uses_its_own_batch_sizes() -> None:
    """Cả batch train LẪN batch dự đoán phải nhỏ lại: lượt dự đoán vẫn chạy ESM2, nên
    4096 dòng có thể chạm tới hàng nghìn protein phân biệt một lúc và hết VRAM."""
    assert train.effective_batch_sizes(BATCH_SETTINGS, "finetune", False) == (64, 128)


def test_smoke_never_uses_a_bigger_batch_than_the_real_finetune_run() -> None:
    """Lỗi cũ: smoke cắt batch xuống 128 nhưng BỎ QUA finetune_overrides, nên smoke
    finetune chạy batch 128 - tốn VRAM gấp đôi lần train thật (64) và hết bộ nhớ ở
    chính bước lẽ ra để kiểm tra an toàn."""
    smoke_batch, _ = train.effective_batch_sizes(BATCH_SETTINGS, "finetune", True)
    train_batch, _ = train.effective_batch_sizes(BATCH_SETTINGS, "finetune", False)
    assert smoke_batch == 64
    assert smoke_batch <= train_batch


def test_smoke_caps_the_frozen_batch() -> None:
    assert train.effective_batch_sizes(BATCH_SETTINGS, "frozen", True) == (128, 4096)


def test_finetune_falls_back_when_overrides_are_missing() -> None:
    settings = {"batch_size": 512, "eval_batch_size": 4096, "finetune_overrides": {}}
    assert train.effective_batch_sizes(settings, "finetune", False) == (512, 4096)
