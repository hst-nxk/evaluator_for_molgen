"""Baseline tầm thường của Stage 3, và lời hứa train.py nạp được khi chưa cài torch."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest

from molopt.evaluator import data, train
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


def test_override_reaches_the_model_block() -> None:
    """dropout la nut chong khop qua du lieu nhung nam o stage3.model, khong phai
    stage3.train - --set phai voi toi duoc no."""
    settings = train_settings()
    model = {"dropout": 0.1, "ligand_out": 256}
    applied = run_evaluator.apply_overrides(settings, ["model.dropout=0.3"], model)
    assert applied == {"model.dropout": 0.3}
    assert model["dropout"] == 0.3
    assert "model.dropout" not in settings


def test_override_rejects_an_unknown_key_in_the_model_block() -> None:
    with pytest.raises(ValueError, match="dropuot"):
        run_evaluator.apply_overrides(train_settings(), ["model.dropuot=0.3"],
                                      {"dropout": 0.1})


def test_override_rejects_the_model_prefix_with_no_key() -> None:
    with pytest.raises(ValueError, match="thiếu tên tham số"):
        run_evaluator.apply_overrides(train_settings(), ["model.=0.3"], {"dropout": 0.1})


def encoder_blocks() -> dict:
    """Hai khối CLI truyền thêm: protein_encoder và paths, giống config.json."""
    return {"protein_encoder": {"model": "facebook/esm2_t12_35M_UR50D",
                                "embedding_dim": 480},
            "paths": {"cache_dir": "work/evaluator_cache_v4", "runs_dir": "runs/evaluator"}}


def test_override_switches_the_protein_encoder_and_its_cache() -> None:
    """Thử ESM2 650M phải làm được bằng --set, không sửa config.json, và phải trỏ
    cache sang thư mục khác để không đè cache của model đang dùng."""
    blocks = encoder_blocks()
    settings = train_settings()
    applied = run_evaluator.apply_overrides(
        settings, ["protein_encoder.model=facebook/esm2_t33_650M_UR50D",
                   "protein_encoder.embedding_dim=1280",
                   "paths.cache_dir=work/evaluator_cache_v4_esm650m"], {}, blocks)
    assert applied == {"protein_encoder.model": "facebook/esm2_t33_650M_UR50D",
                       "protein_encoder.embedding_dim": 1280,
                       "paths.cache_dir": "work/evaluator_cache_v4_esm650m"}
    assert blocks["protein_encoder"]["model"] == "facebook/esm2_t33_650M_UR50D"
    assert isinstance(blocks["protein_encoder"]["embedding_dim"], int)
    assert blocks["paths"] == {"cache_dir": "work/evaluator_cache_v4_esm650m",
                               "runs_dir": "runs/evaluator"}
    assert settings == train_settings()


def test_override_rejects_an_unknown_key_in_the_encoder_block() -> None:
    with pytest.raises(ValueError, match="protein_encoder.embeding_dim"):
        run_evaluator.apply_overrides(train_settings(), ["protein_encoder.embeding_dim=1280"],
                                      {}, encoder_blocks())


def test_override_of_the_encoder_needs_the_block_to_be_passed() -> None:
    """Không truyền khối thì khóa đó không được âm thầm ghi vào stage3.train."""
    settings = train_settings()
    with pytest.raises(ValueError, match="protein_encoder"):
        run_evaluator.apply_overrides(settings, ["protein_encoder.model=x"])
    assert settings == train_settings()


def test_override_accepts_true_and_false_for_a_boolean_key() -> None:
    settings = {**train_settings(), 'min_edit_rows_control': False}
    assert run_evaluator.apply_overrides(
        settings, ['min_edit_rows_control=true']) == {'min_edit_rows_control': True}
    assert settings['min_edit_rows_control'] is True
    assert run_evaluator.apply_overrides(
        settings, ['min_edit_rows_control=off']) == {'min_edit_rows_control': False}
    assert settings['min_edit_rows_control'] is False


def test_override_refuses_a_word_it_cannot_read_as_a_boolean() -> None:
    """bool("khong") là True, nên viết sai phải báo lỗi chứ không được BẬT cờ."""
    settings = {**train_settings(), 'min_edit_rows_control': False}
    with pytest.raises(ValueError, match='true/false'):
        run_evaluator.apply_overrides(settings, ['min_edit_rows_control=khong'])
    assert settings['min_edit_rows_control'] is False


# ------------------------------------------------------------ ghi đè list số (11/11/2026)
# Vì sao cần: các khóa chiều ẩn đều là list, và thí nghiệm "một lớp Linear không bias trên
# vector hiệu" chính là antisymmetric_hidden rỗng. Không có đường này thì thí nghiệm đó chỉ
# chạy được bằng cách sửa tay config.json, mà sửa tay thì mọi lần chạy trong cùng một
# script đều đổi theo — tức không tách được nó khỏi lần chạy bên cạnh.


def test_override_sets_a_list_of_hidden_sizes() -> None:
    model = {'antisymmetric_hidden': [256, 128], 'head_hidden': [8]}
    applied = run_evaluator.apply_overrides(
        train_settings(), ['model.antisymmetric_hidden=[64, 32]'], model)
    assert applied == {'model.antisymmetric_hidden': [64, 32]}
    assert model['antisymmetric_hidden'] == [64, 32]


def test_override_sets_an_empty_list() -> None:
    """[] là chính thí nghiệm cần: bỏ hết lớp ẩn, còn một lớp Linear không bias."""
    model = {'antisymmetric_hidden': [256, 128]}
    assert run_evaluator.apply_overrides(
        train_settings(), ['model.antisymmetric_hidden=[]'], model) == {
            'model.antisymmetric_hidden': []}
    assert model['antisymmetric_hidden'] == []


def test_override_keeps_whole_numbers_as_int() -> None:
    """nn.Linear cần int; 128.0 sẽ báo lỗi ở tận lúc dựng model."""
    model = {'antisymmetric_hidden': [256]}
    run_evaluator.apply_overrides(train_settings(), ['model.antisymmetric_hidden=[128]'],
                                  model)
    assert model['antisymmetric_hidden'] == [128]
    assert all(isinstance(size, int) for size in model['antisymmetric_hidden'])


def test_override_refuses_a_list_value_without_brackets() -> None:
    model = {'antisymmetric_hidden': [256, 128]}
    with pytest.raises(ValueError, match='ngoặc vuông'):
        run_evaluator.apply_overrides(train_settings(),
                                      ['model.antisymmetric_hidden=256'], model)
    assert model['antisymmetric_hidden'] == [256, 128]


@pytest.mark.parametrize('raw', ('[1,,2]', '[abc]', '[1, x]'))
def test_override_refuses_a_list_element_that_is_not_a_number(raw: str) -> None:
    model = {'antisymmetric_hidden': [256]}
    with pytest.raises(ValueError, match='antisymmetric_hidden'):
        run_evaluator.apply_overrides(train_settings(),
                                      [f'model.antisymmetric_hidden={raw}'], model)
    assert model['antisymmetric_hidden'] == [256]


def test_override_refuses_a_list_of_names() -> None:
    """`heads` và `configurations` là danh sách tên: ghi đè chúng làm manifest nói sai."""
    settings = train_settings()
    settings['heads'] = ['direct', 'twin']
    with pytest.raises(ValueError, match='list số'):
        run_evaluator.apply_overrides(settings, ['heads=[1]'])
    assert settings['heads'] == ['direct', 'twin']


# ------------------------------------------------------------ cộng dồn gradient


def run_accum(monkeypatch: pytest.MonkeyPatch, rows: int, batch_size: int,
              accumulate: int, eval_every: int = 0, on_eval=None):
    monkeypatch.setattr(train, "batch_tensors", fake_batch)
    model = FakeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return train.train_one_epoch(model, FakeArrays(rows), None, None, "cpu", optimizer,
                                 torch.nn.MSELoss(), batch_size, 1.0,
                                 np.random.default_rng(0), None, eval_every, on_eval,
                                 False, accumulate)


def test_accumulation_counts_optimizer_steps_not_micro_batches(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """100 dòng, batch 10 = 10 batch con; gộp 5 -> chỉ 2 lần cập nhật tham số."""
    _, _, steps = run_accum(monkeypatch, 100, 10, 5)
    assert steps == 2


def test_accumulation_of_one_is_the_old_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, steps = run_accum(monkeypatch, 100, 10, 1)
    assert steps == 10


def test_accumulation_still_steps_on_the_leftover_batches(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """10 batch con, gộp 4 -> 2 lần đủ nhóm cộng 1 lần cho 2 batch lẻ cuối."""
    _, _, steps = run_accum(monkeypatch, 100, 10, 4)
    assert steps == 3


def test_accumulation_gives_the_mean_gradient_of_the_group(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Gộp N batch phải cho gradient TRUNG BÌNH, không phải tổng: quên chia thì bước đi
    lớn gấp N lần, tức learning rate bị đổi ngầm."""
    monkeypatch.setattr(train, "batch_tensors", fake_batch)

    def gradient_at_first_step(accumulate: int) -> torch.Tensor:
        model = FakeModel()
        captured: list[torch.Tensor] = []

        class RecordingSGD(torch.optim.SGD):
            def step(self, *args, **kwargs):  # noqa: ANN002, ANN003, D102
                if not captured:
                    captured.append(model.layer.weight.grad.clone())
                return super().step(*args, **kwargs)

        optimizer = RecordingSGD(model.parameters(), lr=0.0)
        train.train_one_epoch(model, FakeArrays(40), None, None, "cpu", optimizer,
                              torch.nn.MSELoss(), 10, 0.0, None, None, 0, None,
                              False, accumulate)
        return captured[0]

    # Mọi batch giả giống hệt nhau, nên trung bình của 4 batch phải bằng 1 batch.
    assert torch.allclose(gradient_at_first_step(4), gradient_at_first_step(1), atol=1e-6)


def test_accumulation_evaluates_on_optimizer_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    run_accum(monkeypatch, 200, 10, 4, 2, lambda step: seen.append(step) or False)
    # 20 batch con / gộp 4 = 5 lần cập nhật; đo sau mỗi 2 lần cập nhật.
    assert seen == [2, 4]


def test_selection_by_mae_keeps_the_old_order() -> None:
    better = {"regression": {"mae": 0.58}, "ranking": {"pairwise_accuracy_macro": 0.50}}
    worse = {"regression": {"mae": 0.59}, "ranking": {"pairwise_accuracy_macro": 0.56}}
    assert (train.selection_score(better, train.SELECT_MAE)
            < train.selection_score(worse, train.SELECT_MAE))


def test_selection_by_pairwise_prefers_the_higher_accuracy() -> None:
    # Ca 06/11/2026: mô hình sụp có MAE ngang bình thường nhưng đúng chiều thấp hẳn.
    collapsed = {"regression": {"mae": 0.5879}, "ranking": {"pairwise_accuracy_macro": 0.168}}
    healthy = {"regression": {"mae": 0.5880}, "ranking": {"pairwise_accuracy_macro": 0.557}}
    metric = train.SELECT_PAIRWISE_MACRO
    assert train.selection_score(healthy, metric) < train.selection_score(collapsed, metric)


def test_selection_by_pairwise_skips_a_point_with_no_ranking() -> None:
    no_queries = {"regression": {"mae": 0.5}, "ranking": {"pairwise_accuracy_macro": None}}
    assert train.selection_score(no_queries, train.SELECT_PAIRWISE_MACRO) is None
    assert train.selection_score({"regression": {"mae": 0.5}},
                                 train.SELECT_PAIRWISE_MACRO) is None


def test_config_selection_metric_is_known_and_defaults_to_mae() -> None:
    import json

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert config["stage3"]["train"]["selection_metric"] == train.SELECT_MAE
    assert train.SELECTION_METRICS == (train.SELECT_MAE, train.SELECT_PAIRWISE_MACRO)


def test_ligand_features_switch_only_allows_the_two_known_modes() -> None:
    from molopt.evaluator import models

    assert models.LIGAND_FEATURES == (models.LIGAND_FINGERPRINT, models.LIGAND_EMBEDDING)
    assert models.LIGAND_FINGERPRINT == "fingerprint"


def test_config_keeps_fingerprint_as_the_default_ligand_input() -> None:
    """Đổi mặc định là mọi số đã đo trước 07/11/2026 không so được nữa."""
    import json

    root = Path(__file__).resolve().parents[1]
    stage3 = json.loads((root / "config.json").read_text(encoding="utf-8"))["stage3"]
    assert stage3["model"]["ligand_features"] == "fingerprint"
    # Chiều embedding phải để trống trong config: nó được ĐO từ model lúc nạp.
    assert stage3["molecule_encoder"]["embedding_dim"] is None
    assert stage3["files"]["molecule_embeddings"].endswith(".parquet")


def test_the_molecule_encoder_name_must_be_declared_in_allowed_models() -> None:
    import json

    root = Path(__file__).resolve().parents[1]
    encoder = json.loads((root / "config.json").read_text(
        encoding="utf-8"))["stage3"]["molecule_encoder"]
    assert encoder["model"] in encoder["allowed_models"]


def test_batch_uses_the_embedding_matrix_when_the_cache_is_loaded() -> None:
    """Ở chế độ embedding, batch phải lấy thẳng vector chứ không giải nén bit."""
    torch = pytest.importorskip("torch", reason="batch_tensors cần torch")

    bits, dimension, molecules = 16, 4, 5
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(molecules, dimension)).astype(np.float32)

    class FakeCaches:
        molecule_packed = np.packbits(
            rng.integers(0, 2, (molecules, bits)).astype(np.uint8), axis=1)
        fragment_packed = np.packbits(
            rng.integers(0, 2, (3, bits)).astype(np.uint8), axis=1)
        ligand_bits = dimension
        fragment_bits = bits
        molecule_vectors = vectors

    arrays = data.PairArrays(
        molecule_a=np.array([0, 2]), molecule_b=np.array([1, 3]),
        fragment_a=np.array([0, 1]), fragment_b=np.array([1, 2]),
        protein=np.array([0, 0]), endpoint=np.array([0, 0]),
        target=np.array([0.5, -0.5], dtype=np.float32), slices={})
    rows = np.array([0, 1])
    batch = train.batch_tensors(arrays, rows, FakeCaches(), None, "cpu", uses_edit=True)

    assert batch["ligand_a"].shape == (2, dimension)
    assert np.allclose(batch["ligand_a"].numpy(), vectors[[0, 2]])
    assert np.allclose(batch["ligand_b"].numpy(), vectors[[1, 3]])
    # Nhánh edit vẫn là fingerprint ở cả hai chế độ, nên vẫn ra đúng số bit.
    assert batch["fragment_a"].shape == (2, bits)


def test_batch_still_unpacks_bits_when_there_is_no_embedding_cache() -> None:
    torch = pytest.importorskip("torch", reason="batch_tensors cần torch")

    bits = 16
    rng = np.random.default_rng(1)

    class FakeCaches:
        molecule_packed = np.packbits(rng.integers(0, 2, (4, bits)).astype(np.uint8), axis=1)
        fragment_packed = np.packbits(rng.integers(0, 2, (3, bits)).astype(np.uint8), axis=1)
        ligand_bits = bits
        fragment_bits = bits
        molecule_vectors = None

    arrays = data.PairArrays(
        molecule_a=np.array([0]), molecule_b=np.array([1]),
        fragment_a=np.array([0]), fragment_b=np.array([1]),
        protein=np.array([0]), endpoint=np.array([0]),
        target=np.array([0.0], dtype=np.float32), slices={})
    batch = train.batch_tensors(arrays, np.array([0]), FakeCaches(), None, "cpu",
                                uses_edit=True)
    assert batch["ligand_a"].shape == (1, bits)
    assert set(np.unique(batch["ligand_a"].numpy())) <= {0.0, 1.0}


# ------------------------------------------ chuẩn hóa nhãn (việc 14, họp 09/11/2026)
# Anh Tiến chốt ở dòng 219 biên bản: mọi tham số chuẩn hóa CHỈ tính trên train rồi áp công
# thức sang validation và test. Các test dưới đây ghim điều đó, và ghim thêm một điều
# không có trong biên bản: chuẩn hóa về [0, 1] không tương thích với tính phản đối xứng.

TRAIN_LABELS = np.array([-2.0, -0.5, 0.0, 1.0, 3.5], dtype=np.float32)


def test_label_normalization_none_is_the_identity() -> None:
    scaler = train.fit_label_scaler(train.LABEL_NONE, TRAIN_LABELS)
    assert scaler.active is False
    assert scaler.forward(TRAIN_LABELS) is TRAIN_LABELS
    assert scaler.inverse(TRAIN_LABELS) is TRAIN_LABELS
    assert scaler.preserves_antisymmetry is True


@pytest.mark.parametrize("name", (train.LABEL_SCALE, train.LABEL_ZSCORE,
                                  train.LABEL_MINMAX))
def test_every_label_normalization_round_trips(name: str) -> None:
    """inverse(forward(y)) phải trả về đúng y: nếu không thì mọi MAE báo cáo đều sai thang."""
    scaler = train.fit_label_scaler(name, TRAIN_LABELS)
    back = scaler.inverse(scaler.forward(TRAIN_LABELS))
    assert np.allclose(back, TRAIN_LABELS, atol=1e-5)


def test_scale_normalization_divides_by_the_train_standard_deviation() -> None:
    scaler = train.fit_label_scaler(train.LABEL_SCALE, TRAIN_LABELS)
    assert scaler.shift == 0.0
    assert scaler.scale == pytest.approx(float(TRAIN_LABELS.astype(np.float64).std()))
    # Không trừ trung bình nên phép biến đổi là hàm LẺ: đây là lý do nó dùng được cùng
    # augment đảo chiều và cùng kiến trúc phản đối xứng.
    assert scaler.preserves_antisymmetry is True
    assert scaler.forward(-TRAIN_LABELS) == pytest.approx(-scaler.forward(TRAIN_LABELS))


def test_zscore_and_minmax_do_not_preserve_antisymmetry() -> None:
    for name in (train.LABEL_ZSCORE, train.LABEL_MINMAX):
        scaler = train.fit_label_scaler(name, TRAIN_LABELS)
        assert scaler.shift != 0.0
        assert scaler.preserves_antisymmetry is False
        assert not np.allclose(scaler.forward(-TRAIN_LABELS),
                               -scaler.forward(TRAIN_LABELS))


def test_minmax_puts_the_train_labels_exactly_in_the_unit_interval() -> None:
    scaler = train.fit_label_scaler(train.LABEL_MINMAX, TRAIN_LABELS)
    values = scaler.forward(TRAIN_LABELS)
    assert values.min() == pytest.approx(0.0)
    assert values.max() == pytest.approx(1.0)


def test_label_normalization_refuses_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="label_normalization"):
        train.fit_label_scaler("standardise", TRAIN_LABELS)


def test_label_normalization_refuses_labels_with_no_spread() -> None:
    with pytest.raises(ValueError, match="biến thiên"):
        train.fit_label_scaler(train.LABEL_ZSCORE, np.full(5, 1.5, dtype=np.float32))


def test_a_shifting_normalization_is_refused_with_reverse_augmentation() -> None:
    scaler = train.fit_label_scaler(train.LABEL_MINMAX, TRAIN_LABELS)
    with pytest.raises(ValueError, match="augment_reverse"):
        train.check_label_scaler(scaler, augment=True, antisymmetric=False)


def test_a_shifting_normalization_is_refused_with_an_antisymmetric_head() -> None:
    scaler = train.fit_label_scaler(train.LABEL_ZSCORE, TRAIN_LABELS)
    with pytest.raises(ValueError, match="phản đối xứng"):
        train.check_label_scaler(scaler, augment=False, antisymmetric=True)


def test_scale_normalization_is_allowed_everywhere() -> None:
    scaler = train.fit_label_scaler(train.LABEL_SCALE, TRAIN_LABELS)
    train.check_label_scaler(scaler, augment=True, antisymmetric=True)


def test_the_label_scaler_survives_a_checkpoint_round_trip() -> None:
    """Checkpoint phải mang theo thang đo của chính nó, nếu không eval quy đổi sai thang."""
    scaler = train.fit_label_scaler(train.LABEL_SCALE, TRAIN_LABELS)
    restored = train.LabelScaler.from_state(scaler.state())
    assert restored == scaler


def test_an_old_checkpoint_without_a_scaler_reads_as_the_identity() -> None:
    # Mọi checkpoint train trước ngày 10/11/2026 không có khóa này.
    assert train.LabelScaler.from_state(None).active is False
    assert train.LabelScaler.from_state({}).active is False


def test_the_config_default_keeps_every_earlier_number_comparable() -> None:
    """Mặc định phải là 'none' và augment phải TẮT, nếu không mọi số đã đo thành vô nghĩa."""
    import json

    root = Path(__file__).resolve().parents[1]
    stage3 = json.loads((root / "config.json").read_text(encoding="utf-8"))["stage3"]
    assert stage3["train"]["label_normalization"] == train.LABEL_NONE
    assert stage3["train"]["augment_reverse"] is False
    assert stage3["inputs"]["train_subset"] == ""
