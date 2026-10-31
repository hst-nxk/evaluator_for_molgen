"""Số hạng phạt chuẩn gradient đầu ra (IGNITE). Cần torch; bỏ qua cả file nếu chưa cài."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="IGNITE cần torch")

from molopt.evaluator import sharpness  # noqa: E402

SETTINGS: dict[str, Any] = {
    "enabled": False,
    "lambda_init": 0.01,
    "lambda_learning_rate": 0.001,
    "rho": 0.05,
    "finite_difference_radius": 0.05,
    "epsilon": 0.1,
}


def toy_linear() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Linear(3, 1)


def toy_mlp() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.ReLU(),
                               torch.nn.Linear(6, 1))


def toy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return torch.rand(8, 3, generator=generator), torch.rand(8, generator=generator)


def plain_gradient(build: Any) -> list[torch.Tensor]:
    """Gradient của loss.backward() thường, để đối chiếu."""
    model = build()
    features, target = toy_batch()
    torch.nn.functional.mse_loss(model(features).squeeze(-1), target).backward()
    return [parameter.grad.detach().clone() for parameter in model.parameters()]


def run_one_step(settings: dict[str, Any], build: Any = toy_mlp
                 ) -> tuple[sharpness.SharpnessPenalty, torch.nn.Module,
                            list[torch.Tensor]]:
    """Một bước apply() trên model nhỏ; trả về (penalty, model, bản sao tham số ban đầu)."""
    model = build()
    features, target = toy_batch()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    penalty = sharpness.SharpnessPenalty(settings)
    prediction = model(features).squeeze(-1)
    loss = torch.nn.functional.mse_loss(prediction, target)
    penalty.apply(loss, prediction, list(model.parameters()),
                  lambda: model(features).squeeze(-1).mean())
    return penalty, model, before


def test_is_enabled_reads_config_when_no_override() -> None:
    assert sharpness.is_enabled({"enabled": True}) is True
    assert sharpness.is_enabled({"enabled": False}) is False


def test_is_enabled_lets_the_command_line_win() -> None:
    # Ablation có/không số hạng phạt phải chạy được mà không sửa config.
    assert sharpness.is_enabled({"enabled": False}, "on") is True
    assert sharpness.is_enabled({"enabled": True}, "off") is False


def test_apply_writes_a_gradient_on_every_parameter() -> None:
    _, model, _ = run_one_step(dict(SETTINGS))
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_apply_puts_the_parameters_back_where_they_were() -> None:
    # Sai phân hữu hạn phải làm nhiễu tham số rồi TRẢ LẠI; nếu không, model bị bỏ lại ở
    # điểm nhiễu và bước optimizer sau đó sai hoàn toàn.
    _, model, before = run_one_step(dict(SETTINGS))
    for parameter, original in zip(model.parameters(), before):
        assert torch.allclose(parameter.detach(), original, atol=1e-7)


def test_lambda_zero_gives_exactly_the_plain_gradient() -> None:
    # lambda = 0 nghĩa là không phạt: gradient phải trùng loss.backward() thường.
    _, model, _ = run_one_step(dict(SETTINGS, lambda_init=0.0))
    with_penalty = [parameter.grad.detach().clone() for parameter in model.parameters()]
    for mine, reference in zip(with_penalty, plain_gradient(toy_mlp)):
        assert torch.allclose(mine, reference, atol=1e-6)


def test_lambda_grows_when_the_gradient_norm_is_above_the_threshold() -> None:
    settings = dict(SETTINGS, epsilon=0.0, lambda_learning_rate=1.0)
    penalty, _, _ = run_one_step(settings)
    assert penalty.lambda_value > SETTINGS["lambda_init"]


def test_lambda_never_goes_below_zero() -> None:
    # Dạng Lagrange của bài báo cần lambda > 0; vi phạm âm lớn không được kéo nó xuống âm.
    settings = dict(SETTINGS, epsilon=1000.0, lambda_learning_rate=1.0)
    penalty, _, _ = run_one_step(settings)
    assert penalty.lambda_value == 0.0


def test_state_reports_the_sharpness_estimate() -> None:
    penalty, _, _ = run_one_step(dict(SETTINGS))
    state = penalty.state()
    assert state["output_gradient_norm"] > 0.0
    # Bài báo xấp xỉ độ sắc bằng rho * ||grad đầu ra|| (công thức 10).
    assert state["sharpness_estimate"] == pytest.approx(
        SETTINGS["rho"] * state["output_gradient_norm"])


def test_penalty_changes_the_gradient_when_lambda_is_large() -> None:
    _, model, _ = run_one_step(dict(SETTINGS, lambda_init=10.0))
    with_penalty = [parameter.grad.detach().clone() for parameter in model.parameters()]
    assert not all(torch.allclose(mine, reference, atol=1e-6)
                   for mine, reference in zip(with_penalty, plain_gradient(toy_mlp)))


def test_penalty_is_zero_on_a_linear_model() -> None:
    # Sai phân hữu hạn của bài báo thay cho tích Hessian-vector; mô hình tuyến tính có
    # Hessian bằng 0 nên số hạng phạt phải triệt tiêu đúng bằng 0, dù lambda lớn.
    # Test này ghim rằng cách xấp xỉ được cài đúng chứ không phải trùng hợp.
    _, model, _ = run_one_step(dict(SETTINGS, lambda_init=10.0), toy_linear)
    with_penalty = [parameter.grad.detach().clone() for parameter in model.parameters()]
    for mine, reference in zip(with_penalty, plain_gradient(toy_linear)):
        assert torch.allclose(mine, reference, atol=1e-6)


def test_radius_must_be_positive() -> None:
    with pytest.raises(ValueError, match="finite_difference_radius"):
        sharpness.SharpnessPenalty(dict(SETTINGS, finite_difference_radius=0.0))


def test_rho_must_be_positive() -> None:
    with pytest.raises(ValueError, match="rho"):
        sharpness.SharpnessPenalty(dict(SETTINGS, rho=0.0))


def test_apply_refuses_a_model_with_nothing_to_train() -> None:
    model = toy_mlp()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    features, target = toy_batch()
    penalty = sharpness.SharpnessPenalty(dict(SETTINGS))
    prediction = model(features).squeeze(-1)
    with pytest.raises(ValueError, match="gradient"):
        penalty.apply(torch.nn.functional.mse_loss(prediction, target), prediction,
                      list(model.parameters()), lambda: prediction.mean())


def test_penalty_adds_to_an_existing_gradient_when_accumulating() -> None:
    """Lỗi âm thầm nếu quên: apply() vốn GHI ĐÈ .grad, nên cộng dồn qua nhiều batch con
    sẽ xóa sạch gradient của những batch trước mà không báo gì."""
    model = toy_mlp()
    inputs, targets = toy_batch()
    penalty = sharpness.SharpnessPenalty(SETTINGS)
    parameters = list(model.parameters())

    prediction = model(inputs).squeeze(-1)
    loss = torch.nn.functional.mse_loss(prediction, targets)
    penalty.apply(loss, prediction, parameters,
                  lambda: model(inputs).squeeze(-1).mean(), 1.0, False)
    once = [p.grad.clone() for p in parameters]

    prediction = model(inputs).squeeze(-1)
    loss = torch.nn.functional.mse_loss(prediction, targets)
    penalty.apply(loss, prediction, parameters,
                  lambda: model(inputs).squeeze(-1).mean(), 1.0, True)
    for first, parameter in zip(once, parameters):
        assert torch.allclose(parameter.grad, first * 2, atol=1e-5)


def test_penalty_weight_scales_the_whole_gradient() -> None:
    """Trọng số phải nhân vào gradient TỔNG, kể cả số hạng phạt - không thể thay bằng
    cách chia mất mát, vì số hạng phạt tính từ prediction.mean()."""
    def gradient(weight: float) -> list[Any]:
        model = toy_mlp()
        inputs, targets = toy_batch()
        penalty = sharpness.SharpnessPenalty(dict(SETTINGS, lambda_init=5.0))
        parameters = list(model.parameters())
        prediction = model(inputs).squeeze(-1)
        loss = torch.nn.functional.mse_loss(prediction, targets)
        penalty.apply(loss, prediction, parameters,
                      lambda: model(inputs).squeeze(-1).mean(), weight, False)
        return [p.grad.clone() for p in parameters]

    for half, full in zip(gradient(0.5), gradient(1.0)):
        assert torch.allclose(half * 2, full, atol=1e-5)
