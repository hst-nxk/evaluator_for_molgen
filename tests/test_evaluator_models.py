"""Kiến trúc evaluator của Stage 3. Cần torch; bỏ qua cả file nếu máy chưa cài."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="Stage 3 models cần torch")

from molopt.evaluator import models  # noqa: E402

LIGAND_BITS = 32
FRAGMENT_BITS = 16
PROTEIN_DIM = 48
ENDPOINTS = 4
BATCH = 5

SETTINGS: dict[str, Any] = {
    "ligand_hidden": [16], "ligand_out": 8,
    "edit_hidden": [8], "edit_out": 4,
    "protein_out": 6, "endpoint_dim": 3,
    "head_hidden": [8], "dropout": 0.0,
}


def inputs(protein: bool = True) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(0)
    return {
        "ligand_a": torch.rand(BATCH, LIGAND_BITS, generator=generator),
        "ligand_b": torch.rand(BATCH, LIGAND_BITS, generator=generator),
        "fragment_a": torch.rand(BATCH, FRAGMENT_BITS, generator=generator),
        "fragment_b": torch.rand(BATCH, FRAGMENT_BITS, generator=generator),
        "protein": torch.rand(BATCH, PROTEIN_DIM, generator=generator) if protein else None,
        "endpoint": torch.zeros(BATCH, dtype=torch.long),
    }


def build(head: str, protein_mode: str) -> models.EvaluatorModel:
    # Trọng số khởi tạo cố định: kết quả test không phụ thuộc trạng thái ngẫu nhiên mà các
    # test chạy trước để lại.
    torch.manual_seed(0)
    dimension = None if protein_mode == models.PROTEIN_NONE else PROTEIN_DIM
    return models.EvaluatorModel(SETTINGS, LIGAND_BITS, FRAGMENT_BITS, dimension,
                                 ENDPOINTS, head, protein_mode)


def test_direct_head_returns_one_number_per_row() -> None:
    model = build(models.HEAD_DIRECT, models.PROTEIN_FROZEN)
    output = model(**inputs())
    assert output.shape == (BATCH,)


def test_direct_head_uses_the_edit_branch() -> None:
    assert build(models.HEAD_DIRECT, models.PROTEIN_FROZEN).uses_edit_branch is True


def test_twin_head_drops_the_edit_branch() -> None:
    # Một phép sửa là thuộc tính của cặp, không của một phân tử, nên twin không có chỗ
    # đặt nhánh edit. Đây là bất đối xứng đã biết của ablation, không phải lỗi.
    assert build(models.HEAD_TWIN, models.PROTEIN_FROZEN).uses_edit_branch is False


def test_twin_head_gives_zero_when_both_sides_are_the_same_molecule() -> None:
    model = build(models.HEAD_TWIN, models.PROTEIN_FROZEN).eval()
    batch = inputs()
    batch["ligand_b"] = batch["ligand_a"]
    output = model(**batch)
    assert torch.allclose(output, torch.zeros(BATCH), atol=1e-6)


def test_twin_head_is_antisymmetric_when_the_two_sides_swap() -> None:
    model = build(models.HEAD_TWIN, models.PROTEIN_FROZEN).eval()
    batch = inputs()
    forward = model(**batch)
    swapped = dict(batch)
    swapped["ligand_a"], swapped["ligand_b"] = batch["ligand_b"], batch["ligand_a"]
    assert torch.allclose(forward, -model(**swapped), atol=1e-6)


def test_twin_head_runs_without_the_fragment_inputs() -> None:
    model = build(models.HEAD_TWIN, models.PROTEIN_FROZEN).eval()
    batch = inputs()
    batch["fragment_a"] = batch["fragment_b"] = None
    assert model(**batch).shape == (BATCH,)


def test_direct_head_refuses_missing_fragment_inputs() -> None:
    model = build(models.HEAD_DIRECT, models.PROTEIN_FROZEN)
    batch = inputs()
    batch["fragment_a"] = None
    with pytest.raises(ValueError, match="variable_a"):
        model(**batch)


def test_no_protein_configuration_runs_without_a_protein_vector() -> None:
    model = build(models.HEAD_DIRECT, models.PROTEIN_NONE)
    assert model(**inputs(protein=False)).shape == (BATCH,)


def test_no_protein_configuration_refuses_a_protein_vector() -> None:
    model = build(models.HEAD_DIRECT, models.PROTEIN_NONE)
    with pytest.raises(ValueError, match="none"):
        model(**inputs(protein=True))


def test_protein_branch_refuses_a_missing_protein_vector() -> None:
    model = build(models.HEAD_DIRECT, models.PROTEIN_FROZEN)
    with pytest.raises(ValueError, match="protein"):
        model(**inputs(protein=False))


def test_the_same_encoder_weights_are_used_for_both_sides() -> None:
    # Một encoder dùng chung: cùng fingerprint ở hai phía phải cho cùng vector ẩn.
    model = build(models.HEAD_DIRECT, models.PROTEIN_FROZEN).eval()
    batch = inputs()
    with torch.no_grad():
        assert torch.allclose(model.ligand_encoder(batch["ligand_a"]),
                              model.ligand_encoder(batch["ligand_a"]))
    names = [name for name, _ in model.named_parameters() if "ligand_encoder" in name]
    assert names and not any("ligand_encoder_b" in name for name in names)


def test_model_refuses_an_unknown_head() -> None:
    with pytest.raises(ValueError, match="head"):
        build("ranking", models.PROTEIN_FROZEN)


def test_model_refuses_an_unknown_protein_mode() -> None:
    with pytest.raises(ValueError, match="protein_mode"):
        build(models.HEAD_DIRECT, "half-frozen")


def test_model_refuses_a_missing_protein_dimension() -> None:
    with pytest.raises(ValueError, match="protein_dim"):
        models.EvaluatorModel(SETTINGS, LIGAND_BITS, FRAGMENT_BITS, None, ENDPOINTS,
                              models.HEAD_DIRECT, models.PROTEIN_FROZEN)


def test_protein_mode_for_maps_each_configuration() -> None:
    assert models.protein_mode_for("frozen") == models.PROTEIN_FROZEN
    assert models.protein_mode_for("finetune") == models.PROTEIN_FINETUNE
    assert models.protein_mode_for("no_protein") == models.PROTEIN_NONE


def test_protein_mode_for_refuses_an_unknown_configuration() -> None:
    with pytest.raises(ValueError, match="cấu hình lạ"):
        models.protein_mode_for("partial")


def test_count_parameters_counts_trainable_separately() -> None:
    model = build(models.HEAD_DIRECT, models.PROTEIN_FROZEN)
    for parameter in model.ligand_encoder.parameters():
        parameter.requires_grad_(False)
    counts = models.count_parameters(model)
    assert counts["trainable"] < counts["total"]


def test_build_mlp_has_a_linear_last_layer_so_the_output_can_be_negative() -> None:
    mlp = models.build_mlp(4, [8], 1, dropout=0.0)
    assert isinstance(mlp[-1], torch.nn.Linear)


def test_direct_no_edit_head_drops_the_edit_branch() -> None:
    assert build(models.HEAD_DIRECT_NO_EDIT, models.PROTEIN_FROZEN).uses_edit_branch is False


def test_direct_no_edit_head_returns_one_number_per_row() -> None:
    model = build(models.HEAD_DIRECT_NO_EDIT, models.PROTEIN_FROZEN)
    assert model(**inputs()).shape == (BATCH,)


def test_direct_no_edit_head_runs_without_the_fragment_inputs() -> None:
    model = build(models.HEAD_DIRECT_NO_EDIT, models.PROTEIN_FROZEN).eval()
    batch = inputs()
    batch["fragment_a"] = batch["fragment_b"] = None
    assert model(**batch).shape == (BATCH,)


def test_direct_no_edit_head_still_sees_both_molecules() -> None:
    # Khác twin: hai phía vẫn vào cùng một MLP, nên đổi phía B thì đầu ra phải đổi.
    model = build(models.HEAD_DIRECT_NO_EDIT, models.PROTEIN_FROZEN).eval()
    batch = inputs()
    changed = dict(batch)
    changed["ligand_b"] = torch.rand(BATCH, LIGAND_BITS,
                                     generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        assert not torch.allclose(model(**batch), model(**changed))


def test_direct_no_edit_head_has_fewer_parameters_than_direct() -> None:
    # Chênh lệch đúng bằng nhánh edit; đó là yếu tố mà ablation muốn tách ra.
    with_edit = models.count_parameters(build(models.HEAD_DIRECT, models.PROTEIN_FROZEN))
    without = models.count_parameters(build(models.HEAD_DIRECT_NO_EDIT,
                                            models.PROTEIN_FROZEN))
    assert without["total"] < with_edit["total"]


def test_only_the_direct_head_is_antisymmetric_by_construction() -> None:
    # Dự án CHỌN đo mức bất đối xứng của hai đầu joint chứ không cưỡng chế nó, nên test
    # này ghim lựa chọn đó: chỉ twin cho f(A,B) = −f(B,A).
    for head in (models.HEAD_DIRECT, models.HEAD_DIRECT_NO_EDIT):
        model = build(head, models.PROTEIN_FROZEN).eval()
        batch = inputs()
        swapped = dict(batch)
        swapped["ligand_a"], swapped["ligand_b"] = batch["ligand_b"], batch["ligand_a"]
        swapped["fragment_a"], swapped["fragment_b"] = (batch["fragment_b"],
                                                        batch["fragment_a"])
        with torch.no_grad():
            assert not torch.allclose(model(**batch), -model(**swapped), atol=1e-6)



# --------------------------------------------------------------------------- cách ghép
# Việc giao họp 02/11/2026: projector, attention, normalization + skip connection.

FUSION_SETTINGS: dict[str, Any] = {
    **SETTINGS, "fusion_dim": 8, "attention_heads": 2, "attention_layers": 1,
    "attention_dropout": 0.0, "residual_width": 12, "residual_blocks": 2,
}
ALL_PROTEIN_MODES = (models.PROTEIN_FROZEN, models.PROTEIN_NONE)


def build_with(head: str, protein_mode: str, **changes: Any) -> models.EvaluatorModel:
    torch.manual_seed(0)
    dimension = None if protein_mode == models.PROTEIN_NONE else PROTEIN_DIM
    return models.EvaluatorModel({**FUSION_SETTINGS, **changes}, LIGAND_BITS,
                                 FRAGMENT_BITS, dimension, ENDPOINTS, head, protein_mode)


def swap_sides(batch: dict[str, Any]) -> dict[str, Any]:
    swapped = dict(batch)
    swapped["ligand_a"], swapped["ligand_b"] = batch["ligand_b"], batch["ligand_a"]
    swapped["fragment_a"], swapped["fragment_b"] = batch["fragment_b"], batch["fragment_a"]
    return swapped


@pytest.mark.parametrize("fusion", models.FUSIONS)
@pytest.mark.parametrize("head_block", models.HEAD_BLOCKS)
@pytest.mark.parametrize("head", models.HEADS)
@pytest.mark.parametrize("protein_mode", ALL_PROTEIN_MODES)
def test_every_fusion_and_head_block_returns_one_number_per_row(
        fusion: str, head_block: str, head: str, protein_mode: str) -> None:
    model = build_with(head, protein_mode, fusion=fusion, head_block=head_block)
    output = model(**inputs(protein=protein_mode != models.PROTEIN_NONE))
    assert output.shape == (BATCH,)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("fusion", models.FUSIONS)
def test_gradient_reaches_every_parameter(fusion: str) -> None:
    """Không nhánh nào bị bỏ quên trong forward: mọi tham số đều nhận gradient."""
    model = build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN, fusion=fusion,
                       head_block=models.HEAD_BLOCK_RESIDUAL)
    model(**inputs()).sum().backward()
    missing = [name for name, parameter in model.named_parameters()
               if parameter.grad is None]
    assert missing == []


def test_settings_without_the_new_keys_mean_concat_and_plain() -> None:
    """Config và checkpoint cũ không có fusion/head_block phải ra đúng kiến trúc v1."""
    model = build(models.HEAD_DIRECT, models.PROTEIN_FROZEN)
    assert model.fusion == models.FUSION_CONCAT
    assert model.head_block == models.HEAD_BLOCK_PLAIN


def test_concat_keeps_the_v1_parameter_names_so_old_checkpoints_load() -> None:
    old = build(models.HEAD_DIRECT, models.PROTEIN_FROZEN)
    new = build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN,
                     fusion=models.FUSION_CONCAT, head_block=models.HEAD_BLOCK_PLAIN)
    assert list(old.state_dict()) == list(new.state_dict())
    new.load_state_dict(old.state_dict())
    batch = inputs()
    with torch.no_grad():
        assert torch.equal(old.eval()(**batch), new.eval()(**batch))


@pytest.mark.parametrize("fusion", (models.FUSION_PROJECTOR, models.FUSION_ATTENTION))
def test_twin_head_stays_antisymmetric_with_every_fusion(fusion: str) -> None:
    model = build_with(models.HEAD_TWIN, models.PROTEIN_FROZEN, fusion=fusion,
                       head_block=models.HEAD_BLOCK_RESIDUAL).eval()
    batch = inputs()
    with torch.no_grad():
        assert torch.allclose(model(**batch), -model(**swap_sides(batch)), atol=1e-5)


def test_attention_can_tell_side_a_from_side_b() -> None:
    """A và B dùng chung encoder; nhờ embedding vai trò mà đổi chỗ hai phía vẫn đổi kết quả."""
    model = build_with(models.HEAD_DIRECT_NO_EDIT, models.PROTEIN_FROZEN,
                       fusion=models.FUSION_ATTENTION).eval()
    batch = inputs()
    with torch.no_grad():
        assert not torch.allclose(model(**batch), model(**swap_sides(batch)), atol=1e-6)


def test_attention_mixes_the_branches() -> None:
    """Đổi protein phải đổi phần biểu diễn của phân tử A sau attention: các token nhìn nhau."""
    model = build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN,
                       fusion=models.FUSION_ATTENTION).eval()
    batch = inputs()
    width = FUSION_SETTINGS["fusion_dim"]

    def ligand_a_token(protein: Any) -> Any:
        hidden_a = model.project(models.KIND_LIGAND, model.ligand_encoder(batch["ligand_a"]))
        hidden_b = model.project(models.KIND_LIGAND, model.ligand_encoder(batch["ligand_b"]))
        edit_a = model.project(models.KIND_EDIT, model.edit_encoder(batch["fragment_a"]))
        edit_b = model.project(models.KIND_EDIT, model.edit_encoder(batch["fragment_b"]))
        context = model.context_pieces(protein, batch["endpoint"])
        return model.combine([hidden_a, hidden_b, edit_a, edit_b, *context])[:, :width]

    with torch.no_grad():
        before = ligand_a_token(batch["protein"])
        after = ligand_a_token(batch["protein"] + 1.0)
    assert not torch.allclose(before, after, atol=1e-6)


def test_token_kinds_follow_the_head_and_protein_mode() -> None:
    direct = build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN)
    assert direct.token_kinds == ("ligand", "ligand", "edit", "edit", "protein", "endpoint")
    twin = build_with(models.HEAD_TWIN, models.PROTEIN_NONE)
    assert twin.token_kinds == ("ligand", "endpoint")


def test_projector_brings_every_branch_to_the_shared_width() -> None:
    model = build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN,
                       fusion=models.FUSION_PROJECTOR)
    batch = inputs()
    pieces = [model.project(models.KIND_LIGAND, model.ligand_encoder(batch["ligand_a"])),
              model.project(models.KIND_EDIT, model.edit_encoder(batch["fragment_a"])),
              *model.context_pieces(batch["protein"], batch["endpoint"])]
    assert {piece.shape for piece in pieces} == {(BATCH, FUSION_SETTINGS["fusion_dim"])}


def test_residual_block_starts_from_the_identity_path() -> None:
    """Thân khối bằng 0 thì khối dư trả lại đúng đầu vào: tín hiệu luôn đi qua nhánh tắt."""
    block = models.ResidualBlock(6, 1.0, 0.0, 0.0)
    for parameter in block.body.parameters():
        torch.nn.init.zeros_(parameter)
    values = torch.rand(BATCH, 6)
    assert torch.equal(block(values), values)


def test_model_refuses_an_unknown_fusion() -> None:
    with pytest.raises(ValueError, match="fusion"):
        build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN, fusion="sum")


def test_model_refuses_an_unknown_head_block() -> None:
    with pytest.raises(ValueError, match="head_block"):
        build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN, head_block="dense")


def test_attention_refuses_a_width_that_the_heads_do_not_divide() -> None:
    with pytest.raises(ValueError, match="attention_heads"):
        build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN,
                   fusion=models.FUSION_ATTENTION, attention_heads=3)


def test_residual_head_refuses_zero_blocks() -> None:
    with pytest.raises(ValueError, match="residual_blocks"):
        build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN,
                   head_block=models.HEAD_BLOCK_RESIDUAL, residual_blocks=0)


def test_architecture_settings_drop_the_notes() -> None:
    settings = {**FUSION_SETTINGS, "note": "x", "fusion_note": "y"}
    kept = models.architecture_settings(settings)
    assert "note" not in kept and "fusion_note" not in kept
    assert kept["fusion_dim"] == FUSION_SETTINGS["fusion_dim"]


def test_head_norm_defaults_to_none_and_keeps_the_v1_parameter_names() -> None:
    old = build(models.HEAD_DIRECT, models.PROTEIN_FROZEN)
    new = build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN,
                     fusion=models.FUSION_CONCAT, head_norm=models.HEAD_NORM_NONE)
    assert old.head_norm == models.HEAD_NORM_NONE
    assert list(old.state_dict()) == list(new.state_dict())


@pytest.mark.parametrize("fusion", models.FUSIONS)
@pytest.mark.parametrize("head", models.HEADS)
def test_head_norm_puts_a_layernorm_before_every_relu_of_the_head(fusion: str,
                                                                   head: str) -> None:
    model = build_with(head, models.PROTEIN_FROZEN, fusion=fusion,
                       head_norm=models.HEAD_NORM_LAYER)
    layers = list(model.predictor)
    relus = [index for index, layer in enumerate(layers) if isinstance(layer, torch.nn.ReLU)]
    assert relus
    for index in relus:
        assert isinstance(layers[index - 1], torch.nn.LayerNorm)
        assert isinstance(layers[index - 2], torch.nn.Linear)
    output = model(**inputs())
    assert output.shape == (BATCH,)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.predictor.parameters())


def test_head_norm_keeps_units_alive_when_the_hidden_bias_is_pushed_negative() -> None:
    """Đúng ca 06/11/2026: lệch âm làm lớp ẩn chết hết ở head thường, không chết khi có LN."""
    batch = inputs()
    outputs = {}
    for norm in models.HEAD_NORMS:
        model = build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN,
                           fusion=models.FUSION_PROJECTOR, head_norm=norm).eval()
        with torch.no_grad():
            model.predictor[0].bias.fill_(-10.0)
            outputs[norm] = model(**batch)
    assert torch.unique(outputs[models.HEAD_NORM_NONE]).numel() == 1
    assert torch.unique(outputs[models.HEAD_NORM_LAYER]).numel() == BATCH


def test_model_refuses_an_unknown_head_norm() -> None:
    with pytest.raises(ValueError, match="head_norm"):
        build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN, head_norm="batchnorm")


def test_head_norm_is_refused_with_the_residual_head() -> None:
    with pytest.raises(ValueError, match="head_norm"):
        build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN,
                   head_block=models.HEAD_BLOCK_RESIDUAL, head_norm=models.HEAD_NORM_LAYER)


def test_hidden_factor_widens_only_the_inner_layer_of_a_residual_block() -> None:
    """Bảng 14 (Gorishniy 2021): hidden factor là tỉ lệ lớp trong / chiều khối."""
    block = models.ResidualBlock(64, hidden_factor=3.0, hidden_dropout=0.0,
                                 residual_dropout=0.0)
    first, second = block.body[0], block.body[3]
    assert (first.in_features, first.out_features) == (64, 192)
    assert (second.in_features, second.out_features) == (192, 64)
    assert block(torch.zeros(BATCH, 64)).shape == (BATCH, 64)


def test_hidden_factor_of_one_is_the_old_square_block() -> None:
    block = models.ResidualBlock(64, 1.0, 0.0, 0.0)
    assert block.body[0].out_features == 64 == block.body[3].in_features


def test_hidden_width_never_falls_below_one() -> None:
    assert models.hidden_width(2, 0.1) == 1
    assert models.hidden_width(100, 2.5) == 250


def test_the_two_residual_dropouts_are_separate_knobs() -> None:
    """Mục F.1 của bài: dropout trong khối và dropout nhánh dư thường KHÁC nhau."""
    block = models.ResidualBlock(32, 1.0, hidden_dropout=0.25, residual_dropout=0.0)
    assert block.body[2].p == 0.25
    assert block.body[4].p == 0.0


def test_residual_dropout_defaults_back_to_the_model_dropout() -> None:
    # -1 là quy ước "chưa đặt riêng": giữ đúng hành vi của các lần chạy trước 07/11/2026.
    assert models.dropout_or_default(-1.0, 0.5) == 0.5
    assert models.dropout_or_default(None, 0.5) == 0.5
    assert models.dropout_or_default(0.0, 0.5) == 0.0
    assert models.dropout_or_default(0.3, 0.5) == 0.3


def test_residual_head_defaults_reproduce_the_runs_before_the_sweep() -> None:
    """Config mặc định phải cho ĐÚNG kiến trúc cũ, nếu không mọi số đã đo thành vô nghĩa."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    stage3 = json.loads((root / "config.json").read_text(encoding="utf-8"))["stage3"]
    model_settings = stage3["model"]
    assert model_settings["residual_hidden_factor"] == 1.0
    assert model_settings["residual_hidden_dropout"] == -1.0
    assert model_settings["residual_dropout"] == -1.0

    built = build_with(models.HEAD_DIRECT, models.PROTEIN_FROZEN,
                       head_block=models.HEAD_BLOCK_RESIDUAL, residual_width=32,
                       residual_blocks=2, dropout=0.5,
                       residual_hidden_factor=model_settings["residual_hidden_factor"],
                       residual_hidden_dropout=model_settings["residual_hidden_dropout"],
                       residual_dropout=model_settings["residual_dropout"])
    block = built.predictor.blocks[0]
    assert block.body[0].out_features == 32
    assert block.body[2].p == 0.5 and block.body[4].p == 0.5
