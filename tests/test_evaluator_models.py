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

