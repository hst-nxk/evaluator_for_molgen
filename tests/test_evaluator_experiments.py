"""Hai thí nghiệm thêm ngày 07/11/2026: lọc phép sửa hiếm và learning rate theo nhánh.

Cả hai đều phải có tính chất "mặc định không đổi gì": min_edit_rows = 1 và mọi hệ số
bằng 1,0 phải cho đúng hành vi cũ, nếu không thì mọi số đo trước đó không còn so được.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="phần optimizer cần torch")

from molopt.evaluator import data, models, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Phép sửa {1,2} có 3 dòng (kể cả dòng ghi ngược chiều), {3,4} và {5,5} mỗi cái 1 dòng.
FRAGMENT_A = np.array([1, 2, 3, 1, 5], dtype=np.int64)
FRAGMENT_B = np.array([2, 1, 4, 2, 5], dtype=np.int64)
TARGET = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
KEYS = {"query": np.array([0, 0, 1, 1, 2], dtype=np.int64)}


def arrays() -> data.PairArrays:
    """Năm dòng đã mã hóa, chỉ đủ cột để hàm lọc chạy."""
    count = TARGET.size
    return data.PairArrays(
        molecule_a=np.arange(count, dtype=np.int64),
        molecule_b=np.arange(count, dtype=np.int64) + 100,
        fragment_a=FRAGMENT_A, fragment_b=FRAGMENT_B,
        protein=np.zeros(count, dtype=np.int64),
        endpoint=np.zeros(count, dtype=np.int64),
        target=TARGET,
        slices={"all": np.ones(count, dtype=bool),
                "ki": np.array([True, False, True, False, True])})


# ------------------------------------------------------------------ lọc phép sửa hiếm


def test_min_edit_rows_one_changes_nothing() -> None:
    """Mặc định phải là không lọc: số cũ còn so được."""
    before = arrays()
    after, keys, report = train.filter_rare_edits(before, KEYS, 1)
    assert len(after) == len(before)
    assert np.array_equal(after.target, TARGET)
    assert np.array_equal(keys["query"], KEYS["query"])
    assert report["rows_dropped"] == 0
    assert report["edits_dropped"] == 0


def test_rows_of_single_occurrence_edits_are_dropped() -> None:
    after, _, report = train.filter_rare_edits(arrays(), KEYS, 2)
    # Giữ lại ba dòng của phép sửa {1,2}; bỏ {3,4} và {5,5}.
    assert np.array_equal(after.target, np.array([10.0, 11.0, 13.0]))
    assert report == {"min_edit_rows": 2, "rows_before": 5, "rows_after": 3,
                      "rows_dropped": 2, "edits_dropped": 2,
                      "control_random_rows": False}


def test_the_control_drops_the_same_number_of_rows_at_random() -> None:
    """Đối chứng phải CÙNG CỠ với lần lọc thật, nếu không thì so không được."""
    real, _, real_report = train.filter_rare_edits(arrays(), KEYS, 2)
    fake, _, fake_report = train.filter_rare_edits(arrays(), KEYS, 2, control=True,
                                                   seed=1)
    assert len(fake) == len(real)
    assert fake_report["rows_dropped"] == real_report["rows_dropped"]
    assert fake_report["control_random_rows"] is True
    # Đối chứng không bỏ theo phép sửa, nên không đếm được phép sửa nào bị bỏ.
    assert fake_report["edits_dropped"] == 0


def test_the_control_is_repeatable_and_depends_on_the_seed() -> None:
    kept = []
    for seed in (1, 1, 2, 3, 4, 5):
        after, _, _ = train.filter_rare_edits(arrays(), KEYS, 2, control=True, seed=seed)
        kept.append(tuple(after.target.tolist()))
    assert kept[0] == kept[1]
    assert len(set(kept)) > 1, "mọi seed cho cùng một tập dòng: seed không có tác dụng"


def test_the_control_keeps_slices_and_keys_aligned() -> None:
    after, keys, _ = train.filter_rare_edits(arrays(), KEYS, 2, control=True, seed=7)
    assert keys["query"].size == len(after)
    for name, mask in after.slices.items():
        assert mask.size == len(after), name


def test_an_edit_is_counted_without_its_direction() -> None:
    """Dòng (2,1) phải cùng phép sửa với (1,2), nếu không thì ngưỡng 3 bỏ hết."""
    after, _, _ = train.filter_rare_edits(arrays(), KEYS, 3)
    assert np.array_equal(after.target, np.array([10.0, 11.0, 13.0]))


def test_slices_and_query_keys_are_filtered_together() -> None:
    after, keys, _ = train.filter_rare_edits(arrays(), KEYS, 2)
    assert np.array_equal(after.slices["ki"], np.array([True, False, False]))
    assert np.array_equal(keys["query"], np.array([0, 0, 1]))
    for name, mask in after.slices.items():
        assert mask.size == len(after), name


def test_a_threshold_that_drops_everything_is_refused() -> None:
    """Thà lỗi ngay còn hơn train trên bảng rỗng."""
    with pytest.raises(ValueError, match="min_edit_rows"):
        train.filter_rare_edits(arrays(), KEYS, 4)


# ------------------------------------------------- learning rate riêng từng nhánh

BASE_RATE = 1e-3
SETTINGS: dict[str, Any] = {
    "ligand_hidden": [16], "ligand_out": 8,
    "edit_hidden": [8], "edit_out": 4,
    "protein_out": 6, "endpoint_dim": 3,
    "head_hidden": [8], "dropout": 0.0,
    "fusion_dim": 8, "attention_heads": 2, "attention_layers": 1,
    "attention_dropout": 0.0, "residual_width": 12, "residual_blocks": 2,
}


def model(**changes: Any) -> models.EvaluatorModel:
    torch.manual_seed(0)
    return models.EvaluatorModel({**SETTINGS, **changes}, 32, 16, 48, 4,
                                 models.HEAD_DIRECT, models.PROTEIN_FROZEN)


def unit_scales() -> dict[str, float]:
    return {name: 1.0 for name in train.BRANCH_MODULES}


def test_scales_all_one_give_a_single_group_like_before() -> None:
    groups = train.branch_parameter_groups(model(), BASE_RATE, unit_scales())
    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(BASE_RATE)


def test_the_config_default_gives_a_single_group() -> None:
    """Hành vi mặc định của dự án phải khớp với test trên, đọc thẳng từ config."""
    settings = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    scales = settings["stage3"]["train"]["branch_learning_rate_scale"]
    assert set(scales) == set(train.BRANCH_MODULES)
    assert len(train.branch_parameter_groups(model(), BASE_RATE, scales)) == 1


def test_a_branch_with_its_own_scale_gets_its_own_group() -> None:
    scales = {**unit_scales(), "edit_encoder": 0.1}
    groups = train.branch_parameter_groups(model(), BASE_RATE, scales)
    rates = {group["branch"]: group["lr"] for group in groups}
    assert rates["edit_encoder"] == pytest.approx(BASE_RATE * 0.1)
    assert rates[train.OTHER_BRANCH] == pytest.approx(BASE_RATE)


def test_a_scale_of_zero_freezes_the_branch() -> None:
    """Hệ số 0 là hợp lệ: AdamW với lr = 0 không đổi tham số nào."""
    scales = {**unit_scales(), "edit_encoder": 0.0}
    groups = train.branch_parameter_groups(model(), BASE_RATE, scales)
    assert {group["branch"]: group["lr"] for group in groups}["edit_encoder"] == 0.0


@pytest.mark.parametrize("fusion", ["concat", "projector", "attention"])
@pytest.mark.parametrize("head_block", ["plain", "residual"])
def test_no_parameter_is_lost_in_any_fusion_mode(fusion: str, head_block: str) -> None:
    """Tham số bị sót sẽ im lặng không được train; phải đếm lại cho khớp."""
    built = model(fusion=fusion, head_block=head_block)
    scales = {**unit_scales(), "predictor": 0.5, "ligand_encoder": 2.0}
    groups = train.branch_parameter_groups(built, BASE_RATE, scales)
    grouped = sum(len(group["params"]) for group in groups)
    assert grouped == len(list(built.named_parameters()))


@pytest.mark.parametrize("fusion", ["concat", "projector", "attention"])
def test_branch_modules_covers_every_real_branch_name(fusion: str) -> None:
    """Bắt lỗi gõ sai tên: mọi tiền tố tham số thật phải nằm trong BRANCH_MODULES."""
    built = model(fusion=fusion)
    prefixes = {name.split(".", 1)[0] for name, _ in built.named_parameters()}
    assert prefixes <= set(train.BRANCH_MODULES), sorted(prefixes)


def test_an_unknown_branch_name_is_refused() -> None:
    with pytest.raises(ValueError, match="ligand_encodr"):
        train.branch_parameter_groups(model(), BASE_RATE, {"ligand_encodr": 0.5})


def test_a_negative_scale_is_refused() -> None:
    with pytest.raises(ValueError, match="edit_encoder"):
        train.branch_parameter_groups(model(), BASE_RATE, {"edit_encoder": -1.0})
