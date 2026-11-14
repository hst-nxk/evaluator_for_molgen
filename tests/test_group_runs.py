"""scripts/group_runs.py: gộp các lần chạy chỉ khác hạt giống.

File này sinh ra sau một lỗi thật ngày 13/11/2026: bảng `--min-seeds 3` KHÔNG có dòng nào
cho `augment_reverse=True` (việc 2 biên bản 09/11) dù cả ba hạt giống đã chạy xong. Lý do:
`summarize_runs.label` duyệt `overrides` theo thứ tự cờ `--set`, hạt 20261023 phóng bằng
script cũ nên chuỗi nhãn khác hai hạt kia, `name_of` ra hai chuỗi, thành hai nhóm 1 + 2 và
bị lọc sạch. Lỗi kiểu này IM LẶNG — không có test thì không ai thấy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import group_runs  # noqa: E402  (scripts/ không phải package)


def manifest(overrides: dict[str, object], seed: int = 20261023,
             **changes: object) -> dict[str, object]:
    """Manifest tối thiểu đủ cho label() và identity()."""
    base: dict[str, object] = {
        "configuration": "frozen", "head": "direct", "seed": seed,
        "overrides": overrides, "protein_mode": "frozen",
        "model": {"dropout": 0.5}, "batch_size": 256,
        "learning_rate": 9.86e-05, "weight_decay": 0.0, "loss": "huber",
        "selection_metric": "pairwise_accuracy_macro", "selection_slice": "strict",
    }
    base.update(changes)
    return base


def test_the_same_flags_in_a_different_order_make_one_group() -> None:
    """Chính cái lỗi của việc 2: thứ tự cờ --set không được tạo ra nhóm mới."""
    first = manifest({"model.head_block": "residual", "learning_rate": 9.86e-05,
                      "augment_reverse": True}, seed=20261023)
    second = manifest({"learning_rate": 9.86e-05, "augment_reverse": True,
                       "model.head_block": "residual"}, seed=20261024)
    assert group_runs.name_of(first) == group_runs.name_of(second)
    assert group_runs.identity(first) == group_runs.identity(second)


def test_a_run_with_one_more_flag_stays_a_separate_group() -> None:
    """Sắp xếp KHÔNG được làm nhòe khác biệt thật: thêm một cờ là một thí nghiệm khác."""
    plain = manifest({"model.head_block": "residual"})
    extra = manifest({"model.head_block": "residual", "augment_reverse": True})
    assert group_runs.name_of(plain) != group_runs.name_of(extra)


def test_a_different_value_stays_a_separate_group() -> None:
    assert (group_runs.name_of(manifest({"model.pair_features": "difference"}))
            != group_runs.name_of(manifest({"model.pair_features": "none"})))


def test_the_head_and_configuration_keep_the_front_of_the_label() -> None:
    """Nhãn còn phải đọc được: hai mảnh đầu không bị trộn vào phần sắp xếp."""
    name = group_runs.name_of(manifest({"augment_reverse": True, "loss": "mse"}))
    assert name.split(" ")[:2] == ["frozen", "direct"]


def test_the_seed_is_the_only_thing_allowed_to_differ() -> None:
    overrides = {"model.head_block": "residual"}
    assert (group_runs.identity(manifest(overrides, seed=20261023))
            == group_runs.identity(manifest(overrides, seed=20261025)))


def test_a_structural_difference_outside_the_label_still_splits() -> None:
    """Khác biệt không hiện trong nhãn (ở đây: loss) vẫn phải tách nhóm."""
    overrides = {"model.head_block": "residual"}
    assert (group_runs.identity(manifest(overrides))
            != group_runs.identity(manifest(overrides, loss="mse")))


def test_the_measured_lambda_of_ignite_does_not_split_the_seeds() -> None:
    """`sharpness_penalty.lambda_final` là số ĐO ĐƯỢC, khác nhau theo hạt giống."""
    overrides = {"model.head_block": "residual"}
    first = manifest(overrides, seed=20261023,
                     sharpness_penalty={"mode": "on", "lambda_final": 0.113})
    second = manifest(overrides, seed=20261024,
                      sharpness_penalty={"mode": "on", "lambda_final": 0.207})
    assert group_runs.identity(first) == group_runs.identity(second)


def test_the_antisymmetry_weight_splits_groups() -> None:
    """Khóa của việc 4: phạt 1.0 và phạt 0.0 là hai thí nghiệm khác nhau."""
    overrides = {"antisymmetry_penalty": 1.0}
    assert (group_runs.identity(manifest(overrides, antisymmetry_penalty=1.0))
            != group_runs.identity(manifest({}, antisymmetry_penalty=0.0)))


def test_a_group_that_mixes_two_rulers_is_called_out(capsys) -> None:
    """Khối `metrics` khác nhau = đo bằng hai thước; trung bình nhóm đó vô nghĩa."""
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    overrides = {"model.head_block": "residual"}
    runs = [
        manifest(overrides, seed=20261023,
                 metrics={"expand_both_directions": True},
                 validation_metrics={"strict": {"ranking": {"pairwise_accuracy_macro": 0.55}}}),
        manifest(overrides, seed=20261024,
                 metrics={"expand_both_directions": False},
                 validation_metrics={"strict": {"ranking": {"pairwise_accuracy_macro": 0.56}}}),
    ]
    with tempfile.TemporaryDirectory() as folder:
        for index, run in enumerate(runs):
            (_Path(folder) / f"train_{index}.json").write_text(
                _json.dumps(run), encoding="utf-8")
        sys.argv = ["group_runs.py", "--slice", "strict", "--min-seeds", "2",
                    "--runs-dir", folder]
        assert group_runs.main() == 0
    assert "CẢNH BÁO" in capsys.readouterr().err


def test_runs_without_the_ruler_key_do_not_warn(capsys) -> None:
    """Các lần chạy trước 13/11/2026 không có khóa `metrics`: không được báo động giả."""
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    overrides = {"model.head_block": "residual"}
    runs = [
        manifest(overrides, seed=seed,
                 validation_metrics={"strict": {"ranking": {"pairwise_accuracy_macro": 0.55}}})
        for seed in (20261023, 20261024)
    ]
    with tempfile.TemporaryDirectory() as folder:
        for index, run in enumerate(runs):
            (_Path(folder) / f"train_{index}.json").write_text(
                _json.dumps(run), encoding="utf-8")
        sys.argv = ["group_runs.py", "--slice", "strict", "--min-seeds", "2",
                    "--runs-dir", folder]
        assert group_runs.main() == 0
    assert "CẢNH BÁO" not in capsys.readouterr().err


def write_runs(folder, runs) -> None:
    """Ghi các manifest ra thư mục tạm để gọi main()."""
    import json as _json
    from pathlib import Path as _Path

    for index, run in enumerate(runs):
        (_Path(folder) / f"train_{index}.json").write_text(
            _json.dumps(run), encoding="utf-8")


def score(value: float) -> dict:
    return {"strict": {"ranking": {"pairwise_accuracy_macro": value}}}


def test_the_table_shows_what_actually_differs(capsys) -> None:
    """Mảnh phân biệt nằm ở ĐẦU nhãn sau khi sắp xếp, không được bị cắt mất.

    Lỗi thật đêm 13/11/2026: năm nhóm mới in ra cùng một cái đuôi
    `residual_* selection_* weight_decay=0.0`, không phân biệt được MoLFormer với Morgan,
    không thấy `no_protein` hay `twin`. Số đo tốt mà bảng không đọc được thì như không có.
    """
    import tempfile

    long_tail = {f"model.residual_width_{index}": 102 for index in range(12)}
    runs = []
    for seed in (20261023, 20261024):
        runs.append(manifest({**long_tail, "model.ligand_features": "embedding"},
                             seed=seed, validation_metrics=score(0.5588)))
        runs.append(manifest({**long_tail, "antisymmetry_penalty": 3.0}, seed=seed,
                             antisymmetry_penalty=3.0, validation_metrics=score(0.5532)))
    with tempfile.TemporaryDirectory() as folder:
        write_runs(folder, runs)
        sys.argv = ["group_runs.py", "--slice", "strict", "--min-seeds", "2",
                    "--runs-dir", folder, "--width", "40"]
        assert group_runs.main() == 0
    out = capsys.readouterr().out
    assert "ligand_features=embedding" in out
    assert "antisymmetry_penalty=3.0" in out
    # Phần chung phải được in MỘT lần ở dòng đầu, không lặp lại trên từng dòng.
    assert out.count("residual_width_11=102") == 1
