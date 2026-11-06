"""Bộ lấy mẫu của scripts/sweep_residual.py: mọi điểm quét phải nằm trong Bảng 14.

Nguồn khoảng: Gorishniy và cs. 2021, arXiv:2106.11959, Bảng 14 (mục F.1).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sweep_residual as sweep  # noqa: E402  (scripts/ không phải package)

TRIALS = 200


def sample(count: int = TRIALS, seed: int = 1) -> list[dict]:
    rng = random.Random(seed)
    return [sweep.sample_trial(rng) for _ in range(count)]


def test_every_sampled_value_stays_inside_table_14() -> None:
    for trial in sample():
        assert 1 <= trial["model.residual_blocks"] <= 8
        assert 64 <= trial["model.residual_width"] <= 512
        assert 1.0 <= trial["model.residual_hidden_factor"] <= 4.0
        assert 0.0 <= trial["model.residual_hidden_dropout"] <= 0.5
        assert 0.0 <= trial["model.residual_dropout"] <= 0.5
        assert 1e-5 <= trial["learning_rate"] <= 1e-2
        assert trial["weight_decay"] == 0.0 or 1e-6 <= trial["weight_decay"] <= 1e-3


def test_the_two_set_valued_parameters_really_hit_zero_sometimes() -> None:
    """Bảng 14 viết {0, Uniform[...]}: nhánh 0 phải xuất hiện, nếu không là quét thiếu."""
    trials = sample()
    assert any(trial["model.residual_dropout"] == 0.0 for trial in trials)
    assert any(trial["weight_decay"] == 0.0 for trial in trials)
    assert any(trial["model.residual_dropout"] > 0.0 for trial in trials)
    assert any(trial["weight_decay"] > 0.0 for trial in trials)


def test_the_same_seed_gives_the_same_sweep() -> None:
    assert sample(20, seed=7) == sample(20, seed=7)
    assert sample(20, seed=7) != sample(20, seed=8)


def test_log_uniform_covers_both_ends_of_the_range() -> None:
    rng = random.Random(0)
    drawn = [sweep.log_uniform(rng, 1e-5, 1e-2) for _ in range(TRIALS)]
    assert all(1e-5 <= value <= 1e-2 for value in drawn)
    # Thang log: một nửa số mẫu phải nằm dưới trung bình nhân, khác hẳn thang thẳng.
    middle = (1e-5 * 1e-2) ** 0.5
    below = sum(value < middle for value in drawn)
    assert TRIALS * 0.35 < below < TRIALS * 0.65


def test_the_command_never_touches_the_test_partition() -> None:
    trial = sample(1)[0]
    line = sweep.command(trial, encoder_dropout=0.5, seed=20261023)
    assert "test" not in line
    assert "--mode train" in line
    assert "--set model.head_block=residual" in line
    assert f"--set selection_metric={sweep.SELECTION_METRIC}" in line
    for key, value in trial.items():
        assert f"--set {key}={value}" in line


def test_gpu_list_is_handed_out_round_robin() -> None:
    assert sweep.parse_gpus("1,2,3,7") == [1, 2, 3, 7]
    assert sweep.parse_gpus(None) == []
    assert sweep.parse_gpus("  ") == []
    script = sweep.build_script(sample(6), shards=4, gpus=[1, 2],
                                encoder_dropout=0.5, train_seed=1, sampler_seed=2)
    prefixes = [line.split(" ")[0] for line in script.splitlines()
                if line.startswith("CUDA_VISIBLE_DEVICES=")]
    assert prefixes == ["CUDA_VISIBLE_DEVICES=1", "CUDA_VISIBLE_DEVICES=2",
                        "CUDA_VISIBLE_DEVICES=1", "CUDA_VISIBLE_DEVICES=2"]


def test_gpu_list_refuses_anything_that_is_not_a_number() -> None:
    import pytest

    with pytest.raises(ValueError, match="--gpu"):
        sweep.parse_gpus("1,gpu2")


def test_every_trial_lands_in_exactly_one_shard() -> None:
    trials = sample(20)
    script = sweep.build_script(trials, shards=4, gpus=[], encoder_dropout=0.5,
                                train_seed=20261023, sampler_seed=1)
    assert script.count("--mode train") == len(trials)
    for trial in trials:
        assert script.count(f"--set model.residual_width={trial['model.residual_width']} "
                            f"--set model.residual_hidden_factor="
                            f"{trial['model.residual_hidden_factor']}") >= 1


def test_the_script_never_glues_two_flags_together() -> None:
    """Sự cố 07/11/2026: dán tay qua terminal nuốt mất dấu cách -> '--setlearning_rate'."""
    script = sweep.build_script(sample(20), shards=4, gpus=[6], encoder_dropout=0.5,
                                train_seed=1, sampler_seed=1)
    assert "--setlearning_rate" not in script
    assert "--setmodel" not in script
    # Sau mỗi "--set" phải là đúng một dấu cách rồi mới tới tên tham số.
    head, *rest = script.split("--set")
    assert rest, "script phải có ít nhất một --set"
    for piece in rest:
        assert piece.startswith(" ") and not piece.startswith("  ")
    assert "--set" not in head


def test_the_script_waits_between_shards_so_checkpoints_do_not_collide() -> None:
    # Tên checkpoint đặt theo giây; hai chuỗi khởi động cùng giây sẽ ghi đè nhau.
    script = sweep.build_script(sample(4), shards=2, gpus=[], encoder_dropout=0.5,
                                train_seed=1, sampler_seed=1)
    assert script.count("sleep 20") == 2


def test_the_generated_script_is_valid_bash() -> None:
    import shutil
    import subprocess

    import pytest

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("máy không có bash")
    script = sweep.build_script(sample(8), shards=2, gpus=[1, 2], encoder_dropout=0.5,
                                train_seed=1, sampler_seed=1)
    # -n: chỉ kiểm tra cú pháp, KHÔNG chạy lệnh nào.
    done = subprocess.run([bash, "-n"], input=script, text=True, encoding="utf-8",
                          capture_output=True, check=False)
    assert done.returncode == 0, done.stderr
