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
