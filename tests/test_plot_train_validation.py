"""`scripts/plot_train_validation.py`: the figures that go with every result.

The meeting of 2026-11-18 asked about train against validation and a table could not
answer it. These tests check the three things a figure can get wrong without anyone
noticing: drawing two halves that are not the same model, drawing a number the run never
selected, and putting train loss on the same axis as validation MAE, which is the
misreading the whole figure exists to prevent.

Each figure is actually written to a temporary file, so a broken matplotlib call fails
here rather than on the server.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_train_validation", ROOT / "scripts" / "plot_train_validation.py")
assert SPEC is not None and SPEC.loader is not None
plots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plots)

CHECKPOINT = "runs/evaluator/frozen_direct_20261120_092300_t3a-4174688.pt"


def scores(mae: float, pairwise: float) -> dict[str, Any]:
    return {"regression": {"n": 24946, "mae": mae, "r2": 0.2395},
            "ranking": {"pairwise_accuracy_macro": pairwise,
                        "pairwise_accuracy_micro": pairwise + 0.02}}


def train_manifest(**changes: Any) -> dict[str, Any]:
    history = []
    for epoch in range(3):
        for step in (200, 400, None):
            record: dict[str, Any] = {
                "epoch": epoch, "step": step,
                "validation": scores(0.60 - 0.02 * epoch, 0.70 + 0.02 * epoch),
                "validation_selection": scores(0.60 - 0.02 * epoch,
                                               0.72 + 0.02 * epoch)}
            if step is None:
                record["train_loss"] = 0.30 - 0.05 * epoch
            history.append(record)
    payload: dict[str, Any] = {
        "stage": "stage3_train", "mode": "train",
        "selection_metric": "pairwise_accuracy_macro", "selection_slice": "strict",
        "best_epoch": 2, "best_step": None, "final_metrics_index": len(history) - 1,
        "checkpoint": CHECKPOINT, "history": history,
        "validation_metrics": {"all": scores(0.5144, 0.7778),
                               "strict": scores(0.5144, 0.7778)}}
    payload.update(changes)
    return payload


def eval_manifest(partition: str = "train", checkpoint: str = CHECKPOINT) -> dict[str, Any]:
    return {"stage": "stage3_eval", "partition": partition, "checkpoint": checkpoint,
            f"{partition}_metrics": {"all": scores(0.3698, 0.8425),
                                     "strict": {"regression": {"n": 0, "mae": None,
                                                               "r2": None},
                                                "ranking": {
                                                    "pairwise_accuracy_macro": None}}}}


def classical_manifest(**changes: Any) -> dict[str, Any]:
    results = {
        "linear": {"model": "linear", "alpha": 0.0,
                   "train_metrics": {"all": scores(0.5478, 0.6576)},
                   "validation_metrics": {"strict": scores(0.6039, 0.6330)}},
        "ridge_alpha0.1": {"model": "ridge", "alpha": 0.1,
                           "train_metrics": {"all": scores(0.5479, 0.6578)},
                           "validation_metrics": {"strict": scores(0.6026, 0.6332)}},
        "ridge_alpha1.0": {"model": "ridge", "alpha": 1.0,
                           "train_metrics": {"all": scores(0.5500, 0.6500)},
                           "validation_metrics": {"strict": scores(0.6100, 0.6300)}},
    }
    payload: dict[str, Any] = {
        "stage": "stage3_classical", "results": results,
        "selected_alpha": {"ridge": {"alpha": 0.1, "label": "ridge_alpha0.1"}},
        "selected_alpha_unavailable": []}
    payload.update(changes)
    return payload


def write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ------------------------------------------------------------------ the learning curve
def test_the_curve_is_written(tmp_path: Path) -> None:
    manifest = write(tmp_path / "train_frozen_direct_1_t3a-1.json", train_manifest())
    out = tmp_path / "curve.png"
    assert plots.curve(manifest, out, "strict") == 0
    assert out.is_file() and out.stat().st_size > 0


def test_every_measurement_is_a_point_not_just_the_epoch_ends(tmp_path: Path) -> None:
    """The best point can sit INSIDE epoch 0, which is why the run measures intra-epoch at
    all. A curve drawn only at epoch ends would hide the shape it exists to show.
    """
    labels, series, ends = plots.measurements(train_manifest(), "strict")
    assert len(labels) == 9
    assert labels[:3] == ["e0+200", "e0+400", "e0"]
    # The epoch ends are marked, and they are the only points with a train loss.
    assert ends == [2, 5, 8]
    assert [index for index, value in enumerate(series["train_loss"])
            if value is not None] == ends


def test_the_selection_slice_is_read_when_the_run_used_one(tmp_path: Path) -> None:
    # The run selected on `strict`, and the manifest keeps that slice separately; reading
    # `validation` (always the `all` slice) would plot a series the run did not optimise.
    _, strict, _ = plots.measurements(train_manifest(), "strict")
    _, every, _ = plots.measurements(train_manifest(), "all")
    assert strict["pairwise_accuracy_macro"][0] == pytest.approx(0.72)
    assert every["pairwise_accuracy_macro"][0] == pytest.approx(0.70)


def test_an_empty_history_is_refused_not_drawn(tmp_path: Path) -> None:
    manifest = write(tmp_path / "train_empty_t3a-1.json", train_manifest(history=[]))
    assert plots.curve(manifest, tmp_path / "x.png", "strict") == 1
    assert not (tmp_path / "x.png").exists()


def test_train_loss_is_kept_off_the_validation_axes() -> None:
    """The one thing this figure must not do. Train loss is the optimiser's objective on
    scaled labels under dropout; drawn next to validation MAE on one axis it reads as a
    train-validation gap that was never measured.
    """
    source = (ROOT / "scripts" / "plot_train_validation.py").read_text(encoding="utf-8")
    assert "twinx" not in source
    assert "own panel" in source
    assert "NOT the same quantity as" in source


# ------------------------------------------------------------------ the train/validation gap
def test_the_gap_figure_is_written(tmp_path: Path) -> None:
    manifest = write(tmp_path / "train_frozen_direct_1_t3a-1.json", train_manifest())
    evaluated = write(tmp_path / "eval_frozen_direct_1_t3a-eval-1.json", eval_manifest())
    out = tmp_path / "gap.png"
    assert plots.gap(manifest, evaluated, out, "strict", "all") == 0
    assert out.is_file() and out.stat().st_size > 0


def test_two_halves_of_different_checkpoints_are_refused(tmp_path: Path) -> None:
    manifest = write(tmp_path / "train_a_t3a-1.json", train_manifest())
    evaluated = write(tmp_path / "eval_b_t3a-eval-1.json",
                      eval_manifest(checkpoint="runs/evaluator/another.pt"))
    out = tmp_path / "gap.png"
    assert plots.gap(manifest, evaluated, out, "strict", "all") == 1
    assert not out.exists()


def test_an_eval_of_the_validation_partition_is_not_a_train_half(tmp_path: Path) -> None:
    manifest = write(tmp_path / "train_a_t3a-1.json", train_manifest())
    evaluated = write(tmp_path / "eval_v_t3a-eval-1.json",
                      eval_manifest(partition="validation"))
    assert plots.gap(manifest, evaluated, tmp_path / "gap.png", "strict", "all") == 1


def test_the_train_half_is_not_read_at_a_slice_that_is_empty_here(tmp_path: Path) -> None:
    """On a pair-keyed split `build_pair_split.py` writes unseen_pair = is_validation, so
    the strict slice of the TRAIN partition is empty. Asking for it must stop the figure
    instead of drawing a bar of nothing.
    """
    manifest = write(tmp_path / "train_a_t3a-1.json", train_manifest())
    evaluated = write(tmp_path / "eval_t_t3a-eval-1.json", eval_manifest())
    assert plots.gap(manifest, evaluated, tmp_path / "gap.png", "strict", "strict") == 1


def test_the_gap_needs_the_eval_manifest_at_all(tmp_path: Path) -> None:
    manifest = write(tmp_path / "train_a_t3a-1.json", train_manifest())
    assert plots.main(["gap", str(manifest)]) == 2


# ------------------------------------------------------------------ the classical figure
def test_the_classical_figure_is_written(tmp_path: Path) -> None:
    manifest = write(tmp_path / "classical_frozen_1_t1c-1.json", classical_manifest())
    out = tmp_path / "classical.png"
    assert plots.classical(manifest, out, "strict", "all") == 0
    assert out.is_file() and out.stat().st_size > 0


def test_a_grid_without_a_recorded_winner_is_left_out(tmp_path: Path) -> None:
    """Two ridge fits and no selection: putting either on the figure would show a number
    the run never chose. Linear still has exactly one fit, so the figure is drawn without
    ridge rather than refused.
    """
    manifest = write(tmp_path / "classical_frozen_1_t1c-1.json",
                     classical_manifest(selected_alpha={},
                                        selected_alpha_unavailable=["ridge"]))
    out = tmp_path / "classical.png"
    assert plots.classical(manifest, out, "strict", "all") == 0
    assert out.is_file()


def test_a_manifest_with_nothing_selectable_is_refused(tmp_path: Path) -> None:
    manifest = write(tmp_path / "classical_frozen_1_t1c-1.json",
                     classical_manifest(results={}, selected_alpha={}))
    assert plots.classical(manifest, tmp_path / "c.png", "strict", "all") == 1


def test_a_model_missing_its_train_half_stops_the_figure(tmp_path: Path) -> None:
    payload = classical_manifest()
    payload["results"]["linear"]["train_metrics"] = {}
    manifest = write(tmp_path / "classical_frozen_1_t1c-1.json", payload)
    assert plots.classical(manifest, tmp_path / "c.png", "strict", "all") == 1


# ------------------------------------------------------------------ the command line
def test_a_missing_manifest_exits_two(tmp_path: Path) -> None:
    assert plots.main(["curve", str(tmp_path / "nope.json")]) == 2


def test_the_default_output_name_says_which_figure_it_is(tmp_path: Path) -> None:
    manifest = write(tmp_path / "train_frozen_direct_1_t3a-1.json", train_manifest())
    out = tmp_path / "named.png"
    assert plots.main(["curve", str(manifest), "--out", str(out)]) == 0
    assert out.is_file()
    assert plots.FIGURES.name == "figures"


def test_the_pairing_rule_is_not_reimplemented_here() -> None:
    # One opinion about when two halves are one model, and it lives in diagnostics_tables.
    source = (ROOT / "scripts" / "plot_train_validation.py").read_text(encoding="utf-8")
    assert "_tables.same_checkpoint(" in source
    assert "def same_checkpoint" not in source

# ------------------------------------------- hai đường cùng một metric (20/11/2026)
def measured_manifest() -> dict[str, Any]:
    """Manifest của một lần chạy có `measure_train_metrics`: mỗi mốc đo có cả phía train."""
    payload = train_manifest()
    for index, record in enumerate(payload["history"]):
        epoch = record["epoch"]
        # Phía train tốt hơn validation và tiếp tục tốt lên - hình dạng của overfitting,
        # thứ mà `train_loss` một mình không chứng minh được.
        record["train"] = scores(0.40 - 0.05 * epoch, 0.80 + 0.03 * epoch)
        record["train_measured_rows"] = 75000
    payload["measure_train_metrics"] = True
    payload["train_measured_rows"] = 75000
    return payload


def test_the_train_line_is_drawn_when_the_run_measured_it(tmp_path: Path) -> None:
    series = plots.measurements(measured_manifest(), "all")[1]
    assert any(value is not None for value in series["train_mae"])
    assert any(value is not None
               for value in series["train_pairwise_accuracy_macro"])
    # Và cùng số điểm với phía validation: cùng các mốc đo, không phải lưới khác.
    assert len(series["train_mae"]) == len(series["mae"])


def test_the_figure_says_how_many_training_rows_were_measured(tmp_path: Path) -> None:
    # Có trần số dòng, nên "phía train" có thể là một mẫu; hình phải nói ra số dòng thật.
    path = write(tmp_path / "train_frozen_direct_t9a1-1.json", measured_manifest())
    assert plots.curve(path, tmp_path / "curve.png", "all") == 0
    assert (tmp_path / "curve.png").is_file()
    source = (ROOT / "scripts" / "plot_train_validation.py").read_text(encoding="utf-8")
    assert "train_measured_rows" in source
    assert "SAME metric measured the same way" in source


def test_a_run_without_the_flag_says_so_instead_of_leaving_a_gap(tmp_path: Path) -> None:
    """Một hình chỉ có một đường, không chú thích, sẽ bị đọc là "không overfit"."""
    source = (ROOT / "scripts" / "plot_train_validation.py").read_text(encoding="utf-8")
    assert "measure_train_metrics was off" in source
    series = plots.measurements(train_manifest(), "all")[1]
    assert all(value is None for value in series["train_mae"])


def test_the_two_slices_are_named_on_the_bars_not_only_in_the_caption(
        tmp_path: Path) -> None:
    # Trên SUB100 hai phía đọc ở hai lát khác nhau; người xem hình phải thấy ngay.
    source = (ROOT / "scripts" / "plot_train_validation.py").read_text(encoding="utf-8")
    # So khớp với HAI ký tự (backslash + n) trong mã nguồn script, không phải
    # một newline thật - nên phải ghép qua chr(92).
    assert 'f"train' + chr(92) + 'n({train_slice})"' in source
    assert 'f"validation' + chr(92) + 'n({slice_name})"' in source
    assert "Train columns: {train_slice!r} slice" in source
