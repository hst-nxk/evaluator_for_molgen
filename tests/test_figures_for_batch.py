"""`scripts/figures_for_batch.py`: all of one batch's figures, found from its own log.

"Every result gets a figure" only holds if nobody has to remember. These tests check that
the batch's own run ids decide which manifests are drawn, and - the part that matters most
- that a result WITHOUT a figure is named out loud instead of quietly missing.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from test_plot_train_validation import (classical_manifest, eval_manifest,  # noqa: E402
                                        train_manifest)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "figures_for_batch", ROOT / "scripts" / "figures_for_batch.py")
assert SPEC is not None and SPEC.loader is not None
figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figures)

PID = 4174688


def batch(directory: Path, *, with_eval: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / f"train_frozen_direct_1_t3a-{PID}.json", train_manifest())
    write(directory / f"train_frozen_direct_2_t3b-{PID}.json", train_manifest())
    write(directory / f"classical_frozen_3_t1c-{PID}.json", classical_manifest())
    if with_eval:
        write(directory / f"eval_frozen_direct_4_t3a-eval-{PID}.json", eval_manifest())
    return directory


def write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def log_file(tmp_path: Path, pid: int = PID) -> Path:
    path = tmp_path / "batch.log"
    path.write_text(f"+ python run_evaluator.py --mode train --run-id t3a-{pid} "
                    f"--config frozen\n", encoding="utf-8")
    return path


def test_every_manifest_of_the_batch_gets_what_it_supports(tmp_path: Path) -> None:
    runs = batch(tmp_path / "evaluator")
    out = tmp_path / "figures"
    printed: list[str] = []
    assert figures.build(runs, PID, out, "strict", "all", log=printed.append) == 0
    names = sorted(path.name for path in out.glob("*.png"))
    # t3a: a curve and, because its eval half is here, a train/validation figure.
    # t3b: a curve only. t1c: the classical train/validation figure.
    assert len(names) == 4
    assert any(name.startswith("curve_train_frozen_direct_1_t3a") for name in names)
    assert any(name.startswith("gap_train_frozen_direct_1_t3a") for name in names)
    assert any(name.startswith("curve_train_frozen_direct_2_t3b") for name in names)
    assert any(name.startswith("classical_classical_frozen_3_t1c") for name in names)


def test_a_result_without_a_figure_is_named(tmp_path: Path) -> None:
    """The whole point. t3a has no eval half here, so there is no train/validation figure
    for it - and that has to be stated, because a missing figure is otherwise invisible.
    """
    runs = batch(tmp_path / "evaluator", with_eval=False)
    printed: list[str] = []
    assert figures.build(runs, PID, tmp_path / "figures", "strict", "all",
                         log=printed.append) == 0
    text = "\n".join(printed)
    assert "These results have NO figure:" in text
    assert "t3a: no train/validation figure" in text
    assert "t3a-eval has 0 manifests" in text


def test_a_task_of_another_batch_is_not_drawn(tmp_path: Path) -> None:
    runs = batch(tmp_path / "evaluator")
    write(runs / "train_frozen_direct_9_t3c-999999.json", train_manifest())
    out = tmp_path / "figures"
    assert figures.build(runs, PID, out, "strict", "all", log=lambda _: None) == 0
    assert not any("t3c" in path.name for path in out.glob("*.png"))


def test_two_manifests_with_one_run_id_are_refused_for_that_task(tmp_path: Path) -> None:
    # Two files claiming the same run id: neither can be attributed, so the task is named
    # rather than one of the two being picked.
    runs = batch(tmp_path / "evaluator")
    write(runs / f"train_frozen_direct_5_t3b-{PID}.json", train_manifest())
    printed: list[str] = []
    assert figures.build(runs, PID, tmp_path / "figures", "strict", "all",
                         log=printed.append) == 0
    assert any("t3b: 2 manifests carry this run id" in line for line in printed)


def test_an_empty_batch_is_a_failure_not_a_silent_success(tmp_path: Path) -> None:
    runs = tmp_path / "evaluator"
    runs.mkdir()
    assert figures.build(runs, PID, tmp_path / "figures", "strict", "all",
                         log=lambda _: None) == 1


def test_the_pid_comes_from_the_log(tmp_path: Path) -> None:
    batch(tmp_path / "evaluator")
    path = log_file(tmp_path)
    assert figures.main([str(tmp_path / "evaluator"), str(path),
                         "--out", str(tmp_path / "figures")]) == 0
    assert list((tmp_path / "figures").glob("*.png"))


def test_a_log_of_another_batch_renders_nothing(tmp_path: Path) -> None:
    batch(tmp_path / "evaluator")
    path = log_file(tmp_path, pid=123456)
    assert figures.main([str(tmp_path / "evaluator"), str(path),
                         "--out", str(tmp_path / "figures")]) == 1


def test_a_log_without_run_ids_exits_two(tmp_path: Path) -> None:
    path = tmp_path / "empty.log"
    path.write_text("nothing here\n", encoding="utf-8")
    batch(tmp_path / "evaluator")
    assert figures.main([str(tmp_path / "evaluator"), str(path)]) == 2


def test_a_missing_runs_directory_exits_two(tmp_path: Path) -> None:
    assert figures.main([str(tmp_path / "nope"), str(log_file(tmp_path))]) == 2


def test_the_protein_seed_pairs_are_included() -> None:
    # The three-pair batch writes t5r1..t5s3; its runs need curves like any other.
    tasks = [task for task, _ in figures.TRAIN_TASKS]
    for name in ("t5r1", "t5s1", "t5r2", "t5s2", "t5r3", "t5s3"):
        assert name in tasks


def test_only_the_tasks_with_an_eval_half_ask_for_one() -> None:
    # t3b and t3d are protein-shuffled or SUB100 runs with no eval-on-train pass, so
    # asking for a train half would report a missing figure that was never expected.
    expected = {task: half for task, half in figures.TRAIN_TASKS}
    assert expected["t3a"] == "t3a-eval"
    assert expected["t3c"] == "t3c-eval"
    assert expected["t3b"] is None and expected["t3d"] is None
