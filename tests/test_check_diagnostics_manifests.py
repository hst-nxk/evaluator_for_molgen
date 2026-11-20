"""Is this batch's output fit to build the three tables from?

Why these tests exist. The check this file covers replaced a one-line glob over
``runs/evaluator/*.json``, which looked right and was not: the directory also holds
manifests from earlier months (two train manifests from 2026-10-31 predate the selection
fields altogether), a task that died leaves no file and a glob says nothing about it, and
``--mode classical`` rewrites its manifest after every fit, so a half-swept grid already
carries both selection fields and a provisional winner with ``complete: false``.

Each test below is one of those three ways of being wrong, plus the conditions the tables
actually need.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (Path(__file__).resolve().parents[1] / "scripts"
               / "check_diagnostics_manifests.py")
SPEC = importlib.util.spec_from_file_location("check_diagnostics_manifests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

PID = 31415
STAMP = "20261121_030405"


def classical(runs: Path, task: str, *, pid: int = PID, complete: bool = True,
              metric: str = "pairwise_accuracy_macro", chosen_slice: str = "strict",
              models: tuple[str, ...] = ("ridge", "lasso")) -> Path:
    """One classical manifest, named the way run_stamp names it."""
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"classical_frozen_{STAMP}_{task}-{pid}.json"
    path.write_text(json.dumps({
        "stage": "stage3_classical", "complete": complete,
        "selection_metric": metric, "selection_slice": chosen_slice,
        "models_finished": ["linear", "ridge"],
        "selected_alpha": {name: {"alpha": 1.0, "label": f"{name}_alpha1",
                                  "value": 0.6, "alphas_compared": 3}
                           for name in models},
        "selected_alpha_unavailable": [name for name in ("ridge", "lasso")
                                       if name not in models],
    }), encoding="utf-8")
    return path


def train(runs: Path, task: str, *, pid: int = PID, checkpoint: str | None = "",
          exists: bool = True, metric: str = "pairwise_accuracy_macro",
          chosen_slice: str = "strict") -> Path:
    """One train manifest, with its .pt beside it unless `exists` says otherwise."""
    runs.mkdir(parents=True, exist_ok=True)
    point = runs / f"frozen_direct_{STAMP}_{task}-{pid}.pt"
    if exists:
        point.write_bytes(b"not a real checkpoint")
    named = str(point) if checkpoint == "" else checkpoint
    path = runs / f"train_frozen_direct_{STAMP}_{task}-{pid}.json"
    path.write_text(json.dumps({
        "mode": "train", "checkpoint": named,
        "selection_metric": metric, "selection_slice": chosen_slice,
    }), encoding="utf-8")
    return path


def whole_batch(runs: Path) -> None:
    """Every manifest of one good batch."""
    for task in checker.CLASSICAL_TASKS:
        classical(runs, task)
    for task in checker.TRAIN_TASKS:
        train(runs, task)


def log_text(pid: int = PID, tasks: tuple[str, ...] = ("t1a", "t3a")) -> str:
    return "\n".join(
        f"+ python run_evaluator.py --run-id {task}-{pid} --mode train --config frozen"
        for task in tasks)


# ------------------------------------------------------------------ the pid, from the log


def test_the_pid_is_read_back_from_the_batchs_own_log() -> None:
    # The pid file is removed when the batch exits (the trap in gpu_batch_lock.sh), so the
    # log is where the batch's identity survives.
    assert checker.pid_from_log(log_text()) == PID


def test_a_log_holding_two_batches_is_refused() -> None:
    # Then no task can be attributed to a batch, and picking one of the two would check
    # half of each.
    with pytest.raises(ValueError, match="several batches"):
        checker.pid_from_log(log_text(PID) + "\n" + log_text(99999))


def test_a_log_with_no_run_id_is_refused() -> None:
    with pytest.raises(ValueError, match="no '--run-id"):
        checker.pid_from_log("=== TASK 1a ===\nsome output\n")


# ------------------------------------------------------- one manifest per task, no glob


def test_a_complete_batch_passes(tmp_path: Path) -> None:
    whole_batch(tmp_path)
    good, lines = checker.check(tmp_path, PID)
    assert good, lines
    assert len(lines) == 7
    assert all(": ok" in line for line in lines)


def test_a_task_that_wrote_nothing_is_named(tmp_path: Path) -> None:
    """The decisive case against a glob: a dead task leaves no file, and a glob simply does
    not list it - nothing says a row of the protein-control table has no number."""
    whole_batch(tmp_path)
    (tmp_path / f"train_frozen_direct_{STAMP}_t3c-{PID}.json").unlink()
    good, lines = checker.check(tmp_path, PID)
    assert not good
    assert any(line.startswith("t3c: FAIL") and "found 0" in line for line in lines)


def test_two_manifests_for_one_task_are_refused(tmp_path: Path) -> None:
    # Two files with the same run id mean the batch was run twice with the same pid, or a
    # file was copied in; either way it is not clear which numbers are this batch's.
    whole_batch(tmp_path)
    duplicate = tmp_path / f"classical_frozen_20261121_999999_t1a-{PID}.json"
    duplicate.write_text((tmp_path / f"classical_frozen_{STAMP}_t1a-{PID}.json")
                         .read_text(encoding="utf-8"), encoding="utf-8")
    good, lines = checker.check(tmp_path, PID)
    assert not good
    assert any(line.startswith("t1a: FAIL") and "found 2" in line for line in lines)


def test_the_history_in_the_directory_is_ignored(tmp_path: Path) -> None:
    """Manifests from earlier months carry no run-id suffix and no selection fields; they
    must neither fail the check nor make an empty batch look populated."""
    whole_batch(tmp_path)
    (tmp_path / "train_frozen_direct_20261031_180400.json").write_text(
        json.dumps({"mode": "train", "checkpoint": "whatever.pt"}), encoding="utf-8")
    good, lines = checker.check(tmp_path, PID)
    assert good, lines
    # And with nothing of this batch present, the history does not rescue it.
    for path in tmp_path.glob(f"*-{PID}.json"):
        path.unlink()
    good, lines = checker.check(tmp_path, PID)
    assert not good
    assert all("FAIL" in line for line in lines)


def test_a_manifest_of_another_batch_does_not_count(tmp_path: Path) -> None:
    whole_batch(tmp_path)
    (tmp_path / f"classical_frozen_{STAMP}_t1b-{PID}.json").unlink()
    classical(tmp_path, "t1b", pid=PID + 1)
    good, lines = checker.check(tmp_path, PID)
    assert not good
    assert any(line.startswith("t1b: FAIL") and "found 0" in line for line in lines)


# ------------------------------------------------------- both sides chose the same way


@pytest.mark.parametrize("field", ["metric", "chosen_slice"])
def test_a_task_that_selected_by_another_rule_is_refused(tmp_path: Path,
                                                         field: str) -> None:
    whole_batch(tmp_path)
    (tmp_path / f"classical_frozen_{STAMP}_t1c-{PID}.json").unlink()
    classical(tmp_path, "t1c", **{field: "mae" if field == "metric" else "all"})
    good, lines = checker.check(tmp_path, PID)
    assert not good
    assert any("expected" in line for line in lines)


def test_a_train_task_that_selected_by_another_rule_is_refused(tmp_path: Path) -> None:
    whole_batch(tmp_path)
    (tmp_path / f"train_frozen_direct_{STAMP}_t3b-{PID}.json").unlink()
    train(tmp_path, "t3b", chosen_slice="all")
    good, lines = checker.check(tmp_path, PID)
    assert not good
    assert any("selection_slice is 'all'" in line for line in lines)


# ------------------------------------------- a classical run that has not finished its grid


def test_a_classical_manifest_written_mid_grid_is_refused(tmp_path: Path) -> None:
    """Review 2026-11-20: `--mode classical` rewrites its manifest after every fit, so a
    run that died in the middle already carries both selection fields and an alpha chosen
    among however many levels had finished. Quoting it would quote a grid never swept."""
    whole_batch(tmp_path)
    (tmp_path / f"classical_frozen_{STAMP}_t1a-{PID}.json").unlink()
    classical(tmp_path, "t1a", complete=False)
    good, lines = checker.check(tmp_path, PID)
    assert not good
    assert any("did not finish" in line for line in lines)
    assert any("models finished" in line for line in lines)


def test_a_classical_manifest_missing_one_models_winner_is_refused(tmp_path: Path) -> None:
    whole_batch(tmp_path)
    (tmp_path / f"classical_frozen_{STAMP}_t1b-{PID}.json").unlink()
    classical(tmp_path, "t1b", models=("ridge",))
    good, lines = checker.check(tmp_path, PID)
    assert not good
    assert any("no winner for lasso" in line for line in lines)
    # The manifest's own explanation is quoted, so the reader is not left guessing.
    assert any("selected_alpha_unavailable: ['lasso']" in line for line in lines)


# --------------------------------------------------------- a train run without a checkpoint


def test_a_train_task_without_a_checkpoint_is_refused(tmp_path: Path) -> None:
    whole_batch(tmp_path)
    (tmp_path / f"train_frozen_direct_{STAMP}_t3a-{PID}.json").unlink()
    train(tmp_path, "t3a", checkpoint=None)
    good, lines = checker.check(tmp_path, PID)
    assert not good
    assert any("names no checkpoint" in line for line in lines)


def test_a_checkpoint_the_manifest_names_but_disk_does_not_hold_is_refused(
        tmp_path: Path) -> None:
    whole_batch(tmp_path)
    (tmp_path / f"train_frozen_direct_{STAMP}_t3d-{PID}.json").unlink()
    # The .pt that whole_batch wrote has to go as well, or the manifest would still be
    # naming a file that is there.
    (tmp_path / f"frozen_direct_{STAMP}_t3d-{PID}.pt").unlink()
    train(tmp_path, "t3d", exists=False)
    good, lines = checker.check(tmp_path, PID)
    assert not good
    assert any("is missing" in line for line in lines)


def test_a_relative_checkpoint_is_resolved_against_the_project_root(tmp_path: Path) -> None:
    # Same rule as scripts/pick_train_checkpoint.py: runs/evaluator -> two levels up.
    runs = tmp_path / "runs" / "evaluator"
    whole_batch(runs)
    (runs / f"train_frozen_direct_{STAMP}_t3a-{PID}.json").unlink()
    train(runs, "t3a", checkpoint="runs/evaluator/kept.pt", exists=False)
    good, _ = checker.check(runs, PID)
    assert not good
    (runs / "kept.pt").write_bytes(b"not a real checkpoint")
    good, lines = checker.check(runs, PID)
    assert good, lines


# ------------------------------------------------------------------------------ the CLI


def test_the_cli_exits_zero_only_for_a_complete_batch(tmp_path: Path,
                                                      capsys: pytest.CaptureFixture[str]
                                                      ) -> None:
    runs = tmp_path / "runs" / "evaluator"
    whole_batch(runs)
    log = tmp_path / "diagnostics_1911_gpu1.log"
    log.write_text(log_text(), encoding="utf-8")
    assert checker.main([str(runs), str(log)]) == 0
    assert "can be built from these files" in capsys.readouterr().out

    (runs / f"classical_frozen_{STAMP}_t1a-{PID}.json").unlink()
    assert checker.main([str(runs), str(log)]) == 1
    assert "Not fit to quote" in capsys.readouterr().out


def test_the_cli_refuses_to_guess_which_batch(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "evaluator"
    whole_batch(runs)
    log = tmp_path / "log.txt"
    log.write_text(log_text(), encoding="utf-8")
    # Neither the log nor --pid, and both at once, are usage errors: the pid decides which
    # files are even looked at.
    assert checker.main([str(runs)]) == 2
    assert checker.main([str(runs), str(log), "--pid", str(PID)]) == 2
    assert checker.main([str(runs), "--pid", str(PID)]) == 0


def test_the_cli_refuses_a_directory_that_is_not_there(tmp_path: Path) -> None:
    assert checker.main([str(tmp_path / "nope"), "--pid", str(PID)]) == 2


def test_an_unreadable_manifest_is_a_failure_not_a_crash(tmp_path: Path) -> None:
    whole_batch(tmp_path)
    (tmp_path / f"classical_frozen_{STAMP}_t1c-{PID}.json").write_text(
        "{not json", encoding="utf-8")
    good, lines = checker.check(tmp_path, PID)
    assert not good
    assert any("cannot be read as JSON" in line for line in lines)

# ------------------------------------------- tìm task từ log (review 20/11)
LAUNCHES = """+ python run_evaluator.py --mode train --run-id t9a1-4242 --config frozen
+ python run_evaluator.py --mode eval --run-id t9a1-eval-4242 --checkpoint a --partition train
+ python run_evaluator.py --mode train --run-id t9b1-4242 --config no_protein
+ python run_evaluator.py --mode errors --run-id t7s1-4242 --partition validation
+ python run_evaluator.py --mode eval --run-id t9zz-eval-4242 --partition validation
"""


def test_the_tasks_of_a_batch_are_read_from_its_log() -> None:
    """Danh sách task từng là hằng số trong file, nên một batch đặt tên mới (chín lượt
    t9* ngày 20/11) sinh ra manifest đúng rồi vẽ được 0 hình và lập bảng toàn dòng thiếu.
    Log là chỗ duy nhất không thể lệch khỏi thứ đã chạy.
    """
    assert checker.tasks_from_log(LAUNCHES) == (("t9a1", "t9a1-eval"), ("t9b1", None))


def test_an_errors_task_is_not_listed_as_a_figure_task() -> None:
    # `--mode errors` ghi row dump, không ghi train manifest; liệt kê nó vào đây thì mỗi
    # lần chạy đều có nó trong mục "NO figure", và người đọc sẽ học cách bỏ qua mục đó.
    assert all(task != "t7s1" for task, _ in checker.tasks_from_log(LAUNCHES))


def test_an_eval_of_validation_is_not_a_train_half() -> None:
    # Nửa train phải là `--mode eval --partition train`; nửa validation không phải.
    assert all(half != "t9zz-eval" for _, half in checker.tasks_from_log(LAUNCHES))


def test_a_log_with_no_training_run_yields_nothing() -> None:
    assert checker.tasks_from_log("+ python run_evaluator.py --mode classical "
                               "--run-id t1a-7-7") == ()
