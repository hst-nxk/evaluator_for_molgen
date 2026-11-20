"""The run id that ties a checkpoint to the run that wrote it.

Why these tests exist. The 2026-11-19 batch scores and diagnoses the checkpoint of the run
it has just finished, and three ways of naming that checkpoint were tried and found wrong:

1. "the newest .pt in runs/evaluator" — belongs to the previous task when this one died
   before writing anything, or to an unfinished run, since `--mode train` saves its best
   checkpoint DURING the run;
2. "the newest manifest written since I started" — loses a race between two batches sharing
   the directory: job A writes its manifest and exits 0, job B writes its own a moment
   later, and A's next step then opens B's manifest. With the same split and shuffle seed
   every check inside the loader passes, so nothing announces it;
3. the timestamp in the file name — accurate only to the second, so two runs started in the
   same second write the SAME paths.

What replaced them: each train run is given a unique `--run-id` that lands in the name of
both its checkpoint and its manifest (`run_stamp` in run_evaluator.py), and the picker opens
the manifest carrying that id. The decisive test here is `test_two_successful_jobs_do_not
_take_each_others_checkpoint`.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pick_train_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("pick_train_checkpoint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
picker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(picker)


def write_run(runs: Path, run_id: str, *, stamp: str = "20261119_084500",
              when: float | None = None, checkpoint: bool = True,
              manifest: bool = True) -> Path:
    """One train run on disk, named the way run_stamp names it; returns its .pt path."""
    runs.mkdir(parents=True, exist_ok=True)
    point = runs / f"frozen_direct_{stamp}_{run_id}.pt"
    if checkpoint:
        point.write_bytes(b"not a real checkpoint")
    path = runs / f"train_frozen_direct_{stamp}_{run_id}.json"
    if manifest:
        path.write_text(json.dumps({"mode": "train", "checkpoint": str(point)}),
                        encoding="utf-8")
        if when is not None:
            os.utime(path, (when, when))
    return point


def test_it_returns_the_checkpoint_the_manifest_names(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "evaluator"
    expected = write_run(runs, "t3a-5417")
    assert picker.pick(runs, "t3a-5417") == expected


def test_two_successful_jobs_do_not_take_each_others_checkpoint(tmp_path: Path) -> None:
    """THE RACE. Job A finishes, job B finishes a moment later, and only then does A ask
    for its checkpoint. "The newest manifest" would hand A the model B trained."""
    runs = tmp_path / "runs" / "evaluator"
    first = write_run(runs, "t3a-1111", stamp="20261119_084500", when=1000.0)
    second = write_run(runs, "t3a-2222", stamp="20261119_084500", when=2000.0)
    assert picker.pick(runs, "t3a-1111") == first
    assert picker.pick(runs, "t3a-2222") == second
    # And the two runs did not write the same paths either, although their timestamps are
    # identical to the second.
    assert first != second
    assert first.is_file() and second.is_file()


def test_a_run_that_wrote_no_manifest_is_not_picked(tmp_path: Path) -> None:
    """The task died: its checkpoint may exist (saved mid-run), its manifest does not."""
    runs = tmp_path / "runs" / "evaluator"
    write_run(runs, "t3a-dead", manifest=False)
    write_run(runs, "t3a-older")
    with pytest.raises(FileNotFoundError, match="did not finish"):
        picker.pick(runs, "t3a-dead")


def test_another_runs_manifest_is_not_picked(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "evaluator"
    write_run(runs, "t3b-5417")
    with pytest.raises(FileNotFoundError, match="no train manifest with run id"):
        picker.pick(runs, "t3a-5417")


def test_an_ambiguous_id_is_refused_instead_of_guessed(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "evaluator"
    write_run(runs, "t3a-1", stamp="20261119_084500")
    write_run(runs, "t3a-1", stamp="20261119_090000")
    with pytest.raises(FileNotFoundError, match="is not unique"):
        picker.pick(runs, "t3a-1")


def test_a_manifest_whose_checkpoint_is_gone_is_refused(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "evaluator"
    point = write_run(runs, "t3a-5417")
    point.unlink()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        picker.pick(runs, "t3a-5417")


def test_a_run_that_selected_no_checkpoint_is_refused(tmp_path: Path) -> None:
    """`run_train` writes checkpoint=None when no measurement ever improved."""
    runs = tmp_path / "runs" / "evaluator"
    runs.mkdir(parents=True)
    (runs / "train_frozen_direct_20261119_084500_t3a-5417.json").write_text(
        json.dumps({"mode": "train", "checkpoint": None}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="records no checkpoint"):
        picker.pick(runs, "t3a-5417")


def test_an_empty_id_is_refused(tmp_path: Path) -> None:
    # An unset $RUN_ID must not quietly match some other run's manifest.
    with pytest.raises(FileNotFoundError, match="no run id given"):
        picker.pick(tmp_path, "")


def test_a_relative_checkpoint_path_is_resolved(tmp_path: Path) -> None:
    """Manifests record whatever path the run used; a relative one is read against the
    project root, which is two levels above runs/evaluator."""
    runs = tmp_path / "runs" / "evaluator"
    runs.mkdir(parents=True)
    point = runs / "frozen_direct_20261119_084500_t3a-1.pt"
    point.write_bytes(b"x")
    (runs / "train_frozen_direct_20261119_084500_t3a-1.json").write_text(
        json.dumps({"mode": "train",
                    "checkpoint": "runs/evaluator/frozen_direct_20261119_084500_t3a-1.pt"}),
        encoding="utf-8")
    assert picker.pick(runs, "t3a-1") == point


def test_the_cli_prints_the_path_and_reports_failure(tmp_path: Path,
                                                     capsys: pytest.CaptureFixture[str]
                                                     ) -> None:
    runs = tmp_path / "runs" / "evaluator"
    expected = write_run(runs, "t3a-5417")
    assert picker.main(["pick", str(runs), "t3a-5417"]) == 0
    assert capsys.readouterr().out.strip() == str(expected)
    assert picker.main(["pick", str(runs), "t3a-absent"]) == 1
    assert "no usable checkpoint" in capsys.readouterr().err
    assert picker.main(["pick", str(runs)]) == 2
