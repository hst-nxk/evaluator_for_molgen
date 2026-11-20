"""Is the 2026-11-19 batch's output fit to build the three result tables from?

    python scripts/check_diagnostics_manifests.py runs/evaluator runs/diagnostics_1911_gpu1.log
    python scripts/check_diagnostics_manifests.py runs/evaluator --pid 31415

Prints one line per task and exits 0 only when every condition below holds. Exit 1 means
the batch's output is not fit to quote; exit 2 is a usage error.

WHY THIS EXISTS. The obvious check - glob `runs/evaluator/*.json` and print the two
selection fields - is wrong in three ways, each of which lets a bad batch look fine:

1. A GLOB SEES THE WHOLE HISTORY. `runs/evaluator/` holds manifests from earlier months.
   Two train manifests from 2026-10-31 predate the selection fields entirely and print
   `None None`, which reads as "the new batch chose nothing" when the new batch may have
   been perfect - or, the other way round, their presence makes the directory look
   populated when THIS batch wrote nothing at all.
2. A MISSING FILE IS SILENT. If task 3c died, a glob simply does not list it, and nothing
   says a row of the protein-control table has no number behind it. The condition has to be
   "exactly one manifest per task", which only a per-task lookup can state.
3. A HALF-FINISHED CLASSICAL RUN ALREADY LOOKS RIGHT. `--mode classical` rewrites its
   manifest after EVERY fit (deliberately: a four-model run that dies in xgboost must not
   lose the three models before it). Such a file already carries both selection fields and
   an alpha chosen among however many levels had finished, with `complete: false`. Quoting
   it would quote a grid that was never swept.

So a task is identified by the run id the batch gave it - `<task>-<pid>`, the suffix of its
manifest's filename (see `run_stamp` in run_evaluator.py and `new_run_id` in the batch
script) - and every task is checked on its own.

WHAT IS CHECKED
    all seven tasks   : exactly one manifest, and selection_metric / selection_slice equal
                        to what the MLP side used (pairwise_accuracy_macro / strict), since
                        an alpha and a checkpoint chosen by two different rules do not
                        belong in one table
    t1a, t1b, t1c     : complete = true, and selected_alpha names a winner for BOTH ridge
                        and lasso
    t3a, t3b, t3c, t3d: the manifest names a checkpoint, and the file is there

This reads files only. It never writes, and it never judges the numbers themselves - only
whether the set of manifests is the one the three tables need.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# The run ids the batch mints (scripts/server_run_diagnostics_1911.sh). The scoring runs
# (t3a-eval ...) and the capacity tasks are deliberately absent: the capacity verdict is a
# threshold on train R2 and takes no selection metric, and the scoring runs are extra
# readings of a checkpoint that is already checked through its own train task.
CLASSICAL_TASKS = ("t1a", "t1b", "t1c")
TRAIN_TASKS = ("t3a", "t3b", "t3c", "t3d")

# What both sides have to agree on; see $SELECTION in the batch script.
WANTED_METRIC = "pairwise_accuracy_macro"
WANTED_SLICE = "strict"

RUN_ID = re.compile(r"--run-id (t[0-9a-z]+(?:-eval)?)-(\d+)\b")


def pid_from_log(text: str) -> int:
    """The process id the batch stamped into its run ids, read back from its own log.

    The pid file is removed when the batch exits, so the log is what is left. Every run id
    of one batch carries the same pid; two different pids in one log mean two batches wrote
    to it, and then no task can be attributed, so this refuses instead of picking one.
    """
    found = {int(pid) for _, pid in RUN_ID.findall(text)}
    if not found:
        raise ValueError("no '--run-id <task>-<pid>' line in the log; is this the log of a "
                         "2026-11-19 batch?")
    if len(found) > 1:
        raise ValueError(f"the log holds run ids from several batches ({sorted(found)}); "
                         f"pass --pid to say which one to check")
    return found.pop()


def manifests_for(runs_dir: Path, task: str, pid: int) -> list[Path]:
    """Every manifest written by one task of one batch, by its run-id suffix."""
    return sorted(runs_dir.glob(f"*_{task}-{pid}.json"))


def read(path: Path) -> dict:
    """One manifest; a file that is not readable JSON is a failure, not a crash."""
    try:
        with path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name} cannot be read as JSON ({error})") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} does not hold a JSON object")
    return loaded


def selection_problems(manifest: dict) -> list[str]:
    """What is wrong with this manifest's two selection fields."""
    problems = []
    metric = manifest.get("selection_metric")
    chosen_slice = manifest.get("selection_slice")
    if metric != WANTED_METRIC:
        problems.append(f"selection_metric is {metric!r}, expected {WANTED_METRIC!r}")
    if chosen_slice != WANTED_SLICE:
        problems.append(f"selection_slice is {chosen_slice!r}, expected {WANTED_SLICE!r}")
    return problems


def classical_problems(manifest: dict) -> list[str]:
    """What is wrong with a classical manifest beyond its selection fields."""
    problems = []
    if manifest.get("complete") is not True:
        finished = manifest.get("models_finished")
        problems.append(f"complete is {manifest.get('complete')!r}, so the baseline list "
                        f"did not finish (models finished: {finished})")
    chosen = manifest.get("selected_alpha") or {}
    for model in ("ridge", "lasso"):
        if model not in chosen:
            unavailable = manifest.get("selected_alpha_unavailable")
            problems.append(f"selected_alpha names no winner for {model} "
                            f"(selected_alpha_unavailable: {unavailable})")
        elif chosen[model].get("alpha") is None:
            problems.append(f"selected_alpha[{model!r}] holds no alpha")
    return problems


def train_problems(manifest: dict, runs_dir: Path) -> list[str]:
    """What is wrong with a train manifest beyond its selection fields."""
    checkpoint = manifest.get("checkpoint")
    if not checkpoint:
        return ["the manifest names no checkpoint"]
    path = Path(checkpoint)
    if not path.is_absolute():
        # Same rule as scripts/pick_train_checkpoint.py: relative to the project root,
        # which is the parent of the parent of runs/evaluator.
        path = runs_dir.parent.parent / path
    if not path.is_file():
        return [f"the checkpoint {checkpoint} is missing"]
    return []


def check(runs_dir: Path, pid: int) -> tuple[bool, list[str]]:
    """(is the batch fit to quote, one report line per task)."""
    lines = []
    good = True
    for task in CLASSICAL_TASKS + TRAIN_TASKS:
        found = manifests_for(runs_dir, task, pid)
        if len(found) != 1:
            good = False
            names = [path.name for path in found] or "none"
            lines.append(f"{task}: FAIL - expected exactly one manifest, found "
                         f"{len(found)} ({names})")
            continue
        path = found[0]
        try:
            manifest = read(path)
        except ValueError as error:
            good = False
            lines.append(f"{task}: FAIL - {error}")
            continue
        problems = selection_problems(manifest)
        problems += (classical_problems(manifest) if task in CLASSICAL_TASKS
                     else train_problems(manifest, runs_dir))
        if problems:
            good = False
            lines.append(f"{task}: FAIL - {path.name}")
            lines.extend(f"      {problem}" for problem in problems)
        else:
            lines.append(f"{task}: ok   - {path.name}")
    return good, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that one batch of the 2026-11-19 diagnostics wrote the seven "
                    "manifests the three result tables need, all chosen the same way.")
    parser.add_argument("runs_dir", help="the directory holding the manifests, normally "
                                         "runs/evaluator")
    parser.add_argument("log", nargs="?", help="the batch's log, to read its pid from")
    parser.add_argument("--pid", type=int, help="the batch's pid, instead of its log")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"{runs_dir} is not a directory", file=sys.stderr)
        return 2
    if (args.log is None) == (args.pid is None):
        print("pass either the batch's log or --pid, not both and not neither",
              file=sys.stderr)
        return 2

    if args.pid is not None:
        pid = args.pid
    else:
        log = Path(args.log)
        if not log.is_file():
            print(f"{log} does not exist", file=sys.stderr)
            return 2
        try:
            pid = pid_from_log(log.read_text(encoding="utf-8", errors="replace"))
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

    print(f"batch pid {pid} in {runs_dir}")
    good, lines = check(runs_dir, pid)
    for line in lines:
        print(line)
    if good:
        print("\nAll seven manifests are present and chosen the same way: the three tables "
              "can be built from these files.")
        return 0
    print("\nNot fit to quote. Every FAIL above has to be resolved - by rerunning the task "
          "or by stating in the report that its row has no number - before a table is "
          "built.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
