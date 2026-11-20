"""Every figure of one batch, found from the batch's own log. No GPU, no data.

    python scripts/figures_for_batch.py runs/evaluator runs/diagnostics_1911_gpu1.log
    python scripts/figures_for_batch.py runs/evaluator runs/diagnostics_1911_gpu1.log \
        --out runs/figures/batch_1911

WHY. "Every result gets a figure" only holds if nobody has to remember to make them. This
finds the manifests of a batch the same way the gate and the tables do - by the run id
`<task>-<pid>` the batch minted - and renders what each one supports:

  * a learning curve for every `--mode train` task the log launched - the task names are
    read from the log's own `--run-id` arguments, so a batch that invents new ones (the
    nine t9* runs of 2026-11-20) is rendered without this file being edited,
  * a train-against-validation figure for every train task that also has its
    `--mode eval --partition train` half (t3a-eval, t3c-eval),
  * a train-against-validation figure for every classical task (t1a..t1c).

A task that cannot support a figure is NAMED and skipped - a missing eval half, an empty
history, a grid with no recorded winner. The point of printing it is that a batch with a
missing figure says so here instead of being noticed, or not noticed, later.

The pid comes from the log, so the figures belong to the batch that log describes; running
it against an older log renders that older batch, which is the useful behaviour when a
figure has to be rebuilt for a result already reported.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load("check_diagnostics_manifests")
plots = load("plot_train_validation")

# (train task, its eval-on-train task). The eval half is optional: without it the curve is
# still drawn and the gap figure is reported as missing.
# Only the fallback, for a caller that has no log text. The real list comes from the log;
# see gate.tasks_from_log.
TRAIN_TASKS = (("t3a", "t3a-eval"), ("t3b", None), ("t3c", "t3c-eval"), ("t3d", None),
               ("t5r1", None), ("t5s1", None), ("t5r2", None), ("t5s2", None),
               ("t5r3", None), ("t5s3", None))
CLASSICAL_TASKS = ("t1a", "t1b", "t1c")


def build(runs_dir: Path, pid: int, out_dir: Path, slice_name: str,
          train_slice: str, log=print,
          tasks: tuple[tuple[str, str | None], ...] | None = None) -> int:
    drawn, skipped = 0, []
    # The tasks the LOG launched, not a list in this file: a batch with new task names
    # used to render zero figures and say nothing useful about why.
    for task, eval_task in (tasks if tasks is not None else TRAIN_TASKS):
        found = gate.manifests_for(runs_dir, task, pid)
        if len(found) != 1:
            if found:
                skipped.append(f"{task}: {len(found)} manifests carry this run id")
            continue
        manifest = found[0]
        target = out_dir / f"curve_{manifest.stem}.png"
        if plots.curve(manifest, target, slice_name) == 0:
            drawn += 1
        else:
            skipped.append(f"{task}: no curve (see the message above)")
        if eval_task is None:
            continue
        halves = gate.manifests_for(runs_dir, eval_task, pid)
        if len(halves) != 1:
            skipped.append(f"{task}: no train/validation figure - {eval_task} has "
                           f"{len(halves)} manifests, so there is no train half")
            continue
        target = out_dir / f"gap_{manifest.stem}.png"
        if plots.gap(manifest, halves[0], target, slice_name, train_slice) == 0:
            drawn += 1
        else:
            skipped.append(f"{task}: the two halves were refused (see above)")

    for task in CLASSICAL_TASKS:
        found = gate.manifests_for(runs_dir, task, pid)
        if len(found) != 1:
            if found:
                skipped.append(f"{task}: {len(found)} manifests carry this run id")
            continue
        target = out_dir / f"classical_{found[0].stem}.png"
        if plots.classical(found[0], target, slice_name, train_slice) == 0:
            drawn += 1
        else:
            skipped.append(f"{task}: no classical figure (see the message above)")

    log("")
    log(f"{drawn} figure(s) in {out_dir}")
    if skipped:
        log("These results have NO figure:")
        for line in skipped:
            log(f"  - {line}")
    return 0 if drawn else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir")
    parser.add_argument("log", help="the batch log, for the pid the run ids carry")
    parser.add_argument("--out", default=None)
    parser.add_argument("--slice", dest="slice_name", default="strict")
    parser.add_argument("--train-slice", default="all")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"no such runs directory: {runs_dir}", file=sys.stderr)
        return 2
    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"no such log: {log_path}", file=sys.stderr)
        return 2
    # The gate reads the pid out of the log text and refuses a log that names two
    # batches, because then no task can be attributed to one of them. Same rule here.
    text = log_path.read_text(encoding="utf-8", errors="replace")
    try:
        pid = gate.pid_from_log(text)
    except ValueError as error:
        print(f"{log_path}: {error}", file=sys.stderr)
        return 2
    tasks = gate.tasks_from_log(text)
    print(f"batch pid {pid}")
    print(f"tasks in this log: " + ", ".join(
        task if half is None else f"{task} (+{half})" for task, half in tasks)
        if tasks else "no --mode train task in this log")
    out_dir = Path(args.out) if args.out else plots.FIGURES / f"batch_{pid}"
    return build(runs_dir, pid, out_dir, args.slice_name, args.train_slice,
                 tasks=tasks or None)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
