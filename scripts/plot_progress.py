"""Draw the figure of a run that is still running, from its progress stream.

    python scripts/plot_progress.py runs/evaluator/progress_20261120_121104.jsonl
    python scripts/plot_progress.py runs/evaluator/progress_20261120_121104.jsonl \
        --watch 60
    python scripts/plot_progress.py --latest runs/evaluator --watch 60

WHY. `scripts/plot_train_validation.py curve` needs the manifest, which exists only once a
run has finished. The stream (src/molopt/evaluator/progress.py) is written and flushed at
every measurement, so the same curve can be drawn while the run is in flight. `--watch N`
redraws every N seconds until the process is interrupted, which is what makes a figure
that keeps up with the run.

A stream is NOT a manifest, and the figure says so: there is no selected point yet, so
nothing is marked as the selection - the best measurement so far is annotated instead, and
the caption states that the run may still move past it. Once the run finishes, the figure
to quote is the one drawn from the manifest.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "plot_train_validation", ROOT / "scripts" / "plot_train_validation.py")
assert _SPEC is not None and _SPEC.loader is not None
_plots = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_plots)

METRICS = _plots.METRICS


def records(path: Path) -> list[dict[str, Any]]:
    """Every complete line of the stream.

    A run writing its next line while this reads is normal, so a trailing partial line is
    dropped rather than treated as corruption: it will be complete at the next redraw.
    """
    found = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            found.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return found


def latest(runs_dir: Path) -> Path | None:
    streams = sorted(runs_dir.glob("progress_*.jsonl"),
                     key=lambda path: path.stat().st_mtime)
    return streams[-1] if streams else None


def best_so_far(history: list[dict[str, Any]], slice_name: str) -> tuple[int, float] | None:
    """(index, value) of the best `pairwise_accuracy_macro` measured so far, or None.

    The run's own selection rule is `train.selection_score`; this is deliberately NOT a
    reimplementation of it - it marks the best measurement of the headline ranking metric
    so the eye has something to find, and the caption says it is not a selection.
    """
    best = None
    for index, record in enumerate(history):
        scores = (record.get("validation_selection") if slice_name != "all"
                  else None) or record.get("validation") or {}
        value = ((scores.get("ranking") or {}).get("pairwise_accuracy_macro"))
        if value is None:
            continue
        if best is None or float(value) > best[1]:
            best = (index, float(value))
    return best


def draw(stream: Path, out: Path, slice_name: str) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = records(stream)
    if not history:
        print(f"{stream} holds no complete measurement yet", file=sys.stderr)
        return 1
    labels, series, ends = _plots.measurements({"history": history}, slice_name)
    xs = list(range(len(labels)))
    best = best_so_far(history, slice_name)

    figure, axes = plt.subplots(3, 1, figsize=(11.0, 9.5), sharex=True)
    for axis, (name, _, title, _) in zip(axes, METRICS):
        drawn = [(x, y) for x, y in zip(xs, series[name]) if y is not None]
        axis.plot([x for x, _ in drawn], [y for _, y in drawn], marker="o", markersize=4,
                  linewidth=1.6, color="#1f77b4", label="validation")
        for index in ends:
            axis.axvline(index, color="lightgrey", linewidth=0.8, zorder=0)
        if best is not None and series[name][best[0]] is not None:
            axis.axvline(best[0], color="#ff7f0e", linewidth=1.2, linestyle=":")
        axis.set_ylabel(title, fontsize=9)
        axis.grid(alpha=0.3, linestyle=":")
        axis.legend(loc="best", fontsize=8)

    axis = axes[2]
    drawn = [(x, y) for x, y in zip(xs, series["train_loss"]) if y is not None]
    axis.plot([x for x, _ in drawn], [y for _, y in drawn], marker="s", markersize=4,
              linewidth=1.6, color="#2ca02c", label="train loss (end of epoch only)")
    axis.set_ylabel("train loss", fontsize=9)
    axis.grid(alpha=0.3, linestyle=":")
    axis.legend(loc="best", fontsize=8)
    axis.set_xticks(xs)
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    axis.set_xlabel("measurement")

    marked = ("" if best is None else
              f" Best {METRICS[1][2]} so far: {best[1]:.4f} at {labels[best[0]]} "
              f"(dotted line).")
    figure.suptitle(f"{stream.stem}: run IN PROGRESS, {len(history)} measurements, "
                    f"slice {slice_name!r}", fontsize=12)
    figure.text(0.5, 0.012,
                "Drawn from the progress stream of a run that has not finished, so NO "
                "point here is a selection: the run may still move past the best "
                "measurement, and the selected checkpoint is decided by "
                "train.selection_score when the run ends." + marked +
                " Train loss keeps its own panel: it is the optimiser's objective on "
                "scaled labels under dropout, not validation MAE. Quote the figure drawn "
                "from the manifest, not this one.",
                ha="center", fontsize=7.5, color="dimgrey", wrap=True)
    figure.tight_layout(rect=(0, 0.055, 1, 0.985))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    print(f"wrote {out} ({len(history)} measurements)")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stream", nargs="?", default=None)
    parser.add_argument("--latest", default=None,
                        help="a runs directory; draws its most recent stream")
    parser.add_argument("--out", default=None)
    parser.add_argument("--slice", dest="slice_name", default="strict")
    parser.add_argument("--watch", type=int, default=0,
                        help="redraw every N seconds until interrupted")
    args = parser.parse_args(argv)

    if args.latest:
        runs_dir = Path(args.latest)
        if not runs_dir.is_dir():
            print(f"no such runs directory: {runs_dir}", file=sys.stderr)
            return 2
        stream = latest(runs_dir)
        if stream is None:
            print(f"no progress_*.jsonl in {runs_dir}; is a run writing one? "
                  f"(stage3.train keeps the stream on by default)", file=sys.stderr)
            return 2
    elif args.stream:
        stream = Path(args.stream)
        if not stream.is_file():
            print(f"no such stream: {stream}", file=sys.stderr)
            return 2
    else:
        print("give a progress_*.jsonl, or --latest <runs dir>", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else _plots.FIGURES / f"{stream.stem}.png"
    if args.watch <= 0:
        return draw(stream, out, args.slice_name)
    print(f"redrawing {out} every {args.watch} s; interrupt to stop")
    while True:
        draw(stream, out, args.slice_name)
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
