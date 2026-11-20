"""Figures for every run: what training did over time, and train against validation.

    python scripts/plot_train_validation.py curve     runs/evaluator/train_..._t3a-4174688.json
    python scripts/plot_train_validation.py gap       runs/evaluator/train_..._t3a-4174688.json \
        --eval runs/evaluator/eval_..._t3a-eval-4174688.json
    python scripts/plot_train_validation.py classical runs/evaluator/classical_..._t1c-4174688.json

WHY. A table of final numbers was asked about in the 2026-11-18 meeting and could not be
answered from a table: whether a number is the top of a curve or a point on a plateau, and
how far train sits above validation, are properties of the shape and not of the last row.
Every result that has a train side and a validation side gets a figure of BOTH from now on.

Three figures, all built from manifests alone - no GPU, no data, nothing recomputed:

  curve     - validation over the course of training, at every measurement the run made
              (end of epoch AND intra-epoch), with the selected point marked. Train loss
              gets its OWN panel, see the limitation below.
  gap       - train against validation at the selected checkpoint, per metric, with the
              gap written on the figure. The train side comes from a separate
              `--mode eval --partition train` run of the same checkpoint.
  classical - the same train-against-validation bars for the classical baselines, at the
              alpha each model's run selected.

THE LIMITATION, STATED ON THE FIGURE ITSELF. A training run records `train_loss` once per
epoch and validation metrics at every measurement. It does NOT record train MAE or train
pairwise accuracy per epoch, so a two-line curve of the SAME metric over time cannot be
drawn from existing manifests - train loss is a different quantity (the optimiser's
objective on scaled labels, under dropout) and sharing an axis with validation MAE would
invite exactly the misreading the meeting was worried about. It therefore gets its own
panel, and the caption says why. Measuring the train partition at every measurement point
is what would turn that panel into a second line; `stage3.train.measure_train_metrics`
switches it on for future runs (see docs §4.12) and costs one forward pass over the
training rows per measurement, which is affordable on PAIR75 and SUB100 and not on FULL.

Incomplete input is refused rather than drawn around: a missing eval manifest, two halves
from different checkpoints, or an eval of the wrong partition all stop the script.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "runs" / "figures"

# The pairing rule lives in one place: diagnostics_tables decides when a train half and a
# validation half are the same model, and this script must not have a second opinion.
_SPEC = importlib.util.spec_from_file_location(
    "diagnostics_tables", ROOT / "scripts" / "diagnostics_tables.py")
assert _SPEC is not None and _SPEC.loader is not None
_tables = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_tables)

# (key in the metrics block, part, label on the figure, bigger is better)
METRICS = (("mae", "regression", "MAE (lower is better)", False),
           ("pairwise_accuracy_macro", "ranking", "pairwise accuracy, macro", True))


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pick(scores: dict[str, Any], name: str, part: str) -> float | None:
    value = ((scores or {}).get(part) or {}).get(name)
    return None if value is None else float(value)


def measurements(manifest: dict[str, Any], slice_name: str
                 ) -> tuple[list[str], dict[str, list[float | None]], list[int]]:
    """(label per measurement, one series per metric, the indices that end an epoch).

    The x axis is the measurement INDEX and not a fractional epoch: the manifest records
    which step each intra-epoch measurement happened at, but not how many steps the epoch
    had, so a fractional position would be invented. The epoch ends are marked instead,
    which is what the eye needs to read the shape.
    """
    history = manifest.get("history") or []
    key = ("validation_selection"
           if slice_name != "all" and any("validation_selection" in record
                                          for record in history)
           else "validation")
    labels, ends = [], []
    series: dict[str, list[float | None]] = {name: [] for name, _, _, _ in METRICS}
    series["train_loss"] = []
    # The train side of the SAME metric, when the run measured it
    # (stage3.train.measure_train_metrics). This is what makes the curve two lines of one
    # quantity instead of validation beside a loss that is not comparable with it.
    for name, _, _, _ in METRICS:
        series[f"train_{name}"] = []
    for index, record in enumerate(history):
        step = record.get("step")
        labels.append(f"e{record.get('epoch')}" if step is None
                      else f"e{record.get('epoch')}+{step}")
        if step is None:
            ends.append(index)
        scores = record.get(key) or record.get("validation") or {}
        trained = record.get("train") or {}
        for name, part, _, _ in METRICS:
            series[name].append(pick(scores, name, part))
            series[f"train_{name}"].append(pick(trained, name, part))
        loss = record.get("train_loss")
        series["train_loss"].append(None if loss is None else float(loss))
    return labels, series, ends


def curve(manifest_path: Path, out: Path, slice_name: str) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = read(manifest_path)
    labels, series, ends = measurements(manifest, slice_name)
    if not labels:
        print(f"{manifest_path.name} has no history to plot", file=sys.stderr)
        return 1
    best = manifest.get("final_metrics_index")
    xs = list(range(len(labels)))

    figure, axes = plt.subplots(3, 1, figsize=(11.0, 9.5), sharex=True)
    measured_train = any(any(value is not None for value in series[f"train_{name}"])
                         for name, _, _, _ in METRICS)
    for axis, (name, _, title, higher) in zip(axes, METRICS):
        values = series[name]
        drawn = [(x, y) for x, y in zip(xs, values) if y is not None]
        axis.plot([x for x, _ in drawn], [y for _, y in drawn], marker="o",
                  markersize=4, linewidth=1.6, color="#1f77b4", label="validation")
        # Same metric, same evaluate(), same query definition - so this line and the one
        # above ARE comparable, which train_loss never was.
        trained = [(x, y) for x, y in zip(xs, series[f"train_{name}"]) if y is not None]
        if trained:
            axis.plot([x for x, _ in trained], [y for _, y in trained], marker="^",
                      markersize=4, linewidth=1.6, color="#2ca02c", label="train")
        for index in ends:
            axis.axvline(index, color="lightgrey", linewidth=0.8, zorder=0)
        if best is not None and 0 <= int(best) < len(xs) and values[int(best)] is not None:
            axis.axvline(int(best), color="#d62728", linewidth=1.4, linestyle="--")
            axis.annotate(f"selected: {values[int(best)]:.4f}",
                          (int(best), values[int(best)]), textcoords="offset points",
                          xytext=(8, 8), fontsize=9, color="#d62728")
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
    axis.set_xlabel("measurement (e<epoch> for the end of an epoch, e<epoch>+<step> "
                    "inside one)")

    figure.suptitle(f"{manifest_path.stem}: validation over training, "
                    f"slice {slice_name!r}", fontsize=12)
    missing_train = (
        "" if measured_train else
        "This run did not measure the train side at each point "
        "(stage3.train.measure_train_metrics was off), so the upper panels show "
        "validation only. ")
    measured_note = (
        f"The train line is the SAME metric measured the same way, over "
        f"{manifest.get('train_measured_rows') or '?'} training rows. "
        if measured_train else "")
    figure.text(0.5, 0.012,
                "Train loss is in its own panel on purpose: it is the optimiser's "
                "objective on scaled labels and under dropout, NOT the same quantity as "
                "validation MAE, so the two do not belong on one axis. "
                + missing_train + measured_note +
                f"Selected point: {manifest.get('best_epoch')} / step "
                f"{manifest.get('best_step')}, by "
                f"{manifest.get('selection_metric')} on the "
                f"{manifest.get('selection_slice')!r} slice.",
                ha="center", fontsize=7.5, color="dimgrey", wrap=True)
    figure.tight_layout(rect=(0, 0.055, 1, 0.985))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    print(f"wrote {out}")
    return 0


def gap(manifest_path: Path, eval_path: Path, out: Path, slice_name: str,
        train_slice: str) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = read(manifest_path)
    evaluated = read(eval_path)
    problems = _tables.same_checkpoint(manifest, evaluated)
    if problems:
        print("these two manifests are not two halves of one model:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    # Train and validation are read at the slices that exist on this split: on a
    # pair-keyed split the strict slice of the TRAIN partition is empty by construction
    # (build_pair_split.py writes unseen_pair = is_validation), so reading both halves at
    # `strict` would plot nothing for train.
    validation = (manifest.get("validation_metrics") or {}).get(slice_name) or {}
    train = (evaluated.get("train_metrics") or {}).get(train_slice) or {}
    pairs = []
    for name, part, title, higher in METRICS:
        first, second = pick(train, name, part), pick(validation, name, part)
        if first is None or second is None:
            print(f"{name}: train {first!r}, validation {second!r} - one half is "
                  f"missing, so the gap cannot be drawn", file=sys.stderr)
            return 1
        pairs.append((title, first, second, higher))

    figure, axes = plt.subplots(1, len(pairs), figsize=(5.0 * len(pairs), 4.6))
    axes = axes if len(pairs) > 1 else [axes]
    for axis, (title, first, second, higher) in zip(axes, pairs):
        # Lát cắt ghi ngay trên nhãn cột, không chỉ trong chú thích: trên SUB100 hai
        # phía đọc ở hai lát khác nhau (train ở `all` vì lát strict của partition train
        # rỗng do cấu tạo trên split theo pair), và một người xem hình phải thấy điều đó
        # mà không phải đọc dòng chữ nhỏ bên dưới.
        bars = axis.bar([f"train\n({train_slice})", f"validation\n({slice_name})"],
                        [first, second],
                        color=["#2ca02c", "#1f77b4"], width=0.55)
        for bar, value in zip(bars, (first, second)):
            axis.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, value),
                          textcoords="offset points", xytext=(0, 4), ha="center",
                          fontsize=10)
        difference = second - first
        axis.set_title(f"{title}\ngap {difference:+.4f}", fontsize=10)
        axis.grid(alpha=0.3, linestyle=":", axis="y")
        axis.set_ylim(0, max(first, second) * 1.22)

    figure.suptitle(f"{manifest_path.stem}: train against validation at the selected "
                    f"checkpoint", fontsize=11)
    figure.text(0.5, 0.015,
                f"Train columns: {train_slice!r} slice. Validation columns: "
                f"{slice_name!r} slice. Both halves are the SAME checkpoint "
                f"({Path(str(manifest.get('checkpoint'))).name}); the train half is a "
                f"separate --mode eval --partition train run, because a --mode train "
                f"manifest holds no pass over the training rows. The validation "
                f"configuration was selected ON validation, so the gap is not an "
                f"independent estimate of either side.",
                ha="center", fontsize=7.5, color="dimgrey", wrap=True)
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    print(f"wrote {out}")
    return 0


def classical(manifest_path: Path, out: Path, slice_name: str,
              train_slice: str) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    manifest = read(manifest_path)
    results = manifest.get("results") or {}
    selected = manifest.get("selected_alpha") or {}
    rows = []
    for model in ("linear", "ridge", "lasso", "xgboost"):
        if model in selected:
            label = str(selected[model].get("label"))
        else:
            labels = [name for name, block in results.items()
                      if str(block.get("model")) == model]
            if len(labels) != 1:
                # Either the model was not fitted, or it has a grid with no recorded
                # winner. Quoting the first fit of a grid would put a number on the
                # figure that the run never selected.
                continue
            label = labels[0]
        block = results.get(label)
        if block is None:
            continue
        rows.append((label, (block.get("train_metrics") or {}).get(train_slice) or {},
                     (block.get("validation_metrics") or {}).get(slice_name) or {}))
    if not rows:
        print(f"{manifest_path.name} holds no model with a recorded selection",
              file=sys.stderr)
        return 1

    figure, axes = plt.subplots(1, len(METRICS), figsize=(6.0 * len(METRICS), 4.8))
    axes = axes if len(METRICS) > 1 else [axes]
    positions = np.arange(len(rows))
    for axis, (name, part, title, _) in zip(axes, METRICS):
        train_values = [pick(train, name, part) for _, train, _ in rows]
        validation_values = [pick(validation, name, part) for _, _, validation in rows]
        if any(value is None for value in train_values + validation_values):
            print(f"{name}: a model is missing one half, so the figure would compare "
                  f"different things", file=sys.stderr)
            plt.close(figure)
            return 1
        axis.bar(positions - 0.19, train_values, width=0.36, label=f"train ({train_slice})",
                 color="#2ca02c")
        axis.bar(positions + 0.19, validation_values, width=0.36,
                 label=f"validation ({slice_name})", color="#1f77b4")
        for position, (first, second) in enumerate(zip(train_values, validation_values)):
            axis.annotate(f"{first:.3f}", (position - 0.19, first), ha="center",
                          textcoords="offset points", xytext=(0, 3), fontsize=8)
            axis.annotate(f"{second:.3f}", (position + 0.19, second), ha="center",
                          textcoords="offset points", xytext=(0, 3), fontsize=8)
        axis.set_xticks(positions)
        axis.set_xticklabels([label for label, _, _ in rows], rotation=20, fontsize=8,
                             ha="right")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3, linestyle=":", axis="y")
        axis.legend(fontsize=8)

    figure.suptitle(f"{manifest_path.stem}: classical baselines, train against "
                    f"validation", fontsize=11)
    figure.text(0.5, 0.015,
                "Each bar pair is the alpha that run SELECTED on validation, never the "
                "first of a grid; a model whose winner is not recorded is left out rather "
                "than guessed. The train numbers are computed on the very rows the model "
                "was fitted on.",
                ha="center", fontsize=7.5, color="dimgrey", wrap=True)
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    print(f"wrote {out}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=("curve", "gap", "classical"))
    parser.add_argument("manifest")
    parser.add_argument("--eval", default=None,
                        help="the --mode eval --partition train manifest (figure 'gap')")
    parser.add_argument("--out", default=None)
    parser.add_argument("--slice", dest="slice_name", default="strict",
                        help="the validation slice to read (default strict)")
    parser.add_argument("--train-slice", default="all",
                        help="the train slice to read (default all; see the docstring)")
    args = parser.parse_args(argv)

    path = Path(args.manifest)
    if not path.is_file():
        print(f"no such manifest: {path}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else FIGURES / f"{args.figure}_{path.stem}.png"
    if args.figure == "curve":
        return curve(path, out, args.slice_name)
    if args.figure == "classical":
        return classical(path, out, args.slice_name, args.train_slice)
    if not args.eval:
        print("figure 'gap' needs --eval <the --mode eval --partition train manifest>: "
              "a --mode train manifest holds no pass over the training rows, so there is "
              "no train half to draw without it", file=sys.stderr)
        return 2
    eval_path = Path(args.eval)
    if not eval_path.is_file():
        print(f"no such eval manifest: {eval_path}", file=sys.stderr)
        return 2
    return gap(path, eval_path, out, args.slice_name, args.train_slice)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
