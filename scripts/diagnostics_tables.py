"""Build the three tables of the 2026-11-19 diagnostics, from manifests and nothing else.

    python scripts/diagnostics_tables.py runs/evaluator runs/diagnostics_1911_gpu1.log
    python scripts/diagnostics_tables.py runs/evaluator --pid 4174688 > tables.md

Prints Markdown on stdout and the footnotes a reader needs to not over-read the numbers.

WHY THIS EXISTS. The numbers were first carried from the manifests into a report by hand,
and that is the one step in this pipeline with no test behind it: a transposed digit, a
number read off the `all` slice while the table says `strict`, a ridge row quoted at an
alpha nobody selected. Every number this prints is read out of a manifest at a fixed key,
and every row that has no manifest behind it prints `-` and is named in the footnotes
instead of being filled in from memory.

WHAT IT REFUSES TO DO
  * It does not choose which alpha to show. Ridge and lasso are shown at the level the RUN
    selected on validation (`selected_alpha`, chosen by the metric the MLP selects its
    checkpoint by). A level that was swept but not selected never reaches the table.
  * It does not hide a degenerate or unconverged fit. The flags column carries
    `unconverged` when the solver spent its whole iteration budget, `constant` when every
    coefficient is zero, and the non-zero count so a fit held up by one coefficient is
    visible.
  * It does not compute anything beyond a difference between two numbers it has just read
    (the protein control), and it prints both sides of that difference next to it.
  * It does not say a null result is a proven absence: the protein table prints how many
    seeds are behind each side, because a difference of 0.003 from one seed per side has no
    noise scale to be read against.

Run `scripts/check_diagnostics_manifests.py` first. This script prints what it finds
whether or not the batch is fit to quote - mid-batch is a normal time to look at it - so
the gate, not this, is what decides a table may be built.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

# The gate script holds the manifest lookup (by `_<task>-<pid>.json`) and the pid-from-log
# reader; both scripts have to agree on which file belongs to which task, so there is one
# copy of that rule and this imports it. `scripts/` is on sys.path when this file is run
# directly, but not when a test loads it by path, so put it there explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_diagnostics_manifests as gate  # noqa: E402

CAPACITY_TASKS = ("t2a", "t2b", "t2c", "t2d")

# Which manifest holds which side of which table. Written out rather than derived, because
# a wrong guess here is a mislabelled table, and the batch script is the only authority.
PAIR75 = {"classical": "t1c", "mlp": "t3a", "mlp_shuffled": "t3b",
          "mlp_train": "t3a-eval",
          "label": "PAIR75 (work/splits/pair_split_1411.parquet)"}
SUB100 = {"classical": "t1a", "mlp": "t3c", "mlp_shuffled": "t3d",
          "mlp_train": "t3c-eval",
          "label": "SUB100 (work/splits/train_subset_c200_r500.parquet)"}

# The MLP's TRAIN numbers are not in its train manifest - a `--mode train` run records the
# validation history and the selected checkpoint, not a pass over the training rows. The
# batch scores that checkpoint on the train partition in a separate `--mode eval` run
# (`score_train` in the batch script), whose manifest keys its scores under
# `<partition>_metrics`. The meeting asked for a train-and-validation table, so both
# manifests are read and a missing eval run shows as `-`, not as a silently
# validation-only table.
EVAL_TASKS = (PAIR75["mlp_train"], SUB100["mlp_train"])

MISSING = "-"


def fmt(value: Any, digits: int = 4) -> str:
    """A number as the table shows it; anything that is not a number becomes `-`."""
    if isinstance(value, bool) or value is None:
        return MISSING
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def slice_scores(metrics: Mapping[str, Any] | None, name: str) -> tuple[dict, dict]:
    """(regression, ranking) of one slice of a `train.evaluate` result."""
    scores = (metrics or {}).get(name) or {}
    return (scores.get("regression") or {}), (scores.get("ranking") or {})


def flags_of(block: Mapping[str, Any]) -> str:
    """What has to be said about a classical fit next to its numbers."""
    notes = []
    convergence = block.get("convergence") or {}
    coefficients = block.get("coefficients") or {}
    if convergence.get("hit_the_cap"):
        notes.append(f"unconverged (spent all {convergence.get('max_iter')} iterations)")
    if coefficients.get("constant_predictor"):
        notes.append("constant predictor")
    nonzero, total = coefficients.get("nonzero"), coefficients.get("total")
    if nonzero is not None:
        notes.append(f"{nonzero:,}/{total:,} non-zero")
    return "; ".join(notes) if notes else ""


def classical_rows(manifest: Mapping[str, Any]) -> list[tuple[str, dict]]:
    """One row per baseline: regularised models at the level the RUN selected.

    A model that was fitted but has no selected level (its metric was undefined at every
    level) is returned with an empty block, so the table shows a named row with `-` in it
    rather than leaving the model out.
    """
    results = manifest.get("results") or {}
    chosen = manifest.get("selected_alpha") or {}
    unavailable = set(manifest.get("selected_alpha_unavailable") or [])
    grids = manifest.get("alphas") or {}
    rows: list[tuple[str, dict]] = []
    seen: set[str] = set()

    def needs_a_winner(model: str) -> bool:
        """Is this a model whose row may only ever be a SELECTED level?

        Read from the run's own recorded grid: a model fitted at anything other than the
        single level 0.0 was swept, and quoting one of its levels without a selection
        behind it would put a number in the table that nothing chose. When the manifest
        has no `alphas` block (an older file), fall back to counting its fits.
        """
        grid = grids.get(model)
        if grid is not None:
            return [float(level) for level in grid] != [0.0]
        return len([1 for block in results.values()
                    if str(block.get("model")) == model]) > 1

    for label, block in results.items():
        model = str(block.get("model"))
        if model in seen or model in unavailable:
            continue
        seen.add(model)
        if model not in chosen:
            if needs_a_winner(model):
                # The hole this closes: with neither a selected level nor a place in
                # selected_alpha_unavailable, the code used to fall through and print the
                # FIRST fit of the grid - an alpha nobody selected, in a table whose whole
                # claim is that it only shows selected ones.
                rows.append((f"{model} (swept {grids.get(model)} and no level is "
                             f"recorded as selected)", {}))
                continue
            # One level only (linear, xgboost): the fit itself is the row.
            rows.append((label, dict(block)))
            continue
        winner = chosen[model].get("label")
        picked = results.get(winner)
        if picked is None:
            rows.append((f"{model} (the selected level {winner!r} is not in results)", {}))
            continue
        rows.append((f"{model}, alpha {picked.get('alpha')} "
                     f"(selected of {chosen[model].get('alphas_compared')})", dict(picked)))
    for model in sorted(unavailable):
        rows.append((f"{model} (no level selected: the metric was undefined at every "
                     f"level)", {}))
    return rows


def same_checkpoint(train_manifest: Mapping[str, Any],
                    eval_manifest: Mapping[str, Any]) -> list[str]:
    """Why these two manifests are not two halves of ONE model's row.

    A train half and a validation half only form a gap if they belong to the same
    checkpoint. Nothing in the file names enforces that - both manifests carry the path
    they used, so it is checked - and the eval manifest also has to be the TRAIN one: an
    eval on validation writes `validation_metrics` under the same key shape and would slip
    in as a train column that is really a second validation column.
    """
    problems = []
    partition = eval_manifest.get("partition")
    if partition != "train":
        problems.append(f"the eval manifest scored the {partition!r} partition, not "
                        f"'train'")
    left = str(train_manifest.get("checkpoint") or "")
    right = str(eval_manifest.get("checkpoint") or "")
    if not left or not right:
        problems.append("one of the two manifests names no checkpoint")
    elif Path(left).name != Path(right).name:
        problems.append(f"the two halves are different checkpoints "
                        f"({Path(left).name} and {Path(right).name})")
    return problems


def baseline_table(classical: Mapping[str, Any] | None, mlp: Mapping[str, Any] | None,
                   mlp_on_train: Mapping[str, Any] | None, split: Mapping[str, str],
                   slice_name: str, train_slice: str) -> list[str]:
    """Baselines and the MLP on one split, train AND validation, each at its own slice.

    WHY TWO SLICE NAMES. The train columns are read at `train_slice` and the validation
    columns at `slice_name`, and the header says so. On a pair-keyed split they CANNOT be
    the same: `scripts/build_pair_split.py` writes `unseen_pair = is_validation`, so every
    train row carries false and the `strict` slice of the train partition holds no rows at
    all. Reading `strict` on both sides therefore prints `-` for the train half of a run
    that completed perfectly - which is exactly what this table did until 2026-11-20.
    Passing `--train-slice all --slice strict` is the honest reading and the default;
    `all` on both sides measures the gap over the two whole partitions.
    """
    lines = [f"### Baselines vs the MLP - {split['label']}", "",
             f"Train columns: `{train_slice}` slice. Validation columns: `{slice_name}` "
             f"slice.", "",
             f"| Model | train MAE ({train_slice}) | train pairwise ({train_slice}) | "
             f"validation MAE ({slice_name}) | validation R2 ({slice_name}) | "
             f"validation pairwise ({slice_name}) | notes |",
             "|---|---|---|---|---|---|---|"]

    def row(name: str, train_metrics: Mapping[str, Any] | None,
            validation_metrics: Mapping[str, Any] | None, notes: str) -> str:
        train_regression, train_ranking = slice_scores(train_metrics, train_slice)
        regression, ranking = slice_scores(validation_metrics, slice_name)
        return (f"| {name} | {fmt(train_regression.get('mae'))} | "
                f"{fmt(train_ranking.get('pairwise_accuracy_macro'))} | "
                f"{fmt(regression.get('mae'))} | {fmt(regression.get('r2'))} | "
                f"{fmt(ranking.get('pairwise_accuracy_macro'))} | {notes} |")

    if classical is None:
        lines.append(row(f"classical baselines ({split['classical']})", None, None,
                         "manifest missing"))
    else:
        for name, block in classical_rows(classical):
            lines.append(row(name, block.get("train_metrics"),
                             block.get("validation_metrics"), flags_of(block)))
    if mlp is None:
        lines.append(row(f"**MLP ({split['mlp']})**", None, None, "manifest missing"))
    else:
        notes = (f"best epoch {mlp.get('best_epoch')} of "
                 f"{len(mlp.get('history') or [])}")
        train_metrics = None
        if mlp_on_train is None:
            notes += f"; no train numbers: the `{split['mlp_train']}` eval run is missing"
        else:
            problems = same_checkpoint(mlp, mlp_on_train)
            if problems:
                notes += ("; TRAIN HALF REFUSED: " + "; ".join(problems))
            else:
                train_metrics = mlp_on_train.get("train_metrics")
        lines.append(row(f"**MLP ({split['mlp']})**", train_metrics,
                         mlp.get("validation_metrics"), notes))
    lines.append("")
    return lines


def capacity_table(manifests: Mapping[str, Mapping[str, Any] | None]) -> list[str]:
    """One row per capacity task: can this architecture fit the sample it was given."""
    lines = ["### Capacity - can the architecture fit its own training data", "",
             "| Task | Sample rows | Regularisers | Epochs run | train R2 | Threshold | "
             "train pairwise_macro | Verdict |", "|---|---|---|---|---|---|---|---|"]
    for task in CAPACITY_TASKS:
        manifest = manifests.get(task)
        if manifest is None:
            lines.append(f"| {task} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | "
                         f"{MISSING} | {MISSING} | manifest missing |")
            continue
        final = manifest.get("final") or {}
        sample = manifest.get("sample") or {}
        rows = sample.get("rows")
        lines.append(
            f"| {task} | {rows:,} | "
            f"{'as configured' if manifest.get('regularized') else 'off'} | "
            f"{manifest.get('epochs_run')} of {manifest.get('epochs_requested')} | "
            f"{fmt(final.get('train_r2'))} | "
            f"{fmt(manifest.get('fit_train_r2_threshold'), 2)} | "
            f"{fmt(final.get('train_pairwise_accuracy_macro'))} | "
            f"{manifest.get('verdict')} |")
    lines.append("")
    return lines


def protein_table(manifests: Mapping[str, Mapping[str, Any] | None],
                  slice_name: str) -> list[str]:
    """The shuffled-protein control: the same configuration with the protein destroyed."""
    lines = ["### Protein control - the protein column shuffled across rows", "",
             "| Split | real protein | protein shuffled | difference | seeds per side |",
             "|---|---|---|---|---|"]
    for split in (PAIR75, SUB100):
        real = manifests.get(split["mlp"])
        shuffled = manifests.get(split["mlp_shuffled"])
        values = []
        for manifest in (real, shuffled):
            if manifest is None:
                values.append(None)
                continue
            _, ranking = slice_scores(manifest.get("validation_metrics"), slice_name)
            value = ranking.get("pairwise_accuracy_macro")
            values.append(float(value) if isinstance(value, (int, float)) else None)
        if values[0] is None or values[1] is None:
            difference = MISSING
        else:
            difference = f"{values[1] - values[0]:+.4f}"
        lines.append(f"| {split['label']} | {fmt(values[0])} | {fmt(values[1])} | "
                     f"{difference} | 1 |")
    lines.append("")
    lines.append("One seed per side. A difference of this size has no noise scale to be "
                 "read against until the same two runs exist at several seeds, so the "
                 "reading this table supports is \"no effect was measured\", not \"there "
                 "is no effect\".")
    lines.append("")
    return lines


def ablation_table(with_protein: Mapping[str, Any] | None,
                   without_protein: Mapping[str, Any] | None,
                   slice_name: str) -> list[str]:
    """The same baselines with and without the protein block - the classical control."""
    lines = ["### Protein control on the classical side - the block removed, not shuffled",
             "", "| Model | with protein (t1a) | without protein (t1b) | difference |",
             "|---|---|---|---|"]
    if with_protein is None or without_protein is None:
        lines.append(f"| all | {MISSING} | {MISSING} | a manifest is missing |")
        lines.append("")
        return lines
    left = {name.split(",")[0]: block for name, block in classical_rows(with_protein)}
    right = {name.split(",")[0]: block for name, block in classical_rows(without_protein)}
    for model in left:
        if model not in right:
            continue
        values, alphas = [], []
        for block in (left[model], right[model]):
            _, ranking = slice_scores(block.get("validation_metrics"), slice_name)
            value = ranking.get("pairwise_accuracy_macro")
            values.append(float(value) if isinstance(value, (int, float)) else None)
            alphas.append(block.get("alpha"))
        difference = (MISSING if values[0] is None or values[1] is None
                      else f"{values[1] - values[0]:+.6f}")
        # Each side is shown at the alpha ITS OWN run selected; the two can differ, and
        # pretending otherwise would mean quoting a level that lost on one of the sides.
        name = model if alphas[0] == alphas[1] else f"{model} (alpha {alphas[0]} vs {alphas[1]})"
        lines.append(f"| {name} | {fmt(values[0])} | {fmt(values[1])} | {difference} |")
    lines.append("")
    lines.append("Columns are the number of feature columns, not a hyper-parameter: "
                 f"{with_protein.get('feature_columns')} with the protein block, "
                 f"{without_protein.get('feature_columns')} without it.")
    lines.append("")
    return lines


def footnotes(manifests: Mapping[str, Mapping[str, Any] | None],
              slice_name: str, train_slice: str) -> list[str]:
    """What a reader has to know before quoting any row above."""
    lines = ["### Footnotes", ""]
    # The eval-on-train runs are left out of this list on purpose: a missing one is
    # reported by the footnote below, which says what is actually lost (the MLP's train
    # half), so naming it twice would read as two separate gaps.
    absent = [task for task, manifest in manifests.items()
              if manifest is None and task not in EVAL_TASKS]
    if absent:
        lines.append(f"* **No manifest for {', '.join(sorted(absent))}.** Every row of "
                     f"those tasks is `-` above. None of those rows may be filled in from "
                     f"another batch: a different batch is a different split, seed or "
                     f"configuration unless its own manifest says otherwise.")
    for task in (PAIR75["mlp"], PAIR75["mlp_shuffled"], SUB100["mlp"],
                 SUB100["mlp_shuffled"]):
        manifest = manifests.get(task)
        if manifest is None:
            continue
        regression_all, ranking_all = slice_scores(manifest.get("validation_metrics"),
                                                   "all")
        regression_slice, ranking_slice = slice_scores(
            manifest.get("validation_metrics"), slice_name)
        if (slice_name != "all" and regression_all.get("n") == regression_slice.get("n")
                and ranking_all.get("pairwise_accuracy_macro")
                == ranking_slice.get("pairwise_accuracy_macro")):
            lines.append(f"* **{task}: the `{slice_name}` slice holds every validation "
                         f"row** ({regression_slice.get('n'):,}), so its numbers are the "
                         f"`all` numbers. On this split the two slices are not two "
                         f"measurements.")
        if ranking_slice.get("queries_ranked") is not None:
            ranked = ranking_slice["queries_ranked"]
            total = ranking_slice.get("queries_total")
            # `queries_ranked` counts the queries that CLEARED min_candidates. It is not
            # the macro's denominator: a query whose candidates all sit within
            # direction_tolerance of each other contributes no term to the mean yet is
            # still counted here. The real denominator is `pairwise_accuracy_queries`,
            # recorded from 2026-11-20 on; a manifest written before that does not carry
            # it, and this footnote then says what it can rather than implying the
            # smaller set is the larger one. (Measured on t3a: 5,645 cleared the bar,
            # 3,892 entered the macro.)
            counted = ranking_slice.get("pairwise_accuracy_queries")
            if total and counted is not None:
                lines.append(f"* **{task}: {counted:,} of {total:,} queries are in the "
                             f"pairwise macro** ({100.0 * counted / total:.1f} %). "
                             f"{ranked:,} cleared the min_candidates bar; the macro "
                             f"averages only the {counted:,} that also had two "
                             f"candidates further apart than the direction tolerance.")
            elif total:
                lines.append(f"* **{task}: {ranked:,} of {total:,} queries cleared the "
                             f"min_candidates bar** ({100.0 * ranked / total:.1f} %). "
                             f"The macro averages a SUBSET of those - the ones with a "
                             f"countable comparison - and this manifest predates the "
                             f"field that records how many, so the share above is an "
                             f"upper bound on the macro's own denominator.")
    for task in (SUB100["classical"], PAIR75["classical"], "t1b"):
        manifest = manifests.get(task)
        if manifest is None:
            continue
        for name, block in classical_rows(manifest):
            flags = flags_of(block)
            if "unconverged" in flags or "constant" in flags:
                lines.append(f"* **{task}: the selected {name} is flagged** - {flags}. "
                             f"The number above is that fit's number.")
    if train_slice != slice_name:
        lines.append(f"* **The train columns are the `{train_slice}` slice, the "
                     f"validation columns the `{slice_name}` slice.** On a pair-keyed "
                     f"split they cannot be the same: the split table writes "
                     f"`unseen_pair = is_validation` "
                     f"(`scripts/build_pair_split.py`), so every TRAIN row carries false "
                     f"and the `strict` slice of the train partition is empty. A train "
                     f"column headed `strict` on this split would be `-` for a run that "
                     f"finished perfectly.")
    for split in (PAIR75, SUB100):
        mlp, on_train = manifests.get(split["mlp"]), manifests.get(split["mlp_train"])
        if mlp is not None and on_train is not None:
            problems = same_checkpoint(mlp, on_train)
            if problems:
                lines.append(f"* **{split['mlp']}: the train half was refused.** "
                             f"{'; '.join(problems)}. Two halves that are not the same "
                             f"model do not make a gap, so the train columns are `-`.")
        if mlp is not None and on_train is None:
            lines.append(f"* **{split['mlp']}: no train numbers.** The checkpoint was "
                         f"never scored on the train partition in this batch "
                         f"(`{split['mlp_train']}`), so the train-validation gap of the "
                         f"MLP on this split is not in the table. The classical rows' "
                         f"train numbers come from their own manifests and are not a "
                         f"substitute for it.")
    lines.append("* The train and validation columns of one row come from the SAME "
                 "model: the classical rows are scored on the rows they were fitted on, "
                 "and the MLP's train numbers are the selected checkpoint scored on the "
                 "train partition. A train number from one run next to a validation "
                 "number from another would not be a gap.")
    lines.append("* Every validation score above belongs to a configuration that was "
                 "SELECTED on validation (the alpha on the classical side, the checkpoint "
                 "on the MLP side). An independent estimate needs a held-out set that no "
                 "selection touched.")
    lines.append("")
    return lines


def discovered_table(manifests: Mapping[str, Mapping[str, Any] | None],
                     tasks: tuple[tuple[str, str | None], ...], slice_name: str,
                     train_slice: str) -> list[str]:
    """One row per task this file has no fixed table for, read from its manifest.

    WHY. The 2026-11-20 review ran nine simulated t9* runs through the real CLI: the
    figures were drawn, the tables exited 0, and not one row of the nine appeared. Loading
    a manifest is not tabulating it - every table in this file names its tasks, so a batch
    that minted new names produced nothing and said nothing. This is the general row: it
    reports whatever a train manifest can report without knowing what the task means.

    The train column needs the task's `--mode eval --partition train` half and is checked
    with `same_checkpoint`, the same rule the fixed tables use; without it the column is
    `-` rather than the end of the last epoch, which is a different quantity.
    """
    rows = []
    for task, half in tasks:
        manifest = manifests.get(task)
        if manifest is None:
            rows.append((task, None, None, []))
            continue
        other = manifests.get(half) if half else None
        problems = same_checkpoint(manifest, other) if other else ["no train half"]
        rows.append((task, manifest, None if problems else other, problems))
    if not any(manifest is not None for _, manifest, _, _ in rows):
        return []

    out = ["### Every run of this batch", "",
           f"Read from the manifests themselves. Validation columns at the "
           f"`{slice_name}` slice, train columns at `{train_slice}`.", "",
           "| task | configuration | head | split | seed | train rows | "
           f"val MAE | val pw-macro | train MAE | train pw-macro | note |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for task, manifest, half, problems in rows:
        if manifest is None:
            out.append(f"| {task} | - | - | - | - | - | - | - | - | - | "
                       f"no single manifest carries this run id |")
            continue
        regression, ranking = slice_scores(manifest.get("validation_metrics"), slice_name)
        split = (manifest.get("partition_override_pairs")
                 or manifest.get("train_subset") or "FULL")
        note = []
        if manifest.get("measure_train_metrics"):
            note.append(f"train side measured at every point "
                        f"({manifest.get('train_measured_rows')} rows)")
        if problems:
            note.append("; ".join(problems))
        if half is None:
            train_regression: Mapping[str, Any] = {}
            train_ranking: Mapping[str, Any] = {}
        else:
            train_regression, train_ranking = slice_scores(
                half.get("train_metrics") or half.get("validation_metrics"), train_slice)
        out.append(
            f"| {task} | {manifest.get('configuration', MISSING)} | "
            f"{manifest.get('head', MISSING)} | {Path(str(split)).name} | "
            f"{manifest.get('seed', MISSING)} | "
            f"{(manifest.get('rows') or {}).get('train', MISSING)} | "
            f"{fmt(regression.get('mae'))} | "
            f"{fmt(ranking.get('pairwise_accuracy_macro'))} | "
            f"{fmt(train_regression.get('mae'))} | "
            f"{fmt(train_ranking.get('pairwise_accuracy_macro'))} | "
            + ("; ".join(note) if note else "") + " |")
    out += ["",
            "This table knows nothing about what each task was for; it reports what the "
            "manifest holds. A pair of runs is only a comparison if everything but the "
            "one thing under test is equal in these columns - the split, the seed and the "
            "configuration are printed so that can be checked by eye.",
            ""]
    return out


def tables(runs_dir: Path, pid: int, slice_name: str, train_slice: str = "all",
           extra_tasks: tuple[str, ...] = (),
           discovered: tuple[tuple[str, str | None], ...] = ()) -> list[str]:
    """Every table, in report order.

    `extra_tasks` carries the task names read from the batch's own log, so a batch that
    minted names this file has never heard of (the nine t9* runs of 2026-11-20) is
    tabulated instead of producing a table of missing rows.
    """
    tasks = (gate.CLASSICAL_TASKS + gate.TRAIN_TASKS + CAPACITY_TASKS + EVAL_TASKS)
    tasks = tasks + tuple(task for task in extra_tasks if task not in tasks)
    manifests: dict[str, Mapping[str, Any] | None] = {}
    for task in tasks:
        found = gate.manifests_for(runs_dir, task, pid)
        if len(found) != 1:
            manifests[task] = None
            continue
        try:
            manifests[task] = gate.read(found[0])
        except ValueError:
            manifests[task] = None

    lines = [f"## Diagnostics of 2026-11-19, batch pid {pid}", "",
             f"Read from the manifests in `{runs_dir}`. Selection metric and slice of "
             f"every run: `{gate.WANTED_METRIC}` / `{gate.WANTED_SLICE}`.", ""]
    for split in (PAIR75, SUB100):
        lines += baseline_table(manifests[split["classical"]], manifests[split["mlp"]],
                                manifests[split["mlp_train"]], split, slice_name,
                                train_slice)
    lines += capacity_table(manifests)
    lines += protein_table(manifests, slice_name)
    lines += ablation_table(manifests.get("t1a"), manifests.get("t1b"), slice_name)
    # Every task the log launched that no fixed table above covers. Without this the nine
    # t9* runs produced figures and an empty report.
    covered = set(gate.CLASSICAL_TASKS + gate.TRAIN_TASKS + CAPACITY_TASKS + EVAL_TASKS)
    unknown = tuple((task, half) for task, half in discovered if task not in covered)
    lines += discovered_table(manifests, unknown, slice_name, train_slice)
    lines += footnotes(manifests, slice_name, train_slice)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the three diagnostics tables as Markdown, read from manifests.")
    parser.add_argument("runs_dir", help="the directory holding the manifests, normally "
                                         "runs/evaluator")
    parser.add_argument("log", nargs="?", help="the batch's log, to read its pid from")
    parser.add_argument("--pid", type=int, help="the batch's pid, instead of its log")
    parser.add_argument("--slice", default=gate.WANTED_SLICE,
                        help=f"the slice the VALIDATION columns read (default "
                             f"{gate.WANTED_SLICE})")
    parser.add_argument("--train-slice", default="all", dest="train_slice",
                        help="the slice the TRAIN columns read (default all). On a "
                             "pair-keyed split the strict slice of the train partition "
                             "is empty by construction, so it cannot be strict here")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"{runs_dir} is not a directory", file=sys.stderr)
        return 2
    if (args.log is None) == (args.pid is None):
        print("pass either the batch's log or --pid, not both and not neither",
              file=sys.stderr)
        return 2
    extra: tuple[str, ...] = ()
    pairs: tuple[tuple[str, str | None], ...] = ()
    if args.pid is not None:
        pid = args.pid
    else:
        log = Path(args.log)
        if not log.is_file():
            print(f"{log} does not exist", file=sys.stderr)
            return 2
        text = log.read_text(encoding="utf-8", errors="replace")
        try:
            pid = gate.pid_from_log(text)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        # The task names this batch actually used. Given --pid instead of a log there is
        # nothing to read them from, and only the built-in lists are tabulated.
        found = gate.tasks_from_log(text)
        extra = tuple(task for pair in found for task in pair if task)
        pairs = found

    for line in tables(runs_dir, pid, args.slice, args.train_slice, extra_tasks=extra,
                       discovered=pairs):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
