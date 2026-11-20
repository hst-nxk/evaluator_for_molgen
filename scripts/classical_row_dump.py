"""Write a `--mode errors` style row dump for a CLASSICAL baseline, so the seen/unseen
molecule breakdown can be run on the baselines too.

    python scripts/classical_row_dump.py runs/evaluator/classical_frozen_..._t1c-4174688.json
    python scripts/classical_row_dump.py <manifest> --models ridge,xgboost

WHY THIS EXISTS. The breakdown of 2026-11-20 showed where the MLP's PAIR75 score comes
from: on validation rows whose two molecules both appeared in training it gets the
direction right 80.9 % of the time, and on rows where neither did, 56.8 %. The obvious
next question is whether the MLP's advantage over the classical baselines (+0.0977 on
pairwise_accuracy_macro) survives on the rows with new molecules. It cannot be answered
from the numbers already in hand: the MLP's 0.6673 for that group is a macro over 215
queries, while xgboost's 0.6801 is a macro over all 3,892 - two different sets of queries,
so the two numbers do not compare. The baselines have to be scored on the SAME groups.

`--mode errors` only knows how to dump a checkpoint, and a classical fit has no
checkpoint: `run_classical` fits, scores, records the numbers and throws the estimator
away. This script refits the baseline the manifest describes, predicts the validation
partition, and writes the dump in the layout `scripts/seen_molecules.py` already reads, so
that analysis runs over a baseline without being modified at all.

THE REFIT HAS TO BE THE SAME MODEL, and that is checked rather than assumed. Everything
that decides the fit comes out of the manifest - the configuration, the split, the
selected alpha, the row cap and the seed that picks the capped sample - and before any
dump is written the refit's own validation metrics are compared against the metrics the
manifest recorded. A mismatch means this is not the model whose numbers were reported, so
nothing is written and the script exits non-zero. There is no flag to skip that.

WHAT IT DOES NOT DO. It does not touch the MLP side, it does not re-select an alpha (it
uses the one the run selected, and refuses a regularised model that has none), and it
refuses a manifest whose protein column was shuffled: a dump of a protein-control run
would quietly answer a different question.

COST. CPU only, no GPU. One fit per model on the capped train rows plus one prediction
pass over the validation partition - minutes per model, and the feature matrix is the
memory high-water mark, exactly as in the original run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from molopt.evaluator import analysis, classical, data, train  # noqa: E402

# How close the refit has to land to the recorded numbers. Every model here is
# deterministic given its seed (xgboost included, which is why the seed is read from the
# manifest), so this is a guard against the wrong model or the wrong rows, not a
# tolerance for genuine variation.
TOLERANCE = 1e-9
CHECKED_METRICS = ("mae", "r2")
CHECKED_RANKING = ("pairwise_accuracy_macro", "pairwise_accuracy_micro")


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "stage3_classical":
        raise ValueError(f"{path} is not a classical manifest (stage "
                         f"{manifest.get('stage')!r})")
    return manifest


def refusals(manifest: dict[str, Any]) -> list[str]:
    """Reasons this manifest must not be turned into a dump."""
    problems = []
    permute = int(manifest.get("protein_permute_seed", -1))
    if permute >= 0:
        problems.append(
            f"the run recorded protein_permute_seed {permute}, so its protein column was "
            f"SHUFFLED. A seen/unseen molecule breakdown of a protein-control run would "
            f"answer a different question than the one it is quoted for")
    if not manifest.get("partition_override_pairs"):
        problems.append(
            "the run carries no partition_override_pairs, so it is not one of the "
            "pair-keyed splits the breakdown is about")
    return problems


def alpha_for(manifest: dict[str, Any], model: str) -> tuple[float, str]:
    """(the alpha to refit with, the label the manifest filed that fit under).

    For a regularised model this is the alpha the RUN selected - never the first of the
    grid, never a new selection made here. A model whose alpha could not be selected is
    refused: quoting a fit the run did not choose would put a number in a table that the
    run never reported.
    """
    results = manifest.get("results") or {}
    if model in (classical.LINEAR, classical.XGBOOST):
        labels = [label for label, block in results.items()
                  if str(block.get("model")) == model]
        if not labels:
            raise ValueError(f"the manifest holds no {model} fit")
        return float(results[labels[0]].get("alpha") or 0.0), labels[0]
    selected = (manifest.get("selected_alpha") or {}).get(model)
    if selected is None:
        unavailable = manifest.get("selected_alpha_unavailable") or []
        reason = (f"{model} is listed in selected_alpha_unavailable"
                  if model in unavailable else
                  f"the manifest records no selected alpha for {model}")
        raise ValueError(f"{reason}; refusing to pick one here")
    if isinstance(selected, dict):
        # `best_alpha` files the winner's own label beside its alpha; using that label
        # rather than rebuilding one means a change in how labels are spelled cannot
        # silently point this script at a different fit.
        alpha = float(selected["alpha"])
        label = str(selected.get("label") or classical.run_label(model, alpha))
    else:
        alpha = float(selected)
        label = classical.run_label(model, alpha)
    if label not in results:
        raise ValueError(f"the selected alpha {alpha} of {model} has no fit in the "
                         f"manifest under {label!r}")
    return alpha, label


def settings_from(manifest: dict[str, Any]) -> dict[str, Any]:
    """The config overrides that make this process read what the run read."""
    inputs = {"partition_override_pairs": manifest.get("partition_override_pairs") or "",
              "partition_override": manifest.get("partition_override") or "",
              "train_subset": manifest.get("train_subset") or ""}
    return {f"inputs.{name}": value for name, value in inputs.items() if value}


def load_config(manifest: dict[str, Any]) -> dict[str, Any]:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    stage = config["stage3"]
    for key, value in settings_from(manifest).items():
        block, name = key.split(".", 1)
        stage[block][name] = value
    # The seed decides which rows the cap keeps, so it has to be the run's own seed and
    # not today's config value.
    stage["train"]["seed"] = int(manifest["seed"])
    stage["classical"]["max_train_rows"] = int(manifest["max_train_rows"])
    stage["classical"]["chunk_rows"] = int(manifest["chunk_rows"])
    return config


def differences(recorded: dict[str, Any], measured: dict[str, Any]) -> list[str]:
    """Where the refit's metrics disagree with the ones the manifest recorded."""
    problems = []
    for slice_name, recorded_slice in (recorded or {}).items():
        measured_slice = (measured or {}).get(slice_name)
        if measured_slice is None:
            problems.append(f"the refit produced no {slice_name!r} slice")
            continue
        for part, names in (("regression", CHECKED_METRICS),
                            ("ranking", CHECKED_RANKING)):
            for name in names:
                first = (recorded_slice.get(part) or {}).get(name)
                second = (measured_slice.get(part) or {}).get(name)
                if first is None or second is None:
                    if first != second:
                        problems.append(f"{slice_name}/{part}/{name}: manifest "
                                        f"{first!r}, refit {second!r}")
                    continue
                if abs(float(first) - float(second)) > TOLERANCE:
                    problems.append(f"{slice_name}/{part}/{name}: manifest {first:.6f}, "
                                    f"refit {second:.6f}")
    return problems


def dump(folder: Path, table: Any, arrays: Any, keys: dict[str, np.ndarray],
         prediction: np.ndarray, caches: Any, vocabulary: Any, config: dict[str, Any],
         extra: dict[str, Any]) -> None:
    """The two parquet files and the report, in the layout `--mode errors` writes.

    Only the columns the analysis scripts actually read are written: `error_analysis`
    requires delta_p, prediction, sign_counted, sign_correct, protein_hash and
    protein_rows, and `seen_molecules` adds smiles_a, smiles_b, unseen_pair and endpoint.
    The per-row rarity columns of a real dump that nothing here reads (molecule_a_rows and
    the rest) are left out rather than filled with a stand-in, and `report.json` says the
    dump came from this script.
    """
    metric_settings = config["stage3"]["metrics"]
    tolerance = float(metric_settings["direction_tolerance"])
    truth = arrays.target.astype(np.float64)
    predicted = prediction.astype(np.float64)
    folder.mkdir(parents=True, exist_ok=True)
    counted, correct = analysis.sign_agreement(truth, predicted, tolerance)
    pq.write_table(pa.table({
        "pair_id": table.column("pair_id"),
        "endpoint": table.column("endpoint"),
        "protein_hash": table.column("protein_hash"),
        "smiles_a": table.column("smiles_a"),
        "smiles_b": table.column("smiles_b"),
        "unseen_pair": table.column("unseen_pair"),
        "delta_p": pa.array(truth),
        "prediction": pa.array(predicted),
        "error": pa.array(predicted - truth),
        "sign_counted": pa.array(counted),
        "sign_correct": pa.array(correct),
        "protein_rows": pa.array(analysis.code_frequency(arrays.protein)),
    }), folder / "rows.parquet")
    queries = analysis.query_table(keys, truth, predicted, tolerance,
                                   int(metric_settings["min_candidates"]),
                                   bool(metric_settings["expand_both_directions"]))
    endpoints = list(config["endpoints"])
    pq.write_table(pa.table({
        "reference_smiles": caches.molecule_keys.take(pa.array(queries["reference"])),
        "protein_hash": vocabulary.take(pa.array(queries["protein"])),
        "endpoint": pa.array([endpoints[int(code)] for code in queries["endpoint"]],
                             pa.string()),
        **{name: pa.array(queries[name]) for name in ("candidates", "agree", "counted",
                                                      "spearman", "top_1")},
    }), folder / "queries.parquet")
    report = dict(extra, stage="stage3_errors_classical",
                  generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                  partition=data.VALIDATION, rows=len(arrays),
                  direction_tolerance=tolerance,
                  metrics_settings={key: value for key, value in metric_settings.items()
                                    if not key.endswith("note")},
                  note=("Written by scripts/classical_row_dump.py, not by --mode errors: "
                        "a classical fit has no checkpoint to dump, so the baseline was "
                        "refitted from its manifest and checked against the metrics that "
                        "manifest recorded before this folder was written."))
    (folder / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                        encoding="utf-8")


def build(manifest_path: Path, models: list[str], out_root: Path | None,
          log=print) -> int:
    manifest = read_manifest(manifest_path)
    problems = refusals(manifest)
    if problems:
        log("THIS MANIFEST MUST NOT BE DUMPED:")
        for problem in problems:
            log(f"  - {problem}")
        return 1

    config = load_config(manifest)
    configuration = str(manifest["configuration"])
    settings = config["stage3"]["classical"]
    endpoint_count = len(config["endpoints"])
    chunk_rows = int(settings["chunk_rows"])
    seed = int(manifest["seed"])
    use_protein = str(manifest["protein_mode"]) == "frozen"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    _, runs_dir = train.layout(config, ROOT)
    out_root = out_root or runs_dir

    log(f"manifest  : {manifest_path.name}")
    log(f"split     : {manifest.get('partition_override_pairs')}")
    log(f"features  : {manifest.get('feature_columns')} columns, protein block "
        f"{'in' if use_protein else 'out'}")

    caches = train.Caches(config, ROOT, need_protein=use_protein)
    runtime = train.build_protein_runtime(config, ROOT, configuration, caches, "cpu")
    train_arrays, train_keys, _ = train.prepare_arrays(config, ROOT, data.TRAIN, caches,
                                                       runtime.vocabulary, None)
    validation_arrays, validation_keys, validation_table = train.prepare_arrays(
        config, ROOT, data.VALIDATION, caches, runtime.vocabulary, None)

    # The same capped sample of train rows the run fitted on, picked by the same seed.
    cap = int(settings["max_train_rows"])
    rows = np.arange(len(train_arrays))
    if cap > 0 and len(train_arrays) > cap:
        rows = np.sort(np.random.default_rng(seed).permutation(len(train_arrays))[:cap])
    fit_arrays = data.take_rows(train_arrays, rows)
    recorded_rows = (manifest.get("rows") or {}).get("train_fit")
    if recorded_rows is not None and int(recorded_rows) != len(fit_arrays):
        log(f"REFUSED: the manifest fitted {int(recorded_rows):,} train rows, this "
            f"process reconstructed {len(fit_arrays):,}. The dump would not be the "
            f"model whose numbers were reported.")
        return 1
    log(f"fit rows  : {len(fit_arrays):,} (cap {cap:,}, seed {seed})")

    matrix = classical.build_matrix(fit_arrays, np.arange(len(fit_arrays)), caches,
                                    endpoint_count, use_protein, chunk_rows)
    target = fit_arrays.target.astype(np.float64)

    written, failed = [], 0
    for model in models:
        try:
            alpha, label = alpha_for(manifest, model)
        except ValueError as error:
            log(f"{model}: REFUSED - {error}")
            failed += 1
            continue
        log(f"{label}: refitting")
        estimator = classical.build_estimator(model, alpha, settings, seed)
        estimator.fit(matrix, target)
        prediction = classical.predict_rows(estimator, validation_arrays, caches,
                                            endpoint_count, use_protein, chunk_rows)
        measured = train.evaluate(validation_arrays, prediction, validation_keys, config)
        recorded = (manifest["results"][label] or {}).get("validation_metrics")
        disagreements = differences(recorded, measured)
        if disagreements:
            log(f"{label}: REFUSED - the refit does not reproduce the manifest:")
            for line in disagreements:
                log(f"    {line}")
            failed += 1
            continue
        folder = out_root / f"errors_classical_{model}_{configuration}_validation_{stamp}"
        dump(folder, validation_table, validation_arrays, validation_keys, prediction,
             caches, runtime.vocabulary, config,
             {"model": model, "alpha": alpha, "label": label,
              "configuration": configuration,
              "source_manifest": str(manifest_path),
              "partition_override_pairs": manifest.get("partition_override_pairs") or "",
              "seed": seed})
        log(f"{label}: reproduces the manifest; dump -> {folder.name}")
        written.append((label, folder))

    log("")
    if not written:
        log("No dump was written.")
        return 1
    log(f"{len(written)} dump(s) written. Each one reproduces its manifest's recorded "
        f"validation metrics to {TOLERANCE:g}, so the breakdown below is of the very fit "
        f"the manifest reports.")
    log("Run the breakdown with the TRAIN dump of the MLP run on the same split - the "
        "train side only supplies the set of molecules seen in training, which is a "
        "property of the split and not of the model:")
    for label, folder in written:
        log(f"  python scripts/seen_molecules.py --train <t6train dump> \\")
        log(f"    --validation {folder} --slice all --label {label!r}")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="a classical manifest in runs/evaluator")
    parser.add_argument("--models", default="all",
                        help="comma separated, or 'all' for every model the manifest holds")
    parser.add_argument("--out-root", default=None,
                        help="where to write the dump folders (default: the runs dir)")
    args = parser.parse_args(argv)
    path = Path(args.manifest)
    if not path.is_file():
        print(f"no such manifest: {path}", file=sys.stderr)
        return 2
    if args.models == "all":
        models = [name for name in classical.MODELS
                  if any(str(block.get("model")) == name for block in
                         (read_manifest(path).get("results") or {}).values())]
    else:
        models = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in models if name not in classical.MODELS]
    if unknown:
        print(f"unknown model(s): {unknown}; only {list(classical.MODELS)}",
              file=sys.stderr)
        return 2
    if not models:
        print("the manifest holds no fitted model to dump", file=sys.stderr)
        return 2
    return build(path, models,
                 Path(args.out_root) if args.out_root else None)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
