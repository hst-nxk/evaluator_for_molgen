"""CLI for Stage 3 — the evaluator baseline (predicting the Δp of one edit pair).

    python run_evaluator.py --mode prepare  [--skip-protein] [--molecule-embeddings]
                                           [--set key=value ...]
    python run_evaluator.py --mode baseline
    python run_evaluator.py --mode classical [--config frozen|no_protein]
                                           [--baseline linear|ridge|lasso|xgboost|all]
    python run_evaluator.py --mode capacity [--config ...] [--head ...] [--regularized]
    python run_evaluator.py --mode smoke   [--config frozen|finetune|no_protein]
                                           [--head direct|direct_no_edit|twin|antisymmetric]
                                           [--sharpness on|off]
    python run_evaluator.py --mode train   [--config ...] [--head ...]
                                           [--sharpness on|off] [--set key=value ...]
    python run_evaluator.py --mode eval    --checkpoint runs/evaluator/<file>.pt
                                           [--partition validation|train]
    python run_evaluator.py --mode errors  --checkpoint runs/evaluator/<file>.pt
                                           [--partition validation|train]
    python run_evaluator.py --mode features --checkpoint runs/evaluator/<file>.pt
    python run_evaluator.py --mode diagnose --checkpoint runs/evaluator/<file>.pt
                                           [--partition validation|train]
"""
# ---------------------------------------------------------------------------------------
# This file is only the entry point: it reads config.json, turns on logging and calls into
# src/molopt/evaluator. No data handling lives here.
#   prepare  : build the fingerprint cache (RDKit) and the protein embedding cache (ESM2).
#              --skip-protein builds only the fingerprints, for a machine without
#              torch/transformers. --molecule-embeddings additionally builds the molecule
#              embedding cache (MoLFormer), needed only when running
#              --set model.ligand_features=embedding.
#   baseline : the metrics of two trivial baselines (always predict 0; the per-endpoint
#              mean) as a lower reference. Needs neither torch nor the caches, so it runs
#              before prepare.
#   classical: classical ML baselines (Linear/Ridge/Lasso/XGBoost) on EXACTLY the features,
#              the split and the metrics of a real training run — the bar the deep model
#              has to clear (2026-11-18 meeting). Needs scikit-learn, scipy and xgboost.
#   capacity : the capacity check — hand the model a very small sample, switch off dropout
#              and weight decay, run for a long time, then measure on THAT SAME sample.
#              Answers "can the model fit its own data at all" (2026-11-18 meeting).
#   smoke    : train on a small sample, runs on the CPU, only confirms the pipeline works.
#   train    : the real training run; the checkpoint is selected on validation MAE.
#   eval     : load a checkpoint and report the metrics slice by slice.
#   errors   : load a checkpoint and break the errors down by group (task 6, 2026-11-02).
#   features : load a checkpoint and export the post-encoder vectors of a train and a
#              validation sample, for t-SNE (task 4).
#   diagnose : load a checkpoint and measure dead ReLU units and the sensitivity to each
#              input, for when a run predicts something close to a constant.
#
# Order of use: baseline -> prepare -> smoke -> train -> eval.
# The `test` partition is used by no mode at all (see molopt.evaluator.data).
# ---------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from molopt.evaluator import capacity as evaluator_capacity  # noqa: E402
from molopt.evaluator import classical as evaluator_classical  # noqa: E402
from molopt.evaluator import train as evaluator_train  # noqa: E402

MODEL_PREFIX = "model"
CONFIG_PATH = ROOT / "config.json"
MODES = ("prepare", "baseline", "classical", "capacity", "smoke", "train", "eval",
         "errors", "features", "diagnose")
# The --baseline value meaning "run them all"; see molopt.evaluator.classical.MODELS.
ALL_BASELINES = "all"
# Partitions --mode eval may score. No test: test is scored once, at the very end.
EVAL_PARTITIONS = ("validation", "train")
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def load_config(path: Path) -> dict[str, Any]:
    """Read config.json; every Stage 3 parameter lives under the 'stage3' key."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def setup_logging(log_path: Path) -> None:
    """Log to the console and to this run's log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_path, encoding="utf-8")])


# Words accepted for a true/false key of --set. Never bool(raw): bool("false") is True,
# so a typo would TURN ON the very flag the user meant to turn off.
BOOLEAN_WORDS = {"true": True, "false": False, "1": True, "0": False,
                 "on": True, "off": False}


def parse_number_list(raw: str, key: str, current: Sequence[object]) -> list[int | float]:
    """Parse a ``--set`` value of the form ``[]`` or ``[256,128]`` into a list of numbers.

    Takes: the part after ``=``; the key name (for error messages only); the value
    currently in the config.
    Returns: a list of int/float, keeping ints where written without a decimal point.

    Why this path is needed: the hidden-width keys (``model.antisymmetric_hidden``,
    ``model.head_hidden``, ``model.ligand_hidden``) are all lists, and before 2026-11-11
    ``--set`` refused every type that was not a number, string or bool. The consequence was
    that the "single bias-free Linear layer on the difference vector" experiment (an empty
    ``antisymmetric_hidden``) could only be run by hand-editing ``config.json`` — and a
    hand edit changes EVERY run in the same script along with it, so that experiment could
    no longer be separated from the run beside it.

    ONLY lists whose current elements are all numbers are accepted. The other lists in the
    config are lists of names (``configurations``, ``heads``) or lists of dicts
    (``slices``); overriding those from the command line would make the manifest say one
    thing while the data says another, so it is refused.
    """
    for item in current:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"--set: cannot override {key!r} (the list contains "
                             f"{type(item).__name__}; only lists of numbers)")
    text = raw.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"--set: {key!r} is a list, write it in square brackets, e.g. "
                         f"{key}=[256,128] or {key}=[]; got {raw!r}")
    inner = text[1:-1].strip()
    if not inner:
        return []
    values: list[int | float] = []
    for piece in inner.split(","):
        piece = piece.strip()
        if not piece:
            raise ValueError(f"--set: {key!r} has an empty element in {raw!r}")
        try:
            values.append(int(piece) if piece.lstrip("+-").isdigit() else float(piece))
        except ValueError as error:
            raise ValueError(f"--set: {key!r} has a non-numeric element: {piece!r}") from error
    return values


def apply_overrides(settings: dict, pairs: list[str],
                    model_settings: dict | None = None,
                    blocks: dict[str, dict] | None = None) -> dict:
    """Override training parameters from the command line, as ``key=value``.

    Only keys that ALREADY EXIST in ``stage3.train`` and hold a number, a string, a
    true/false or a list of numbers may be overridden: a mistyped key fails immediately
    instead of being silently ignored, because otherwise a whole parameter sweep can finish
    before anyone notices it was meaningless. The type is inferred from the value already in
    the config, so no type has to be declared on the command line.

    Lists of numbers are written in square brackets: ``model.antisymmetric_hidden=[]`` for a
    single bias-free Linear layer, ``model.head_hidden=[256,128]`` for two hidden layers.
    See ``parse_number_list``.

    A dotted key descends into a sub-block, e.g. ``finetune_overrides.batch_size=32``. A key
    starting with ``model.`` is routed to ``stage3.model`` instead of ``stage3.train``, e.g.
    ``model.dropout=0.3`` — dropout is an anti-overfitting knob that nevertheless lives in
    the block describing the architecture, not in the block of training parameters.

    ``blocks`` adds further blocks the same way, block name -> the block's dict. The CLI
    passes ``protein_encoder``, ``molecule_encoder`` and ``paths`` so another encoder can be
    tried without editing config.json::

        --set protein_encoder.model=facebook/esm2_t30_150M_UR50D \
        --set protein_encoder.embedding_dim=640 --set paths.cache_dir=work/cache_esm150
        --set molecule_encoder.model=DeepChem/ChemBERTa-77M-MLM \
        --set paths.cache_dir=work/cache_chemberta

    ``embedding_dim`` has to be passed ALONG WITH the model: ``features.py`` refuses a
    cache build when the declared dimension differs from the model's real
    ``hidden_size`` (480 for ESM2 35M, 640 for 150M, 1280 for 650M), and refusing is the
    right behaviour — the alternative is a cache of the wrong width discovered three hours
    into a run. Changing ``paths.cache_dir`` too is required in practice: the embedding
    cache is a single file, and ``train.Caches`` refuses a cache built with a model other
    than the configured one, so sharing the directory means the new build overwrites the
    cache of the model in use.
    The ``inputs`` block serves the experiments that change the split or the size of the
    training set: ``inputs.partition_override=work/splits/partition_override_id30.parquet``
    and ``inputs.train_subset=work/splits/train_subset_c200_r500.parquet``. Here
    ``paths.cache_dir`` does NOT need changing: the caches are keyed by molecule and by
    protein, not by partition or by row, so one cache serves every threshold and every
    subset.

    The ``metrics`` block changes how a run is SCORED without changing how it trains, e.g.
    ``metrics.expand_both_directions=false``. This path is needed because the current
    both-directions expansion flips the sign of the STORED PREDICTION rather than running
    the model again on the reversed input (see metrics.expand_directions), so it does not
    test the reverse direction but only changes how queries are grouped; switching it off is
    the only way to learn the number without the expansion. Section 11.3 of
    docs/STAGE3_EVALUATOR.md has recorded this check for a long time and nobody has run it.
    Returns: a dict of key -> new value, to be recorded in the run's manifest.
    """
    routes: dict[str, dict | None] = {MODEL_PREFIX: model_settings, **(blocks or {})}
    applied: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set needs the form key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        target = settings
        path = key.split(".")
        if path[0] in routes:
            if routes[path[0]] is None:
                raise ValueError(f"--set: {key!r} is not accepted in this mode")
            target = routes[path[0]]
            prefix = path[0]
            path = path[1:]
            if not path or not all(path):
                raise ValueError(f"--set: missing a parameter name after '{prefix}.'")
        for step in path[:-1]:
            if not isinstance(target, dict) or step not in target:
                raise ValueError(f"--set: no block {step!r} inside {key!r}")
            target = target[step]
        leaf = path[-1]
        if not isinstance(target, dict) or leaf not in target:
            raise ValueError(f"--set: no parameter {key!r} in stage3.train")
        current = target[leaf]
        if isinstance(current, list):
            target[leaf] = parse_number_list(raw, key, current)
            applied[key] = target[leaf]
            continue
        if not isinstance(current, (bool, int, float, str)):
            raise ValueError(f"--set: cannot override {key!r} "
                             f"(type {type(current).__name__})")
        value: object = raw
        # bool has to be checked BEFORE int (bool is a subclass of int) and parsed
        # explicitly: bool("false") is True, so letting Python convert would turn on the very
        # flag the user just turned off.
        if isinstance(current, bool):
            if raw.strip().lower() not in BOOLEAN_WORDS:
                raise ValueError(f"--set: {key!r} is true/false, got {raw!r}; "
                                 f"write one of {sorted(BOOLEAN_WORDS)}")
            value = BOOLEAN_WORDS[raw.strip().lower()]
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        target[leaf] = value
        applied[key] = value
    return applied


# Characters allowed in --run-id. Everything else is refused rather than stripped: the id
# goes into file names, and an id that is silently rewritten no longer matches the one the
# caller is waiting for.
RUN_ID_CHARACTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")


def run_stamp(moment: datetime, run_id: str | None) -> str:
    """The stamp that names this run's files: timestamp, plus ``run_id`` when given.

    Takes: the moment the run starts; the ``--run-id`` value or None.
    Returns: ``20261119_084500`` or ``20261119_084500_3a-5417``.

    WHY A RUN ID EXISTS. The timestamp is only accurate to the second, and every mode
    writes into one shared ``runs/evaluator/``. Two consequences, both of which produce
    numbers for a model nobody chose:

    * two runs started in the same second write the SAME checkpoint and manifest path, so
      one overwrites the other;
    * a batch script that wants the checkpoint of the run it just finished cannot tell its
      own manifest from the manifest of a second batch running on another GPU, because
      "the newest manifest" belongs to whichever job happened to finish last.

    With an id passed in by the caller, the manifest is ``train_<configuration>_<head>_
    <timestamp>_<id>.json`` and the caller can open exactly that one. See
    ``scripts/pick_train_checkpoint.py`` and ``scripts/server_run_diagnostics_1911.sh``.
    """
    stamp = moment.strftime(evaluator_train.TIMESTAMP_FORMAT)
    if not run_id:
        return stamp
    unknown = sorted(set(run_id) - RUN_ID_CHARACTERS)
    if unknown:
        raise ValueError(f"--run-id may only hold letters, digits and '-'; got {run_id!r} "
                         f"with {unknown}")
    return f"{stamp}_{run_id}"


def main() -> int:
    """Parse the command line and run the chosen mode. Returns the exit code (0 = done)."""
    config = load_config(CONFIG_PATH)
    settings = config["stage3"]["train"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--config", dest="configuration", default="frozen",
                        choices=list(settings["configurations"]),
                        help="protein branch configuration (default frozen: the cheapest)")
    parser.add_argument("--head", default=settings["head"], choices=list(settings["heads"]),
                        help="direct: predict Δp straight away; direct_no_edit: like "
                             "direct but without the edit branch; twin: score each molecule "
                             "separately and take the difference (antisymmetric by "
                             "construction); antisymmetric: an odd function of the "
                             "difference vector times a context gate (antisymmetric by "
                             "construction, still able to learn within-pair interaction)")
    parser.add_argument("--skip-protein", action="store_true",
                        help="prepare: build only the fingerprint cache, skip ESM2")
    parser.add_argument("--molecule-embeddings", action="store_true",
                        help="prepare: additionally build the molecule embedding cache "
                             "(MoLFormer), needed for --set model.ligand_features=embedding")
    parser.add_argument("--sharpness", choices=("on", "off"), default=None,
                        help="smoke/train: switch the output gradient-norm penalty on or "
                             "off (IGNITE, arXiv:2503.04242); left out, the config key "
                             "stage3.train.sharpness.enabled decides")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override one parameter without editing config.json, e.g. "
                             "--set weight_decay=0.1 (stage3.train), --set model.dropout=0.6 "
                             "(stage3.model), --set protein_encoder.model=... with "
                             "--set protein_encoder.embedding_dim=... and "
                             "--set paths.cache_dir=..., --set molecule_encoder.model=... "
                             "(also used by prepare); may be repeated")
    parser.add_argument("--baseline", dest="baselines", action="append", default=[],
                        choices=list(evaluator_classical.MODELS) + [ALL_BASELINES],
                        help="classical: which baseline to run; may be repeated, default "
                             "all (run them all, cheapest first)")
    parser.add_argument("--regularized", action="store_true",
                        help="capacity: KEEP the configured dropout and weight decay "
                             "instead of switching them off; only for measuring whether "
                             "those two are what blocks fitting")
    parser.add_argument("--run-id", dest="run_id", default=None,
                        help="a label added to every file this run writes, so that two "
                             "runs started in the same second cannot overwrite each "
                             "other's checkpoint and a batch script can find its OWN "
                             "manifest while another job writes into the same directory "
                             "(see scripts/pick_train_checkpoint.py)")
    parser.add_argument("--checkpoint", type=Path,
                        help="eval, errors, features, diagnose: the .pt file to load")
    parser.add_argument("--partition", default=EVAL_PARTITIONS[0], choices=EVAL_PARTITIONS,
                        help="eval, errors, diagnose: the partition to score; use train to "
                             "compare against validation and see whether the model overfits "
                             "or has not learned (test is always refused)")
    args = parser.parse_args()

    try:
        overrides = apply_overrides(settings, args.overrides, config["stage3"]["model"],
                                    {"protein_encoder": config["stage3"]["protein_encoder"],
                                     # molecule_encoder too: ChemBERTa is already in
                                     # allowed_models, but with no --set route the only way
                                     # to reach it was editing config.json by hand - which
                                     # changes EVERY run of a batch script at once.
                                     "molecule_encoder": config["stage3"]["molecule_encoder"],
                                     "paths": config["stage3"]["paths"],
                                     "inputs": config["stage3"]["inputs"],
                                     "metrics": config["stage3"]["metrics"],
                                     "classical": config["stage3"]["classical"],
                                     "capacity": config["stage3"]["capacity"]})
    except ValueError as error:
        parser.error(str(error))

    try:
        stamp = run_stamp(datetime.now(), args.run_id)
    except ValueError as error:
        parser.error(str(error))
    runs_dir = ROOT / config["stage3"]["paths"]["runs_dir"]
    setup_logging(runs_dir / f"{args.mode}_{stamp}.log")
    if overrides:
        logging.info("parameter overrides: %s", overrides)

    if args.mode == "prepare":
        evaluator_train.run_prepare(config, ROOT, args.skip_protein,
                                    args.molecule_embeddings)
        return 0
    if args.mode == "baseline":
        evaluator_train.run_baselines(config, ROOT, stamp)
        return 0
    if args.mode == "classical":
        chosen = args.baselines or [ALL_BASELINES]
        baselines = (list(evaluator_classical.MODELS) if ALL_BASELINES in chosen
                     else [name for name in evaluator_classical.MODELS if name in chosen])
        evaluator_classical.run_classical(config, ROOT, args.configuration, baselines,
                                          stamp, overrides)
        return 0
    if args.mode == "capacity":
        evaluator_capacity.run_capacity(config, ROOT, args.configuration, args.head,
                                        stamp, regularized=args.regularized,
                                        overrides=overrides)
        return 0
    if args.mode in ("smoke", "train"):
        evaluator_train.run_train(config, ROOT, args.configuration, args.head,
                                  smoke=args.mode == "smoke", stamp=stamp,
                                  sharpness_mode=args.sharpness, overrides=overrides)
        return 0
    if args.checkpoint is None:
        parser.error(f"--mode {args.mode} needs --checkpoint")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.mode == "errors":
        evaluator_train.run_errors(config, ROOT, args.checkpoint, stamp, args.partition)
    elif args.mode == "features":
        evaluator_train.run_features(config, ROOT, args.checkpoint, stamp)
    elif args.mode == "diagnose":
        evaluator_train.run_diagnose(config, ROOT, args.checkpoint, stamp, args.partition)
    else:
        evaluator_train.run_eval(config, ROOT, args.checkpoint, stamp, args.partition)
    return 0


if __name__ == "__main__":
    sys.exit(main())
