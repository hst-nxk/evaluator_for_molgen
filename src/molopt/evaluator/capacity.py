"""Stage 3: capacity check — can the model fit a very small sample at all?

WHY THIS MODULE EXISTS. The 2026-11-18 meeting left two contradictory readings of the
same observation (quotes translated from the Vietnamese minutes):

    Supervisor : "an architecture this size, with the tiny amount of data you feed it,
                  cannot possibly train to a good result."
    Reviewer   : "but the point is that it is not even overfitting. Too little data and
                  too large a model should, in principle, overfit once trained to the
                  limit."

Argument cannot settle that; one direct check can. Give the model a VERY small sample,
switch off everything that fights fitting (dropout, weight decay), let it run for a long
time without early stopping on validation, then score it on THAT SAME sample.

    Fits     -> lack of capacity is not the explanation for the model failing to overfit
                real data. What is left to blame is the input REPRESENTATION or the noise
                in the labels, not the size of the model.
    Does not -> the fault is somewhere else entirely: a branch that receives no gradient,
                a representation that cannot tell two rows with different labels apart,
                or self-contradictory labels (same input, two labels — which does occur
                in MMP data).

The verdict only applies to the exact architecture and hyper-parameters of the run. The
config default is the `plain` 512-256 head, so drawing a conclusion about the residual
7x102 architecture of the reference sweeps requires passing that sweep's `--set` flags.

This is a DIAGNOSTIC, not a training run meant to report results. It saves no checkpoint
and never touches validation: the only thing to learn is whether the model can fit the
sample it was just handed.

One limitation to state when reading the result: the small sample is drawn from the first
``read_rows`` rows of the train partition, not from all of train. The dataset is written
in (endpoint, protein bucket) order, and MEASURED on 2026-11-19 the first 200,000 train
rows of dataset_v4 are ALL `Ki` — so the default window does not cover every protein and
holds a single endpoint. The verdict of a default run is therefore "this configuration can
(or cannot) fit 1,024 Ki rows", which is exactly the question "can it fit SOME small
sample" and nothing more. The manifest records `distinct_endpoints` and
`distinct_proteins` so this is visible in the result and not only here.

To ask instead "can it fit the 100k rows of SUB100", point the run at that subset and
widen both the sample and the window::

    python run_evaluator.py --mode capacity --config frozen --head direct \
      --set inputs.train_subset=work/splits/train_subset_c200_r500.parquet \
      --set capacity.rows=100000 --set capacity.read_rows=100000

That run fits every row it reads, so there is no sampling window left to skew it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import data, train

LOGGER = logging.getLogger(__name__)

# EVERY dropout key of the `stage3.model` block. All of them must go off, not just
# `dropout`: the residual block has its own `residual_hidden_dropout` and
# `residual_dropout` (the 2026-11-14 reference configuration sets 0.3936), and the
# `attention` fusion has `attention_dropout`. Switching off `dropout` alone writes 0 into
# the manifest while the layers keep dropping units — a capacity check run with one hand
# tied and nobody able to see it. The "negative means use model.dropout" convention is no
# help here: these must be set to an explicit 0.0, see models.dropout_or_default.
DROPOUT_KEYS = ("dropout", "residual_hidden_dropout", "residual_dropout",
                "attention_dropout")

# Verdict of the check, written straight into the manifest so nobody has to re-read the
# table of numbers to find out how it came out.
FIT = "fits"
NOT_FIT = "does_not_fit"
UNDECIDED = "undecided"


def sample_rows(total: int, rows: int, seed: int) -> np.ndarray:
    """Indices of ``rows`` rows drawn without replacement out of ``total`` rows.

    Random sampling rather than the first rows: the dataset is written in (endpoint,
    protein bucket) order, so the head of a read window is systematically skewed towards
    one endpoint — and a small sample containing a single endpoint is easy to fit for the
    wrong reason. With ``rows`` >= ``total`` the whole range is returned in original order.
    """
    if total <= 0:
        raise ValueError("no rows to sample from")
    if rows >= total:
        return np.arange(total)
    return np.sort(np.random.default_rng(seed).permutation(total)[:rows])


def verdict(train_r2: float | None, threshold: float) -> str:
    """Verdict of the check, from R² measured on the very sample that was trained on."""
    if train_r2 is None:
        return UNDECIDED
    return FIT if train_r2 >= threshold else NOT_FIT


def verdict_note(regularized: bool, sample: Mapping[str, Any],
                 dropouts: Mapping[str, float] | None = None,
                 weight_decay: float | None = None, epochs_run: int | None = None) -> str:
    """The sentence that goes with the verdict in the manifest.

    Takes: whether the run KEPT the configured regularisers (``--regularized``); the
    ``sample`` block of the manifest; the dropout keys and their values as actually used;
    the weight decay as actually used; how many epochs were run.
    Returns: one paragraph stating what the verdict does and does not license.

    Why this is built instead of being one fixed string. A ``--regularized`` run is the
    opposite experiment, so a note saying "weight decay and every dropout were switched
    off" would describe it incorrectly in the one field a reader goes to for the
    conclusion. And "the knobs are ON" is just as wrong when ``--regularized`` runs with
    `weight_decay=0.0` passed on the command line, as the 2026-11-19 batch does: the pair
    of runs then isolates dropout alone. So the note states the VALUES that were used
    rather than a word for them. It also repeats what the sample held, because a sample
    with ONE endpoint is easier to fit than the dataset, and how many epochs were run,
    because "did not fit within this budget" is not "cannot fit".
    """
    values = (", ".join(f"{key}={value:g}" for key, value in dropouts.items())
              if dropouts else "no dropout key in the model block")
    decay = "unknown" if weight_decay is None else f"{weight_decay:g}"
    knobs = (f"The regularisers were used AS CONFIGURED (--regularized): {values}; weight "
             f"decay {decay}" if regularized else
             f"Weight decay and every dropout were switched off: {values}; weight decay "
             f"{decay}")
    scope = (f"The sample held {sample.get('rows')} rows, "
             f"{sample.get('distinct_endpoints')} endpoint(s) and "
             f"{sample.get('distinct_proteins')} protein(s) out of a read window of "
             f"{sample.get('window_rows')} rows")
    budget = ("" if epochs_run is None else
              f", reached after {epochs_run} epoch(s) of training on that sample")
    fits = (f"fits: the model reached R2 >= threshold ON THIS SAMPLE{budget}. That is all "
            f"it establishes: for this architecture, these hyper-parameters and this "
            f"sample, the fit is attainable, so a lack of capacity is not what stops this "
            f"configuration from fitting a sample of this size. It says nothing about "
            f"100k rows, about every endpoint, or about which branch does the work — this "
            f"check does NOT inspect per-branch gradients; use --mode diagnose "
            f"--partition train for that.")
    fails = ("does_not_fit: what the check measured is that the target R2 was NOT reached "
             "on this sample within the epochs that were run — nothing more. Why it was "
             "not reached is a separate question with at least four candidate answers, to "
             "be told apart afterwards: the gradient path (use --mode diagnose "
             "--partition train), the input representation (two rows with different labels "
             "sharing one feature vector), self-contradictory labels, and the optimisation "
             "schedule plus the number of update steps. \"Not within this budget\" is not "
             "\"cannot\".")
    if regularized:
        fails = ("does_not_fit: with the regularisers configured, this does NOT mean a "
                 "lack of capacity; read it against the run with the same "
                 "hyper-parameters and the regularisers off, which is what isolates them. "
                 + fails)
    return (f"{fits} {fails} {knobs}. {scope}. Neither verdict says anything about "
            f"prediction quality on unseen data, and both apply only to the exact "
            f"architecture, hyper-parameters and sample of this run.")


def run_capacity(config: Mapping[str, Any], root: Path, configuration: str, head: str,
                 stamp: str, regularized: bool = False,
                 overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run the capacity check; returns the manifest (already written to disk).

    Takes: config; project root; protein-branch configuration; output head; run stamp;
    ``regularized`` keeps the configured dropout and weight decay instead of switching
    them off (only for MEASURING whether those two are what blocks fitting);
    ``overrides`` is the list of applied ``--set`` flags, recorded in the manifest only.
    """
    from . import models

    settings = config["stage3"]["train"]
    limits = config["stage3"]["capacity"]
    device = train.resolve_device(settings["device"])
    train.limit_gpu_memory(settings, device)
    seed = int(settings["seed"])
    train.set_seeds(seed)
    _, runs_dir = train.layout(config, root)

    # Switch off dropout and weight decay: this check measures the CAPACITY to fit, so
    # everything deliberately standing in the way of fitting has to go. Done on copies,
    # never in place, so the manifest can report the configured values next to the ones
    # actually used.
    fit_settings = dict(settings)
    model_settings = dict(config["stage3"]["model"])
    if not regularized:
        fit_settings["weight_decay"] = 0.0
        for key in DROPOUT_KEYS:
            if key in model_settings:
                model_settings[key] = 0.0
    dropouts = {key: float(model_settings[key]) for key in DROPOUT_KEYS
                if key in model_settings}
    LOGGER.info("capacity check: %s | weight decay %.4g | %s",
                " | ".join(f"{key} {value:.4g}" for key, value in dropouts.items()),
                float(fit_settings["weight_decay"]),
                "KEEPING the configured values (--regularized)" if regularized
                else "weight decay and EVERY dropout switched off")

    protein_mode = models.protein_mode_for(configuration)
    caches = train.Caches(config, root, need_protein=protein_mode == models.PROTEIN_FROZEN)
    runtime = train.build_protein_runtime(config, root, configuration, caches, device)

    window = int(limits["read_rows"])
    all_arrays, all_keys, _ = train.prepare_arrays(config, root, data.TRAIN, caches,
                                                   runtime.vocabulary, window)
    chosen = sample_rows(len(all_arrays), int(limits["rows"]), seed)
    arrays = data.take_rows(all_arrays, chosen)
    keys = {name: column[chosen] for name, column in all_keys.items()}
    sample = {
        "rows": len(arrays), "read_window": window, "window_rows": len(all_arrays),
        "distinct_proteins": int(np.unique(arrays.protein).size),
        "distinct_endpoints": int(np.unique(arrays.endpoint).size),
        "distinct_molecules": int(np.unique(
            np.concatenate([arrays.molecule_a, arrays.molecule_b])).size),
        "target_std": float(np.std(arrays.target)),
    }
    LOGGER.info("small sample: %s rows | %s proteins | %s endpoints | %s molecules | "
                "label sd %.4f", f"{sample['rows']:,}", sample["distinct_proteins"],
                sample["distinct_endpoints"], f"{sample['distinct_molecules']:,}",
                sample["target_std"])

    model = models.EvaluatorModel(model_settings, caches.ligand_bits, caches.fragment_bits,
                                  runtime.dimension, len(config["endpoints"]), head,
                                  protein_mode)
    model.to(device)
    optimizer = train.build_optimizer(model, runtime.source, fit_settings)
    criterion = train.loss_function(fit_settings)
    # Batch size and batch grouping must come through run_train's own path instead of
    # reading `batch_size` straight from the config: under the finetune configuration ESM2
    # sits inside the computation graph, so memory scales with the number of DISTINCT
    # proteins per batch — which is why `finetune_overrides` and protein-grouped batches
    # exist. Ignoring both runs out of VRAM on the first batch.
    configured_batch, _ = train.effective_batch_sizes(fit_settings, protein_mode,
                                                      smoke=False)
    batch_size = min(configured_batch, len(arrays))
    group_by_protein = protein_mode == models.PROTEIN_FINETUNE
    rng = np.random.default_rng(seed)
    epochs = int(limits["epochs"])
    threshold = float(limits["fit_train_r2"])
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(epochs):
        loss, _, steps = train.train_one_epoch(
            model, arrays, caches, runtime.source, device, optimizer, criterion,
            batch_size, float(fit_settings["grad_clip"]), rng,
            group_by_protein=group_by_protein)
        # Scoring the very sample just trained on is the point of the check, not an
        # oversight.
        scores = train.evaluate(arrays, train.predict(model, arrays, caches, runtime.source,
                                                      device, batch_size),
                                keys, config)["all"]
        regression = scores["regression"]
        ranking = scores.get("ranking") or {}
        record = {"epoch": epoch, "steps": steps, "train_loss": loss,
                  "train_mae": regression["mae"], "train_r2": regression["r2"],
                  "train_pairwise_accuracy_macro": ranking.get("pairwise_accuracy_macro")}
        history.append(record)
        LOGGER.info("epoch %s: loss %.6f | train MAE %.4f | train R2 %s | pairwise "
                    "accuracy macro %s", epoch, loss, regression["mae"],
                    "None" if regression["r2"] is None else f"{regression['r2']:.4f}",
                    "None" if record["train_pairwise_accuracy_macro"] is None
                    else f"{record['train_pairwise_accuracy_macro']:.4f}")
        # Stop on EXACTLY the condition `verdict` uses, never on a second threshold.
        # Before 2026-11-19 this stopped at MAE <= 0.01 while the verdict looked at
        # R² >= 0.95: on a small sample with little label variance, MAE passes its
        # threshold while R² stays low, so one run could both log "sample fitted" and
        # record `does_not_fit`.
        if verdict(regression["r2"], threshold) == FIT:
            LOGGER.info("stopping: train R2 %.4f reached the threshold %.4f, the small "
                        "sample has been fitted", regression["r2"], threshold)
            break

    last = history[-1] if history else {}
    conclusion = verdict(last.get("train_r2"), threshold)
    manifest = {
        "stage": "stage3_capacity",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configuration": configuration, "head": head, "protein_mode": protein_mode,
        "device": device, "seed": seed, "regularized": regularized,
        "dropout": float(model_settings["dropout"]),
        "weight_decay": float(fit_settings["weight_decay"]),
        "learning_rate": float(fit_settings["learning_rate"]),
        "batch_size": batch_size, "group_by_protein": group_by_protein,
        "epochs_requested": epochs, "epochs_run": len(history),
        "dropout_keys": dropouts,
        "model": models.architecture_settings(model_settings),
        "parameters": models.count_parameters(model),
        "sample": sample,
        "train_subset": str(config["stage3"]["inputs"].get("train_subset") or ""),
        "fit_train_r2_threshold": threshold,
        "final": last,
        "verdict": conclusion,
        "verdict_note": verdict_note(regularized, sample, dropouts,
                                     float(fit_settings["weight_decay"]), len(history)),
        "overrides": dict(overrides or {}),
        "history": history,
        "seconds": round(time.perf_counter() - started, 1),
        "gpu_peak_reserved_mib": train.peak_gpu_memory_mib(device),
    }
    LOGGER.info("VERDICT: %s (final train R2 %s, threshold %.2f)", conclusion,
                last.get("train_r2"), threshold)
    path = runs_dir / f"capacity_{configuration}_{head}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("capacity check manifest: %s", path)
    return manifest
