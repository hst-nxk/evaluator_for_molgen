"""Stage 3: classical machine-learning baselines on EXACTLY the evaluator's features.

WHY THIS MODULE EXISTS. From the 2026-11-18 meeting (translated from the Vietnamese
minutes):

    "That is the most basic step, the minimum baseline you have to try first. Because if
     this complicated 3.7-million-parameter deep learning model of yours cannot even beat
     a linear regression, then it is not solving anything."

For the answer to mean anything the comparison has to be FAIR, so this module deliberately
reuses the real training run untouched and swaps only the model:

    features : fingerprint A, fingerprint B, substituent fingerprint A, substituent B,
               protein embedding, one-hot endpoint — exactly the five inputs the MLP sees,
               read from the very caches it uses (`train.Caches`)
    splitting: `data.read_partition`, so it also obeys `inputs.train_subset`,
               `inputs.partition_override` and `inputs.partition_override_pairs`
    metrics  : `train.evaluate`, so the same slices, the same query definition and the
               same direction-accuracy computation

What is left is a difference between MODELS, not between feature sets or splits. The
module also scores the FITTED TRAIN ROWS themselves, which is what the meeting asked for
("you have to look at the metrics on the training set as well, otherwise you cannot tell
whether the model is learning at all").

THREE THINGS TO KNOW WHEN READING THE RESULTS:

1. Ridge and Lasso have a regularisation parameter. Every value in
   ``stage3.classical.alphas`` is a separate fit, reported separately; whichever value is
   quoted has to be named, and it has to be chosen on validation exactly the way the MLP's
   checkpoint is chosen, or the two sides are not comparable. The manifest therefore makes
   that choice itself, in `selected_alpha`, through `train.selection_score` with
   `stage3.train.selection_metric` and `selection_slice` - the same code path --mode train
   uses. Every fit also carries `coefficients`, so a fit whose alpha zeroed the whole
   vector shows up as `constant_predictor` rather than as a baseline.
2. Features are NOT standardised. Fingerprints are 0/1 bits and therefore on one scale,
   protein embeddings are not — so for one alpha the effective amount of regularisation
   differs between the two blocks. The reason for that choice: the usual standardisation
   SUBTRACTS THE MEAN before dividing by the standard deviation, and subtracting the mean
   turns every zero into a non-zero, destroying sparsity and making the train matrix fully
   dense. Scaling ALONE (for example `StandardScaler(with_mean=False)`, or scaling the
   protein block separately) does keep sparsity — a legitimate option the project has not
   tried yet, not something that cannot be done. For now the features are raw and that is
   stated plainly.
3. Memory. The train matrix is sparse; the expensive part is not the fingerprints (a few
   dozen bits set per molecule) but the protein embedding (480 dense floats per row). At
   100,000 rows the matrix is on the order of half a gigabyte. ``chunk_rows`` bounds the
   memory of the validation prediction pass; ``max_train_rows`` bounds the memory of the
   fit.

scikit-learn, scipy and xgboost are imported INSIDE functions, following the convention
`train.py` uses for torch: every other mode keeps working on a machine where those three
are not installed.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import data, features, metrics, train

LOGGER = logging.getLogger(__name__)

LINEAR = "linear"
RIDGE = "ridge"
LASSO = "lasso"
XGBOOST = "xgboost"
# Deliberate order: cheap first, expensive last, so a run cut short still has the number
# for the simplest model — that is the reference point the meeting actually asked for.
MODELS = (LINEAR, RIDGE, LASSO, XGBOOST)
# The two models with a regularisation parameter; the others ignore `alphas`.
REGULARIZED = (RIDGE, LASSO)


def endpoint_one_hot(codes: np.ndarray, count: int) -> np.ndarray:
    """One-hot encoding of endpoint codes; (rows x count), float32.

    One-hot rather than a single integer column: for a linear model a 0..3 column would
    assert that endpoint 3 is "three times" endpoint 1, when the four endpoints are simply
    four kinds of measurement.
    """
    if codes.size and (codes.min() < 0 or codes.max() >= count):
        raise ValueError(f"endpoint code outside [0, {count}): "
                         f"[{codes.min()}, {codes.max()}]")
    matrix = np.zeros((codes.size, count), dtype=np.float32)
    matrix[np.arange(codes.size), codes] = 1.0
    return matrix


def molecule_block(caches: train.Caches, indices: np.ndarray) -> np.ndarray:
    """Feature block for one molecule side: unpacked fingerprint, or cached embedding.

    Follows whatever `stage3.model.ligand_features` the current training run uses, so the
    baseline gets the same molecular representation as the MLP rather than a different one.
    """
    if caches.molecule_vectors is None:
        return features.unpack_rows(caches.molecule_packed, indices, caches.ligand_bits)
    return np.asarray(caches.molecule_vectors[indices], dtype=np.float32)


def feature_blocks(arrays: data.PairArrays, rows: np.ndarray, caches: train.Caches,
                   endpoint_count: int, use_protein: bool) -> list[np.ndarray]:
    """Feature blocks of a batch of rows, in a fixed order.

    Order: molecule A, molecule B, substituent A, substituent B, [protein embedding],
    endpoint. That order is part of the contract: a linear model's coefficients can only
    be read if one knows which columns belong to which block (see `block_layout`).
    """
    blocks = [
        molecule_block(caches, arrays.molecule_a[rows]),
        molecule_block(caches, arrays.molecule_b[rows]),
        features.unpack_rows(caches.fragment_packed, arrays.fragment_a[rows],
                             caches.fragment_bits),
        features.unpack_rows(caches.fragment_packed, arrays.fragment_b[rows],
                             caches.fragment_bits),
    ]
    if use_protein:
        if caches.protein_vectors is None:
            raise ValueError("this configuration asks for the protein block but the "
                             "protein embedding cache is not loaded; run --mode prepare "
                             "or use --config no_protein")
        blocks.append(np.asarray(caches.protein_vectors[arrays.protein[rows]],
                                 dtype=np.float32))
    blocks.append(endpoint_one_hot(arrays.endpoint[rows], endpoint_count))
    return blocks


def block_layout(caches: train.Caches, endpoint_count: int,
                 use_protein: bool) -> list[dict[str, int | str]]:
    """Name and column range of every feature block, for the manifest."""
    widths = [("molecule_a", caches.ligand_bits), ("molecule_b", caches.ligand_bits),
              ("fragment_a", caches.fragment_bits), ("fragment_b", caches.fragment_bits)]
    if use_protein:
        widths.append(("protein", int(caches.protein_dim or 0)))
    widths.append(("endpoint", endpoint_count))
    layout: list[dict[str, int | str]] = []
    start = 0
    for name, width in widths:
        layout.append({"block": name, "start": start, "width": int(width)})
        start += int(width)
    return layout


def build_matrix(arrays: data.PairArrays, rows: np.ndarray, caches: train.Caches,
                 endpoint_count: int, use_protein: bool, chunk_rows: int) -> Any:
    """Sparse (CSR) feature matrix for ``rows``, assembled chunk by chunk.

    Chunked because the fingerprint blocks are unpacked into DENSE matrices before being
    compressed back to sparse; doing that once for all 100,000 rows would need several
    gigabytes for that intermediate step alone.
    """
    from scipy import sparse

    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    pieces = []
    for start in range(0, rows.size, chunk_rows):
        batch = rows[start:start + chunk_rows]
        blocks = feature_blocks(arrays, batch, caches, endpoint_count, use_protein)
        pieces.append(sparse.hstack([sparse.csr_matrix(block) for block in blocks],
                                    format="csr"))
    return sparse.vstack(pieces, format="csr") if len(pieces) > 1 else pieces[0]


def predict_rows(estimator: Any, arrays: data.PairArrays, caches: train.Caches,
                 endpoint_count: int, use_protein: bool, chunk_rows: int) -> np.ndarray:
    """Predict a whole partition chunk by chunk; float32, same length as ``arrays``.

    Chunked because validation has nearly 500,000 rows: building its entire feature matrix
    at once costs several gigabytes, while prediction only ever needs one chunk at a time.
    """
    outputs = np.empty(len(arrays), dtype=np.float32)
    for rows in data.iter_batches(len(arrays), chunk_rows):
        matrix = build_matrix(arrays, rows, caches, endpoint_count, use_protein, chunk_rows)
        outputs[rows] = np.asarray(estimator.predict(matrix), dtype=np.float32)
    return outputs


def build_estimator(name: str, alpha: float, settings: Mapping[str, Any],
                    seed: int) -> Any:
    """A configured, not yet fitted scikit-learn or xgboost model.

    Takes: model name; regularisation strength (ignored for linear and xgboost); the
    ``stage3.classical`` block; the seed.
    """
    if name == LINEAR:
        from sklearn.linear_model import LinearRegression

        # No regularisation: this is precisely the "linear regression" reference point the
        # meeting asked for.
        return LinearRegression(fit_intercept=True)
    if name == RIDGE:
        from sklearn.linear_model import Ridge

        # sparse_cg is the only Ridge solver that takes a sparse matrix without densifying
        # it first.
        return Ridge(alpha=alpha, fit_intercept=True, solver="sparse_cg",
                     max_iter=int(settings["max_iter"]), random_state=seed)
    if name == LASSO:
        from sklearn.linear_model import Lasso

        return Lasso(alpha=alpha, fit_intercept=True, max_iter=int(settings["max_iter"]),
                     random_state=seed)
    if name == XGBOOST:
        from xgboost import XGBRegressor

        return XGBRegressor(random_state=seed, **dict(settings["xgboost"]))
    raise ValueError(f"unknown baseline: {name!r}; only {MODELS}")


def run_label(name: str, alpha: float) -> str:
    """Key of one fit in the manifest; carries the alpha only where one exists."""
    return f"{name}_alpha{alpha:g}" if name in REGULARIZED else name


def model_alphas(name: str, settings: Mapping[str, Any]) -> list[float]:
    """The regularisation levels to fit for one model; `[0.0]` for a model without one.

    WHY PER MODEL AND NOT ONE SHARED LIST. The features are not standardised (point 2),
    and ridge and lasso penalise on different scales anyway, so one list cannot serve
    both: a level that barely regularises ridge drives every lasso coefficient to zero.
    Until 2026-11-20 the shared list held the single value 1.0, which for lasso is almost
    certainly that degenerate case - so the "baseline" the meeting asked for would have
    been a constant. ``alphas_by_model`` gives each model its own grid and ``alphas``
    stays the default for a model not named there.
    """
    if name not in REGULARIZED:
        return [0.0]
    by_model = dict(settings.get("alphas_by_model") or {})
    values = by_model.get(name, settings["alphas"])
    alphas = [float(value) for value in values]
    if not alphas:
        raise ValueError(f"no alphas configured for {name}; stage3.classical.alphas or "
                         f"alphas_by_model.{name} has to hold at least one value")
    return alphas


def convergence_summary(estimator: Any) -> dict[str, Any] | None:
    """Did the solver stop because it converged, or because it ran out of iterations?

    None for a model that does not report an iteration count - that is the right value
    there, because "no number" and "zero iterations" are different things.

    What `hit_the_cap` says is exactly this: the solver used its whole iteration budget.
    It does not say the solution is bad, and not hitting the cap does not prove the
    solution is good. Lasso's coordinate descent and ridge's sparse_cg both stop at
    ``max_iter`` whatever the state of the solution, and scikit-learn only says so through
    a warning that nothing in a batch log keeps. An unconverged fit is not wrong to
    report, but it has to be reported AS one.
    """
    iterations = getattr(estimator, "n_iter_", None)
    if iterations is None:
        return None
    # Lasso returns an int; some estimators return one entry per target.
    if not isinstance(iterations, (int, float)):
        values = [int(value) for value in np.asarray(iterations).ravel()]
        if not values:
            return None
        count = max(values)
    else:
        count = int(iterations)
    cap = int(getattr(estimator, "max_iter", 0) or 0)
    return {"iterations": count, "max_iter": cap,
            "hit_the_cap": bool(cap and count >= cap)}


def coefficient_summary(estimator: Any) -> dict[str, Any] | None:
    """How many coefficients the fit actually kept; None for a model without a flat
    coefficient vector (xgboost).

    WHY IT IS RECORDED. Lasso with too large an alpha drives every coefficient to zero,
    and such a fit predicts the intercept for every row: a constant. Its MAE is then the
    mean absolute deviation of the labels and its direction accuracy is whatever the
    tie-breaking rule gives - numbers that look like a baseline and are not one. With
    `alphas` holding a single value nobody had checked, that could have been quoted
    against the MLP without anyone noticing; `constant_predictor` makes it visible.

    IT ONLY WORKS IN ONE DIRECTION. All-zero coefficients prove the fit predicts a
    constant. Non-zero coefficients do NOT prove the opposite: a fit can still produce
    (near-)constant predictions on a particular set of rows, for instance when every
    surviving coefficient sits on a column that barely varies there. The flag is a
    sufficient condition for a degenerate fit, not a certificate that a fit is sound.
    """
    coefficients = getattr(estimator, "coef_", None)
    if coefficients is None:
        return None
    flat = np.asarray(coefficients, dtype=np.float64).ravel()
    nonzero = int(np.count_nonzero(flat))
    return {"total": int(flat.size), "nonzero": nonzero,
            "constant_predictor": nonzero == 0}


def selection_value(scores: Mapping[str, Any], selection: str) -> float | None:
    """The selection metric as a reader would quote it: bigger is better for pairwise
    accuracy, smaller for mae. None when that slice carries no number for it.

    `train.selection_score` normalises every metric to "smaller is better", which is what
    comparing needs and not what quoting needs; this undoes the sign.
    """
    score = train.selection_score(scores, selection)
    if score is None:
        return None
    return float(score) if selection == train.SELECT_MAE else -float(score)


def best_alpha(results: Mapping[str, Any], name: str, selection: str,
               selection_slice: str) -> dict[str, Any] | None:
    """Which alpha of model `name` wins on VALIDATION; None when none can be compared.

    Chosen exactly the way --mode train chooses a checkpoint: same metric, same slice,
    same "smaller is better" normalisation, through `train.selection_score` itself. Point
    1 of the module docstring demands that - an alpha picked on train, or picked by eye
    afterwards, is not comparable with the MLP's number. Writing the choice into the
    manifest also means the result table is not built by sifting the alphas for whichever
    one happens to flatter or embarrass the MLP.

    A tie goes to the smaller alpha: the same validation score for more regularisation.
    """
    candidates = []
    for label, block in results.items():
        if block.get("model") != name:
            continue
        scores = (block.get("validation_metrics") or {}).get(selection_slice)
        if not isinstance(scores, Mapping):
            continue
        score = train.selection_score(scores, selection)
        if score is None:
            continue
        candidates.append((float(score), float(block["alpha"]), label))
    if not candidates:
        return None
    score, alpha, label = min(candidates)
    return {"alpha": alpha, "label": label,
            "value": score if selection == train.SELECT_MAE else -score,
            "alphas_compared": len(candidates)}


def metrics_lines(label: str, scores: Mapping[str, Any]) -> list[str]:
    """Log lines for one metrics block; same formatting as --mode train."""
    return metrics.format_metrics(label, scores["regression"], scores.get("ranking"),
                                  scores.get("change_classes"))


def run_classical(config: Mapping[str, Any], root: Path, configuration: str,
                  baselines: Sequence[str], stamp: str,
                  overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit and score the classical baselines; returns the manifest (written to disk).

    Takes: config; project root; protein-branch configuration ('frozen' includes the
    protein block, 'no_protein' drops it); list of baselines; run stamp; ``overrides`` is
    recorded in the manifest only.
    """
    from . import models

    unknown = [name for name in baselines if name not in MODELS]
    if unknown:
        raise ValueError(f"unknown baseline: {unknown}; only {list(MODELS)}")
    settings = config["stage3"]["classical"]
    train_settings = config["stage3"]["train"]
    seed = int(train_settings["seed"])
    endpoint_count = len(config["endpoints"])
    chunk_rows = int(settings["chunk_rows"])
    protein_mode = models.protein_mode_for(configuration)
    if protein_mode == models.PROTEIN_FINETUNE:
        raise ValueError("the 'finetune' configuration is meaningless here: a classical "
                         "baseline does not train ESM2. Use --config frozen or no_protein")
    use_protein = protein_mode == models.PROTEIN_FROZEN
    _, runs_dir = train.layout(config, root)

    caches = train.Caches(config, root, need_protein=use_protein)
    runtime = train.build_protein_runtime(config, root, configuration, caches, "cpu")
    train_arrays, train_keys, _ = train.prepare_arrays(config, root, data.TRAIN, caches,
                                                       runtime.vocabulary, None)
    validation_arrays, validation_keys, _ = train.prepare_arrays(
        config, root, data.VALIDATION, caches, runtime.vocabulary, None)

    selection = str(train_settings.get("selection_metric", train.SELECT_MAE))
    if selection not in train.SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {train.SELECTION_METRICS}, "
                         f"got {selection!r}")
    # The row counts are passed in, exactly as run_train passes them: that turns "the
    # chosen slice has no validation rows" into an error BEFORE nine fits are run, instead
    # of a manifest whose selected_alpha is quietly missing. A table built from such a
    # manifest would simply have no alpha to quote and nothing would have said why.
    validation_slice_rows = {name: int(mask.sum())
                             for name, mask in validation_arrays.slices.items()}
    selection_slice = train.resolve_selection_slice(train_settings,
                                                    config["stage3"]["metrics"],
                                                    validation_slice_rows)

    # Cap on fitted rows: the feature matrix lives entirely in RAM, so a cap is required.
    # Sampled at random with the training run's own seed, not taken from the head of the
    # partition.
    cap = int(settings["max_train_rows"])
    rows = np.arange(len(train_arrays))
    if cap > 0 and len(train_arrays) > cap:
        rows = np.sort(np.random.default_rng(seed).permutation(len(train_arrays))[:cap])
        LOGGER.info("train has %s rows, cut to %s by max_train_rows (seed %s)",
                    f"{len(train_arrays):,}", f"{rows.size:,}", seed)
    fit_arrays = data.take_rows(train_arrays, rows)
    fit_keys = {name: column[rows] for name, column in train_keys.items()}

    layout_blocks = block_layout(caches, endpoint_count, use_protein)
    columns = sum(int(block["width"]) for block in layout_blocks)
    LOGGER.info("features: %s columns in %s blocks (%s) | fitting %s rows | validation "
                "%s rows", f"{columns:,}", len(layout_blocks),
                ", ".join(str(block["block"]) for block in layout_blocks),
                f"{len(fit_arrays):,}", f"{len(validation_arrays):,}")

    built = time.perf_counter()
    matrix = build_matrix(fit_arrays, np.arange(len(fit_arrays)), caches, endpoint_count,
                          use_protein, chunk_rows)
    target = fit_arrays.target.astype(np.float64)
    density = float(matrix.nnz) / float(matrix.shape[0] * matrix.shape[1])
    LOGGER.info("train matrix %s x %s, %s non-zero entries (%.3f %%), built in %.1f s",
                f"{matrix.shape[0]:,}", f"{matrix.shape[1]:,}", f"{matrix.nnz:,}",
                100.0 * density, time.perf_counter() - built)

    alphas = {name: model_alphas(name, settings) for name in baselines}


    base_manifest = {
        "stage": "stage3_classical",
        "configuration": configuration, "protein_mode": protein_mode,
        "baselines": list(baselines), "alphas": alphas, "seed": seed,
        "selection_metric": selection, "selection_slice": selection_slice,
        "ligand_features": caches.ligand_features,
        "feature_blocks": layout_blocks, "feature_columns": columns,
        "train_matrix_density": density,
        "rows": {"train_available": len(train_arrays), "train_fit": len(fit_arrays),
                 "validation": len(validation_arrays)},
        "max_train_rows": cap, "chunk_rows": chunk_rows,
        "xgboost": dict(settings["xgboost"]),
        "max_iter": int(settings["max_iter"]),
        "train_subset": str(config["stage3"]["inputs"].get("train_subset") or ""),
        "partition_override": str(
            config["stage3"]["inputs"].get("partition_override") or ""),
        "partition_override_pairs": str(
            config["stage3"]["inputs"].get("partition_override_pairs") or ""),
        "protein_permute_seed": int(
            train_settings.get("protein_permute_seed", train.PROTEIN_PERMUTE_OFF)),
        "metrics": {key: value for key, value in config["stage3"]["metrics"].items()
                    if not key.endswith("note")},
        "overrides": dict(overrides or {}),
        "note": ("Classical baselines on exactly the features, the split and the metrics "
                 "of a real training run, so these numbers compare directly with a "
                 "--mode train manifest. Features are NOT standardised (see the module "
                 "docstring). Train metrics are computed on the very rows that were "
                 "fitted. `complete` = false means the run did not get through the whole "
                 "baseline list (an error, or it was cut short); the models present in "
                 "`results` still carry their own complete numbers."),
    }

    results: dict[str, Any] = {}
    path = runs_dir / f"classical_{configuration}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(complete: bool) -> dict[str, Any]:
        """Write the manifest with whatever exists; `complete` states whether all ran.

        Called after EVERY fit, not only at the end: `--baseline all` runs four models, and
        if xgboost fails (missing library, out of memory) after the other three finished,
        their per-slice metrics would be lost entirely had the manifest only been written
        at the end. Rewriting the whole file each time is cheap: the file is a few hundred
        kilobytes while a single fit takes minutes.
        """
        chosen = {name: best_alpha(results, name, selection, selection_slice)
                  for name in REGULARIZED}
        # A regularised model that was fitted but whose alpha could not be chosen: the
        # slice is not empty (that is refused above), so the metric itself came back None -
        # on `pairwise_accuracy_macro` that means no query in the slice had enough
        # candidates to be ranked. Named in the manifest rather than left as an absence.
        unavailable = sorted(
            name for name, block in chosen.items()
            if block is None and any(entry.get("model") == name
                                     for entry in results.values()))
        manifest = dict(base_manifest, results=results, complete=complete,
                        models_finished=sorted(results),
                        selected_alpha={name: block for name, block in chosen.items()
                                        if block is not None},
                        selected_alpha_unavailable=unavailable,
                        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return manifest

    for name in baselines:
        for alpha in alphas[name]:
            label = run_label(name, alpha)
            estimator = build_estimator(name, alpha, settings, seed)
            started = time.perf_counter()
            estimator.fit(matrix, target)
            seconds = round(time.perf_counter() - started, 1)
            on_train = train.evaluate(
                fit_arrays, np.asarray(estimator.predict(matrix), dtype=np.float32),
                fit_keys, config)
            on_validation = train.evaluate(
                validation_arrays,
                predict_rows(estimator, validation_arrays, caches, endpoint_count,
                             use_protein, chunk_rows),
                validation_keys, config)
            results[label] = {"model": name, "alpha": alpha, "fit_seconds": seconds,
                              "coefficients": coefficient_summary(estimator),
                              "convergence": convergence_summary(estimator),
                              "train_metrics": on_train,
                              "validation_metrics": on_validation}
            write(complete=False)
            kept = results[label]["coefficients"]
            if kept is not None:
                LOGGER.info(
                    "%s: %s of %s coefficients are non-zero%s", label,
                    f"{kept['nonzero']:,}", f"{kept['total']:,}",
                    " - THIS FIT PREDICTS A CONSTANT" if kept["constant_predictor"]
                    else "")
            stopped = results[label]["convergence"]
            if stopped is not None and stopped["hit_the_cap"]:
                LOGGER.info("%s: the solver used all %s iterations and may not have "
                            "converged", label, f"{stopped['max_iter']:,}")
            LOGGER.info("%s: fit %.1f s | manifest updated: %s", label, seconds, path)
            # Log EVERY slice, not just 'all': the 'strict' slice is the one the project
            # has ranked models on since 2026-11-12, and if a later model in the loop
            # fails, the log is the only thing left at that moment.
            for partition, scores in (("train", on_train), ("validation", on_validation)):
                for slice_name, block in scores.items():
                    if slice_name == "by_endpoint" or not isinstance(block, dict):
                        continue
                    for line in metrics_lines(f"{label} / {partition} / {slice_name}",
                                              block):
                        LOGGER.info("%s", line)

    manifest = write(complete=True)
    for name in manifest["selected_alpha_unavailable"]:
        LOGGER.warning("%s: no alpha could be chosen - %s is None on the %r slice of "
                       "validation for every level fitted; the manifest records the fits "
                       "but names no winner", name, selection, selection_slice)
    for name, block in manifest["selected_alpha"].items():
        LOGGER.info("%s: alpha %g wins on validation (%s = %s on the %r slice, %s levels "
                    "compared)", name, block["alpha"], selection,
                    f"{block['value']:.4f}" if block["value"] is not None else "-",
                    selection_slice, block["alphas_compared"])
    LOGGER.info("classical baseline manifest: %s", path)
    return manifest
