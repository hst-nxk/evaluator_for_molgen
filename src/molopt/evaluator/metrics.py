"""Stage 3: regression metrics, and ranking metrics WITHIN one query.

Two groups of metrics, deliberately kept apart:

Regression - per row, no grouping needed: MAE, RMSE, R2, Spearman over everything.
Ranking    - ONLY within one query. A query is (reference molecule, protein, endpoint)
             with several candidates. Candidates of two different queries are never
             ranked against each other in one table.

Every function here works on numpy arrays and imports no torch, so it is testable on a
machine with no deep-learning library installed. No scipy either: the Spearman
coefficient is computed here (scipy is not among the project's requirements).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

# What a metric returns when it is undefined (the Spearman coefficient of a query whose
# labels are all equal, for example). None, not 0.0: a zero would be averaged in as
# though it were a measured result.
UNDEFINED = None


def average_ranks(values: np.ndarray) -> np.ndarray:
    """The rank of every element; equal values share their average rank.

    Takes: a one-dimensional array.
    Returns: a float64 array of the same length, ranks starting at 1.
    WHY: a Spearman coefficient over data with many ties (rounded Δp values tie often)
    comes out wrong if the ties are broken arbitrarily; the average rank is the standard
    definition.
    """
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            # The whole block [start, index) holds one value -> one average rank.
            ranks[order[start:index]] = 0.5 * (start + index - 1) + 1.0
            start = index
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    """The Pearson correlation coefficient; None when either side has no variance."""
    if left.size < 2:
        return UNDEFINED
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = math.sqrt(float(left_centered @ left_centered)
                            * float(right_centered @ right_centered))
    if denominator == 0.0:
        return UNDEFINED
    return float(left_centered @ right_centered) / denominator


def spearman(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    """Spearman = Pearson over ranks; None when either side is all one value."""
    if truth.size < 2:
        return UNDEFINED
    return pearson(average_ranks(truth), average_ranks(prediction))


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    """MAE, RMSE, R2, Spearman and the row count.

    Takes: the true labels and the predictions, of the same length.
    Returns: a dict of metrics; R2 and Spearman are None when the labels have no variance.
    R2 is taken around the mean of the TRUE labels (the usual definition), so it goes
    negative when the model does worse than predicting that mean.
    """
    if truth.shape != prediction.shape:
        raise ValueError(f"length mismatch: {truth.shape} and {prediction.shape}")
    if truth.size == 0:
        return {"n": 0, "mae": UNDEFINED, "rmse": UNDEFINED, "r2": UNDEFINED,
                "spearman": UNDEFINED}
    error = prediction - truth
    residual = float(error @ error)
    centered = truth - truth.mean()
    total = float(centered @ centered)
    return {
        "n": int(truth.size),
        "mae": float(np.abs(error).mean()),
        "rmse": math.sqrt(residual / truth.size),
        "r2": (1.0 - residual / total) if total > 0.0 else UNDEFINED,
        "spearman": spearman(truth, prediction),
    }


CHANGE_CLASSES = ("worse", "neutral", "better")


def change_class(values: np.ndarray, threshold: float) -> np.ndarray:
    """The three classes of Δp: 0 = worse (< -threshold), 1 = neutral (|Δp| <= threshold),
    2 = better (> threshold)."""
    return np.where(values > threshold, 2, np.where(values < -threshold, 0, 1)).astype(np.int8)


def change_class_metrics(truth: np.ndarray, prediction: np.ndarray,
                         threshold: float) -> dict[str, object]:
    """Score the model as a three-class problem: worse / neutral / better.

    Takes: the true labels, the predictions, and the threshold below which a difference
    counts as "neutral" (in log units). Returns: accuracy, per-class recall, precision and
    F1, macro-F1, the confusion matrix [true][predicted], and the row count of every true
    class.
    WHY: a Δp difference of 0.3 log or less is inside experimental error (Landrum &
    Riniker 2024, J. Chem. Inf. Model. 64, 1560; Nelen et al. 2025, J. Cheminform. 17, 8),
    so getting the direction wrong inside that band means nothing; three classes separate
    "worse" from "neutral" and "better". A class with no rows at all (neither true nor
    predicted) has F1 None and does not enter the macro-F1.
    """
    if truth.shape != prediction.shape:
        raise ValueError(f"length mismatch: {truth.shape} and {prediction.shape}")
    size = len(CHANGE_CLASSES)
    result: dict[str, object] = {"n": int(truth.size), "threshold": float(threshold)}
    if truth.size == 0:
        result.update({"accuracy": UNDEFINED, "macro_f1": UNDEFINED})
        return result
    actual = change_class(truth, threshold)
    guessed = change_class(prediction, threshold)
    confusion = np.zeros((size, size), dtype=np.int64)
    np.add.at(confusion, (actual, guessed), 1)
    scores = []
    for index, name in enumerate(CHANGE_CLASSES):
        hit = int(confusion[index, index])
        real, predicted = int(confusion[index].sum()), int(confusion[:, index].sum())
        recall = hit / real if real else UNDEFINED
        precision = hit / predicted if predicted else UNDEFINED
        f1 = (2.0 * hit / (real + predicted)) if (real + predicted) else UNDEFINED
        if f1 is not None:
            scores.append(f1)
        result[f"{name}_rows"] = real
        result[f"{name}_recall"] = recall
        result[f"{name}_precision"] = precision
        result[f"{name}_f1"] = f1
    result["accuracy"] = float(np.trace(confusion) / truth.size)
    result["macro_f1"] = float(np.mean(scores)) if scores else UNDEFINED
    result["confusion"] = confusion.tolist()
    return result


def expand_directions(reference: np.ndarray, candidate: np.ndarray, truth: np.ndarray,
                      prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                       np.ndarray]:
    """Every pair (A, B, Δp) becomes two directions: A->B keeps the sign, B->A flips it.

    Takes: the molecule code of A, of B, the true Δp, the predicted Δp.
    Returns: (reference molecule code, candidate code, true Δp, predicted Δp), doubled.
    WHY: the dataset stores every pair in the key order A < B (see Stage 2b), so grouping
    by the A side alone cuts a query up along molecule-key order rather than along
    chemistry. For a fixed reference molecule, ranking candidates by Δp is exactly ranking
    them by their own activity, so flipping the sign is a valid transformation and not
    invented data.
    """
    return (np.concatenate([reference, candidate]),
            np.concatenate([candidate, reference]),
            np.concatenate([truth, -truth]),
            np.concatenate([prediction, -prediction]))


def query_groups(columns: Sequence[np.ndarray]) -> np.ndarray:
    """The group (query) code of every row, from key columns already encoded as integers.

    Takes: a list of integer arrays of the same length (reference molecule, protein,
    endpoint, for example).
    Returns: an int64 array; the same code means the same query.
    """
    if not columns:
        raise ValueError("at least one key column is needed")
    stacked = np.stack([np.asarray(column, dtype=np.int64) for column in columns], axis=1)
    _, inverse = np.unique(stacked, axis=0, return_inverse=True)
    return inverse.astype(np.int64).ravel()


def group_bounds(group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by group code; returns (row order, the bounds of every group).

    The bounds read [0, ..., n], so group i is order[bounds[i]:bounds[i + 1]].
    """
    order = np.argsort(group_ids, kind="stable")
    sorted_ids = group_ids[order]
    changes = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
    bounds = np.concatenate([[0], changes, [group_ids.size]])
    return order, bounds


def pairwise_direction(truth: np.ndarray, prediction: np.ndarray,
                       tolerance: float) -> tuple[int, int]:
    """How many candidate pairs got the direction right, and how many were counted.

    Takes: the labels and predictions of the candidates WITHIN ONE query; the threshold
    below which two labels count as equal.
    Returns: (pairs with the right direction, pairs counted).
    A pair whose two labels differ by no more than `tolerance` is dropped from the
    denominator: there is no right direction to compare against. A pair the model predicts
    as a tie counts as wrong, because the model did not choose.
    """
    difference_truth = truth[:, None] - truth[None, :]
    difference_prediction = prediction[:, None] - prediction[None, :]
    upper = np.triu(np.ones_like(difference_truth, dtype=bool), k=1)
    counted = upper & (np.abs(difference_truth) > tolerance)
    total = int(counted.sum())
    if total == 0:
        return 0, 0
    agree = np.sign(difference_truth) == np.sign(difference_prediction)
    return int((counted & agree).sum()), total


def top_k_hit(truth: np.ndarray, prediction: np.ndarray, k: int) -> bool:
    """Is the best candidate by label among the k best candidates by prediction?

    Ties at the top: it counts as a hit if ANY candidate holding the largest label is
    inside the predicted top-k.
    """
    best = np.flatnonzero(truth >= truth.max())
    chosen = np.argsort(-prediction, kind="stable")[:k]
    return bool(np.intersect1d(best, chosen).size > 0)


def ranking_metrics(group_ids: np.ndarray, truth: np.ndarray, prediction: np.ndarray,
                    top_k: Sequence[int], min_candidates: int,
                    tolerance: float = 0.0) -> dict[str, object]:
    """Ranking metrics within every query, summarised both macro and micro.

    Takes: the query code of every row, the labels, the predictions, the list of k values,
    the minimum number of candidates for a query to be ranked at all, and the threshold
    below which two labels count as equal.
    Returns: a dict holding
        queries_total / queries_ranked / queries_skipped - the denominators, stated;
        spearman_macro, spearman_median - the mean and median Spearman PER QUERY;
        spearman_undefined - how many queries had all labels equal, so it was undefined;
        pairwise_accuracy_macro - the per-query direction accuracy, averaged over queries;
        pairwise_accuracy_micro - every candidate pair of every query pooled, then divided;
        pairs_counted - the denominator of the micro figure;
        top_{k} and top_{k}_queries - the hit rate and its own denominator for every k.

    WHY BOTH MACRO AND MICRO ARE REPORTED: a long query holds many pairs and therefore
    dominates the micro figure, while the macro figure gives every query one equal vote.
    The two numbers answer two different questions and the meeting did not settle which
    one counts, so neither is dropped.
    WHY THERE IS NO spearman_micro: a Spearman coefficient only means something within one
    query; pooling the ranks of different queries into one table is the very thing to
    avoid. For top-k, macro and micro coincide (the metric is 0/1 per query), so there is
    one figure and its denominator.
    """
    if not (truth.shape == prediction.shape == group_ids.shape):
        raise ValueError("group_ids, truth and prediction must be the same length")
    if min_candidates < 2:
        raise ValueError(f"min_candidates has to be 2 or more, got {min_candidates}")
    order, bounds = group_bounds(group_ids)
    spearman_values: list[float] = []
    accuracy_values: list[float] = []
    agree_total = 0
    pairs_total = 0
    undefined = 0
    skipped = 0
    hits = {int(k): 0 for k in top_k}
    denominators = {int(k): 0 for k in top_k}
    for index in range(bounds.size - 1):
        rows = order[bounds[index]:bounds[index + 1]]
        if rows.size < min_candidates:
            skipped += 1
            continue
        group_truth = truth[rows]
        group_prediction = prediction[rows]
        value = spearman(group_truth, group_prediction)
        if value is None:
            undefined += 1
        else:
            spearman_values.append(value)
        agree, counted = pairwise_direction(group_truth, group_prediction, tolerance)
        if counted:
            agree_total += agree
            pairs_total += counted
            accuracy_values.append(agree / counted)
        for k in hits:
            # top-k only means something for a query with at least k candidates, so
            # every k carries its own denominator.
            if rows.size >= k:
                denominators[k] += 1
                if top_k_hit(group_truth, group_prediction, k):
                    hits[k] += 1
    ranked = int(bounds.size - 1 - skipped)
    result: dict[str, object] = {
        "queries_total": int(bounds.size - 1),
        "queries_ranked": ranked,
        "queries_skipped_too_few_candidates": skipped,
        "spearman_macro": float(np.mean(spearman_values)) if spearman_values else UNDEFINED,
        "spearman_median": float(np.median(spearman_values)) if spearman_values else UNDEFINED,
        "spearman_queries": len(spearman_values),
        "spearman_undefined": undefined,
        "pairwise_accuracy_macro": (float(np.mean(accuracy_values)) if accuracy_values
                                    else UNDEFINED),
        "pairwise_accuracy_micro": (agree_total / pairs_total) if pairs_total else UNDEFINED,
        "pairs_counted": pairs_total,
    }
    for k in sorted(hits):
        result[f"top_{k}"] = (hits[k] / denominators[k]) if denominators[k] else UNDEFINED
        result[f"top_{k}_queries"] = denominators[k]
    return result


def metrics_by_endpoint(endpoints: np.ndarray, truth: np.ndarray, prediction: np.ndarray,
                        names: Sequence[str]) -> dict[str, dict[str, float | None]]:
    """Regression metrics, one block per endpoint.

    Takes: the endpoint code of every row, the labels, the predictions, and the endpoint
    names in code order.
    Returns: {endpoint name: regression metrics}. An endpoint with no rows still appears,
    with n = 0, so the report table does not lose a column between runs.
    """
    result = {}
    for code, name in enumerate(names):
        mask = endpoints == code
        result[name] = regression_metrics(truth[mask], prediction[mask])
    return result


def asymmetry_metrics(forward: np.ndarray, backward: np.ndarray, identity: np.ndarray,
                      label_spread: float) -> dict[str, float | None]:
    """How far the predictions depart from two properties the `direct` head does NOT
    guarantee.

    Takes:
        forward      : f(A, B) over a sample of validation rows
        backward     : f(B, A) over exactly those rows, with the two sides swapped
        identity     : f(A, A) over exactly those rows, with the B side replaced by A
        label_spread : the standard deviation of the true labels over that sample, which
                       serves as the yardstick
    Returns: the measurements, plus two normalised ratios.

    The symmetric part of a prediction is r = (f(A,B) + f(B,A)) / 2. A perfectly
    antisymmetric model has r = 0 for every pair; `twin` gives exactly 0 by construction,
    `direct` does not. An absolute number cannot be read without a yardstick, so this also
    returns the ratio to the size of the predictions and to the spread of the labels: a
    departure of 0.1 log is small when the labels span 1.5 log and large when they span
    0.2 log.
    """
    if not (forward.size == backward.size == identity.size):
        raise ValueError("the three arrays must hold the same number of rows")
    if forward.size == 0:
        return {"n": 0, "antisymmetry_mean_abs": UNDEFINED, "antisymmetry_p90": UNDEFINED,
                "antisymmetry_max": UNDEFINED, "antisymmetry_over_prediction": UNDEFINED,
                "antisymmetry_over_label_spread": UNDEFINED, "identity_mean_abs": UNDEFINED,
                "identity_p90": UNDEFINED, "identity_max": UNDEFINED,
                "identity_over_label_spread": UNDEFINED, "prediction_mean_abs": UNDEFINED,
                "label_spread": UNDEFINED}
    residual = np.abs((forward + backward) / 2.0)
    scale = float(np.mean(np.abs(forward)))
    identity_abs = np.abs(identity)
    spread = float(label_spread)
    return {
        "n": int(forward.size),
        "antisymmetry_mean_abs": float(np.mean(residual)),
        "antisymmetry_p90": float(np.percentile(residual, 90)),
        "antisymmetry_max": float(np.max(residual)),
        "antisymmetry_over_prediction": float(np.mean(residual) / scale) if scale else UNDEFINED,
        "antisymmetry_over_label_spread": float(np.mean(residual) / spread) if spread
        else UNDEFINED,
        "identity_mean_abs": float(np.mean(identity_abs)),
        "identity_p90": float(np.percentile(identity_abs, 90)),
        "identity_max": float(np.max(identity_abs)),
        "identity_over_label_spread": float(np.mean(identity_abs) / spread) if spread
        else UNDEFINED,
        "prediction_mean_abs": scale,
        "label_spread": spread,
    }


def format_asymmetry(label: str, scores: Mapping[str, float | None]) -> list[str]:
    """Two log lines for the asymmetry measurements; None prints as '-'.

    The second line prints the LARGEST value in scientific notation and compares it with
    the 1e-8 target set at the 2026-11-09 meeting (line 141 of the minutes). Scientific
    notation is required: on an architecture that is antisymmetric by construction this
    number is around 1e-8 or below, which `%.4f` would print as 0.0000 - and that cannot
    be told apart from "zero because the model predicts a constant".
    """

    def show(value: object) -> str:
        if value is None:
            return "-"
        return f"{value:.4f}" if isinstance(value, float) else f"{value:,}"

    def exact(value: object) -> str:
        return "-" if value is None else f"{float(value):.3e}"

    target = scores.get("antisymmetry_target")
    meets = scores.get("meets_target")
    verdict = "-" if meets is None else ("MET" if meets else "NOT MET")
    lines = [f"{label}: n = {show(scores.get('n'))}"
             f" | mean |f(A,B)+f(B,A)|/2 {show(scores.get('antisymmetry_mean_abs'))}"
             f" (= {show(scores.get('antisymmetry_over_label_spread'))} x label sd)"
             f" | mean |f(A,A)| {show(scores.get('identity_mean_abs'))}"
             f" (= {show(scores.get('identity_over_label_spread'))} x label sd)"]
    if target is not None:
        lines.append(
            f"{label}: largest |f(A,B)+f(B,A)|/2 = "
            f"{exact(scores.get('antisymmetry_max'))} against the target {exact(target)}"
            f" -> {verdict} | antisymmetric by construction: "
            f"{'yes' if scores.get('antisymmetric_by_construction') else 'no'}")
    return lines


def format_metrics(label: str, regression: Mapping[str, float | None],
                   ranking: Mapping[str, object] | None = None,
                   classes: Mapping[str, object] | None = None) -> list[str]:
    """Short text lines for the log; None prints as '-', never as 0."""

    def show(value: object, digits: int = 4) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return f"{value:,}"

    lines = [f"{label}: n = {show(regression.get('n'))}"
             f" | MAE {show(regression.get('mae'))}"
             f" | RMSE {show(regression.get('rmse'))}"
             f" | R2 {show(regression.get('r2'))}"
             f" | Spearman {show(regression.get('spearman'))}"]
    if ranking is not None:
        lines.append(f"    ranking: queries ranked {show(ranking.get('queries_ranked'))}"
                     f"/{show(ranking.get('queries_total'))}"
                     f" | Spearman macro {show(ranking.get('spearman_macro'))}"
                     f" | direction accuracy macro"
                     f" {show(ranking.get('pairwise_accuracy_macro'))}"
                     f" micro {show(ranking.get('pairwise_accuracy_micro'))}")
        tops = [f"top-{k} {show(ranking.get(f'top_{k}'))}"
                f" (over {show(ranking.get(f'top_{k}_queries'))} queries)"
                for k in sorted(int(name.split('_')[1]) for name in ranking
                                if name.startswith('top_') and name.count('_') == 1)]
        if tops:
            lines.append("    " + " | ".join(tops))
    if classes is not None:
        lines.append(f"    three classes (+/-{show(classes.get('threshold'), 2)}):"
                     f" accuracy {show(classes.get('accuracy'))}"
                     f" | macro-F1 {show(classes.get('macro_f1'))} | "
                     + " | ".join(f"{name} F1 {show(classes.get(f'{name}_f1'))}"
                                  for name in CHANGE_CLASSES))
    return lines
