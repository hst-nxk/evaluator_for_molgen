"""Classical baselines: the feature-matrix contract (molopt.evaluator.classical).

None of these tests fit a real model: what has to stay correct is the feature matrix —
right blocks, right order, right values — plus the wiring of a full run. If the block
order ever shifts, every linear coefficient becomes unreadable and nothing reports an
error.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from molopt.evaluator import classical, data, train

LIGAND_BITS = 16
FRAGMENT_BITS = 8
PROTEIN_DIM = 4
ENDPOINTS = 3
MOLECULES = 6
FRAGMENTS = 4
PROTEINS = 2
ROWS = 5


class FakeCaches:
    """Same interface as train.Caches, but entirely in memory."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.molecule_bits_matrix = rng.integers(0, 2, (MOLECULES, LIGAND_BITS)).astype(
            np.uint8)
        self.fragment_bits_matrix = rng.integers(0, 2, (FRAGMENTS, FRAGMENT_BITS)).astype(
            np.uint8)
        self.molecule_packed = np.packbits(self.molecule_bits_matrix, axis=1)
        self.fragment_packed = np.packbits(self.fragment_bits_matrix, axis=1)
        self.ligand_bits = LIGAND_BITS
        self.fragment_bits = FRAGMENT_BITS
        self.molecule_keys = pa.array([f"m{index}" for index in range(MOLECULES)])
        self.fragment_keys = pa.array([f"f{index}" for index in range(FRAGMENTS)])
        self.molecule_vectors = None
        self.ligand_features = "fingerprint"
        self.protein_keys = pa.array([f"h{index}" for index in range(PROTEINS)])
        self.protein_vectors = rng.normal(size=(PROTEINS, PROTEIN_DIM)).astype(np.float32)
        self.protein_dim = PROTEIN_DIM


@pytest.fixture()
def fixture() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    arrays = data.PairArrays(
        molecule_a=np.array([0, 1, 2, 3, 4]),
        molecule_b=np.array([5, 4, 3, 2, 1]),
        fragment_a=np.array([0, 1, 2, 3, 0]),
        fragment_b=np.array([3, 2, 1, 0, 2]),
        protein=np.array([0, 1, 1, 0, 1]),
        endpoint=np.array([0, 1, 2, 0, 1]),
        target=np.arange(ROWS, dtype=np.float32),
        slices={"all": np.ones(ROWS, dtype=bool)})
    return {"arrays": arrays, "caches": FakeCaches(rng)}


def test_one_hot_marks_exactly_one_column_per_row() -> None:
    matrix = classical.endpoint_one_hot(np.array([0, 2, 1, 2]), 3)
    assert matrix.shape == (4, 3)
    assert matrix.sum(axis=1).tolist() == [1.0, 1.0, 1.0, 1.0]
    assert matrix[1].tolist() == [0.0, 0.0, 1.0]


def test_one_hot_rejects_a_code_outside_the_vocabulary() -> None:
    # A code outside the range means the config's endpoint vocabulary differs from the one
    # the data was encoded with; silently ignoring it would leave an all-zero one-hot row
    # that nobody notices.
    with pytest.raises(ValueError, match="outside"):
        classical.endpoint_one_hot(np.array([0, 3]), 3)


def test_the_blocks_come_in_the_documented_order_with_the_documented_widths(
        fixture: dict[str, Any]) -> None:
    rows = np.arange(ROWS)
    blocks = classical.feature_blocks(fixture["arrays"], rows, fixture["caches"],
                                      ENDPOINTS, use_protein=True)
    assert [block.shape[1] for block in blocks] == [
        LIGAND_BITS, LIGAND_BITS, FRAGMENT_BITS, FRAGMENT_BITS, PROTEIN_DIM, ENDPOINTS]
    layout = classical.block_layout(fixture["caches"], ENDPOINTS, use_protein=True)
    assert [item["block"] for item in layout] == [
        "molecule_a", "molecule_b", "fragment_a", "fragment_b", "protein", "endpoint"]
    # Each block's start must equal the previous start plus its width, otherwise the column
    # ranges recorded in the manifest do not match the real matrix.
    assert [item["start"] for item in layout] == [
        0, LIGAND_BITS, 2 * LIGAND_BITS, 2 * LIGAND_BITS + FRAGMENT_BITS,
        2 * LIGAND_BITS + 2 * FRAGMENT_BITS,
        2 * LIGAND_BITS + 2 * FRAGMENT_BITS + PROTEIN_DIM]


def test_the_blocks_carry_the_values_of_the_right_rows(fixture: dict[str, Any]) -> None:
    """Row i's features must be the fingerprint of row i's OWN molecules and protein."""
    arrays, caches = fixture["arrays"], fixture["caches"]
    rows = np.array([1, 3])
    blocks = classical.feature_blocks(arrays, rows, caches, ENDPOINTS, use_protein=True)
    for position, row in enumerate(rows):
        assert blocks[0][position].tolist() == (
            caches.molecule_bits_matrix[arrays.molecule_a[row]].tolist())
        assert blocks[1][position].tolist() == (
            caches.molecule_bits_matrix[arrays.molecule_b[row]].tolist())
        assert blocks[2][position].tolist() == (
            caches.fragment_bits_matrix[arrays.fragment_a[row]].tolist())
        assert blocks[4][position].tolist() == pytest.approx(
            caches.protein_vectors[arrays.protein[row]].tolist())


def test_dropping_the_protein_block_only_drops_that_block(fixture: dict[str, Any]) -> None:
    rows = np.arange(ROWS)
    with_protein = classical.feature_blocks(fixture["arrays"], rows, fixture["caches"],
                                            ENDPOINTS, use_protein=True)
    without = classical.feature_blocks(fixture["arrays"], rows, fixture["caches"],
                                       ENDPOINTS, use_protein=False)
    assert len(without) == len(with_protein) - 1
    assert [block.shape[1] for block in without] == [
        LIGAND_BITS, LIGAND_BITS, FRAGMENT_BITS, FRAGMENT_BITS, ENDPOINTS]


def test_chunking_does_not_change_the_matrix(fixture: dict[str, Any]) -> None:
    """Chunking is a memory decision; it must not change a single matrix entry."""
    pytest.importorskip("scipy", reason="the sparse matrix needs scipy")
    rows = np.arange(ROWS)
    whole = classical.build_matrix(fixture["arrays"], rows, fixture["caches"], ENDPOINTS,
                                   True, chunk_rows=ROWS)
    chunked = classical.build_matrix(fixture["arrays"], rows, fixture["caches"], ENDPOINTS,
                                     True, chunk_rows=2)
    assert whole.shape == chunked.shape
    assert (whole != chunked).nnz == 0


def test_the_run_label_records_the_alpha_only_where_it_exists() -> None:
    # The label is a fit's key in the manifest: linear has no alpha and must not carry a
    # meaningless number, while ridge and lasso must.
    assert classical.run_label(classical.LINEAR, 0.0) == "linear"
    assert classical.run_label(classical.RIDGE, 1.0) == "ridge_alpha1"
    assert classical.run_label(classical.LASSO, 0.001) == "lasso_alpha0.001"


def test_the_module_imports_the_heavy_libraries_only_inside_functions() -> None:
    """classical.py must import on a machine without scikit-learn, scipy or xgboost.

    The same promise train.py keeps about torch: every other CLI mode has to keep working,
    because run_evaluator.py imports this module at start-up.
    """
    tree = ast.parse(Path(classical.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"sklearn", "scipy", "xgboost"}


class StubEstimator:
    """Fake model: records the shape of the matrix it was fitted on, predicts a constant.

    It exists to test run_classical's WIRING (reading data, building the matrix, scoring
    train and validation, writing the manifest) without depending on scikit-learn — the
    actual learning is scikit-learn's job, not something this project has to test.
    """

    def __init__(self) -> None:
        self.fitted_shape: tuple[int, int] | None = None

    def fit(self, matrix: Any, target: np.ndarray) -> "StubEstimator":
        self.fitted_shape = tuple(matrix.shape)
        assert matrix.shape[0] == target.size
        return self

    def predict(self, matrix: Any) -> np.ndarray:
        return np.full(matrix.shape[0], 0.25, dtype=np.float32)


def test_a_full_run_scores_both_train_and_validation_and_writes_one_manifest(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict[str, Any],
        fixture: dict[str, Any]) -> None:
    """One complete run_classical pass over fake data with a fake model."""
    pytest.importorskip("scipy", reason="the sparse matrix needs scipy")
    run_config = json.loads(json.dumps(config))
    run_config["endpoints"] = list(run_config["endpoints"])[:ENDPOINTS]
    run_config["stage3"]["classical"].update({"max_train_rows": 3, "chunk_rows": 2,
                                              "alphas": [0.5, 1.0]})
    # Deterministic selection for the assertion below: the stub predicts one constant, so
    # every alpha scores the same and the tie has to go to the smaller one.
    run_config["stage3"]["train"].update({"selection_metric": "mae",
                                          "selection_slice": "all"})
    arrays, caches = fixture["arrays"], fixture["caches"]
    keys = {"molecule_a": arrays.molecule_a, "molecule_b": arrays.molecule_b,
            "protein": arrays.protein, "endpoint": arrays.endpoint}
    stubs: list[StubEstimator] = []

    def build_stub(*args: Any, **kwargs: Any) -> StubEstimator:
        stubs.append(StubEstimator())
        return stubs[-1]

    monkeypatch.setattr(train, "Caches", lambda *args, **kwargs: caches)
    monkeypatch.setattr(train, "prepare_arrays",
                        lambda *args, **kwargs: (arrays, dict(keys), None))
    monkeypatch.setattr(train, "layout", lambda config, root: (tmp_path, tmp_path))
    monkeypatch.setattr(classical, "build_estimator", build_stub)

    manifest = classical.run_classical(run_config, tmp_path, "frozen",
                                       [classical.LINEAR, classical.RIDGE], stamp="t")

    # linear once, ridge once per alpha.
    assert sorted(manifest["results"]) == ["linear", "ridge_alpha0.5", "ridge_alpha1"]
    assert len(stubs) == 3
    # max_train_rows = 3 has to really cut, and the fitted matrix must have exactly as many
    # columns as the layout claims.
    assert manifest["rows"] == {"train_available": ROWS, "train_fit": 3,
                                "validation": ROWS}
    assert stubs[0].fitted_shape == (3, manifest["feature_columns"])
    for block in manifest["results"].values():
        # Scored on both train (the fitted rows) and validation, over the same slices.
        assert block["train_metrics"]["all"]["regression"]["n"] == 3
        assert block["validation_metrics"]["all"]["regression"]["n"] == ROWS
    assert sorted(path.name for path in tmp_path.glob("classical_*")) == [
        "classical_frozen_t.json"]
    # The alpha that gets quoted is recorded by the run itself, not chosen afterwards.
    assert manifest["selection_metric"] == "mae"
    assert manifest["selection_slice"] == "all"
    assert manifest["selected_alpha"]["ridge"] == {
        "alpha": 0.5, "label": "ridge_alpha0.5",
        "value": manifest["results"]["ridge_alpha0.5"]["validation_metrics"]["all"][
            "regression"]["mae"],
        "alphas_compared": 2}
    # Lasso was not among the baselines, so there is nothing to choose for it.
    assert "lasso" not in manifest["selected_alpha"]
    # The stub has neither coef_ nor n_iter_; the keys still exist, holding None, so a
    # reader never has to guess whether a fit was checked or the check is missing.
    for block in manifest["results"].values():
        assert block["coefficients"] is None
        assert block["convergence"] is None


def test_a_finetune_configuration_is_refused(config: dict[str, Any],
                                             tmp_path: Path) -> None:
    # A classical baseline does not train ESM2, so 'finetune' here is a meaningless command
    # rather than a variant: it has to fail instead of producing a number nobody can place.
    with pytest.raises(ValueError, match="finetune"):
        classical.run_classical(config, tmp_path, "finetune", [classical.LINEAR],
                                stamp="t")


# --------------------------------------------------------------- the alpha grid per model
# One shared list cannot serve ridge and lasso: the features are not standardised and the
# two penalties are on different scales, so the level that barely regularises ridge zeroes
# every lasso coefficient. Until 2026-11-20 the list held the single value 1.0.


def test_each_regularised_model_can_have_its_own_alpha_grid() -> None:
    settings = {"alphas": [1.0],
                "alphas_by_model": {"lasso": [0.001, 0.01]}}
    assert classical.model_alphas(classical.LASSO, settings) == [0.001, 0.01]
    # Ridge is not named, so it keeps the default grid.
    assert classical.model_alphas(classical.RIDGE, settings) == [1.0]


def test_a_model_without_a_regularisation_parameter_is_fitted_once() -> None:
    settings = {"alphas": [0.1, 1.0, 10.0]}
    assert classical.model_alphas(classical.LINEAR, settings) == [0.0]
    assert classical.model_alphas(classical.XGBOOST, settings) == [0.0]


def test_an_empty_alpha_grid_is_refused() -> None:
    # Silently fitting nothing would leave the model missing from the manifest with no
    # error anywhere.
    with pytest.raises(ValueError, match="at least one value"):
        classical.model_alphas(classical.RIDGE, {"alphas": [],
                                                 "alphas_by_model": {}})


# ------------------------------------------------- a fit that zeroed everything is visible


class Fitted:
    """Carries only the attributes the two summaries read."""

    def __init__(self, **attributes: Any) -> None:
        for name, value in attributes.items():
            setattr(self, name, value)


def test_a_fit_whose_coefficients_are_all_zero_is_marked_as_a_constant() -> None:
    summary = classical.coefficient_summary(Fitted(coef_=np.zeros(5)))
    assert summary == {"total": 5, "nonzero": 0, "constant_predictor": True}


def test_the_coefficient_summary_counts_what_the_fit_kept() -> None:
    summary = classical.coefficient_summary(
        Fitted(coef_=np.array([0.0, -0.5, 0.0, 2.0])))
    assert summary == {"total": 4, "nonzero": 2, "constant_predictor": False}


def test_a_model_without_a_coefficient_vector_reports_nothing() -> None:
    # xgboost: there is no flat coefficient vector to count, and inventing a zero would
    # read as "this fit predicts a constant".
    assert classical.coefficient_summary(Fitted()) is None


def test_a_solver_that_used_every_iteration_is_marked() -> None:
    assert classical.convergence_summary(Fitted(n_iter_=2000, max_iter=2000)) == {
        "iterations": 2000, "max_iter": 2000, "hit_the_cap": True}
    assert classical.convergence_summary(Fitted(n_iter_=31, max_iter=2000)) == {
        "iterations": 31, "max_iter": 2000, "hit_the_cap": False}


def test_an_iteration_count_per_target_is_summarised_by_the_worst_one() -> None:
    assert classical.convergence_summary(
        Fitted(n_iter_=np.array([12, 2000]), max_iter=2000))["hit_the_cap"] is True


def test_a_model_that_does_not_count_iterations_reports_nothing() -> None:
    assert classical.convergence_summary(Fitted(max_iter=2000)) is None


# ------------------------------------------------------ the alpha is chosen on validation


def scored(model: str, alpha: float, mae: float, pairwise: float) -> dict[str, Any]:
    """One entry of a manifest's `results`, carrying only what best_alpha reads."""
    return {"model": model, "alpha": alpha,
            "validation_metrics": {"all": {"regression": {"mae": mae},
                                           "ranking": {"pairwise_accuracy_macro": pairwise}},
                                   "strict": {"regression": {"mae": mae},
                                              "ranking": {"pairwise_accuracy_macro": None}}}}


def test_the_chosen_alpha_follows_the_direction_of_the_metric() -> None:
    results = {"ridge_alpha0.1": scored(classical.RIDGE, 0.1, mae=0.60, pairwise=0.70),
               "ridge_alpha10": scored(classical.RIDGE, 10.0, mae=0.55, pairwise=0.64)}
    # Smaller is better for mae, bigger for pairwise accuracy - so the two metrics pick
    # different alphas out of the same two fits, which is exactly why the manifest has to
    # say which metric was used.
    by_mae = classical.best_alpha(results, classical.RIDGE, "mae", "all")
    assert (by_mae["alpha"], by_mae["value"]) == (10.0, 0.55)
    by_rank = classical.best_alpha(results, classical.RIDGE,
                                   "pairwise_accuracy_macro", "all")
    assert (by_rank["alpha"], by_rank["value"]) == (0.1, 0.70)
    assert by_rank["label"] == "ridge_alpha0.1"
    assert by_rank["alphas_compared"] == 2


def test_only_the_asked_model_is_considered() -> None:
    results = {"ridge_alpha1": scored(classical.RIDGE, 1.0, mae=0.60, pairwise=0.70),
               "lasso_alpha0.001": scored(classical.LASSO, 0.001, mae=0.50, pairwise=0.80)}
    chosen = classical.best_alpha(results, classical.RIDGE, "mae", "all")
    assert chosen["label"] == "ridge_alpha1"


def test_an_equal_score_goes_to_the_stronger_regularisation() -> None:
    results = {"ridge_alpha1": scored(classical.RIDGE, 1.0, mae=0.60, pairwise=0.70),
               "ridge_alpha10": scored(classical.RIDGE, 10.0, mae=0.60, pairwise=0.70)}
    assert classical.best_alpha(results, classical.RIDGE, "mae", "all")["alpha"] == 1.0


def test_no_alpha_is_chosen_when_the_slice_carries_no_number() -> None:
    # The 'strict' slice here has no pairwise accuracy (an empty slice gives None), and
    # inventing a winner out of that would put an unmeasured number in the table.
    results = {"ridge_alpha1": scored(classical.RIDGE, 1.0, mae=0.60, pairwise=0.70)}
    assert classical.best_alpha(results, classical.RIDGE,
                                "pairwise_accuracy_macro", "strict") is None
    assert classical.best_alpha(results, classical.LASSO, "mae", "all") is None


# ------------------------------------- the manifest's own choice, on a machine without scipy
# The full run above needs scipy for the sparse matrix, so on a machine without it the test
# SKIPS - and everything it asserts about the manifest goes unchecked there. The matrix is
# scipy's job, not this project's, so this test replaces it with a stand-in that reports a
# shape and a non-zero count, and checks what IS this project's job: that the run records
# which alpha won on validation, and that a fit which zeroed its whole coefficient vector
# says so in the manifest.


class FakeMatrix:
    """Only what run_classical and a stub estimator read off a matrix."""

    def __init__(self, rows: int, columns: int) -> None:
        self.shape = (rows, columns)
        self.nnz = rows * columns


class Recorded:
    """Predicts the constant it was given; carries coef_ and n_iter_ like a real fit."""

    def __init__(self, constant: float, nonzero: int) -> None:
        self.constant = constant
        # `nonzero` coefficients survive, the rest are zero: nonzero = 0 is the degenerate
        # lasso fit that predicts its intercept for every row.
        self.coef_ = np.array([1.0] * nonzero + [0.0] * (4 - nonzero))
        self.n_iter_ = 2000
        self.max_iter = 2000

    def fit(self, matrix: Any, target: np.ndarray) -> "Recorded":
        assert matrix.shape[0] == target.size
        return self

    def predict(self, matrix: Any) -> np.ndarray:
        return np.full(matrix.shape[0], self.constant, dtype=np.float32)


def test_the_run_records_which_alpha_won_on_validation(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict[str, Any],
        fixture: dict[str, Any]) -> None:
    run_config = json.loads(json.dumps(config))
    run_config["endpoints"] = list(run_config["endpoints"])[:ENDPOINTS]
    run_config["stage3"]["classical"].update({
        "max_train_rows": ROWS, "chunk_rows": 2, "alphas": [1.0],
        "alphas_by_model": {"ridge": [0.5, 1.0]}})
    run_config["stage3"]["train"].update({"selection_metric": "mae",
                                          "selection_slice": "all"})
    arrays, caches = fixture["arrays"], fixture["caches"]
    keys = {"molecule_a": arrays.molecule_a, "molecule_b": arrays.molecule_b,
            "protein": arrays.protein, "endpoint": arrays.endpoint}
    columns = sum(int(block["width"]) for block
                  in classical.block_layout(caches, ENDPOINTS, True))

    # The labels are 0, 1, 2, 3, 4: a constant 2.0 is their mean and scores mae 1.2, a
    # constant 0.25 scores 1.85 (the absolute errors sum to 9.25). So alpha 1.0 wins -
    # which also proves the choice is not just "the smallest alpha in the grid".
    fits = {0.5: Recorded(0.25, nonzero=0), 1.0: Recorded(2.0, nonzero=3)}
    monkeypatch.setattr(train, "Caches", lambda *args, **kwargs: caches)
    monkeypatch.setattr(train, "prepare_arrays",
                        lambda *args, **kwargs: (arrays, dict(keys), None))
    monkeypatch.setattr(train, "layout", lambda config, root: (tmp_path, tmp_path))
    monkeypatch.setattr(classical, "build_matrix",
                        lambda arrays_in, indices, *args: FakeMatrix(len(indices), columns))
    monkeypatch.setattr(classical, "build_estimator",
                        lambda name, alpha, settings, seed: fits[alpha])

    manifest = classical.run_classical(run_config, tmp_path, "frozen",
                                       [classical.RIDGE], stamp="t")

    assert manifest["alphas"] == {"ridge": [0.5, 1.0]}
    assert manifest["selected_alpha"]["ridge"]["alpha"] == 1.0
    assert manifest["selected_alpha"]["ridge"]["value"] == pytest.approx(1.2)
    assert manifest["selected_alpha"]["ridge"]["alphas_compared"] == 2
    # The losing fit is still in the manifest, and it is marked for what it is.
    degenerate = manifest["results"]["ridge_alpha0.5"]
    assert degenerate["coefficients"] == {"total": 4, "nonzero": 0,
                                          "constant_predictor": True}
    assert degenerate["convergence"] == {"iterations": 2000, "max_iter": 2000,
                                         "hit_the_cap": True}
    assert manifest["results"]["ridge_alpha1"]["coefficients"]["nonzero"] == 3


def test_the_slice_decides_the_winner_as_much_as_the_metric() -> None:
    """Review 2026-11-20: the case that matters for the real batch.

    The project ranks on pairwise_accuracy_macro over the 'strict' slice. Here alpha 0.1
    wins on 'all' and alpha 10 wins on 'strict', with the same metric - so a run that chose
    its alpha on the wrong slice would quote the other fit and nothing would look wrong.
    """
    results = {
        "ridge_alpha0.1": {
            "model": classical.RIDGE, "alpha": 0.1,
            "validation_metrics": {
                "all": {"ranking": {"pairwise_accuracy_macro": 0.71}},
                "strict": {"ranking": {"pairwise_accuracy_macro": 0.58}}}},
        "ridge_alpha10": {
            "model": classical.RIDGE, "alpha": 10.0,
            "validation_metrics": {
                "all": {"ranking": {"pairwise_accuracy_macro": 0.66}},
                "strict": {"ranking": {"pairwise_accuracy_macro": 0.63}}}},
    }
    on_all = classical.best_alpha(results, classical.RIDGE, "pairwise_accuracy_macro", "all")
    on_strict = classical.best_alpha(results, classical.RIDGE, "pairwise_accuracy_macro",
                                     "strict")
    assert (on_all["alpha"], on_all["value"]) == (0.1, 0.71)
    assert (on_strict["alpha"], on_strict["value"]) == (10.0, 0.63)


def test_a_selection_slice_with_no_validation_rows_is_refused(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict[str, Any],
        fixture: dict[str, Any]) -> None:
    """Review 2026-11-20, taken one step further: the alpha is chosen on a slice, so a
    slice with no validation rows has to stop the run BEFORE the fits, the way run_train
    stops. Otherwise nine fits are paid for and the manifest names no winner."""
    run_config = json.loads(json.dumps(config))
    run_config["endpoints"] = list(run_config["endpoints"])[:ENDPOINTS]
    run_config["stage3"]["classical"].update({"max_train_rows": ROWS, "chunk_rows": 2,
                                              "alphas": [1.0]})
    run_config["stage3"]["train"].update({"selection_metric": "mae",
                                          "selection_slice": "strict"})
    arrays, caches = fixture["arrays"], fixture["caches"]
    # The fixture's arrays carry an 'all' slice only, so 'strict' has no rows at all.
    run_config["stage3"]["metrics"]["slices"] = [{"name": "all", "filters": {}},
                                                 {"name": "strict",
                                                  "filters": {"unseen_pair": True}}]
    keys = {"molecule_a": arrays.molecule_a, "molecule_b": arrays.molecule_b,
            "protein": arrays.protein, "endpoint": arrays.endpoint}
    monkeypatch.setattr(train, "Caches", lambda *args, **kwargs: caches)
    monkeypatch.setattr(train, "prepare_arrays",
                        lambda *args, **kwargs: (arrays, dict(keys), None))
    monkeypatch.setattr(train, "layout", lambda config, root: (tmp_path, tmp_path))
    monkeypatch.setattr(classical, "build_matrix",
                        lambda *args: (_ for _ in ()).throw(
                            AssertionError("no fit may be reached")))

    with pytest.raises(ValueError, match="strict"):
        classical.run_classical(run_config, tmp_path, "frozen", [classical.RIDGE],
                                stamp="t")


def test_a_model_whose_metric_is_undefined_is_named_not_left_out(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict[str, Any],
        fixture: dict[str, Any]) -> None:
    """With pairwise accuracy there may be no rankable query at all, and then no alpha can
    be chosen. The manifest has to SAY so: an absent key reads as "not implemented"."""
    run_config = json.loads(json.dumps(config))
    run_config["endpoints"] = list(run_config["endpoints"])[:ENDPOINTS]
    run_config["stage3"]["classical"].update({"max_train_rows": ROWS, "chunk_rows": 2,
                                              "alphas": [1.0]})
    run_config["stage3"]["train"].update(
        {"selection_metric": "pairwise_accuracy_macro", "selection_slice": "all"})
    # No query reaches 99 candidates, so every query is skipped and the ranking metric
    # comes back None - the same state a real slice with no rankable query would be in.
    run_config["stage3"]["metrics"]["min_candidates"] = 99
    arrays, caches = fixture["arrays"], fixture["caches"]
    keys = {"molecule_a": arrays.molecule_a, "molecule_b": arrays.molecule_b,
            "protein": arrays.protein, "endpoint": arrays.endpoint}
    columns = sum(int(block["width"]) for block
                  in classical.block_layout(caches, ENDPOINTS, True))
    monkeypatch.setattr(train, "Caches", lambda *args, **kwargs: caches)
    monkeypatch.setattr(train, "prepare_arrays",
                        lambda *args, **kwargs: (arrays, dict(keys), None))
    monkeypatch.setattr(train, "layout", lambda config, root: (tmp_path, tmp_path))
    monkeypatch.setattr(classical, "build_matrix",
                        lambda arrays_in, indices, *args: FakeMatrix(len(indices), columns))
    monkeypatch.setattr(classical, "build_estimator",
                        lambda *args, **kwargs: Recorded(0.25, nonzero=2))

    manifest = classical.run_classical(run_config, tmp_path, "frozen",
                                       [classical.RIDGE], stamp="t")

    # The ranking metric is None, so there is nothing to choose between.
    assert manifest["results"]["ridge_alpha1"]["validation_metrics"]["all"]["ranking"][
        "pairwise_accuracy_macro"] is None
    assert manifest["selected_alpha"] == {}
    assert manifest["selected_alpha_unavailable"] == ["ridge"]
