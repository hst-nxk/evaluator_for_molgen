"""`scripts/classical_row_dump.py`: refit a classical baseline from its own manifest and
dump its validation rows, so the seen/unseen molecule breakdown can be run on the
baselines as well as on the MLP.

Everything that decides which model gets refitted is tested here. The fit itself needs the
real feature caches and the dataset, so it is not exercised; what IS exercised is the part
that would silently produce the wrong answer - picking an alpha the run did not select,
dumping a protein-control run, or accepting a refit whose numbers are not the manifest's.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "classical_row_dump", ROOT / "scripts" / "classical_row_dump.py")
assert SPEC is not None and SPEC.loader is not None
dumper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dumper)

SPLIT = "work/splits/pair_split_1411.parquet"


def scores(mae: float, r2: float, pairwise: float) -> dict[str, Any]:
    return {"regression": {"n": 24946, "mae": mae, "r2": r2},
            "ranking": {"pairwise_accuracy_macro": pairwise,
                        "pairwise_accuracy_micro": pairwise}}


def fit(model: str, alpha: float, pairwise: float) -> dict[str, Any]:
    return {"model": model, "alpha": alpha,
            "train_metrics": {"all": scores(0.5478, 0.3, pairwise + 0.02)},
            "validation_metrics": {"all": scores(0.6039, 0.0127, pairwise),
                                   "strict": scores(0.6039, 0.0127, pairwise)}}


def manifest(**changes: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": "stage3_classical", "configuration": "frozen",
        "protein_mode": "frozen", "seed": 20261023,
        "partition_override_pairs": SPLIT, "partition_override": "", "train_subset": "",
        "protein_permute_seed": -1, "max_train_rows": 100000, "chunk_rows": 20000,
        "feature_columns": 6628, "complete": True,
        "rows": {"train_available": 75000, "train_fit": 75000, "validation": 24946},
        "results": {"linear": fit("linear", 0.0, 0.6330),
                    "ridge_a0.1": fit("ridge", 0.1, 0.6332),
                    "ridge_a1.0": fit("ridge", 1.0, 0.6300),
                    "lasso_a0.0001": fit("lasso", 0.0001, 0.6267),
                    "xgboost": fit("xgboost", 0.0, 0.6801)},
        "selected_alpha": {"ridge": {"alpha": 0.1, "label": "ridge_a0.1",
                                     "value": 0.6332, "alphas_compared": 2}},
        "selected_alpha_unavailable": [],
    }
    payload.update(changes)
    return payload


def write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "classical_frozen_20261120_t1c-4174688.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------- what must be refused
def test_a_protein_control_run_is_not_dumped() -> None:
    """A run whose protein column was shuffled answers a different question. Its rows
    would look like an ordinary baseline's in the dump, and the breakdown quoted from it
    would be read as one.
    """
    problems = dumper.refusals(manifest(protein_permute_seed=20261119))
    assert len(problems) == 1
    assert "SHUFFLED" in problems[0]


def test_a_run_without_a_pair_split_is_not_dumped() -> None:
    problems = dumper.refusals(manifest(partition_override_pairs=""))
    assert any("partition_override_pairs" in problem for problem in problems)


def test_an_ordinary_run_is_accepted() -> None:
    assert dumper.refusals(manifest()) == []


def test_a_manifest_of_another_stage_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, manifest(stage="stage3_train"))
    with pytest.raises(ValueError, match="not a classical manifest"):
        dumper.read_manifest(path)


# --------------------------------------------------------------- which fit gets refitted
def test_the_selected_alpha_is_used_and_not_the_first_of_the_grid() -> None:
    """The run compared two ridge alphas and selected 0.1. Refitting 1.0 - the other one,
    or whichever the dictionary happens to yield first - would dump a model whose numbers
    the manifest never reported.
    """
    alpha, label = dumper.alpha_for(manifest(), "ridge")
    assert alpha == 0.1
    assert label == "ridge_a0.1"


def test_the_label_recorded_by_the_run_is_preferred() -> None:
    # A label spelled differently from what run_label would build today still points at
    # the right fit, because the manifest's own label is used.
    payload = manifest()
    payload["results"]["ridge at one tenth"] = payload["results"].pop("ridge_a0.1")
    payload["selected_alpha"]["ridge"] = {"alpha": 0.1, "label": "ridge at one tenth"}
    assert dumper.alpha_for(payload, "ridge")[1] == "ridge at one tenth"


def test_a_regularised_model_without_a_selected_alpha_is_refused() -> None:
    payload = manifest(selected_alpha={}, selected_alpha_unavailable=["ridge"])
    with pytest.raises(ValueError, match="selected_alpha_unavailable"):
        dumper.alpha_for(payload, "ridge")


def test_a_regularised_model_missing_from_both_places_is_still_refused() -> None:
    with pytest.raises(ValueError, match="records no selected alpha"):
        dumper.alpha_for(manifest(selected_alpha={}), "lasso")


def test_a_selected_alpha_with_no_fit_behind_it_is_refused() -> None:
    payload = manifest(selected_alpha={"ridge": {"alpha": 7.0, "label": "ridge_a7.0"}})
    with pytest.raises(ValueError, match="has no fit in the manifest"):
        dumper.alpha_for(payload, "ridge")


def test_an_unregularised_model_needs_no_selection() -> None:
    # linear and xgboost have one fit each; there is nothing to select between.
    assert dumper.alpha_for(manifest(), "linear") == (0.0, "linear")
    assert dumper.alpha_for(manifest(), "xgboost") == (0.0, "xgboost")


def test_a_model_the_manifest_never_fitted_is_refused() -> None:
    payload = manifest()
    del payload["results"]["xgboost"]
    with pytest.raises(ValueError, match="no xgboost fit"):
        dumper.alpha_for(payload, "xgboost")


# --------------------------------------------------------------- reading what the run read
def test_the_split_and_the_seed_come_from_the_manifest() -> None:
    config = dumper.load_config(manifest())
    assert config["stage3"]["inputs"]["partition_override_pairs"] == SPLIT
    # The seed decides which rows the row cap keeps, so today's config value must not win.
    assert config["stage3"]["train"]["seed"] == 20261023
    assert config["stage3"]["classical"]["max_train_rows"] == 100000
    assert config["stage3"]["classical"]["chunk_rows"] == 20000


def test_an_empty_override_is_not_written_into_the_config() -> None:
    # A manifest with no train_subset must not turn into inputs.train_subset = "" if the
    # config happens to carry something there; only what the run set is applied.
    assert dumper.settings_from(manifest()) == {
        "inputs.partition_override_pairs": SPLIT}


# --------------------------------------------------------------- the reproduce check
def test_identical_metrics_agree() -> None:
    recorded = {"all": scores(0.6039, 0.0127, 0.6332)}
    assert dumper.differences(recorded, {"all": scores(0.6039, 0.0127, 0.6332)}) == []


def test_a_different_macro_is_reported_with_both_numbers() -> None:
    recorded = {"all": scores(0.6039, 0.0127, 0.6332)}
    problems = dumper.differences(recorded, {"all": scores(0.6039, 0.0127, 0.6500)})
    # Both pairwise metrics move together in this fixture, and both are reported: the
    # message names the metric and prints the two numbers, so the log says what did not
    # match rather than only that something did not.
    assert len(problems) == 2
    assert any("pairwise_accuracy_macro" in problem for problem in problems)
    assert all("0.633200" in problem and "0.650000" in problem for problem in problems)


def test_a_difference_in_mae_is_caught_too() -> None:
    recorded = {"all": scores(0.6039, 0.0127, 0.6332)}
    problems = dumper.differences(recorded, {"all": scores(0.6100, 0.0127, 0.6332)})
    assert any("mae" in problem for problem in problems)


def test_a_missing_slice_is_a_disagreement() -> None:
    recorded = {"all": scores(0.6039, 0.0127, 0.6332),
                "strict": scores(0.6039, 0.0127, 0.6332)}
    problems = dumper.differences(recorded, {"all": scores(0.6039, 0.0127, 0.6332)})
    assert problems == ["the refit produced no 'strict' slice"]


def test_an_undefined_metric_has_to_be_undefined_on_both_sides() -> None:
    recorded = {"all": {"regression": {"mae": 0.6, "r2": None},
                        "ranking": {"pairwise_accuracy_macro": None}}}
    same = {"all": {"regression": {"mae": 0.6, "r2": None},
                    "ranking": {"pairwise_accuracy_macro": None}}}
    assert dumper.differences(recorded, same) == []
    other = {"all": {"regression": {"mae": 0.6, "r2": 0.1},
                     "ranking": {"pairwise_accuracy_macro": None}}}
    assert dumper.differences(recorded, other) == ["all/regression/r2: manifest None, "
                                                   "refit 0.1"]


def test_the_tolerance_is_tight_enough_to_mean_the_same_model() -> None:
    # These models are deterministic given their seed, so the check is for the wrong model
    # or the wrong rows - not a budget for genuine run-to-run variation.
    assert dumper.TOLERANCE <= 1e-8
    recorded = {"all": scores(0.6039, 0.0127, 0.6332)}
    nudged = {"all": scores(0.6039, 0.0127, 0.6332 + 1e-6)}
    assert dumper.differences(recorded, nudged) != []


# --------------------------------------------------------------- the command line
def test_a_missing_manifest_exits_two(tmp_path: Path) -> None:
    assert dumper.main([str(tmp_path / "nope.json")]) == 2


def test_an_unknown_model_exits_two(tmp_path: Path) -> None:
    path = write(tmp_path, manifest())
    assert dumper.main([str(path), "--models", "randomforest"]) == 2


def test_all_means_every_model_the_manifest_holds(tmp_path: Path, monkeypatch) -> None:
    path = write(tmp_path, manifest())
    seen: dict[str, Any] = {}

    def fake(manifest_path: Path, models: list[str], out_root: Path | None) -> int:
        seen["models"] = models
        return 0

    monkeypatch.setattr(dumper, "build", fake)
    assert dumper.main([str(path)]) == 0
    assert seen["models"] == ["linear", "ridge", "lasso", "xgboost"]


def test_a_manifest_with_no_fits_exits_two(tmp_path: Path) -> None:
    path = write(tmp_path, manifest(results={}))
    assert dumper.main([str(path)]) == 2


def test_a_refusal_before_any_fit_costs_nothing(tmp_path: Path) -> None:
    """The protein-control refusal has to happen BEFORE the feature matrix is built: this
    runs on CPU for minutes per model, and a refusal afterwards would have spent it.
    """
    path = write(tmp_path, manifest(protein_permute_seed=20261119))
    printed: list[str] = []
    assert dumper.build(path, ["ridge"], None, log=printed.append) == 1
    assert any("MUST NOT BE DUMPED" in line for line in printed)
    assert not any("refitting" in line for line in printed)


# --------------------------------------------------------------- the server batch
SCRIPT = (ROOT / "scripts" / "server_run_classical_seen_2011.sh").read_text(
    encoding="utf-8")


def test_the_batch_takes_no_gpu_lock() -> None:
    # Every model here is CPU-only, so claiming a GPU would block a training batch for
    # nothing.
    assert "claim_gpu" not in SCRIPT
    assert "CUDA_VISIBLE_DEVICES" not in SCRIPT
    assert "NO GPU" in SCRIPT


def test_the_batch_refuses_a_train_dump_that_is_not_one() -> None:
    assert '[[ ! -f "$TRAIN_DUMP/rows.parquet" ]]' in SCRIPT


def test_the_batch_runs_the_dump_before_the_breakdown() -> None:
    assert SCRIPT.index("classical_row_dump.py") < SCRIPT.index("seen_molecules.py")


def test_the_batch_passes_the_same_slice_the_mlp_was_read_at() -> None:
    # The MLP breakdown was run at `all`, which on a pair-keyed split is every validation
    # row; reading the baselines at another slice would not compare.
    assert "--slice all" in SCRIPT


def test_the_batch_carries_the_numbers_it_is_to_be_compared_with() -> None:
    assert "0.6673 / 0.7103 / 0.8311" in SCRIPT


def test_a_failed_step_is_carried_through_as_a_failure() -> None:
    assert "FINISHED WITH $FAILED FAILED COMMAND(S)" in SCRIPT
    assert "exit 1" in SCRIPT
