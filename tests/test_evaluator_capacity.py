"""The capacity check (molopt.evaluator.capacity) and the protein-shuffle control.

Two groups of tests:

1. Pure functions (row sampling, verdict, shuffle seed) — no torch needed.
2. Real `run_capacity` passes over fake data on the CPU, to catch wiring mistakes without
   a GPU and without touching the dataset. They do NOT check model quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from molopt.evaluator import capacity, data, train

# Reuse the fake caches and fake PairArrays of the training-loop test: a second copy of the
# same fake data would drift apart over time, and then two test files would be checking
# different things under the same names.
from test_evaluator_train_loop import FakeCaches, ROWS, fake_arrays  # noqa: E402


def test_the_sample_is_random_but_reproducible() -> None:
    first = capacity.sample_rows(1000, 10, seed=1)
    assert first.size == 10
    assert np.array_equal(first, capacity.sample_rows(1000, 10, seed=1))
    assert not np.array_equal(first, capacity.sample_rows(1000, 10, seed=2))
    # Not the first rows: the dataset is written in (endpoint, protein bucket) order, so
    # the first ten rows are a single endpoint.
    assert first.tolist() != list(range(10))
    assert np.unique(first).size == 10


def test_asking_for_more_rows_than_there_are_returns_everything() -> None:
    assert capacity.sample_rows(5, 50, seed=0).tolist() == [0, 1, 2, 3, 4]


def test_the_verdict_follows_the_threshold() -> None:
    assert capacity.verdict(0.99, 0.95) == capacity.FIT
    assert capacity.verdict(0.95, 0.95) == capacity.FIT
    assert capacity.verdict(0.40, 0.95) == capacity.NOT_FIT
    # R² is None when the labels have no variance: nothing can be concluded, and that must
    # not default to "does not fit".
    assert capacity.verdict(None, 0.95) == capacity.UNDECIDED


def test_the_permutation_seed_is_off_by_default_and_differs_per_partition() -> None:
    assert train.protein_permutation_seed({}, data.TRAIN) is None
    off = {"protein_permute_seed": train.PROTEIN_PERMUTE_OFF}
    assert train.protein_permutation_seed(off, data.TRAIN) is None
    on = {"protein_permute_seed": 100}
    seeds = {partition: train.protein_permutation_seed(on, partition)
             for partition in (data.TRAIN, data.VALIDATION)}
    assert seeds[data.TRAIN] != seeds[data.VALIDATION]
    assert all(seed is not None for seed in seeds.values())


def test_shuffling_protein_keeps_the_multiset_and_changes_nothing_else() -> None:
    rng = np.random.default_rng(3)
    arrays, _ = fake_arrays(rng, [{"name": "all"}])
    shuffled = data.permute_protein_rows(arrays, seed=5)
    assert sorted(shuffled.protein.tolist()) == sorted(arrays.protein.tolist())
    assert not np.array_equal(shuffled.protein, arrays.protein)
    for column in ("molecule_a", "molecule_b", "fragment_a", "fragment_b", "endpoint",
                   "target"):
        assert np.array_equal(getattr(shuffled, column), getattr(arrays, column))
    assert np.array_equal(data.permute_protein_rows(arrays, seed=5).protein,
                          shuffled.protein)


# --------------------------------------------------------------------- real run

torch = pytest.importorskip("torch", reason="the capacity check needs torch")


@pytest.fixture()
def small(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
          config: dict[str, Any]) -> dict[str, Any]:
    """A small config plus the monkeypatching run_capacity needs to run on the CPU."""
    rng = np.random.default_rng(0)
    run_config = json.loads(json.dumps(config))
    run_config["stage3"]["train"].update(
        {"device": "cpu", "batch_size": 16, "gpu_memory_limit_mib": None,
         "weight_decay": 0.05})
    run_config["stage3"]["model"].update(
        {"ligand_hidden": [16], "ligand_out": 8, "edit_hidden": [8], "edit_out": 4,
         "protein_out": 4, "endpoint_dim": 3, "head_hidden": [8], "dropout": 0.4})
    run_config["stage3"]["capacity"].update({"rows": ROWS, "epochs": 2, "read_rows": ROWS})
    arrays, keys = fake_arrays(rng, run_config["stage3"]["metrics"]["slices"])
    caches = FakeCaches(rng)
    monkeypatch.setattr(train, "Caches", lambda *args, **kwargs: caches)
    monkeypatch.setattr(train, "prepare_arrays",
                        lambda *args, **kwargs: (arrays, dict(keys), None))
    monkeypatch.setattr(train, "layout", lambda config, root: (tmp_path, tmp_path))
    return {"config": run_config, "root": tmp_path}


def test_a_capacity_check_turns_off_both_regularizers(small: dict[str, Any]) -> None:
    manifest = capacity.run_capacity(small["config"], small["root"], "frozen", "direct",
                                     stamp="t")
    assert manifest["dropout"] == 0.0
    assert manifest["weight_decay"] == 0.0
    # And it must not edit the run's config: copies, not in place.
    assert small["config"]["stage3"]["model"]["dropout"] == 0.4
    assert small["config"]["stage3"]["train"]["weight_decay"] == 0.05


def test_regularized_keeps_the_configured_values(small: dict[str, Any]) -> None:
    manifest = capacity.run_capacity(small["config"], small["root"], "frozen", "direct",
                                     stamp="t", regularized=True)
    assert manifest["dropout"] == 0.4
    assert manifest["weight_decay"] == 0.05


def test_the_manifest_records_the_sample_the_history_and_a_verdict(
        small: dict[str, Any]) -> None:
    manifest = capacity.run_capacity(small["config"], small["root"], "frozen", "direct",
                                     stamp="t")
    assert manifest["stage"] == "stage3_capacity"
    assert manifest["sample"]["rows"] == ROWS
    assert manifest["sample"]["distinct_proteins"] >= 1
    assert manifest["epochs_run"] == len(manifest["history"]) == 2
    assert [record["epoch"] for record in manifest["history"]] == [0, 1]
    assert manifest["verdict"] in (capacity.FIT, capacity.NOT_FIT, capacity.UNDECIDED)
    # A diagnostic: exactly one manifest, and NO checkpoint.
    runs_dir = small["root"]
    assert sorted(path.name for path in runs_dir.glob("capacity_*")) == [
        "capacity_frozen_direct_t.json"]
    assert not list(runs_dir.glob("*.pt"))


def test_the_dropout_keys_cover_every_dropout_in_the_model_block(
        config: dict[str, Any]) -> None:
    """Every dropout key of `stage3.model` has to appear in DROPOUT_KEYS.

    This test is a contract with the config: adding a new dropout knob to the architecture
    and forgetting it here would make the capacity check run with one hand tied while the
    manifest still reports dropout 0.
    """
    in_config = {key for key in config["stage3"]["model"]
                 if "dropout" in key and not key.endswith("note")}
    assert in_config <= set(capacity.DROPOUT_KEYS), in_config - set(capacity.DROPOUT_KEYS)


def test_every_dropout_layer_of_the_built_model_is_really_off(
        small: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Not just the manifest reporting 0: the REAL Dropout layers must have p = 0.

    The residual architecture has its own `residual_hidden_dropout` (the reference
    configuration sets 0.3936), so switching off `model.dropout` alone is not enough.
    """
    from molopt.evaluator import models

    small["config"]["stage3"]["model"].update(
        {"head_block": "residual", "residual_blocks": 2, "residual_width": 16,
         "residual_hidden_factor": 1.5, "residual_hidden_dropout": 0.3936,
         "residual_dropout": 0.25, "attention_dropout": 0.2})
    built: list[Any] = []
    real = models.EvaluatorModel

    def record(*args: Any, **kwargs: Any) -> Any:
        model = real(*args, **kwargs)
        built.append(model)
        return model

    monkeypatch.setattr(models, "EvaluatorModel", record)
    manifest = capacity.run_capacity(small["config"], small["root"], "frozen", "direct",
                                     stamp="t")
    assert built, "no model was built"
    probabilities = [module.p for module in built[0].modules()
                     if isinstance(module, torch.nn.Dropout)]
    assert probabilities, "this architecture has no Dropout layer to check"
    assert set(probabilities) == {0.0}, probabilities
    assert set(manifest["dropout_keys"].values()) == {0.0}


def test_the_run_stops_on_the_same_condition_the_verdict_uses(
        small: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Early stopping and the verdict have to use the SAME threshold.

    Before 2026-11-19 the loop stopped on MAE while the verdict looked at R², so one run
    could both log "sample fitted" and record `does_not_fit`.
    """
    small["config"]["stage3"]["capacity"]["epochs"] = 5

    # Pretend every measurement clears the threshold: the run must stop after epoch 0.
    def perfect(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"all": {"regression": {"mae": 0.5, "r2": 0.99, "n": 1},
                        "ranking": {"pairwise_accuracy_macro": 0.6}}}

    monkeypatch.setattr(train, "evaluate", perfect)
    manifest = capacity.run_capacity(small["config"], small["root"], "frozen", "direct",
                                     stamp="t")
    assert manifest["epochs_run"] == 1
    assert manifest["verdict"] == capacity.FIT


def test_a_low_r2_keeps_running_to_the_budget(small: dict[str, Any],
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """A small MAE with a low R² must NOT stop the run: that is the old mixed-up case."""
    small["config"]["stage3"]["capacity"]["epochs"] = 3

    def tiny_mae_low_r2(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"all": {"regression": {"mae": 0.001, "r2": 0.10, "n": 1},
                        "ranking": {"pairwise_accuracy_macro": 0.5}}}

    monkeypatch.setattr(train, "evaluate", tiny_mae_low_r2)
    manifest = capacity.run_capacity(small["config"], small["root"], "frozen", "direct",
                                     stamp="t")
    assert manifest["epochs_run"] == 3
    assert manifest["verdict"] == capacity.NOT_FIT


# ------------------------------------------- the shuffle seed inside the checkpoint


def test_a_checkpoint_carries_the_permutation_seed_it_was_trained_with(
        small: dict[str, Any]) -> None:
    """Without the seed in the .pt, re-scoring silently becomes train-shuffled/eval-clean."""
    from molopt.evaluator import train as train_module

    settings = small["config"]["stage3"]["train"]
    settings.update({"epochs": 1, "eval_every_steps": 0, "eval_batch_size": 32,
                     "protein_permute_seed": 777})
    manifest = train_module.run_train(small["config"], small["root"], "frozen", "direct",
                                      smoke=False, stamp="p")
    assert manifest["protein_permute_seed"] == 777
    state = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False)
    assert state["protein_permute_seed"] == 777


@pytest.mark.parametrize("saved,current,ok", (
    (-1, -1, True),      # an ordinary run, re-scored ordinarily
    (777, 777, True),    # a shuffled run, re-scored with the same seed
    (777, -1, False),    # THE TRAP: trained shuffled, scored clean — the number comes out
                         # higher than the truth
    (-1, 777, False),    # and the other way round
))
def test_mismatched_permutation_seeds_are_refused(saved: int, current: int,
                                                  ok: bool) -> None:
    state = {"protein_permute_seed": saved}
    settings = {"protein_permute_seed": current}
    if ok:
        assert train.check_protein_permutation(state, settings) == saved
    else:
        with pytest.raises(ValueError, match="protein_permute_seed"):
            train.check_protein_permutation(state, settings)


def test_an_old_checkpoint_without_the_key_is_accepted_only_without_the_flag() -> None:
    # Checkpoints from before 2026-11-19 do not carry the key; all of them are unshuffled
    # runs.
    assert train.check_protein_permutation({}, {"protein_permute_seed": -1}) is None
    with pytest.raises(ValueError, match="before 2026-11-19"):
        train.check_protein_permutation({}, {"protein_permute_seed": 5})


# ------------------------------------------- the note that goes with the verdict


SAMPLE = {"rows": 1024, "distinct_endpoints": 1, "distinct_proteins": 37,
          "window_rows": 200000}
DROPOUTS = {"dropout": 0.5, "residual_hidden_dropout": 0.3936}


def note(regularized: bool, **changes: Any) -> str:
    settings = {"sample": SAMPLE, "dropouts": DROPOUTS, "weight_decay": 0.0,
                "epochs_run": 12, **changes}
    return capacity.verdict_note(regularized, settings["sample"], settings["dropouts"],
                                 settings["weight_decay"], settings["epochs_run"])


def test_the_note_says_the_regularisers_were_off_only_when_they_were() -> None:
    """A --regularized run is the OPPOSITE experiment, and the note is the one field a
    reader goes to for the conclusion: it must not describe the wrong run."""
    off = note(False, dropouts={"dropout": 0.0, "residual_hidden_dropout": 0.0})
    on = note(True)
    assert "switched off" in off and "--regularized" not in off
    assert "AS CONFIGURED (--regularized)" in on and "switched off" not in on
    # And with the regularisers configured, a failure to fit says nothing about capacity.
    assert "does NOT mean a lack of capacity" in on
    assert "does NOT mean a lack of capacity" not in off


def test_the_note_prints_the_values_used_not_a_word_for_them() -> None:
    """Review point of 2026-11-19: the 19/11 batch passes `--set weight_decay=0.0` to its
    --regularized run as well, so calling the knobs "ON" would be false - that pair
    isolates dropout alone. The note states the numbers instead."""
    on = note(True)
    assert "dropout=0.5" in on and "residual_hidden_dropout=0.3936" in on
    assert "weight decay 0" in on
    off = note(False, dropouts={"dropout": 0.0, "residual_hidden_dropout": 0.0})
    assert "dropout=0" in off and "weight decay 0" in off


def test_fits_does_not_claim_more_than_this_sample() -> None:
    """Review point of 2026-11-19: `fits` only establishes the fit on THAT sample, with
    those hyper-parameters, within that budget."""
    text = note(False)
    assert "ON THIS SAMPLE" in text
    assert "after 12 epoch(s)" in text
    assert "says nothing about 100k rows" in text


def test_the_note_repeats_the_scope_of_the_sample() -> None:
    """1 endpoint out of a 200,000-row window: the verdict covers that and nothing wider."""
    text = note(False)
    assert "1024 rows" in text and "1 endpoint(s)" in text and "37 protein(s)" in text
    assert "200000 rows" in text


def test_a_failure_lists_the_optimisation_schedule_and_the_budget() -> None:
    # A fixed epoch budget means "did not fit" also has the schedule and the number of
    # update steps as candidate causes, and is not the same statement as "cannot fit".
    text = note(False)
    # What the check really returned, stated first: the threshold was not reached on this
    # sample within the epochs that ran. The causes come after that, as candidates.
    assert "NOT reached" in text and "within the epochs that were run" in text
    assert "optimisation schedule" in text
    assert 'is not "cannot"' in text


def test_the_manifest_uses_the_built_note(small: dict[str, Any]) -> None:
    manifest = capacity.run_capacity(small["config"], small["root"], "frozen", "direct",
                                     stamp="n", regularized=True)
    assert manifest["verdict_note"] == capacity.verdict_note(
        True, manifest["sample"], manifest["dropout_keys"], manifest["weight_decay"],
        manifest["epochs_run"])
    # The values in the note are the values the run really used.
    assert f"weight decay {manifest['weight_decay']:g}" in manifest["verdict_note"]
