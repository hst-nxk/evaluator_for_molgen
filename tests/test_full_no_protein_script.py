"""The FULL no_protein run of 2026-11-21: its flags, and the claims in its header.

The run's whole value is that it differs from t9c1 in ONE thing - the number of training
rows. These tests hold that: every flag that must match the reference batch, the absence of
any budget override, and the four control numbers plus the comparability argument written
into the header, so a later edit cannot quietly leave a stale baseline behind.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts" / "server_run_full_no_protein_2111.sh").read_text(
    encoding="utf-8")
REFERENCE = (ROOT / "scripts" / "server_run_protein_seeds_2011.sh").read_text(
    encoding="utf-8")
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def test_the_reference_flags_come_from_the_batch_being_compared_with() -> None:
    for flag in ("model.dropout=0.5", "learning_rate=9.86e-05", "weight_decay=0.0",
                 "model.residual_width=102", "model.residual_hidden_factor=1.849",
                 "model.residual_hidden_dropout=0.3936", "model.residual_dropout=0.0",
                 "model.residual_blocks=7", "model.head_block=residual"):
        assert flag in REFERENCE, f"{flag} is no longer what the reference batch used"
        assert flag in SCRIPT, flag
    assert "seed=20261023" in SCRIPT
    assert "selection_metric=pairwise_accuracy_macro" in SCRIPT
    assert "selection_slice=strict" in SCRIPT


def test_the_full_partition_is_reached_by_clearing_the_subset_not_by_a_new_file() -> None:
    """An empty value is a real override here: these settings are strings in the config."""
    for key in ("inputs.train_subset", "inputs.partition_override",
                "inputs.partition_override_pairs"):
        assert f"--set {key}=" in SCRIPT, key
        # The key has to exist and be a string, or apply_overrides refuses the run.
        leaf = key.split(".")[1]
        assert isinstance(CONFIG["stage3"]["inputs"][leaf], str), key


def test_the_eval_batch_size_is_t9c_s_and_not_the_config_default() -> None:
    assert "eval_batch_size=2048" in SCRIPT
    assert "eval_batch_size=2048" in REFERENCE
    assert CONFIG["stage3"]["train"]["eval_batch_size"] != 2048   # the default differs


def test_the_training_budget_is_not_overridden() -> None:
    for flag in ("--set epochs", "--set batch_size", "--set eval_every_steps",
                 "--set early_stopping", "--set max_steps"):
        assert flag not in SCRIPT, flag


def test_the_header_carries_the_four_control_numbers_of_t9c1() -> None:
    for number in ("0.55057", "0.54915", "0.62558", "0.07643"):
        assert number in SCRIPT, number
    assert "same seed 20261023" in SCRIPT
    # And the reason the comparison is legitimate at all.
    assert "subsets" in SCRIPT and "TRAIN partition only" in SCRIPT
    assert "6,627,559" in SCRIPT


def test_the_header_counts_batches_and_refuses_to_promise_a_time() -> None:
    """66x is batches per epoch - 12,945 against 196 - not a wall-clock factor."""
    rows = 6627559
    batch = int(CONFIG["stage3"]["train"]["batch_size"])
    assert batch == 512
    assert -(-rows // batch) == 12945
    assert -(-100000 // batch) == 196
    # The header wraps, so the numbers are checked rather than one unbroken phrase.
    assert "12,945 batches per" in SCRIPT
    assert "196" in SCRIPT
    assert "batches per epoch and says" in SCRIPT
    assert "wall-clock" in SCRIPT
    assert "FIRST validation mark" in SCRIPT


def test_the_header_states_what_one_seed_forbids_without_inventing_a_threshold() -> None:
    assert "0.00046" in SCRIPT
    assert "do NOT turn 0.001 into a threshold" in SCRIPT
    # The paired difference itself IS measured - that much must not be hedged away.
    assert "measured directly" in SCRIPT


def test_the_header_limits_the_scope_of_the_result() -> None:
    assert "does not establish the" in SCRIPT and "best configuration on FULL" in SCRIPT
    assert "more update steps" in SCRIPT
    assert "dropout 0.5" in SCRIPT and "too strong at this scale" in SCRIPT


def test_the_lock_is_recorded_after_the_redirection_and_not_before_it() -> None:
    """A run refused the lock must not write into the log of the batch that holds it."""
    claim = SCRIPT.index("claim_gpu ")
    redirect = SCRIPT.index('exec >>"$log" 2>&1')
    recorded = SCRIPT.index("GPU $gpu claimed by pid $$")
    assert claim < redirect < recorded
    assert "must not write into the log of the batch already holding it" in SCRIPT


def test_a_failure_stops_the_run_and_says_where() -> None:
    assert "set -Eeuo pipefail" in SCRIPT      # -E so the ERR trap reaches functions
    assert "FAILED (exit $?) at line $LINENO" in SCRIPT
    # One CALL, however many times the comments name it.
    assert SCRIPT.count('claim_gpu "$gpu"') == 1
    # The previous log is moved aside, with the pid in the kept name, never removed.
    assert 'mv -- "$log" "$log.$stamp-$$"' in SCRIPT


def test_the_train_side_is_measured_at_the_selected_checkpoint() -> None:
    assert "measure_train_metrics=true" in SCRIPT
    assert "measure_train_max_rows=200000" in SCRIPT
    assert "pick_train_checkpoint.py" in SCRIPT
    assert "--mode eval --partition train --checkpoint" in SCRIPT
    # The cap is for the in-training marks; the final pass covers every train row.
    assert "final eval covers every train row" in SCRIPT
