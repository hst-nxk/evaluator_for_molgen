"""The two extra FULL seeds: the batch that gives t13full's difference a spread.

Only the seed may differ from t13full. These tests hold that, hold the per-seed controls so
nobody pairs a FULL run with the wrong SUB100 number, and hold the two places where this
batch must not repeat a mistake already made: the patience that the eval_every=0 branch
reads, and the 200,000-row train measurement that is not comparable to t9c's train/all.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts" / "server_run_full_seeds_2211.sh").read_text(encoding="utf-8")
EPOCH_RULE = (ROOT / "scripts" / "server_run_full_epoch_rule_2111.sh").read_text(
    encoding="utf-8")
TRAIN = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["stage3"]["train"]


def test_the_two_seeds_are_the_ones_sub100_already_has() -> None:
    assert "SEEDS=(20261024 20261025)" in SCRIPT
    # And the seed t13full used is NOT repeated here.
    assert "20261023" not in SCRIPT.split("SEEDS=")[1]


def test_each_seed_is_paired_with_its_own_control() -> None:
    for number in ("0.55057", "0.54994", "0.55084"):
        assert number in SCRIPT, number
    assert "t9c2" in SCRIPT and "t9c3" in SCRIPT
    # The measured difference being given a spread is stated, not left implied.
    assert "0.54394" in SCRIPT and "0.56266" in SCRIPT
    assert "-0.00663" in SCRIPT and "+0.01351" in SCRIPT


def test_the_rule_is_t13full_s_and_the_patience_is_written_out() -> None:
    assert "--set eval_every_steps=0" in SCRIPT
    assert "--set early_stopping_patience=5" in SCRIPT
    assert TRAIN["early_stopping_patience"] == 2, "the default would change the rule"
    assert "train.py:1671" in SCRIPT


def test_nothing_but_the_seed_differs_from_t13full() -> None:
    for flag in ("model.dropout=0.5", "learning_rate=9.86e-05", "weight_decay=0.0",
                 "model.residual_blocks=7", "model.residual_width=102",
                 "model.residual_hidden_factor=1.849",
                 "model.residual_hidden_dropout=0.3936", "model.residual_dropout=0.0",
                 "model.fusion=concat", "model.ligand_features=fingerprint",
                 "eval_batch_size=2048", "selection_slice=strict",
                 "selection_metric=pairwise_accuracy_macro", "alignment_weight=0.0",
                 "augment_reverse=false", "antisymmetry_penalty=0.0", "min_edit_rows=1"):
        assert flag in EPOCH_RULE, flag
        assert flag in SCRIPT, flag
    assert "--set epochs" not in SCRIPT and "--set batch_size" not in SCRIPT


def test_the_full_partition_is_reached_by_clearing_the_subset() -> None:
    for key in ("inputs.train_subset", "inputs.partition_override",
                "inputs.partition_override_pairs"):
        assert f"--set {key}=" in SCRIPT, key


def test_the_train_side_is_scored_on_the_whole_partition_with_the_caveat_recorded() -> None:
    assert "--mode eval --partition train" in SCRIPT
    assert "0.77484" in SCRIPT and "0.62818" in SCRIPT
    assert "NOT comparable" in SCRIPT


def test_a_trained_seed_without_a_checkpoint_counts_as_one_failure() -> None:
    assert SCRIPT.count("FAILED=$((FAILED + 1))") == 2      # the run() path and this one
    assert "trained but produced NO usable checkpoint" in SCRIPT
    # The picker is only consulted when the training command itself succeeded.
    assert "if [[ $trained -eq 0 ]]; then" in SCRIPT
    assert "|| true" in SCRIPT and "2>> \"$LOG\"" in SCRIPT


def test_one_lock_one_log_one_pid() -> None:
    assert SCRIPT.count('claim_gpu "$GPU"') == 1
    assert 'if ! mv "$LOG" "$KEPT"; then' in SCRIPT
    assert "exit 4" in SCRIPT
    assert SCRIPT.count("$$") >= 4      # pid in the kept log name and in every run id


def test_every_run_id_is_unique_per_seed() -> None:
    labels = re.findall(r'label="([^"]+)"', SCRIPT)
    assert labels == ['t14full$stamp$((position + 2))'], labels
    assert '--run-id "$label-$$"' in SCRIPT
    assert '--run-id "$label-eval-$$"' in SCRIPT
