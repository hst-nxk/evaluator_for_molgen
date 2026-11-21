"""The dropout 2x2 of 2026-11-21, and the flag mix-up it was built on.

The batch exists because the train-versus-validation gap on SUB100 (0.0894 at `all` on
both sides) cannot be attributed: the only earlier run with dropout off (t2d) also changed
the budget and the stopping rule. The first version of this script claimed to sweep the
residual head's dropout and swept `model.dropout`, which - with the residual dropouts
passed explicitly, as the reference flag set does - reaches the two ENCODERS and never the
head. `test_model_dropout_does_not_reach_the_residual_head` holds that fact against the
model itself, so the naming cannot drift back.
"""
from pathlib import Path
from typing import Any

import torch

from molopt.evaluator import models
from test_evaluator_models import ENDPOINTS, FRAGMENT_BITS, FUSION_SETTINGS, LIGAND_BITS

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts" / "server_run_dropout_2111.sh").read_text(encoding="utf-8")
REFERENCE = (ROOT / "scripts" / "server_run_protein_seeds_2011.sh").read_text(
    encoding="utf-8")
OVERNIGHT = (ROOT / "scripts" / "server_run_overnight_2011.sh").read_text(encoding="utf-8")


def dropouts(**changes: Any) -> dict[str, list[float]]:
    """Every nn.Dropout of the reference architecture, by the branch it sits in."""
    settings = {**FUSION_SETTINGS, "head_block": models.HEAD_BLOCK_RESIDUAL,
                "residual_blocks": 7, "residual_width": 102,
                "residual_hidden_factor": 1.849, "residual_hidden_dropout": 0.3936,
                "residual_dropout": 0.0, **changes}
    model = models.EvaluatorModel(settings, LIGAND_BITS, FRAGMENT_BITS, None, ENDPOINTS,
                                  models.HEAD_DIRECT, models.PROTEIN_NONE)
    found: dict[str, list[float]] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            found.setdefault(name.split(".")[0], []).append(round(float(module.p), 4))
    return found


def test_model_dropout_does_not_reach_the_residual_head() -> None:
    """The review of 2026-11-21, held against the model and not against a comment.

    `model.dropout` is the dropout of the ligand and edit encoders. The head's is
    `residual_hidden_dropout`, in all seven blocks, and it does not move when
    `model.dropout` does - which is why a `model.dropout` sweep is not a head sweep.
    """
    high = dropouts(dropout=0.5)
    low = dropouts(dropout=0.0)
    assert high["ligand_encoder"] and high["edit_encoder"]
    assert all(value == 0.5 for value in high["ligand_encoder"] + high["edit_encoder"])
    # At zero the encoders hold no Dropout at all: build_mlp leaves it out.
    assert "ligand_encoder" not in low and "edit_encoder" not in low
    # And the head is untouched by either, at 0.3936 inside each block and 0.0 after it.
    assert high["predictor"] == low["predictor"] == [0.3936, 0.0] * 7


def test_the_head_dropout_moves_only_with_its_own_flag() -> None:
    assert dropouts(residual_hidden_dropout=0.0)["predictor"] == [0.0, 0.0] * 7


def test_the_two_factors_are_the_two_flags_that_actually_do_the_job() -> None:
    assert "ENCODER=(0.5 0.5 0.0 0.0)" in SCRIPT
    assert "HEAD=(0.3936 0.0 0.3936 0.0)" in SCRIPT
    assert "--set \"model.dropout=$encoder\"" in SCRIPT
    assert "--set \"model.residual_hidden_dropout=$head\"" in SCRIPT
    # Neither swept flag may also sit in the shared string: it would be set twice.
    shared = SCRIPT.split("SEEDS=(")[0].split("SETTINGS=")[-1]
    assert "model.dropout=" not in shared
    assert "model.residual_hidden_dropout=" not in shared
    # residual_dropout stays 0.0 everywhere, so it belongs in the shared string.
    assert "--set model.residual_dropout=0.0" in SCRIPT


def test_the_script_says_what_model_dropout_reaches() -> None:
    assert "is the ENCODERS' dropout" in SCRIPT
    assert "never reaches it" in SCRIPT
    assert "ligand_encoder [0.5], edit_encoder [0.5]" in SCRIPT


def test_the_held_flags_are_read_from_the_batch_being_compared_with() -> None:
    for flag in ("learning_rate=9.86e-05", "weight_decay=0.0",
                 "model.residual_width=102", "model.residual_hidden_factor=1.849",
                 "model.residual_dropout=0.0", "model.residual_blocks=7",
                 "model.head_block=residual"):
        assert flag in REFERENCE, f"{flag} is no longer what the reference batch used"
        assert flag in SCRIPT, flag
    assert "SEEDS=(20261023 20261024 20261025)" in REFERENCE
    assert "SEEDS=(20261023 20261024 20261025)" in SCRIPT
    assert ("--set selection_metric=pairwise_accuracy_macro "
            "--set selection_slice=strict") in SCRIPT


def test_the_epoch_budget_is_not_overridden() -> None:
    # t2d's failure mode: dropout off AND a different budget AND a different stopping rule.
    for flag in ("--set epochs", "--set max_steps", "--set patience", "--set early_stop"):
        assert flag not in SCRIPT, flag


def test_the_control_is_t9c_and_not_t9a() -> None:
    """The 0.0894 gap was measured on t9a, which has the ESM2 branch; this batch has not."""
    assert "must reproduce t9c" in SCRIPT
    assert "0.5504" in SCRIPT
    assert "t9a is" in SCRIPT and "NOT this batch's control" in SCRIPT
    assert "that one belongs to" in SCRIPT      # the 0.0894 gap, to t9a
    assert "CONFIGURATION=no_protein" in OVERNIGHT      # which arm t9c was
    assert "--config no_protein" in SCRIPT
    assert "--config frozen" not in SCRIPT


def test_the_reproduction_check_is_read_at_the_slice_t9c_was_reported_at() -> None:
    assert "validation/strict" in SCRIPT
    # Both tables are printed: `all`/`all` for the gap, the default strict for t9c.
    assert "--slice all --train-slice all" in SCRIPT
    assert SCRIPT.count("scripts/diagnostics_tables.py") == 2
    assert "train/strict holds 0 rows" in SCRIPT


def test_a_second_batch_on_one_gpu_does_not_share_a_log() -> None:
    """Both readers refuse a log with two pids, so the old log is moved aside, not deleted.

    It happens after the lock and before the first write: before the lock, two batches
    could rotate each other's log, and after the first write the new batch's own lines
    would travel into the kept file.
    """
    lock = SCRIPT.index("claim_gpu \"$GPU\"")
    rotate = SCRIPT.index('mv "$LOG" "$KEPT"')
    first_say = SCRIPT.index('say "dropout 2x2')
    assert lock < rotate < first_say
    assert 'KEPT="$LOG.$(date' in SCRIPT
    # Kept, never removed: no `rm` command anywhere in the script (as a command, not as
    # the tail of a word like "arm", which is why this is anchored).
    import re
    assert not re.search(r"(?m)^\s*rm", SCRIPT)
    assert "pid_from_log" in SCRIPT  # and the reason is named


def test_both_sides_of_the_gap_are_measured_at_the_selected_checkpoint() -> None:
    assert "measure_train_metrics=true" in SCRIPT
    assert "--mode eval" in SCRIPT and "--partition train" in SCRIPT
    assert "the SAME checkpoint scored on TRAIN" in SCRIPT


def test_the_batch_takes_the_gpu_lock_once() -> None:
    assert SCRIPT.count("claim_gpu") == 1
    assert 'claim_gpu "$GPU" "$ROOT/.cache" "dropout_2111"' in SCRIPT


def test_a_dead_training_run_skips_only_its_own_half() -> None:
    assert "usable checkpoint; its train half is skipped" in SCRIPT
    assert "LAST_STATUS=$status" in SCRIPT
    assert "set -u -o pipefail" in SCRIPT
    assert "\nset -e" not in SCRIPT


def test_twelve_runs_are_named_so_the_tables_can_find_them() -> None:
    assert 'train_and_score "t10$arm$((position + 1))"' in SCRIPT
    assert "ARMS=(a b c d)" in SCRIPT


def test_the_gap_it_explains_is_quoted_with_its_slice() -> None:
    assert "train/all" in SCRIPT and "validation/all" in SCRIPT
    assert "0.0894" in SCRIPT
    assert "CAUSE IS UNKNOWN" in SCRIPT

# ------------------------------------------- review of 2026-11-21, second round
def test_a_trained_run_without_a_checkpoint_is_counted_as_a_failure() -> None:
    """`|| true` swallows the picker's exit code, so the caller has to count the gap lost.

    Otherwise the batch can print "FINISHED with 0 failed command(s)" while one of the
    twelve gaps - the quantity it exists to measure - was never computed.
    """
    assert "trained but produced NO usable checkpoint" in SCRIPT
    body = SCRIPT[SCRIPT.index("train_and_score() {"):SCRIPT.index('say "dropout 2x2')]
    assert "FAILED=$((FAILED + 1))" in body
    # And not counted twice when the training run itself already failed.
    assert "no checkpoint because its training run failed" in body
    assert body.count("FAILED=$((FAILED + 1))") == 1
    assert '-z "$checkpoint" && ${LAST_STATUS:-1} -eq 0' in SCRIPT


def test_a_failed_rotation_stops_the_batch_before_the_gpu_is_used() -> None:
    # An unchecked mv would announce a saved log and then append to the old file, and both
    # readers refuse a log with two pids - there is nothing to salvage from that.
    assert 'if ! mv "$LOG" "$KEPT"; then' in SCRIPT
    assert "exit 4" in SCRIPT
    assert "not writing over it" in SCRIPT
    # The pid is in the kept name too: two starts inside one second keep both logs.
    assert 'KEPT="$LOG.$(date +%Y%m%d_%H%M%S)-$$"' in SCRIPT


def test_the_baseline_is_t9c_measured_and_not_the_frozen_number() -> None:
    """The gap to beat is t9c's 0.07381; 0.0894 is t9a's, which has the ESM2 branch."""
    for number in ("0.07381", "0.62301", "0.54920", "0.55045",
                   "0.55057", "0.54994", "0.55084", "0.57223", "0.58824"):
        assert number in SCRIPT, number
    assert "The gap to beat is therefore 0.07381" in SCRIPT
    assert "0.0894" in SCRIPT and "belongs to" in SCRIPT


def test_the_contrasts_are_written_down_and_are_per_seed() -> None:
    assert "b - a" in SCRIPT and "c - a" in SCRIPT and "d - c - b + a" in SCRIPT
    assert "Seed by seed" in SCRIPT
    # And the warning that a gap also shrinks when the train score falls.
    assert "a gap also narrows when the TRAIN score" in SCRIPT
