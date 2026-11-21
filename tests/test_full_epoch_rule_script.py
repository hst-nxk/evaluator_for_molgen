"""The second FULL run of 2026-11-21: the epoch rule, and what its header may claim.

The first FULL run was scored under a different rule from its own control, because one
`since_best` counter serves both the intra-epoch marks and the end-of-epoch check
(train.py:1671, 1249, 1872). This script exists to remove that difference, so these tests
hold the two flags that define the rule - including the patience that MUST be written out,
since the eval_every=0 branch reads a setting whose default is 2 - and they hold the line
that keeps the observation run out of the comparison.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts" / "server_run_full_epoch_rule_2111.sh").read_text(
    encoding="utf-8")
FIRST = (ROOT / "scripts" / "server_run_full_no_protein_2111.sh").read_text(
    encoding="utf-8")
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
TRAIN = CONFIG["stage3"]["train"]


def test_the_rule_is_end_of_epoch_selection() -> None:
    assert "--set eval_every_steps=0" in SCRIPT
    assert TRAIN["eval_every_steps"] > 0, "the default is intra-epoch, so this must be set"


def test_the_patience_is_written_out_because_the_default_would_change_the_rule() -> None:
    """eval_every=0 reads early_stopping_patience, which is 2 - not the 5 t9c1 got."""
    assert "--set early_stopping_patience=5" in SCRIPT
    assert TRAIN["early_stopping_patience"] == 2
    assert TRAIN["early_stopping_patience_evals"] == 5
    assert "default is 2" in SCRIPT


def test_the_header_cites_the_three_lines_that_explain_the_early_stop() -> None:
    for reference in ("train.py:1671", "train.py:1249", "train.py:1872"):
        assert reference in SCRIPT, reference
    assert "One `since_best` counter serves both" in SCRIPT
    # And the measured facts it is reacting to.
    assert "0.54545" in SCRIPT and "0.55057" in SCRIPT
    assert "12,945" in SCRIPT and "196 steps per epoch" in SCRIPT


def test_the_control_is_named_as_not_needing_a_rerun() -> None:
    """On SUB100 the intra-epoch marks never fire, so t9c1 already ran under this rule."""
    assert "NOT RERUN" in SCRIPT
    # The header wraps: "the intra-epoch marks never / fired".
    assert "intra-epoch marks never" in SCRIPT
    for number in ("0.55057", "0.54915", "0.62558", "0.07643"):
        assert number in SCRIPT, number


def test_the_budget_apart_from_the_rule_is_untouched() -> None:
    for flag in ("--set batch_size", "--set learning_rate=1", "--set epochs=10"):
        assert flag not in SCRIPT, flag
    # The epoch ceiling stays at the config's value for the comparison run; only the
    # observation run pins epochs, and it pins it to one.
    assert SCRIPT.count("--set epochs=1 ") == 1 or "--set epochs=1 \\" in SCRIPT


def test_the_reference_flags_still_match_the_first_full_run() -> None:
    for flag in ("model.dropout=0.5", "learning_rate=9.86e-05", "weight_decay=0.0",
                 "model.residual_blocks=7", "model.residual_width=102",
                 "model.residual_hidden_factor=1.849",
                 "model.residual_hidden_dropout=0.3936", "model.residual_dropout=0.0",
                 "seed=20261023", "selection_slice=strict",
                 "selection_metric=pairwise_accuracy_macro", "eval_batch_size=2048"):
        assert flag in FIRST, flag
        assert flag in SCRIPT, flag


def test_the_observation_run_is_separate_and_excluded_from_the_comparison() -> None:
    assert "--set eval_every_steps=500" in SCRIPT
    assert "must never be quoted beside t9c1" in SCRIPT
    assert "only its CURVE" in SCRIPT
    # Its own stopping must not cut the curve short.
    assert "--set early_stopping_patience_evals=100" in SCRIPT
    # And it must not be mistaken for the main run: a run id of its own.
    assert '"$task-curve-$$"' in SCRIPT
    assert SCRIPT.count('--run-id "$task-$$"') == 1


def test_the_train_side_caveat_travels_with_the_script() -> None:
    assert "200,000-row sample is NOT comparable" in SCRIPT
    assert "sparser" in SCRIPT and "6,627,559" in SCRIPT


def test_one_log_one_pid_and_one_lock() -> None:
    assert SCRIPT.count('claim_gpu "$gpu"') == 1
    assert 'mv -- "$log" "$log.$stamp-$$"' in SCRIPT
    claim = SCRIPT.index("claim_gpu ")
    redirect = SCRIPT.index('exec >>"$log" 2>&1')
    assert claim < redirect < SCRIPT.index("GPU $gpu claimed by pid $$")
    assert "set -Eeuo pipefail" in SCRIPT


def test_the_task_id_is_new_and_matches_the_readers_convention() -> None:
    assert 'task="t13full$stamp"' in SCRIPT
    # t12full is named in the prose on purpose; what matters is that no run
    # id of this script reuses it, which the loop below checks.
    for run_id in re.findall(r'--run-id "([^"]+)"', SCRIPT):
        assert run_id.startswith("$task"), run_id
