"""Task 12 of the 2026-11-09 meeting: the CLIP-style alignment term.

What has to hold for the experiment to measure anything:
  * the term is 0-ish when a molecule's token already matches its own fragment's token,
    and near ln(batch) when the pairing is random - otherwise its scale means nothing;
  * it is symmetric in the two directions, as CLIP's is;
  * it reaches the weights, and the reported train loss does NOT include it;
  * it is refused, loudly, where it cannot mean anything: fusion='concat' (no shared
    space) and an architecture without an edit branch (no partner to align with);
  * off by default, and a run with it off calls the model exactly as before.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from molopt.evaluator import models, train
from test_evaluator_models import ENDPOINTS, FRAGMENT_BITS, FUSION_SETTINGS, LIGAND_BITS

ROOT = Path(__file__).resolve().parents[1]


def parts_of(ligand: torch.Tensor, edit: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"ligand_a": ligand, "ligand_b": ligand, "edit_a": edit, "edit_b": edit}


def test_the_scale_of_the_term_is_what_the_config_says_it_is() -> None:
    """The three landmarks the log line and the config note promise.

    ln(batch) is the value when every similarity is EQUAL - a flat representation - and it
    does not depend on the temperature. It is NOT the value of a random pairing: at the
    default 0.07 the factor 1/t is about 14, which stretches the spread, so a random
    pairing lands well ABOVE ln(batch). The first version of this note said "random =
    ln(batch)", which was wrong and would have made every logged number unreadable.
    """
    torch.manual_seed(0)
    rows, width = 32, 16
    vectors = torch.randn(rows, width)
    flat = train.alignment_term(parts_of(torch.ones(rows, width),
                                         torch.ones(rows, width)), 0.07)
    perfect = train.alignment_term(parts_of(vectors, vectors.clone()), 0.07)
    random_pairing = train.alignment_term(parts_of(vectors, torch.randn(rows, width)), 0.07)
    # Every similarity identical -> exactly the uniform softmax, at any temperature.
    assert float(flat) == pytest.approx(math.log(rows), abs=1e-5)
    assert float(train.alignment_term(parts_of(torch.ones(rows, width),
                                               torch.ones(rows, width)), 1.0)
                 ) == pytest.approx(math.log(rows), abs=1e-5)
    assert float(perfect) < 0.01                       # aligned
    assert float(random_pairing) > math.log(rows)      # actively mismatched, amplified


def test_the_term_does_not_depend_on_the_order_of_the_two_sides() -> None:
    """CLIP takes the cross entropy both ways; swapping the roles must change nothing."""
    torch.manual_seed(1)
    left, right = torch.randn(12, 8), torch.randn(12, 8)
    forward = train.alignment_term(parts_of(left, right), 0.05)
    backward = train.alignment_term(parts_of(right, left), 0.05)
    assert float(forward) == pytest.approx(float(backward), abs=1e-6)


def test_only_the_length_direction_matters_not_the_length() -> None:
    # Both sides are L2-normalised, so scaling a whole side is not a change.
    torch.manual_seed(2)
    left, right = torch.randn(10, 8), torch.randn(10, 8)
    plain = train.alignment_term(parts_of(left, right), 0.07)
    scaled = train.alignment_term(parts_of(left * 7.0, right * 0.2), 0.07)
    assert float(plain) == pytest.approx(float(scaled), abs=1e-5)


def test_a_lower_temperature_sharpens_the_term() -> None:
    torch.manual_seed(3)
    left = torch.randn(16, 8)
    right = left + 0.3 * torch.randn(16, 8)
    sharp = float(train.alignment_term(parts_of(left, right), 0.02))
    soft = float(train.alignment_term(parts_of(left, right), 1.0))
    assert sharp != soft


def test_it_carries_a_gradient_to_both_sides() -> None:
    torch.manual_seed(4)
    left = torch.randn(8, 6, requires_grad=True)
    right = torch.randn(8, 6, requires_grad=True)
    train.alignment_term(parts_of(left, right), 0.07).backward()
    assert left.grad is not None and float(left.grad.abs().sum()) > 0.0
    assert right.grad is not None and float(right.grad.abs().sum()) > 0.0


def test_one_row_is_refused_instead_of_being_a_silent_zero() -> None:
    single = torch.randn(1, 4)
    with pytest.raises(ValueError, match="batch >= 2"):
        train.alignment_term(parts_of(single, single), 0.07)


def test_a_missing_edit_token_is_refused() -> None:
    vectors = torch.randn(4, 4)
    with pytest.raises(ValueError, match="edit"):
        train.alignment_term({"ligand_a": vectors, "ligand_b": vectors}, 0.07)


def test_a_temperature_of_zero_is_refused() -> None:
    vectors = torch.randn(4, 4)
    with pytest.raises(ValueError, match="temperature"):
        train.alignment_term(parts_of(vectors, vectors), 0.0)


# ------------------------------------------------------- the tokens come from the model
def build(**changes: Any) -> models.EvaluatorModel:
    torch.manual_seed(0)
    settings = {**FUSION_SETTINGS, **changes}
    mode = (models.PROTEIN_NONE if changes.get("protein_none")
            else models.PROTEIN_FROZEN)
    settings.pop("protein_none", None)
    dimension = None if mode == models.PROTEIN_NONE else 8
    return models.EvaluatorModel(settings, LIGAND_BITS, FRAGMENT_BITS, dimension,
                                 ENDPOINTS, changes.pop("head", models.HEAD_DIRECT), mode)


def inputs(rows: int = 5) -> dict[str, Any]:
    torch.manual_seed(1)
    return {"ligand_a": torch.rand(rows, LIGAND_BITS).round(),
            "ligand_b": torch.rand(rows, LIGAND_BITS).round(),
            "fragment_a": torch.rand(rows, FRAGMENT_BITS).round(),
            "fragment_b": torch.rand(rows, FRAGMENT_BITS).round(),
            "protein": torch.randn(rows, 8),
            "endpoint": torch.zeros(rows, dtype=torch.long)}


def test_the_model_hands_out_the_projected_tokens_in_the_shared_width() -> None:
    """The alignment is only a constraint if both tokens live in ONE space."""
    model = build(fusion=models.FUSION_PROJECTOR)
    parts: dict[str, torch.Tensor] = {}
    prediction = model(**inputs(), parts=parts)
    shared = int(FUSION_SETTINGS["fusion_dim"])
    assert set(parts) == {"ligand_a", "ligand_b", "edit_a", "edit_b"}
    for name, token in parts.items():
        assert token.shape == (5, shared), name
    assert prediction.shape == (5,)


def test_asking_for_the_tokens_does_not_change_the_prediction() -> None:
    model = build(fusion=models.FUSION_PROJECTOR).eval()
    with torch.no_grad():
        plain = model(**inputs())
        with_parts = model(**inputs(), parts={})
    assert torch.allclose(plain, with_parts)


def test_the_tokens_are_the_ones_the_prediction_was_made_from() -> None:
    """No second pass through the encoders: with dropout on, a second pass would draw
    different masks and the term would be aligning vectors the head never saw."""
    model = build(fusion=models.FUSION_PROJECTOR, dropout=0.5).train()
    parts: dict[str, torch.Tensor] = {}
    batch = inputs()
    model(**batch, parts=parts)
    again: dict[str, torch.Tensor] = {}
    model(**batch, parts=again)
    # Two independent passes differ under dropout - which is why the tokens have to be
    # handed out by the pass itself rather than recomputed.
    assert not torch.allclose(parts["ligand_a"], again["ligand_a"])


# ------------------------------------------------------- the two hard conditions
def test_concat_fusion_is_refused_because_there_is_no_shared_space() -> None:
    assert build(fusion=models.FUSION_CONCAT).shares_one_space is False
    assert build(fusion=models.FUSION_PROJECTOR).shares_one_space is True
    assert build(fusion=models.FUSION_ATTENTION).shares_one_space is True
    source = (ROOT / "src" / "molopt" / "evaluator" / "train.py").read_text(
        encoding="utf-8")
    assert "alignment_weight > 0 đòi fusion='projector'" in source
    assert "alignment_weight > 0 đòi nhánh edit" in source


def test_an_architecture_without_an_edit_branch_cannot_align() -> None:
    assert build(fusion=models.FUSION_PROJECTOR,
                 head=models.HEAD_DIRECT_NO_EDIT).uses_edit_branch is False


# ------------------------------------------------------- the loop and the config
def test_the_weight_reaches_the_loss_and_the_reported_loss_excludes_it() -> None:
    """The train-loss column must stay comparable with runs that have the term off."""
    source = (ROOT / "src" / "molopt" / "evaluator" / "train.py").read_text(
        encoding="utf-8")
    body = source[source.index("def train_one_epoch("):source.index("def effective_batch")]
    data_loss = body.index("total += float(loss.detach()")
    added = body.index("loss = loss + alignment_weight * alignment")
    assert data_loss < added, "số hạng phụ bị cộng vào TRƯỚC khi ghi lại mất mát dữ liệu"
    assert "alignment_total" in body and 'stats["alignment_loss"]' in body
    # And nothing is passed to the model when the term is off.
    assert "model(*arguments) if parts is None" in body


def test_the_config_has_the_flag_off_and_says_what_it_costs() -> None:
    import json
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    settings = config["stage3"]["train"]
    assert settings["alignment_weight"] == 0.0
    assert settings["alignment_temperature"] == 0.07
    note = settings["alignment_note"]
    assert "arXiv:2103.00020" in note                 # CLIP, the equation being used
    assert "KHONG co luot forward them" in note       # unlike antisymmetry_penalty
    assert "ln(batch_size)" in note                   # how to read the number
    assert "KHONG suy nguoc" in note                  # the reading does not reverse
    assert "31,2 %" in note                           # the duplicate rate it was built for
    assert "arXiv:2004.11362" in note                 # the multi-positive form
    assert "w * alignment" in note                    # both numbers are reported
    assert "projector" in note and "attention" in note

# ------------------------------------------------------- the batch that measures it
BATCH = (ROOT / "scripts" / "server_run_alignment_2111.sh").read_text(encoding="utf-8")
REFERENCE = (ROOT / "scripts" / "server_run_protein_seeds_2011.sh").read_text(
    encoding="utf-8")


def test_the_control_is_inside_the_batch_and_shares_the_fusion() -> None:
    """A difference against the concat reference runs would measure two changes at once."""
    assert "WEIGHTS=(0.0 0.1 1.0)" in BATCH
    assert "--set model.fusion=projector" in BATCH
    assert "NOT the concat reference runs" in BATCH
    # The weight is set per arm only, never in the shared flag string.
    shared = BATCH.split("SEEDS=(")[0].split("SETTINGS=")[-1]
    assert "alignment_weight" not in shared
    assert '--set "alignment_weight=$weight"' in BATCH


def test_the_batch_holds_the_reference_flags_it_shares_with_the_other_runs() -> None:
    for flag in ("model.dropout=0.5", "learning_rate=9.86e-05", "weight_decay=0.0",
                 "model.residual_width=102", "model.residual_hidden_factor=1.849",
                 "model.residual_hidden_dropout=0.3936", "model.residual_dropout=0.0",
                 "model.residual_blocks=7"):
        assert flag in REFERENCE, f"{flag} is no longer what the reference batch used"
        assert flag in BATCH, flag
    assert "SEEDS=(20261023 20261024 20261025)" in BATCH


def test_the_batch_reports_a_lost_train_side_as_a_failure() -> None:
    assert "trained but produced NO usable checkpoint" in BATCH
    body = BATCH[BATCH.index("train_and_score() {"):BATCH.index('say "CLIP-style')]
    assert body.count("FAILED=$((FAILED + 1))") == 1
    assert 'if ! mv "$LOG" "$KEPT"; then' in BATCH and "exit 4" in BATCH
    assert BATCH.count("claim_gpu") == 1


def test_the_batch_says_how_to_tell_a_run_that_did_not_test_anything() -> None:
    # A validation score means nothing if the term was a rounding error beside the loss.
    assert "alignment_loss_last_epoch" in BATCH
    assert "alignment_contribution_last_epoch" in BATCH
    assert "did not test the hypothesis" in BATCH
    # And the batch must not claim the reading runs backwards.
    assert "It does not" in BATCH and "run backwards" in BATCH
    assert "FLAT representation" not in BATCH


def test_the_batch_states_the_duplicate_rate_it_was_designed_around() -> None:
    assert "31.2 %" in BATCH
    assert "31,219 of 100,000" in BATCH
    assert "MASK, not a diagonal" in BATCH
    assert "train.positive_mask" in BATCH

# =================================================== review 2026-11-21: nhieu o duong
def test_the_decisive_case_two_rows_sharing_one_fragment() -> None:
    """Batch (M1,F), (M2,F), (M3,G): matching M1 or M2 with F must COUNT, in either column.

    This is the case the review named. With a diagonal loss, row 0's molecule paired with
    row 1's fragment is a negative - although both fragments are the same F - so the loss
    punishes exactly the pairing the hypothesis is about. On SUB100 that is not a corner
    case: 31.2 % of rows share their variable_a with another row of the same batch.
    """
    molecule = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    fragment = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])   # F, F, G
    mask = train.positive_mask(molecule, fragment)
    assert mask.tolist() == [[True, True, False],
                             [True, True, False],
                             [False, False, True]]


def test_the_same_molecule_in_two_rows_is_also_a_positive() -> None:
    # Row j holding (M_i, F_j) makes that pair an observed one, so it is not a negative.
    molecule = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    fragment = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    mask = train.positive_mask(molecule, fragment)
    assert bool(mask[0, 1]) and bool(mask[1, 0])
    # Cell (0, 2) is M1 with F2, and row 1 holds exactly that pair, so under the exact
    # rule it is a POSITIVE - it was a negative under the earlier cheap rule, which is
    # the inconsistency the second review round found.
    assert bool(mask[0, 2])
    # A cell no row stands for is still a negative: M2 never appears with F1.
    assert not bool(mask[2, 0])


def test_a_batch_with_no_repeats_gives_exactly_the_diagonal() -> None:
    molecule = torch.eye(4)
    fragment = torch.eye(4)[[1, 2, 3, 0]]
    assert torch.equal(train.positive_mask(molecule, fragment),
                       torch.eye(4, dtype=torch.bool))


def test_the_mask_changes_the_loss_where_a_fragment_repeats() -> None:
    """The point of the fix: the shared-fragment cell stops being a negative."""
    torch.manual_seed(0)
    rows, width = 6, 8
    tokens = torch.randn(rows, width)
    edits = torch.randn(rows, width)
    molecule = torch.eye(rows)
    fragment = torch.eye(rows)
    fragment[1] = fragment[0]                       # rows 0 and 1 share one fragment
    mask = train.positive_mask(molecule, fragment)
    diagonal = train.alignment_term(parts_of(tokens, edits), 0.07)
    masked = train.alignment_term(parts_of(tokens, edits), 0.07,
                                  {"a": mask, "b": mask})
    assert float(diagonal) != pytest.approx(float(masked), abs=1e-6)


def test_a_row_with_many_partners_does_not_outweigh_a_row_with_one() -> None:
    """|P_i| divides, so the 31 % of repeated rows cannot dominate the term.

    Built so every similarity is equal: then each row's mean log-probability is log(1/B)
    whatever the number of positives, and the whole term is ln(B) - the same value the
    one-positive case gives.
    """
    rows = 8
    flat = torch.ones(rows, 4)
    full = torch.ones(rows, rows, dtype=torch.bool)
    assert float(train.alignment_term(parts_of(flat, flat), 0.07,
                                      {"a": full, "b": full})
                 ) == pytest.approx(math.log(rows), abs=1e-5)
    assert float(train.alignment_term(parts_of(flat, flat), 0.07)
                 ) == pytest.approx(math.log(rows), abs=1e-5)


def test_a_mask_of_the_wrong_shape_is_refused() -> None:
    tokens = torch.randn(4, 4)
    with pytest.raises(ValueError, match="mặt nạ"):
        train.alignment_term(parts_of(tokens, tokens), 0.07,
                             {"a": torch.ones(3, 3, dtype=torch.bool)})


def test_the_real_training_path_builds_the_mask_and_not_a_diagonal() -> None:
    source = (ROOT / "src" / "molopt" / "evaluator" / "train.py").read_text(
        encoding="utf-8")
    body = source[source.index("def train_one_epoch("):source.index("def effective_batch")]
    assert 'positive_mask(batch["ligand_a"], batch["fragment_a"])' in body
    assert 'positive_mask(batch["ligand_b"], batch["fragment_b"])' in body
    assert "alignment_term(parts, alignment_temperature, positives)" in body


# =================================================== review: doc so cho dung
def test_both_numbers_are_reported_not_only_the_weight() -> None:
    """"w = 0.1" says nothing about how heavy the term is beside the Huber loss."""
    source = (ROOT / "src" / "molopt" / "evaluator" / "train.py").read_text(
        encoding="utf-8")
    assert 'stats["alignment_contribution"]' in source
    assert 'stats["data_loss"]' in source
    assert "data loss %.6g | alignment %.6g" in source
    assert "alignment_contribution_last_epoch" in source


def test_the_docstring_does_not_run_the_reading_backwards() -> None:
    doc = train.alignment_term.__doc__ or ""
    assert "It does NOT run backwards" in doc
    assert "does not prove the representation is flat" in doc


# =================================================== review: temperature 0 bi nuot
def test_a_temperature_of_zero_survives_the_config_path() -> None:
    """`settings.get(name, 0.07) or 0.07` turned a user's 0 into 0.07 in silence.

    The old tests called alignment_term directly, so they stayed green while the real
    training path ignored the value that was passed. This one goes through the config
    reader, which is where the bug was.
    """
    with pytest.raises(ValueError, match="alignment_temperature"):
        train.alignment_settings({"alignment_weight": 1.0, "alignment_temperature": 0.0})
    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="alignment_temperature"):
            train.alignment_settings({"alignment_weight": 1.0,
                                      "alignment_temperature": bad})


def test_the_default_applies_only_when_the_key_is_absent() -> None:
    assert train.alignment_settings({}) == (0.0, 0.07)
    assert train.alignment_settings({"alignment_weight": 0.5}) == (0.5, 0.07)
    assert train.alignment_settings({"alignment_weight": 0.5,
                                     "alignment_temperature": 0.2}) == (0.5, 0.2)
    # A temperature of 0 with the term OFF is not a run that can go wrong, so it passes
    # through unchanged rather than being rewritten to 0.07.
    assert train.alignment_settings({"alignment_weight": 0.0,
                                     "alignment_temperature": 0.0}) == (0.0, 0.0)


def test_a_negative_weight_is_refused() -> None:
    with pytest.raises(ValueError, match="alignment_weight"):
        train.alignment_settings({"alignment_weight": -0.1})


def test_the_training_path_uses_the_validated_reader() -> None:
    source = (ROOT / "src" / "molopt" / "evaluator" / "train.py").read_text(
        encoding="utf-8")
    assert "alignment_weight, alignment_temperature = alignment_settings(settings)" in source
    assert 'settings.get("alignment_temperature", 0.07) or 0.07' not in source

# ============================================ review 2026-11-21 (lan 2): du cap da thay
def test_a_pair_seen_only_through_a_third_row_is_a_positive() -> None:
    """The case the cheap rule missed, and the reason it had to go.

    Rows: (M1, F1), (M2, F2), (M1, F2). Row 0's molecule M1 does appear with F2 - in row 2 -
    so cell (0, 1) stands for a pair that WAS observed, even though row 0 and row 1 share
    neither their molecule nor their fragment. Under the earlier rule that cell was a
    negative while cell (0, 2), the same fingerprint pair (M1, F2), was a positive: the loss
    would have pushed one and pulled the other.
    """
    molecule = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    fragment = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    mask = train.positive_mask(molecule, fragment)
    assert bool(mask[0, 1]), "cap (M1, F2) da thay o dong 2 nen khong phai cap am"
    assert bool(mask[0, 2])
    # And the two cells for one fingerprint pair carry the SAME label - the real point.
    assert bool(mask[0, 1]) == bool(mask[0, 2])
    assert bool(mask[2, 0]) == bool(mask[0, 0])


def test_every_cell_of_one_fingerprint_pair_gets_one_label() -> None:
    """No two cells standing for the same (molecule, fragment) may disagree, ever."""
    torch.manual_seed(5)
    rows = 24
    molecule = torch.randint(0, 4, (rows, 3)).float()
    fragment = torch.randint(0, 3, (rows, 3)).float()
    mask = train.positive_mask(molecule, fragment)
    seen: dict[tuple, bool] = {}
    for i in range(rows):
        for j in range(rows):
            key = (tuple(molecule[i].tolist()), tuple(fragment[j].tolist()))
            value = bool(mask[i, j])
            assert seen.setdefault(key, value) == value, key


def test_an_unobserved_pair_is_still_a_negative() -> None:
    # The mask must not become "everything": a pair no row holds stays negative.
    molecule = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    fragment = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = train.positive_mask(molecule, fragment)
    assert mask.tolist() == [[True, False], [False, True]]


def test_the_docstring_carries_the_measured_cost_of_the_cheap_rule() -> None:
    doc = train.positive_mask.__doc__ or ""
    for number in ("208,960", "209,112", "152", "210,226", "210,482", "256"):
        assert number in doc, number
    assert "0.073 %" in doc and "0.122 %" in doc
    assert "not the count" in doc
