"""A null result needs a noise scale, and the summary must not print one without the other.

Why these tests exist. The 2026-11-19 batch measured +0.0027 for shuffling the protein
column, from one run per side. The failure mode this script exists to prevent is reporting
a mean difference as if it were an effect, so the tests below check that the spread is
always printed next to the mean, that a missing side refuses the difference instead of
filling it in, and that the batch script asks the two sides exactly the same question.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "protein_control_seeds.py"
SPEC = importlib.util.spec_from_file_location("protein_control_seeds", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)

SCRIPT = (ROOT / "scripts" / "server_run_protein_seeds_2011.sh").read_text(
    encoding="utf-8")
PID = 54321


SPLIT = "work/splits/pair_split_1411.parquet"


def manifest(path: Path, *, pairwise: float, seed: int, permute: int = -1,
             metric: str = "pairwise_accuracy_macro", slice_name: str = "strict",
             split: str = SPLIT, head: str = "direct") -> None:
    """One train manifest, with the fields a pair is identified by at the top level -
    where `--mode train` writes them (src/molopt/evaluator/train.py). `permute` of -1 is
    an ordinary run, which is what the real manifests record; >= 0 is a shuffled one."""
    payload = {"selection_metric": metric, "selection_slice": slice_name,
               "seed": seed, "protein_permute_seed": permute,
               "partition_override_pairs": split, "train_subset": "",
               "configuration": "frozen", "head": head,
               "overrides": {"seed": seed},
               "validation_metrics": {"strict": {
                   "regression": {"n": 24946, "mae": 0.51},
                   "ranking": {"pairwise_accuracy_macro": pairwise,
                               "queries_ranked": 5645, "queries_total": 42777}}}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def pairs(directory: Path, scores: list[tuple[float, float]]) -> None:
    """One manifest per side of each pair, named the way the batch names them."""
    for index, (real, shuffled) in enumerate(scores, start=1):
        seed = 20261022 + index
        manifest(directory / f"train_frozen_direct_a{index}_t5r{index}-{PID}.json",
                 pairwise=real, seed=seed)
        manifest(directory / f"train_frozen_direct_b{index}_t5s{index}-{PID}.json",
                 pairwise=shuffled, seed=seed, permute=20261019 + 100 * index)


@pytest.fixture()
def runs(tmp_path: Path) -> Path:
    directory = tmp_path / "evaluator"
    directory.mkdir()
    pairs(directory, [(0.7778, 0.7805), (0.7801, 0.7760), (0.7742, 0.7799)])
    return directory


def text(runs: Path) -> str:
    good, lines = summary.report(summary.rows(runs, PID, "strict"), "strict")
    return ("OK\n" if good else "INCOMPLETE\n") + "\n".join(lines)


def test_each_pair_prints_both_scores_and_their_own_difference(runs: Path) -> None:
    out = text(runs)
    assert "| 1 | 20261023 | 0.7778 | 0.7805 | +0.0027 |" in out
    assert "| 2 | 20261024 | 0.7801 | 0.7760 | -0.0041 |" in out
    assert "| 3 | 20261025 | 0.7742 | 0.7799 | +0.0057 |" in out
    assert out.startswith("OK")


def test_the_mean_is_never_printed_without_the_spread(runs: Path) -> None:
    # This is the whole point of the script: a mean difference alone reads as an effect.
    out = text(runs)
    assert "Mean difference (shuffled - real): +0.0014 over 3 pair(s)." in out
    assert "Spread of the differences: 0.0098 (from -0.0041 to +0.0057)." in out
    assert "Standard deviation of the differences: 0.0050." in out
    assert out.index("Mean difference") < out.index("Spread of the differences")


def test_the_reading_offered_is_a_bound_not_a_proof(runs: Path) -> None:
    out = text(runs)
    # The earlier wording said the effect was "bounded above by the spread", which does
    # not follow from three differences; what the script may say is what the numbers are.
    assert "bounded" not in out
    assert "These are descriptive statistics over 3 pairs on PAIR75" in out
    assert "NOT a bound on the effect, NOT a significance test" in out
    assert "three pairs cannot provide either" in out


def test_pair_one_is_not_offered_as_a_fourth_pair(runs: Path) -> None:
    # Pair 1 repeats the 2026-11-19 configuration, so adding t3a/t3b to the table would
    # count one draw twice.
    out = text(runs)
    assert "reproducibility check of t3a/t3b - not a fourth, independent pair" in out


def test_two_sides_with_different_seeds_are_not_a_pair(runs: Path) -> None:
    manifest(runs / f"train_frozen_direct_b2_t5s2-{PID}.json", pairwise=0.9,
             seed=20261099, permute=20261219)
    out = text(runs)
    assert "the two sides differ in seed (20261024 vs 20261099)" in out
    assert "| 2 | 20261024 | 0.7801 | 0.9000 | - |" in out
    assert "over 2 pair(s)" in out


def test_two_shuffled_sides_are_not_a_pair(runs: Path) -> None:
    # Both sides shuffled is a valid pair of runs and a meaningless difference.
    manifest(runs / f"train_frozen_direct_a3_t5r3-{PID}.json", pairwise=0.7742,
             seed=20261025, permute=20261319)
    out = text(runs)
    assert "the real side was itself SHUFFLED (protein_permute_seed=20261319)" in out
    assert out.startswith("INCOMPLETE")


def test_an_unshuffled_shuffled_side_is_not_a_pair(runs: Path) -> None:
    manifest(runs / f"train_frozen_direct_b1_t5s1-{PID}.json", pairwise=0.7805,
             seed=20261023, permute=-1)
    assert "the shuffled side was NOT shuffled (protein_permute_seed=-1)" in text(runs)


def test_two_sides_on_different_splits_are_not_a_pair(runs: Path) -> None:
    manifest(runs / f"train_frozen_direct_b1_t5s1-{PID}.json", pairwise=0.7805,
             seed=20261023, permute=20261219, split="work/splits/other.parquet")
    assert "the two sides differ in split" in text(runs)


def test_a_manifest_from_before_the_control_existed_is_refused(runs: Path) -> None:
    # Manifests written before 2026-11-19 have no protein_permute_seed at all, so a real
    # run and a shuffled run look identical in them.
    path = runs / f"train_frozen_direct_a1_t5r1-{PID}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["protein_permute_seed"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert "records no protein_permute_seed" in text(runs)


def test_a_missing_side_refuses_the_difference(runs: Path) -> None:
    (runs / f"train_frozen_direct_b2_t5s2-{PID}.json").unlink()
    out = text(runs)
    assert "| 2 | 20261024 | 0.7801 | - | - |" in out
    assert "t5s2: expected exactly one manifest, found 0 (none)" in out
    assert out.startswith("INCOMPLETE")
    # The mean is still printed, over the pairs that exist, and says how many.
    assert "over 2 pair(s)" in out


def test_a_side_selected_by_another_rule_is_not_compared(runs: Path, tmp_path: Path
                                                         ) -> None:
    # Two runs that chose their checkpoint differently measure the rule, not the protein.
    manifest(runs / f"train_frozen_direct_b1_t5s1-{PID}.json", pairwise=0.9,
             seed=20261023, permute=20261119, metric="mae")
    out = text(runs)
    assert "selection_metric is 'mae'" in out
    assert "0.9000" not in out


def test_the_scale_of_one_side_alone_is_printed_beside_the_difference(runs: Path
                                                                      ) -> None:
    """A mean difference means nothing without the scale it is read against: how much the
    plain number moves when only the training seed changes. The paired design is meant to
    make the difference steadier than either side, and printing both is what shows
    whether it did. Printed by the script so that no number in a report is arithmetic
    somebody did by hand over the rows of the table.
    """
    out = text(runs)
    assert "the REAL side alone across these 3 seeds: spread 0.0059" in out
    assert "(from 0.7742 to 0.7801), standard deviation 0.0030." in out
    assert out.index("Standard deviation of the differences") < out.index("REAL side")


def test_the_measured_batch_of_20_11_reads_the_way_it_was_reported(tmp_path: Path
                                                                   ) -> None:
    """The three pairs actually measured on the server on 2026-11-20. Kept as a test so
    the sentences quoted from them stay reproducible from the code: the differences are
    an order of magnitude steadier than the side they are taken from, which is the whole
    argument for pairing on the seed.
    """
    directory = tmp_path / "measured"
    directory.mkdir()
    pairs(directory, [(0.7778, 0.7805), (0.7696, 0.7719), (0.7834, 0.7874)])
    good, lines = summary.report(summary.rows(directory, PID, "strict"), "strict")
    out = "\n".join(lines)
    assert good
    assert "Mean difference (shuffled - real): +0.0030 over 3 pair(s)." in out
    # 0.0009 and not the 0.0008 the server printed: these fixtures carry the scores
    # rounded to four decimals, and the real manifests carry full precision. The point
    # the test fixes is the ORDER OF MAGNITUDE between the two lines below, which is the
    # same either way.
    assert "Standard deviation of the differences: 0.0009." in out
    assert "the REAL side alone across these 3 seeds: spread 0.0138" in out
    assert "standard deviation 0.0069." in out


def test_one_pair_is_told_it_has_no_noise_scale(tmp_path: Path) -> None:
    directory = tmp_path / "one"
    directory.mkdir()
    pairs(directory, [(0.7778, 0.7805)])
    good, lines = summary.report(summary.rows(directory, PID, "strict"), "strict")
    out = "\n".join(lines)
    assert not good
    assert "One pair only: this is the number that has no noise scale yet" in out
    assert "Standard deviation" not in out


def test_the_cli_exit_codes(runs: Path, tmp_path: Path) -> None:
    assert summary.main([str(runs), "--pid", str(PID)]) == 0
    assert summary.main([str(runs), "--pid", "999"]) == 1
    assert summary.main([str(runs)]) == 2
    assert summary.main([str(tmp_path / "nope"), "--pid", "1"]) == 2


def test_the_two_sides_of_a_pair_differ_in_one_flag_only() -> None:
    # Read out of the script itself: every flag except protein_permute_seed has to be the
    # same on both sides, or the pair measures two changes at once.
    real = [line for line in SCRIPT.splitlines() if "--run-id \"t5r$pair" in line]
    shuffled = [line for line in SCRIPT.splitlines() if "--run-id \"t5s$pair" in line]
    assert len(real) == 1 and len(shuffled) == 1
    # Only ONE command in the whole script may carry the shuffle flag; everything else the
    # two sides are given comes from the same $REF and $RESIDUAL variables above them.
    assert "--set \"protein_permute_seed=$permute\"" in SCRIPT
    assert SCRIPT.count("--set \"protein_permute_seed=") == 1
    assert SCRIPT.count("--set \"inputs.partition_override_pairs=$PAIR_SPLIT\"") == 2
    assert SCRIPT.count("--set \"seed=$seed\"") == 2


def test_the_batch_holds_the_selection_rule_of_the_1911_batch() -> None:
    assert ("SELECTION=\"--set selection_metric=pairwise_accuracy_macro "
            "--set selection_slice=strict\"") in SCRIPT
    assert "REF=\"$SELECTION\"" in SCRIPT
    # The reference hyper-parameters, so these runs compare with t3a/t3b and not only
    # with each other.
    for flag in ("model.dropout=0.5", "learning_rate=9.86e-05", "weight_decay=0.0",
                 "model.residual_blocks=7", "model.residual_width=102",
                 "model.residual_hidden_factor=1.849",
                 "model.residual_hidden_dropout=0.3936"):
        assert flag in SCRIPT


def test_the_first_seed_reproduces_the_1911_batch() -> None:
    # Pair 1 repeats t3a/t3b exactly, so a mismatch there says something else changed.
    assert "SEEDS=(20261023 20261024 20261025)" in SCRIPT
    assert "PERMUTE=(20261119 20261219 20261319)" in SCRIPT


def test_the_batch_takes_the_lock_and_points_at_the_summary() -> None:
    assert "source \"$ROOT/scripts/gpu_batch_lock.sh\"" in SCRIPT
    assert "claim_gpu \"$GPU\" \"$ROOT/.cache\" \"protein_seeds_2011\"" in SCRIPT
    assert SCRIPT.count("summary_hint") == 3
    assert "python scripts/protein_control_seeds.py runs/evaluator $LOG" in SCRIPT


# --------------------------------------------- what the batch IS, not just that it agrees
# Review 2026-11-20: pair_problems only asks whether the two sides of a pair agree with
# each other. Two manifests on the DEFAULT split with seed 42 agreed perfectly and were
# still reported as "PAIR75" and as a repetition of t3a/t3b.


def test_the_default_split_is_not_reported_as_pair75(tmp_path: Path) -> None:
    directory = tmp_path / "other"
    directory.mkdir()
    for index, (real, shuffled) in enumerate([(0.70, 0.71), (0.72, 0.70)], start=1):
        manifest(directory / f"a{index}_t5r{index}-{PID}.json", pairwise=real,
                 seed=40 + index, split="")
        manifest(directory / f"b{index}_t5s{index}-{PID}.json", pairwise=shuffled,
                 seed=40 + index, permute=99, split="")
    good, lines = summary.report(summary.rows(directory, PID, "strict"), "strict")
    out = "\n".join(lines)
    assert not good
    assert "this is not the PAIR75 batch" in out
    assert "pairs on the split below" in out
    # The two claims that depend on it are withheld, not reworded. (The words still
    # appear inside the REFUSAL, which is why the affirmative sentence is what is
    # searched for.)
    assert "pairs on PAIR75" not in out
    assert "so it is a reproducibility check" not in out
    assert "WHAT THIS BATCH IS COULD NOT BE CONFIRMED" in out


def test_one_seed_reused_across_pairs_is_not_a_spread_across_seeds(tmp_path: Path
                                                                   ) -> None:
    directory = tmp_path / "oneseed"
    directory.mkdir()
    for index, (real, shuffled) in enumerate([(0.70, 0.71), (0.72, 0.70)], start=1):
        manifest(directory / f"a{index}_t5r{index}-{PID}.json", pairwise=real,
                 seed=20261023)
        manifest(directory / f"b{index}_t5s{index}-{PID}.json", pairwise=shuffled,
                 seed=20261023, permute=20261119 + index)
    good, lines = summary.report(summary.rows(directory, PID, "strict"), "strict")
    out = "\n".join(lines)
    assert not good
    assert "do not have one training seed each" in out
    assert "not the spread across seeds" in out


def test_pair_one_must_repeat_the_1911_configuration_to_be_called_that(tmp_path: Path
                                                                      ) -> None:
    directory = tmp_path / "shifted"
    directory.mkdir()
    pairs(directory, [(0.7778, 0.7805), (0.7801, 0.7760), (0.7742, 0.7799)])
    # Pair 1 on a different training seed: everything still agrees inside the pair.
    manifest(directory / f"train_frozen_direct_a1_t5r1-{PID}.json", pairwise=0.7778,
             seed=19990101)
    manifest(directory / f"train_frozen_direct_b1_t5s1-{PID}.json", pairwise=0.7805,
             seed=19990101, permute=20261119)
    good, lines = summary.report(summary.rows(directory, PID, "strict"), "strict")
    out = "\n".join(lines)
    assert not good
    assert "pair 1 does not repeat the 2026-11-19 configuration" in out
    assert "expected 20261023 and 20261119" in out
    assert "reproducibility check of t3a/t3b - not a fourth" not in out


def test_the_verified_batch_still_prints_both_labels(runs: Path) -> None:
    good, lines = summary.report(summary.rows(runs, PID, "strict"), "strict")
    out = "\n".join(lines)
    assert good
    assert "pairs on PAIR75" in out
    assert "reproducibility check of t3a/t3b" in out
    assert "COULD NOT BE CONFIRMED" not in out
