"""`scripts/seen_molecules_seeds.py` and the overnight batch.

The number the seen/unseen argument is missing is the seed's own movement: the MLP's macro
for the group with no molecule seen in training is 0.6673 over 215 queries, and a margin
taken from 215 queries cannot be quoted until it is known how far that figure moves when
only the seed changes. These tests hold the two things that would make such a summary
worse than none: averaging reports that are not the same measurement, and printing a mean
without the spread beside it.
"""
from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "seen_molecules_seeds", ROOT / "scripts" / "seen_molecules_seeds.py")
assert SPEC is not None and SPEC.loader is not None
seeds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seeds)

TRAIN_DUMP = "errors_frozen_direct_train_20261120_124613_t6train-3831959"


def report(macro: tuple[float, float, float], micro: tuple[float, float, float],
           **changes: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": "stage3_seen_molecules", "train": TRAIN_DUMP,
        "validation": "errors_frozen_direct_validation_x", "slice": "all",
        "label": "PAIR75 t5r1", "rows": 24946, "train_molecules": 94080,
        "direction_tolerance": 0.3, "min_candidates": 2,
        "min_candidates_source": "report.json",
        # Chứng minh đây là các lần chạy KHÁC NHAU trên CÙNG các dòng. `identity` và
        # `validation_seed` là thứ được so, không phải đường dẫn checkpoint: một fit cổ
        # điển không có `.pt` nào cả.
        "validation_fingerprint": "a1b2c3d4" * 4,
        "validation_fingerprint_columns":
            "pair_id,protein_hash,smiles_a,smiles_b,endpoint",
        "validation_checkpoint": "runs/evaluator/best_t5r1.pt",
        "identity": "checkpoint:best_t5r1.pt",
        "kind": "mlp",
        "validation_seed": 20261023,
        "by_row": [{"molecules_seen": group, "rows": rows, "mae": 0.5,
                    "direction_accuracy_micro": value, "comparisons": 4268}
                   for group, rows, value in zip((0, 1, 2), (6925, 7606, 10415), micro)],
        "by_seen_macro": [{"molecules_seen": group, "rows": rows, "queries": 13500,
                           "queries_eligible": 339, "queries_ranked": ranked,
                           "comparisons": 233, "pairwise_accuracy_macro": value}
                          for group, rows, ranked, value in
                          zip((0, 1, 2), (6925, 7606, 10415), (215, 494, 2392), macro)],
        "by_query": [], "overall": {"queries": 42777, "queries_eligible": 5645,
                                    "queries_ranked": 3892,
                                    "pairwise_accuracy_macro": 0.7778},
        "check": {"checked": True, "agrees": True}}
    payload.update(changes)
    return payload


def write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def three(tmp_path: Path) -> list[Path]:
    return [write(tmp_path / f"seen_molecules_t7s{index}.json",
                  report(macro, micro, **mlp_changes(index)))
            for index, (macro, micro) in enumerate(
                [((0.6673, 0.7103, 0.8311), (0.5677, 0.6727, 0.8092)),
                 ((0.6421, 0.7050, 0.8265), (0.5590, 0.6680, 0.8041)),
                 ((0.6890, 0.7210, 0.8350), (0.5731, 0.6766, 0.8130))], start=1)]


def mlp_changes(index: int) -> dict[str, Any]:
    return {"validation_checkpoint": f"runs/evaluator/best_t5r{index}.pt",
            "identity": f"checkpoint:best_t5r{index}.pt",
            "validation_seed": 20261022 + index}


# ------------------------------------------------------------------ the table
def test_every_group_is_a_mean_with_its_spread(tmp_path: Path) -> None:
    paths = three(tmp_path)
    out = "\n".join(seeds.lines(paths, [seeds.read(path) for path in paths]))
    macro = [0.6673, 0.6421, 0.6890]
    expected = (f"{statistics.fmean(macro):.4f} ±{statistics.stdev(macro):.4f} "
                f"[0.6421..0.6890]")
    assert expected in out
    # And no group is printed as a bare number.
    for line in out.splitlines():
        if line.startswith("| 0 |") or line.startswith("| 2 |"):
            assert "±" in line


def test_the_seed_spread_is_not_called_the_bar_for_beating_a_baseline(
        tmp_path: Path) -> None:
    """Review 20/11, lượt hai: câu "chỉ đáng trích nếu hiệu lớn hơn sd của MLP" vẫn còn.

    Sd của riêng MLP nói MLP dao động bao nhiêu; nó KHÔNG phải ngưỡng để nói MLP thắng
    lasso. Đại lượng mang kết luận đó là hiệu ghép cặp trên cùng các dòng.
    """
    paths = three(tmp_path)
    out = "\n".join(seeds.lines(paths, [seeds.read(path) for path in paths]))
    assert "The seed alone moves the macro of group" in out
    assert "how unstable the MLP is" in out
    assert "NOT the bar for beating a baseline" in out
    assert "paired difference on the same rows" in out
    # Câu cũ phải mất hẳn, không chỉ bị thêm câu khác vào cạnh.
    assert "only worth quoting when it is larger than this" not in out


def test_no_confidence_interval_is_claimed(tmp_path: Path) -> None:
    paths = three(tmp_path)
    out = "\n".join(seeds.lines(paths, [seeds.read(path) for path in paths]))
    assert "NOT a confidence interval" in out
    assert "NOT a significance test" in out
    assert "bound" not in out.lower()


def test_the_better_supported_unit_is_named(tmp_path: Path) -> None:
    # The macro of a group averages the queries that cleared min_candidates; the micro
    # column counts every comparable row. Where they disagree, that matters.
    paths = three(tmp_path)
    out = "\n".join(seeds.lines(paths, [seeds.read(path) for path in paths]))
    assert "the row-level one is the better supported" in out


def test_one_report_is_told_it_has_no_scale(tmp_path: Path) -> None:
    path = write(tmp_path / "seen_molecules_one.json",
                 report((0.6673, 0.7103, 0.8311), (0.5677, 0.6727, 0.8092)))
    out = "\n".join(seeds.lines([path], [seeds.read(path)]))
    assert "ONE SEED ONLY" in out
    assert "one seed: no scale yet" in out
    assert "±" not in out


# ------------------------------------------------------------------ what must be refused
def test_a_different_train_dump_is_not_the_same_measurement(tmp_path: Path) -> None:
    """The train dump supplies the SET of molecules seen in training. Change it and
    "0 molecules seen" means something else, so the two numbers are not one quantity at
    two seeds.
    """
    first = write(tmp_path / "a.json", report((0.66, 0.71, 0.83), (0.56, 0.67, 0.80)))
    second = write(tmp_path / "b.json", report((0.64, 0.70, 0.82), (0.55, 0.66, 0.80),
                                               train="errors_other_train_dump"))
    found = seeds.problems([(path, seeds.read(path)) for path in (first, second)])
    assert any("train" in problem for problem in found)
    assert seeds.main([str(first), str(second)]) == 1


def test_a_different_slice_or_threshold_is_refused(tmp_path: Path) -> None:
    first = write(tmp_path / "a.json", report((0.66, 0.71, 0.83), (0.56, 0.67, 0.80)))
    for key, value in (("slice", "strict"), ("min_candidates", 3),
                       ("direction_tolerance", 0.5), ("rows", 24000)):
        second = write(tmp_path / "b.json",
                       report((0.64, 0.70, 0.82), (0.55, 0.66, 0.80), **{key: value}))
        found = seeds.problems([(path, seeds.read(path)) for path in (first, second)])
        assert any(key in problem for problem in found), key


def test_a_report_whose_own_check_failed_is_refused(tmp_path: Path) -> None:
    # seen_molecules.py refuses to let such a breakdown be quoted; averaging it would be
    # quoting it three times.
    path = write(tmp_path / "a.json",
                 report((0.66, 0.71, 0.83), (0.56, 0.67, 0.80),
                        check={"checked": True, "agrees": False}))
    found = seeds.problems([(path, seeds.read(path))])
    assert any("must not be quoted" in problem for problem in found)
    assert seeds.main([str(path)]) == 1


def test_a_file_of_another_kind_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "a.json", {"stage": "stage3_train"})
    assert seeds.main([str(path)]) == 1


def test_a_missing_file_exits_two(tmp_path: Path) -> None:
    assert seeds.main([str(tmp_path / "nope.json")]) == 2


def test_three_good_reports_exit_zero(tmp_path: Path) -> None:
    assert seeds.main([str(path) for path in three(tmp_path)]) == 0


# ------------------------------------------------------------------ the overnight batch
SCRIPT = (ROOT / "scripts" / "server_run_overnight_2011.sh").read_text(encoding="utf-8")


def test_the_batch_takes_the_gpu_lock_once() -> None:
    assert SCRIPT.count("claim_gpu") == 1
    assert 'claim_gpu "$GPU" "$ROOT/.cache" "overnight_2011"' in SCRIPT


def test_the_batch_trains_nine_runs_and_scores_each_on_train() -> None:
    """Analysis is minutes; the night is for the training that is still owed.

    Three seeds of SUB100 (the only split with unseen protein clusters in validation, and
    the one that has been run exactly once), and the no-protein arm on both splits -
    shuffling the protein column is not the same measurement as removing the branch,
    because a shuffle keeps the parameters and the input size.
    """
    assert "--mode train" in SCRIPT
    assert SCRIPT.count("train_and_score \"t9") == 3   # one call per loop, three loops
    assert all(f"t9{letter}$((index + 1))" in SCRIPT for letter in "abc")
    # Each training run is followed by the SAME checkpoint scored on the train partition,
    # which is the only way to get the train side AT THE SELECTED checkpoint.
    assert "--mode eval" in SCRIPT and "--partition train" in SCRIPT
    assert "the SAME checkpoint scored on TRAIN" in SCRIPT


def test_the_no_protein_arm_is_the_point_of_two_of_the_three() -> None:
    assert SCRIPT.count("CONFIGURATION=no_protein") == 2
    assert SCRIPT.count("CONFIGURATION=frozen") >= 1
    assert "a shuffle keeps the branch" in SCRIPT


def test_the_training_runs_measure_the_train_side_at_every_point() -> None:
    # Without this the figure has validation beside a loss that is not the same quantity -
    # exactly what was asked about in the meeting.
    assert 'MEASURE="--set measure_train_metrics=true"' in SCRIPT
    assert SCRIPT.count("$MEASURE") == 3


def test_the_training_runs_carry_the_whole_reference_flag_set() -> None:
    """Review 20/11: the first version set only head_block and residual_blocks.

    config.json's own defaults are width 512, dropout 0.1, lr 1e-3, weight decay 0.01,
    while every run these nine must be compared with - t3a, t3c, t5r1-t5r3 - was trained
    at width 102, dropout 0.5, lr 9.86e-05, weight decay 0. Step 4b would then have
    changed the protein branch, the architecture AND the optimiser at once: nine runs
    spent on a number comparable with nothing. Each flag below is checked by value.
    """
    reference = ("--set model.dropout=0.5", "--set learning_rate=9.86e-05",
                 "--set weight_decay=0.0", "--set model.head_block=residual",
                 "--set model.residual_blocks=7", "--set model.residual_width=102",
                 "--set model.residual_hidden_factor=1.849",
                 "--set model.residual_hidden_dropout=0.3936",
                 "--set model.residual_dropout=0.0")
    for flag in reference:
        assert flag in SCRIPT, flag
    assert "SEEDS=(20261023 20261024 20261025)" in SCRIPT
    assert "selection_metric=pairwise_accuracy_macro" in SCRIPT
    assert "selection_slice=strict" in SCRIPT


def test_the_reference_flags_match_the_batch_they_came_from() -> None:
    """Checked against the protein batch's own file, not against this test's memory.

    If that batch is ever retuned, this fails instead of letting the two drift apart
    silently - which is the whole failure mode the review found.
    """
    source = (ROOT / "scripts" / "server_run_protein_seeds_2011.sh").read_text(
        encoding="utf-8")
    for flag in ("model.dropout=0.5", "learning_rate=9.86e-05", "weight_decay=0.0",
                 "model.residual_width=102", "model.residual_hidden_factor=1.849",
                 "model.residual_hidden_dropout=0.3936", "model.residual_dropout=0.0",
                 "model.residual_blocks=7"):
        assert flag in source, f"{flag} is no longer what the reference batch used"
        assert flag in SCRIPT, flag


def test_the_memory_note_does_not_overclaim_the_measured_peak() -> None:
    # MEMORY_LIMIT caps PyTorch's allocator; the CUDA context takes more on top. And the
    # 250 MiB peak was measured at width 102, not at config.json's 512.
    assert "NOT the process" in SCRIPT
    assert "width 102" in SCRIPT


def test_a_dead_training_run_skips_only_its_own_half() -> None:
    assert "produced no usable checkpoint; its train half is skipped" in SCRIPT
    assert "LAST_STATUS=$status" in SCRIPT


def test_the_memory_limit_can_be_lowered_without_editing_the_script() -> None:
    # Every GPU on this machine is 73-81 GB full of other people's jobs, so the cap has to
    # be settable from outside at launch.
    assert "gpu_memory_limit_mib=${MEMORY_LIMIT:-4500}" in SCRIPT


def test_every_step_can_be_skipped_without_killing_the_rest() -> None:
    # A missing split, log or checkpoint must cost its own step and nothing more: the
    # whole point of a sequence left running overnight.
    assert SCRIPT.count("SKIPPED") >= 3
    assert "set -u -o pipefail" in SCRIPT
    assert "set -e" not in SCRIPT


def test_the_steps_run_in_the_stated_order() -> None:
    # Analysis before training on purpose: it is minutes, it answers questions already
    # asked, and a training run that fails must not cost the cheap answers.
    for first, second in (("step 1", "step 2"), ("step 2", "step 3"),
                          ("step 3", "step 4a"), ("step 4c", "step 5")):
        assert SCRIPT.index(f'say "{first}') < SCRIPT.index(f'say "{second}')


def test_the_seed_sweep_uses_its_own_run_ids() -> None:
    # Dumps are found by the run id this script minted, so an older dump cannot be
    # analysed by mistake.
    assert '--run-id "t7s$index-$$"' in SCRIPT
    assert 'errors_*_validation_*t7s$index-$$' in SCRIPT


def test_the_train_dump_is_reused_and_said_to_be() -> None:
    assert "it only" in SCRIPT and "SET OF MOLECULES" in SCRIPT
    assert "PAIR_TRAIN_DUMP" in SCRIPT


def test_the_sub100_step_passes_the_subset() -> None:
    assert "inputs.train_subset=$SUB_SPLIT" in SCRIPT
    assert "unseen protein clusters" in SCRIPT


def test_the_batch_timestamps_its_steps() -> None:
    # The classical batch had no timestamps and its elapsed time had to be asked of `ps`.
    assert 'echo "=== $(date +%H:%M:%S) $* ==="' in SCRIPT


def test_the_batch_says_how_to_read_its_log() -> None:
    assert "FINISHED with $FAILED failed command(s)" in SCRIPT
    assert "step 2 summary" in SCRIPT


def test_a_count_that_never_moves_is_printed_once(tmp_path: Path) -> None:
    # The validation rows do not change with the seed, so "6925 +-0 [6925..6925]" is
    # noise around one number and reads as if it had varied.
    paths = three(tmp_path)
    out = "\n".join(seeds.lines(paths, [seeds.read(path) for path in paths]))
    assert "6925 (same at every seed)" in out
    assert "215 (same at every seed)" in out

# --------------------------------- chứng minh là ba lần chạy khác nhau (review 20/11)
def test_the_same_report_twice_is_refused(tmp_path: Path) -> None:
    """Review 20/11 đưa cùng một báo cáo hai lần và bản cũ chấp nhận.

    Độ lệch chuẩn của ba bản sao của một số là 0, và sd = 0 sẽ bị đọc thành "kết quả cực
    kỳ vững" - đúng ngược với sự thật là chưa đo gì cả.
    """
    path = three(tmp_path)[0]
    assert seeds.main([str(path), str(path)]) == 1
    found = seeds.problems([(path, seeds.read(path)), (path, seeds.read(path))])
    assert any("passed more than once" in problem for problem in found)


def test_two_reports_of_the_same_run_are_refused(tmp_path: Path) -> None:
    # Cùng một lần chạy chấm điểm hai lần không phải hai seed.
    first = write(tmp_path / "a.json", report((0.66, 0.71, 0.83), (0.56, 0.67, 0.80)))
    second = write(tmp_path / "b.json", report((0.64, 0.70, 0.82), (0.55, 0.66, 0.80)))
    found = seeds.problems([(first, seeds.read(first)), (second, seeds.read(second))])
    assert any("SAME run" in problem for problem in found)
    assert seeds.main([str(first), str(second)]) == 1


def test_two_checkpoints_of_the_same_seed_are_refused(tmp_path: Path) -> None:
    """Review 20/11, lượt hai: hai checkpoint khác TÊN mà cùng seed đã được chấp nhận.

    Một lần chạy lại, hay một bản copy, không phải một replicate: độ lệch chuẩn qua chúng
    đo nhiễu của phép lưu file, không phải nhiễu của seed.
    """
    first = write(tmp_path / "a.json",
                  report((0.66, 0.71, 0.83), (0.56, 0.67, 0.80), **mlp_changes(1)))
    second = write(tmp_path / "b.json",
                   report((0.64, 0.70, 0.82), (0.55, 0.66, 0.80),
                          validation_checkpoint="runs/evaluator/best_rerun.pt",
                          identity="checkpoint:best_rerun.pt",
                          validation_seed=20261023))
    assert seeds.read(first)["validation_seed"] == seeds.read(second)["validation_seed"]
    assert seeds.read(first)["identity"] != seeds.read(second)["identity"]
    found = seeds.problems([(first, seeds.read(first)), (second, seeds.read(second))])
    assert any("SAME SEED" in problem for problem in found)
    assert seeds.main([str(first), str(second)]) == 1


def test_an_unreadable_seed_is_reported_not_assumed(tmp_path: Path) -> None:
    first = write(tmp_path / "a.json",
                  report((0.66, 0.71, 0.83), (0.56, 0.67, 0.80), **mlp_changes(1)))
    second = write(tmp_path / "b.json",
                   report((0.64, 0.70, 0.82), (0.55, 0.66, 0.80),
                          validation_checkpoint="runs/evaluator/best_x.pt",
                          identity="checkpoint:best_x.pt", validation_seed=None))
    found = seeds.problems([(first, seeds.read(first)), (second, seeds.read(second))])
    assert any("WITHOUT that check" in problem for problem in found)


def test_different_validation_rows_are_refused_even_at_the_same_count(
        tmp_path: Path) -> None:
    """Số dòng bằng nhau KHÔNG chứng minh cùng một tập dòng.

    Đây là chỗ bản cũ chỉ so `rows`, nên hai lát của hai split khác nhau mà tình cờ cùng
    số dòng sẽ được gộp.
    """
    first = write(tmp_path / "a.json",
                  report((0.66, 0.71, 0.83), (0.56, 0.67, 0.80),
                         validation_checkpoint="runs/evaluator/best_a.pt"))
    second = write(tmp_path / "b.json",
                   report((0.64, 0.70, 0.82), (0.55, 0.66, 0.80),
                          validation_checkpoint="runs/evaluator/best_b.pt",
                          validation_fingerprint="ffffffff" * 4))
    assert seeds.read(first)["rows"] == seeds.read(second)["rows"]
    found = seeds.problems([(first, seeds.read(first)), (second, seeds.read(second))])
    assert any("DIFFERENT SET of validation rows" in problem for problem in found)


def test_reports_that_do_not_name_their_run_are_not_silently_averaged(
        tmp_path: Path) -> None:
    # Báo cáo viết trước 20/11 không có `identity`; phải nói là KHÔNG kiểm được, chứ
    # không được coi như đã kiểm.
    first = write(tmp_path / "a.json",
                  report((0.66, 0.71, 0.83), (0.56, 0.67, 0.80), identity=""))
    second = write(tmp_path / "b.json",
                   report((0.64, 0.70, 0.82), (0.55, 0.66, 0.80), identity=""))
    found = seeds.problems([(first, seeds.read(first)), (second, seeds.read(second))])
    assert any("do not say which run produced them" in problem for problem in found)


# ------------------------------------- so paired voi baseline thay cho sd cua MLP
def baseline_report(**changes: Any) -> dict[str, Any]:
    """Giống dump cổ điển THẬT: KHÔNG có checkpoint, không có seed theo nghĩa replicate.

    Review 20/11 lượt hai bắt đúng chỗ này: fixture cũ tự gán
    `validation_checkpoint="...classical_lasso.manifest"`, nên test xanh trong khi CLI
    thật từ chối mọi dump lasso thật vì "thiếu checkpoint". `classical_row_dump.py` ghi
    `source_manifest`, `model`, `alpha`, `label` - và không ghi `.pt` nào.
    """
    payload = report((0.5831, 0.6464, 0.7164), (0.5527, 0.6301, 0.7569),
                     label="SUB100 lasso")
    payload.update({"validation_checkpoint": "",
                    "validation_model": "lasso",
                    "validation_source_manifest":
                        "runs/evaluator/classical_frozen_20261120_120000_t1b-3831957.json",
                    "validation_alpha": 0.001,
                    "identity": "classical:classical_frozen_20261120_120000_t1b-"
                                "3831957.json#lasso_alpha0.001@0.001",
                    "kind": "classical",
                    "validation_seed": None})
    payload.update(changes)
    return payload


def test_a_real_classical_baseline_has_no_checkpoint_and_is_still_accepted(
        tmp_path: Path) -> None:
    """Chính ca review đưa ra: dump lasso thật không có `.pt`.

    CLI cũ áp quy tắc "hai checkpoint khác nhau" cho cả baseline rồi từ chối vì thiếu
    checkpoint, nên bảng hiệu MLP − Lasso không bao giờ in ra được.
    """
    paths = three(tmp_path)
    base = write(tmp_path / "lasso.json", baseline_report())
    assert seeds.read(base)["validation_checkpoint"] == ""
    assert seeds.read(base)["validation_seed"] is None
    assert seeds.main([*(str(path) for path in paths), "--baseline", str(base)]) == 0


def test_a_baseline_that_does_not_say_which_fit_it_is_refused(tmp_path: Path) -> None:
    paths = three(tmp_path)
    base = write(tmp_path / "lasso.json", baseline_report(identity=""))
    assert seeds.main([*(str(path) for path in paths), "--baseline", str(base)]) == 1


def test_the_paired_table_names_the_fit(tmp_path: Path) -> None:
    paths = three(tmp_path)
    base = write(tmp_path / "lasso.json", baseline_report())
    out = "\n".join(seeds.lines(paths, [seeds.read(path) for path in paths],
                                (base, seeds.read(base))))
    assert "lasso_alpha0.001@0.001" in out


def test_the_paired_differences_are_printed_per_seed(tmp_path: Path) -> None:
    """Điểm chính của review: sd của riêng MLP không phải ngưỡng để nói MLP thắng lasso.

    Đại lượng mang kết luận là HIỆU trên cùng các dòng, mỗi seed một lần, và việc cả ba
    hiệu có cùng dấu hay không.
    """
    paths = three(tmp_path)
    base = write(tmp_path / "lasso.json", baseline_report())
    out = "\n".join(seeds.lines(paths, [seeds.read(path) for path in paths],
                                (base, seeds.read(base))))
    assert "Paired against lasso.json" in out
    assert "MLP - baseline, per seed" in out
    assert "all one sign" in out
    # Nhóm 0, macro: 0.6673/0.6421/0.6890 so với 0.5831 -> cả ba đều dương.
    assert "+0.0842" in out and "+0.0590" in out and "+0.1059" in out
    assert "| 0 | macro |" in out


def test_a_mean_difference_with_mixed_signs_is_called_out(tmp_path: Path) -> None:
    paths = three(tmp_path)
    # Baseline cao hơn hai trong ba seed ở nhóm 0 -> dấu không đồng nhất.
    base = write(tmp_path / "lasso.json", baseline_report())
    payload = seeds.read(base)
    for block in payload["by_seen_macro"]:
        if block["molecules_seen"] == 0:
            block["pairwise_accuracy_macro"] = 0.6700
    out = "\n".join(seeds.lines(paths, [seeds.read(path) for path in paths],
                                (base, payload)))
    assert "| NO |" in out
    assert "whose per-seed signs disagree is not a result" in out


def test_a_baseline_on_other_rows_is_refused(tmp_path: Path) -> None:
    paths = three(tmp_path)
    base = write(tmp_path / "lasso.json",
                 baseline_report(validation_fingerprint="ffffffff" * 4))
    assert seeds.main([*(str(path) for path in paths), "--baseline", str(base)]) == 1


def test_a_good_baseline_exits_zero(tmp_path: Path) -> None:
    paths = three(tmp_path)
    base = write(tmp_path / "lasso.json", baseline_report())
    assert seeds.main([*(str(path) for path in paths), "--baseline", str(base)]) == 0


def test_no_p_value_is_claimed_for_three_differences(tmp_path: Path) -> None:
    paths = three(tmp_path)
    base = write(tmp_path / "lasso.json", baseline_report())
    out = "\n".join(seeds.lines(paths, [seeds.read(path) for path in paths],
                                (base, seeds.read(base))))
    assert "not a confidence interval and not a significance test" in out.lower()

def test_the_batch_asks_for_the_paired_table(tmp_path: Path) -> None:
    """Review 20/11 lượt hai: batch gọi bộ tổng hợp mà KHÔNG truyền --baseline, nên sáng
    mai sẽ không có bảng hiệu MLP − Lasso dù code đã có."""
    assert "--baseline" in SCRIPT
    assert "BASELINE_REPORT" in SCRIPT
    # Và thiếu nó thì nói ra, chứ không im lặng bỏ bảng.
    assert "no BASELINE_REPORT passed" in SCRIPT
