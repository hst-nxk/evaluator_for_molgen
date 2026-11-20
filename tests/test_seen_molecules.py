"""Is the PAIR75 score carried by rows whose molecules were already in training?

Why these tests exist. The breakdown is only evidence if it decomposes the number the run
itself recorded, so the script rebuilds the queries from the row dump and checks the result
against the dump's own `queries.parquet`. These tests cover that check in both directions
(agreeing and disagreeing), the two units the report is written in, and the one claim the
report is NOT allowed to make - that a higher score on seen molecules means memorisation,
which is why each row group carries its label statistics.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "seen_molecules.py"
SPEC = importlib.util.spec_from_file_location("seen_molecules", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
seen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seen)

TOLERANCE = 0.3


def dump(folder: Path, rows: list[dict], *, queries: pa.Table | None = None,
         tolerance: float = TOLERANCE) -> Path:
    """A folder in the shape `--mode errors` writes: rows.parquet (+ queries.parquet)."""
    folder.mkdir(parents=True, exist_ok=True)
    truth = np.array([row["delta_p"] for row in rows], dtype=np.float64)
    prediction = np.array([row["prediction"] for row in rows], dtype=np.float64)
    counted = np.abs(truth) > tolerance
    correct = counted & (np.sign(truth) == np.sign(prediction))
    pq.write_table(pa.table({
        "pair_id": pa.array([index for index, _ in enumerate(rows)], pa.int64()),
        "endpoint": pa.array([row.get("endpoint", "IC50") for row in rows]),
        "protein_hash": pa.array([row.get("protein", "P1") for row in rows]),
        "smiles_a": pa.array([row["a"] for row in rows]),
        "smiles_b": pa.array([row["b"] for row in rows]),
        "unseen_pair": pa.array([row.get("unseen_pair", True) for row in rows]),
        "delta_p": pa.array(truth),
        "prediction": pa.array(prediction),
        "error": pa.array(prediction - truth),
        "sign_counted": pa.array(counted),
        "sign_correct": pa.array(correct),
        "protein_rows": pa.array([len(rows)] * len(rows), pa.int64()),
    }), folder / "rows.parquet")
    # A real report.json as `--mode errors` writes it: the metrics block under
    # "metrics_settings", the flat direction_tolerance beside it, and "settings" holding
    # the stage3.ANALYSIS block - which is NOT where a metric threshold lives.
    (folder / "report.json").write_text(
        json.dumps({"direction_tolerance": tolerance,
                    "metrics_settings": {"direction_tolerance": tolerance,
                                         "min_candidates": 2},
                    "settings": {"min_group_queries": 25, "min_group_rows": 200}}),
        encoding="utf-8")
    if queries is not None:
        pq.write_table(queries, folder / "queries.parquet")
    return folder


# Two molecules are in train (M1, M2); M9 and M8 are not. The validation rows then cover
# all three groups: both seen, one seen, neither seen.
TRAIN_ROWS = [{"a": "M1", "b": "M2", "delta_p": 1.0, "prediction": 1.0}]
VALIDATION_ROWS = [
    {"a": "M1", "b": "M2", "delta_p": 1.0, "prediction": 0.9},    # 2 seen, right sign
    {"a": "M1", "b": "M9", "delta_p": 1.0, "prediction": 0.8},    # 1 seen, right sign
    {"a": "M9", "b": "M8", "delta_p": 1.0, "prediction": -0.7},   # 0 seen, wrong sign
    {"a": "M8", "b": "M9", "delta_p": 0.9, "prediction": -0.6},   # 0 seen, wrong sign
]


@pytest.fixture()
def dumps(tmp_path: Path) -> tuple[Path, Path]:
    train = dump(tmp_path / "errors_train", TRAIN_ROWS)
    validation = dump(tmp_path / "errors_validation", VALIDATION_ROWS)
    return train, validation


def report_of(dumps: tuple[Path, Path], slice_name: str = "all") -> dict:
    return seen.build(dumps[0], dumps[1], slice_name, "test")


def test_the_rows_are_split_by_how_many_molecules_were_seen(dumps) -> None:
    blocks = {block["molecules_seen"]: block for block in report_of(dumps)["by_row"]}
    assert blocks[2]["rows"] == 1 and blocks[1]["rows"] == 1 and blocks[0]["rows"] == 2
    assert blocks[2]["share"] == pytest.approx(0.25)
    # The seen rows were predicted with the right sign, the unseen ones with the wrong
    # one, which is the shape the hypothesis predicts and the table has to show.
    assert blocks[2]["direction_accuracy_micro"] == pytest.approx(1.0)
    assert blocks[0]["direction_accuracy_micro"] == pytest.approx(0.0)
    assert blocks[2]["mae"] == pytest.approx(0.1)


def test_each_row_group_carries_its_label_statistics(dumps) -> None:
    # Without these, "the model remembers these rows" cannot be told from "these rows are
    # easier", and the report would invite the first reading by omission.
    for block in report_of(dumps)["by_row"]:
        if block["rows"]:
            assert block["label_mean"] is not None
            assert block["label_sd"] is not None
            assert block["abs_label_mean"] is not None


def test_the_queries_are_rebuilt_the_way_the_metric_builds_them(dumps) -> None:
    # Four pairs, each doubled: references are M1, M2, M1, M9, M9, M8, M8, M9 and a query
    # is (reference, protein, endpoint) - so M1, M2, M9, M8 under one protein and one
    # endpoint: four queries.
    report = report_of(dumps)
    assert report["overall"]["queries"] == 4
    assert report["direction_tolerance"] == TOLERANCE


def test_a_query_group_knows_whether_its_reference_was_seen(dumps) -> None:
    groups = {(block["reference_seen"], block["candidates"]): block
              for block in report_of(dumps)["by_query"]}
    # M1's query: reference seen, candidates M2 (seen) and M9 (unseen).
    assert (True, "some candidates seen") in groups
    # M9's query: reference unseen; its candidates are M1 (seen) and M8 (unseen).
    assert any(not reference_seen for reference_seen, _ in groups)
    for block in groups.values():
        assert block["queries_ranked"] <= block["queries"]


def test_the_reconstruction_is_checked_against_the_runs_own_queries(tmp_path: Path
                                                                   ) -> None:
    train = dump(tmp_path / "t", TRAIN_ROWS)
    validation = dump(tmp_path / "v", VALIDATION_ROWS)
    first = seen.build(train, validation, "all", "")
    # No queries.parquet yet: the script must say the check did not happen rather than
    # imply it passed.
    assert first["check"]["checked"] is False
    assert "could not" in first["check"]["note"]
    assert seen.main(["--train", str(train), "--validation", str(validation)]) == 1


def test_a_disagreeing_queries_file_refuses_the_breakdown(tmp_path: Path,
                                                          capsys) -> None:
    train = dump(tmp_path / "t", TRAIN_ROWS)
    validation = dump(tmp_path / "v", VALIDATION_ROWS)
    # A file that records five queries where three clear the bar: the breakdown does not
    # decompose the recorded number, so it may not be quoted.
    wrong = pa.table({"reference_smiles": pa.array(["X"] * 5),
                      "protein_hash": pa.array(["P1"] * 5),
                      "endpoint": pa.array(["IC50"] * 5),
                      "candidates": pa.array([1] * 5, pa.int64()),
                      "agree": pa.array([1] * 5, pa.int64()),
                      "counted": pa.array([1] * 5, pa.int64()),
                      "spearman": pa.array([None] * 5, pa.float64()),
                      "top_1": pa.array([1] * 5, pa.int64())})
    pq.write_table(wrong, validation / "queries.parquet")
    report = seen.build(train, validation, "all", "")
    assert report["check"]["checked"] is True
    assert report["check"]["agrees"] is False
    assert report["check"]["queries_recorded"] == 5
    assert report["check"]["queries_rebuilt"] == 3
    assert seen.main(["--train", str(train), "--validation", str(validation)]) == 1
    out = capsys.readouterr().out
    assert "RECONSTRUCTION DISAGREES" in out
    assert "must not be quoted" in out


def test_an_agreeing_queries_file_passes_and_exits_zero(tmp_path: Path, capsys) -> None:
    train = dump(tmp_path / "t", TRAIN_ROWS)
    validation = dump(tmp_path / "v", VALIDATION_ROWS)
    report = seen.build(train, validation, "all", "")
    # Build a queries.parquet from the rebuilt queries themselves: same count, same macro.
    agree, counted = [], []
    for block in report["by_query"]:
        value = block["pairwise_accuracy_macro"]
        for _ in range(block["queries_ranked"]):
            agree.append(int(round((value or 0.0) * 2)))
            counted.append(2)
        # Only the queries the evaluator would have written down: those with at least
        # min_candidates candidates. A singleton query is rebuilt here but never written.
        for _ in range(block["queries_eligible"] - block["queries_ranked"]):
            agree.append(0)
            counted.append(0)
    pq.write_table(pa.table({"reference_smiles": pa.array(["X"] * len(agree)),
                             "protein_hash": pa.array(["P1"] * len(agree)),
                             "endpoint": pa.array(["IC50"] * len(agree)),
                             "candidates": pa.array([2] * len(agree), pa.int64()),
                             "agree": pa.array(agree, pa.int64()),
                             "counted": pa.array(counted, pa.int64()),
                             "spearman": pa.array([None] * len(agree), pa.float64()),
                             "top_1": pa.array([1] * len(agree), pa.int64())}),
                   validation / "queries.parquet")
    assert seen.main(["--train", str(train), "--validation", str(validation)]) == 0
    out = capsys.readouterr().out
    assert f"Reconstruction agrees with {seen.QUERIES_FILE}" in out


def test_singleton_queries_do_not_make_a_correct_run_fail(tmp_path: Path) -> None:
    """Review 2026-11-20: `analysis.query_table` writes down only the queries with at
    least `min_candidates` candidates, so queries.parquet is the rankable part of the
    query set, not all of it. The check compared raw totals and would have refused a
    perfectly correct pair of GPU runs AFTER both had been paid for.

    Two rows A-B and A-C: three queries are rebuilt (A with two candidates, B and C with
    one each), the evaluator writes ONE (A), and both sides agree on the macro.
    """
    train = dump(tmp_path / "t", [{"a": "A", "b": "B", "delta_p": 1.0,
                                   "prediction": 1.0}])
    validation = dump(tmp_path / "v", [
        {"a": "A", "b": "B", "delta_p": 1.0, "prediction": 0.9},
        {"a": "A", "b": "C", "delta_p": -1.0, "prediction": -0.8}])
    pq.write_table(pa.table({"reference_smiles": pa.array(["A"]),
                             "protein_hash": pa.array(["P1"]),
                             "endpoint": pa.array(["IC50"]),
                             "candidates": pa.array([2], pa.int64()),
                             "agree": pa.array([1], pa.int64()),
                             "counted": pa.array([1], pa.int64()),
                             "spearman": pa.array([1.0], pa.float64()),
                             "top_1": pa.array([1], pa.int64())}),
                   validation / "queries.parquet")
    report = seen.build(train, validation, "all", "")
    assert report["min_candidates"] == 2
    assert report["overall"]["queries"] == 3            # A, B and C
    assert report["overall"]["queries_eligible"] == 1   # only A clears the bar
    assert report["overall"]["queries_ranked"] == 1
    assert report["overall"]["pairwise_accuracy_macro"] == pytest.approx(1.0)
    assert report["check"]["agrees"] is True
    assert report["check"]["queries_rebuilt_including_singletons"] == 3
    assert seen.main(["--train", str(train), "--validation", str(validation)]) == 0


def test_the_group_table_separates_coverage_from_the_rankable_part(dumps) -> None:
    # The reviewer's second ask: keep the full total beside the comparable one, so the
    # coverage of each group is visible rather than implied.
    report = report_of(dumps)
    text = "\n".join(seen.lines(report))
    assert f"with >= {report['min_candidates']} candidates" in text
    assert "rebuilt, of which" in text
    for block in report["by_seen_macro"]:
        assert block["queries_eligible"] <= block["queries"]
        assert block["queries_ranked"] <= block["queries_eligible"]


def test_min_candidates_comes_from_the_dump_not_from_this_script(tmp_path: Path) -> None:
    train = dump(tmp_path / "t", TRAIN_ROWS)
    validation = dump(tmp_path / "v", VALIDATION_ROWS)
    (validation / "report.json").write_text(
        json.dumps({"metrics_settings": {"direction_tolerance": 0.3,
                                         "min_candidates": 3}}), encoding="utf-8")
    report = seen.build(train, validation, "all", "")
    assert report["min_candidates"] == 3
    assert report["min_candidates_source"] == "report.json"
    # Nothing in report.json: config.json is the fallback, the project's value is 2, and
    # the report SAYS which of the two it used - the dump of a past run was written with
    # whatever config.json held then, not with what it holds today.
    (validation / "report.json").write_text("{}", encoding="utf-8")
    report = seen.build(train, validation, "all", "")
    assert report["min_candidates"] == 2
    assert report["min_candidates_source"] == "config.json"
    assert "read from config.json" in "@".join(seen.lines(report))


def test_the_analysis_block_is_not_mistaken_for_the_metrics_block(tmp_path: Path) -> None:
    """Review 2026-11-20: report.json["settings"] is config["stage3"]["analysis"]
    (train.py), so a metric threshold must not be read out of it. It holds
    min_group_queries and min_group_rows - different settings that could be picked up by
    name. A metrics value placed there is ignored and the fallback is used instead.
    """
    train = dump(tmp_path / "t", TRAIN_ROWS)
    validation = dump(tmp_path / "v", VALIDATION_ROWS)
    (validation / "report.json").write_text(
        json.dumps({"settings": {"min_candidates": 9, "direction_tolerance": 1.5}}),
        encoding="utf-8")
    report = seen.build(train, validation, "all", "")
    assert report["min_candidates"] == 2
    assert report["min_candidates_source"] == "config.json"
    assert report["direction_tolerance"] == 0.3


def test_a_query_below_the_bar_is_skipped_and_not_merely_uncounted(tmp_path: Path
                                                                   ) -> None:
    """Review 2026-11-20: with min_candidates = 3 the macro still scored the queries
    below the bar, so eligible was 0 while ranked was 1 and the macro was 1.0.
    `metrics.ranking_report` SKIPS such a query before scoring anything
    (queries_skipped_too_few_candidates), so this macro has to skip it too or it is a
    different quantity from the one the manifests report. At 2 the two behaviours agree,
    which is why it took a threshold of 3 to show.
    """
    train = dump(tmp_path / "t", [{"a": "A", "b": "B", "delta_p": 1.0,
                                   "prediction": 1.0}])
    validation = dump(tmp_path / "v", [
        {"a": "A", "b": "B", "delta_p": 1.0, "prediction": 0.9},
        {"a": "A", "b": "C", "delta_p": -1.0, "prediction": -0.8}])
    (validation / "report.json").write_text(
        json.dumps({"metrics_settings": {"direction_tolerance": 0.3,
                                         "min_candidates": 3}}), encoding="utf-8")
    report = seen.build(train, validation, "all", "")
    assert report["min_candidates"] == 3
    overall = report["overall"]
    assert overall["queries"] == 3           # coverage: A, B and C were rebuilt
    assert overall["queries_eligible"] == 0  # none of them has three candidates
    assert overall["queries_ranked"] == 0    # so none of them is scored
    assert overall["pairwise_accuracy_macro"] is None
    for block in report["by_seen_macro"] + report["by_query"]:
        assert block["queries_ranked"] <= block["queries_eligible"]
        assert block["comparisons"] == 0


def test_the_tolerance_comes_from_the_dump_not_from_this_script(tmp_path: Path) -> None:
    train = dump(tmp_path / "t", TRAIN_ROWS, tolerance=0.9)
    validation = dump(tmp_path / "v", VALIDATION_ROWS, tolerance=0.9)
    assert seen.build(train, validation, "all", "")["direction_tolerance"] == 0.9
    # With the field gone, config.json is the fallback and the project's value is 0.3.
    (validation / "report.json").write_text("{}", encoding="utf-8")
    assert seen.build(train, validation, "all", "")["direction_tolerance"] == 0.3


def test_an_old_dump_without_the_smiles_columns_is_refused(tmp_path: Path) -> None:
    train = dump(tmp_path / "t", TRAIN_ROWS)
    validation = dump(tmp_path / "v", VALIDATION_ROWS)
    table = pq.read_table(validation / "rows.parquet").drop(["smiles_a"])
    pq.write_table(table, validation / "rows.parquet")
    with pytest.raises(ValueError, match="2026-11-10"):
        seen.build(train, validation, "all", "")


def test_the_text_report_states_which_unit_each_table_is_in(dumps) -> None:
    text = "\n".join(seen.lines(report_of(dumps)))
    assert "Per ROW (one MMP pair)" in text
    assert "MICRO quantity" in text
    assert "pairwise_accuracy_macro PER GROUP, the groups taken as slices" in text
    assert "Per QUERY of the WHOLE set" in text
    assert "placed by which SIDE was seen" in text


def test_the_macro_per_group_is_measured_the_way_a_slice_is(dumps) -> None:
    """What the 2026-11-20 review asked for: pairwise_accuracy_macro per 0/1/2 group,
    with the rows and the ranked queries of each group. The rows are selected first and
    the queries built afterwards, exactly as train.evaluate measures every slice.
    """
    blocks = {block["molecules_seen"]: block
              for block in report_of(dumps)["by_seen_macro"]}
    assert set(blocks) == {0, 1, 2}
    assert blocks[0]["rows"] == 2 and blocks[1]["rows"] == 1 and blocks[2]["rows"] == 1
    for block in blocks.values():
        assert block["queries_ranked"] <= block["queries"]
        assert "comparisons" in block
    # Group 0 is the two rows between M9 and M8, both predicted with the wrong sign: one
    # query of two candidates, one comparison, and it disagrees.
    assert blocks[0]["queries_ranked"] == 2
    assert blocks[0]["pairwise_accuracy_macro"] == pytest.approx(0.0)
    # Groups 1 and 2 hold a single pair each: one comparison needs two candidates of one
    # reference, so there is nothing to rank and the macro is undefined rather than 0.
    assert blocks[2]["pairwise_accuracy_macro"] is None
    assert blocks[2]["queries_ranked"] == 0


def test_the_groups_are_not_claimed_to_add_up_to_the_whole(dumps) -> None:
    # Restricting the rows cuts a query into smaller queries, so the three group macros
    # do not decompose the overall macro - and the report says so instead of inviting the
    # reader to average them.
    text = "\n".join(seen.lines(report_of(dumps)))
    assert "do not add up to the number of the whole set" in text
    assert "Compare the groups with each other, not with the overall macro" in text


def test_an_empty_group_is_undefined_not_zero(tmp_path: Path) -> None:
    # Every molecule seen: groups 0 and 1 hold no rows at all.
    train = dump(tmp_path / "t", [{"a": "M1", "b": "M2", "delta_p": 1.0,
                                   "prediction": 1.0}])
    validation = dump(tmp_path / "v", [{"a": "M1", "b": "M2", "delta_p": 1.0,
                                        "prediction": 0.9}])
    blocks = {block["molecules_seen"]: block
              for block in seen.build(train, validation, "all", "")["by_seen_macro"]}
    assert blocks[0]["rows"] == 0 and blocks[0]["pairwise_accuracy_macro"] is None
    assert blocks[0]["queries"] == 0
    assert blocks[2]["rows"] == 1


# ------------------------------------------------------------------ the batch script
SCRIPT = (ROOT / "scripts" / "server_run_seen_molecules_2011.sh").read_text(
    encoding="utf-8")


def test_both_dumps_are_given_the_same_split() -> None:
    # "Seen in train" is only defined against the train set the checkpoint was trained
    # on, so a dump taken under a different split would answer another question.
    assert "$SETTINGS" in SCRIPT
    assert "--set inputs.partition_override_pairs=$PAIR_SPLIT" in SCRIPT
    # One command, inside the loop over the two partitions - not two commands that could
    # drift apart.
    commands = [line for line in SCRIPT.splitlines()
                if line.strip().startswith("run python run_evaluator.py --mode errors")]
    assert len(commands) == 1, commands
    assert "for partition in train validation; do" in SCRIPT


def test_the_protein_shuffle_is_never_passed_here() -> None:
    # A t3b checkpoint records protein_permute_seed >= 0 and --mode errors refuses a
    # command that does not match it, so leaving the flag out is what keeps a shuffled
    # checkpoint from being analysed as if it were the real one.
    assert "--set protein_permute_seed" not in SCRIPT
    assert "$SETTINGS" in SCRIPT and "protein_permute_seed" not in SCRIPT.split(
        "SETTINGS=")[1].split("say()")[0]


def test_the_dumps_are_found_by_this_runs_own_id() -> None:
    # Globbing errors_*_train_* alone would pick up a dump from an older run.
    assert 'errors_*_train_*t6train-$$' in SCRIPT
    assert 'errors_*_validation_*t6validation-$$' in SCRIPT
    assert '--run-id "t6$partition-$$"' in SCRIPT


def test_a_failed_dump_stops_before_the_analysis() -> None:
    first = SCRIPT.index("no analysis was run")
    assert first < SCRIPT.index("analysis - molecules seen in training")


def test_the_analysis_refusing_itself_fails_the_batch() -> None:
    # seen_molecules.py exits 1 when its reconstruction does not reproduce
    # queries.parquet; the batch has to carry that through instead of printing DONE.
    assert "THE ANALYSIS REFUSED ITS OWN OUTPUT" in SCRIPT
    assert SCRIPT.rindex("exit 1") < SCRIPT.index('say "DONE"')


def test_the_dump_is_made_to_carry_the_metrics_block() -> None:
    """The consumer side above falls back to config.json, which is a guess about a past
    run. So the writer was changed to record the block: new dumps say what they used.
    (Checked at the source, because writing one needs a GPU and the real data.)
    """
    source = (ROOT / "src" / "molopt" / "evaluator" / "train.py").read_text(
        encoding="utf-8")
    assert '"metrics_settings": {key: value for key, value in metric_settings.items()'         in source
    # And it is the metrics block, not the analysis block that sits beside it.
    assert '"settings": dict(settings),' in source


def test_the_picker_is_called_with_its_real_interface() -> None:
    """Review 2026-11-20: the header (and the command sent in chat) used
    `--task t3a --pid 4174688`, which the picker refuses - it takes two positional
    arguments. The wrong form must not survive anywhere in the script.
    """
    # No line that CALLS the picker may carry those flags. (They still appear in the
    # comment that records the mistake, which is where they belong.)
    calls = [line for line in SCRIPT.splitlines()
             if "pick_train_checkpoint.py" in line and not line.lstrip().startswith("#")]
    assert calls, SCRIPT
    for line in calls:
        assert "--task" not in line and "--pid" not in line, line
    assert 'python scripts/pick_train_checkpoint.py runs/evaluator "$TARGET"' in SCRIPT
    assert "python scripts/pick_train_checkpoint.py runs/evaluator t3a-4174688" in SCRIPT
    picker = (ROOT / "scripts" / "pick_train_checkpoint.py").read_text(encoding="utf-8")
    assert "usage: python scripts/pick_train_checkpoint.py <runs_dir> <run id>" in picker


def test_a_run_id_or_a_checkpoint_path_are_both_accepted() -> None:
    assert '[[ "$TARGET" == *.pt ]]' in SCRIPT
    assert 'CHECKPOINT="$TARGET"' in SCRIPT


def test_the_batch_takes_the_gpu_lock() -> None:
    assert 'source "$ROOT/scripts/gpu_batch_lock.sh"' in SCRIPT
    assert 'claim_gpu "$GPU" "$ROOT/.cache" "seen_molecules_2011"' in SCRIPT
