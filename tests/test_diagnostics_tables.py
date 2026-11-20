"""The tables are read out of manifests, never filled in by hand.

Why these tests exist. Carrying numbers from a manifest into a report was the one step of
this pipeline with nothing behind it: a row quoted at an alpha nobody selected, a `strict`
column holding `all` numbers, a missing task silently filled in from another batch. Each
test below is one of those ways of being wrong.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics_tables.py"
SPEC = importlib.util.spec_from_file_location("diagnostics_tables", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
tables = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tables)

PID = 4174688


def scores(mae: float, r2: float, pairwise: float, rows: int = 100,
           ranked: int = 50, total: int = 100, counted: int | None = None) -> dict:
    """One slice of a manifest. `counted` is `pairwise_accuracy_queries`, the macro's own
    denominator, which the metric records from 2026-11-20 on; None leaves it out, the way
    an older manifest is."""
    ranking = {"pairwise_accuracy_macro": pairwise,
               "pairwise_accuracy_micro": pairwise,
               "queries_ranked": ranked, "queries_total": total}
    if counted is not None:
        ranking["pairwise_accuracy_queries"] = counted
    return {"regression": {"n": rows, "mae": mae, "r2": r2, "spearman": 0.4},
            "ranking": ranking}


def empty_scores() -> dict:
    """What `train.evaluate` writes for a slice with no rows in it.

    This is not a hypothetical: on a pair-keyed split `build_pair_split.py` writes
    `unseen_pair = is_validation`, so the `strict` slice of the TRAIN partition is empty
    by construction and every number in it is None (metrics.regression_metrics returns
    UNDEFINED for an empty array). A fixture that gives train/strict real numbers hides
    the bug this file is meant to catch.
    """
    return {"rows": 0,
            "regression": {"n": 0, "mae": None, "rmse": None, "r2": None,
                           "spearman": None},
            "ranking": {"pairwise_accuracy_macro": None,
                        "pairwise_accuracy_micro": None,
                        "queries_ranked": 0, "queries_total": 0}}


def fit(model: str, alpha: float, pairwise: float, mae: float = 0.6, r2: float = 0.01,
        nonzero: int | None = None, cap: bool | None = None) -> dict:
    block = {"model": model, "alpha": alpha, "fit_seconds": 1.0,
             "coefficients": None if nonzero is None else
             {"total": 6628, "nonzero": nonzero, "constant_predictor": nonzero == 0},
             "convergence": None if cap is None else
             {"iterations": 2000, "max_iter": 2000, "hit_the_cap": cap},
             # Train at `all` only; `strict` is present and empty, as on PAIR75.
             "train_metrics": {"all": scores(0.3, 0.5, 0.9),
                               "strict": empty_scores()},
             "validation_metrics": {"all": scores(mae, r2, pairwise),
                                    "strict": scores(mae, r2, pairwise)}}
    return block


def classical(path: Path, *, columns: int = 6628, protein: str = "frozen",
              ridge_winner: str = "ridge_alpha100", lasso_winner: str = "lasso_alpha0.001",
              extra_results: dict | None = None, unavailable: list | None = None) -> None:
    results = {
        "linear": fit("linear", 0.0, 0.5238, mae=1.1511, r2=-2.57, nonzero=6628),
        "ridge_alpha1": fit("ridge", 1.0, 0.5266, nonzero=6628),
        "ridge_alpha100": fit("ridge", 100.0, 0.5332, mae=0.6290, r2=-0.0528,
                              nonzero=6628),
        "lasso_alpha0.0001": fit("lasso", 0.0001, 0.5369, nonzero=3940, cap=True),
        "lasso_alpha0.001": fit("lasso", 0.001, 0.5423, mae=0.5982, r2=0.0141,
                                nonzero=274, cap=False),
        "xgboost": fit("xgboost", 0.0, 0.5313, mae=0.6097, r2=-0.0111),
    }
    results.update(extra_results or {})
    chosen = {}
    if ridge_winner:
        chosen["ridge"] = {"alpha": results[ridge_winner]["alpha"],
                           "label": ridge_winner, "value": 0.5332, "alphas_compared": 4}
    if lasso_winner:
        chosen["lasso"] = {"alpha": results[lasso_winner]["alpha"],
                           "label": lasso_winner, "value": 0.5423, "alphas_compared": 3}
    payload = {"stage": "stage3_classical", "complete": True,
               "selection_metric": "pairwise_accuracy_macro", "selection_slice": "strict",
               "protein_mode": protein, "feature_columns": columns,
               "baselines": ["linear", "ridge", "lasso", "xgboost"],
               "results": results, "selected_alpha": chosen,
               "selected_alpha_unavailable": unavailable or []}
    path.write_text(json.dumps(payload), encoding="utf-8")


def train(path: Path, *, pairwise: float, rows: int = 24946, ranked: int = 5645,
          total: int = 42777, permuted: int | None = None,
          all_pairwise: float | None = None, counted: int | None = None) -> None:
    payload = {"stage": "stage3_train", "selection_metric": "pairwise_accuracy_macro",
               "selection_slice": "strict", "selection_slice_rows": rows,
               "best_epoch": 9, "history": [{"epoch": index} for index in range(10)],
               "checkpoint": "runs/evaluator/checkpoint.pt",
               "overrides": {} if permuted is None else
               {"protein_permute_seed": permuted},
               "validation_metrics": {
                   "all": scores(0.5144, 0.2395,
                                 pairwise if all_pairwise is None else all_pairwise,
                                 rows, ranked, total, counted),
                   "strict": scores(0.5144, 0.2395, pairwise, rows, ranked, total,
                                    counted)}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def eval_on_train(path: Path, *, mae: float, pairwise: float,
                  checkpoint: str = "runs/evaluator/checkpoint.pt",
                  partition: str = "train") -> None:
    """The manifest of `--mode eval --partition train`: the SAME checkpoint scored on the
    training rows. It keys its scores under `<partition>_metrics`, so a train eval writes
    `train_metrics` (src/molopt/evaluator/train.py)."""
    payload = {"stage": "stage3_eval", "partition": partition,
               "checkpoint": checkpoint,
               "train_metrics": {"all": scores(mae, 0.5, pairwise),
                                 "strict": empty_scores()}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def capacity(path: Path, *, rows: int, epochs: int, r2: float, regularized: bool,
             verdict: str = "fits") -> None:
    payload = {"stage": "stage3_capacity", "regularized": regularized,
               "epochs_run": epochs, "epochs_requested": 200,
               "sample": {"rows": rows}, "fit_train_r2_threshold": 0.95,
               "final": {"train_r2": r2, "train_mae": 0.12,
                         "train_pairwise_accuracy_macro": 0.98},
               "verdict": verdict}
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def runs(tmp_path: Path) -> Path:
    directory = tmp_path / "evaluator"
    directory.mkdir()
    classical(directory / f"classical_frozen_1_t1a-{PID}.json")
    classical(directory / f"classical_no_protein_2_t1b-{PID}.json", columns=6148,
              protein="none")
    classical(directory / f"classical_frozen_3_t1c-{PID}.json",
              ridge_winner="ridge_alpha1", lasso_winner="lasso_alpha0.0001")
    train(directory / f"train_frozen_direct_4_t3a-{PID}.json", pairwise=0.7778)
    train(directory / f"train_frozen_direct_5_t3b-{PID}.json", pairwise=0.7805,
          permuted=20261119)
    train(directory / f"train_frozen_direct_6_t3c-{PID}.json", pairwise=0.6500,
          rows=493852, ranked=100000, total=400000)
    train(directory / f"train_frozen_direct_7_t3d-{PID}.json", pairwise=0.6400,
          rows=493852, ranked=100000, total=400000, permuted=20261119)
    eval_on_train(directory / f"eval_frozen_direct_train_12_t3a-eval-{PID}.json",
                  mae=0.2727, pairwise=0.8310)
    capacity(directory / f"capacity_frozen_direct_8_t2a-{PID}.json", rows=1024, epochs=14,
             r2=0.9594, regularized=False)
    capacity(directory / f"capacity_frozen_direct_9_t2b-{PID}.json", rows=1024, epochs=25,
             r2=0.9512, regularized=False)
    capacity(directory / f"capacity_frozen_direct_10_t2c-{PID}.json", rows=1024,
             epochs=106, r2=0.9526, regularized=True)
    capacity(directory / f"capacity_frozen_direct_11_t2d-{PID}.json", rows=100000,
             epochs=13, r2=0.9508, regularized=False)
    return directory


def text(runs: Path, slice_name: str = "strict") -> str:
    return "\n".join(tables.tables(runs, PID, slice_name))


def test_a_regularised_model_is_shown_at_the_level_the_run_selected(runs: Path) -> None:
    # The whole point of the grid: the table may only quote the winner. t1c selected
    # ridge_alpha1 (0.5266) and t1a selected ridge_alpha100 (0.5332), so the two splits'
    # tables must not show the same ridge row.
    pair75, sub100 = text(runs).split("### Baselines vs the MLP")[1:3]
    assert "ridge, alpha 1.0 (selected of 4)" in pair75 and "0.5266" in pair75
    assert "ridge, alpha 100.0 (selected of 4)" in sub100 and "0.5332" in sub100
    # A level that was swept but not selected never appears as a row of its own.
    assert "| ridge_alpha1 |" not in pair75 and "| ridge_alpha100 |" not in sub100


def test_a_level_nobody_selected_is_not_quoted(runs: Path) -> None:
    pair75, sub100 = text(runs).split("### Baselines vs the MLP")[1:3]
    # lasso_alpha0.001 won in t1a (SUB100), lasso_alpha0.0001 in t1c (PAIR75). Each table
    # shows ONE lasso row, and it is its own split's winner.
    assert pair75.count("| lasso,") == 1 and "lasso, alpha 0.0001" in pair75
    assert sub100.count("| lasso,") == 1 and "lasso, alpha 0.001 (" in sub100


def test_an_unconverged_winner_is_flagged_in_its_row_and_in_the_footnotes(
        runs: Path) -> None:
    # t1c's lasso winner spent its whole iteration budget. A reader must not meet that
    # number without being told, in the row AND where the caveats are collected.
    out = text(runs)
    assert "unconverged (spent all 2000 iterations)" in out
    assert "the selected lasso, alpha 0.0001" in out


def test_a_fit_held_up_by_one_coefficient_is_visible(runs: Path, tmp_path: Path) -> None:
    # constant_predictor is false with one non-zero coefficient (it is not literally a
    # constant), so the non-zero count is what has to carry the information.
    degenerate = {"lasso_alpha0.01": fit("lasso", 0.01, 0.2047, nonzero=1, cap=False)}
    directory = tmp_path / "one"
    directory.mkdir()
    classical(directory / f"classical_frozen_1_t1a-{PID}.json",
              extra_results=degenerate, lasso_winner="lasso_alpha0.01")
    out = "\n".join(tables.tables(directory, PID, "strict"))
    assert "1/6,628 non-zero" in out
    assert "constant predictor" not in out


def test_a_missing_task_prints_a_dash_and_is_named(runs: Path) -> None:
    (runs / f"train_frozen_direct_6_t3c-{PID}.json").unlink()
    out = text(runs)
    assert "| **MLP (t3c)** | - | - | - | - | - | manifest missing |" in out
    assert "**No manifest for t3c.**" in out
    # The side that DOES exist is still printed; what is refused is the difference, which
    # would otherwise be computed against a number from somewhere else.
    block = out.split("### Protein control")[1].split("###")[0]
    assert "| - | 0.6400 | - | 1 |" in block


def test_the_protein_difference_is_the_two_numbers_it_prints(runs: Path) -> None:
    out = text(runs)
    block = out.split("### Protein control")[1]
    assert "| 0.7778 | 0.7805 | +0.0027 |" in block
    assert "| 0.6500 | 0.6400 | -0.0100 |" in block
    assert "no effect was measured" in block


def test_the_classical_ablation_names_both_alphas_when_they_differ(runs: Path) -> None:
    out = text(runs).split("### Protein control on the classical side")[1]
    assert "6628 with the protein block, 6148 without it" in out
    assert "| linear | 0.5238 | 0.5238 | +0.000000 |" in out


def test_the_slice_the_tables_read_is_the_slice_they_say(runs: Path) -> None:
    # Asking for a slice that is not in the manifests must empty the columns, not fall
    # back to another slice: a table headed `strict` holding `all` numbers is the error
    # this test exists for.
    out = text(runs, "non_cliff")
    assert "`non_cliff` slice" in out
    assert "0.7778" not in out


def test_the_macros_own_denominator_is_quoted_when_the_manifest_has_it(
        tmp_path: Path) -> None:
    """`queries_ranked` counts the queries that cleared min_candidates, NOT the terms of
    the macro: a query whose candidates all sit inside the direction tolerance adds no
    term yet is counted. The t3a dump of 2026-11-20 showed how far apart the two are -
    5,645 cleared the bar, 3,892 entered the macro, 9.1 % and not 13.2 % of the 42,777
    queries - so the metric now records the real denominator and the footnote quotes it.
    """
    directory = tmp_path / "evaluator"
    directory.mkdir()
    train(directory / f"train_frozen_direct_1_t3a-{PID}.json", pairwise=0.7778,
          ranked=5645, total=42777, counted=3892)
    out = text(directory)
    assert "t3a: 3,892 of 42,777 queries are in the pairwise macro** (9.1 %)" in out
    assert "5,645 cleared the min_candidates bar" in out
    assert "further apart than the direction tolerance" in out
    # The phrasing that implied the larger set WAS the macro is gone for good.
    assert "queries could be ranked" not in out


def test_a_slice_that_covers_every_row_is_called_out(runs: Path) -> None:
    out = text(runs)
    assert "the `strict` slice holds every validation row** (24,946)" in out
    assert "5,645 of 42,777 queries cleared the min_candidates bar** (13.2 %)" in out
    assert "an upper bound on the macro's own denominator" in out


def test_the_capacity_table_carries_the_threshold_and_the_verdict(runs: Path) -> None:
    out = text(runs).split("### Capacity")[1]
    assert "| t2d | 100,000 | off | 13 of 200 | 0.9508 | 0.95 | 0.9800 | fits |" in out
    assert "| t2c | 1,024 | as configured | 106 of 200 |" in out


def test_the_mlp_train_half_comes_from_the_eval_run_of_the_same_checkpoint(runs: Path
                                                                           ) -> None:
    # The meeting asked for train AND validation. A `--mode train` manifest has no pass
    # over the training rows, so the train half can only come from the eval run.
    pair75 = text(runs).split("### Baselines vs the MLP")[1]
    assert "| **MLP (t3a)** | 0.2727 | 0.8310 | 0.5144 | 0.2395 | 0.7778 |" in pair75
    # And the header has to say which slice each half is: the train half CANNOT be strict
    # on this split.
    assert "Train columns: `all` slice. Validation columns: `strict` slice." in pair75


def test_a_split_without_its_train_eval_says_so_instead_of_dropping_the_columns(
        runs: Path) -> None:
    # SUB100's eval-on-train run is not in the fixture, which is the real batch's state.
    sub100 = text(runs).split("### Baselines vs the MLP")[2]
    assert "no train numbers: the `t3c-eval` eval run is missing" in sub100
    assert "| **MLP (t3c)** | - | - | 0.5144 | 0.2395 | 0.6500 |" in sub100
    assert "**t3c: no train numbers.**" in text(runs)


def test_the_classical_train_half_is_the_rows_it_was_fitted_on(runs: Path) -> None:
    pair75 = text(runs).split("### Baselines vs the MLP")[1]
    # The fixture's fits all carry train MAE 0.3 / train pairwise 0.9 at the `all` slice.
    assert "| linear | 0.3000 | 0.9000 | 1.1511 |" in pair75


def test_the_train_half_is_not_read_at_a_slice_that_is_empty_on_this_split(runs: Path
                                                                          ) -> None:
    """Review 2026-11-20: both halves were read at `strict`, and on PAIR75 the strict
    slice of the TRAIN partition holds no rows at all (`unseen_pair = is_validation`), so
    the train columns were `-` for a batch that had completed perfectly. The fixture now
    carries an empty train/strict, which is what the real manifests carry.
    """
    both_strict = "\n".join(tables.tables(runs, PID, "strict", "strict"))
    assert "| **MLP (t3a)** | - | - | 0.5144" in both_strict
    assert "| linear | - | - | 1.1511" in both_strict
    # The default reading says what it read, and gets the numbers.
    default = text(runs)
    assert "| **MLP (t3a)** | 0.2727 | 0.8310 |" in default
    assert "train MAE (all)" in default and "validation MAE (strict)" in default
    assert "the `strict` slice of the train partition is empty" in default


def test_asking_for_all_on_both_sides_is_allowed(runs: Path) -> None:
    out = "\n".join(tables.tables(runs, PID, "all", "all"))
    assert "Train columns: `all` slice. Validation columns: `all` slice." in out
    # With one slice on both sides there is nothing to warn about.
    assert "the `strict` slice of the train partition is empty" not in out


def test_two_halves_of_different_checkpoints_are_refused(runs: Path) -> None:
    """Review 2026-11-20: the two halves were joined into one MLP row without checking
    they belong to one model. A train number from checkpoint A beside a validation number
    from checkpoint B is not a gap.
    """
    eval_on_train(runs / f"eval_frozen_direct_train_12_t3a-eval-{PID}.json",
                  mae=0.2727, pairwise=0.8310, checkpoint="runs/evaluator/other.pt")
    out = text(runs)
    assert "TRAIN HALF REFUSED" in out
    assert "different checkpoints (checkpoint.pt and other.pt)" in out
    assert "| **MLP (t3a)** | - | - | 0.5144" in out
    assert "**t3a: the train half was refused.**" in out


def test_an_eval_of_the_validation_partition_is_not_a_train_half(runs: Path) -> None:
    # An eval on validation writes the same shape of manifest; taken as a train half it
    # would put a second validation column where the gap should be.
    eval_on_train(runs / f"eval_frozen_direct_train_12_t3a-eval-{PID}.json",
                  mae=0.2727, pairwise=0.8310, partition="validation")
    out = text(runs)
    assert "scored the 'validation' partition, not 'train'" in out
    assert "| **MLP (t3a)** | - | - | 0.5144" in out


def test_a_swept_model_with_no_recorded_winner_is_not_quoted(tmp_path: Path) -> None:
    """The hole review 13 found: with neither a selected level nor a place in
    selected_alpha_unavailable, the first fit of the grid used to be printed - an alpha
    nobody selected, in a table whose claim is that it shows only selected ones."""
    directory = tmp_path / "nowinner"
    directory.mkdir()
    path = directory / f"classical_frozen_1_t1a-{PID}.json"
    classical(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["alphas"] = {"linear": [0.0], "ridge": [1.0, 100.0],
                         "lasso": [0.0001, 0.001], "xgboost": [0.0]}
    del payload["selected_alpha"]["ridge"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    out = "\n".join(tables.tables(directory, PID, "strict"))
    assert "ridge (swept [1.0, 100.0] and no level is recorded as selected)" in out
    # Neither level's score reaches the table.
    assert "0.5266" not in out and "0.5332" not in out
    # A model with a one-level grid is still shown from its own fit.
    assert "| linear | 0.3000 | 0.9000 | 1.1511 |" in out


def test_the_train_and_validation_of_one_row_are_the_same_model(runs: Path) -> None:
    assert ("The train and validation columns of one row come from the SAME model"
            in text(runs))


def test_every_table_warns_that_the_scores_are_post_selection(runs: Path) -> None:
    assert "An independent estimate needs a held-out set" in text(runs)


def test_a_model_with_no_selected_level_is_named_not_dropped(tmp_path: Path) -> None:
    directory = tmp_path / "none"
    directory.mkdir()
    classical(directory / f"classical_frozen_1_t1a-{PID}.json", lasso_winner="",
              unavailable=["lasso"])
    out = "\n".join(tables.tables(directory, PID, "strict"))
    assert "lasso (no level selected: the metric was undefined at every level)" in out
    assert "| lasso_alpha0.001 |" not in out


def test_the_cli_refuses_a_log_and_a_pid_together(tmp_path: Path) -> None:
    assert tables.main([str(tmp_path), "some.log", "--pid", "1"]) == 2
    assert tables.main([str(tmp_path / "nope")]) == 2


def test_the_cli_reads_the_pid_out_of_the_log(runs: Path, tmp_path: Path,
                                              capsys: pytest.CaptureFixture) -> None:
    log = tmp_path / "batch.log"
    log.write_text(f"+ python run_evaluator.py --run-id t1a-{PID} --mode classical\n",
                   encoding="utf-8")
    assert tables.main([str(runs), str(log)]) == 0
    assert f"batch pid {PID}" in capsys.readouterr().out
