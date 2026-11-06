"""Phân tích ca sai của Stage 3 (việc 6, họp 02/11/2026). Phần numpy không cần torch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molopt.evaluator import analysis, data, metrics

TOLERANCE = 0.3


def test_code_frequency_counts_each_code_over_the_whole_array() -> None:
    assert analysis.code_frequency(np.array([2, 0, 2, 2])).tolist() == [3, 1, 3, 3]


def test_molecule_frequency_counts_both_sides_of_the_pair() -> None:
    left, right = analysis.molecule_frequency(np.array([0, 1]), np.array([1, 2]))
    assert left.tolist() == [1, 2]
    assert right.tolist() == [2, 1]


def test_edit_codes_ignore_the_direction_of_the_edit() -> None:
    codes = analysis.edit_codes(np.array([3, 5, 3]), np.array([5, 3, 4]))
    assert codes[0] == codes[1]
    assert codes[0] != codes[2]


def test_quantile_groups_merge_equal_edges_and_keep_every_row() -> None:
    values = np.array([1, 1, 1, 1, 2, 5], dtype=np.float64)
    codes, names = analysis.quantile_groups(values, 5)
    assert len(names) == len(set(codes.tolist()))
    assert (codes >= 0).all()
    assert codes[0] == codes[3]


def test_quantile_groups_put_missing_values_in_their_own_group() -> None:
    codes, names = analysis.quantile_groups(np.array([1.0, np.nan, 3.0]), 2)
    assert names[codes[1]] == analysis.MISSING_LABEL
    assert names[codes[0]] != analysis.MISSING_LABEL


def test_quantile_groups_handle_a_constant_column() -> None:
    codes, names = analysis.quantile_groups(np.array([2.0, 2.0]), 5)
    assert codes.tolist() == [0, 0]
    assert len(names) == 1


def test_quantile_groups_refuse_zero_bins() -> None:
    with pytest.raises(ValueError, match="quantile_bins"):
        analysis.quantile_groups(np.array([1.0]), 0)


def test_magnitude_groups_keep_the_noise_band_separate() -> None:
    truth = np.array([0.1, -0.2, 0.5, -1.5, 2.0])
    codes, names = analysis.magnitude_groups(truth, TOLERANCE, 2)
    assert codes[:2].tolist() == [0, 0]
    assert (codes[2:] > 0).all()
    assert "trong sai số" in names[0]


def test_sign_agreement_skips_rows_inside_the_tolerance() -> None:
    counted, correct = analysis.sign_agreement(np.array([0.1, 1.0, -1.0]),
                                               np.array([-5.0, 2.0, 3.0]), TOLERANCE)
    assert counted.tolist() == [False, True, True]
    assert correct.tolist() == [False, True, False]


def test_spread_report_sees_a_shrunk_prediction() -> None:
    truth = np.array([-2.0, -1.0, 1.0, 2.0])
    report = analysis.spread_report(truth, 0.25 * truth)
    assert report["std_ratio"] == pytest.approx(0.25)
    assert report["slope"] == pytest.approx(0.25)
    assert report["bias"] == pytest.approx(0.0)


def test_group_table_reports_each_non_empty_group() -> None:
    codes = np.array([0, 0, 2])
    table = analysis.group_table(codes, ["a", "b", "c"], np.array([1.0, -1.0, 2.0]),
                                 np.array([1.0, 1.0, 2.5]), TOLERANCE)
    assert [row["group"] for row in table] == ["a", "c"]
    assert table[0]["sign_accuracy"] == pytest.approx(0.5)
    assert table[1]["mae"] == pytest.approx(0.5)


def random_queries(seed: int = 0, rows: int = 400) -> tuple[dict[str, np.ndarray],
                                                            np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = {"molecule_a": rng.integers(0, 15, rows), "molecule_b": rng.integers(15, 40, rows),
            "protein": rng.integers(0, 4, rows), "endpoint": rng.integers(0, 2, rows)}
    truth = rng.normal(size=rows)
    prediction = truth + rng.normal(scale=1.0, size=rows)
    return keys, truth, prediction


def test_query_table_matches_the_headline_ranking_metric() -> None:
    """Trung bình theo query phải bằng đúng pairwise_accuracy_macro của chỉ số chính."""
    keys, truth, prediction = random_queries()
    queries = analysis.query_table(keys, truth, prediction, TOLERANCE, 2, True)
    reference, _, expanded_truth, expanded_prediction = metrics.expand_directions(
        keys["molecule_a"], keys["molecule_b"], truth, prediction)
    groups = metrics.query_groups([reference, np.concatenate([keys["protein"]] * 2),
                                   np.concatenate([keys["endpoint"]] * 2)])
    headline = metrics.ranking_metrics(groups, expanded_truth, expanded_prediction, [1], 2,
                                       TOLERANCE)
    used = queries["counted"] > 0
    macro = float(np.mean(queries["agree"][used] / queries["counted"][used]))
    assert macro == pytest.approx(headline["pairwise_accuracy_macro"])
    assert queries["agree"].sum() == pytest.approx(
        headline["pairwise_accuracy_micro"] * headline["pairs_counted"])
    assert queries["top_1"].mean() == pytest.approx(headline["top_1"])
    assert queries["agree"].size == headline["queries_ranked"]


def test_query_group_table_splits_queries() -> None:
    keys, truth, prediction = random_queries(1)
    queries = analysis.query_table(keys, truth, prediction, TOLERANCE, 2, True)
    codes = (queries["candidates"] > np.median(queries["candidates"])).astype(np.int64)
    table = analysis.query_group_table(codes, ["nhỏ", "lớn"], queries)
    assert sum(row["queries"] for row in table) == queries["agree"].size


def test_worst_proteins_are_sorted_and_respect_the_minimum() -> None:
    queries = {"protein": np.array([0, 0, 1, 1, 1, 2]), "agree": np.array([1, 1, 0, 1, 0, 0]),
               "counted": np.array([1, 1, 1, 1, 1, 1])}
    worst = analysis.worst_proteins(queries, 2, 5)
    assert [entry["protein"] for entry in worst] == [1, 0]
    assert worst[0]["pairwise_accuracy_macro"] == pytest.approx(1 / 3)


def test_worst_edits_report_a_representative_fragment_pair() -> None:
    fragment_a = np.array([0, 1, 0, 2, 2])
    fragment_b = np.array([1, 0, 1, 3, 3])
    edits = analysis.edit_codes(fragment_a, fragment_b)
    truth = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    prediction = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
    worst = analysis.worst_edits(edits, fragment_a, fragment_b, truth, prediction,
                                 TOLERANCE, 2, 1)
    assert len(worst) == 1
    assert {worst[0]["fragment_a"], worst[0]["fragment_b"]} == {0, 1}
    assert worst[0]["sign_accuracy"] == pytest.approx(1 / 3)
    assert worst[0]["rows"] == 3


def test_format_group_table_prints_one_line_per_group() -> None:
    lines = analysis.format_group_table("x", [{"group": "a", "rows": 3, "mae": None}])
    assert len(lines) == 3
    assert lines[-1].strip().endswith("-")


# --------------------------------------------------------------------------- chạy trọn


torch = pytest.importorskip("torch", reason="run_errors và run_features cần torch")

from molopt.evaluator import models, train  # noqa: E402

LIGAND_BITS = 32
FRAGMENT_BITS = 16
PROTEIN_DIM = 8
ROWS = 60


def fake_partition(rng: np.random.Generator) -> tuple[data.PairArrays, dict[str, np.ndarray],
                                                      pa.Table]:
    molecule_a = rng.integers(0, 6, ROWS)
    molecule_b = rng.integers(6, 12, ROWS)
    arrays = data.PairArrays(
        molecule_a=molecule_a, molecule_b=molecule_b,
        fragment_a=rng.integers(0, 4, ROWS), fragment_b=rng.integers(4, 8, ROWS),
        protein=rng.integers(0, 3, ROWS), endpoint=rng.integers(0, 4, ROWS),
        target=rng.normal(size=ROWS).astype(np.float32), slices={})
    keys = {"molecule_a": molecule_a, "molecule_b": molecule_b, "protein": arrays.protein,
            "endpoint": arrays.endpoint}
    table = pa.table({
        "pair_id": pa.array([f"p{i}" for i in range(ROWS)]),
        "endpoint": pa.array(["Ki"] * ROWS),
        "protein_hash": pa.array([f"h{i}" for i in arrays.protein]),
        "n_assays": pa.array(rng.integers(1, 4, ROWS)),
        "n_documents": pa.array(rng.integers(1, 3, ROWS)),
        "n_distinct_values": pa.array(rng.integers(1, 3, ROWS)),
        "plausible_range": pa.array([True] * (ROWS - 1) + [None], pa.bool_()),
        "source_quality_warning": pa.array([None] * (ROWS - 2) + ["x", "y"], pa.string()),
        "unseen_pair": pa.array([False] * ROWS),
        "qc_status": pa.array(["ok"] * ROWS),
    })
    return arrays, keys, table


@pytest.fixture()
def fake_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
             config: dict[str, Any]) -> dict[str, Any]:
    """Thay phần đọc dữ liệu và cache bằng dữ liệu nhỏ, giữ nguyên mọi phần còn lại."""
    rng = np.random.default_rng(0)
    arrays, keys, table = fake_partition(rng)
    torch.manual_seed(0)
    settings = {"ligand_hidden": [16], "ligand_out": 8, "edit_hidden": [8], "edit_out": 4,
                "protein_out": 4, "endpoint_dim": 3, "head_hidden": [8], "dropout": 0.0}
    model = models.EvaluatorModel(settings, LIGAND_BITS, FRAGMENT_BITS, PROTEIN_DIM, 4,
                                  models.HEAD_DIRECT, models.PROTEIN_FROZEN)

    class FakeCaches:
        molecule_packed = np.packbits(rng.integers(0, 2, (12, LIGAND_BITS)).astype(np.uint8),
                                      axis=1)
        fragment_packed = np.packbits(rng.integers(0, 2, (8, FRAGMENT_BITS)).astype(np.uint8),
                                      axis=1)
        ligand_bits = LIGAND_BITS
        fragment_bits = FRAGMENT_BITS
        molecule_keys = pa.array([f"m{i}" for i in range(12)])
        fragment_keys = pa.array([f"f{i}" for i in range(8)])

    vectors = rng.normal(size=(3, PROTEIN_DIM)).astype(np.float32)
    loaded = train.LoadedCheckpoint(
        state={}, model=model, caches=FakeCaches(), protein_source=vectors,
        protein_vocabulary=pa.array(["h0", "h1", "h2"]),
        protein_mode=models.PROTEIN_FROZEN, configuration="frozen", head="direct",
        device="cpu", eval_batch_size=16)
    extra = pa.table({
        "pair_id": table.column("pair_id"),
        "max_replicate_std": pa.array([0.1] * (ROWS - 1) + [None], pa.float64()),
        "p_a_range": pa.array(rng.normal(size=ROWS)),
        "mixed_assay_type": pa.array([False] * ROWS),
        "assay_types": pa.array(["B"] * (ROWS - 1) + [None], pa.string()),
    })
    seen: list[str] = []

    def fake_prepare(config, root, partition, caches, vocabulary, limit):  # noqa: ANN001
        seen.append(partition)
        return arrays, keys, table

    monkeypatch.setattr(train, "load_checkpoint", lambda *args: loaded)
    monkeypatch.setattr(train, "prepare_arrays", fake_prepare)
    monkeypatch.setattr(train.data, "read_columns", lambda *args, **kwargs: extra)
    monkeypatch.setattr(train, "layout", lambda config, root: (tmp_path, tmp_path))
    run_config = json.loads(json.dumps(config))
    run_config["stage3"]["analysis"]["feature_rows_per_partition"] = 10
    return {"config": run_config, "root": tmp_path, "seen": seen, "rows": ROWS, "model": model}


def test_run_errors_writes_the_report_and_the_row_tables(fake_run: dict[str, Any]) -> None:
    report = train.run_errors(fake_run["config"], fake_run["root"], Path("x.pt"), "s",
                              data.TRAIN)
    folder = fake_run["root"] / "errors_frozen_direct_train_s"
    assert (folder / "report.json").is_file()
    rows = pq.read_table(folder / "rows.parquet")
    assert rows.num_rows == fake_run["rows"]
    assert pq.read_table(folder / "queries.parquet").num_rows == report[
        "queries_overall"]["queries"]
    for name in ("abs_delta_p", "endpoint", "protein_rows", "mixed_assay_type",
                 "source_quality_warning"):
        assert sum(group["rows"] for group in report["by_row"][name]) == fake_run["rows"]
    assert fake_run["seen"] == [data.TRAIN]


def test_run_features_samples_train_and_validation_only(fake_run: dict[str, Any]) -> None:
    manifest = train.run_features(fake_run["config"], fake_run["root"], Path("x.pt"), "s")
    table = pq.read_table(manifest["file"])
    assert sorted(set(table.column("partition").to_pylist())) == [data.TRAIN, data.VALIDATION]
    assert table.num_rows == 20
    assert manifest["feature_dim"] == 2 * 8 + 2 * 4 + 4 + 3
    assert data.TEST not in fake_run["seen"]


def test_representations_feed_the_predictor_exactly() -> None:
    """Vector xuất cho t-SNE đúng là đầu vào của MLP cuối: đưa qua predictor ra lại Δp."""
    torch.manual_seed(0)
    settings = {"ligand_hidden": [16], "ligand_out": 8, "edit_hidden": [8], "edit_out": 4,
                "protein_out": 4, "endpoint_dim": 3, "head_hidden": [8], "dropout": 0.0}
    for head in models.HEADS:
        model = models.EvaluatorModel(settings, LIGAND_BITS, FRAGMENT_BITS, PROTEIN_DIM, 4,
                                      head, models.PROTEIN_FROZEN).eval()
        inputs = (torch.rand(5, LIGAND_BITS), torch.rand(5, LIGAND_BITS),
                  torch.rand(5, FRAGMENT_BITS), torch.rand(5, FRAGMENT_BITS),
                  torch.rand(5, PROTEIN_DIM), torch.zeros(5, dtype=torch.long))
        with torch.no_grad():
            pieces = model.representations(*inputs)
            scores = [model.predictor(piece).squeeze(1) for piece in pieces]
            expected = scores[1] - scores[0] if head == models.HEAD_TWIN else scores[0]
            assert torch.allclose(model(*inputs), expected)


def test_run_diagnose_reports_layers_and_every_input(fake_run: dict[str, Any]) -> None:
    report = train.run_diagnose(fake_run["config"], fake_run["root"], Path("x.pt"), "s")
    assert (fake_run["root"] / "diagnose_frozen_direct_validation_s.json").is_file()
    assert report["rows"] == 10
    assert set(report["sensitivity"]) == {"ligand_b", "edit", "protein", "endpoint"}
    names = {layer["layer"] for layer in report["relu_layers"]}
    assert any(name.startswith("predictor") for name in names)
    assert any(name.startswith("ligand_encoder") for name in names)
    for layer in report["relu_layers"]:
        assert 0.0 <= layer["dead_fraction"] <= 1.0
        assert 0.0 <= layer["active_rate"] <= 1.0
    assert fake_run["seen"] == [data.VALIDATION]


def test_run_diagnose_sees_a_dead_head(fake_run: dict[str, Any]) -> None:
    """Lớp ẩn cuối của head chết hết -> dự đoán hằng số, độ nhạy 0, lớp đó chết 100 %."""
    hidden = fake_run["model"].predictor[0]
    with torch.no_grad():
        hidden.weight.zero_()
        hidden.bias.fill_(-1.0)
    report = train.run_diagnose(fake_run["config"], fake_run["root"], Path("x.pt"), "s")
    dead = {layer["layer"]: layer["dead_fraction"] for layer in report["relu_layers"]}
    assert dead["predictor.1"] == 1.0
    assert report["prediction"]["distinct_1e-6"] == 1
    assert report["prediction"]["std"] == 0.0
    for shift in report["sensitivity"].values():
        assert shift["mean_abs_shift"] == 0.0
