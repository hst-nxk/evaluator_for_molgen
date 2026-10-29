"""Logic thuần của src/molopt/dataset/pipeline.py (không đọc dữ liệu thật)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molopt.dataset import aggregate, pipeline

ENDPOINTS = ["Ki", "Kd", "IC50", "EC50"]
BUCKETS = 16
SUFFIX = "-REL"
HASH_A = "00" + "a" * 62
HASH_B = "0f" + "b" * 62
HASH_C = "10" + "c" * 62


@pytest.fixture()
def settings(config: dict[str, Any]) -> dict[str, Any]:
    return config["stage2b"]


def load_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Bảng như sau khi rename_roles: mọi vai trò trong LOAD_ROLES, cộng assay_key."""
    columns: dict[str, pa.Array] = {}
    for role in pipeline.LOAD_ROLES:
        if role == "p_activity":
            columns[role] = pa.array([row.get(role) for row in rows], pa.float64())
        elif role == "keep":
            columns[role] = pa.array([row.get(role, True) for row in rows], pa.bool_())
        else:
            columns[role] = pa.array([row.get(role) for row in rows], pa.string())
    table = pa.table(columns)
    return table.append_column("assay_key", pipeline.assay_keys(table))


def filtered(rows: list[dict[str, Any]], config: dict[str, Any]) -> pa.Table:
    table = load_table(rows)
    steps = pipeline.combined_mask(pipeline.filter_steps(
        table, config["qualifiers"]["exact"],
        config["stage2b"]["filter"]["resolved_status"]))
    return table.filter(steps[-1][1])


def complete_row(**overrides: Any) -> dict[str, Any]:
    """Một dòng đi qua được mọi bước lọc; override để thử từng bước."""
    row = {"observation_id": "o1", "source": "chembl37", "assay_id": "7",
           "assay_entry_id": None, "protein_hash": HASH_A, "protein_sequence": "MKT",
           "protein_accession": "P1", "protein_name": "kinase", "endpoint": "Ki",
           "inchikey": "AAAAAAAAAAAAAA-BBBBBBBBBB-N", "smiles": "Cc1ccccc1",
           "cx_extension": None, "molecule_source_id": "M1", "qualifier": "=",
           "p_activity": 7.5, "protein_status": "resolved", "assay_status": "resolved",
           "keep": True, "activity_context": None}
    row.update(overrides)
    return row


# --------------------------------------------------------------------------- định dạng


def test_fmt_int_uses_dots_for_thousands() -> None:
    assert pipeline.fmt_int(5338213) == "5.338.213"


def test_fmt_float_uses_a_comma_for_decimals_and_a_dash_for_nan() -> None:
    assert pipeline.fmt_float(1234.5) == "1.234,50"
    assert pipeline.fmt_float(None) == "–"
    assert pipeline.fmt_float(math.nan) == "–"


def test_fmt_pct_handles_zero_total() -> None:
    assert pipeline.fmt_pct(1, 4) == "25,0 %"
    assert pipeline.fmt_pct(1, 0) == "–"


def test_format_table_aligns_the_columns() -> None:
    lines = pipeline.format_table(["endpoint", "dòng"], [["Ki", "1.000"], ["EC50", "7"]])
    assert len(lines) == 4
    assert lines[0].startswith("endpoint")
    assert all(len(line) == len(lines[0]) for line in lines[1:])


def test_percentile_label_names_the_usual_percentiles() -> None:
    assert [pipeline.percentile_label(value) for value in (0, 50, 100, 99)] == [
        "min", "median", "max", "P99"]


# --------------------------------------------------------------------------- khóa, bucket


def test_assay_keys_joins_source_entry_and_assay() -> None:
    table = load_table([complete_row(), complete_row(assay_entry_id="1025", assay_id="1")])
    assert table.column("assay_key").to_pylist() == ["chembl37::7", "chembl37:1025:1"]


def test_bucket_of_uses_the_first_two_hex_digits() -> None:
    buckets = pipeline.bucket_of(pa.array([HASH_A, HASH_B, HASH_C]), BUCKETS)
    assert buckets.tolist() == [0, 15, 0]


def test_bucket_of_refuses_a_hash_that_is_not_hex() -> None:
    with pytest.raises(ValueError, match="không phải chuỗi hex"):
        pipeline.bucket_of(pa.array(["zz" + "a" * 62]), BUCKETS)


def test_endpoint_codes_marks_unknown_endpoints() -> None:
    codes = pipeline.endpoint_codes(pa.array(["Ki", "EC50", "Papp"]), ENDPOINTS)
    assert codes.tolist() == [0, 3, -1]


def test_smoke_mask_is_stable_and_keeps_whole_assays() -> None:
    keys = pa.array([f"chembl37::{number}" for number in range(2000)])
    first = pipeline.smoke_mask(keys, 200).to_pylist()
    assert first == pipeline.smoke_mask(keys, 200).to_pylist()
    assert 0 < sum(first) < len(first)


# --------------------------------------------------------------------------- lọc dòng


def test_non_empty_rejects_null_and_empty_strings() -> None:
    assert pipeline.non_empty(pa.array(["a", "", None])).to_pylist() == [True, False, False]


def test_filter_steps_labels_match_the_constant(config: dict[str, Any]) -> None:
    steps = pipeline.filter_steps(load_table([complete_row()]),
                                  config["qualifiers"]["exact"],
                                  config["stage2b"]["filter"]["resolved_status"])
    assert [label for label, _ in steps] == list(pipeline.FILTER_LABELS)


@pytest.mark.parametrize("overrides", [
    {"keep": False}, {"qualifier": ">"}, {"protein_status": "unresolved"},
    {"assay_status": "unresolved"}, {"p_activity": None}, {"inchikey": None},
    {"smiles": ""}])
def test_filter_steps_drops_rows_failing_any_step(overrides: dict[str, Any],
                                                  config: dict[str, Any]) -> None:
    assert filtered([complete_row(**overrides)], config).num_rows == 0


def test_filter_steps_keeps_a_complete_row(config: dict[str, Any]) -> None:
    assert filtered([complete_row()], config).num_rows == 1


def test_combined_mask_is_cumulative() -> None:
    steps = [("a", pa.array([True, True, None])), ("b", pa.array([True, False, True]))]
    masks = pipeline.combined_mask(steps)
    assert masks[0][1].to_pylist() == [True, True, False]
    assert masks[1][1].to_pylist() == [True, False, False]


# --------------------------------------------------------------------------- cờ độc lập


def test_independent_flags_marks_flagged_chembl_rows(settings: dict[str, Any]) -> None:
    table = load_table([
        complete_row(observation_id="o1", activity_context='{"potential_duplicate": 1}'),
        complete_row(observation_id="o2", activity_context='{"potential_duplicate": 0}'),
        complete_row(observation_id="o3", activity_context='{"potential_duplicate": "1"}'),
        complete_row(observation_id="o4", activity_context=None),
    ])
    flags = pipeline.independent_flags(table, settings["filter"]).to_pylist()
    assert flags == [False, True, False, True]


def test_independent_flags_reports_broken_json(settings: dict[str, Any]) -> None:
    table = load_table([complete_row(activity_context='{"potential_duplicate": 1')])
    with pytest.raises(ValueError, match="o1"):
        pipeline.independent_flags(table, settings["filter"])


# --------------------------------------------------------------------------- dòng shard


def shard_of(rows: list[dict[str, Any]], config: dict[str, Any]) -> pa.Table:
    return pipeline.shard_rows(filtered(rows, config), config["stage2b"]["filter"],
                               config["stage2a"]["dedup"]["relative_stereo_pattern"], SUFFIX)


def test_shard_rows_matches_the_schema(config: dict[str, Any]) -> None:
    assert shard_of([complete_row()], config).schema == pipeline.SHARD_SCHEMA


def test_shard_rows_adds_the_suffix_for_relative_stereo(config: dict[str, Any]) -> None:
    rows = [complete_row(), complete_row(cx_extension="|r|")]
    keys = shard_of(rows, config).column("molecule_key").to_pylist()
    assert keys[1] == keys[0] + SUFFIX


def test_write_shards_routes_rows_by_endpoint_and_bucket(
        tmp_path: Path, settings: dict[str, Any], config: dict[str, Any]) -> None:
    rows = [complete_row(observation_id="o1", endpoint="Ki", protein_hash=HASH_A),
            complete_row(observation_id="o2", endpoint="Kd", protein_hash=HASH_B)]
    table = filtered(rows, config)
    shard = pipeline.shard_rows(table, settings["filter"],
                                config["stage2a"]["dedup"]["relative_stereo_pattern"], SUFFIX)
    layout = pipeline.Layout(tmp_path, settings["files"])
    writer = pipeline.ShardWriter(layout, "shards_dir")
    pipeline.write_shards(writer, shard, pipeline.endpoint_codes(table.column("endpoint"),
                                                                 ENDPOINTS), ENDPOINTS, BUCKETS)
    writer.close()
    first = pq.read_table(layout.bucket_file("shards_dir", "Ki", 0))
    second = pq.read_table(layout.bucket_file("shards_dir", "Kd", 15))
    assert first.column("observation_id").to_pylist() == ["o1"]
    assert second.column("observation_id").to_pylist() == ["o2"]


# --------------------------------------------------------------------------- gom dần


def test_group_accumulator_takes_the_smallest_value_and_sums_rows() -> None:
    accumulator = pipeline.GroupAccumulator("key", ["value"], merge_every=2)
    for value in ("b", "a", "c"):
        accumulator.add(pa.table({"key": ["k"], "value": [value],
                                  "rows": pa.array([1], pa.int64())}))
    (row,) = accumulator.table().to_pylist()
    assert (row["key"], row["value"], row["rows"]) == ("k", "a", 3)


def test_group_accumulator_of_nothing_is_empty() -> None:
    assert pipeline.GroupAccumulator("key", ["value"], 2).table().num_rows == 0


# --------------------------------------------------------------------------- bước 1 → 4


def group_rows(rows: list[dict[str, Any]]) -> pa.Table:
    """Bảng bước 1 tối giản (chỉ các cột mà iter_tasks và attach_smiles cần)."""
    return pa.table({
        "assay_key": pa.array([row.get("assay_key", "s::1") for row in rows], pa.string()),
        "protein_hash": pa.array([row.get("protein_hash", HASH_A) for row in rows], pa.string()),
        "molecule_key": pa.array([row["molecule_key"] for row in rows], pa.string()),
        "smiles": pa.array([row.get("smiles", "Cc1ccccc1") for row in rows], pa.string()),
    })


def test_attach_smiles_reports_a_missing_molecule_key() -> None:
    groups = group_rows([{"molecule_key": "KEY-A"}]).drop_columns(["smiles"])
    molecules = pa.table({"molecule_key": ["KEY-B"], "smiles": ["CC"]})
    with pytest.raises(ValueError, match="không có trong bảng phân tử"):
        pipeline.attach_smiles(groups, molecules)


def test_attach_smiles_adds_the_representative_smiles() -> None:
    groups = group_rows([{"molecule_key": "KEY-A"}]).drop_columns(["smiles"])
    molecules = pa.table({"molecule_key": ["KEY-A"], "smiles": ["CCO"]})
    assert pipeline.attach_smiles(groups, molecules).column("smiles").to_pylist() == ["CCO"]


def test_iter_tasks_skips_groups_with_one_molecule() -> None:
    groups = group_rows([{"molecule_key": "KEY-A"},
                         {"molecule_key": "KEY-B", "assay_key": "s::2"},
                         {"molecule_key": "KEY-C", "assay_key": "s::2"}])
    tasks = list(pipeline.iter_tasks(groups, SUFFIX, groups_per_task=10))
    assert len(tasks) == 1
    (group, members) = tasks[0][0]
    assert [member.row for member in members] == [1, 2]


def test_iter_tasks_splits_into_batches() -> None:
    rows = []
    for assay in range(4):
        rows += [{"molecule_key": f"KEY-{assay}-{index}", "assay_key": f"s::{assay}"}
                 for index in range(2)]
    tasks = list(pipeline.iter_tasks(group_rows(rows), SUFFIX, groups_per_task=3))
    assert [len(task) for task in tasks] == [3, 1]


def test_iter_tasks_marks_relative_stereo_members() -> None:
    groups = group_rows([{"molecule_key": "KEY-A"}, {"molecule_key": "KEY-B" + SUFFIX}])
    (_, members) = list(pipeline.iter_tasks(groups, SUFFIX, 10))[0][0]
    assert [member.relative for member in members] == [False, True]


def test_find_group_pairs_runs_in_process(settings: dict[str, Any]) -> None:
    groups = group_rows([{"molecule_key": "KEY-A", "smiles": "Cc1ccccc1"},
                         {"molecule_key": "KEY-B", "smiles": "CCc1ccccc1"}])
    pairs, lost, counters = pipeline.find_group_pairs(groups, settings["mmp"], SUFFIX, None)
    assert len(pairs) == 1 and lost == []
    assert counters["molecules"] == 2 and counters["failed_smiles"] == 0


def test_find_group_pairs_counts_pairs_lost_to_relative_stereo(
        settings: dict[str, Any]) -> None:
    groups = group_rows([{"molecule_key": "KEY-A", "smiles": "Cc1ccccc1"},
                         {"molecule_key": "KEY-B" + SUFFIX, "smiles": "CCc1ccccc1"}])
    pairs, lost, _ = pipeline.find_group_pairs(groups, settings["mmp"], SUFFIX, None)
    assert pairs == [] and lost == [(0, 1)]


def test_dispersion_counts_splits_by_report_level() -> None:
    records = pa.table({"dispersion": pa.array([0.1, 0.5, 1.5, None], pa.float64())})
    counts = pipeline.dispersion_counts(records, [0.3, 1.0])
    assert counts["measured"] == 3 and counts["single"] == 1
    assert counts["le_0.3"] == 1 and counts["gt_1.0"] == 1


def test_merge_stats_adds_counters_and_numbers() -> None:
    merged = pipeline.merge_stats([{"rows": 2, "qc": {"pass": 1}, "dispersion": {"measured": 1}},
                                   {"rows": 3, "qc": {"pass": 2}, "dispersion": {"measured": 2}}])
    assert merged["rows"] == 5
    assert merged["qc"] == {"pass": 3}
    assert merged["dispersion"] == {"measured": 3}


# --------------------------------------------------------------------------- bước 5


def record_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.table({
        "pair_id": pa.array([row.get("pair_id", "p1") for row in rows], pa.string()),
        "endpoint": pa.array([row.get("endpoint", "Ki") for row in rows], pa.string()),
        "protein_hash": pa.array([row.get("protein_hash", HASH_A) for row in rows], pa.string()),
        "partition": pa.array([row.get("partition", "train") for row in rows], pa.string()),
        "smiles_a": pa.array([row.get("smiles_a", "A") for row in rows], pa.string()),
        "smiles_b": pa.array([row.get("smiles_b", "B") for row in rows], pa.string()),
        "constant_smiles": pa.array(["core"] * len(rows), pa.string()),
        "variable_a": pa.array([row.get("variable_a", "va") for row in rows], pa.string()),
        "variable_b": pa.array([row.get("variable_b", "vb") for row in rows], pa.string()),
        "delta_p_mean": pa.array([row["delta_p_mean"] for row in rows], pa.float64()),
        "n_assays": pa.array([row.get("n_assays", 2) for row in rows], pa.int64()),
        "n_distinct_values": pa.array([row.get("n_distinct_values", 2) for row in rows],
                                      pa.int64()),
        "n_documents": pa.array([row.get("n_documents", 2) for row in rows], pa.int64()),
        "qc_status": pa.array([row.get("qc_status", "pass") for row in rows], pa.string()),
        "min_target_confidence": pa.array([row.get("min_target_confidence", 9) for row in rows],
                                          pa.int64()),
        "p_a_range": pa.array([row.get("p_a_range", 0.0) for row in rows], pa.float64()),
        "max_replicate_std": pa.array([row.get("max_replicate_std") for row in rows],
                                      pa.float64()),
        "inactive_comment": pa.array([row.get("inactive_comment", False) for row in rows],
                                     pa.bool_()),
        "nonstandard_sequence": pa.array(
            [row.get("nonstandard_sequence", False) for row in rows], pa.bool_()),
        "molecule_a_key": pa.array([row.get("molecule_a_key", "KEY-A") for row in rows],
                                   pa.string()),
        "molecule_b_key": pa.array([row.get("molecule_b_key", "KEY-B") for row in rows],
                                   pa.string()),
        "unseen_pair": pa.array([row.get("unseen_pair", False) for row in rows], pa.bool_()),
        "plausible_range": pa.array([row.get("plausible_range", True) for row in rows],
                                    pa.bool_()),
        "short_protein": pa.array([row.get("short_protein", False) for row in rows], pa.bool_()),
        "source_quality_warning": pa.array(
            [row.get("source_quality_warning") for row in rows], pa.string()),
        "assay_types": pa.array([row.get("assay_types", "B") for row in rows], pa.string()),
        "mixed_assay_type": pa.array([row.get("mixed_assay_type", False) for row in rows],
                                     pa.bool_()),
    })


def test_oriented_generator_rows_keeps_the_direction_of_a_gain() -> None:
    table = pipeline.oriented_generator_rows(record_table([{"delta_p_mean": 1.2}]), 0.3)
    (row,) = table.to_pylist()
    assert (row["smiles_weak"], row["smiles_strong"]) == ("A", "B")
    assert (row["variable_weak"], row["variable_strong"]) == ("va", "vb")
    assert row["gain"] == pytest.approx(1.2)


def test_oriented_generator_rows_swaps_a_negative_delta() -> None:
    (row,) = pipeline.oriented_generator_rows(record_table([{"delta_p_mean": -1.2}]),
                                              0.3).to_pylist()
    assert (row["smiles_weak"], row["smiles_strong"]) == ("B", "A")
    assert (row["variable_weak"], row["variable_strong"]) == ("vb", "va")
    assert row["gain"] == pytest.approx(1.2)


def test_oriented_generator_rows_applies_the_strict_minimum_gain() -> None:
    rows = record_table([{"delta_p_mean": 0.3}, {"delta_p_mean": 0.31},
                         {"delta_p_mean": -0.3}])
    table = pipeline.oriented_generator_rows(rows, 0.3)
    assert table.num_rows == 1
    assert table.column("gain").to_pylist() == pytest.approx([0.31])


def test_oriented_generator_rows_has_the_declared_columns() -> None:
    table = pipeline.oriented_generator_rows(record_table([{"delta_p_mean": 1.0}]), 0.3)
    assert table.column_names == pipeline.GENERATOR_COLUMNS


def test_oriented_generator_rows_carries_the_descriptive_flags() -> None:
    rows = record_table([{"delta_p_mean": -1.0, "unseen_pair": True, "short_protein": True,
                          "plausible_range": False}])
    (row,) = pipeline.oriented_generator_rows(rows, 0.3).to_pylist()
    assert (row["unseen_pair"], row["plausible_range"],
            row["short_protein"]) == (True, False, True)


def test_oriented_generator_rows_carries_the_evidence_columns() -> None:
    rows = record_table([{"delta_p_mean": -1.0, "n_documents": 1, "n_distinct_values": 1,
                          "qc_status": "identical_values", "min_target_confidence": 8,
                          "p_a_range": 3.0, "inactive_comment": True}])
    (row,) = pipeline.oriented_generator_rows(rows, 0.3).to_pylist()
    assert (row["n_documents"], row["n_distinct_values"], row["qc_status"],
            row["min_target_confidence"], row["p_a_range"], row["inactive_comment"]) == (
        1, 1, "identical_values", 8, 3.0, True)


def test_evaluator_columns_carry_every_evidence_and_flag_column() -> None:
    for name in [*pipeline.EVIDENCE_COLUMNS, *pipeline.FLAG_COLUMNS]:
        assert name in pipeline.EVALUATOR_COLUMNS and name in pipeline.GENERATOR_COLUMNS


# --------------------------------------------------------------------------- cờ mô tả bước 5


def test_pair_codes_gives_one_code_per_molecule_pair() -> None:
    codes: dict[str, int] = {}
    rows = record_table([{"delta_p_mean": 1.0},
                         {"delta_p_mean": 1.0, "molecule_b_key": "KEY-C"},
                         {"delta_p_mean": 1.0}])
    values = pipeline.pair_codes(rows, codes, grow=True)
    assert values[0] == values[2] and values[0] != values[1]
    assert set(codes) == {"KEY-A", "KEY-B", "KEY-C"}


def test_pair_codes_marks_an_unknown_key_with_minus_one() -> None:
    codes = {"KEY-A": 0, "KEY-B": 1}
    rows = record_table([{"delta_p_mean": 1.0, "molecule_b_key": "KEY-LA"}])
    assert pipeline.pair_codes(rows, codes, grow=False).tolist() == [-1]
    assert "KEY-LA" not in codes


def test_pair_codes_is_order_sensitive() -> None:
    codes: dict[str, int] = {}
    forward = pipeline.pair_codes(record_table([{"delta_p_mean": 1.0}]), codes, grow=True)
    swapped = pipeline.pair_codes(
        record_table([{"delta_p_mean": 1.0, "molecule_a_key": "KEY-B",
                       "molecule_b_key": "KEY-A"}]), codes, grow=True)
    assert forward[0] != swapped[0]


def test_unseen_pair_column_is_false_for_a_pair_seen_in_train() -> None:
    codes: dict[str, int] = {}
    train = record_table([{"delta_p_mean": 1.0}])
    train_codes = np.unique(pipeline.pair_codes(train, codes, grow=True))
    rows = record_table([{"delta_p_mean": 1.0, "partition": "test"},
                         {"delta_p_mean": 1.0, "partition": "test",
                          "molecule_b_key": "KEY-C"}])
    assert pipeline.unseen_pair_column(rows, codes, train_codes).to_pylist() == [False, True]


def test_unseen_pair_column_without_any_train_pair() -> None:
    rows = record_table([{"delta_p_mean": 1.0, "partition": "test"}])
    empty = np.zeros(0, dtype=np.int64)
    assert pipeline.unseen_pair_column(rows, {}, empty).to_pylist() == [True]


def write_deltas(path: Path, rows: list[tuple[str, float, float]]) -> Path:
    table = pa.table({"pair_id": pa.array([r[0] for r in rows], pa.string()),
                      "p_a": pa.array([r[1] for r in rows], pa.float64()),
                      "p_b": pa.array([r[2] for r in rows], pa.float64())})
    pq.write_table(table, path)
    return path


def test_value_bounds_takes_the_widest_range_over_all_assays(tmp_path: Path) -> None:
    path = write_deltas(tmp_path / "b00.parquet",
                        [("p1", 6.0, 7.0), ("p1", 5.0, 9.0), ("p2", 3.0, 3.5)])
    lo, hi = pipeline.value_bounds(path, pa.array(["p2", "p1"]))
    assert lo.tolist() == [3.0, 5.0]
    assert hi.tolist() == [3.5, 9.0]


def test_value_bounds_refuses_a_record_without_any_assay_delta(tmp_path: Path) -> None:
    path = write_deltas(tmp_path / "b00.parquet", [("p1", 6.0, 7.0)])
    with pytest.raises(ValueError, match="pair_id"):
        pipeline.value_bounds(path, pa.array(["p9"]))


# ------------------------------------------------------------------- cảnh báo chất lượng nguồn


def write_warning_deltas(path: Path, rows: list[tuple[str, list[str], list[str]]]) -> Path:
    table = pa.table({
        "pair_id": pa.array([r[0] for r in rows], pa.string()),
        "observation_ids_a": pa.array([r[1] for r in rows], pa.list_(pa.string())),
        "observation_ids_b": pa.array([r[2] for r in rows], pa.list_(pa.string())),
    })
    pq.write_table(table, path)
    return path


def test_warning_column_gathers_both_sides_without_repeating_a_value(tmp_path: Path) -> None:
    path = write_warning_deltas(tmp_path / "b00.parquet",
                                [("p1", ["o1"], ["o2"]), ("p1", ["o3"], ["o9"]),
                                 ("p2", ["o9"], ["o9"])])
    comments = {"o1": "Outside typical range", "o2": "Potential author error",
                "o3": "Outside typical range"}
    column = pipeline.warning_column(path, pa.array(["p1", "p2"]), comments, "; ")
    # 'Outside typical range' có ở cả hai dòng của p1 nhưng chỉ được kể một lần, và thứ tự
    # đã sắp xếp nên giá trị không phụ thuộc thứ tự dòng.
    assert column.to_pylist() == ["Outside typical range; Potential author error", None]


def test_warning_column_is_null_when_no_measurement_carries_a_warning(tmp_path: Path) -> None:
    path = write_warning_deltas(tmp_path / "b00.parquet", [("p1", ["o1"], ["o2"])])
    column = pipeline.warning_column(path, pa.array(["p1"]), {"o7": "Potential author error"},
                                     "; ")
    assert column.to_pylist() == [None]


def test_warning_column_skips_reading_when_the_source_has_no_warning(tmp_path: Path) -> None:
    missing = tmp_path / "khong-ton-tai.parquet"
    assert pipeline.warning_column(missing, pa.array(["p1", "p2"]), {}, "; ").to_pylist() == [
        None, None]


def test_observation_hits_returns_nothing_when_no_id_matches() -> None:
    column = pa.chunked_array([pa.array([["o1"], ["o2"]], pa.list_(pa.string()))])
    rows, which = pipeline.observation_hits(column, pa.array(["o9"], pa.string()))
    assert rows.tolist() == [] and which.tolist() == []


def test_observation_hits_maps_each_hit_to_its_row() -> None:
    column = pa.chunked_array([pa.array([["o1", "o2"], [], ["o3"]], pa.list_(pa.string()))])
    rows, which = pipeline.observation_hits(column, pa.array(["o3", "o1"], pa.string()))
    assert rows.tolist() == [0, 2] and which.tolist() == [1, 0]


def test_observation_hits_refuses_a_null_list() -> None:
    column = pa.chunked_array([pa.array([["o1"], None], pa.list_(pa.string()))])
    with pytest.raises(ValueError, match="null"):
        pipeline.observation_hits(column, pa.array(["o1"], pa.string()))


def test_source_quality_comments_keeps_only_rows_whose_key_has_a_value(
        tmp_path: Path, config: dict[str, Any]) -> None:
    names = config["stage2b"]["columns"]
    table = pa.table({
        names["observation_id"]: pa.array(["o1", "o2", "o3", "o4"], pa.string()),
        names["activity_context"]: pa.array([
            '{"data_validity_comment": "Outside typical range"}',
            '{"data_validity_comment": null}',
            '{"potential_duplicate": 0}',
            '{"data_validity_comment": "Potential author error"}',
        ], pa.string()),
    })
    pq.write_table(table, tmp_path / "src.parquet")
    local = {"stage2b": dict(config["stage2b"], inputs={"chembl": "src.parquet"})}
    found = pipeline.source_quality_comments(local, tmp_path, config["stage2b"]["export"])
    assert found == {"o1": "Outside typical range", "o4": "Potential author error"}


def test_generator_rows_carry_the_source_quality_warning() -> None:
    rows = record_table([{"delta_p_mean": 1.0,
                          "source_quality_warning": "Outside typical range"}])
    table = pipeline.oriented_generator_rows(rows, 0.3)
    assert table.column("source_quality_warning").to_pylist() == ["Outside typical range"]


# ------------------------------------------------------------------- tài liệu độc lập (bước 4)


def test_normalize_patent_matches_both_source_spellings(config: dict[str, Any]) -> None:
    dedup = config["stage2a"]["dedup"]
    values = pa.array(["US-9321727-B2", "US9321727", "us 9321727 b1", None])
    normalized = pipeline.normalize_patent(values, dedup["patent_strip_pattern"],
                                           dedup["patent_kind_code_pattern"],
                                           dedup["patent_kind_code_replacement"])
    assert normalized.to_pylist() == ["US9321727", "US9321727", "US9321727", None]


def test_normalize_doi_drops_url_prefix_and_case(config: dict[str, Any]) -> None:
    pattern = config["stage2a"]["dedup"]["doi_prefix_pattern"]
    values = pa.array(["https://doi.org/10.1021/JM001", "10.1021/jm001", None])
    assert pipeline.normalize_doi(values, pattern).to_pylist() == [
        "10.1021/jm001", "10.1021/jm001", None]


def document_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.table({role: pa.array([row.get(role) for row in rows], pa.string())
                     for role in pipeline.DOCUMENT_ROLES})


def test_document_identifiers_prefers_pmid_doi_patent_over_source_id(
        config: dict[str, Any]) -> None:
    table = document_table([
        {"source": "chembl", "document_pmid": "123", "document_doi": "10.1/X",
         "document_id": "CHEMBL1"},
        {"source": "bindingdb", "document_patent": "US-1234567-B2", "document_id": "9"},
        {"source": "chembl", "document_id": "CHEMBL2"},
        {"source": "bindingdb"},
    ])
    found = pipeline.document_identifiers(table, config["stage2a"]["dedup"])
    assert found == [{"pmid:123", "doi:10.1/x"}, {"patent:US1234567"},
                     {"doc:chembl:CHEMBL2"}, set()]


def test_assay_documents_merges_rows_of_one_assay(tmp_path: Path,
                                                  config: dict[str, Any]) -> None:
    names = config["stage2b"]["columns"]
    rows = {"source": ["chembl", "chembl", "bindingdb"], "assay_id": ["A1", "A1", "7"],
            "assay_entry_id": [None, None, "50"], "document_pmid": ["1", None, None],
            "document_doi": [None, "10.1/x", None], "document_patent": [None, None, None],
            "document_id": ["D1", "D1", None]}
    pq.write_table(pa.table({names[role]: pa.array(values, pa.string())
                             for role, values in rows.items()}), tmp_path / "src.parquet")
    local = dict(config, stage2b=dict(config["stage2b"], inputs={"chembl": "src.parquet"}))
    documents = pipeline.assay_documents(local, tmp_path)
    assert documents == {"chembl::A1": frozenset({"pmid:1", "doi:10.1/x"}),
                         "bindingdb:50:7": frozenset()}


# ------------------------------------------------------------------- cờ của ChEMBL (bước 5)


def test_chembl_observation_info_reads_confidence_and_inactive_comments(
        tmp_path: Path, config: dict[str, Any]) -> None:
    names = config["stage2b"]["columns"]
    table = pa.table({
        names["observation_id"]: pa.array(["o1", "o2", "o3", "o4", "o5"], pa.string()),
        names["assay_context"]: pa.array([
            '{"confidence_score": 9}', '{"confidence_score": 8}',
            '{"confidence_score": null}', '{"confidence_score": "7"}', None], pa.string()),
        names["activity_context"]: pa.array([
            '{"activity_comment": "Not Active"}', '{"activity_comment": "active"}',
            '{"activity_comment": null}', None, '{"activity_comment": "inconclusive "}'],
            pa.string()),
    })
    pq.write_table(table, tmp_path / "src.parquet")
    local = {"stage2b": dict(config["stage2b"], inputs={"chembl": "src.parquet"})}
    export = config["stage2b"]["export"]
    confidence, inactive = pipeline.chembl_observation_info(local, tmp_path, export)
    assert confidence == {"o2": 8, "o3": export["missing_confidence"], "o4": 7}
    assert inactive == {"o1", "o5"}


def test_observation_flag_columns_by_source(tmp_path: Path) -> None:
    path = write_warning_deltas(tmp_path / "b00.parquet",
                                [("p1", ["o1"], ["o2"]), ("p1", ["o3"], ["o4"]),
                                 ("p2", ["o5"], ["o6"]), ("p3", ["o7"], ["o8"])])
    records = pa.table({
        "pair_id": pa.array(["p1", "p2", "p3"], pa.string()),
        "assay_keys": pa.array([["chembl::A1", "chembl::A2"], ["chembl::A3"],
                                ["bindingdb:5:1"]], pa.list_(pa.string())),
    })
    minimum, inactive = pipeline.observation_flag_columns(
        path, records, {"o2": 8, "o4": 0}, {"o5"}, 9, "chembl:")
    # p1: mức nhỏ nhất của các phép đo đóng góp; p2: ChEMBL không có mức thấp -> 9;
    # p3: chỉ BindingDB -> null.
    assert minimum.to_pylist() == [0, 9, None]
    assert inactive.to_pylist() == [False, True, False]


def test_excluded_type_keys_flags_any_excluded_label_and_counts_conflicts() -> None:
    pairs = pa.table({"inchikey": ["K1", "K2", "K2", "K3", "K4"],
                      "molecule_chembl_type": ["Protein", "Protein", "Small molecule",
                                               "Small molecule", "Unknown"]})
    flagged, stats = pipeline.excluded_type_keys(pairs, ["Protein", "Oligonucleotide"])
    # K2 mang hai nhãn mâu thuẫn: bị loại (thận trọng) và được đếm riêng.
    assert flagged == {"K1", "K2"}
    assert stats["excluded_molecules"] == 2 and stats["conflicting_molecules"] == 1
    assert stats["molecules_by_type"] == {"Protein": 2, "Small molecule": 2, "Unknown": 1}


def test_excluded_type_column_ignores_the_relative_stereo_suffix() -> None:
    records = record_table([
        {"delta_p_mean": 1.0, "molecule_a_key": "K1" + SUFFIX, "molecule_b_key": "K9"},
        {"delta_p_mean": 1.0, "molecule_a_key": "K8", "molecule_b_key": "K1"},
        {"delta_p_mean": 1.0, "molecule_a_key": "K8", "molecule_b_key": "K9"}])
    flags = pipeline.excluded_type_column(records, {"K1"}, SUFFIX)
    assert flags.to_pylist() == [True, True, False]


def test_strip_suffix_only_removes_a_trailing_suffix() -> None:
    assert pipeline.strip_suffix("AAA-BBB-N" + SUFFIX, SUFFIX) == "AAA-BBB-N"
    assert pipeline.strip_suffix("AAA-BBB-N", SUFFIX) == "AAA-BBB-N"


def test_chembl_type_pairs_reads_distinct_labelled_rows(tmp_path: Path,
                                                        config: dict[str, Any]) -> None:
    names = config["stage2b"]["columns"]
    table = pa.table({names["inchikey"]: ["K1", "K1", "K2", "K3", None],
                      names["molecule_chembl_type"]: ["Protein", "Protein", None,
                                                      "Small molecule", "Protein"]})
    pq.write_table(table, tmp_path / "src.parquet")
    local = {"stage2b": dict(config["stage2b"], inputs={"chembl": "src.parquet",
                                                        "bindingdb": "src.parquet"})}
    pairs = pipeline.chembl_type_pairs(local, tmp_path)
    assert sorted(zip(pairs.column("inchikey").to_pylist(),
                      pairs.column("molecule_chembl_type").to_pylist())) == [
        ("K1", "Protein"), ("K3", "Small molecule")]


def assay_type_source(path: Path, config: dict[str, Any], types: list[str | None]) -> None:
    names = config["stage2b"]["columns"]
    rows = len(types)
    pq.write_table(pa.table({
        names["source"]: ["chembl"] * (rows - 1) + ["bindingdb"],
        names["assay_id"]: ["A1"] * (rows - 1) + ["7"],
        names["assay_entry_id"]: [None] * (rows - 1) + ["50"],
        names["assay_type"]: pa.array(types, pa.string())}), path)


def test_assay_type_map_keeps_typed_assays_only(tmp_path: Path,
                                                config: dict[str, Any]) -> None:
    assay_type_source(tmp_path / "src.parquet", config, ["B", "B", None])
    local = dict(config, stage2b=dict(config["stage2b"], inputs={"chembl": "src.parquet"}))
    assert pipeline.assay_type_map(local, tmp_path) == {"chembl::A1": "B"}


def test_assay_type_map_refuses_an_assay_with_two_types(tmp_path: Path,
                                                        config: dict[str, Any]) -> None:
    assay_type_source(tmp_path / "src.parquet", config, ["B", "F", None])
    local = dict(config, stage2b=dict(config["stage2b"], inputs={"chembl": "src.parquet"}))
    with pytest.raises(ValueError, match="hai loại"):
        pipeline.assay_type_map(local, tmp_path)


def test_assay_type_columns_marks_mixed_binding_and_functional() -> None:
    records = pa.table({"assay_keys": pa.array(
        [["chembl::A1", "chembl::A2"], ["chembl::A1", "chembl::A3"],
         ["bindingdb:5:1"], ["chembl::A2", "bindingdb:5:1"]], pa.list_(pa.string()))})
    types = {"chembl::A1": "F", "chembl::A2": "B", "chembl::A3": "F"}
    joined, mixed = pipeline.assay_type_columns(records, types, ";")
    assert joined.to_pylist() == ["B;F", "F", None, "B"]
    assert mixed.to_pylist() == [True, False, False, False]


def test_scope_and_flag_columns_do_not_overlap() -> None:
    assert not set(pipeline.SCOPE_COLUMNS) & set(pipeline.FLAG_COLUMNS)
    assert not set(pipeline.SCOPE_COLUMNS) & set(pipeline.EVALUATOR_COLUMNS)


def test_nonstandard_proteins_ignores_case_and_whitespace() -> None:
    proteins = pa.table({"protein_hash": ["h1", "h2", "h3", "h4"],
                         "protein_sequence": ["mktay", "MKT AY\n", "MKT1AY",
                                              "biotin-C6linker-MKT"]})
    alphabet = "ACDEFGHIKLMNPQRSTVWYBXZUO"
    assert pipeline.nonstandard_proteins(proteins, alphabet) == {"h3", "h4"}


# --------------------------------------------------------------------------- split trong export


def test_partition_column_reports_a_protein_missing_from_the_split() -> None:
    records = record_table([{"delta_p_mean": 1.0}])
    assignment = pa.table({"protein_sequence_hash": [HASH_B], "partition": ["train"]})
    with pytest.raises(ValueError, match="protein_split"):
        pipeline.partition_column(records, assignment)


def test_partition_column_reads_the_partition_of_each_protein() -> None:
    records = record_table([{"delta_p_mean": 1.0}, {"delta_p_mean": 1.0,
                                                    "protein_hash": HASH_B}])
    assignment = pa.table({"protein_sequence_hash": [HASH_A, HASH_B],
                           "partition": ["train", "test"]})
    assert pipeline.partition_column(records, assignment).to_pylist() == ["train", "test"]


# --------------------------------------------------------------------------- verify, split


def stored_records(config: dict[str, Any]) -> tuple[pa.Table, dict[str, frozenset[str]]]:
    """Bản ghi do chính aggregate_deltas tạo, và bảng tài liệu dùng cho nó."""
    documents = {"s:e:1": frozenset({"pmid:1"}), "s:e:2": frozenset({"pmid:2"}),
                 "s:e:3": frozenset({"pmid:1"})}
    rows = [("KEY-B", "s:e:1", 1.0), ("KEY-B", "s:e:2", 1.0), ("KEY-B", "s:e:3", 2.5),
            ("KEY-C", "s:e:1", 0.4)]
    deltas = pa.table({
        "protein_hash": pa.array([HASH_A] * len(rows), pa.string()),
        "molecule_a_key": pa.array(["KEY-A"] * len(rows), pa.string()),
        "molecule_b_key": pa.array([row[0] for row in rows], pa.string()),
        "assay_key": pa.array([row[1] for row in rows], pa.string()),
        "delta_p": pa.array([row[2] for row in rows], pa.float64()),
        "p_a": pa.array([6.0] * len(rows), pa.float64()),
        "p_std_a": pa.array([None] * len(rows), pa.float64()),
        "p_std_b": pa.array([None] * len(rows), pa.float64()),
        "independent": pa.array([True] * len(rows), pa.bool_()),
        **{name: pa.array([name] * len(rows), pa.string())
           for name in aggregate.RECORD_CARRIED},
    })
    qc = config["stage2b"]["qc"]
    records = aggregate.aggregate_deltas(deltas, qc["dispersion_thresholds"]["Ki"],
                                         qc["dispersion_ddof"], qc["collapse_tolerance"],
                                         config["stage2b"]["replicate"]["float_slack"],
                                         documents)
    return records, documents


def recompute(records: pa.Table, documents: dict[str, frozenset[str]],
              config: dict[str, Any]) -> list[bool]:
    stage = config["stage2b"]
    return pipeline.recompute_mismatch(
        records.select(list(pipeline.RECOMPUTED_COLUMNS)),
        stage["qc"]["dispersion_thresholds"]["Ki"], stage["qc"]["dispersion_ddof"],
        stage["qc"]["collapse_tolerance"], stage["replicate"]["float_slack"],
        stage["run"]["verify_tolerance"], documents).tolist()


def test_recompute_mismatch_accepts_records_written_by_step_four(
        config: dict[str, Any]) -> None:
    records, documents = stored_records(config)
    assert recompute(records, documents, config) == [False, False]


def test_recompute_mismatch_catches_a_wrong_document_count(config: dict[str, Any]) -> None:
    records, documents = stored_records(config)
    index = records.schema.get_field_index("n_documents")
    wrong = records.set_column(index, "n_documents",
                               pa.array([9, 1], pa.int64()))
    assert recompute(wrong, documents, config) == [True, False]


def test_recompute_mismatch_catches_a_wrong_status(config: dict[str, Any]) -> None:
    records, documents = stored_records(config)
    index = records.schema.get_field_index("qc_status")
    # Bản ghi đầu đúng là quarantine (hai giá trị 1,0 và 2,5); bản ghi sau đúng là
    # single_assay nên ghi 'pass' là sai.
    wrong = records.set_column(index, "qc_status",
                               pa.array(["quarantine", "pass"], pa.string()))
    assert recompute(wrong, documents, config) == [False, True]


def test_stored_records_fixture_has_the_expected_meaning(config: dict[str, Any]) -> None:
    records, _ = stored_records(config)
    first, second = records.to_pylist()
    # 1,0 xuất hiện ở hai assay nhưng chỉ tính một lần; s:e:1 và s:e:3 chung PMID.
    assert (first["n_assays"], first["n_distinct_values"], first["n_documents"]) == (3, 2, 2)
    assert first["qc_status"] == aggregate.QC_QUARANTINE
    assert second["qc_status"] == aggregate.QC_SINGLE


def test_close_or_both_nan_accepts_two_nans() -> None:
    left = np.array([1.0, np.nan, 1.0])
    right = np.array([1.0 + 1e-12, np.nan, 2.0])
    assert pipeline.close_or_both_nan(left, right, 1e-9).tolist() == [True, True, False]


def test_list_offsets_rebases_the_offsets() -> None:
    column = pa.chunked_array([pa.array([["a", "b"], ["c"]])])
    offsets, values = pipeline.list_offsets(column)
    assert offsets.tolist() == [0, 2, 3]
    assert values.to_pylist() == ["a", "b", "c"]


def test_load_cluster_tsv_reads_representative_and_member(tmp_path: Path) -> None:
    path = tmp_path / "clusters.tsv"
    path.write_text("r1\taa\nr1\tbb\n\nr2\tcc\n", encoding="utf-8")
    assert pipeline.load_cluster_tsv(path) == {"aa": "r1", "bb": "r1", "cc": "r2"}


def test_load_cluster_tsv_refuses_a_line_without_two_columns(tmp_path: Path) -> None:
    path = tmp_path / "clusters.tsv"
    path.write_text("r1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hai cột"):
        pipeline.load_cluster_tsv(path)


def test_load_cluster_tsv_refuses_a_protein_in_two_clusters(tmp_path: Path) -> None:
    path = tmp_path / "clusters.tsv"
    path.write_text("r1\taa\nr2\taa\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hai nhóm"):
        pipeline.load_cluster_tsv(path)


def test_layout_builds_bucket_paths(tmp_path: Path, settings: dict[str, Any]) -> None:
    layout = pipeline.Layout(tmp_path, settings["files"])
    assert layout.file("molecules").name == "molecules.parquet"
    path = layout.bucket_file("records_dir", "Ki", 3)
    assert path.parent.name == "Ki" and path.name == "b03.parquet"


def test_layout_lists_only_existing_bucket_files(tmp_path: Path,
                                                 settings: dict[str, Any]) -> None:
    layout = pipeline.Layout(tmp_path, settings["files"])
    assert layout.bucket_files("records_dir", "Ki") == []
    path = layout.bucket_file("records_dir", "Ki", 1)
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"x": [1]}), path)
    assert layout.bucket_files("records_dir", "Ki") == [path]
