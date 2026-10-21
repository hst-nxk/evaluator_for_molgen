"""Logic thuần của src/molopt/dataset/pipeline.py (không đọc dữ liệu thật)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molopt.dataset import pipeline

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
        "qc_status": pa.array([row.get("qc_status", "pass") for row in rows], pa.string()),
        "molecule_a_key": pa.array([row.get("molecule_a_key", "KEY-A") for row in rows],
                                   pa.string()),
        "molecule_b_key": pa.array([row.get("molecule_b_key", "KEY-B") for row in rows],
                                   pa.string()),
        "unseen_pair": pa.array([row.get("unseen_pair", False) for row in rows], pa.bool_()),
        "plausible_range": pa.array([row.get("plausible_range", True) for row in rows],
                                    pa.bool_()),
        "short_protein": pa.array([row.get("short_protein", False) for row in rows], pa.bool_()),
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
