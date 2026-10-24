"""Logic thuần của src/molopt/dataset/aggregate.py (bước 1 và bước 4), trên dữ liệu tự tạo."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from molopt.dataset import aggregate

TOLERANCE = 0.01
SLACK = 1e-9
DDOF = 1


def replicate_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Bảng đầu vào của collapse_replicates; cột thiếu lấy giá trị mặc định dễ đọc."""
    return pa.table({
        "assay_key": pa.array([row.get("assay_key", "s:e:1") for row in rows], pa.string()),
        "source": pa.array([row.get("source", "chembl37") for row in rows], pa.string()),
        "protein_hash": pa.array([row.get("protein_hash", "aa") for row in rows], pa.string()),
        "molecule_key": pa.array([row.get("molecule_key", "KEY-A") for row in rows], pa.string()),
        "observation_id": pa.array([row["observation_id"] for row in rows], pa.string()),
        "source_id": pa.array([row.get("source_id", "M1") for row in rows], pa.string()),
        "p_activity": pa.array([row["p_activity"] for row in rows], pa.float64()),
        "independent": pa.array([row.get("independent", True) for row in rows], pa.bool_()),
    })


def collapse(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate.collapse_replicates(replicate_table(rows), TOLERANCE, SLACK,
                                         DDOF).to_pylist()


# --------------------------------------------------------------------------- tiện ích chung


def test_group_bounds_treats_two_nulls_as_one_value() -> None:
    table = pa.table({"k": pa.array(["a", "a", None, None, "b"], pa.string())})
    assert aggregate.group_bounds(table, ["k"]).tolist() == [0, 2, 4, 5]


def test_group_bounds_of_empty_table() -> None:
    table = pa.table({"k": pa.array([], pa.string())})
    assert aggregate.group_bounds(table, ["k"]).tolist() == [0]


def test_integer_codes_gives_equal_codes_to_equal_values() -> None:
    codes = aggregate.integer_codes(pa.chunked_array([pa.array(["x", "y"]),
                                                      pa.array(["x", None])]))
    assert codes[0] == codes[2] and codes[0] != codes[1] and codes[3] == -1


def test_list_column_splits_values_by_bounds() -> None:
    column = aggregate.list_column(np.array([0, 2, 3], dtype=np.int64),
                                   pa.array(["a", "b", "c"]))
    assert column.to_pylist() == [["a", "b"], ["c"]]


# --------------------------------------------------------------------------- bước 1


def test_merge_values_anchors_cluster_at_first_value() -> None:
    values = np.array([6.000, 6.005, 6.02])
    merged, flags = aggregate.merge_values(values, np.ones(3, dtype=bool), TOLERANCE, SLACK)
    assert merged == pytest.approx([6.0025, 6.02])
    assert flags.tolist() == [True, True]


def test_merge_values_does_not_chain_beyond_tolerance() -> None:
    # 6,016 − 6,000 = 0,016 > 0,01: không được nối vào cụm đầu dù cách 6,008 chỉ 0,008.
    values = np.array([6.000, 6.008, 6.016])
    merged, _ = aggregate.merge_values(values, np.ones(3, dtype=bool), TOLERANCE, SLACK)
    assert merged == pytest.approx([6.004, 6.016])


def test_merge_values_marks_cluster_independent_if_any_row_is() -> None:
    values = np.array([7.0, 7.005, 8.0])
    independent = np.array([False, True, False])
    _, flags = aggregate.merge_values(values, independent, TOLERANCE, SLACK)
    assert flags.tolist() == [True, False]


def test_collapse_replicates_merges_close_values_then_takes_mean() -> None:
    rows = [{"observation_id": "o1", "p_activity": 6.000},
            {"observation_id": "o2", "p_activity": 6.005},
            {"observation_id": "o3", "p_activity": 6.02}]
    (group,) = collapse(rows)
    assert group["n_raw"] == 3
    assert group["n_merged"] == 2
    assert group["p_mean"] == pytest.approx((6.0025 + 6.02) / 2)
    assert group["p_std"] == pytest.approx(np.std([6.0025, 6.02], ddof=DDOF))
    assert group["observation_ids"] == ["o1", "o2", "o3"]


def test_collapse_replicates_single_row_has_no_std() -> None:
    (group,) = collapse([{"observation_id": "o1", "p_activity": 5.5}])
    assert group["p_std"] is None
    assert (group["n_merged"], group["n_independent"], group["n_raw"]) == (1, 1, 1)


def test_collapse_replicates_keeps_flagged_rows_but_not_as_independent() -> None:
    rows = [{"observation_id": "o1", "p_activity": 6.0, "independent": False},
            {"observation_id": "o2", "p_activity": 7.0, "independent": False},
            {"observation_id": "o3", "p_activity": 8.0}]
    (group,) = collapse(rows)
    assert group["n_merged"] == 3
    assert group["n_independent"] == 1
    assert group["p_mean"] == pytest.approx(7.0)


def test_collapse_replicates_flags_multiple_source_ids() -> None:
    rows = [{"observation_id": "o1", "p_activity": 6.0, "source_id": "M1"},
            {"observation_id": "o2", "p_activity": 7.0, "source_id": "M2"}]
    (group,) = collapse(rows)
    assert (group["n_source_ids"], group["multi_source_id"]) == (2, True)


def test_collapse_replicates_single_source_id_is_not_flagged() -> None:
    rows = [{"observation_id": "o1", "p_activity": 6.0},
            {"observation_id": "o2", "p_activity": 7.0}]
    (group,) = collapse(rows)
    assert (group["n_source_ids"], group["multi_source_id"]) == (1, False)


def test_collapse_replicates_separates_assays_and_molecules() -> None:
    rows = [{"observation_id": "o1", "p_activity": 6.0},
            {"observation_id": "o2", "p_activity": 9.0, "assay_key": "s:e:2"},
            {"observation_id": "o3", "p_activity": 8.0, "molecule_key": "KEY-B"}]
    groups = collapse(rows)
    assert len(groups) == 3
    assert {group["p_mean"] for group in groups} == {6.0, 9.0, 8.0}


# --------------------------------------------------------------------------- bước 4


def test_dispersion_uses_std_from_three_assays() -> None:
    sizes = np.array([3])
    spread = aggregate.dispersion(sizes, np.array([0.5]), np.array([0.0]), np.array([2.0]))
    assert spread == pytest.approx([0.5])


def test_dispersion_uses_range_for_two_assays() -> None:
    spread = aggregate.dispersion(np.array([2]), np.array([np.nan]), np.array([1.0]),
                                  np.array([2.5]))
    assert spread == pytest.approx([1.5])


def test_dispersion_is_nan_for_one_assay() -> None:
    spread = aggregate.dispersion(np.array([1]), np.array([np.nan]), np.array([1.0]),
                                  np.array([1.0]))
    assert math.isnan(spread[0])


def test_qc_status_separates_identical_values_from_pass() -> None:
    status = aggregate.qc_status(np.array([1, 2, 2, 3, 2]), np.array([1, 1, 2, 3, 2]),
                                 np.array([np.nan, np.nan, 2.0, 0.1, 1.0]), 1.0)
    assert status.tolist() == [aggregate.QC_SINGLE, aggregate.QC_IDENTICAL,
                              aggregate.QC_QUARANTINE, aggregate.QC_PASS, aggregate.QC_PASS]


def test_collapse_values_sorts_then_merges_copies() -> None:
    merged = aggregate.collapse_values(np.array([6.02, 6.005, 6.0]), TOLERANCE, SLACK)
    assert merged == pytest.approx([6.0025, 6.02])


def test_summarize_values_does_not_count_a_copy_twice() -> None:
    # Hai assay chép cùng giá trị 1,0, một assay đo 2,0: hai giá trị khác nhau, mean 1,5
    # (không phải 4/3), phân tán là hiệu 1,0.
    mean, median, std, low, high, spread, distinct = aggregate.summarize_values(
        np.array([1.0, 2.0, 1.0]), TOLERANCE, SLACK, DDOF)
    assert (mean, median, low, high, spread, distinct) == pytest.approx(
        (1.5, 1.5, 1.0, 2.0, 1.0, 2))
    assert std == pytest.approx(np.std([1.0, 2.0], ddof=DDOF))


def test_summarize_values_of_identical_copies_has_no_spread() -> None:
    mean, _, std, _, _, spread, distinct = aggregate.summarize_values(
        np.array([0.7, 0.7, 0.705]), TOLERANCE, SLACK, DDOF)
    assert distinct == 1
    assert mean == pytest.approx((0.7 + 0.7 + 0.705) / 3)
    assert math.isnan(std) and math.isnan(spread)


def test_document_count_joins_assays_sharing_any_identifier() -> None:
    # a và c không chung gì, nhưng cùng chung với b nên cùng một tài liệu.
    documents = {"a": frozenset({"pmid:1"}), "b": frozenset({"pmid:1", "doi:x"}),
                 "c": frozenset({"doi:x"}), "d": frozenset({"patent:US1"})}
    assert aggregate.document_count(["a", "b", "c"], documents) == 1
    assert aggregate.document_count(["a", "c"], documents) == 2
    assert aggregate.document_count(["a", "b", "c", "d"], documents) == 2


def test_document_count_counts_assays_without_identifier_separately() -> None:
    documents = {"a": frozenset(), "b": frozenset()}
    assert aggregate.document_count(["a", "b"], documents) == 2


def test_document_count_rejects_unknown_assay() -> None:
    with pytest.raises(KeyError):
        aggregate.document_count(["a"], {})


def test_group_max_ignores_nan_and_keeps_all_nan_groups() -> None:
    values = np.array([np.nan, 0.2, np.nan, np.nan, 0.5])
    result = aggregate.group_max(values, np.array([0, 2, 4, 5], dtype=np.int64))
    assert result[0] == pytest.approx(0.2)
    assert math.isnan(result[1])
    assert result[2] == pytest.approx(0.5)


# Mỗi assay của test một tài liệu riêng, trừ khi test tự truyền bảng khác.
DOCUMENTS = {f"s:e:{index}": frozenset({f"pmid:{index}"}) for index in range(1, 6)}


def delta_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Bảng Δp trong assay tối giản cho aggregate_deltas."""
    size = len(rows)
    columns = {
        "protein_hash": pa.array([row.get("protein_hash", "aa") for row in rows], pa.string()),
        "molecule_a_key": pa.array([row.get("a", "KEY-A") for row in rows], pa.string()),
        "molecule_b_key": pa.array([row.get("b", "KEY-B") for row in rows], pa.string()),
        "assay_key": pa.array([row["assay_key"] for row in rows], pa.string()),
        "delta_p": pa.array([row["delta_p"] for row in rows], pa.float64()),
        "p_a": pa.array([row.get("p_a", 6.0) for row in rows], pa.float64()),
        "p_std_a": pa.array([row.get("p_std_a") for row in rows], pa.float64()),
        "p_std_b": pa.array([row.get("p_std_b") for row in rows], pa.float64()),
        "independent": pa.array([row.get("independent", True) for row in rows], pa.bool_()),
    }
    for name in aggregate.RECORD_CARRIED:
        columns[name] = pa.array([name] * size, pa.string())
    return pa.table(columns)


def aggregate_rows(rows: list[dict[str, Any]], threshold: float = 1.0,
                   documents: dict[str, frozenset[str]] | None = None) -> list[dict[str, Any]]:
    return aggregate.aggregate_deltas(delta_table(rows), threshold, DDOF, TOLERANCE, SLACK,
                                      DOCUMENTS if documents is None else documents
                                      ).to_pylist()


def test_aggregate_deltas_summarises_each_record() -> None:
    rows = [{"assay_key": "s:e:1", "delta_p": 1.0},
            {"assay_key": "s:e:2", "delta_p": 2.0},
            {"assay_key": "s:e:3", "delta_p": 0.0, "independent": False}]
    (record,) = aggregate_rows(rows)
    assert record["n_assays"] == 3
    assert record["n_independent_assays"] == 2
    assert record["n_distinct_values"] == 3
    assert record["n_documents"] == 3
    assert record["delta_p_mean"] == pytest.approx(1.0)
    assert record["delta_p_median"] == pytest.approx(1.0)
    assert record["delta_p_std"] == pytest.approx(1.0)
    assert record["delta_p_min"] == pytest.approx(0.0)
    assert record["delta_p_max"] == pytest.approx(2.0)
    assert record["delta_p_values"] == [1.0, 2.0, 0.0]
    assert record["assay_keys"] == ["s:e:1", "s:e:2", "s:e:3"]
    assert record["qc_status"] == aggregate.QC_PASS


def test_aggregate_deltas_marks_copied_values_as_identical() -> None:
    # Cùng Δp ở hai tài liệu nhưng p(A) lệch đúng 3 log: dấu hiệu bản sao sai đơn vị.
    rows = [{"assay_key": "s:e:1", "delta_p": 0.5, "p_a": 5.0},
            {"assay_key": "s:e:2", "delta_p": 0.5, "p_a": 8.0}]
    (record,) = aggregate_rows(rows)
    assert record["qc_status"] == aggregate.QC_IDENTICAL
    assert record["n_distinct_values"] == 1
    assert record["n_documents"] == 2
    assert record["delta_p_mean"] == pytest.approx(0.5)
    assert record["delta_p_std"] is None and record["dispersion"] is None
    assert record["p_a_range"] == pytest.approx(3.0)


def test_aggregate_deltas_mean_is_not_pulled_by_a_copy() -> None:
    rows = [{"assay_key": "s:e:1", "delta_p": 1.0}, {"assay_key": "s:e:2", "delta_p": 1.0},
            {"assay_key": "s:e:3", "delta_p": 2.0}]
    (record,) = aggregate_rows(rows)
    assert record["delta_p_mean"] == pytest.approx(1.5)
    assert record["dispersion"] == pytest.approx(1.0)
    assert record["qc_status"] == aggregate.QC_PASS


def test_aggregate_deltas_counts_shared_document_once() -> None:
    documents = {"s:e:1": frozenset({"pmid:1"}), "s:e:2": frozenset({"pmid:1"})}
    rows = [{"assay_key": "s:e:1", "delta_p": 1.0}, {"assay_key": "s:e:2", "delta_p": 1.2}]
    (record,) = aggregate_rows(rows, documents=documents)
    assert record["n_documents"] == 1
    assert record["n_distinct_values"] == 2


def test_aggregate_deltas_keeps_largest_replicate_std() -> None:
    rows = [{"assay_key": "s:e:1", "delta_p": 1.0, "p_std_b": 0.1},
            {"assay_key": "s:e:2", "delta_p": 1.2, "p_std_a": 0.3},
            {"assay_key": "s:e:3", "delta_p": 1.0, "b": "KEY-C"}]
    records = aggregate_rows(rows)
    by_b = {record["molecule_b_key"]: record for record in records}
    assert by_b["KEY-B"]["max_replicate_std"] == pytest.approx(0.3)
    assert by_b["KEY-C"]["max_replicate_std"] is None


def test_aggregate_deltas_quarantines_dispersed_two_assay_record() -> None:
    rows = [{"assay_key": "s:e:1", "delta_p": 0.0}, {"assay_key": "s:e:2", "delta_p": 1.5}]
    (record,) = aggregate_rows(rows)
    assert record["dispersion"] == pytest.approx(1.5)
    assert record["qc_status"] == aggregate.QC_QUARANTINE


def test_aggregate_deltas_keeps_records_apart_by_protein() -> None:
    rows = [{"assay_key": "s:e:1", "delta_p": 1.0},
            {"assay_key": "s:e:1", "delta_p": 2.0, "protein_hash": "bb"}]
    records = aggregate_rows(rows)
    assert [record["n_assays"] for record in records] == [1, 1]
    assert {record["qc_status"] for record in records} == {aggregate.QC_SINGLE}
    assert [record["p_a_range"] for record in records] == [0.0, 0.0]


def test_aggregate_deltas_rejects_assay_missing_from_documents() -> None:
    with pytest.raises(KeyError):
        aggregate_rows([{"assay_key": "s:e:9", "delta_p": 1.0}])


def test_aggregate_deltas_of_empty_input_returns_schema() -> None:
    empty = aggregate.aggregate_deltas(delta_table([]), 1.0, DDOF, TOLERANCE, SLACK, {})
    assert empty.num_rows == 0
    assert empty.schema == aggregate.RECORD_SCHEMA
