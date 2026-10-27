"""Logic thuần của gói molopt.audit, trên dữ liệu tự tạo (không đọc dữ liệu thật)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molopt.audit import common, duplicates, molecules, stats, thresholds

PEPTIDE_SMARTS = "[NX3][CX4][CX3](=O)[NX3][CX4][CX3](=O)"
REPORT_LABELS = ("p1", "p25", "p50", "p75", "p95", "p99", "p99.9", "p100")


# --------------------------------------------------------------------------- common


def test_group_ids_equal_rows_share_id() -> None:
    first = np.array([1, 1, 2, 1, 2])
    second = np.array([5, 5, 5, 6, 5])
    ids = common.group_ids([first, second])
    assert ids[0] == ids[1]
    assert ids[2] == ids[4]
    assert len({ids[0], ids[2], ids[3]}) == 3
    assert sorted(set(ids.tolist())) == [0, 1, 2]


def test_group_ids_empty_and_no_columns() -> None:
    assert common.group_ids([np.zeros(0, dtype=np.int64)]).size == 0
    with pytest.raises(ValueError):
        common.group_ids([])


def test_distinct_per_group() -> None:
    ids = np.array([0, 0, 0, 1, 1, 2])
    values = np.array([7, 7, 8, 3, 3, 9])
    assert common.distinct_per_group(ids, values).tolist() == [2, 1, 1]


def test_encode_maps_null_to_null_code() -> None:
    codes = common.encode(pa.chunked_array([["a", None], ["a", "b"]]))
    assert codes[0] == codes[2]
    assert codes[1] == common.NULL_CODE
    assert codes[3] not in (codes[0], common.NULL_CODE)


def test_boolean_null_is_false() -> None:
    assert common.boolean(pa.chunked_array([[True, None], [False]])).tolist() == [
        True, False, False]


def test_bucket_counts_includes_below_and_open_end() -> None:
    rows = common.bucket_counts(np.array([-1.0, 0.0, 0.5, 1.0, 5.0, np.nan]), [0, 1])
    assert [row["range"] for row in rows] == ["< 0", "[0, 1)", ">= 1"]
    assert [row["count"] for row in rows] == [1, 2, 2]


def test_quantile_summary_labels_and_empty() -> None:
    summary = common.quantile_summary(np.array([1.0, 2.0, 3.0, np.nan]), [0.5, 0.999])
    assert summary["n"] == 3
    assert summary["p50"] == pytest.approx(2.0)
    assert "p99.9" in summary
    empty = common.quantile_summary(np.zeros(0), [0.5])
    assert empty["n"] == 0 and empty["p50"] is None


def test_number_format() -> None:
    assert common.number(1234567) == "1.234.567"
    assert common.number(0.5) == "0,5000"
    assert common.number(None) == "-"


def test_write_report(tmp_path: Path) -> None:
    json_path, text_path = common.write_report(
        tmp_path, "x", {"value": np.int64(3), "rate": np.float64(np.nan)}, ["dòng"])
    assert '"value": 3' in json_path.read_text(encoding="utf-8")
    assert '"rate": null' in json_path.read_text(encoding="utf-8")
    assert text_path.read_text(encoding="utf-8") == "dòng\n"


# --------------------------------------------------------------------------- duplicates


def test_classify_groups_by_source_assay_and_document() -> None:
    # nhóm 0: ChEMBL + BindingDB; nhóm 1: BindingDB, cùng assay; nhóm 2: BindingDB, hai
    # assay cùng tài liệu; nhóm 3: BindingDB, hai tài liệu; nhóm 4: một dòng.
    ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4])
    is_chembl = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    assays = np.array([1, 2, 3, 3, 4, 5, 6, 7, 8])
    documents = np.array([1, 1, 2, 2, 3, 3, 4, 5, 6])
    result = duplicates.classify_groups(ids, is_chembl, assays, documents)
    assert result["groups"] == 5
    assert result["duplicated"]["groups"] == 4
    assert result["duplicated"]["extra_rows"] == 4
    assert result["cross_source"] == {"groups": 1, "rows": 2, "extra_rows": 1,
                                      "chembl_rows": 1, "bindingdb_rows": 1}
    only = result["bindingdb_only"]
    assert only["all"]["groups"] == 3
    assert only["same_assay"]["groups"] == 1
    assert only["one_document_assays"]["groups"] == 1
    assert only["many_documents"]["groups"] == 1
    assert result["chembl_only"]["all"]["groups"] == 0


def test_twin_counts() -> None:
    ids = np.array([0, 0, 1, 2, 2])
    is_chembl = np.array([1, 0, 0, 1, 0], dtype=bool)
    targets = np.array([0, 1, 1, 0, 1], dtype=bool)
    round_values = np.array([0, 1, 0, 0, 0], dtype=bool)
    assert duplicates.twin_counts(ids, is_chembl, targets, round_values) == {
        "targets": 3, "with_chembl_twin": 2, "with_twin_round_value": 1, "without_twin": 1}


def test_level_rows_skips_null_and_rejects_unknown_column() -> None:
    valid = {"inchikey": np.array([1, 1, 0], dtype=bool),
             "value": np.array([1, 0, 1], dtype=bool)}
    subset = np.ones(3, dtype=bool)
    assert duplicates.level_rows({"columns": ["inchikey", "value"]}, subset,
                                 valid).tolist() == [0]
    with pytest.raises(KeyError):
        duplicates.level_rows({"columns": ["unknown"]}, subset, valid)


def test_sample_groups_returns_all_rows_of_group() -> None:
    ids = np.array([2, 0, 2, 1])
    picked = duplicates.sample_groups(ids, np.array([False, False, True]), 5,
                                      np.random.default_rng(0))
    assert len(picked) == 1
    assert sorted(picked[0].tolist()) == [0, 2]


def test_dataset_distinctness(tmp_path: Path) -> None:
    table = pa.table({
        "pair_id": ["p1", "p2", "p2", "p4", "p5"],
        "smiles_a": ["A", "B", "B", "A", "C"],
        "smiles_b": ["B", "A", "A", "B", "C"],
        "protein_hash": ["h", "h", "h", "h", "h"],
        "endpoint": ["Ki", "Ki", "Ki", "Ki", "Ki"],
    })
    path = tmp_path / "evaluator.parquet"
    pq.write_table(table, path)
    result = duplicates.dataset_distinctness(path)
    assert result["rows"] == 5
    assert result["duplicate_pair_id_rows"] == 1
    # (A, B) hai lần, (B, A) hai lần.
    assert result["duplicate_pair_rows"] == 2
    # bốn dòng A-B/B-A, cộng dòng C-C tự khớp với chiều ngược của chính nó.
    assert result["both_directions_rows"] == 5
    assert result["same_molecule_both_sides"] == 1


# --------------------------------------------------------------------------- thresholds


def evidence(n_assays: list[int], n_distinct: list[int],
             n_documents: list[int]) -> dict[str, np.ndarray]:
    return {"n_assays": np.array(n_assays), "n_distinct_values": np.array(n_distinct),
            "n_documents": np.array(n_documents)}


def test_identical_summary() -> None:
    data = evidence([1, 2, 2, 3, 7], [1, 1, 2, 1, 1], [1, 1, 2, 2, 1])
    result = thresholds.identical_summary(data, np.ones(5, dtype=bool), [2, 3, 6])
    assert result["multi_assay"] == 4
    assert result["identical"] == 3
    assert result["identical_one_document"] == 2
    assert result["identical_many_documents"] == 1
    assert [row["n_assays"] for row in result["by_n_assays"]] == ["2-2", "3-5", ">= 6"]
    assert [row["identical"] for row in result["by_n_assays"]] == [1, 1, 1]


LEVELS = {"within_error": 0.3, "meaningful": 1.0, "unit_copy": 3.0}


def test_pairs_within_counts_per_record() -> None:
    # bản ghi 0: 1,0 / 1,2 / 2,5; bản ghi 1: 0,0 / 3,0
    flat = np.array([1.0, 2.5, 1.2, 0.0, 3.0])
    owners = np.array([0, 0, 0, 1, 1])
    assert thresholds.pairs_within(flat, owners, 0.3).tolist() == [1, 0]
    assert thresholds.pairs_within(flat, owners, 1.5).tolist() == [3, 0]
    assert thresholds.pairs_within(flat, owners, 3.0).tolist() == [3, 1]
    assert thresholds.pairs_within(np.zeros(0), np.zeros(0, dtype=np.int64), 1.0).size == 0


def test_pair_level_matches_paper_counting() -> None:
    flat = np.array([1.0, 1.0, 1.2, 0.0, 3.0, 5.0, 5.5])
    owners = np.array([0, 0, 0, 1, 1, 2, 2])
    result = thresholds.pair_level(flat, owners, LEVELS, 0.01)
    # cặp: (1,1) (1,1.2) (1,1.2) | (0,3) | (5,5.5) -> 5 cặp
    assert result["assay_pairs"] == 5
    assert result["identical"] == 1
    assert result["over_within_error"] == 2
    assert result["over_meaningful"] == 1
    assert result["unit_copy"] == 1


def test_record_level_groups_sum_to_multi_assay() -> None:
    data = {"n_assays": np.array([1, 2, 2, 3, 2]),
            "n_distinct_values": np.array([1, 1, 2, 3, 2]),
            "range": np.array([0.0, 0.0, 0.2, 0.8, 2.0])}
    result = thresholds.record_level(data, np.ones(5, dtype=bool), LEVELS)
    assert result["multi_assay"] == 4
    assert [result[key] for key in ("identical", "near_identical", "between",
                                    "over_meaningful")] == [1, 1, 1, 1]


def test_flatten_lists_selected_rows() -> None:
    values = pa.chunked_array([[[1.0, 2.0], [3.0]], [[4.0, 5.0, 6.0]]])
    flat, owners = thresholds.flatten_lists(values, np.array([0, 2]))
    assert flat.tolist() == [1.0, 2.0, 4.0, 5.0, 6.0]
    assert owners.tolist() == [0, 0, 1, 1, 1]


# --------------------------------------------------------------------------- molecules


def test_describe_counts_hydrogens_and_peptide() -> None:
    table = molecules.describe(["CCO", "not a smiles", "NCC(=O)NCC(=O)O", "[*]C"],
                               PEPTIDE_SMARTS)
    heavy, total, _, _, peptide = (molecules.METRICS.index(name) for name in (
        "heavy_atoms", "total_atoms", "mw", "rotatable_bonds", "peptide_like"))
    assert table[0, heavy] == 3 and table[0, total] == 9
    assert np.isnan(table[1]).all()
    assert table[2, peptide] == 1.0 and table[0, peptide] == 0.0
    assert table[3, heavy] == 1


def test_describe_all_matches_describe_single_process() -> None:
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    settings = {"chunk_size": 2, "workers": 1, "peptide_smarts": PEPTIDE_SMARTS}
    np.testing.assert_array_equal(molecules.describe_all(smiles, settings),
                                  molecules.describe(smiles, PEPTIDE_SMARTS))


def test_positions_and_row_metric() -> None:
    unique = pa.array(["x", "y"], type=pa.large_string())
    index = molecules.positions(pa.chunked_array([["y", None, "x"]]), unique)
    assert index.tolist() == [1, common.NULL_CODE, 0]
    values = molecules.row_metric(np.array([[10.0], [20.0]]), index, 0)
    assert values[0] == 20.0 and np.isnan(values[1]) and values[2] == 10.0


def test_cutoff_impact() -> None:
    pair_values = np.array([30.0, 60.0, 80.0, np.nan])
    partitions = np.array(["train", "train", "validation", "test"])
    rows = molecules.cutoff_impact(pair_values, partitions, np.array([30.0, 60.0, 80.0]),
                                   [50, 70], ["train", "validation", "test"])
    assert rows[0]["pairs_removed"] == 2 and rows[0]["molecules_over"] == 2
    assert rows[0]["train"]["removed"] == 1 and rows[0]["test"]["removed"] == 0
    assert rows[1]["pairs_removed"] == 1


# --------------------------------------------------------------------------- stats


def test_per_group_counts() -> None:
    codes = np.array([1, 1, 2, 3])
    groups = np.array([5, 5, 5, 9])
    assert sorted(stats.per_group_counts(codes, groups).tolist()) == [1, 2]
    assert stats.per_group_counts(np.zeros(0), np.zeros(0)).size == 0


def test_value_counts_null_label() -> None:
    result = stats.value_counts(pa.chunked_array([[9, None, 9], [8]]))
    assert result == {"9": 2, "null": 1, "8": 1}


# --------------------------------------------------------------------------- config


def test_audit_config_has_report_quantiles(config: dict[str, Any]) -> None:
    labels = {common.quantile_label(q) for q in config["audit"]["quantiles"]}
    assert set(REPORT_LABELS) <= labels


def test_audit_key_levels_use_known_columns(config: dict[str, Any]) -> None:
    for level in config["audit"]["duplicates"]["key_levels"]:
        assert set(level["columns"]) <= set(duplicates.KEY_COLUMNS)


def test_audit_literature_levels_present(config: dict[str, Any]) -> None:
    literature = config["audit"]["literature"]
    assert set(thresholds.LEVEL_KEYS) <= set(literature)
    assert literature["source"]


def test_audit_peptide_smarts_parses(config: dict[str, Any]) -> None:
    from rdkit import Chem

    assert Chem.MolFromSmarts(config["audit"]["molecules"]["peptide_smarts"]) is not None
