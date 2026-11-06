"""Đọc và mã hóa dữ liệu của Stage 3 (không cần torch, không đọc dataset thật)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from molopt.evaluator import data

ENDPOINTS = ["Ki", "Kd", "IC50", "EC50"]


def pair_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Bảng cặp tối giản, đủ các cột mà data.py cần."""
    return pa.table({
        "pair_id": pa.array([row.get("pair_id", "p1") for row in rows], pa.string()),
        "endpoint": pa.array([row.get("endpoint", "Ki") for row in rows], pa.string()),
        "protein_hash": pa.array([row.get("protein_hash", "h1") for row in rows], pa.string()),
        "partition": pa.array([row.get("partition", "train") for row in rows], pa.string()),
        "smiles_a": pa.array([row.get("smiles_a", "CC") for row in rows], pa.string()),
        "smiles_b": pa.array([row.get("smiles_b", "CCC") for row in rows], pa.string()),
        "variable_a": pa.array([row.get("variable_a", "[*]C") for row in rows], pa.string()),
        "variable_b": pa.array([row.get("variable_b", "[*]CC") for row in rows], pa.string()),
        "delta_p_mean": pa.array([row.get("delta_p_mean", 1.0) for row in rows], pa.float64()),
        "n_assays": pa.array([row.get("n_assays", 1) for row in rows], pa.int64()),
        "n_documents": pa.array([row.get("n_documents", 1) for row in rows], pa.int64()),
        "n_distinct_values": pa.array([row.get("n_distinct_values", 1) for row in rows],
                                      pa.int64()),
        "qc_status": pa.array([row.get("qc_status", "pass") for row in rows], pa.string()),
        "unseen_pair": pa.array([row.get("unseen_pair", False) for row in rows], pa.bool_()),
        "plausible_range": pa.array([row.get("plausible_range", True) for row in rows],
                                    pa.bool_()),
        "source_quality_warning": pa.array(
            [row.get("source_quality_warning") for row in rows], pa.string()),
    })


def test_check_partition_blocks_test() -> None:
    with pytest.raises(ValueError, match="test"):
        data.check_partition("test")


def test_check_partition_allows_test_only_when_asked_explicitly() -> None:
    assert data.check_partition("test", allow_test=True) == "test"


def test_check_partition_refuses_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="lạ"):
        data.check_partition("holdout")


def test_slice_mask_with_no_filter_keeps_every_row() -> None:
    table = pair_table([{}, {}])
    assert data.slice_mask(table, {}).tolist() == [True, True]


def test_slice_mask_on_the_strict_slice() -> None:
    table = pair_table([{"unseen_pair": True}, {"unseen_pair": False}])
    assert data.slice_mask(table, {"unseen_pair": True}).tolist() == [True, False]


def test_slice_mask_on_the_clean_slice_combines_two_conditions() -> None:
    table = pair_table([
        {"plausible_range": True, "source_quality_warning": None},
        {"plausible_range": True, "source_quality_warning": "Outside typical range"},
        {"plausible_range": False, "source_quality_warning": None},
    ])
    mask = data.slice_mask(table, {"plausible_range": True, "no_source_warning": True})
    assert mask.tolist() == [True, False, False]


def test_slice_mask_on_the_multi_assay_slice() -> None:
    table = pair_table([{"n_assays": 1}, {"n_assays": 2}, {"n_assays": 7}])
    assert data.slice_mask(table, {"min_n_assays": 2}).tolist() == [False, True, True]


def test_slice_mask_on_the_independent_agreement_slice() -> None:
    # Cần CẢ hai: ≥ 2 tài liệu và ≥ 2 giá trị khác nhau. Hai tài liệu cùng một giá trị là
    # bản sao, không phải xác nhận độc lập.
    table = pair_table([{"n_documents": 2, "n_distinct_values": 1},
                        {"n_documents": 1, "n_distinct_values": 2},
                        {"n_documents": 3, "n_distinct_values": 2}])
    mask = data.slice_mask(table, {"min_n_documents": 2, "min_distinct_values": 2})
    assert mask.tolist() == [False, False, True]


def test_pair_columns_cover_every_slice_column() -> None:
    for column in data.MINIMUM_KEYS.values():
        assert column in data.PAIR_COLUMNS


def test_config_slices_use_only_known_keys(config: dict[str, Any]) -> None:
    for entry in config["stage3"]["metrics"]["slices"]:
        assert set(entry["filters"]) <= set(data.SLICE_KEYS), entry["name"]


def test_slice_mask_refuses_an_unknown_filter_key() -> None:
    with pytest.raises(ValueError, match="khóa lát cắt lạ"):
        data.slice_mask(pair_table([{}]), {"qc_status": "pass"})


def test_code_column_maps_values_to_their_row_in_the_cache() -> None:
    vocabulary = pa.array(["CC", "CCC"], pa.string())
    codes = data.code_column(pa.array(["CCC", "CC"], pa.string()), vocabulary, "smiles")
    assert codes.tolist() == [1, 0]


def test_code_column_reports_a_value_missing_from_the_cache() -> None:
    vocabulary = pa.array(["CC"], pa.string())
    with pytest.raises(ValueError, match="không có trong cache"):
        data.code_column(pa.array(["CCO"], pa.string()), vocabulary, "smiles_a")


def test_encode_pairs_builds_indices_labels_and_slice_masks() -> None:
    table = pair_table([
        {"smiles_a": "CC", "smiles_b": "CCC", "endpoint": "Kd", "delta_p_mean": -0.5,
         "unseen_pair": True},
        {"smiles_a": "CCC", "smiles_b": "CC", "endpoint": "Ki", "delta_p_mean": 2.0},
    ])
    molecules = pa.array(["CC", "CCC"], pa.string())
    fragments = pa.array(["[*]C", "[*]CC"], pa.string())
    proteins = pa.array(["h1"], pa.string())
    slices = [{"name": "all", "filters": {}}, {"name": "strict",
                                               "filters": {"unseen_pair": True}}]
    arrays = data.encode_pairs(table, molecules, fragments, proteins, ENDPOINTS, slices)
    assert len(arrays) == 2
    assert arrays.molecule_a.tolist() == [0, 1]
    assert arrays.molecule_b.tolist() == [1, 0]
    assert arrays.endpoint.tolist() == [1, 0]
    assert arrays.protein.tolist() == [0, 0]
    assert arrays.target.tolist() == [-0.5, 2.0]
    assert arrays.slices["all"].tolist() == [True, True]
    assert arrays.slices["strict"].tolist() == [True, False]


def test_query_key_columns_encodes_both_sides_of_the_pair() -> None:
    table = pair_table([{"smiles_a": "CC", "smiles_b": "CCC"}])
    keys = data.query_key_columns(table, pa.array(["CC", "CCC"], pa.string()),
                                  pa.array(["h1"], pa.string()), ENDPOINTS)
    assert keys["molecule_a"].tolist() == [0] and keys["molecule_b"].tolist() == [1]
    assert keys["protein"].tolist() == [0] and keys["endpoint"].tolist() == [0]


def test_iter_batches_covers_every_row_including_a_short_last_batch() -> None:
    batches = list(data.iter_batches(5, 2))
    assert [batch.tolist() for batch in batches] == [[0, 1], [2, 3], [4]]


def test_iter_batches_shuffles_without_losing_or_repeating_a_row() -> None:
    rng = np.random.default_rng(0)
    rows = np.concatenate(list(data.iter_batches(10, 3, rng)))
    assert sorted(rows.tolist()) == list(range(10))


def test_iter_batches_refuses_a_batch_size_below_one() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        list(data.iter_batches(3, 0))


def test_distinct_strings_sorts_the_values_so_the_cache_is_reproducible() -> None:
    table = pair_table([{"smiles_a": "CCC", "smiles_b": "CC"},
                        {"smiles_a": "CC", "smiles_b": "CCCC"}])
    assert data.distinct_strings(table, ["smiles_a", "smiles_b"]).to_pylist() == [
        "CC", "CCC", "CCCC"]


def probe_arrays() -> data.PairArrays:
    """PairArrays hai dòng, dùng cho các phép dựng đầu vào chẩn đoán."""
    table = pair_table([
        {"smiles_a": "CC", "smiles_b": "CCC", "variable_a": "[*]C", "variable_b": "[*]CC",
         "delta_p_mean": 1.5, "unseen_pair": True},
        {"smiles_a": "CCC", "smiles_b": "CC", "variable_a": "[*]CC", "variable_b": "[*]C",
         "delta_p_mean": -0.5},
    ])
    return data.encode_pairs(table, pa.array(["CC", "CCC"], pa.string()),
                             pa.array(["[*]C", "[*]CC"], pa.string()),
                             pa.array(["h1"], pa.string()), ENDPOINTS,
                             [{"name": "all", "filters": {}},
                              {"name": "strict", "filters": {"unseen_pair": True}}])


def test_take_rows_keeps_the_slice_masks_aligned() -> None:
    subset = data.take_rows(probe_arrays(), np.array([1]))
    assert len(subset) == 1
    assert subset.molecule_a.tolist() == [1]
    assert subset.target.tolist() == [-0.5]
    assert subset.slices["strict"].tolist() == [False]


def test_swap_sides_swaps_both_sides_and_flips_the_label() -> None:
    arrays = probe_arrays()
    swapped = data.swap_sides(arrays)
    assert swapped.molecule_a.tolist() == arrays.molecule_b.tolist()
    assert swapped.molecule_b.tolist() == arrays.molecule_a.tolist()
    assert swapped.fragment_a.tolist() == arrays.fragment_b.tolist()
    assert swapped.target.tolist() == [-1.5, 0.5]
    # Protein và endpoint không đổi: chỉ hai phía của cặp bị đổi chỗ.
    assert swapped.protein.tolist() == arrays.protein.tolist()
    assert swapped.endpoint.tolist() == arrays.endpoint.tolist()


def test_swapping_twice_gives_back_the_original() -> None:
    arrays = probe_arrays()
    twice = data.swap_sides(data.swap_sides(arrays))
    assert twice.molecule_a.tolist() == arrays.molecule_a.tolist()
    assert twice.target.tolist() == arrays.target.tolist()


def test_identity_sides_sets_b_to_a_and_the_label_to_zero() -> None:
    arrays = probe_arrays()
    identity = data.identity_sides(arrays)
    assert identity.molecule_b.tolist() == arrays.molecule_a.tolist()
    assert identity.fragment_b.tolist() == arrays.fragment_a.tolist()
    assert identity.target.tolist() == [0.0, 0.0]



# ------------------------------------------------- gom batch theo protein (finetune)


def test_protein_batches_use_every_row_exactly_once() -> None:
    """Như iter_batches: không được mất dòng nào, không được lặp dòng nào."""
    proteins = np.array([0, 1, 0, 2, 1, 1, 2, 0, 3])
    rows = np.concatenate(list(data.iter_protein_batches(proteins, 2,
                                                         np.random.default_rng(0))))
    assert sorted(rows.tolist()) == list(range(proteins.size))


def test_protein_batches_keep_rows_of_one_protein_together() -> None:
    """4 dòng mỗi protein, batch 4 -> mỗi batch đúng một protein."""
    proteins = np.repeat([10, 20, 30], 4)
    batches = list(data.iter_protein_batches(proteins, 4, np.random.default_rng(1)))
    assert len(batches) == 3
    for batch in batches:
        assert len(set(proteins[batch])) == 1


def test_protein_batches_cut_far_fewer_proteins_than_random_batches() -> None:
    """Lý do tồn tại của hàm này: bộ nhớ của finetune tỉ lệ với số protein phân biệt."""
    rng = np.random.default_rng(2)
    proteins = rng.integers(0, 50, size=4000)
    grouped = max(len(set(proteins[b]))
                  for b in data.iter_protein_batches(proteins, 64, rng))
    shuffled = max(len(set(proteins[b]))
                   for b in data.iter_batches(proteins.size, 64, rng))
    assert grouped <= 3
    assert shuffled >= 30


def test_protein_batches_are_deterministic_without_an_rng() -> None:
    proteins = np.array([2, 0, 1, 0, 2])
    once = [b.tolist() for b in data.iter_protein_batches(proteins, 2)]
    twice = [b.tolist() for b in data.iter_protein_batches(proteins, 2)]
    assert once == twice


def test_protein_batches_shuffle_when_given_an_rng() -> None:
    proteins = np.repeat(np.arange(20), 10)
    first = np.concatenate(list(data.iter_protein_batches(proteins, 10,
                                                          np.random.default_rng(0))))
    second = np.concatenate(list(data.iter_protein_batches(proteins, 10,
                                                           np.random.default_rng(1))))
    assert first.tolist() != second.tolist()


def test_protein_batches_keep_the_short_last_batch() -> None:
    proteins = np.array([0, 0, 0, 0, 0])
    sizes = [b.size for b in data.iter_protein_batches(proteins, 2)]
    assert sizes == [2, 2, 1]


def test_protein_batches_reject_a_bad_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        list(data.iter_protein_batches(np.array([0, 1]), 0))


def test_protein_cap_is_never_exceeded() -> None:
    """Giới hạn CHẮC CHẮN, không phải thống kê: một batch vượt trần cũng đủ làm hết
    bộ nhớ khi đang chia GPU với người khác."""
    rng = np.random.default_rng(3)
    proteins = rng.integers(0, 400, size=5000)
    for batch in data.iter_protein_batches(proteins, 64, rng, max_proteins=4):
        assert len(set(proteins[batch])) <= 4


def test_protein_cap_still_uses_every_row_exactly_once() -> None:
    rng = np.random.default_rng(4)
    proteins = rng.integers(0, 50, size=600)
    rows = np.concatenate(list(data.iter_protein_batches(proteins, 32, rng,
                                                         max_proteins=3)))
    assert sorted(rows.tolist()) == list(range(proteins.size))


def test_protein_cap_cuts_the_batch_short_rather_than_mixing_in_one_more() -> None:
    """3 protein x 1 dòng, batch 10, trần 2 -> batch đầu phải dừng ở 2 dòng."""
    proteins = np.array([0, 1, 2])
    sizes = [b.size for b in data.iter_protein_batches(proteins, 10, None, max_proteins=2)]
    assert sizes == [2, 1]


def test_protein_cap_does_not_split_a_big_protein_needlessly() -> None:
    """Một protein nhiều dòng vẫn được cắt theo batch_size, không bị trần làm vụn thêm."""
    proteins = np.zeros(100, dtype=np.int64)
    sizes = [b.size for b in data.iter_protein_batches(proteins, 40, None, max_proteins=2)]
    assert sizes == [40, 40, 20]


def test_protein_cap_of_zero_means_no_cap() -> None:
    proteins = np.arange(20)
    capped = [b.tolist() for b in data.iter_protein_batches(proteins, 8, None, 0)]
    plain = [b.tolist() for b in data.iter_protein_batches(proteins, 8, None)]
    assert capped == plain


def test_cliff_slice_follows_the_moleculeace_fold_change_rule() -> None:
    """Cliff = chênh hoạt tính HƠN 10 lần = |Δp| > 1, dấu > ngặt như find_cliffs."""
    table = pair_table([{"delta_p_mean": 1.5}, {"delta_p_mean": -1.5},
                        {"delta_p_mean": 1.0}, {"delta_p_mean": -1.0},
                        {"delta_p_mean": 0.2}])
    mask = data.slice_mask(table, {"abs_delta_p_above": 1.0})
    assert mask.tolist() == [True, True, False, False, False]


def test_cliff_slice_uses_the_absolute_value_so_direction_does_not_matter() -> None:
    # Dataset lưu cặp theo chiều khóa A < B, nên dấu của Δp là quy ước lưu trữ.
    table = pair_table([{"delta_p_mean": 2.5}, {"delta_p_mean": -2.5}])
    assert data.slice_mask(table, {"abs_delta_p_above": 1.0}).tolist() == [True, True]


def test_the_two_cliff_slices_cover_every_row_exactly_once() -> None:
    table = pair_table([{"delta_p_mean": value}
                        for value in (-3.0, -1.0, -0.4, 0.0, 0.4, 1.0, 3.0)])
    cliff = data.slice_mask(table, {"abs_delta_p_above": 1.0})
    rest = data.slice_mask(table, {"abs_delta_p_at_most": 1.0})
    assert not (cliff & rest).any(), "hai lát cắt không được chồng nhau"
    assert (cliff | rest).all(), "hai lát cắt phải phủ kín"


def test_cliff_slice_can_be_combined_with_another_filter() -> None:
    table = pair_table([{"delta_p_mean": 2.0, "unseen_pair": True},
                        {"delta_p_mean": 2.0, "unseen_pair": False},
                        {"delta_p_mean": 0.1, "unseen_pair": True}])
    mask = data.slice_mask(table, {"abs_delta_p_above": 1.0, "unseen_pair": True})
    assert mask.tolist() == [True, False, False]


def test_config_defines_both_cliff_slices_at_the_sourced_threshold() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    metrics = json.loads((root / "config.json").read_text(encoding="utf-8"))
    slices = {definition["name"]: definition["filters"]
              for definition in metrics["stage3"]["metrics"]["slices"]}
    # Ngưỡng 1.0 đến từ MoleculeACE potency_fold=10; đổi số là lệch khỏi định nghĩa.
    assert slices["cliff"] == {"abs_delta_p_above": 1.0}
    assert slices["non_cliff"] == {"abs_delta_p_at_most": 1.0}
