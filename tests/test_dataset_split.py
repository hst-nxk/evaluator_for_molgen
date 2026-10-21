"""Logic thuần của src/molopt/dataset/split.py (dựng split protein, không gom chuỗi)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow as pa
import pytest

from molopt.dataset import split

PARTITIONS = ["train", "validation", "test"]
RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
SEED = 20261021


def split_table(protein_pairs: dict[str, int], membership: dict[str, str]) -> pa.Table:
    return split.build_split_table(protein_pairs, membership, RATIOS, PARTITIONS, SEED)


# --------------------------------------------------------------------------- đếm cặp, FASTA


def test_protein_pair_counts_adds_up_across_tables() -> None:
    counts = split.protein_pair_counts([pa.array(["aa", "aa", "bb"]), pa.array(["aa"])])
    assert counts == Counter({"aa": 3, "bb": 1})


def test_write_fasta_wraps_lines_and_sorts_by_hash(tmp_path: Path) -> None:
    path = tmp_path / "input.fasta"
    written = split.write_fasta({"bb": "MKT", "aa": "A" * 7}, ["bb", "aa"], path, line_width=5)
    assert written == 2
    assert path.read_text(encoding="ascii").splitlines() == [">aa", "AAAAA", "AA", ">bb", "MKT"]


def test_write_fasta_refuses_a_protein_without_sequence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="không có chuỗi"):
        split.write_fasta({"aa": "MKT"}, ["aa", "bb"], tmp_path / "input.fasta")


# --------------------------------------------------------------------------- nhóm, phân chia


def test_cluster_id_does_not_depend_on_member_order() -> None:
    assert split.cluster_id(["aa", "bb"]) == split.cluster_id(["bb", "aa"])
    assert split.cluster_id(["aa"]) != split.cluster_id(["bb"])


def test_assign_partitions_sends_the_largest_cluster_to_train() -> None:
    assignment = split.assign_partitions({"big": 800, "small": 100, "tiny": 100},
                                         RATIOS, PARTITIONS, SEED)
    assert assignment["big"] == "train"
    assert set(assignment.values()) == set(PARTITIONS)


def test_assign_partitions_is_deterministic_for_one_seed() -> None:
    sizes = {f"c{number}": number % 7 + 1 for number in range(40)}
    first = split.assign_partitions(sizes, RATIOS, PARTITIONS, SEED)
    assert first == split.assign_partitions(sizes, RATIOS, PARTITIONS, SEED)


def test_assign_partitions_gets_close_to_the_target_ratios() -> None:
    sizes = {f"c{number}": 1 for number in range(1000)}
    assignment = split.assign_partitions(sizes, RATIOS, PARTITIONS, SEED)
    counts = Counter(assignment.values())
    assert counts["train"] == pytest.approx(800, abs=5)
    assert counts["validation"] == pytest.approx(100, abs=5)


def test_build_split_table_keeps_a_cluster_in_one_partition() -> None:
    table = split_table({"aa": 500, "bb": 300, "cc": 100, "dd": 100},
                        {"aa": "r1", "bb": "r1", "cc": "r2", "dd": "r3"})
    rows = {row["protein_sequence_hash"]: row for row in table.to_pylist()}
    assert rows["aa"]["partition"] == rows["bb"]["partition"]
    assert rows["aa"]["protein_cluster_id"] == rows["bb"]["protein_cluster_id"]
    assert rows["aa"]["cluster_proteins"] == 2
    assert rows["aa"]["cluster_pairs"] == 800
    assert rows["cc"]["pairs"] == 100


def test_build_split_table_refuses_a_protein_without_a_cluster() -> None:
    with pytest.raises(ValueError, match="không có nhóm"):
        split_table({"aa": 10, "bb": 10}, {"aa": "r1"})


def test_build_split_table_matches_the_declared_schema() -> None:
    table = split_table({"aa": 10}, {"aa": "r1"})
    assert table.schema == split.SPLIT_SCHEMA


# --------------------------------------------------------------------------- kiểm tra


def test_checks_pass_on_a_table_built_by_the_module() -> None:
    counts = {"aa": 500, "bb": 300, "cc": 100, "dd": 100}
    table = split_table(counts, {"aa": "r1", "bb": "r1", "cc": "r2", "dd": "r3"})
    assert split.check_clusters_single_partition(table)["passed"]
    assert split.check_proteins_single_partition(table)["passed"]
    totals = split.check_pair_totals(table, counts)
    assert totals["passed"]
    assert sum(totals["pairs_by_partition"].values()) == 1000


def test_check_clusters_single_partition_finds_a_split_cluster() -> None:
    table = pa.table({"protein_sequence_hash": ["aa", "bb"],
                      "protein_cluster_id": ["c1", "c1"],
                      "partition": ["train", "test"],
                      "cluster_proteins": [2, 2], "cluster_pairs": [2, 2], "pairs": [1, 1]},
                     schema=split.SPLIT_SCHEMA)
    result = split.check_clusters_single_partition(table)
    assert not result["passed"] and result["clusters_in_multiple_partitions"] == 1


def test_check_proteins_single_partition_finds_a_repeated_protein() -> None:
    table = pa.table({"protein_sequence_hash": ["aa", "aa"],
                      "protein_cluster_id": ["c1", "c2"],
                      "partition": ["train", "test"],
                      "cluster_proteins": [1, 1], "cluster_pairs": [1, 1], "pairs": [1, 1]},
                     schema=split.SPLIT_SCHEMA)
    result = split.check_proteins_single_partition(table)
    assert not result["passed"] and result["proteins_in_multiple_partitions"] == 1


def test_check_pair_totals_reports_a_protein_missing_from_the_split() -> None:
    table = split_table({"aa": 10}, {"aa": "r1"})
    result = split.check_pair_totals(table, {"aa": 10, "bb": 5})
    assert not result["passed"] and result["proteins_missing_from_split"] == 1


def test_check_pair_totals_reports_a_wrong_pair_count() -> None:
    table = split_table({"aa": 10}, {"aa": "r1"})
    result = split.check_pair_totals(table, {"aa": 11})
    assert not result["passed"] and result["proteins_with_wrong_pair_count"] == 1


def test_ratio_report_gives_actual_ratio_and_deviation() -> None:
    report = split.ratio_report({"train": 700, "validation": 200, "test": 100}, RATIOS)
    assert report["train"]["actual"] == pytest.approx(0.7)
    assert report["train"]["deviation"] == pytest.approx(-0.1)
    assert report["validation"]["deviation"] == pytest.approx(0.1)


def test_compare_old_split_counts_proteins_in_the_same_partition() -> None:
    table = pa.table({"protein_sequence_hash": ["aa", "bb", "cc"],
                      "protein_cluster_id": ["c1", "c2", "c3"],
                      "partition": ["train", "test", "train"],
                      "cluster_proteins": [1, 1, 1], "cluster_pairs": [1, 1, 1],
                      "pairs": [1, 1, 1]}, schema=split.SPLIT_SCHEMA)
    old = pa.table({"target_sequence_hash": ["aa", "bb", "dd"],
                    "partition": ["train", "train", "test"]})
    result = split.compare_old_split(table, old)
    assert (result["proteins_shared"], result["same_partition"]) == (2, 1)
    assert result["same_partition_share"] == pytest.approx(0.5)
    assert result["old_to_new"] == {"train -> test": 1, "train -> train": 1}


def test_compare_old_split_without_shared_proteins() -> None:
    table = split_table({"aa": 1}, {"aa": "r1"})
    old = pa.table({"target_sequence_hash": ["zz"], "partition": ["train"]})
    result = split.compare_old_split(table, old)
    assert result["proteins_shared"] == 0 and result["same_partition_share"] is None
