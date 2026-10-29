"""Phần lipinski của audit bổ sung (src/molopt/audit/followup.py), trên dữ liệu tự tạo."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molopt.audit import followup

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"
# Tetracontan: MW ≈ 563 > 500 và logP rất cao, không có N/O -> vượt đúng 2 tiêu chí.
LONG_ALKANE = "C" * 40


@pytest.fixture()
def limits(config: dict[str, Any]) -> dict[str, Any]:
    return config["audit"]["followup"]["lipinski"]


def test_lipinski_properties_counts_donors_and_acceptors_like_the_paper() -> None:
    (weight, logp, donors, acceptors), broken = followup.lipinski_properties(
        [ASPIRIN, "not a smiles"])
    # Aspirin: một OH (HBD 1), bốn nguyên tử O (HBA 4), MW 180,16.
    assert (donors, acceptors) == (1, 4)
    assert weight == pytest.approx(180.159, abs=0.01)
    assert 0 < logp < 3
    assert all(math.isnan(value) for value in broken)


def test_violation_counts_uses_strictly_greater_than(limits: dict[str, Any]) -> None:
    properties = np.array([[600, 6, 6, 11], [400, 1, 1, 1], [500, 5, 5, 10],
                           [501, 5, 5, 10], [np.nan] * 4], dtype=np.float64)
    assert followup.violation_counts(properties, limits).tolist() == [4, 0, 0, 1, -1]


def test_pair_violations_takes_the_worse_side_and_keeps_unparsed() -> None:
    violations = np.array([0, 2, -1])
    worst = followup.pair_violations(violations, np.array([0, 0, 1]), np.array([1, 2, 0]))
    assert worst.tolist() == [2, -1, 2]


def test_ordered_counts_sorts_by_value() -> None:
    assert followup.ordered_counts(np.array([2, 0, 2, -1])) == {"-1": 1, "0": 1, "2": 2}


def test_lipinski_check_on_a_tiny_evaluator(tmp_path: Path, config: dict[str, Any]) -> None:
    local = copy.deepcopy(config)
    local["stage2b"]["paths"]["dataset_dir"] = "ds"
    local["audit"]["molecules"]["workers"] = 1
    path = tmp_path / "ds" / local["stage2b"]["files"]["evaluator"]
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"smiles_a": [ASPIRIN, ETHANOL], "smiles_b": [ETHANOL, LONG_ALKANE],
                             "endpoint": ["Ki", "Ki"], "partition": ["train", "test"]}), path)
    part = followup.lipinski_check(local, tmp_path)
    assert part["molecules"] == 3 and part["pairs"] == 2
    assert part["molecule_violations"] == {"0": 2, "2": 1}
    assert part["pair_worst_violations"] == {"0": 1, "2": 1}
    assert part["by_partition"] == {"test": {"pairs": 1, "zero": 0, "at_most_one": 0},
                                    "train": {"pairs": 1, "zero": 1, "at_most_one": 1}}
    assert [row["count"] for row in part["molecule_weight"]] == [2, 1, 0]
    assert followup.lipinski_lines(part)[0].startswith("## 4.")
