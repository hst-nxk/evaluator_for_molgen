"""Logic thuần của src/molopt/dataset/delta.py (bước 3: Δp trong assay)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from molopt.dataset import delta, mmp

ID_LENGTH = 24


def group_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Bảng bước 1 (đã thêm cột smiles) cho assay_deltas."""
    return pa.table({
        "assay_key": pa.array([row.get("assay_key", "s:e:1") for row in rows], pa.string()),
        "source": pa.array([row.get("source", "chembl37") for row in rows], pa.string()),
        "protein_hash": pa.array([row.get("protein_hash", "aa") for row in rows], pa.string()),
        "molecule_key": pa.array([row["molecule_key"] for row in rows], pa.string()),
        "p_mean": pa.array([row["p_mean"] for row in rows], pa.float64()),
        "p_std": pa.array([row.get("p_std") for row in rows], pa.float64()),
        "n_merged": pa.array([row.get("n_merged", 1) for row in rows], pa.int64()),
        "n_independent": pa.array([row.get("n_independent", 1) for row in rows], pa.int64()),
        "n_raw": pa.array([row.get("n_raw", 1) for row in rows], pa.int64()),
        "n_source_ids": pa.array([row.get("n_source_ids", 1) for row in rows], pa.int64()),
        "multi_source_id": pa.array([row.get("multi_source_id", False) for row in rows],
                                    pa.bool_()),
        "observation_ids": pa.array([row.get("observation_ids", ["o"]) for row in rows],
                                    pa.list_(pa.string())),
        "smiles": pa.array([row.get("smiles", "CC") for row in rows], pa.string()),
    })


def pair(row_x: int, row_y: int) -> mmp.Pair:
    return mmp.Pair(row_x, row_y, "[*:1]c1ccccc1", "[*:1]C", "[*:1]CC")


# --------------------------------------------------------------------------- pair_id, chiều


def test_pair_id_is_stable_and_has_the_configured_length() -> None:
    first = delta.pair_id("Ki", "aa", "KEY-A", "KEY-B", ID_LENGTH)
    assert first == delta.pair_id("Ki", "aa", "KEY-A", "KEY-B", ID_LENGTH)
    assert len(first) == ID_LENGTH


def test_pair_id_changes_with_endpoint_protein_and_direction() -> None:
    base = delta.pair_id("Ki", "aa", "KEY-A", "KEY-B", ID_LENGTH)
    assert base != delta.pair_id("IC50", "aa", "KEY-A", "KEY-B", ID_LENGTH)
    assert base != delta.pair_id("Ki", "bb", "KEY-A", "KEY-B", ID_LENGTH)
    assert base != delta.pair_id("Ki", "aa", "KEY-B", "KEY-A", ID_LENGTH)


def test_orient_puts_the_smaller_key_first() -> None:
    assert delta.orient("KEY-B", "KEY-A") is True
    assert delta.orient("KEY-A", "KEY-B") is False


# --------------------------------------------------------------------------- bảng Δp


def test_assay_deltas_orders_sides_by_key_and_keeps_the_sign() -> None:
    groups = group_table([{"molecule_key": "KEY-B", "p_mean": 8.0},
                          {"molecule_key": "KEY-A", "p_mean": 6.0}])
    (row,) = delta.assay_deltas(groups, [pair(0, 1)], "Ki", ID_LENGTH).to_pylist()
    assert (row["molecule_a_key"], row["molecule_b_key"]) == ("KEY-A", "KEY-B")
    assert (row["p_a"], row["p_b"]) == (6.0, 8.0)
    assert row["delta_p"] == pytest.approx(2.0)
    assert row["endpoint"] == "Ki"


def test_assay_deltas_swaps_the_variable_parts_with_the_sides() -> None:
    groups = group_table([{"molecule_key": "KEY-B", "p_mean": 8.0},
                          {"molecule_key": "KEY-A", "p_mean": 6.0}])
    (row,) = delta.assay_deltas(groups, [pair(0, 1)], "Ki", ID_LENGTH).to_pylist()
    # row_x (KEY-B) mang variable_x; sau khi đổi chỗ, nó phải thành variable_b.
    assert (row["variable_a"], row["variable_b"]) == ("[*:1]CC", "[*:1]C")


def test_assay_deltas_negative_delta_when_b_is_weaker() -> None:
    groups = group_table([{"molecule_key": "KEY-A", "p_mean": 9.0},
                          {"molecule_key": "KEY-B", "p_mean": 7.5}])
    (row,) = delta.assay_deltas(groups, [pair(0, 1)], "Kd", ID_LENGTH).to_pylist()
    assert row["delta_p"] == pytest.approx(-1.5)


def test_assay_deltas_marks_pair_independent_only_if_both_sides_are() -> None:
    groups = group_table([{"molecule_key": "KEY-A", "p_mean": 6.0, "n_independent": 0},
                          {"molecule_key": "KEY-B", "p_mean": 7.0}])
    (row,) = delta.assay_deltas(groups, [pair(0, 1)], "Ki", ID_LENGTH).to_pylist()
    assert row["independent"] is False


def test_assay_deltas_marks_multi_source_id_from_either_side() -> None:
    groups = group_table([{"molecule_key": "KEY-A", "p_mean": 6.0},
                          {"molecule_key": "KEY-B", "p_mean": 7.0,
                           "multi_source_id": True}])
    (row,) = delta.assay_deltas(groups, [pair(0, 1)], "Ki", ID_LENGTH).to_pylist()
    assert row["multi_source_id"] is True


def test_assay_deltas_carries_observation_ids_of_both_sides() -> None:
    groups = group_table([{"molecule_key": "KEY-A", "p_mean": 6.0,
                           "observation_ids": ["o1", "o2"]},
                          {"molecule_key": "KEY-B", "p_mean": 7.0,
                           "observation_ids": ["o3"]}])
    (row,) = delta.assay_deltas(groups, [pair(0, 1)], "Ki", ID_LENGTH).to_pylist()
    assert (row["observation_ids_a"], row["observation_ids_b"]) == (["o1", "o2"], ["o3"])


def test_assay_deltas_matches_the_stored_pair_id() -> None:
    groups = group_table([{"molecule_key": "KEY-A", "p_mean": 6.0},
                          {"molecule_key": "KEY-B", "p_mean": 7.0}])
    (row,) = delta.assay_deltas(groups, [pair(0, 1)], "Ki", ID_LENGTH).to_pylist()
    assert row["pair_id"] == delta.pair_id("Ki", "aa", "KEY-A", "KEY-B", ID_LENGTH)


def test_assay_deltas_without_pairs_returns_the_schema() -> None:
    table = delta.assay_deltas(group_table([]), [], "Ki", ID_LENGTH)
    assert table.num_rows == 0 and table.schema == delta.ASSAY_DELTA_SCHEMA


def test_assay_deltas_handles_many_pairs_in_one_table() -> None:
    groups = group_table([{"molecule_key": f"KEY-{letter}", "p_mean": value}
                          for letter, value in zip("ABC", [6.0, 7.0, 8.0])])
    table = delta.assay_deltas(groups, [pair(0, 1), pair(1, 2), pair(0, 2)], "Ki", ID_LENGTH)
    assert table.num_rows == 3
    assert np.allclose(table.column("delta_p").to_numpy(), [1.0, 1.0, 2.0])


# --------------------------------------------------------------------------- cặp bị loại


def test_lost_pairs_table_orders_the_sides_by_key() -> None:
    groups = group_table([{"molecule_key": "KEY-B", "p_mean": 8.0},
                          {"molecule_key": "KEY-A", "p_mean": 6.0}])
    (row,) = delta.lost_pairs_table(groups, [(0, 1)], "Ki").to_pylist()
    assert (row["molecule_a_key"], row["molecule_b_key"]) == ("KEY-A", "KEY-B")
    assert row["endpoint"] == "Ki"


def test_lost_pairs_table_without_rows_returns_the_schema() -> None:
    table = delta.lost_pairs_table(group_table([]), [], "Ki")
    assert table.num_rows == 0 and table.schema == delta.LOST_PAIR_SCHEMA
