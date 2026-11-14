"""Việc 13 họp 09/11/2026: chia theo CẶP — trùng protein được, trùng cặp thì không.

Hai nửa, và cả hai đều là chỗ dễ sai âm thầm:

1. `data.py` — bảng đổi partition khóa theo pair_id. Phải ghi đè CẢ cột `unseen_pair`
   (cột trong dataset tính theo cách chia CŨ, mà lát `strict` đọc đúng cột đó), và cặp
   không có trong bảng phải bị loại hẳn chứ không được bỏ lại theo cách chia cũ.
2. `build_pair_split.py` — đơn vị chia là cặp KHÔNG kể thứ tự, và một cặp không bao giờ
   được nằm cả hai bên.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_pair_split as builder  # noqa: E402  (scripts/ không phải package)

from molopt.evaluator import data  # noqa: E402


def dataset_table() -> pa.Table:
    """Sáu dòng. Cặp {C, CC} xuất hiện ở hai protein; chiều ngược nằm ở dòng cuối."""
    rows = [
        # pair_id, protein, a,     b,      partition cũ, unseen_pair cũ
        ("p0", "ha", "C", "CC", "train", False),
        ("p1", "ha", "C", "CCC", "train", False),
        ("p2", "hb", "C", "CC", "train", False),
        ("p3", "hb", "CO", "CCO", "train", False),
        ("p4", "hc", "C", "CCCC", "validation", True),
        ("p5", "hc", "CC", "C", "validation", True),
    ]
    return pa.table({
        "pair_id": pa.array([row[0] for row in rows], pa.string()),
        "endpoint": pa.array(["Ki"] * len(rows), pa.string()),
        "protein_hash": pa.array([row[1] for row in rows], pa.string()),
        "smiles_a": pa.array([row[2] for row in rows], pa.string()),
        "smiles_b": pa.array([row[3] for row in rows], pa.string()),
        "partition": pa.array([row[4] for row in rows], pa.string()),
        "unseen_pair": pa.array([row[5] for row in rows], pa.bool_()),
        "delta_p_mean": pa.array([1.0] * len(rows), pa.float64()),
    })


def write_dataset(tmp_path: Path) -> dict[str, Any]:
    target = tmp_path / "evaluator_training.parquet"
    pq.write_table(dataset_table(), target)
    return {"stage3": {"inputs": {"evaluator": target.name, "partition_override": "",
                                  "partition_override_pairs": "", "train_subset": ""}}}


def write_pairs(tmp_path: Path, mapping: dict[str, tuple[str, bool]],
                name: str = "pairs.parquet") -> str:
    table = pa.table({
        "pair_id": pa.array(list(mapping), pa.string()),
        "partition": pa.array([value[0] for value in mapping.values()], pa.string()),
        "unseen_pair": pa.array([value[1] for value in mapping.values()], pa.bool_()),
    })
    pq.write_table(table, tmp_path / name)
    return name


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    data._OVERRIDE_CACHE.clear()
    data._PAIR_OVERRIDE_CACHE.clear()


# ------------------------------------------------------- data.py: áp bảng

def test_pairs_move_between_partitions_and_proteins_may_repeat(tmp_path: Path) -> None:
    """Đúng điều kiện việc 13: hb nằm ở CẢ hai bên, nhưng không cặp nào nằm cả hai bên."""
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override_pairs"] = write_pairs(tmp_path, {
        "p0": ("train", False), "p2": ("train", False), "p1": ("train", False),
        "p3": ("validation", True),
    })
    train = data.read_columns(config, tmp_path, "train", ["protein_hash"])
    validation = data.read_columns(config, tmp_path, "validation", ["protein_hash"])
    assert sorted(train.column("protein_hash").to_pylist()) == ["ha", "ha", "hb"]
    assert validation.column("protein_hash").to_pylist() == ["hb"]


def test_a_pair_missing_from_the_table_is_dropped_not_left_in_the_old_split(
        tmp_path: Path) -> None:
    """p4 và p5 không được liệt kê -> sang 'test', KHÔNG còn ở validation cũ."""
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override_pairs"] = write_pairs(tmp_path, {
        "p0": ("train", False), "p3": ("validation", True)})
    train = data.read_columns(config, tmp_path, "train", ["pair_id"])
    validation = data.read_columns(config, tmp_path, "validation", ["pair_id"])
    assert train.column("pair_id").to_pylist() == ["p0"]
    assert validation.column("pair_id").to_pylist() == ["p3"]


def test_unseen_pair_is_overwritten_because_the_old_column_is_wrong_now(
        tmp_path: Path) -> None:
    """Cột cũ nói p3 là False (nó ở train cũ); cách chia mới đưa p3 sang validation."""
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override_pairs"] = write_pairs(tmp_path, {
        "p0": ("train", False), "p3": ("validation", True)})
    validation = data.read_columns(config, tmp_path, "validation",
                                   ["pair_id", "unseen_pair"])
    assert validation.column("unseen_pair").to_pylist() == [True]
    train = data.read_columns(config, tmp_path, "train", ["pair_id", "unseen_pair"])
    assert train.column("unseen_pair").to_pylist() == [False]


def test_the_strict_slice_sees_every_validation_row_of_this_split(tmp_path: Path) -> None:
    """Hệ quả của việc ghi đè: trên cách chia này lát strict trùng lát all."""
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override_pairs"] = write_pairs(tmp_path, {
        "p0": ("train", False), "p1": ("train", False),
        "p3": ("validation", True), "p4": ("validation", True)})
    validation = data.read_columns(config, tmp_path, "validation", ["unseen_pair"])
    flags = validation.column("unseen_pair").to_pylist()
    assert flags == [True, True] and len(flags) == validation.num_rows


def test_the_pair_id_column_is_not_leaked_into_the_result(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override_pairs"] = write_pairs(tmp_path, {
        "p0": ("train", False), "p3": ("validation", True)})
    table = data.read_columns(config, tmp_path, "train", ["endpoint"])
    assert sorted(table.schema.names) == ["endpoint", "partition"]


def test_a_duplicate_pair_id_in_the_table_is_an_error(tmp_path: Path) -> None:
    target = tmp_path / "dup.parquet"
    pq.write_table(pa.table({"pair_id": pa.array(["p0", "p0"], pa.string()),
                             "partition": pa.array(["train", "validation"], pa.string()),
                             "unseen_pair": pa.array([False, True], pa.bool_())}), target)
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override_pairs"] = target.name
    with pytest.raises(ValueError, match="pair_id trùng"):
        data.read_columns(config, tmp_path, "train", ["pair_id"])


def test_a_missing_table_names_the_builder(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override_pairs"] = "khong_co.parquet"
    with pytest.raises(FileNotFoundError, match="build_pair_split"):
        data.read_columns(config, tmp_path, "train", ["pair_id"])


@pytest.mark.parametrize("other", ["partition_override", "train_subset"])
def test_combining_with_the_other_split_paths_is_refused(tmp_path: Path, other: str) -> None:
    """Chồng hai bảng đổi lên nhau thì cách chia phụ thuộc thứ tự áp — phải chặn."""
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override_pairs"] = write_pairs(
        tmp_path, {"p0": ("train", False)})
    config["stage3"]["inputs"][other] = "bang_khac.parquet"
    with pytest.raises(ValueError, match="không được bật cùng lúc"):
        data.read_columns(config, tmp_path, "train", ["pair_id"])


# --------------------------------------------- build_pair_split.py: dựng bảng

def test_a_pair_and_its_reverse_are_one_group(tmp_path: Path) -> None:
    """p0 là (C, CC) còn p5 là (CC, C): cùng một cặp hóa học, phải cùng một nhóm."""
    codes, groups = builder.pair_group_codes(dataset_table(), with_endpoint=False)
    assert codes[0] == codes[2] == codes[5]
    assert groups == 4  # {C,CC}, {C,CCC}, {CO,CCO}, {C,CCCC}


def test_the_endpoint_variant_splits_the_same_molecules_apart() -> None:
    table = dataset_table()
    table = table.set_column(table.schema.get_field_index("endpoint"), "endpoint",
                             pa.array(["Ki", "Ki", "IC50", "Ki", "Ki", "Ki"], pa.string()))
    codes, groups = builder.pair_group_codes(table, with_endpoint=True)
    assert codes[0] != codes[2]  # cùng {C, CC} nhưng khác endpoint
    assert codes[0] == codes[5]
    assert groups == 5


def test_a_group_never_lands_on_both_sides() -> None:
    codes = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    flags = builder.split_groups(codes, 4, fraction=0.5, seed=7)
    for code in range(4):
        side = flags[codes == code]
        assert side.all() or not side.any()


def test_the_split_never_empties_either_side() -> None:
    codes = np.array([0, 0, 1, 1], dtype=np.int64)
    for fraction in (0.0, 1.0):
        flags = builder.split_groups(codes, 2, fraction=fraction, seed=3)
        assert flags.any() and not flags.all()


def test_building_end_to_end_keeps_every_validation_protein_in_train(
        tmp_path: Path, monkeypatch, capsys) -> None:
    dataset = tmp_path / "evaluator_training.parquet"
    pq.write_table(dataset_table(), dataset)
    out = tmp_path / "pair_split.parquet"
    monkeypatch.setattr(sys, "argv", ["build_pair_split.py", "--dataset", str(dataset),
                                      "--subset", "", "--fraction", "0.5",
                                      "--seed", "11", "--out", str(out)])
    assert builder.main() == 0
    table = pq.read_table(out)
    rows = dict(zip(table.column("pair_id").to_pylist(),
                    table.column("partition").to_pylist()))
    source = dataset_table()
    protein_of = dict(zip(source.column("pair_id").to_pylist(),
                          source.column("protein_hash").to_pylist()))
    train_proteins = {protein_of[key] for key, value in rows.items() if value == "train"}
    validation_proteins = {protein_of[key] for key, value in rows.items()
                           if value == "validation"}
    assert validation_proteins and train_proteins
    assert validation_proteins <= train_proteins
    assert "cặp trùng hai bên: 0" in capsys.readouterr().out


def test_unseen_pair_is_true_for_exactly_the_validation_rows(tmp_path: Path,
                                                             monkeypatch) -> None:
    dataset = tmp_path / "evaluator_training.parquet"
    pq.write_table(dataset_table(), dataset)
    out = tmp_path / "pair_split.parquet"
    monkeypatch.setattr(sys, "argv", ["build_pair_split.py", "--dataset", str(dataset),
                                      "--subset", "", "--fraction", "0.5", "--seed", "11",
                                      "--keep-unseen-proteins", "--out", str(out)])
    builder.main()
    table = pq.read_table(out)
    for partition, flag in zip(table.column("partition").to_pylist(),
                               table.column("unseen_pair").to_pylist()):
        assert flag == (partition == "validation")
