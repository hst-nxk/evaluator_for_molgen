"""Đường lọc subset train của Stage 3 (việc 2 họp 09/11/2026).

Không đọc dataset thật: mọi test dựng một parquet nhỏ trong tmp_path.

Điều kiện quan trọng nhất mà file này ghim: subset CHỈ áp cho partition train. Nếu nó
rò sang validation thì mẫu số của mọi chỉ số sẽ đổi và không một con số nào trong tuần
trước còn so được nữa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molopt.evaluator import data


def dataset_table() -> pa.Table:
    """Tám dòng: sáu train (ha, hb), hai validation (hc)."""
    rows = [("p0", "ha", "train"), ("p1", "ha", "train"), ("p2", "ha", "train"),
            ("p3", "hb", "train"), ("p4", "hb", "train"), ("p5", "hb", "train"),
            ("p6", "hc", "validation"), ("p7", "hc", "validation")]
    return pa.table({
        "pair_id": pa.array([row[0] for row in rows], pa.string()),
        "endpoint": pa.array(["Ki"] * len(rows), pa.string()),
        "protein_hash": pa.array([row[1] for row in rows], pa.string()),
        "partition": pa.array([row[2] for row in rows], pa.string()),
        "delta_p_mean": pa.array([1.0] * len(rows), pa.float64()),
    })


def write_dataset(tmp_path: Path) -> dict[str, Any]:
    """Ghi dataset nhỏ và trả về config trỏ vào nó, chưa bật subset."""
    target = tmp_path / "evaluator_training.parquet"
    pq.write_table(dataset_table(), target)
    return {"stage3": {"inputs": {"evaluator": target.name, "partition_override": "",
                                  "train_subset": ""}}}


def write_subset(tmp_path: Path, pair_ids: list[str],
                 name: str = "subset.parquet") -> str:
    """Ghi bảng subset một cột pair_id và trả về tên file."""
    pq.write_table(pa.table({"pair_id": pa.array(pair_ids, pa.string())}),
                   tmp_path / name)
    return name


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    """Xóa cache giữa các test: nhiều test ghi nhiều bảng khác nhau cùng tên file."""
    data._SUBSET_CACHE.clear()
    data._OVERRIDE_CACHE.clear()


def test_without_a_subset_the_whole_train_partition_is_read(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    assert data.read_columns(config, tmp_path, "train", ["pair_id"]).num_rows == 6


def test_a_subset_keeps_only_the_listed_rows_of_train(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["train_subset"] = write_subset(tmp_path, ["p0", "p4"])
    table = data.read_columns(config, tmp_path, "train", ["pair_id"])
    assert table.column("pair_id").to_pylist() == ["p0", "p4"]


def test_a_subset_never_touches_validation(tmp_path: Path) -> None:
    # Điều kiện cốt yếu: anh Tiến chốt giữ nguyên tập validation (dòng 99 biên bản).
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["train_subset"] = write_subset(tmp_path, ["p0"])
    assert data.read_columns(config, tmp_path, "validation", ["pair_id"]).num_rows == 2


def test_the_subset_key_column_does_not_leak_into_the_result(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["train_subset"] = write_subset(tmp_path, ["p0", "p1"])
    table = data.read_columns(config, tmp_path, "train", ["endpoint"])
    assert sorted(table.schema.names) == ["endpoint", "partition"]
    assert table.num_rows == 2


def test_the_subset_key_column_is_kept_when_it_was_asked_for(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["train_subset"] = write_subset(tmp_path, ["p2"])
    table = data.read_columns(config, tmp_path, "train", ["pair_id", "endpoint"])
    assert sorted(table.schema.names) == ["endpoint", "pair_id", "partition"]
    assert table.column("pair_id").to_pylist() == ["p2"]


def test_a_pair_id_that_is_not_in_train_is_an_error(tmp_path: Path) -> None:
    """Dựng subset từ lần export khác là lỗi, không được bỏ qua im lặng.

    p6 thuộc validation, nên nếu nó có trong bảng subset thì bảng đó không khớp dataset và
    một phần dữ liệu sẽ bị thiếu mà không báo gì.
    """
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["train_subset"] = write_subset(tmp_path, ["p0", "p6"])
    with pytest.raises(ValueError, match="subset"):
        data.read_columns(config, tmp_path, "train", ["pair_id"])


def test_a_missing_subset_file_points_at_the_builder(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["train_subset"] = "work/splits/khong_co.parquet"
    with pytest.raises(FileNotFoundError, match="build_train_subset"):
        data.read_columns(config, tmp_path, "train", ["pair_id"])


def test_an_empty_subset_is_an_error(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["train_subset"] = write_subset(tmp_path, [])
    with pytest.raises(ValueError, match="rỗng"):
        data.read_columns(config, tmp_path, "train", ["pair_id"])


def test_a_duplicated_pair_id_in_the_subset_is_an_error(tmp_path: Path) -> None:
    # Trùng pair_id sẽ nhân đôi dòng đó trong train một cách âm thầm.
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["train_subset"] = write_subset(tmp_path, ["p0", "p0"])
    with pytest.raises(ValueError, match="trùng"):
        data.read_columns(config, tmp_path, "train", ["pair_id"])


def test_the_subset_works_together_with_a_partition_override(tmp_path: Path) -> None:
    """Hai đường độc lập: override chọn protein nào là train, subset chọn dòng nào được giữ."""
    config = write_dataset(tmp_path)
    override = pa.table({"protein_hash": pa.array(["ha", "hb", "hc"], pa.string()),
                         "partition": pa.array(["train", "validation", "train"],
                                               pa.string())})
    pq.write_table(override, tmp_path / "override.parquet")
    config["stage3"]["inputs"]["partition_override"] = "override.parquet"
    # Sau override, train gồm p0..p2 (ha) và p6, p7 (hc).
    config["stage3"]["inputs"]["train_subset"] = write_subset(tmp_path, ["p1", "p7"])
    table = data.read_columns(config, tmp_path, "train", ["pair_id"])
    assert table.column("pair_id").to_pylist() == ["p1", "p7"]
    assert data.read_columns(config, tmp_path, "validation", ["pair_id"]).num_rows == 3
