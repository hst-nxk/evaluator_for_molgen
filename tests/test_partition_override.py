"""Đường đổi partition của Stage 3 (thí nghiệm đổi ngưỡng giống nhau của protein).

Không đọc dataset thật: mọi test dựng một parquet nhỏ trong tmp_path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molopt.evaluator import data

PROTEINS = ("ha", "hb", "hc")


def dataset_table() -> pa.Table:
    """Sáu dòng, ba protein, cách chia gốc: ha và hb train, hc validation."""
    rows = [("ha", "train"), ("ha", "train"), ("hb", "train"),
            ("hb", "train"), ("hc", "validation"), ("hc", "validation")]
    return pa.table({
        "pair_id": pa.array([f"p{number}" for number in range(len(rows))], pa.string()),
        "endpoint": pa.array(["Ki"] * len(rows), pa.string()),
        "protein_hash": pa.array([row[0] for row in rows], pa.string()),
        "partition": pa.array([row[1] for row in rows], pa.string()),
        "delta_p_mean": pa.array([1.0] * len(rows), pa.float64()),
    })


def write_dataset(tmp_path: Path) -> dict[str, Any]:
    """Ghi dataset nhỏ và trả về config trỏ vào nó, chưa bật đường đổi partition."""
    target = tmp_path / "evaluator_training.parquet"
    pq.write_table(dataset_table(), target)
    return {"stage3": {"inputs": {"evaluator": target.name, "partition_override": ""}}}


def write_override(tmp_path: Path, mapping: dict[str, str], name: str = "override.parquet"
                   ) -> str:
    """Ghi bảng đổi partition {protein: partition} và trả về tên file."""
    table = pa.table({"protein_hash": pa.array(list(mapping), pa.string()),
                      "partition": pa.array(list(mapping.values()), pa.string())})
    pq.write_table(table, tmp_path / name)
    return name


@pytest.fixture(autouse=True)
def clear_override_cache() -> None:
    """Xóa cache giữa các test: hai test có thể ghi hai bảng khác nhau cùng tên file."""
    data._OVERRIDE_CACHE.clear()


def test_without_an_override_the_dataset_partition_is_used(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    table = data.read_columns(config, tmp_path, "train", ["protein_hash"])
    assert table.num_rows == 4
    assert sorted(set(table.column("protein_hash").to_pylist())) == ["ha", "hb"]


def test_an_override_moves_whole_proteins_between_partitions(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override"] = write_override(
        tmp_path, {"ha": "train", "hb": "validation", "hc": "train"})
    train = data.read_columns(config, tmp_path, "train", ["protein_hash"])
    validation = data.read_columns(config, tmp_path, "validation", ["protein_hash"])
    assert sorted(set(train.column("protein_hash").to_pylist())) == ["ha", "hc"]
    assert sorted(set(validation.column("protein_hash").to_pylist())) == ["hb"]
    assert train.num_rows + validation.num_rows == 6


def test_the_override_does_not_leak_the_key_column_into_the_result(tmp_path: Path) -> None:
    """Cột protein_hash chỉ được thêm để tra bảng; không ai gọi nó thì phải bỏ đi."""
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override"] = write_override(
        tmp_path, {"ha": "train", "hb": "train", "hc": "validation"})
    table = data.read_columns(config, tmp_path, "train", ["endpoint"])
    assert sorted(table.schema.names) == ["endpoint", "partition"]
    assert table.num_rows == 4


def test_the_override_wins_over_the_column_stored_in_the_dataset(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override"] = write_override(
        tmp_path, {"ha": "validation", "hb": "validation", "hc": "validation"})
    table = data.read_columns(config, tmp_path, "validation", ["protein_hash"])
    assert table.num_rows == 6
    assert set(table.column("partition").to_pylist()) == {"validation"}


def test_a_protein_missing_from_the_override_is_an_error(tmp_path: Path) -> None:
    """Thiếu protein nghĩa là bảng dựng từ lần export khác; phải dừng, không chia một nửa."""
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override"] = write_override(
        tmp_path, {"ha": "train", "hb": "train"})
    with pytest.raises(ValueError, match="không nằm trong bảng đổi partition"):
        data.read_columns(config, tmp_path, "train", ["protein_hash"])


def test_a_duplicated_protein_in_the_override_is_an_error(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    table = pa.table({"protein_hash": pa.array(["ha", "ha", "hb", "hc"], pa.string()),
                      "partition": pa.array(["train", "validation", "train", "train"],
                                            pa.string())})
    pq.write_table(table, tmp_path / "dup.parquet")
    config["stage3"]["inputs"]["partition_override"] = "dup.parquet"
    with pytest.raises(ValueError, match="trùng"):
        data.read_columns(config, tmp_path, "train", ["protein_hash"])


def test_a_missing_override_file_is_an_error(tmp_path: Path) -> None:
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override"] = "khong_co.parquet"
    with pytest.raises(FileNotFoundError, match="build_identity_split"):
        data.read_columns(config, tmp_path, "train", ["protein_hash"])


def test_test_stays_blocked_even_with_an_override(tmp_path: Path) -> None:
    """Đổi cách chia không được mở đường đọc test."""
    config = write_dataset(tmp_path)
    config["stage3"]["inputs"]["partition_override"] = write_override(
        tmp_path, {"ha": "test", "hb": "test", "hc": "test"})
    with pytest.raises(ValueError, match="test"):
        data.read_columns(config, tmp_path, "test", ["protein_hash"])
