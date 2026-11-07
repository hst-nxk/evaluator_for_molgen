"""scripts/count_queries.py: đếm query theo khóa gom nhóm do người dùng chọn.

Điểm cần khóa chặt: con số query THAY ĐỔI theo khóa gom, nên mọi kết quả in ra phải nói
rõ khóa đã dùng (sự cố lệch 669.035 / 636.118 ngày 07/11/2026).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import count_queries as counter  # noqa: E402  (scripts/ không phải package)

# Hai protein: P1 có 2 query (mỗi query 2 dòng), P2 có 1 query (1 dòng).
ROWS = [
    ("train", "P1", "CCO", "Ki", "CC", "O"),
    ("train", "P1", "CCO", "Ki", "CC", "N"),
    ("train", "P1", "CCN", "Ki", "CC", "O"),
    ("train", "P1", "CCN", "Ki", "CC", "F"),
    ("train", "P2", "CCC", "IC50", "CC", "C"),
    ("validation", "P3", "CCO", "Ki", "CC", "O"),
]
COLUMNS = ["partition", "protein_hash", "smiles_a", "endpoint", "constant_smiles",
           "variable_a"]


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    """Một bộ dữ liệu giả tí hon, đúng tên file mà script đi tìm."""
    folder = tmp_path / "dataset_test"
    folder.mkdir()
    frame = pd.DataFrame(ROWS, columns=COLUMNS)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False),
                   folder / counter.EVALUATOR_FILE)
    return folder


def test_key_is_split_on_commas() -> None:
    assert counter.parse_key("a,b , c") == ["a", "b", "c"]


def test_an_empty_key_is_refused_instead_of_guessed() -> None:
    with pytest.raises(ValueError, match="--key"):
        counter.parse_key(" , ")


def test_counts_rows_queries_and_proteins_of_one_partition(dataset: Path) -> None:
    result = counter.report(dataset, list(counter.DEFAULT_KEY))
    train = result["partitions"]["train"]
    assert train["rows"] == 5
    assert train["queries"] == 3
    assert train["proteins"] == 2
    assert train["rows_per_query"] == pytest.approx(5 / 3)


def test_the_count_changes_with_the_grouping_key(dataset: Path) -> None:
    """Lý do tồn tại của script: khóa khác thì số khác, cả hai đều đúng."""
    by_molecule = counter.report(dataset, ["protein_hash", "smiles_a", "endpoint"])
    by_fragment = counter.report(dataset, ["protein_hash", "constant_smiles", "endpoint"])
    assert by_molecule["partitions"]["train"]["queries"] == 3
    # Gom theo khung chung thì CCO và CCN nhập làm một -> ít query hơn.
    assert by_fragment["partitions"]["train"]["queries"] == 2


def test_the_printed_table_always_states_the_key_used(dataset: Path) -> None:
    text = counter.render(counter.report(dataset, ["protein_hash", "smiles_a"]))
    assert "protein_hash + smiles_a" in text
    assert "endpoint" not in text.splitlines()[1]


def test_protein_concentration_is_measured_in_queries(dataset: Path) -> None:
    result = counter.report(dataset, list(counter.DEFAULT_KEY))
    top = result["partitions"]["train"]["concentration"]["top_50pct"]
    # 2 protein, nửa trên = 1 protein = P1, giữ 2 trong 3 query.
    assert top["proteins"] == 1
    assert top["queries"] == 2
    assert top["share"] == pytest.approx(2 / 3)


def test_spread_reports_the_thin_tail(dataset: Path) -> None:
    spread = counter.report(dataset, list(counter.DEFAULT_KEY))["partitions"]["train"]
    assert spread["spread"]["largest_protein_rows"] == 4
    assert spread["spread"]["proteins_with_one_row"] == 1
    assert spread["spread"]["share_with_one_row"] == pytest.approx(0.5)


def test_a_missing_column_is_named_in_the_error(dataset: Path) -> None:
    with pytest.raises(ValueError, match="khong_co_cot_nay"):
        counter.report(dataset, ["khong_co_cot_nay"])


def test_a_missing_dataset_is_reported_not_silently_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        counter.report(tmp_path / "khong-ton-tai", list(counter.DEFAULT_KEY))


def test_the_script_never_touches_the_data_folder(dataset: Path) -> None:
    """Script chỉ đọc; chạy xong thư mục dữ liệu phải y nguyên."""
    before = {p.name: p.stat().st_mtime_ns for p in dataset.iterdir()}
    counter.report(dataset, list(counter.DEFAULT_KEY))
    after = {p.name: p.stat().st_mtime_ns for p in dataset.iterdir()}
    assert before == after
