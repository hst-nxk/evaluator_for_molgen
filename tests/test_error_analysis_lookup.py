"""Lệnh con `lookup` của error_analysis.py: đúng chiều theo tần suất phân tử trong train.

Sinh ra ngày 14/11/2026 để kiểm giả thuyết TRA CỨU đứng sau việc Morgan thắng MoLFormer
+0,1186 trên cách chia theo cặp. Những chỗ dễ sai âm thầm, mỗi chỗ một test:

* Đếm phân tử phải tính CẢ HAI phía A và B của mỗi dòng train.
* Thùng lấy theo min(A, B), không phải max và không phải tổng: tra cứu cần cả hai phía.
* Dump train KHÔNG được cắt lát (nó là kho đối chiếu); chỉ validation mới cắt.
* Dấu hiệu "cùng protein" phải đòi ĐÚNG protein của dòng validation, không phải protein nào.
* Các thùng phải phủ kín và không chồng nhau, nếu không thì tổng số dòng không khớp.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import error_analysis as analysis  # noqa: E402  (scripts/ không phải package)


def rows_table(rows: list[tuple], unseen: list[bool] | None = None) -> pa.Table:
    """Một rows.parquet tối thiểu: (smiles_a, smiles_b, protein, delta_p, prediction)."""
    size = len(rows)
    truth = np.array([row[3] for row in rows], dtype=np.float64)
    prediction = np.array([row[4] for row in rows], dtype=np.float64)
    # Quy ước của dự án: chỉ đếm dòng có |Δp| > 0,3; đúng chiều = cùng dấu.
    counted = np.abs(truth) > 0.3
    correct = np.sign(truth) == np.sign(prediction)
    data = {
        "smiles_a": pa.array([row[0] for row in rows], pa.string()),
        "smiles_b": pa.array([row[1] for row in rows], pa.string()),
        "protein_hash": pa.array([row[2] for row in rows], pa.string()),
        "endpoint": pa.array(["Ki"] * size, pa.string()),
        "delta_p": pa.array(truth),
        "prediction": pa.array(prediction),
        "sign_counted": pa.array(counted),
        "sign_correct": pa.array(correct),
        "protein_rows": pa.array([100] * size, pa.int64()),
    }
    if unseen is not None:
        data["unseen_pair"] = pa.array(unseen, pa.bool_())
    return pa.table(data)


def write_dump(folder: Path, table: pa.Table) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, folder / analysis.ROWS_FILE)
    return folder


def namespace(train: Path, validation: Path, **extra) -> "analysis.argparse.Namespace":
    options = {"train": str(train), "validation": str(validation), "slice": "all",
               "label": "", "out": ""}
    options.update(extra)
    return analysis.argparse.Namespace(**options)


# ------------------------------------------------------- đếm phân tử

def test_both_sides_of_a_train_row_are_counted() -> None:
    table = rows_table([("C", "CC", "ha", 1.0, 1.0)])
    counts, _ = analysis.molecule_occurrences(table)
    assert counts == {"C": 1, "CC": 1}


def test_a_molecule_appearing_on_both_sides_is_counted_twice() -> None:
    table = rows_table([("C", "CC", "ha", 1.0, 1.0),
                        ("CC", "CCC", "ha", 1.0, 1.0)])
    counts, _ = analysis.molecule_occurrences(table)
    assert counts["CC"] == 2
    assert counts["C"] == 1 and counts["CCC"] == 1


def test_the_same_protein_set_keys_on_the_exact_protein() -> None:
    table = rows_table([("C", "CC", "ha", 1.0, 1.0)])
    _, seen = analysis.molecule_occurrences(table)
    assert ("C", "ha") in seen and ("CC", "ha") in seen
    assert ("C", "hb") not in seen


# ------------------------------------------------------- thùng tần suất

def test_the_bucket_uses_the_minimum_of_the_two_sides(tmp_path: Path) -> None:
    """A thấy 3 lần, B thấy 0 lần -> thùng 0, KHÔNG phải thùng 2-4."""
    train = write_dump(tmp_path / "train", rows_table([
        ("C", "CC", "ha", 1.0, 1.0),
        ("C", "CCC", "ha", 1.0, 1.0),
        ("C", "CCCC", "ha", 1.0, 1.0)]))
    validation = write_dump(tmp_path / "val", rows_table([
        ("C", "LAZ", "ha", 1.0, 1.0)]))
    analysis.run_lookup(namespace(train, validation, out=str(tmp_path / "r.json")))
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    buckets = {entry["nhãn"]: entry["dòng"]
               for entry in report["theo min(số lần thấy A, số lần thấy B)"]}
    assert buckets["0"] == 1
    assert sum(buckets.values()) == 1


def test_the_buckets_cover_every_row_exactly_once(tmp_path: Path) -> None:
    train_rows = []
    # "HOT" xuất hiện 30 lần, "MID" 3 lần, "ONE" 1 lần.
    for index in range(15):
        train_rows.append(("HOT", f"X{index}", "ha", 1.0, 1.0))
        train_rows.append((f"Y{index}", "HOT", "ha", 1.0, 1.0))
    for index in range(3):
        train_rows.append(("MID", f"Z{index}", "ha", 1.0, 1.0))
    train_rows.append(("ONE", "W0", "ha", 1.0, 1.0))
    train = write_dump(tmp_path / "train", rows_table(train_rows))
    validation = write_dump(tmp_path / "val", rows_table([
        ("HOT", "HOT", "ha", 1.0, 1.0),
        ("MID", "MID", "ha", 1.0, 1.0),
        ("ONE", "ONE", "ha", 1.0, 1.0),
        ("LAZ", "LAZ", "ha", 1.0, 1.0)]))
    analysis.run_lookup(namespace(train, validation, out=str(tmp_path / "r.json")))
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    buckets = {entry["nhãn"]: entry["dòng"]
               for entry in report["theo min(số lần thấy A, số lần thấy B)"]}
    assert buckets == {"0": 1, "1": 1, "2-4": 1, "5-19": 0, ">=20": 1}
    assert sum(buckets.values()) == report["rows"] == 4


# ------------------------------------------------------- lát cắt

def test_only_the_validation_dump_is_sliced(tmp_path: Path) -> None:
    """Train có unseen_pair=False khắp nơi; lát strict KHÔNG được làm rỗng kho đối chiếu."""
    train = write_dump(tmp_path / "train", rows_table(
        [("C", "CC", "ha", 1.0, 1.0)], unseen=[False]))
    validation = write_dump(tmp_path / "val", rows_table(
        [("C", "CC", "hb", 1.0, 1.0), ("LAZ", "LAZ", "hb", 1.0, 1.0)],
        unseen=[True, True]))
    analysis.run_lookup(namespace(train, validation, slice="strict",
                                  out=str(tmp_path / "r.json")))
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["phân tử phân biệt trong train"] == 2
    assert report["rows"] == 2


def test_a_strict_slice_that_drops_rows_is_honoured(tmp_path: Path) -> None:
    train = write_dump(tmp_path / "train", rows_table(
        [("C", "CC", "ha", 1.0, 1.0)], unseen=[False]))
    validation = write_dump(tmp_path / "val", rows_table(
        [("C", "CC", "hb", 1.0, 1.0), ("LAZ", "LAZ", "hb", 1.0, 1.0)],
        unseen=[True, False]))
    analysis.run_lookup(namespace(train, validation, slice="strict",
                                  out=str(tmp_path / "r.json")))
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["rows"] == 1


# ------------------------------------------------------- bảng "cùng protein"

def test_the_same_protein_table_needs_both_molecules_and_the_right_protein(
        tmp_path: Path) -> None:
    train = write_dump(tmp_path / "train", rows_table([
        ("C", "CC", "ha", 1.0, 1.0),
        ("C", "CCC", "hb", 1.0, 1.0)]))
    validation = write_dump(tmp_path / "val", rows_table([
        ("C", "CC", "ha", 1.0, 1.0),    # cả hai đã thấy cùng ha -> có
        ("C", "CC", "hb", 1.0, 1.0),    # CC chưa thấy cùng hb   -> không
        ("C", "CCC", "ha", 1.0, 1.0)]))  # CCC chưa thấy cùng ha  -> không
    analysis.run_lookup(namespace(train, validation, out=str(tmp_path / "r.json")))
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["cả A và B đã thấy cùng protein"]["dòng"] == 1
    assert report["không"]["dòng"] == 2


# ------------------------------------------------------- số đo

def test_sign_accuracy_follows_the_project_convention(tmp_path: Path) -> None:
    """Dòng |Δp| <= 0,3 bị loại khỏi MẪU SỐ, đúng quy ước direction_tolerance của dự án."""
    train = write_dump(tmp_path / "train", rows_table([("C", "CC", "ha", 1.0, 1.0)]))
    validation = write_dump(tmp_path / "val", rows_table([
        ("C", "CC", "ha", 1.0, 1.0),     # đếm, đúng chiều
        ("C", "CC", "ha", 1.0, -1.0),    # đếm, sai chiều
        ("C", "CC", "ha", 0.1, -1.0)]))  # KHÔNG đếm (|Δp| <= 0,3)
    analysis.run_lookup(namespace(train, validation, out=str(tmp_path / "r.json")))
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    everything = report["tất cả"]
    assert everything["mẫu số"] == 2
    assert everything["đúng chiều"] == pytest.approx(0.5)
    assert everything["dòng"] == 3


def test_mae_is_computed_on_the_rows_of_that_bucket_only(tmp_path: Path) -> None:
    train = write_dump(tmp_path / "train", rows_table([("C", "CC", "ha", 1.0, 1.0)]))
    validation = write_dump(tmp_path / "val", rows_table([
        ("C", "CC", "ha", 1.0, 1.0),       # đã thấy cả hai -> thùng 1, MAE 0
        ("LAZ", "LAZ", "ha", 1.0, 3.0)]))  # chưa thấy      -> thùng 0, MAE 2
    analysis.run_lookup(namespace(train, validation, out=str(tmp_path / "r.json")))
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    buckets = {entry["nhãn"]: entry for entry in
               report["theo min(số lần thấy A, số lần thấy B)"]}
    assert buckets["1"]["mae"] == pytest.approx(0.0)
    assert buckets["0"]["mae"] == pytest.approx(2.0)


def test_an_empty_bucket_reports_none_instead_of_crashing(tmp_path: Path) -> None:
    train = write_dump(tmp_path / "train", rows_table([("C", "CC", "ha", 1.0, 1.0)]))
    validation = write_dump(tmp_path / "val", rows_table([("C", "CC", "ha", 1.0, 1.0)]))
    analysis.run_lookup(namespace(train, validation, out=str(tmp_path / "r.json")))
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    empty = [entry for entry in report["theo min(số lần thấy A, số lần thấy B)"]
             if entry["dòng"] == 0]
    assert empty and all(entry["mae"] is None and entry["đúng chiều"] is None
                         for entry in empty)


def test_the_row_share_sums_to_one_over_the_buckets(tmp_path: Path) -> None:
    train = write_dump(tmp_path / "train", rows_table([
        ("C", "CC", "ha", 1.0, 1.0), ("C", "CCC", "ha", 1.0, 1.0)]))
    validation = write_dump(tmp_path / "val", rows_table([
        ("C", "CC", "ha", 1.0, 1.0), ("LAZ", "C", "ha", 1.0, 1.0),
        ("CCC", "C", "ha", 1.0, 1.0)]))
    analysis.run_lookup(namespace(train, validation, out=str(tmp_path / "r.json")))
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    total = sum(entry["tỉ lệ dòng"]
                for entry in report["theo min(số lần thấy A, số lần thấy B)"])
    assert total == pytest.approx(1.0)


# ------------------------------------------------------- lỗi đầu vào

def test_a_dump_without_smiles_columns_says_to_rerun(tmp_path: Path) -> None:
    table = rows_table([("C", "CC", "ha", 1.0, 1.0)])
    table = table.drop(["smiles_a"])
    train = write_dump(tmp_path / "train", table)
    validation = write_dump(tmp_path / "val", rows_table([("C", "CC", "ha", 1.0, 1.0)]))
    with pytest.raises(ValueError, match="mode errors"):
        analysis.run_lookup(namespace(train, validation))


def test_a_missing_dump_names_the_command_that_builds_it(tmp_path: Path) -> None:
    validation = write_dump(tmp_path / "val", rows_table([("C", "CC", "ha", 1.0, 1.0)]))
    with pytest.raises(FileNotFoundError, match="mode errors"):
        analysis.run_lookup(namespace(tmp_path / "khong_co", validation))


def test_the_bucket_labels_match_the_edges() -> None:
    assert analysis.lookup_bucket_labels() == ["0", "1", "2-4", "5-19", ">=20"]
    assert len(analysis.lookup_bucket_labels()) == len(analysis.LOOKUP_EDGES)
