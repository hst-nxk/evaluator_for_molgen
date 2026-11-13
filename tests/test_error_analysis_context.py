"""scripts/error_analysis.py lệnh `context`: cùng Context P, KHÁC cặp (A, B).

Đây là lát (b) của việc 9 họp 09/11/2026 — nửa còn lại của `flip`, vốn chỉ đo lát (a)
"cùng cặp, khác Context P".

Điểm phải khóa chặt: nhóm phải gom theo (protein_hash, endpoint) chứ không chỉ theo
protein (Δp của IC50 và của Ki không so thẳng được), và phép tách phương sai phải đúng
luật phương sai toàn phần — hai con số đó là thứ sẽ đi vào báo cáo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import error_analysis as analysis  # noqa: E402  (scripts/ không phải package)


def write_rows(folder: Path, rows: list[dict]) -> Path:
    """Dựng một thư mục errors_* giả với đủ các cột mà read_rows đòi."""
    folder.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "smiles_a": [row["a"] for row in rows],
        "smiles_b": [row["b"] for row in rows],
        "endpoint": [row.get("endpoint", "Ki") for row in rows],
        "protein_hash": [row["protein"] for row in rows],
        "protein_rows": [row.get("protein_rows", 100) for row in rows],
        "delta_p": [float(row["truth"]) for row in rows],
        "prediction": [float(row["prediction"]) for row in rows],
        "sign_counted": [True] * len(rows),
        "sign_correct": [bool(np.sign(row["truth"]) == np.sign(row["prediction"]))
                         for row in rows],
    })
    pq.write_table(table, folder / analysis.ROWS_FILE)
    return folder


def arguments(folder: Path, **changes) -> object:
    """Namespace đúng các khóa mà run_context đọc, mặc định như dòng lệnh."""
    import argparse

    values = {"rows": str(folder), "min_pairs": 2, "max_rows_per_group": 500,
              "min_rows_for_correlation": 5, "tolerance": 0.3, "worst": 0,
              "worst_out": "", "seed": 20261023, "out": ""}
    values.update(changes)
    return argparse.Namespace(**values)


def test_the_same_protein_with_two_endpoints_is_two_contexts(tmp_path: Path) -> None:
    """Δp của Ki và của IC50 không so được với nhau, nên chúng KHÔNG chung một nhóm."""
    table = pq.read_table(write_rows(tmp_path / "errors", [
        {"a": "C", "b": "CC", "protein": "P1", "endpoint": "Ki", "truth": 1.0,
         "prediction": 0.5},
        {"a": "C", "b": "CCC", "protein": "P1", "endpoint": "IC50", "truth": -1.0,
         "prediction": -0.5},
    ]) / analysis.ROWS_FILE)
    assert np.unique(analysis.context_codes(table)).size == 2


def test_rows_of_one_protein_and_endpoint_share_a_context(tmp_path: Path) -> None:
    table = pq.read_table(write_rows(tmp_path / "errors", [
        {"a": "C", "b": "CC", "protein": "P1", "truth": 1.0, "prediction": 0.5},
        {"a": "C", "b": "CCC", "protein": "P1", "truth": -1.0, "prediction": -0.5},
        {"a": "C", "b": "CC", "protein": "P2", "truth": 1.0, "prediction": 0.5},
    ]) / analysis.ROWS_FILE)
    codes = analysis.context_codes(table)
    assert codes[0] == codes[1] and codes[0] != codes[2]


def test_variance_split_follows_the_law_of_total_variance() -> None:
    """trong + giữa = toàn phần, với trọng số theo số dòng. Số tay, không chạy lại code."""
    values = np.array([0.0, 2.0, 10.0, 12.0])
    groups = [np.array([0, 1]), np.array([2, 3])]
    split = analysis.variance_split(values, groups)
    # Trung bình nhóm 1,0 và 11,0; phương sai trong mỗi nhóm 1,0 -> phần TRONG = 1,0.
    # Trung bình chung 6,0 -> phần GIỮA = (5^2 + 5^2) / 2 = 25. Toàn phần 26.
    assert split["sd trong nhóm"] == pytest.approx(1.0)
    assert split["sd giữa các nhóm"] == pytest.approx(5.0)
    assert split["sd toàn phần"] == pytest.approx(float(values.std()))
    assert split["tỉ lệ phương sai nằm giữa các nhóm"] == pytest.approx(25.0 / 26.0)


def test_variance_split_weighs_a_big_group_more_than_a_small_one() -> None:
    """Trọng số theo DÒNG, như mọi chỉ số khác của dự án — không phải trung bình các nhóm."""
    values = np.array([0.0, 0.0, 0.0, 10.0])
    groups = [np.array([0, 1, 2]), np.array([3])]
    split = analysis.variance_split(values, groups)
    assert split["sd toàn phần"] == pytest.approx(float(values.std()))


def test_a_model_that_ignores_the_context_has_no_variance_between_contexts(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Đúng hình dạng mà `flip` đã thấy: dự đoán đứng yên khi đổi protein.

    Nhãn khác hẳn nhau giữa hai protein, còn dự đoán thì chỉ phụ thuộc cặp (A, B). Phép
    tách phương sai phải chỉ ra điều đó: phần GIỮA các nhóm của dự đoán bằng 0.
    """
    rows = []
    for protein in ("P1", "P2"):
        offset = 0.0 if protein == "P1" else 5.0
        for index, pair in enumerate(("CC", "CCC", "CCCC", "CCCCC", "CCCCCC")):
            rows.append({"a": "C", "b": pair, "protein": protein,
                         "truth": offset + index, "prediction": float(index)})
    folder = write_rows(tmp_path / "errors", rows)
    assert analysis.run_context(arguments(folder)) == 0
    report = json.loads((folder / "context_pairs.json").read_text(encoding="utf-8"))
    assert report["số Context P giữ lại"] == 2
    assert report["phương sai DỰ ĐOÁN"]["sd giữa các nhóm"] == pytest.approx(0.0)
    assert report["phương sai NHÃN"]["sd giữa các nhóm"] == pytest.approx(2.5)
    # Trong từng protein thì thứ tự vẫn đúng hoàn toàn: đó chính là "học hóa học của cặp".
    assert report["đúng chiều TRONG Context P"] == pytest.approx(1.0)
    assert report["spearman trung bình trong nhóm"] == pytest.approx(1.0)


def test_a_context_with_one_pair_is_dropped(tmp_path: Path) -> None:
    """Nhóm chỉ có một cặp không trả lời được câu hỏi "phân biệt cặp này với cặp kia"."""
    folder = write_rows(tmp_path / "errors", [
        {"a": "C", "b": "CC", "protein": "P1", "truth": 1.0, "prediction": 0.5},
        {"a": "C", "b": "CC", "protein": "P1", "truth": 1.2, "prediction": 0.6},
        {"a": "C", "b": "CC", "protein": "P2", "truth": 1.0, "prediction": 0.5},
        {"a": "C", "b": "CCC", "protein": "P2", "truth": -1.0, "prediction": -0.5},
    ])
    assert analysis.run_context(arguments(folder)) == 0
    report = json.loads((folder / "context_pairs.json").read_text(encoding="utf-8"))
    assert report["số Context P giữ lại"] == 1
    assert report["dòng"] == 2


def test_worst_writes_only_rows_whose_sign_is_wrong(tmp_path: Path) -> None:
    """Bảng đọc tay phải là ca SAI QUY LUẬT, không phải ca lệch nhiều mà vẫn đúng chiều."""
    folder = write_rows(tmp_path / "errors", [
        {"a": "C", "b": "CC", "protein": "P1", "truth": 1.0, "prediction": 9.0},
        {"a": "C", "b": "CCC", "protein": "P1", "truth": 1.0, "prediction": -2.0},
        {"a": "C", "b": "CCCC", "protein": "P1", "truth": -1.0, "prediction": 3.0},
    ])
    assert analysis.run_context(arguments(folder, worst=10)) == 0
    lines = (folder / "worst_rows.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3                      # tiêu đề + hai dòng trái dấu
    assert "CCCC" in lines[1]                   # lệch 4,0, nặng nhất trong hai ca trái dấu
    assert "9.0" not in "\n".join(lines)        # ca lệch 8,0 nhưng ĐÚNG chiều: không vào


def test_missing_rows_file_says_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--mode errors"):
        analysis.run_context(arguments(tmp_path / "chưa-có"))
