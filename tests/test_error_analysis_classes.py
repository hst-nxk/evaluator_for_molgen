"""scripts/error_analysis.py lệnh `classes`: góc classification của việc 8 (dòng 73, 139).

Hai thứ phải khóa chặt, vì chúng đi thẳng vào báo cáo 16/11/2026:

1. Lát cắt. `gap` và `classes` giờ có `--slice`; `strict` phải lọc theo đúng cột
   `unseen_pair` của rows.parquet, và với dump cũ (trước 10/11/2026, chưa có cột đó) phải
   BÁO LỖI chứ không được im lặng trả về lát `all` — đó đúng là cái bẫy mục 5 nhật ký
   13/11 cảnh báo.
2. Con số. Ma trận nhầm và F1 phải trùng khít metrics.change_class_metrics của dự án
   (tức trùng report.json), không được tính lại theo cách khác ở script phân tích.
"""

from __future__ import annotations

import argparse
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

from molopt.evaluator import metrics as metric_module  # noqa: E402


def write_rows(folder: Path, rows: list[dict], with_unseen: bool = True) -> Path:
    """Dựng thư mục errors_* giả; with_unseen=False mô phỏng dump trước 10/11/2026."""
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "smiles_a": [row.get("a", "C") for row in rows],
        "smiles_b": [row.get("b", "CC") for row in rows],
        "endpoint": [row.get("endpoint", "Ki") for row in rows],
        "protein_hash": [row.get("protein", "P1") for row in rows],
        "protein_rows": [row.get("protein_rows", 100) for row in rows],
        "delta_p": [float(row["truth"]) for row in rows],
        "prediction": [float(row["prediction"]) for row in rows],
        "sign_counted": [True] * len(rows),
        "sign_correct": [bool(np.sign(row["truth"]) == np.sign(row["prediction"]))
                         for row in rows],
    }
    if with_unseen:
        data["unseen_pair"] = [bool(row.get("unseen", True)) for row in rows]
    pq.write_table(pa.table(data), folder / analysis.ROWS_FILE)
    return folder


def arguments(folder: Path, **changes) -> argparse.Namespace:
    values = {"train": "", "validation": str(folder), "slice": "strict",
              "threshold": None, "out": ""}
    values.update(changes)
    return argparse.Namespace(**values)


# ---------------------------------------------------------------- lát cắt

def test_strict_keeps_only_unseen_pairs(tmp_path: Path) -> None:
    folder = write_rows(tmp_path / "errors", [
        {"truth": 1.0, "prediction": 0.9, "unseen": True},
        {"truth": -1.0, "prediction": -0.8, "unseen": False},
        {"truth": 2.0, "prediction": 0.1, "unseen": True},
    ])
    table = analysis.read_rows(folder)
    assert table.num_rows == 3
    kept = analysis.select_slice(table, "strict")
    assert kept.num_rows == 2
    assert sorted(kept.column("delta_p").to_pylist()) == [1.0, 2.0]


def test_all_returns_the_table_untouched(tmp_path: Path) -> None:
    folder = write_rows(tmp_path / "errors", [
        {"truth": 1.0, "prediction": 0.9, "unseen": True},
        {"truth": -1.0, "prediction": -0.8, "unseen": False},
    ])
    table = analysis.read_rows(folder)
    assert analysis.select_slice(table, "all") is table


def test_an_old_dump_without_the_column_raises_instead_of_silently_giving_all(
        tmp_path: Path) -> None:
    """Cái bẫy phải chặn: im lặng trả lát `all` khi người đọc đang tưởng là `strict`."""
    folder = write_rows(tmp_path / "errors",
                        [{"truth": 1.0, "prediction": 0.9}], with_unseen=False)
    table = analysis.read_rows(folder)
    with pytest.raises(ValueError, match="unseen_pair"):
        analysis.select_slice(table, "strict")


def test_an_unknown_slice_name_raises(tmp_path: Path) -> None:
    folder = write_rows(tmp_path / "errors", [{"truth": 1.0, "prediction": 0.9}])
    with pytest.raises(ValueError, match="multi_assay"):
        analysis.select_slice(analysis.read_rows(folder), "multi_assay")


def test_strict_with_no_unseen_row_raises(tmp_path: Path) -> None:
    folder = write_rows(tmp_path / "errors",
                        [{"truth": 1.0, "prediction": 0.9, "unseen": False}])
    with pytest.raises(ValueError, match="không còn dòng nào"):
        analysis.select_slice(analysis.read_rows(folder), "strict")


# ---------------------------------------------------------------- ngưỡng

def test_the_default_threshold_comes_from_config_not_from_the_script() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    assert analysis.config_threshold() == pytest.approx(
        float(config["stage3"]["metrics"]["change_class_threshold"]))


# ---------------------------------------------------------------- số đo

def test_classes_matches_the_project_metric_exactly(tmp_path: Path) -> None:
    """Ma trận nhầm và F1 phải trùng metrics.change_class_metrics, không tính lại cách khác."""
    rows = [{"truth": 1.2, "prediction": 0.9}, {"truth": -1.5, "prediction": -0.1},
            {"truth": 0.0, "prediction": 0.05}, {"truth": 2.0, "prediction": 0.2},
            {"truth": -0.9, "prediction": -1.1}, {"truth": 0.1, "prediction": 0.4}]
    folder = write_rows(tmp_path / "errors", rows)
    assert analysis.run_classes(arguments(folder)) == 0
    written = json.loads((folder / "classes.json").read_text(encoding="utf-8"))
    block = written["validation"]
    expected = metric_module.change_class_metrics(
        np.array([row["truth"] for row in rows], dtype=np.float64),
        np.array([row["prediction"] for row in rows], dtype=np.float64),
        analysis.config_threshold())
    assert block["confusion"] == expected["confusion"]
    assert block["accuracy"] == pytest.approx(expected["accuracy"])
    assert block["macro_f1"] == pytest.approx(expected["macro_f1"])
    assert block["rows"] == len(rows)
    assert block["slice"] == "strict"


def test_every_confusion_row_sums_to_the_count_of_that_true_class(tmp_path: Path) -> None:
    folder = write_rows(tmp_path / "errors", [
        {"truth": 1.0, "prediction": 0.1}, {"truth": 1.0, "prediction": 1.0},
        {"truth": -1.0, "prediction": 0.0}, {"truth": 0.0, "prediction": 0.0},
    ])
    analysis.run_classes(arguments(folder))
    block = json.loads((folder / "classes.json").read_text(encoding="utf-8"))["validation"]
    confusion = np.asarray(block["confusion"])
    for index, name in enumerate(metric_module.CHANGE_CLASSES):
        assert int(confusion[index].sum()) == block[f"{name}_rows"]
    assert int(confusion.sum()) == block["rows"]


def test_a_model_that_always_says_neutral_shows_up_as_one_bar(tmp_path: Path) -> None:
    """Chính hiện tượng hình này tồn tại để chỉ ra: dồn hết dự đoán vào lớp giữa."""
    folder = write_rows(tmp_path / "errors", [
        {"truth": 3.0, "prediction": 0.0}, {"truth": -3.0, "prediction": 0.0},
        {"truth": 0.0, "prediction": 0.0}, {"truth": 1.0, "prediction": 0.1},
    ])
    analysis.run_classes(arguments(folder))
    block = json.loads((folder / "classes.json").read_text(encoding="utf-8"))["validation"]
    assert block["prediction_share"]["neutral"] == pytest.approx(1.0)
    assert block["truth_share"]["neutral"] == pytest.approx(0.25)
    assert block["better_recall"] == pytest.approx(0.0)
    assert block["worse_recall"] == pytest.approx(0.0)


def test_both_partitions_land_in_one_figure_and_one_json(tmp_path: Path) -> None:
    train = write_rows(tmp_path / "errors_train", [{"truth": 1.0, "prediction": 0.9}])
    validation = write_rows(tmp_path / "errors_val", [{"truth": -1.0, "prediction": 0.0}])
    target = tmp_path / "figures" / "classes_test.png"
    assert analysis.run_classes(arguments(validation, train=str(train),
                                          out=str(target))) == 0
    assert target.is_file() and target.stat().st_size > 0
    written = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    assert sorted(written) == ["train", "validation"]


def test_the_threshold_flag_overrides_the_config_default(tmp_path: Path) -> None:
    """Cùng dump, ngưỡng rộng hơn thì dòng |Δp| = 0,5 chuyển từ 'tốt hơn' sang 'ngang'."""
    folder = write_rows(tmp_path / "errors", [{"truth": 0.5, "prediction": 0.5}])
    analysis.run_classes(arguments(folder, threshold=0.3,
                                   out=str(tmp_path / "a.png")))
    narrow = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))["validation"]
    analysis.run_classes(arguments(folder, threshold=1.0,
                                   out=str(tmp_path / "b.png")))
    wide = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))["validation"]
    assert narrow["better_rows"] == 1 and narrow["neutral_rows"] == 0
    assert wide["better_rows"] == 0 and wide["neutral_rows"] == 1


# ---------------------------------------------------------------- gap + lát

def test_gap_reads_the_slice_it_was_given(tmp_path: Path, capsys) -> None:
    """Trước 14/11/2026 `gap` không có tham số lát, nên cột validation của nó là lát all."""
    folder = write_rows(tmp_path / "errors", [
        {"truth": 1.0, "prediction": 1.0, "unseen": True},
        {"truth": -4.0, "prediction": 4.0, "unseen": False},
    ])
    analysis.run_gap(argparse.Namespace(train="", validation=str(folder),
                                        slice="strict", out=""))
    strict = capsys.readouterr().out
    analysis.run_gap(argparse.Namespace(train="", validation=str(folder),
                                        slice="all", out=""))
    every = capsys.readouterr().out
    assert "lát cắt: strict" in strict
    assert "KHÔNG dùng để so hai mô hình" in every
    # Lát strict chỉ còn dòng đoán hoàn hảo nên MAE = 0; lát all có thêm dòng lệch 8,0.
    assert "rows" in strict and strict.count("1") >= 1
    assert "0.0000" in strict
    assert "0.0000" not in every.split("mae")[1].split("\n")[0]


# ------------------------------------------------- lát strict trên train

def test_strict_on_train_falls_back_to_all_and_says_so(tmp_path: Path, capsys) -> None:
    """Mọi cặp của train đều đã thấy trong train, nên lát strict ở đó rỗng theo định nghĩa.

    Lùi về `all` thì được, nhưng phải NÓI RA: cảnh báo ra stderr, tên lát thật vào .json
    và vào tiêu đề hình. Lặng lẽ lùi là đổi thước đo giữa chừng.
    """
    train = write_rows(tmp_path / "errors_train",
                       [{"truth": 1.0, "prediction": 0.9, "unseen": False},
                        {"truth": -1.0, "prediction": -0.9, "unseen": False}])
    validation = write_rows(tmp_path / "errors_val",
                            [{"truth": 2.0, "prediction": 0.1, "unseen": True},
                             {"truth": -2.0, "prediction": 0.0, "unseen": False}])
    target = tmp_path / "classes.png"
    assert analysis.run_classes(arguments(validation, train=str(train),
                                          out=str(target))) == 0
    captured = capsys.readouterr()
    assert "CẢNH BÁO" in captured.err and "strict rỗng trên train" in captured.err
    written = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    assert written["train"]["slice"] == "all (strict rỗng trên train)"
    assert written["train"]["slice_requested"] == "strict"
    assert written["train"]["rows"] == 2
    # validation vẫn được lọc thật: chỉ còn 1 dòng unseen_pair = True.
    assert written["validation"]["slice"] == "strict"
    assert written["validation"]["rows"] == 1


def test_strict_on_validation_still_raises_when_empty(tmp_path: Path) -> None:
    """Ở validation, lát strict rỗng là dấu hiệu hỏng — phải dừng, không được lùi."""
    folder = write_rows(tmp_path / "errors",
                        [{"truth": 1.0, "prediction": 0.9, "unseen": False}])
    with pytest.raises(ValueError, match="không còn dòng nào"):
        analysis.run_classes(arguments(folder))


def test_train_with_some_unseen_pairs_is_filtered_not_waved_through(tmp_path: Path) -> None:
    """Nếu train THẬT SỰ có cặp unseen thì vẫn lọc bình thường, không lùi."""
    folder = write_rows(tmp_path / "errors_train",
                        [{"truth": 1.0, "prediction": 0.9, "unseen": True},
                         {"truth": -1.0, "prediction": -0.9, "unseen": False}])
    table, used = analysis.partition_slice(analysis.read_rows(folder), "strict", "train")
    assert used == "strict" and table.num_rows == 1


def test_gap_reports_the_slice_it_actually_used_per_partition(tmp_path: Path) -> None:
    train = write_rows(tmp_path / "errors_train",
                       [{"truth": 1.0, "prediction": 0.9, "unseen": False}])
    validation = write_rows(tmp_path / "errors_val",
                            [{"truth": 2.0, "prediction": 0.1, "unseen": True}])
    out = tmp_path / "gap.json"
    analysis.run_gap(argparse.Namespace(train=str(train), validation=str(validation),
                                        slice="strict", out=str(out)))
    blocks = json.loads(out.read_text(encoding="utf-8"))
    assert blocks["train"]["slice"].startswith("all")
    assert blocks["validation"]["slice"] == "strict"
