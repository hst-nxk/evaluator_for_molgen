"""Task 1 họp 09/11/2026: phân tích lỗi trên checkpoint hiện tại, không cần train lại.

Năm lệnh con, tất cả đều đọc thư mục `rows.parquet` do `--mode errors` ghi ra:

    python scripts/error_analysis.py gap     --train <dir_train> --validation <dir_val>
    python scripts/error_analysis.py hist    --train <dir_train> --validation <dir_val>
    python scripts/error_analysis.py flip    --validation <dir_val> [--train <dir_train>]
    python scripts/error_analysis.py context --rows <dir> [--worst 200]
    python scripts/error_analysis.py scale   --rows <dir>

với <dir> là runs/evaluator/errors_<cấu hình>_<đầu ra>_<partition>_<stamp>/.

Script CHỈ ĐỌC; mọi thứ nó ghi nằm trong thư mục --out (mặc định cạnh file rows.parquet).

VÌ SAO TỪNG LỆNH, và nó trả lời câu nào của ai:

    gap   — dòng 72 biên bản, anh Tiến: "lọc và đối chiếu trực tiếp metric của Regression
            (như MSE/MAE) giữa tập Train và tập Validation xem nó có bị chênh nhau nhiều
            như chỉ số Accuracy này không". Lệnh này in một bảng MSE/MAE/R²/Spearman/đúng
            chiều cho hai partition cạnh nhau.
    hist  — dòng 73 và 139: vẽ phân phối nhãn thật và nhãn dự đoán ĐÈ LÊN NHAU, cả train
            lẫn validation, "để biết model thực sự học tốt trên bài toán Regression hay là
            nó đang tệ trên cả Regression luôn".
    flip  — dòng 189, anh Tiến: nếu cùng cặp (A, B) mà Context P khác nhau cho Δp TRÁI DẤU,
            và mô hình chỉ học vẹt theo cặp (A, B) chứ không học vai trò của P, thì nó sẽ
            sai hoàn toàn trên validation. Lệnh này đo đúng điều đó: nó tách các cặp có đổi
            dấu giữa các P và so độ chính xác trên chúng với các cặp không đổi dấu. Có
            thêm --train thì đo tiếp phép thử mạnh hơn: với các dòng validation trùng cặp
            (A, B) với train, dự đoán của mô hình tương quan với nhãn THẬT của dòng
            validation hơn, hay với nhãn của dòng TRAIN cùng cặp hơn.
    context — NỬA CÒN LẠI của việc 9 họp 09/11/2026. Biên bản giao hai lát: (a) "cùng cặp
            (A, B), khác Context P" — đó là lệnh `flip`; và (b) "cùng Context P, khác cặp
            (A, B)" — đó là lệnh này. Câu hỏi của lát (b) ngược hẳn với lát (a): trong
            CÙNG MỘT protein, mô hình có phân biệt được cặp này với cặp kia không, hay nó
            trả về gần như một con số cho cả protein? Vì mọi protein của validation đều
            chưa từng thấy, đây là phép đo "đọc được Context P mới đến đâu".
            Lệnh này còn tách phương sai của dự đoán thành phần GIỮA các protein và phần
            TRONG một protein, rồi so với phân tách tương ứng của nhãn — hai cặp số này
            nói thẳng mô hình đang biến thiên theo chiều nào. Thêm --worst N thì ghi ra
            một bảng N dòng sai nặng nhất để ĐỌC TRỰC TIẾP, đúng như biên bản yêu cầu.
    scale — dòng 85, Tùng: Context P dưới 1.000 mẫu đúng chiều ~75 %, trên 50.000 mẫu
            ~89 %. Lệnh này bóc theo đúng các ngưỡng TUYỆT ĐỐI đó để tái lập con số, chứ
            không dùng ngưỡng phân vị như `--mode errors` đang làm.

QUY ƯỚC ĐÚNG CHIỀU lấy nguyên của dự án, không định nghĩa lại ở đây: cột `sign_counted`
và `sign_correct` của rows.parquet đã áp ngưỡng stage3.metrics.direction_tolerance (0,3),
tức cặp có hai nhãn lệch nhau không quá 0,3 bị loại khỏi mẫu số. Nhờ vậy mọi con số ở đây
so trực tiếp được với report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from molopt.evaluator import metrics as metric_module  # noqa: E402

ROWS_FILE = "rows.parquet"
# Ngưỡng của Tùng ở dòng 85 biên bản 09/11/2026, tính theo SỐ DÒNG của Context P trong
# chính partition đang xét (cột protein_rows của rows.parquet).
SCALE_EDGES = (1_000, 10_000, 50_000)


def read_rows(folder: Path) -> pa.Table:
    """Đọc rows.parquet của một thư mục --mode errors, kiểm các cột bắt buộc."""
    path = folder / ROWS_FILE if folder.is_dir() else folder
    if not path.is_file():
        raise FileNotFoundError(
            f"chưa có {path}; chạy trước:\n"
            f"  python run_evaluator.py --mode errors --checkpoint <file>.pt "
            f"--partition <train|validation>")
    table = pq.read_table(path)
    needed = {"delta_p", "prediction", "sign_counted", "sign_correct", "protein_hash",
              "protein_rows"}
    missing = sorted(needed - set(table.column_names))
    if missing:
        raise ValueError(f"{path} thiếu cột {missing}; file này do lần chạy --mode errors "
                         f"cũ ghi ra. Chạy lại --mode errors để có đủ cột.")
    return table


def columns(table: pa.Table, *names: str) -> tuple[np.ndarray, ...]:
    """Lấy nhiều cột thành mảng numpy float64 hoặc bool, theo kiểu của cột."""
    out = []
    for name in names:
        values = table.column(name).to_numpy(zero_copy_only=False)
        out.append(np.asarray(values))
    return tuple(out)


def sign_accuracy(counted: np.ndarray, correct: np.ndarray) -> tuple[float | None, int]:
    """(tỉ lệ đúng chiều, mẫu số) theo quy ước của dự án; None khi mẫu số bằng 0."""
    total = int(np.count_nonzero(counted))
    if total == 0:
        return None, 0
    return float(np.count_nonzero(correct & counted)) / total, total


def row_summary(table: pa.Table) -> dict[str, Any]:
    """Chỉ số regression và đúng chiều THEO DÒNG của một partition.

    Dùng lại metrics.regression_metrics của dự án để con số khớp report.json từng chữ số;
    MSE thêm vào đây vì hàm kia chỉ trả RMSE (MSE = RMSE²) và anh Tiến hỏi đúng MSE.
    """
    truth, prediction = columns(table, "delta_p", "prediction")
    counted, correct = columns(table, "sign_counted", "sign_correct")
    counted = counted.astype(bool)
    correct = correct.astype(bool)
    scores = dict(metric_module.regression_metrics(truth.astype(np.float64),
                                                   prediction.astype(np.float64)))
    rmse = scores.get("rmse")
    scores["mse"] = None if rmse is None else float(rmse) ** 2
    accuracy, denominator = sign_accuracy(counted, correct)
    scores["sign_accuracy"] = accuracy
    scores["sign_pairs_counted"] = denominator
    scores["rows"] = table.num_rows
    scores["truth_mean"] = float(truth.mean())
    scores["truth_sd"] = float(truth.std())
    scores["prediction_mean"] = float(prediction.mean())
    scores["prediction_sd"] = float(prediction.std())
    # Độ co: mô hình đoán dải hẹp hơn nhãn bao nhiêu lần. < 1 nghĩa là co về trung bình.
    scores["spread_ratio"] = (float(prediction.std()) / float(truth.std())
                              if truth.std() > 0 else None)
    return scores


def show(value: Any) -> str:
    """Một số thành chuỗi ngắn; None thành '-'."""
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.4f}"


def run_gap(args: argparse.Namespace) -> int:
    """Bảng chỉ số regression train vs validation cạnh nhau (dòng 72 biên bản)."""
    blocks = {}
    for label, folder in (("train", args.train), ("validation", args.validation)):
        if not folder:
            continue
        blocks[label] = row_summary(read_rows(Path(folder)))
    if not blocks:
        raise ValueError("cần ít nhất một trong --train hoặc --validation")
    keys = ("rows", "mse", "mae", "rmse", "r2", "spearman", "sign_accuracy",
            "sign_pairs_counted", "truth_sd", "prediction_sd", "spread_ratio")
    # Hiệu của hai SỐ ĐẾM không có nghĩa gì (train nhiều dòng hơn validation là chuyện
    # đương nhiên của cách chia tập), nên cột chênh để trống ở các dòng này thay vì in ra
    # một con số mà người đọc phải tự bỏ qua.
    count_keys = {"rows", "sign_pairs_counted"}
    labels = list(blocks)
    width = max(len(key) for key in keys) + 2
    print(f"{'chỉ số':<{width}}" + "".join(f"{label:>16}" for label in labels)
          + (f"{'chênh':>16}" if len(labels) == 2 else ""))
    print("-" * (width + 16 * len(labels) + (16 if len(labels) == 2 else 0)))
    for key in keys:
        line = f"{key:<{width}}" + "".join(f"{show(blocks[label].get(key)):>16}"
                                           for label in labels)
        if len(labels) == 2:
            first, second = (blocks[labels[0]].get(key), blocks[labels[1]].get(key))
            if key in count_keys:
                line += f"{'—':>16}"
            elif isinstance(first, (int, float)) and isinstance(second, (int, float)):
                line += f"{show(float(second) - float(first)):>16}"
        print(line)
    print()
    print("Đọc bảng này: nếu MAE/MSE của hai partition GẦN nhau mà đúng chiều chênh nhiều,")
    print("thì khoảng cách train-validation nằm ở việc XẾP HẠNG, không nằm ở hồi quy — tức")
    print("mô hình tệ trên hồi quy ở CẢ HAI tập. Nếu MAE train thấp hơn hẳn thì là overfit.")
    print("spread_ratio < 1 nghĩa là dự đoán co về trung bình so với nhãn.")
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(blocks, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(f"\nđã ghi {target}")
    return 0


def run_hist(args: argparse.Namespace) -> int:
    """Phân phối nhãn thật đè nhãn dự đoán, mỗi partition một panel (dòng 73, 139)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picked = [(label, Path(folder)) for label, folder in
              (("train", args.train), ("validation", args.validation)) if folder]
    if not picked:
        raise ValueError("cần ít nhất một trong --train hoặc --validation")
    figure, axes = plt.subplots(2, len(picked), figsize=(6.0 * len(picked), 8.0),
                                squeeze=False)
    edges = np.linspace(-args.range, args.range, args.bins + 1)
    report: dict[str, Any] = {}
    for index, (label, folder) in enumerate(picked):
        table = read_rows(folder)
        truth, prediction = columns(table, "delta_p", "prediction")
        top = axes[0][index]
        top.hist(truth, bins=edges, histtype="step", linewidth=1.6, density=True,
                 label=f"nhãn thật (sd {truth.std():.3f})")
        top.hist(prediction, bins=edges, histtype="step", linewidth=1.6, density=True,
                 label=f"dự đoán (sd {prediction.std():.3f})")
        top.set_title(f"{label}: {table.num_rows:,} dòng")
        top.set_xlabel("Δp")
        top.set_ylabel("mật độ")
        top.legend(fontsize=8)
        bottom = axes[1][index]
        # hexbin thay vì scatter: 6,6 triệu điểm thì scatter chỉ ra một khối đen.
        bottom.hexbin(truth, prediction, gridsize=60,
                      extent=(-args.range, args.range, -args.range, args.range),
                      bins="log", cmap="viridis")
        bottom.plot([-args.range, args.range], [-args.range, args.range], "r--",
                    linewidth=1.0, label="y = x (dự đoán hoàn hảo)")
        bottom.set_xlabel("Δp thật")
        bottom.set_ylabel("Δp dự đoán")
        bottom.legend(fontsize=8)
        report[label] = {
            "rows": table.num_rows,
            "truth": {"mean": float(truth.mean()), "sd": float(truth.std()),
                      "min": float(truth.min()), "max": float(truth.max())},
            "prediction": {"mean": float(prediction.mean()), "sd": float(prediction.std()),
                           "min": float(prediction.min()), "max": float(prediction.max())},
            "outside_range": int(np.count_nonzero(np.abs(truth) > args.range)),
        }
    figure.tight_layout()
    target = Path(args.out) if args.out else picked[0][1] / "distributions.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150)
    plt.close(figure)
    target.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nđã ghi {target}")
    print(f"       {target.with_suffix('.json')}")
    return 0


def pair_codes(table: pa.Table) -> np.ndarray:
    """Mã nhóm cho từng dòng theo (smiles_a, smiles_b, endpoint) — tức "cùng một cặp".

    Endpoint vào khóa vì Δp của hai endpoint khác nhau không so được với nhau.
    """
    needed = {"smiles_a", "smiles_b", "endpoint"}
    missing = sorted(needed - set(table.column_names))
    if missing:
        raise ValueError(f"rows.parquet thiếu cột {missing}; chạy lại --mode errors "
                         f"(các cột này được thêm ngày 10/11/2026)")
    pieces = []
    for name in ("smiles_a", "smiles_b", "endpoint"):
        encoded = pc.dictionary_encode(table.column(name).combine_chunks())
        pieces.append(np.asarray(encoded.indices.to_numpy(zero_copy_only=False),
                                 dtype=np.int64))
    return metric_module.query_groups(pieces)


def run_flip(args: argparse.Namespace) -> int:
    """Cùng cặp (A, B), khác Context P: nhãn có đổi dấu không, và mô hình có theo kịp không.

    Đây là phép đo cho giả thuyết anh Tiến nêu ở dòng 189 biên bản 09/11/2026.
    """
    table = read_rows(Path(args.validation))
    truth, prediction = columns(table, "delta_p", "prediction")
    counted, correct = columns(table, "sign_counted", "sign_correct")
    counted, correct = counted.astype(bool), correct.astype(bool)
    codes = pair_codes(table)
    proteins = np.asarray(pc.dictionary_encode(
        table.column("protein_hash").combine_chunks()).indices.to_numpy(
            zero_copy_only=False), dtype=np.int64)
    tolerance = float(args.tolerance)

    order, bounds = metric_module.group_bounds(codes)
    multi_rows: list[np.ndarray] = []
    flip_rows: list[np.ndarray] = []
    same_rows: list[np.ndarray] = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        rows = order[start:stop]
        if rows.size < 2 or np.unique(proteins[rows]).size < 2:
            continue
        multi_rows.append(rows)
        values = truth[rows]
        # "Đổi dấu" theo đúng ngưỡng của dự án: phải có một P cho Δp < −tol và một P cho
        # Δp > +tol. Nếu không áp ngưỡng thì nhiễu quanh 0 sẽ bị đếm là đổi dấu.
        if np.any(values > tolerance) and np.any(values < -tolerance):
            flip_rows.append(rows)
        else:
            same_rows.append(rows)

    def block(groups: list[np.ndarray], label: str) -> dict[str, Any]:
        if not groups:
            return {"nhóm": 0, "dòng": 0, "mae": None, "đúng chiều": None, "mẫu số": 0}
        rows = np.concatenate(groups)
        accuracy, denominator = sign_accuracy(counted[rows], correct[rows])
        return {"nhãn": label, "nhóm": len(groups), "dòng": int(rows.size),
                "mae": float(np.mean(np.abs(prediction[rows] - truth[rows]))),
                "đúng chiều": accuracy, "mẫu số": denominator,
                "sd nhãn trong nhóm": float(np.mean(
                    [float(np.std(truth[group])) for group in groups])),
                "sd dự đoán trong nhóm": float(np.mean(
                    [float(np.std(prediction[group])) for group in groups]))}

    report: dict[str, Any] = {
        "partition": Path(args.validation).name,
        "rows_total": table.num_rows,
        "direction_tolerance": tolerance,
        "cùng cặp nhiều P": block(multi_rows, "mọi cặp xuất hiện với >= 2 Context P"),
        "nhãn ĐỔI DẤU giữa các P": block(flip_rows, "đổi dấu"),
        "nhãn GIỮ DẤU giữa các P": block(same_rows, "giữ dấu"),
    }
    for key in ("cùng cặp nhiều P", "nhãn ĐỔI DẤU giữa các P", "nhãn GIỮ DẤU giữa các P"):
        entry = report[key]
        print(f"{key:<34} nhóm {entry['nhóm']:>7,} | dòng {entry['dòng']:>9,} | "
              f"MAE {show(entry['mae']):>8} | đúng chiều {show(entry['đúng chiều']):>8} "
              f"(mẫu số {entry['mẫu số']:,})")
    flip = report["nhãn ĐỔI DẤU giữa các P"]
    same = report["nhãn GIỮ DẤU giữa các P"]
    if flip["đúng chiều"] is not None and same["đúng chiều"] is not None:
        print()
        print(f"CHÊNH đúng chiều (đổi dấu − giữ dấu) = "
              f"{flip['đúng chiều'] - same['đúng chiều']:+.4f}")
        print("Âm nhiều nghĩa là: đúng ở các cặp mà P không đổi quy luật, sai ở các cặp mà")
        print("P đổi quy luật -> mô hình đang học theo cặp (A, B), phớt lờ Context P.")
    if flip["sd dự đoán trong nhóm"] is not None and flip["nhóm"]:
        print()
        print(f"Trong các nhóm đổi dấu: sd NHÃN giữa các P = "
              f"{show(flip['sd nhãn trong nhóm'])}, sd DỰ ĐOÁN giữa các P = "
              f"{show(flip['sd dự đoán trong nhóm'])}.")
        print("Dự đoán gần như không đổi khi chỉ đổi P là bằng chứng trực tiếp nhất cho")
        print("việc mô hình không dùng Context P.")

    if args.train:
        report["ghi nhớ từ train"] = memorization_block(table, Path(args.train), codes,
                                                        truth, prediction)
    target = Path(args.out) if args.out else Path(args.validation) / "pair_context_flip.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nđã ghi {target}")
    return 0


def memorization_block(validation: pa.Table, train_folder: Path, validation_codes: np.ndarray,
                       truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Dự đoán trên validation giống nhãn validation hơn, hay giống nhãn train cùng cặp hơn.

    Phép thử mạnh hơn `flip`: với các dòng validation có cặp (A, B, endpoint) cũng xuất
    hiện trong train (đo 10/11/2026: 69.379 dòng, 14,05 % validation), lấy trung bình nhãn
    train của chính cặp đó rồi so hai tương quan. Nếu mô hình học vẹt theo cặp thì dự đoán
    sẽ bám nhãn TRAIN, dù dòng validation thuộc một Context P khác hẳn.
    """
    train = read_rows(train_folder)
    key_columns = ("smiles_a", "smiles_b", "endpoint")
    missing = sorted(set(key_columns) - set(train.column_names))
    if missing:
        raise ValueError(f"rows.parquet của train thiếu cột {missing}; chạy lại --mode errors")
    # Khóa dạng chuỗi ghép: hai bảng khác nhau nên không dùng được mã dictionary của bảng này.
    def joined(table: pa.Table) -> list[str]:
        parts = [table.column(name).to_pylist() for name in key_columns]
        return ["\x1f".join(str(value) for value in row) for row in zip(*parts)]

    train_truth = np.asarray(train.column("delta_p").to_numpy(zero_copy_only=False))
    totals: dict[str, list[float]] = {}
    for key, value in zip(joined(train), train_truth):
        totals.setdefault(key, []).append(float(value))
    train_mean = {key: float(np.mean(values)) for key, values in totals.items()}

    validation_keys = joined(validation)
    rows = np.array([index for index, key in enumerate(validation_keys)
                     if key in train_mean], dtype=np.int64)
    if rows.size == 0:
        return {"dòng trùng cặp với train": 0,
                "ghi chú": "không có cặp nào của validation xuất hiện trong train"}
    reference = np.array([train_mean[validation_keys[index]] for index in rows])
    own = truth[rows]
    guess = prediction[rows]
    with_own = metric_module.pearson(guess, own)
    with_train = metric_module.pearson(guess, reference)
    block = {
        "dòng trùng cặp với train": int(rows.size),
        "tỉ lệ trên validation": float(rows.size) / validation.num_rows,
        "tương quan dự đoán với nhãn VALIDATION": with_own,
        "tương quan dự đoán với nhãn TRAIN cùng cặp": with_train,
        "mae với nhãn validation": float(np.mean(np.abs(guess - own))),
        "mae với nhãn train cùng cặp": float(np.mean(np.abs(guess - reference))),
        "nhãn hai bên lệch nhau (mae)": float(np.mean(np.abs(own - reference))),
    }
    print()
    print(f"Dòng validation trùng cặp (A, B, endpoint) với train: {rows.size:,} "
          f"({100.0 * rows.size / validation.num_rows:.2f} %)")
    print(f"  tương quan dự đoán ~ nhãn validation      : {show(with_own)}")
    print(f"  tương quan dự đoán ~ nhãn train cùng cặp  : {show(with_train)}")
    print(f"  MAE dự đoán vs nhãn validation            : {show(block['mae với nhãn validation'])}")
    print(f"  MAE dự đoán vs nhãn train cùng cặp        : {show(block['mae với nhãn train cùng cặp'])}")
    if with_own is not None and with_train is not None and with_train > with_own:
        print("  -> Dự đoán bám nhãn TRAIN hơn nhãn thật của dòng validation: đây là bằng")
        print("     chứng cho giả thuyết học vẹt theo cặp (A, B) ở dòng 189 biên bản.")
    return block


def context_codes(table: pa.Table) -> np.ndarray:
    """Mã nhóm theo (protein_hash, endpoint) — tức "cùng một Context P".

    Endpoint vào khóa vì Δp đo bằng IC50 và Δp đo bằng Ki không so thẳng với nhau được;
    đây cũng đúng định nghĩa query mà metrics.query_groups dùng (tham chiếu, protein,
    endpoint), chỉ bỏ phân tử tham chiếu ra để nhóm rộng hơn: cả protein một nhóm.
    """
    missing = sorted({"protein_hash", "endpoint"} - set(table.column_names))
    if missing:
        raise ValueError(f"rows.parquet thiếu cột {missing}; chạy lại --mode errors")
    pieces = []
    for name in ("protein_hash", "endpoint"):
        encoded = pc.dictionary_encode(table.column(name).combine_chunks())
        pieces.append(np.asarray(encoded.indices.to_numpy(zero_copy_only=False),
                                 dtype=np.int64))
    return metric_module.query_groups(pieces)


def variance_split(values: np.ndarray, groups: list[np.ndarray]) -> dict[str, float]:
    """Tách phương sai thành phần GIỮA các nhóm và phần TRONG nhóm (luật phương sai toàn phần).

    Trả về sd của từng phần và tỉ lệ phần giữa nhóm. Trọng số theo số dòng của nhóm, nên
    một protein 10.000 dòng nặng hơn một protein 10 dòng — đúng với cách mọi chỉ số khác
    của dự án tính trên dòng.
    """
    counts = np.array([group.size for group in groups], dtype=np.float64)
    means = np.array([float(values[group].mean()) for group in groups])
    variances = np.array([float(values[group].var()) for group in groups])
    total_rows = float(counts.sum())
    overall = float((counts * means).sum() / total_rows)
    within = float((counts * variances).sum() / total_rows)
    between = float((counts * (means - overall) ** 2).sum() / total_rows)
    whole = within + between
    return {"sd trong nhóm": float(np.sqrt(within)),
            "sd giữa các nhóm": float(np.sqrt(between)),
            "sd toàn phần": float(np.sqrt(whole)),
            "tỉ lệ phương sai nằm giữa các nhóm": (between / whole) if whole > 0 else None}


def run_context(args: argparse.Namespace) -> int:
    """Cùng Context P, KHÁC cặp (A, B): lát (b) của việc 9 họp 09/11/2026.

    `flip` hỏi: giữ nguyên cặp, đổi protein thì mô hình có theo kịp không (câu trả lời
    11/11: không). Lệnh này hỏi chiều ngược lại: trong cùng một protein, mô hình có phân
    biệt được cặp này với cặp kia không. Hai lát cộng lại trả lời được câu "mô hình đang
    biến thiên theo chiều nào".
    """
    table = read_rows(Path(args.rows))
    truth, prediction = columns(table, "delta_p", "prediction")
    truth, prediction = truth.astype(np.float64), prediction.astype(np.float64)
    codes = context_codes(table)
    pairs = pair_codes(table)
    tolerance = float(args.tolerance)
    rng = np.random.default_rng(args.seed)

    order, bounds = metric_module.group_bounds(codes)
    kept: list[np.ndarray] = []
    distinct: list[int] = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        rows = order[start:stop]
        count = int(np.unique(pairs[rows]).size)
        if count < args.min_pairs:
            continue
        kept.append(rows)
        distinct.append(count)
    if not kept:
        print(f"không có Context P nào có >= {args.min_pairs} cặp (A, B) phân biệt")
        return 1

    # Đúng chiều TRONG nhóm: mọi cặp ứng viên của cùng một Context P, đúng hàm mà chỉ số
    # pairwise_accuracy của dự án dùng. O(n^2) theo nhóm, nên nhóm lớn được lấy mẫu
    # `--max-rows-per-group` dòng (hạt giống cố định) thay vì bỏ qua.
    agree_total = 0
    counted_total = 0
    capped = 0
    spearmans: list[float] = []
    shrink: list[float] = []
    for rows in kept:
        sample = rows
        if sample.size > args.max_rows_per_group:
            sample = np.sort(rng.choice(rows, size=args.max_rows_per_group, replace=False))
            capped += 1
        agree, counted = metric_module.pairwise_direction(truth[sample],
                                                          prediction[sample], tolerance)
        agree_total += agree
        counted_total += counted
        if sample.size >= args.min_rows_for_correlation:
            value = metric_module.spearman(truth[sample], prediction[sample])
            if value is not None:
                spearmans.append(float(value))
            spread = float(truth[sample].std())
            if spread > 0:
                shrink.append(float(prediction[sample].std()) / spread)

    rows_kept = np.concatenate(kept)
    report: dict[str, Any] = {
        "nguồn": str(Path(args.rows)),
        "ngưỡng đúng chiều": tolerance,
        "số Context P giữ lại": len(kept),
        "số cặp (A, B) phân biệt mỗi Context P": {
            "trung vị": float(np.median(distinct)), "lớn nhất": int(max(distinct))},
        "dòng": int(rows_kept.size),
        "dòng trên toàn partition": int(table.num_rows),
        "nhóm bị lấy mẫu vì quá lớn": capped,
        "đúng chiều TRONG Context P": (float(agree_total) / counted_total
                                       if counted_total else None),
        "mẫu số (số cặp ứng viên)": counted_total,
        "spearman trung bình trong nhóm": (float(np.mean(spearmans)) if spearmans
                                           else None),
        "số nhóm tính được spearman": len(spearmans),
        "độ co trung bình trong nhóm": float(np.mean(shrink)) if shrink else None,
        "phương sai NHÃN": variance_split(truth, kept),
        "phương sai DỰ ĐOÁN": variance_split(prediction, kept),
    }

    print(f"Nguồn: {Path(args.rows)}")
    print(f"Context P (protein × endpoint) có >= {args.min_pairs} cặp phân biệt: "
          f"{len(kept):,} nhóm | {rows_kept.size:,} dòng "
          f"({100.0 * rows_kept.size / table.num_rows:.1f} % partition)")
    print(f"Số cặp phân biệt mỗi nhóm: trung vị {np.median(distinct):,.0f}, "
          f"lớn nhất {max(distinct):,}")
    if capped:
        print(f"  ({capped:,} nhóm quá lớn, lấy mẫu {args.max_rows_per_group:,} dòng)")
    print()
    print(f"Đúng chiều giữa HAI CẶP trong cùng một Context P : "
          f"{show(report['đúng chiều TRONG Context P'])} "
          f"(mẫu số {counted_total:,} cặp ứng viên)")
    print(f"Spearman trung bình trong một Context P          : "
          f"{show(report['spearman trung bình trong nhóm'])} "
          f"({len(spearmans):,} nhóm >= {args.min_rows_for_correlation} dòng)")
    print(f"Độ co trung bình trong một Context P            : "
          f"{show(report['độ co trung bình trong nhóm'])}  (sd dự đoán / sd nhãn)")
    print()
    print(f"{'phân tách phương sai':<26}{'nhãn':>14}{'dự đoán':>14}")
    print("-" * 54)
    for key in ("sd trong nhóm", "sd giữa các nhóm", "sd toàn phần",
                "tỉ lệ phương sai nằm giữa các nhóm"):
        print(f"{key:<26}{show(report['phương sai NHÃN'][key]):>14}"
              f"{show(report['phương sai DỰ ĐOÁN'][key]):>14}")
    print()
    print("Đọc bảng này cùng với lệnh `flip`. `flip` đã chứng minh mô hình gần như không")
    print("đổi dự đoán khi chỉ đổi Context P. Nếu ở đây 'sd trong nhóm' của dự đoán cũng")
    print("nhỏ hơn hẳn của nhãn thì mô hình co về trung bình ở CẢ HAI chiều; còn nếu")
    print("'đúng chiều trong Context P' vượt rõ 0,5 thì trong một protein mô hình vẫn xếp")
    print("hạng được các cặp — tức cái nó học là hóa học của cặp, không phải vai trò của P.")

    if args.worst:
        report["bảng ca sai"] = write_worst_rows(table, rows_kept, truth, prediction,
                                                 Path(args.rows), args)

    target = Path(args.out) if args.out else Path(args.rows) / "context_pairs.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nđã ghi {target}")
    return 0


def write_worst_rows(table: pa.Table, rows: np.ndarray, truth: np.ndarray,
                     prediction: np.ndarray, folder: Path,
                     args: argparse.Namespace) -> dict[str, Any]:
    """Ghi N dòng sai nặng nhất ra CSV để ĐỌC TRỰC TIẾP (yêu cầu của việc 9).

    "Sai nặng" ở đây là sai NGƯỢC CHIỀU và lệch nhiều: |Δp dự đoán − Δp thật| lớn trong
    số các dòng mà hai bên trái dấu. Dòng lệch nhiều nhưng cùng chiều không vào bảng này,
    vì chúng là lỗi độ lớn chứ không phải lỗi quy luật.
    """
    error = prediction[rows] - truth[rows]
    opposite = np.sign(truth[rows]) != np.sign(prediction[rows])
    candidates = rows[opposite]
    magnitude = np.abs(error)[opposite]
    if candidates.size == 0:
        return {"ghi chú": "không có dòng nào trái dấu"}
    picked = candidates[np.argsort(-magnitude)[:args.worst]]
    wanted = [name for name in ("smiles_a", "smiles_b", "endpoint", "protein_hash",
                                "protein_rows", "delta_p", "prediction", "unseen_pair")
              if name in table.column_names]
    extracted = {name: np.asarray(table.column(name).to_numpy(zero_copy_only=False))[picked]
                 for name in wanted}
    extracted["sai_số"] = prediction[picked] - truth[picked]
    target = Path(args.worst_out) if args.worst_out else folder / "worst_rows.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(extracted))
        for index in range(picked.size):
            writer.writerow([extracted[name][index] for name in extracted])
    print()
    print(f"{picked.size:,} dòng TRÁI DẤU lệch nặng nhất -> {target}")
    print("Mở file này đọc tay: biên bản 09/11 yêu cầu đọc trực tiếp các ca sai để tìm quy")
    print("luật, không chỉ nhìn số tổng hợp.")
    return {"số dòng ghi": int(picked.size), "file": str(target),
            "số dòng trái dấu trong các nhóm đã giữ": int(candidates.size),
            "tỉ lệ trái dấu": float(candidates.size) / float(rows.size)}


def run_scale(args: argparse.Namespace) -> int:
    """Đúng chiều theo QUY MÔ nhóm Context P, ngưỡng tuyệt đối của Tùng (dòng 85)."""
    table = read_rows(Path(args.rows))
    counted, correct = columns(table, "sign_counted", "sign_correct")
    counted, correct = counted.astype(bool), correct.astype(bool)
    truth, prediction = columns(table, "delta_p", "prediction")
    protein_rows = np.asarray(table.column("protein_rows").to_numpy(zero_copy_only=False))
    edges = [0, *SCALE_EDGES, np.inf]
    names = [f"< {SCALE_EDGES[0]:,}"]
    for low, high in zip(SCALE_EDGES[:-1], SCALE_EDGES[1:]):
        names.append(f"{low:,} - {high:,}")
    names.append(f"> {SCALE_EDGES[-1]:,}")
    report: list[dict[str, Any]] = []
    print(f"{'quy mô Context P':<20}{'dòng':>12}{'số protein':>12}{'MAE':>10}"
          f"{'đúng chiều':>12}{'mẫu số':>12}")
    print("-" * 78)
    for index, name in enumerate(names):
        mask = (protein_rows >= edges[index]) & (protein_rows < edges[index + 1])
        if not mask.any():
            continue
        accuracy, denominator = sign_accuracy(counted[mask], correct[mask])
        entry = {
            "quy mô": name, "dòng": int(mask.sum()),
            "số protein": int(np.unique(np.asarray(
                table.column("protein_hash").to_pylist(), dtype=object)[mask]).size),
            "mae": float(np.mean(np.abs(prediction[mask] - truth[mask]))),
            "đúng chiều": accuracy, "mẫu số": denominator,
        }
        report.append(entry)
        print(f"{name:<20}{entry['dòng']:>12,}{entry['số protein']:>12,}"
              f"{entry['mae']:>10.4f}{show(accuracy):>12}{denominator:>12,}")
    print()
    print("Số của Tùng ở dòng 85 biên bản 09/11/2026, để đối chiếu: Context P dưới 1.000")
    print("mẫu đúng chiều ~0,75; trên 50.000 mẫu ~0,89 (trên TRAIN). Trên validation mọi")
    print("Context P đều mới nên con số này không so trực tiếp với train được.")
    target = Path(args.out) if args.out else Path(args.rows) / "context_scale.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"edges": list(SCALE_EDGES), "groups": report},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nđã ghi {target}")
    return 0


def main() -> int:
    """Phân tích tham số và gọi lệnh con. Trả về mã thoát (0 = xong)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gap = subparsers.add_parser("gap", help="bảng MSE/MAE/R² train vs validation")
    gap.add_argument("--train", default="", help="thư mục errors_*_train_*")
    gap.add_argument("--validation", default="", help="thư mục errors_*_validation_*")
    gap.add_argument("--out", default="", help="file .json để ghi; mặc định chỉ in")
    gap.set_defaults(handler=run_gap)

    hist = subparsers.add_parser("hist", help="phân phối nhãn thật đè nhãn dự đoán")
    hist.add_argument("--train", default="")
    hist.add_argument("--validation", default="")
    hist.add_argument("--bins", type=int, default=120)
    hist.add_argument("--range", type=float, default=4.0,
                      help="trục Δp vẽ trong [-range, range] (mặc định 4,0)")
    hist.add_argument("--out", default="", help="file .png; mặc định distributions.png")
    hist.set_defaults(handler=run_hist)

    flip = subparsers.add_parser("flip", help="cùng cặp (A, B) khác Context P")
    flip.add_argument("--validation", required=True)
    flip.add_argument("--train", default="",
                      help="thêm phép thử ghi nhớ: dự đoán bám nhãn train hay nhãn thật")
    flip.add_argument("--tolerance", type=float, default=0.3,
                      help="ngưỡng coi Δp là khác 0; mặc định 0,3 = direction_tolerance")
    flip.add_argument("--out", default="")
    flip.set_defaults(handler=run_flip)

    context = subparsers.add_parser("context", help="cùng Context P, khác cặp (A, B)")
    context.add_argument("--rows", required=True, help="thư mục errors_*")
    context.add_argument("--min-pairs", type=int, default=2,
                         help="bỏ qua Context P có ít cặp (A, B) phân biệt hơn mức này")
    context.add_argument("--max-rows-per-group", type=int, default=500,
                         help="nhóm lớn hơn mức này được lấy mẫu; phép đếm cặp là O(n^2)")
    context.add_argument("--min-rows-for-correlation", type=int, default=5,
                         help="nhóm ít dòng hơn mức này không tính spearman/độ co")
    context.add_argument("--tolerance", type=float, default=0.3,
                         help="ngưỡng coi hai nhãn là bằng; mặc định 0,3 = direction_tolerance")
    context.add_argument("--worst", type=int, default=0,
                         help="ghi ra N dòng TRÁI DẤU lệch nặng nhất để đọc tay")
    context.add_argument("--worst-out", default="", help="file .csv cho --worst")
    context.add_argument("--seed", type=int, default=20261023,
                         help="hạt giống của phép lấy mẫu nhóm lớn")
    context.add_argument("--out", default="")
    context.set_defaults(handler=run_context)

    scale = subparsers.add_parser("scale", help="đúng chiều theo quy mô nhóm Context P")
    scale.add_argument("--rows", required=True, help="thư mục errors_*")
    scale.add_argument("--out", default="")
    scale.set_defaults(handler=run_scale)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
