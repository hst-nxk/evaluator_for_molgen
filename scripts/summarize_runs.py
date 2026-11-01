"""Bảng so sánh MỌI lần train trong runs/evaluator, kể cả các lần chỉ khác nhau ở --set.

    python scripts/summarize_runs.py                 # xếp theo MAE, tốt nhất lên đầu
    python scripts/summarize_runs.py --sort spearman # xếp theo Spearman
    python scripts/summarize_runs.py --slice clean
    python scripts/summarize_runs.py --ranking       # chỉ số xếp hạng trong cùng query

Khác `summarize_evaluator.py`: file kia gom theo (cấu hình, đầu ra) rồi chỉ lấy lần MỚI
NHẤT của mỗi cặp, nên một đợt quét tham số có 15 lần `frozen direct` sẽ chỉ hiện một dòng.
File này liệt kê từng lần một và in kèm các tham số đã ghi đè, để so được với nhau.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
DEFAULT_SLICE = "all"
DIGITS = 4


def load(path: Path) -> dict[str, Any]:
    """Đọc một file JSON (UTF-8)."""
    return json.loads(path.read_text(encoding="utf-8"))


def show(value: float | None) -> str:
    """Định dạng một ô; thiếu số thì in '-'."""
    return "-" if value is None else f"{value:.{DIGITS}f}"


def label(manifest: dict[str, Any]) -> str:
    """Tên ngắn của một lần chạy: cấu hình, đầu ra, IGNITE, cộng dồn, tham số đã đổi."""
    parts = [str(manifest["configuration"]), str(manifest["head"])]
    if manifest.get("sharpness_penalty"):
        parts.append("ignite")
    # Cộng dồn gradient đọc từ KHÓA RIÊNG của manifest, không đọc từ `overrides`: các
    # lần finetune chạy trước 01/11/2026 ghi sai khối overrides (trùng tên biến trong
    # train.py), nên dựa vào overrides là mất dấu đúng tham số cần so.
    steps = int(manifest.get("accumulate_steps") or 1)
    if steps > 1:
        parts.append(f"accum={steps}")
    overrides = manifest.get("overrides") or {}
    # embedding_dim và cache_dir chỉ đi kèm việc đổi model ESM2, không phải tham số
    # riêng; in tên model là đủ để phân biệt lần chạy.
    skip = {"gpu_memory_limit_mib", "eval_every_steps", "accumulate_steps",
            "protein_encoder.embedding_dim", "paths.cache_dir"}
    for key, value in overrides.items():
        if key == "protein_encoder.model":
            parts.append(str(value).split("/")[-1])
        elif key not in skip:
            parts.append(f"{key.split('.')[-1]}={value}")
    return " ".join(parts)


def best_block(manifest: dict[str, Any], slice_name: str) -> dict[str, Any] | None:
    """Khối chỉ số của MỐC TỐT NHẤT (không phải mốc cuối)."""
    if slice_name != DEFAULT_SLICE:
        return (manifest.get("validation_metrics") or {}).get(slice_name)
    # final_metrics_epoch là khóa của các lần chạy TRƯỚC khi có phép đo giữa epoch; hồi
    # đó history xếp theo epoch nên dùng thẳng làm chỉ số được.
    index = manifest.get("final_metrics_index", manifest.get("final_metrics_epoch"))
    history = manifest.get("history") or []
    if index is None or index < 0 or index >= len(history):
        return None
    return history[index].get("validation")


def cadence(manifest: dict[str, Any]) -> str:
    """Cứ bao nhiêu bước thì đo validation một lần; 0 = chỉ đo cuối mỗi epoch.

    Phải hiện thành cột riêng vì đây là một CONFOUND: đo dày hơn thì cơ hội bắt được
    checkpoint tốt cũng nhiều hơn, nên hai lần chạy khác nhịp đo thì chỉ số không so
    trực tiếp với nhau được.
    """
    every = manifest.get("eval_every_steps")
    if every is None:
        return "-"
    return "epoch" if int(every) <= 0 else str(int(every))


def ranking_cells(block: dict[str, Any] | None) -> list[str]:
    """Các ô xếp hạng của một khối lát cắt: đúng chiều, top-1, top-3, 3 lớp.

    Đúng chiều và top-k tính TRONG CÙNG MỘT QUERY (phân tử tham chiếu, protein, endpoint),
    xem docs/STAGE3_EVALUATOR.md mục 4.2–4.3. Ba lớp kém / ngang / tốt chia ở ±0,3 log.
    """
    ranking = (block or {}).get("ranking") or {}
    classes = (block or {}).get("change_classes") or {}
    return [show(ranking.get("pairwise_accuracy_macro")), show(ranking.get("top_1")),
            show(ranking.get("top_3")), show(classes.get("accuracy")),
            show(classes.get("macro_f1"))]


def where(manifest: dict[str, Any]) -> str:
    """Mốc nào cho kết quả tốt nhất, dạng 'epoch 0 bước 4000'."""
    epoch, step = manifest.get("best_epoch"), manifest.get("best_step")
    if epoch is None:
        return "-"
    return f"e{epoch}" if step is None else f"e{epoch}/b{step}"


def main() -> int:
    """Đọc mọi manifest train, in bảng đã sắp xếp. Trả về mã thoát 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", default=DEFAULT_SLICE)
    parser.add_argument("--sort", default="mae", choices=("mae", "spearman", "rmse"))
    parser.add_argument("--ranking", action="store_true",
                        help="in chỉ số xếp hạng (đúng chiều, top-k, 3 lớp) thay cho RMSE/R2")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    runs_dir = ROOT / config["stage3"]["paths"]["runs_dir"]
    if not runs_dir.is_dir():
        print(f"không thấy {runs_dir}", file=sys.stderr)
        return 1

    rows: list[tuple[float, list[str]]] = []
    baseline_path = max(runs_dir.glob("baseline_*.json"), default=None,
                        key=lambda path: path.name)
    if baseline_path is not None:
        for name, scores in load(baseline_path)["validation_metrics"].items():
            block = scores.get(args.slice) or {}
            regression = block.get("regression") or {}
            # Baseline phải xếp CHUNG với các mô hình, không dồn xuống cuối: cả bảng này
            # tồn tại để trả lời "có vượt được phép đoán trung bình không".
            value = regression.get(args.sort)
            order = float("inf") if value is None else (-value if args.sort == "spearman"
                                                        else value)
            middle = (ranking_cells(block) if args.ranking
                      else [show(regression.get("rmse")), show(regression.get("r2"))])
            rows.append((order, [f"baseline {name}", show(regression.get("mae")), *middle,
                                 show(regression.get("spearman")), "-", "-", "-"]))

    for path in sorted(runs_dir.glob("train_*.json")):
        manifest = load(path)
        block = best_block(manifest, args.slice)
        regression = (block or {}).get("regression") or {}
        key = {"mae": "mae", "rmse": "rmse", "spearman": "spearman"}[args.sort]
        value = regression.get(key)
        # Spearman càng lớn càng tốt, MAE/RMSE càng nhỏ càng tốt.
        order = float("inf") if value is None else (-value if args.sort == "spearman"
                                                    else value)
        # Chỉ số xếp hạng chỉ tính ở bước cuối, sau khi nạp lại checkpoint tốt nhất, nên
        # lấy từ validation_metrics chứ không từ history (history chỉ có regression).
        final = (manifest.get("validation_metrics") or {}).get(args.slice)
        middle = (ranking_cells(final) if args.ranking
                  else [show(regression.get("rmse")), show(regression.get("r2"))])
        rows.append((order, [
            label(manifest), show(regression.get("mae")), *middle,
            show(regression.get("spearman")), where(manifest), cadence(manifest),
            f"{manifest.get('seconds', 0) / 60:.0f}p"]))

    middle = (["đúng chiều", "top-1", "top-3", "đúng 3 lớp", "F1 3 lớp"] if args.ranking
              else ["RMSE", "R2"])
    header = ["lần chạy", "MAE", *middle, "Spearman", "mốc", "đo", "giờ"]
    rows.sort(key=lambda row: row[0])
    table = [header] + [row[1] for row in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    print(f"lát cắt: {args.slice} | xếp theo: {args.sort} | {runs_dir}")
    for index, row in enumerate(table):
        print("  ".join(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
                        for i, cell in enumerate(row)))
        if index == 0:
            print("  ".join("-" * width for width in widths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
