"""Tóm tắt kết quả Stage 3 trên validation thành một bảng (chỉ đọc file JSON, không train).

    python scripts/summarize_evaluator.py              # lát cắt 'all'
    python scripts/summarize_evaluator.py --slice clean

Nguồn số, theo thứ tự ưu tiên cho mỗi cặp (cấu hình, đầu ra):
    1. eval_<cấu hình>_<đầu ra>_*.json mới nhất: số của checkpoint TỐT NHẤT, đủ mọi lát cắt;
    2. nếu chưa có: train_<cấu hình>_<đầu ra>_*.json mới nhất, lấy history[best_epoch]
       (chỉ có lát 'all').
Baseline lấy từ baseline_*.json mới nhất. Không đọc gì của partition test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
DEFAULT_SLICE = "all"
COLUMNS = ["MAE", "R2", "Spearman", "đúng chiều", "Spearman/q", "top-1", "top-3", "top-10"]
NAME_WIDTH = 26
CELL_WIDTH = 11
DIGITS = 4


def load(path: Path) -> dict[str, Any]:
    """Đọc một file JSON (UTF-8)."""
    return json.loads(path.read_text(encoding="utf-8"))


def latest(paths: list[Path]) -> Path | None:
    """File mới nhất theo tên (tên chứa nhãn thời gian YYYYmmdd_HHMMSS nên sắp được)."""
    return sorted(paths)[-1] if paths else None


def row_values(block: dict[str, Any] | None) -> list[float | None]:
    """Các ô của một dòng bảng từ khối {regression, ranking} của một lát cắt."""
    if not block:
        return [None] * len(COLUMNS)
    regression = block.get("regression") or {}
    ranking = block.get("ranking") or {}
    return [regression.get("mae"), regression.get("r2"), regression.get("spearman"),
            ranking.get("pairwise_accuracy_macro"), ranking.get("spearman_macro"),
            ranking.get("top_1"), ranking.get("top_3"), ranking.get("top_10")]


def show(value: float | None) -> str:
    """Định dạng một ô; thiếu số thì in '-'."""
    return "-" if value is None else f"{value:.{DIGITS}f}"


def print_row(name: str, values: list[float | None], note: str = "") -> None:
    """In một dòng bảng."""
    cells = "".join(show(value).rjust(CELL_WIDTH) for value in values)
    print(f"{name:<{NAME_WIDTH}}{cells}  {note}")


def main() -> int:
    """Đọc manifest trong runs_dir của Stage 3 và in bảng so sánh."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slice", default=DEFAULT_SLICE, help="lát cắt cần in")
    args = parser.parse_args()
    config = load(CONFIG_PATH)
    runs_dir = ROOT / config["stage3"]["paths"]["runs_dir"]
    if not runs_dir.is_dir():
        print(f"không thấy {runs_dir}", file=sys.stderr)
        return 1

    print(f"lát cắt: {args.slice} | thư mục: {runs_dir}")
    print(f"{'':<{NAME_WIDTH}}" + "".join(name.rjust(CELL_WIDTH) for name in COLUMNS))

    baseline_path = latest(list(runs_dir.glob("baseline_*.json")))
    if baseline_path is not None:
        for name, scores in load(baseline_path)["validation_metrics"].items():
            print_row(f"baseline {name}", row_values(scores.get(args.slice)))

    pairs = sorted({tuple(load(path)[key] for key in ("configuration", "head"))
                    for path in runs_dir.glob("train_*.json")})
    for configuration, head in pairs:
        name = f"{configuration} {head}"
        evaluated = latest(list(runs_dir.glob(f"eval_{configuration}_{head}_*.json")))
        if evaluated is not None:
            block = load(evaluated)["validation_metrics"].get(args.slice)
            print_row(name, row_values(block), "checkpoint tốt nhất (eval)")
            continue
        trained = latest(list(runs_dir.glob(f"train_{configuration}_{head}_*.json")))
        manifest = load(trained)
        if args.slice != DEFAULT_SLICE or manifest["best_epoch"] < 0:
            print_row(name, row_values(None), "chưa có eval cho lát này")
            continue
        block = manifest["history"][manifest["best_epoch"]]["validation"]
        print_row(name, row_values(block), f"epoch tốt nhất {manifest['best_epoch']} (train)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
