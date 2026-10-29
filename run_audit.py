"""CLI audit dữ liệu theo yêu cầu buổi họp 26/10/2026 (chỉ đọc data/, chỉ ghi runs/audit).

    python run_audit.py --task duplicates|thresholds|molecules|stats|followup|all
    python run_audit.py --task followup --parts peptide relative   # chỉ chạy vài phần
"""
# ---------------------------------------------------------------------------------------
# Mỗi task trả lời một yêu cầu:
#   duplicates : (1) bản BindingDB bị loại có đúng là trùng ChEMBL không; còn bản trùng
#                nào chưa bị bắt (so độ phân biệt record bằng numpy, ba mức khóa)
#   stats      : (2) thống kê số lượng, độ đo, phân bố 4 nhãn, split; đếm điểm bất thường
#   thresholds : (3) độ phân tán Δp giữa các assay, số bản ghi Δp y hệt, tác động ngưỡng
#   molecules  : (4) kích thước phân tử (nguyên tử nặng, tổng nguyên tử có H, MW), peptide,
#                kích thước nhóm thế, số cặp mất theo từng ngưỡng ứng viên
#   followup   : việc tồn 27/10 trước dataset_v3: dòng trùng do -REL, peptide còn lại
#                sau ngưỡng MW, assay bị nhập hai lần trong cùng một nguồn; và (29/10)
#                số phân tử, số cặp nằm trong quy tắc 5 của Lipinski
# Kết quả: runs/audit/<task>_<thời điểm>.json, .txt, .log và thư mục _figures.
# molecules dùng ProcessPoolExecutor (spawn trên Windows) nên main() phải nằm sau
# `if __name__ == "__main__"`.
# ---------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from molopt.audit import (common, duplicates, followup, molecules, stats,  # noqa: E402
                          thresholds)

CONFIG_PATH = ROOT / "config.json"
TASKS = ("duplicates", "stats", "thresholds", "molecules", "followup")
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def load_config(path: Path) -> dict[str, Any]:
    """Đọc config.json; tham số audit nằm ở mục 'audit'."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def setup_logging(log_path: Path) -> None:
    """Ghi log ra màn hình và ra file log của lần chạy."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_path, encoding="utf-8")])


def run_task(task: str, config: dict[str, Any], stamp: str,
             parts: list[str] | None = None) -> None:
    """Chạy một task và ghi báo cáo <task>_<stamp>.json/.txt.

    `parts` chỉ dùng cho followup (None = mọi phần).
    """
    output_dir = common.runs_dir(config, ROOT)
    name = f"{task}_{stamp}"
    figures = output_dir / f"{name}_figures"
    logging.info("=== bắt đầu %s", task)
    if task == "duplicates":
        payload, lines = duplicates.run(config, ROOT)
    elif task == "stats":
        payload, lines = stats.run(config, ROOT, figures)
    elif task == "thresholds":
        payload, lines = thresholds.run(config, ROOT, figures)
    elif task == "followup":
        payload, lines = followup.run(config, ROOT, parts or tuple(followup.PARTS))
    else:
        payload, lines = molecules.run(config, ROOT, figures, name)
    common.write_report(output_dir, name, payload, lines)
    logging.info("=== xong %s", task)


def main() -> int:
    """Nhận tham số dòng lệnh, chạy task tương ứng. Trả về mã thoát 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "all"), required=True)
    parser.add_argument("--parts", nargs="+", choices=tuple(followup.PARTS),
                        help="chỉ với --task followup: các phần cần chạy (mặc định mọi phần)")
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    setup_logging(common.runs_dir(config, ROOT) / f"{args.task}_{stamp}.log")
    for task in TASKS if args.task == "all" else (args.task,):
        run_task(task, config, stamp, args.parts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
