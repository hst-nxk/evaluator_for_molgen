"""CLI cho Stage 3 — evaluator baseline (dự đoán Δp của một cặp chỉnh sửa).

    python run_evaluator.py --mode prepare  [--skip-protein]
    python run_evaluator.py --mode baseline
    python run_evaluator.py --mode smoke   [--config frozen|finetune|no_protein]
                                           [--head direct|twin]
    python run_evaluator.py --mode train   [--config ...] [--head ...]
    python run_evaluator.py --mode eval    --checkpoint runs/evaluator/<file>.pt
"""
# ---------------------------------------------------------------------------------------
# File này chỉ là điểm vào: đọc config.json, bật logging, rồi gọi hàm trong
# src/molopt/evaluator. Không có logic xử lý dữ liệu ở đây.
#   prepare  : dựng cache fingerprint (RDKit) và cache embedding protein (ESM2).
#              --skip-protein để chỉ dựng fingerprint khi chưa cài torch/transformers.
#   baseline : chỉ số của hai baseline tầm thường (luôn đoán 0; trung bình theo endpoint)
#              làm mốc dưới. Không cần torch, không cần cache: chạy được trước prepare.
#   smoke    : train trên một mẫu nhỏ, chạy được trên CPU, chỉ xác nhận pipeline thông.
#   train    : train thật; chọn checkpoint theo MAE trên validation.
#   eval     : nạp checkpoint rồi báo cáo chỉ số theo từng lát cắt.
#
# Thứ tự chạy: baseline -> prepare -> smoke -> train -> eval.
# Partition `test` không được dùng ở bất kỳ chế độ nào (xem molopt.evaluator.data).
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

from molopt.evaluator import train as evaluator_train  # noqa: E402

CONFIG_PATH = ROOT / "config.json"
MODES = ("prepare", "baseline", "smoke", "train", "eval")
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def load_config(path: Path) -> dict[str, Any]:
    """Đọc config.json; mọi tham số của Stage 3 nằm ở mục 'stage3'."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def setup_logging(log_path: Path) -> None:
    """Ghi log ra màn hình và ra file log của lần chạy."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_path, encoding="utf-8")])


def main() -> int:
    """Nhận tham số dòng lệnh, chạy chế độ tương ứng. Trả về mã thoát (0 = xong)."""
    config = load_config(CONFIG_PATH)
    settings = config["stage3"]["train"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--config", dest="configuration", default="frozen",
                        choices=list(settings["configurations"]),
                        help="cấu hình nhánh protein (mặc định frozen: rẻ nhất)")
    parser.add_argument("--head", default=settings["head"], choices=list(settings["heads"]),
                        help="direct: dự đoán thẳng Δp; direct_no_edit: như direct nhưng "
                             "bỏ nhánh edit; twin: chấm riêng từng phân tử rồi lấy hiệu")
    parser.add_argument("--skip-protein", action="store_true",
                        help="prepare: chỉ dựng cache fingerprint, bỏ qua ESM2")
    parser.add_argument("--checkpoint", type=Path, help="eval: file .pt cần nạp")
    args = parser.parse_args()

    stamp = datetime.now().strftime(evaluator_train.TIMESTAMP_FORMAT)
    runs_dir = ROOT / config["stage3"]["paths"]["runs_dir"]
    setup_logging(runs_dir / f"{args.mode}_{stamp}.log")

    if args.mode == "prepare":
        evaluator_train.run_prepare(config, ROOT, args.skip_protein)
        return 0
    if args.mode == "baseline":
        evaluator_train.run_baselines(config, ROOT, stamp)
        return 0
    if args.mode in ("smoke", "train"):
        evaluator_train.run_train(config, ROOT, args.configuration, args.head,
                                  smoke=args.mode == "smoke", stamp=stamp)
        return 0
    if args.checkpoint is None:
        parser.error("--mode eval cần --checkpoint")
    if not args.checkpoint.is_file():
        parser.error(f"không thấy checkpoint {args.checkpoint}")
    evaluator_train.run_eval(config, ROOT, args.checkpoint, stamp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
