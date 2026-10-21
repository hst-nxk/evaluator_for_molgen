"""CLI for the MOLOPT Observations -> MMP dataset stage (Stage 2b).

    python run_dataset.py --mode smoke|full|split|export|verify
"""
# ---------------------------------------------------------------------------------------
# File này là điểm vào của Stage 2b. Nó không chứa logic xử lý dữ liệu: chỉ đọc
# config.json, bật logging, rồi gọi đúng hàm trong gói src/molopt/dataset theo chế độ:
#   smoke  : chạy thử trên ~1/200 số assay, ghi vào runs/dataset/smoke_<thời điểm>/
#   full   : chạy toàn bộ bước 1 -> 4, ghi vào data/dataset/full/
#   split  : dựng split protein SAU khi đã có bản ghi bước 4 (xem --clusters)
#   export : bước 5, xuất hai file training và một file evidence vào data/dataset/
#   verify : chạy các phép kiểm tra trên output của full, ghi báo cáo JSON
#
# Thứ tự chạy: smoke -> full -> split -> export -> verify.
# Mọi tiến trình con (tìm cặp MMP) được tạo bằng multiprocessing kiểu spawn trên Windows,
# nên phần gọi main() phải nằm trong `if __name__ == "__main__"` ở cuối file.
# ---------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Thư mục gốc của dự án (E:\eval_training); mọi đường dẫn trong config tính từ đây.
ROOT = Path(__file__).resolve().parent
# Thêm src/ vào sys.path để import được gói molopt mà không cần cài đặt.
sys.path.insert(0, str(ROOT / "src"))

from molopt.dataset import pipeline  # noqa: E402

CONFIG_PATH = ROOT / "config.json"
MODES = ("smoke", "full", "split", "export", "verify")
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def load_config(path: Path) -> dict[str, Any]:
    """Đọc config.json.

    Nhận: đường dẫn config.
    Trả về: dict cấu hình. Mọi tham số của Stage 2b nằm ở mục 'stage2b'.
    """
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def setup_logging(log_path: Path) -> None:
    """Ghi log ra màn hình và ra file log của lần chạy."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_path, encoding="utf-8")])


def main() -> int:
    """Nhận tham số dòng lệnh, chạy chế độ tương ứng.

    Trả về: mã thoát (0 = xong; 1 = verify có phép kiểm tra không đạt; 2 = split chưa có
    kết quả gom chuỗi nên chỉ ghi được FASTA đầu vào).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--force", action="store_true",
                        help="full: chạy lại dù data/dataset/full/ đã có")
    parser.add_argument("--clusters", type=Path,
                        help="split: file TSV hai cột '<đại diện>\\t<thành viên>' do công "
                             "cụ gom chuỗi xuất ra")
    parser.add_argument("--cluster-tool",
                        help="split: tên công cụ gom chuỗi, ghi vào manifest")
    parser.add_argument("--cluster-tool-version",
                        help="split: phiên bản công cụ gom chuỗi, ghi vào manifest")
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    stamp = datetime.now().strftime(pipeline.TIMESTAMP_FORMAT)
    runs_dir = ROOT / config["stage2b"]["paths"]["runs_dir"]
    setup_logging(runs_dir / f"{args.mode}_{stamp}.log")

    if args.mode == "smoke":
        pipeline.run_smoke(config, ROOT, stamp)
        return 0
    if args.mode == "full":
        pipeline.run_full(config, ROOT, args.force)
        return 0
    if args.mode == "split":
        if args.clusters is not None and not (args.cluster_tool and args.cluster_tool_version):
            parser.error("--clusters cần cả --cluster-tool và --cluster-tool-version "
                         "(manifest phải ghi công cụ và phiên bản đã dùng)")
        result = pipeline.run_split(config, ROOT, args.clusters, args.cluster_tool,
                                    args.cluster_tool_version)
        return 0 if result["clustered"] else 2
    if args.mode == "export":
        pipeline.run_export(config, ROOT)
        return 0

    report = pipeline.run_verify(config, ROOT)
    report_path = runs_dir / f"verify_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("báo cáo verify: %s", report_path)
    return 0 if report["passed"] else 1


# Chỉ chạy main() khi gọi trực tiếp; tiến trình con kiểu spawn import lại file này nên
# phần thân phải nằm sau điều kiện này.
if __name__ == "__main__":
    sys.exit(main())
