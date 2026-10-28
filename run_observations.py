"""CLI for the MOLOPT Raw -> Observations stage.

    python run_observations.py --mode inspect|smoke|full|verify
"""
# ---------------------------------------------------------------------------------------
# File này là điểm vào (entry point) của bước Raw -> Observations.
# Nó không chứa logic xử lý dữ liệu: chỉ đọc config.json, bật logging, rồi gọi đúng
# hàm trong gói src/molopt/observations theo chế độ người dùng chọn:
#   inspect : in danh sách file raw, schema, vị trí các cột cần dùng (không đọc dòng dữ liệu)
#   smoke   : chạy thử một mẫu nhỏ của cả hai nguồn vào runs/observations/smoke_<thời điểm>/
#   full    : chạy toàn bộ, ghi Parquet vào paths.observations_dir (hiện là data/observations_v2/)
#   verify  : đối chiếu output của full ngược về raw, ghi báo cáo JSON
# (Ghi chú tiếng Việt đặt bằng comment thay vì trong docstring ở trên, vì docstring của
#  module được dùng làm nội dung `--help` qua argparse(description=__doc__).)
# ---------------------------------------------------------------------------------------

# Cho phép viết kiểu `X | None` trên Python 3.10; không ảnh hưởng logic.
from __future__ import annotations

# Thư viện chuẩn: tham số dòng lệnh, JSON, log, thời gian, đường dẫn.
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Thư mục gốc của dự án (E:\eval_training). Mọi đường dẫn trong config.json đều tính
# tương đối từ đây, nên toàn bộ input/output nằm trong thư mục dự án.
ROOT = Path(__file__).resolve().parent
# Thêm src/ vào sys.path để import được gói molopt mà không cần cài đặt (pip install).
sys.path.insert(0, str(ROOT / "src"))

# Import module của dự án; phải đứng sau sys.path.insert ở trên.
from molopt.observations import bindingdb, chembl, pipeline  # noqa: E402

# Đường dẫn config, bốn chế độ hợp lệ, định dạng một dòng log.
CONFIG_PATH = ROOT / "config.json"
MODES = ("inspect", "smoke", "full", "verify")
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def load_config(path: Path) -> dict[str, Any]:
    """Read config.json.

    Nhận: đường dẫn tới config.json.
    Trả về: dict cấu hình (đường dẫn, phiên bản, endpoint, bảng đơn vị, tên cột...).
    Vì sao cần: mọi tham số của pipeline nằm ở một chỗ, code không viết cứng tên file
    hay tên cột.
    """
    # Đọc config.json (UTF-8) thành dict.
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def setup_logging(log_path: Path | None) -> None:
    """Log to the console, and to a run log file except in read-only inspect mode.

    Nhận: đường dẫn file log, hoặc None (chế độ inspect).
    Trả về: không có; cấu hình logging toàn cục.
    Vì sao cần: các lần chạy smoke/full/verify cần để lại log trong runs/observations/
    để xem lại tiến độ và lỗi; inspect là chế độ chỉ đọc nên không ghi file nào.
    """
    # Luôn in ra màn hình; chỉ thêm file log khi có đường dẫn.
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        # Có đường dẫn thì ghi log thêm ra file (tạo thư mục nếu chưa có).
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    # Áp dụng cấu hình log cho toàn chương trình, từ mức INFO trở lên.
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=handlers)


def run_inspect(config: dict[str, Any]) -> None:
    """Print raw files, schemas and the columns that feed each required field.

    Nhận: config.
    Trả về: không có; in báo cáo ra màn hình.
    Vì sao cần: trước khi chạy thật, cần xác nhận file raw có mặt, schema đúng như code
    giả định và mọi cột cần dùng đều tìm thấy. Chế độ này không đọc dòng dữ liệu nào.
    """
    # In mô tả của ChEMBL rồi BindingDB.
    for line in chembl.describe(config, ROOT) + bindingdb.describe(config, ROOT):
        print(line)


def main() -> int:
    """Nhận tham số dòng lệnh, chạy chế độ tương ứng.

    Trả về: mã thoát của tiến trình (0 = thành công; 1 = verify phát hiện kiểm tra
    không đạt).
    """
    # --- Đọc tham số dòng lệnh: --mode bắt buộc, --force chỉ có tác dụng với full.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--force", action="store_true",
                        help="full: rebuild sources whose output file already exists")
    # Vd. --mode full --force -> args.mode = 'full', args.force = True.
    args = parser.parse_args()
    mode = args.mode
    # Đọc config một lần, rồi truyền xuống mọi hàm.
    config = load_config(CONFIG_PATH)
    # --- inspect: chỉ đọc, không ghi log ra file, không tạo output.
    if mode == "inspect":
        setup_logging(None)
        run_inspect(config)
        return 0

    # --- Các chế độ còn lại ghi log vào runs/observations/<mode>_<thời điểm>.log.
    # Mỗi lần chạy một file log mới, vd. full_20261018_205349.log.
    stamp = datetime.now().strftime(pipeline.TIMESTAMP_FORMAT)
    runs_dir = ROOT / config["paths"]["runs_dir"]
    setup_logging(runs_dir / f"{mode}_{stamp}.log")
    # --- smoke: mẫu nhỏ, luôn ghi vào thư mục mới nên không đụng output của full.
    if mode == "smoke":
        logging.info("smoke rows written: %s", pipeline.run_smoke(config, ROOT))
        return 0
    # --- full: toàn bộ dữ liệu; nguồn nào đã có file hoàn chỉnh thì bỏ qua trừ khi --force.
    if mode == "full":
        # --force chỉ có tác dụng với full.
        logging.info("full rows written: %s", pipeline.run_full(config, ROOT, args.force))
        return 0

    # --- verify: chạy các phép kiểm tra, ghi báo cáo JSON, in PASS/FAIL từng phép.
    report = pipeline.run_verify(config, ROOT)
    # Ghi báo cáo verify ra file JSON.
    report_path = runs_dir / f"verify_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # In PASS/FAIL cho từng phép kiểm tra.
    for check in ("conservation", "sample_reconciliation", "regression_accession_construct",
                  "regression_enhanced_stereo", "regression_pmid_integer_form"):
        logging.info("verify %s: %s", check, "PASS" if report[check]["passed"] else "FAIL")
    logging.info("verify report: %s", report_path)
    # Mã thoát khác 0 khi có phép kiểm tra không đạt, để script/CI bên ngoài nhận biết được.
    return 0 if report["passed"] else 1


# Chỉ chạy main() khi gọi trực tiếp file này, không chạy khi bị import.
if __name__ == "__main__":
    sys.exit(main())
