"""CLI cho Stage 3 — evaluator baseline (dự đoán Δp của một cặp chỉnh sửa).

    python run_evaluator.py --mode prepare  [--skip-protein] [--molecule-embeddings]
                                           [--set khóa=giá_trị ...]
    python run_evaluator.py --mode baseline
    python run_evaluator.py --mode smoke   [--config frozen|finetune|no_protein]
                                           [--head direct|twin] [--sharpness on|off]
    python run_evaluator.py --mode train   [--config ...] [--head ...]
                                           [--sharpness on|off] [--set khóa=giá_trị ...]
    python run_evaluator.py --mode eval    --checkpoint runs/evaluator/<file>.pt
                                           [--partition validation|train]
    python run_evaluator.py --mode errors  --checkpoint runs/evaluator/<file>.pt
                                           [--partition validation|train]
    python run_evaluator.py --mode features --checkpoint runs/evaluator/<file>.pt
    python run_evaluator.py --mode diagnose --checkpoint runs/evaluator/<file>.pt
                                           [--partition validation|train]
"""
# ---------------------------------------------------------------------------------------
# File này chỉ là điểm vào: đọc config.json, bật logging, rồi gọi hàm trong
# src/molopt/evaluator. Không có logic xử lý dữ liệu ở đây.
#   prepare  : dựng cache fingerprint (RDKit) và cache embedding protein (ESM2).
#              --skip-protein để chỉ dựng fingerprint khi chưa cài torch/transformers.
#              --molecule-embeddings để dựng thêm cache embedding phân tử (MoLFormer),
#              chỉ cần khi chạy --set model.ligand_features=embedding.
#   baseline : chỉ số của hai baseline tầm thường (luôn đoán 0; trung bình theo endpoint)
#              làm mốc dưới. Không cần torch, không cần cache: chạy được trước prepare.
#   smoke    : train trên một mẫu nhỏ, chạy được trên CPU, chỉ xác nhận pipeline thông.
#   train    : train thật; chọn checkpoint theo MAE trên validation.
#   eval     : nạp checkpoint rồi báo cáo chỉ số theo từng lát cắt.
#   errors   : nạp checkpoint rồi chia lỗi theo nhóm (việc 6, họp 02/11/2026).
#   features : nạp checkpoint rồi xuất vector sau encoder của một mẫu train và
#              validation, để vẽ t-SNE (việc 4).
#   diagnose : nạp checkpoint rồi đo nơ-ron ReLU chết và độ nhạy với từng đầu vào, khi
#              một lần train cho dự đoán gần như hằng số.
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

MODEL_PREFIX = "model"
CONFIG_PATH = ROOT / "config.json"
MODES = ("prepare", "baseline", "smoke", "train", "eval", "errors", "features", "diagnose")
# Partition mà --mode eval được chấm. Không có test: test chỉ chạy một lần ở cuối.
EVAL_PARTITIONS = ("validation", "train")
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


# Từ được nhận cho khóa true/false của --set. Không dùng bool(raw): bool("false") là
# True, nên viết sai sẽ BẬT cờ mà người dùng vừa định tắt.
BOOLEAN_WORDS = {"true": True, "false": False, "1": True, "0": False,
                 "on": True, "off": False}


def apply_overrides(settings: dict, pairs: list[str],
                    model_settings: dict | None = None,
                    blocks: dict[str, dict] | None = None) -> dict:
    """Ghi đè các tham số train từ dòng lệnh, dạng ``khóa=giá trị``.

    Chỉ cho phép ghi đè khóa ĐÃ CÓ trong ``stage3.train`` và có giá trị là số, chuỗi
    hay true/false: gõ sai tên khóa sẽ báo lỗi ngay thay vì âm thầm bỏ qua, nếu không cả
    một đợt quét tham số có thể chạy xong mới phát hiện là vô nghĩa. Kiểu được suy từ giá
    trị đang có trong config, nên không cần khai báo kiểu ở dòng lệnh.

    Khóa có dấu chấm thì đi vào khối con, ví dụ ``finetune_overrides.batch_size=32``.
    Khóa bắt đầu bằng ``model.`` thì đi sang ``stage3.model`` thay vì ``stage3.train``,
    ví dụ ``model.dropout=0.3`` — dropout là nút chống khớp quá dữ liệu nhưng lại nằm
    trong khối mô tả kiến trúc, không nằm trong khối tham số train.

    ``blocks`` thêm các khối khác theo cùng cách, tên khối -> dict của khối. CLI truyền
    ``protein_encoder`` và ``paths`` để thử một model ESM2 khác mà không sửa config.json:
    ``protein_encoder.model=... protein_encoder.embedding_dim=... paths.cache_dir=...``.
    Đổi cả ``paths.cache_dir`` là bắt buộc trên thực tế: cache embedding chỉ có một file,
    và ``train.Caches`` từ chối cache dựng bằng model khác config, nên dùng chung thư mục
    thì lần dựng mới đè mất cache của model đang dùng.
    Trả về: dict khóa -> giá trị mới, để ghi vào manifest của lần chạy.
    """
    routes: dict[str, dict | None] = {MODEL_PREFIX: model_settings, **(blocks or {})}
    applied: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set cần dạng khóa=giá trị, nhận {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        target = settings
        path = key.split(".")
        if path[0] in routes:
            if routes[path[0]] is None:
                raise ValueError(f"--set: không nhận {key!r} ở chế độ này")
            target = routes[path[0]]
            prefix = path[0]
            path = path[1:]
            if not path or not all(path):
                raise ValueError(f"--set: thiếu tên tham số sau '{prefix}.'")
        for step in path[:-1]:
            if not isinstance(target, dict) or step not in target:
                raise ValueError(f"--set: không có khối {step!r} trong {key!r}")
            target = target[step]
        leaf = path[-1]
        if not isinstance(target, dict) or leaf not in target:
            raise ValueError(f"--set: không có tham số {key!r} trong stage3.train")
        current = target[leaf]
        if not isinstance(current, (bool, int, float, str)):
            raise ValueError(f"--set: không ghi đè được {key!r} "
                             f"(kiểu {type(current).__name__})")
        value: object = raw
        # bool phải xét TRƯỚC int (bool là lớp con của int) và phải phân tích tường minh:
        # bool("false") là True, nên để Python tự chuyển sẽ bật cờ mà người dùng vừa tắt.
        if isinstance(current, bool):
            if raw.strip().lower() not in BOOLEAN_WORDS:
                raise ValueError(f"--set: {key!r} là true/false, nhận {raw!r}; "
                                 f"viết một trong {sorted(BOOLEAN_WORDS)}")
            value = BOOLEAN_WORDS[raw.strip().lower()]
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        target[leaf] = value
        applied[key] = value
    return applied


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
    parser.add_argument("--molecule-embeddings", action="store_true",
                        help="prepare: dựng thêm cache embedding phân tử (MoLFormer), "
                             "cần cho --set model.ligand_features=embedding")
    parser.add_argument("--sharpness", choices=("on", "off"), default=None,
                        help="smoke/train: bật hoặc tắt số hạng phạt chuẩn gradient đầu ra "
                             "(IGNITE, arXiv:2503.04242); không truyền thì theo config "
                             "stage3.train.sharpness.enabled")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KHÓA=GIÁ_TRỊ",
                        help="ghi đè một tham số mà không sửa config.json, ví dụ "
                             "--set weight_decay=0.1 (stage3.train), --set model.dropout=0.6 "
                             "(stage3.model), --set protein_encoder.model=... và "
                             "--set paths.cache_dir=... (dùng cả cho prepare); dùng được "
                             "nhiều lần")
    parser.add_argument("--checkpoint", type=Path,
                        help="eval, errors, features, diagnose: file .pt cần nạp")
    parser.add_argument("--partition", default=EVAL_PARTITIONS[0], choices=EVAL_PARTITIONS,
                        help="eval, errors, diagnose: partition được chấm; train để so với "
                             "validation xem mô hình overfit hay chưa học được (test luôn "
                             "bị chặn)")
    args = parser.parse_args()

    try:
        overrides = apply_overrides(settings, args.overrides, config["stage3"]["model"],
                                    {"protein_encoder": config["stage3"]["protein_encoder"],
                                     "paths": config["stage3"]["paths"]})
    except ValueError as error:
        parser.error(str(error))

    stamp = datetime.now().strftime(evaluator_train.TIMESTAMP_FORMAT)
    runs_dir = ROOT / config["stage3"]["paths"]["runs_dir"]
    setup_logging(runs_dir / f"{args.mode}_{stamp}.log")
    if overrides:
        logging.info("ghi đè tham số: %s", overrides)

    if args.mode == "prepare":
        evaluator_train.run_prepare(config, ROOT, args.skip_protein,
                                    args.molecule_embeddings)
        return 0
    if args.mode == "baseline":
        evaluator_train.run_baselines(config, ROOT, stamp)
        return 0
    if args.mode in ("smoke", "train"):
        evaluator_train.run_train(config, ROOT, args.configuration, args.head,
                                  smoke=args.mode == "smoke", stamp=stamp,
                                  sharpness_mode=args.sharpness, overrides=overrides)
        return 0
    if args.checkpoint is None:
        parser.error(f"--mode {args.mode} cần --checkpoint")
    if not args.checkpoint.is_file():
        parser.error(f"không thấy checkpoint {args.checkpoint}")
    if args.mode == "errors":
        evaluator_train.run_errors(config, ROOT, args.checkpoint, stamp, args.partition)
    elif args.mode == "features":
        evaluator_train.run_features(config, ROOT, args.checkpoint, stamp)
    elif args.mode == "diagnose":
        evaluator_train.run_diagnose(config, ROOT, args.checkpoint, stamp, args.partition)
    else:
        evaluator_train.run_eval(config, ROOT, args.checkpoint, stamp, args.partition)
    return 0


if __name__ == "__main__":
    sys.exit(main())
