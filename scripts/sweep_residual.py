"""Sinh lệnh quét siêu tham số cho head residual, lấy mẫu theo Bảng 14 của bài gốc.

    python scripts/sweep_residual.py --trials 20 --seed 20261107
    python scripts/sweep_residual.py --trials 20 --gpu 6 --shards 3 > sweep.sh

Script này KHÔNG chạy train: nó chỉ in ra các lệnh `run_evaluator.py --mode train` để
người dùng tự chạy trên server.

NGUỒN khoảng lấy mẫu — Gorishniy, Rubachev, Khrulkov, Babenko (2021), "Revisiting Deep
Learning Models for Tabular Data", NeurIPS, arXiv:2106.11959, Bảng 14 (mục F.1, không
gian quét của ResNet), đọc trực tiếp từ bản PDF trang 19 ngày 07/11/2026:
    # Layers        UniformInt[1, 8]
    Layer size      UniformInt[64, 512]
    Hidden factor   Uniform[1, 4]
    Hidden dropout  Uniform[0, 0.5]
    Residual dropout {0, Uniform[0, 0.5]}
    Learning rate   LogUniform[1e-5, 1e-2]
    Weight decay    {0, LogUniform[1e-6, 1e-3]}
    # Iterations    100 (Optuna, Akiba và cs. 2019)
Mục F.1 của bài còn ghi: ở cấu hình tốt nhất, dropout TRONG khối thường cao hơn dropout
nhánh dư, và dropout nhánh dư nhiều khi bằng 0.

HAI CHỖ LỆCH BÀI BÁO, có chủ ý và phải nhớ khi viết paper:
 1. Bài dùng Optuna/TPE 100 vòng. Ở đây mỗi lần train mất 15-25 phút nên 100 vòng là
    25-40 giờ một GPU. Script này dùng LẤY MẪU NGẪU NHIÊN (Bergstra và Bengio 2012,
    "Random Search for Hyper-Parameter Optimization", JMLR 13:281-305), tức đúng cách
    Optuna chạy 10 vòng đầu. Random search KHÔNG mạnh bằng TPE khi số vòng lớn; chọn
    nó vì ngân sách nhỏ và vì nó song song hoá được.
 2. Bài đo trên dữ liệu bảng, không phải cặp phân tử. Khoảng quét là điểm xuất phát có
    nguồn, KHÔNG bảo đảm hợp với bài toán này.

`model.dropout` không nằm trong Bảng 14 (ở đây nó là dropout của các encoder nhánh, bài
gốc không có nhánh nào) nên giữ nguyên giá trị đang dùng, mặc định 0.5; đổi bằng
--encoder-dropout. Checkpoint và dừng sớm chọn theo đúng chiều macro trên validation,
không bao giờ dùng test.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from typing import Any

# Bảng 14 chỉ quét trên validation; test không xuất hiện ở bất kỳ đâu trong quy trình.
SELECTION_METRIC = "pairwise_accuracy_macro"
SOURCE = "Gorishniy et al. 2021, arXiv:2106.11959, Table 14 (section F.1)"


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    """Lấy mẫu đều trên thang log trong [low, high], như LogUniform của Bảng 14."""
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_trial(rng: random.Random) -> dict[str, Any]:
    """Một điểm trong không gian Bảng 14; khóa là tên tham số của run_evaluator.py."""
    # {0, Uniform[0, 0.5]} trong bài là tập hai thành phần: hoặc đúng 0, hoặc lấy mẫu
    # đều. Bài không nói xác suất của nhánh 0 nên ở đây để 50/50 và ghi rõ là lựa chọn
    # của dự án, CHƯA CÓ NGUỒN.
    residual_dropout = 0.0 if rng.random() < 0.5 else rng.uniform(0.0, 0.5)
    weight_decay = 0.0 if rng.random() < 0.5 else log_uniform(rng, 1e-6, 1e-3)
    return {
        "model.residual_blocks": rng.randint(1, 8),
        "model.residual_width": rng.randint(64, 512),
        "model.residual_hidden_factor": round(rng.uniform(1.0, 4.0), 3),
        "model.residual_hidden_dropout": round(rng.uniform(0.0, 0.5), 4),
        "model.residual_dropout": round(residual_dropout, 4),
        "learning_rate": float(f"{log_uniform(rng, 1e-5, 1e-2):.3g}"),
        "weight_decay": float(f"{weight_decay:.3g}"),
    }


def command(trial: dict[str, Any], encoder_dropout: float, seed: int) -> str:
    """Một lệnh train đầy đủ cho một điểm quét."""
    parts = ["python run_evaluator.py --mode train --config frozen --head direct",
             "--set model.head_block=residual",
             f"--set model.dropout={encoder_dropout}",
             f"--set selection_metric={SELECTION_METRIC}",
             f"--set seed={seed}"]
    parts.extend(f"--set {key}={value}" for key, value in trial.items())
    return " ".join(parts)


def main() -> int:
    """In các lệnh quét; trả về mã thoát (0 = xong)."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=20,
                        help="số điểm quét (mặc định 20; bài gốc dùng 100 vòng Optuna)")
    parser.add_argument("--seed", type=int, default=20261107,
                        help="seed của BỘ LẤY MẪU, để quét lặp lại được")
    parser.add_argument("--train-seed", type=int, default=20261023,
                        help="seed của lần train (giữ cố định để so các điểm quét)")
    parser.add_argument("--encoder-dropout", type=float, default=0.5,
                        help="model.dropout, không thuộc Bảng 14 (mặc định 0.5)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="chèn CUDA_VISIBLE_DEVICES=<gpu> vào đầu mỗi chuỗi lệnh")
    parser.add_argument("--shards", type=int, default=1,
                        help="chia các điểm quét thành mấy chuỗi chạy song song")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="in bảng điểm quét dạng JSON thay cho lệnh shell")
    args = parser.parse_args()
    # Console Windows mặc định là cp1252, không in được tiếng Việt khi hứng qua pipe.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.trials < 1:
        parser.error("--trials phải >= 1")
    if args.shards < 1:
        parser.error("--shards phải >= 1")
    rng = random.Random(args.seed)
    trials = [sample_trial(rng) for _ in range(args.trials)]

    if args.as_json:
        print(json.dumps({"source": SOURCE, "sampler": "random search, Bergstra & Bengio "
                                                       "2012 (JMLR 13:281-305)",
                          "sampler_seed": args.seed, "train_seed": args.train_seed,
                          "selection_metric": SELECTION_METRIC, "trials": trials},
                         indent=2, ensure_ascii=False))
        return 0

    print(f"# {args.trials} điểm quét | nguồn khoảng: {SOURCE}")
    print(f"# lấy mẫu ngẫu nhiên, seed bộ lấy mẫu {args.seed}, seed train "
          f"{args.train_seed}")
    print("# chọn checkpoint và dừng sớm theo đúng chiều macro trên validation")
    for shard in range(args.shards):
        chosen = trials[shard::args.shards]
        if not chosen:
            continue
        lines = [command(trial, args.encoder_dropout, args.train_seed)
                 for trial in chosen]
        body = "; ".join(lines) + f"; echo XONG_SWEEP_{shard}"
        prefix = "" if args.gpu is None else f"CUDA_VISIBLE_DEVICES={args.gpu} "
        print(f"\n# chuỗi {shard}: {len(chosen)} lần train")
        print(f"{prefix}setsid nohup bash -c \"{body}\" "
              f"> runs/sweep_{shard}.log 2>&1 &")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
