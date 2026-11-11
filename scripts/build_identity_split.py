"""Dựng bảng đổi partition cho thí nghiệm đổi ngưỡng giống nhau của protein.

    python scripts/build_identity_split.py command --identity 0.3
    python scripts/build_identity_split.py build --identity 0.3 \\
            --clusters runs/dataset/mmseqs_id30/p30_cluster.tsv
    python scripts/build_identity_split.py verify --identity 0.5 \\
            --clusters runs/dataset/mmseqs_v4b/p50_cluster.tsv

Script CHỈ ĐỌC thư mục dữ liệu; mọi thứ nó ghi đều nằm trong splits/ (xem
OUTPUT_DIR). Thư mục đó ĐƯỢC git theo dõi, nên bảng sang server bằng `git pull`;
khác với scripts/build_train_subset.py ghi vào work/splits/ và phải scp.

VÌ SAO CÓ FILE NÀY. Kết quả tuần 03-08/11/2026 loại được bảy nguyên nhân của trần 0,566
và còn đúng một nguyên nhân: mô hình không tổng quát hóa sang HỌ PROTEIN chưa thấy. Phép
thử trực tiếp cho nguyên nhân đó là đổi ngưỡng giống nhau khi gom protein thành họ: ở
ngưỡng 80 % các họ vụn hơn nên protein trong validation giống protein trong train hơn, ở
ngưỡng 30 % thì ngược lại. Nếu điểm đi lên theo ngưỡng thì nguyên nhân được xác nhận.

CÁCH LÀM, và vì sao KHÔNG dựng dataset v5. Cách chia chỉ phụ thuộc protein: mỗi protein
thuộc một họ, mỗi họ thuộc một partition. Cặp phân tử, nhãn Δp, cache fingerprint và cache
embedding không đổi chút nào (hai cache đánh khóa theo phân tử và theo protein, không theo
partition). Nên thứ duy nhất cần đổi là cột `partition`, và Stage 3 chỉ đọc cột đó ở một
chỗ duy nhất (`data.read_columns`). Bảng do script này ghi ra được nạp ở đúng chỗ đó qua
`stage3.inputs.partition_override`.

PHẦN GOM CHUỖI VẪN DO MMseqs2 CHẠY NGOÀI DỰ ÁN, giống Stage 2b: xem
`src/molopt/dataset/split.py`. Lệnh cho từng ngưỡng in ra bằng lệnh con `command`; mọi
tham số khác giữ NGUYÊN như lần chia 50 % đã chốt ngày 21/10/2026
(`config.json` mục `stage2b.split.clustering`), vì nếu đổi hai thứ cùng lúc thì chênh lệch
điểm không quy được cho ngưỡng.

SỐ CẶP MỖI PROTEIN lấy lại từ cột `pairs` của `data/dataset_v4/protein_split.parquet`, tức
đúng những con số đã dùng khi chia 50 %. Số cặp không phụ thuộc ngưỡng identity, nên dùng
lại là đúng, và nhờ vậy lệnh `verify` dựng lại được bảng 50 % để đối chiếu từng dòng.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from molopt.dataset import split as split_module  # noqa: E402

CONFIG_FILE = ROOT / "config.json"
DATASET_DIR = ROOT / "data" / "dataset_v4"
SPLIT_FILE = DATASET_DIR / "protein_split.parquet"
FASTA_FILE = DATASET_DIR / "protein_split_input.fasta"
EVALUATOR_FILE = DATASET_DIR / "evaluator_training.parquet"
OUTPUT_DIR = ROOT / "splits"
OVERRIDE_SCHEMA = pa.schema([
    ("protein_hash", pa.string()),
    ("partition", pa.string()),
    ("protein_cluster_id", pa.string()),
    ("pairs", pa.int64()),
])
# Ngưỡng 0,5 là lần chia đang dùng; hai ngưỡng kia là phép thử của tuần 09-15/11/2026.
KNOWN_IDENTITIES = (0.3, 0.5, 0.8)


def sha256(path: Path) -> str:
    """Băm SHA-256 của một file, để biết bảng đổi partition dựng từ bảng nhóm nào.

    Bảng nhóm của MMseqs2 nằm trong runs/ nên không vào git, giống lần chia 50 % của
    Stage 2b; ghi băm vào báo cáo là cách truy lại.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_tag(identity: float) -> str:
    """Nhãn ngắn của một ngưỡng, ví dụ 0.3 -> 'id30'. Dùng cho tên file và tên lần chạy."""
    percent = round(identity * 100)
    if not 0 < percent < 100:
        raise ValueError(f"ngưỡng identity phải nằm trong khoảng (0, 1), nhận {identity}")
    return f"id{percent:02d}"


def load_config() -> dict[str, Any]:
    """Đọc config.json; chỉ đọc, không bao giờ ghi lại."""
    with CONFIG_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def protein_pairs_from_split() -> dict[str, int]:
    """{protein: số cặp} lấy từ bảng split đã có, tức đúng số đã dùng khi chia 50 %."""
    if not SPLIT_FILE.is_file():
        raise FileNotFoundError(f"chưa có {SPLIT_FILE}")
    table = pq.read_table(SPLIT_FILE, columns=["protein_sequence_hash", "pairs"])
    return dict(zip(table.column("protein_sequence_hash").to_pylist(),
                    table.column("pairs").to_pylist()))


def read_membership(path: Path) -> dict[str, str]:
    """{protein: tên nhóm} từ file TSV hai cột của MMseqs2 (đại diện, thành viên)."""
    if not path.is_file():
        raise FileNotFoundError(f"chưa có bảng nhóm {path}; chạy MMseqs2 trước, xem lệnh "
                                f"con 'command'")
    membership: dict[str, str] = {}
    with path.open(encoding="ascii") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path} dòng {number}: cần đúng hai cột, nhận {len(parts)}")
            representative, member = parts
            if member in membership and membership[member] != representative:
                raise ValueError(f"{path}: protein {member} nằm ở hai nhóm")
            membership[member] = representative
    if not membership:
        raise ValueError(f"{path} rỗng")
    return membership


def build_override(identity: float, clusters: Path) -> tuple[pa.Table, dict[str, Any]]:
    """Dựng bảng đổi partition cho một ngưỡng; trả về (bảng, báo cáo kiểm tra).

    Dùng lại nguyên `molopt.dataset.split` của Stage 2b (cùng cách xếp nhóm, cùng seed,
    cùng tỷ lệ mục tiêu) để chênh lệch giữa các ngưỡng chỉ đến từ bảng nhóm.
    """
    config = load_config()
    settings = config["stage2b"]["split"]
    pairs = protein_pairs_from_split()
    membership = read_membership(clusters)
    extra = sorted(set(membership) - set(pairs))
    if extra:
        raise ValueError(f"{len(extra)} protein có trong bảng nhóm nhưng không có cặp nào, "
                         f"ví dụ {extra[:3]}; bảng nhóm phải dựng từ "
                         f"{FASTA_FILE.name}")
    table = split_module.build_split_table(pairs, membership, settings["target_ratios"],
                                           settings["partitions"], settings["seed"])
    report = {
        "identity": identity,
        "clusters_file": clusters.as_posix(),
        "clusters_sha256": sha256(clusters),
        "proteins": table.num_rows,
        "clusters": len(set(table.column("protein_cluster_id").to_pylist())),
        "pairs": sum(pairs.values()),
        "seed": settings["seed"],
        "target_ratios": settings["target_ratios"],
        "clusters_single_partition":
            split_module.check_clusters_single_partition(table),
        "proteins_single_partition":
            split_module.check_proteins_single_partition(table),
        "pair_totals": split_module.check_pair_totals(table, pairs),
    }
    report["ratios"] = split_module.ratio_report(
        report["pair_totals"]["pairs_by_partition"], settings["target_ratios"])
    override = pa.table({
        "protein_hash": table.column("protein_sequence_hash"),
        "partition": table.column("partition"),
        "protein_cluster_id": table.column("protein_cluster_id"),
        "pairs": table.column("pairs"),
    }, schema=OVERRIDE_SCHEMA)
    return override, report


def compare_with_dataset(override: pa.Table) -> dict[str, Any]:
    """So bảng mới với cột partition đang có trong dataset, theo protein và theo cặp.

    Trả về số protein đổi partition và số cặp đổi partition. Đây là thông tin bắt buộc
    phải báo cùng kết quả: nếu một ngưỡng chỉ đổi vài phần trăm số cặp thì chênh lệch
    điểm không có ý nghĩa, dù nó dương hay âm.
    """
    current = pq.read_table(SPLIT_FILE, columns=["protein_sequence_hash", "partition",
                                                 "pairs"])
    before = dict(zip(current.column("protein_sequence_hash").to_pylist(),
                      current.column("partition").to_pylist()))
    counts = dict(zip(current.column("protein_sequence_hash").to_pylist(),
                      current.column("pairs").to_pylist()))
    after = dict(zip(override.column("protein_hash").to_pylist(),
                     override.column("partition").to_pylist()))
    moved = [protein for protein in after if before.get(protein) != after[protein]]
    crosstab: Counter = Counter()
    for protein, partition in after.items():
        crosstab[f"{before.get(protein, 'thiếu')} -> {partition}"] += 1
    return {
        "proteins_total": len(after),
        "proteins_moved": len(moved),
        "pairs_moved": sum(counts.get(protein, 0) for protein in moved),
        "pairs_total": sum(counts.values()),
        "identical": not moved,
        "old_to_new": dict(sorted(crosstab.items())),
    }


def wsl_path(path: Path) -> str:
    """Đường dẫn Windows -> đường dẫn WSL, ví dụ E:/a/b -> /mnt/e/a/b.

    MMseqs2 cài trong WSL Ubuntu-24.04 (xem config stage2b.split.clustering.tool_install)
    nên lệnh in ra phải dùng đường dẫn mà WSL hiểu, không phải đường dẫn Windows.
    """
    text = path.as_posix()
    if len(text) > 2 and text[1] == ":":
        return f"/mnt/{text[0].lower()}{text[2:]}"
    return text


def mmseqs_command(config: dict[str, Any], identity: float, prefix: str) -> str:
    """Lệnh MMseqs2 cho một ngưỡng: giữ mọi tham số của lần chia 50 %, chỉ đổi identity."""
    clustering = config["stage2b"]["split"]["clustering"]
    return (f"mmseqs easy-cluster {wsl_path(FASTA_FILE)} {prefix} {prefix}_tmp"
            f" --min-seq-id {identity} -c {clustering['coverage']}"
            f" --cov-mode {clustering['coverage_mode']}"
            f" --cluster-mode {clustering['cluster_mode']}"
            f" --single-step-clustering --max-seqs {clustering['max_seqs']}"
            f" --alignment-mode {clustering['alignment_mode']}"
            f" --seq-id-mode {clustering['seq_id_mode']}"
            f" -s {clustering['sensitivity']}")


def run_command(args: argparse.Namespace) -> int:
    """In lệnh MMseqs2 cần chạy trong WSL cho ngưỡng đã chọn."""
    config = load_config()
    tag = identity_tag(args.identity)
    prefix = f"runs/dataset/mmseqs_{tag}/p{round(args.identity * 100):02d}"
    print(f"# Ngưỡng {args.identity:.2f} ({tag}). Chạy trong WSL Ubuntu-24.04, MMseqs2 "
          f"{config['stage2b']['split']['clustering']['tool_version']}.")
    print(f"cd {wsl_path(ROOT)} && mkdir -p runs/dataset/mmseqs_{tag}")
    print(mmseqs_command(config, args.identity, prefix))
    print(f"# Xong thì bảng nhóm nằm ở {prefix}_cluster.tsv")
    return 0


def run_build(args: argparse.Namespace) -> int:
    """Dựng và ghi bảng đổi partition cho một ngưỡng."""
    override, report = build_override(args.identity, Path(args.clusters))
    report["compare_with_dataset"] = compare_with_dataset(override)
    tag = identity_tag(args.identity)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = Path(args.out) if args.out else OUTPUT_DIR / f"partition_override_{tag}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(override, target)
    (target.with_suffix(".json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [name for name in ("clusters_single_partition", "proteins_single_partition",
                                "pair_totals") if not report[name]["passed"]]
    print(json.dumps({"ngưỡng": args.identity, "file": str(target),
                      "số nhóm": report["clusters"], "số protein": report["proteins"],
                      "kiểm tra không đạt": failed,
                      "tỷ lệ thực tế": {name: round(value["actual"], 6)
                                        for name, value in report["ratios"].items()},
                      "protein đổi partition": report["compare_with_dataset"]
                      ["proteins_moved"],
                      "cặp đổi partition": report["compare_with_dataset"]["pairs_moved"]},
                     ensure_ascii=False, indent=2))
    return 1 if failed else 0


def run_verify(args: argparse.Namespace) -> int:
    """Dựng lại một ngưỡng rồi so với cột partition đang có trong dataset.

    Dùng với ngưỡng 0,5 và bảng nhóm gốc: phải ra GIỐNG HỆT. Nếu giống hệt thì đường mã
    mới đã được chứng minh là tương đương đường cũ, và các lần chạy đã có đóng vai mốc
    50 % — khỏi phải train lại mốc, tiết kiệm đúng một lượt train và một lượt chấm.
    """
    override, _ = build_override(args.identity, Path(args.clusters))
    comparison = compare_with_dataset(override)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if comparison["identical"]:
        print("GIỐNG HỆT: đường đổi partition tái tạo đúng cách chia đang dùng.")
        return 0
    print("KHÁC: đường đổi partition KHÔNG tái tạo được cách chia đang dùng; phải chạy "
          "thêm một mốc 50 % qua đúng đường mã này trước khi so hai ngưỡng kia.")
    return 1


def main() -> int:
    """Phân tích tham số và gọi lệnh con. Trả về mã thoát."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    printer = subparsers.add_parser("command", help="in lệnh MMseqs2 cho một ngưỡng")
    printer.add_argument("--identity", type=float, required=True,
                         help=f"ngưỡng giống nhau, ví dụ {KNOWN_IDENTITIES}")
    printer.set_defaults(handler=run_command)

    builder = subparsers.add_parser("build", help="dựng bảng đổi partition")
    builder.add_argument("--identity", type=float, required=True)
    builder.add_argument("--clusters", required=True, help="file *_cluster.tsv của MMseqs2")
    builder.add_argument("--out", default="", help="mặc định splits/...")
    builder.set_defaults(handler=run_build)

    checker = subparsers.add_parser("verify", help="so bảng dựng lại với cách chia đang dùng")
    checker.add_argument("--identity", type=float, required=True)
    checker.add_argument("--clusters", required=True)
    checker.set_defaults(handler=run_verify)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
