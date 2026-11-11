"""Dựng bảng subset train: N nhóm Context P nhiều dữ liệu nhất, có trần số dòng mỗi nhóm.

    python scripts/build_train_subset.py plan
    python scripts/build_train_subset.py build
    python scripts/build_train_subset.py build --clusters 200 --rows-per-cluster 500
    python scripts/build_train_subset.py verify --out work/splits/train_subset_c200.parquet

Script CHỈ ĐỌC thư mục dữ liệu; mọi thứ nó ghi đều nằm trong work/splits. File kết quả
nhỏ (~2,6 MB) và work/ bị .gitignore bỏ qua, nên nó sang server bằng scp, giống cách
mọi file dữ liệu khác sang server.

VÌ SAO CÓ FILE NÀY. Việc 2 họp 09/11/2026. Anh Tiến: "Tạm thời em nên trích xuất ra một
tập dữ liệu con nhỏ để train và debug trước, đừng train trên toàn bộ tập dữ liệu lớn như
thế này nữa" (dòng 75), rồi chốt tiêu chí ở dòng 99: "Em cứ lấy ra khoảng 200 nhóm Context
P có nhiều dữ liệu nhất để làm tập Train thôi, còn tập Validation hiện tại thì cứ giữ
nguyên để đánh giá." Lý do ở dòng 99: lấy cả hàng nghìn nhóm mà mỗi nhóm một ít thì dữ
liệu quá phân tán.

MỘT MÂU THUẪN TRONG BIÊN BẢN, phải biết khi đọc kết quả: dòng 75 nói "100.000 mẫu để
train và 100.000 mẫu để validate", còn dòng 99 nói giữ nguyên tập validation hiện tại
(493.852 dòng). Script này theo DÒNG 99 vì nó muộn hơn và cụ thể hơn, và vì cắt validation
sẽ làm mọi con số cũ không so được nữa. Muốn theo dòng 75 thì phải cắt validation ở chỗ
khác, không phải ở script này.

"NHÓM CONTEXT P" LÀ CỤM PROTEIN, không phải một protein. Khải nói ở dòng 93: ở ngưỡng gom
cụm 50 % có 2.415 nhóm. Số protein phân biệt là 6.313 (đo 24/10/2026). Hai con số đó không
mâu thuẫn — Tùng nói "6.000 Context P" ở dòng 207 là đang đếm protein. Script này gom theo
`protein_cluster_id` của data/dataset_v4/protein_split.parquet, tức theo đúng đơn vị mà
cách chia tập đang dùng, nên "nhóm chưa từng thấy" ở validation vẫn đúng nghĩa.

TRẦN SỐ DÒNG MỖI CỤM là bắt buộc, không phải tuỳ chọn: Tùng đo được có cụm chiếm hơn
50.000 dòng train (dòng 85), nên lấy 200 cụm lớn nhất mà không có trần thì vài cụm đầu
bảng sẽ chiếm gần hết subset và subset lại mất cân bằng đúng như tập gốc — tức không trả
lời được câu hỏi của chính thí nghiệm.

CÁCH LẤY DÒNG TRONG MỘT CỤM: lấy mẫu ngẫu nhiên không hoàn lại với seed cố định
(stage3.train.seed), KHÔNG lấy các dòng đầu tiên. Dataset được ghi theo thứ tự (endpoint,
bucket protein) nên lấy phần đầu sẽ lệch endpoint một cách có hệ thống.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config.json"
DATASET_DIR = ROOT / "data" / "dataset_v4"
SPLIT_FILE = DATASET_DIR / "protein_split.parquet"
OUTPUT_DIR = ROOT / "work" / "splits"
TRAIN = "train"
# Cột ghi ra. pair_id là khóa mà data.read_columns lọc theo; ba cột kia chỉ để truy lại
# và kiểm tra, Stage 3 không đọc chúng.
SUBSET_SCHEMA = pa.schema([
    ("pair_id", pa.string()),
    ("protein_hash", pa.string()),
    ("protein_cluster_id", pa.string()),
    ("endpoint", pa.string()),
])
READ_COLUMNS = ["pair_id", "protein_hash", "endpoint", "partition"]
BATCH_ROWS = 200_000


def load_config() -> dict[str, Any]:
    """Đọc config.json; chỉ đọc, không bao giờ ghi lại."""
    with io.open(CONFIG_FILE, encoding="utf-8") as handle:
        return json.load(handle)


def protein_clusters() -> tuple[dict[str, str], dict[str, str]]:
    """({protein: cụm}, {protein: partition}) từ bảng chia tập đang dùng.

    Đọc từ data/dataset_v4/protein_split.parquet, tức đúng bảng đã dùng khi chia 50 %
    ngày 21/10/2026. Không gom cụm lại ở đây: gom cụm do MMseqs2 chạy ngoài dự án.
    """
    if not SPLIT_FILE.is_file():
        raise FileNotFoundError(f"chưa có {SPLIT_FILE}")
    table = pq.read_table(SPLIT_FILE, columns=["protein_sequence_hash",
                                               "protein_cluster_id", "partition"])
    proteins = table.column("protein_sequence_hash").to_pylist()
    clusters = dict(zip(proteins, table.column("protein_cluster_id").to_pylist()))
    partitions = dict(zip(proteins, table.column("partition").to_pylist()))
    return clusters, partitions


def cluster_lookup(clusters: dict[str, str]) -> tuple[pa.Array, pa.Array]:
    """(mảng protein, mảng cụm) để tra bằng pyarrow thay vì bằng vòng lặp Python.

    Tra kiểu vector là bắt buộc ở đây, không phải tối ưu sớm: dataset có 7,9 triệu dòng,
    và một vòng `for` trên `to_pylist()` mất vài phút cho mỗi lượt đọc, mà script này đọc
    hai lượt.
    """
    proteins = sorted(clusters)
    return (pa.array(proteins, pa.string()),
            pa.array([clusters[protein] for protein in proteins], pa.string()))


def cluster_column(table: pa.Table, lookup: tuple[pa.Array, pa.Array]) -> pa.Array:
    """Cột cụm của từng dòng, tra từ protein_hash; protein lạ là LỖI, không bỏ qua."""
    keys, values = lookup
    position = pc.index_in(table.column("protein_hash"), value_set=keys)
    missing = pc.sum(pc.is_null(position)).as_py() or 0
    if missing:
        unknown = pc.filter(table.column("protein_hash"), pc.is_null(position))
        raise ValueError(f"{missing} dòng có protein không nằm trong {SPLIT_FILE}, ví dụ "
                         f"{unknown.to_pylist()[:3]}; bảng split dựng từ lần export khác")
    return pc.take(values, position)


def count_train_rows(evaluator: Path, lookup: tuple[pa.Array, pa.Array]) -> Counter:
    """{cụm: số dòng train} đếm trên chính dataset, không lấy từ cột `pairs` của bảng split.

    Vì sao đếm lại chứ không dùng cột `pairs`: cột đó đếm số cặp của protein ở Stage 2b,
    TRƯỚC bước làm sạch của v4. Số dòng train thực tế của dataset v4 nhỏ hơn, và subset
    phải xếp hạng cụm theo số dòng mà mô hình thật sự nhìn thấy.
    """
    handle = pq.ParquetFile(evaluator)
    counts: Counter = Counter()
    for batch in handle.iter_batches(batch_size=BATCH_ROWS, columns=READ_COLUMNS):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.equal(table.column("partition"), TRAIN))
        if not table.num_rows:
            continue
        for entry in pc.value_counts(cluster_column(table, lookup)):
            counts[entry["values"].as_py()] += entry["counts"].as_py()
    if not counts:
        raise ValueError(f"không có dòng train nào trong {evaluator}")
    return counts


def chosen_clusters(counts: Counter, wanted: int) -> list[tuple[str, int]]:
    """`wanted` cụm nhiều dòng train nhất, giảm dần; hòa thì xếp theo mã cụm.

    Xếp theo mã cụm khi hòa để hai lần chạy cho đúng cùng một danh sách: mã cụm là sha256
    của danh sách thành viên đã sắp xếp (xem molopt.dataset.split), nên nó cố định.
    """
    if wanted < 1:
        raise ValueError(f"--clusters phải >= 1, nhận {wanted}")
    order = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return order[:wanted]


def pick_rows(evaluator: Path, lookup: tuple[pa.Array, pa.Array], keep: set[str], cap: int,
              seed: int) -> tuple[pa.Table, dict[str, int]]:
    """Lấy tối đa `cap` dòng train ngẫu nhiên cho mỗi cụm trong `keep`.

    Hai lượt đọc: lượt này gom TOÀN BỘ dòng ứng viên của các cụm được chọn (chỉ bốn cột
    nhỏ nên vừa RAM), rồi mới lấy mẫu trong từng cụm. Lấy mẫu theo kiểu "giữ cap dòng đầu
    gặp được" sẽ lệch theo thứ tự ghi của dataset, nên không dùng.
    Trả về: (bảng subset đã sắp xếp theo pair_id, {cụm: số dòng lấy được}).
    """
    handle = pq.ParquetFile(evaluator)
    wanted = pa.array(sorted(keep), pa.string())
    pieces: list[pa.Table] = []
    for batch in handle.iter_batches(batch_size=BATCH_ROWS, columns=READ_COLUMNS):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.equal(table.column("partition"), TRAIN))
        if not table.num_rows:
            continue
        table = table.drop_columns(["partition"]).append_column(
            "protein_cluster_id", cluster_column(table, lookup))
        table = table.filter(pc.is_in(table.column("protein_cluster_id"),
                                      value_set=wanted))
        if table.num_rows:
            pieces.append(table)
    if not pieces:
        raise ValueError("không có dòng train nào thuộc các cụm đã chọn")
    pool = pa.concat_tables(pieces)
    encoded = pc.dictionary_encode(pool.column("protein_cluster_id").combine_chunks())
    codes = np.asarray(encoded.indices.to_numpy(zero_copy_only=False), dtype=np.int64)
    names = encoded.dictionary.to_pylist()
    rng = np.random.default_rng(seed)
    taken: list[np.ndarray] = []
    sizes: dict[str, int] = {}
    for code, name in enumerate(names):
        rows = np.flatnonzero(codes == code)
        if rows.size > cap:
            rows = np.sort(rng.choice(rows, size=cap, replace=False))
        taken.append(rows)
        sizes[str(name)] = int(rows.size)
    picked = np.sort(np.concatenate(taken))
    subset = pool.take(pa.array(picked))
    # Sắp xếp theo pair_id để file dựng lại hai lần cho đúng cùng một thứ tự dòng.
    subset = subset.take(pc.sort_indices(subset.column("pair_id")))
    return pa.table({name: subset.column(name) for name in SUBSET_SCHEMA.names},
                    schema=SUBSET_SCHEMA), sizes


def summarize(subset: pa.Table, sizes: dict[str, int], counts: Counter,
              picked: list[tuple[str, int]], cap: int, seed: int) -> dict[str, Any]:
    """Báo cáo kiểm tra: phủ bao nhiêu cụm, bao nhiêu dòng, mất cân bằng còn lại bao nhiêu."""
    taken = np.array(sorted(sizes.values()), dtype=np.int64)
    total_train = int(sum(counts.values()))
    endpoint_counts = Counter(subset.column("endpoint").to_pylist())
    capped = [name for name, size in sizes.items() if size >= cap]
    return {
        "rows": subset.num_rows,
        "clusters_requested": len(picked),
        "clusters_with_rows": len(sizes),
        "rows_per_cluster_cap": cap,
        "seed": seed,
        "clusters_at_cap": len(capped),
        "rows_min_per_cluster": int(taken.min()) if taken.size else 0,
        "rows_median_per_cluster": float(np.median(taken)) if taken.size else 0.0,
        "rows_max_per_cluster": int(taken.max()) if taken.size else 0,
        "share_of_full_train": subset.num_rows / total_train if total_train else None,
        "full_train_rows": total_train,
        "full_train_clusters": len(counts),
        # Mất cân bằng còn lại: cụm lớn nhất chiếm bao nhiêu phần subset. Ở tập gốc Tùng đo
        # 1 % cụm đầu bảng chiếm 33 % số dòng (dòng 207 biên bản); con số này cho biết trần
        # đã kéo nó xuống bao nhiêu.
        "largest_cluster_share": (int(taken.max()) / subset.num_rows
                                  if subset.num_rows and taken.size else None),
        "endpoints": dict(sorted(endpoint_counts.items())),
        "largest_clusters": [{"protein_cluster_id": name, "rows_in_full_train": rows,
                              "rows_in_subset": sizes.get(name, 0)}
                             for name, rows in picked[:10]],
    }


def run_plan(args: argparse.Namespace) -> int:
    """Chỉ ĐẾM và in: bao nhiêu cụm, mỗi cụm bao nhiêu dòng, chưa ghi file nào.

    Chạy lệnh này trước `build` để chọn `--clusters` và `--rows-per-cluster` cho ra đúng
    cỡ subset mong muốn, mà không phải ghi rồi xóa file.
    """
    config = load_config()
    evaluator = ROOT / config["stage3"]["inputs"]["evaluator"]
    clusters, _ = protein_clusters()
    counts = count_train_rows(evaluator, cluster_lookup(clusters))
    picked = chosen_clusters(counts, args.clusters)
    estimate = sum(min(rows, args.rows_per_cluster) for _, rows in picked)
    total = sum(counts.values())
    print(f"dataset           : {evaluator}")
    print(f"tổng dòng train   : {total:,}")
    print(f"tổng số cụm       : {len(counts):,}")
    print(f"cụm được chọn     : {len(picked):,} (nhiều dòng nhất)")
    print(f"trần dòng mỗi cụm : {args.rows_per_cluster:,}")
    print(f"=> subset ước tính: {estimate:,} dòng "
          f"({100.0 * estimate / total:.2f} % tập train)")
    print()
    print(f"{'hạng':>5}  {'dòng train':>12}  {'lấy vào subset':>15}  cụm")
    for rank, (name, rows) in enumerate(picked[:15], start=1):
        print(f"{rank:>5}  {rows:>12,}  {min(rows, args.rows_per_cluster):>15,}  {name}")
    if len(picked) > 15:
        smallest = picked[-1]
        print(f"{'...':>5}")
        print(f"{len(picked):>5}  {smallest[1]:>12,}  "
              f"{min(smallest[1], args.rows_per_cluster):>15,}  {smallest[0]}")
    return 0


def run_build(args: argparse.Namespace) -> int:
    """Dựng và ghi bảng subset train, kèm file .json báo cáo."""
    config = load_config()
    evaluator = ROOT / config["stage3"]["inputs"]["evaluator"]
    seed = int(config["stage3"]["train"]["seed"])
    clusters, partitions = protein_clusters()
    lookup = cluster_lookup(clusters)
    counts = count_train_rows(evaluator, lookup)
    picked = chosen_clusters(counts, args.clusters)
    keep = {name for name, _ in picked}
    # Kiểm tra bắt buộc: không một cụm nào được chọn mà lại nằm ở validation hay test.
    # Nếu có thì bảng split và cột partition của dataset đã lệch nhau, và subset sẽ kéo
    # dòng validation vào train — đúng kiểu rò rỉ mà cả dự án đang tránh.
    stray = sorted({clusters[protein] for protein, part in partitions.items()
                    if part != TRAIN and clusters[protein] in keep})
    if stray:
        raise ValueError(f"{len(stray)} cụm được chọn có protein KHÔNG thuộc train, ví dụ "
                         f"{stray[:3]}; bảng split và dataset không khớp")
    subset, sizes = pick_rows(evaluator, lookup, keep, args.rows_per_cluster, seed)
    report = summarize(subset, sizes, counts, picked, args.rows_per_cluster, seed)
    report["dataset"] = evaluator.as_posix()
    report["split_file"] = SPLIT_FILE.as_posix()
    report["source"] = ("họp 09/11/2026 dòng 99: 200 nhóm Context P nhiều dữ liệu nhất, "
                        "validation giữ nguyên")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = (Path(args.out) if args.out else
              OUTPUT_DIR / f"train_subset_c{args.clusters}_r{args.rows_per_cluster}.parquet")
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, target)
    report["file"] = target.as_posix()
    target.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("file", "rows", "clusters_requested", "clusters_with_rows",
                       "rows_per_cluster_cap", "clusters_at_cap", "rows_min_per_cluster",
                       "rows_median_per_cluster", "rows_max_per_cluster",
                       "largest_cluster_share", "share_of_full_train", "endpoints")},
                     ensure_ascii=False, indent=2))
    print()
    print("Bật subset bằng cách đặt trong config.json:")
    print(f'  "train_subset": "{target.relative_to(ROOT).as_posix()}"')
    print("hoặc không sửa config: --set inputs.train_subset="
          f"{target.relative_to(ROOT).as_posix()}")
    return 0


def run_verify(args: argparse.Namespace) -> int:
    """Đọc lại một file subset đã ghi và kiểm ba điều kiện bắt buộc.

    1. pair_id không trùng nhau.
    2. Mọi pair_id đều thuộc partition train của dataset (không kéo validation/test vào).
    3. Số dòng mỗi cụm không vượt trần ghi trong file .json kèm theo.
    """
    config = load_config()
    evaluator = ROOT / config["stage3"]["inputs"]["evaluator"]
    path = Path(args.out)
    if not path.is_file():
        raise FileNotFoundError(f"chưa có {path}")
    subset = pq.read_table(path)
    ids = subset.column("pair_id").to_pylist()
    problems: list[str] = []
    if len(set(ids)) != len(ids):
        problems.append(f"pair_id trùng: {len(ids) - len(set(ids))} dòng")
    wanted = set(ids)
    handle = pq.ParquetFile(evaluator)
    found: dict[str, str] = {}
    for batch in handle.iter_batches(batch_size=BATCH_ROWS,
                                     columns=["pair_id", "partition"]):
        table = pa.Table.from_batches([batch])
        for pair_id, partition in zip(table.column("pair_id").to_pylist(),
                                      table.column("partition").to_pylist()):
            if pair_id in wanted:
                found[pair_id] = partition
    missing = sorted(wanted - set(found))
    if missing:
        problems.append(f"{len(missing)} pair_id không có trong dataset, ví dụ {missing[:3]}")
    not_train = sorted(pair_id for pair_id, part in found.items() if part != TRAIN)
    if not_train:
        problems.append(f"{len(not_train)} pair_id KHÔNG thuộc train, ví dụ {not_train[:3]}")
    sizes = Counter(subset.column("protein_cluster_id").to_pylist())
    meta = path.with_suffix(".json")
    cap = None
    if meta.is_file():
        cap = int(json.loads(meta.read_text(encoding="utf-8"))["rows_per_cluster_cap"])
        over = [name for name, size in sizes.items() if size > cap]
        if over:
            problems.append(f"{len(over)} cụm vượt trần {cap} dòng, ví dụ {over[:3]}")
    print(json.dumps({"file": path.as_posix(), "rows": subset.num_rows,
                      "clusters": len(sizes), "rows_per_cluster_cap": cap,
                      "vấn đề": problems or "không có"}, ensure_ascii=False, indent=2))
    return 1 if problems else 0


def main() -> int:
    """Phân tích tham số và gọi lệnh con. Trả về mã thoát (0 = xong)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler, helptext in (
            ("plan", run_plan, "chỉ đếm và in, không ghi file"),
            ("build", run_build, "dựng và ghi bảng subset"),
            ("verify", run_verify, "kiểm lại một file subset đã ghi")):
        sub = subparsers.add_parser(name, help=helptext)
        sub.add_argument("--clusters", type=int, default=200,
                         help="số nhóm Context P nhiều dữ liệu nhất (mặc định 200, theo "
                              "dòng 99 biên bản 09/11/2026)")
        sub.add_argument("--rows-per-cluster", type=int, default=500,
                         help="trần số dòng mỗi nhóm (mặc định 500: 200 x 500 = 100.000 "
                              "dòng, đúng cỡ anh Tiến yêu cầu)")
        sub.add_argument("--out", default="", help="mặc định work/splits/train_subset_*.parquet")
        sub.set_defaults(handler=handler)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
