"""Việc 16 họp 09/11/2026: gói bàn giao cho nhóm baseline agent, và chốt định nghĩa query.

    python scripts/export_baseline_handoff.py --out docs/BASELINE_HANDOFF.md

Biên bản đặt đây là ĐIỀU KIỆN CỨNG: baseline agent sẽ được so với mô hình của paper, nên
hai bên phải chạy trên dữ liệu TRÙNG TỪNG DÒNG. Hai đường làm được việc đó:

  (a) cấp quyền push repo cho nhóm kia — việc không rút lại được, người quyết là anh Khải;
  (b) gửi họ file parquet cố định kèm mã băm — đường này, script này lo.

Script CHỈ ĐỌC. Nó không chạm vào data/, không ghi gì ngoài file --out.

NÓ TRẢ LỜI BA CÂU, mỗi câu bằng số đo chứ không bằng lời:

1. "Trùng từng dòng" nghĩa là gì, kiểm thế nào? -> sha256 của đúng hai file đầu vào, số
   dòng tổng và theo từng partition. Bên kia chạy `sha256sum` trên máy họ; khớp là xong,
   không khớp thì dừng chứ đừng chạy.

2. Stage 3 có lọc bớt dòng nào nữa không? -> KHÔNG. Chỗ lọc duy nhất của toàn Stage 3 là
   `read_columns` trong src/molopt/evaluator/data.py, và nó chỉ lọc theo cột `partition`
   (cộng thêm bảng subset/override nếu config bật, mặc định đều tắt). Nên file parquet một
   mình đã định nghĩa đủ tập dữ liệu; không có bộ lọc ngầm nào bên ngoài nó.

3. Một "query" là gì — 669.000 hay 636.000? -> biên bản ghi "đã đo là khác cách gom nhóm,
   chứ không phải khác dữ liệu", nhưng chưa ghi cách gom nào ra số nào. Script đếm TẤT CẢ
   các cách gom có thể hiểu được, trên đúng dữ liệu thật, rồi in cạnh nhau. Con số nào
   khớp 669.000 và 636.000 thì tự nó chỉ ra hai bên đang nói cách gom nào.

   Định nghĩa của dự án (src/molopt/evaluator/train.py, metrics.query_groups):
   một query = (phân tử tham chiếu, protein, endpoint). Hai chỗ làm số đổi:
     * expand_both_directions: mỗi dòng (A, B) được tính hai lần, A làm tham chiếu rồi B
       làm tham chiếu. Bật thì số query gần như gấp đôi.
     * min_candidates: query có ít hơn ngần ấy ứng viên bị BỎ khỏi mẫu số khi tính
       pairwise_accuracy_macro (queries_ranked), nhưng vẫn tồn tại (queries_total).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from molopt.evaluator import metrics as metric_module  # noqa: E402

KEY_COLUMNS = ["pair_id", "endpoint", "protein_hash", "partition", "smiles_a", "smiles_b"]
CHUNK = 64 * 1024 * 1024


def file_digest(path: Path) -> tuple[str, int]:
    """(sha256, số byte) của một file, đọc theo khối để không nạp cả file vào RAM."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def flat(column: pa.ChunkedArray | pa.Array) -> pa.Array:
    """Một cột parquet thành MỘT mảng liền, để nối hai cột lại rồi mã hóa chung."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if isinstance(column, pa.ChunkedArray):
        return (column.chunk(0) if column.num_chunks == 1
                else pa.concat_arrays(column.chunks))
    return column


def codes(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """Một cột chuỗi thành mảng mã số nguyên; cùng giá trị thì cùng mã."""
    encoded = pc.dictionary_encode(flat(column))
    return np.asarray(encoded.indices.to_numpy(zero_copy_only=False), dtype=np.int64)


def composite_groups(reference: np.ndarray, protein: np.ndarray,
                     endpoint: np.ndarray) -> np.ndarray:
    """Mã query từ ba cột, bằng MỘT khóa int64 gộp thay vì np.unique theo hàng.

    Cùng định nghĩa với metrics.query_groups (và `verify_against_project` kiểm lại điều
    đó trên mẫu con), nhưng rẻ hơn nhiều: query_groups gọi np.unique(axis=0), trên train
    6,6 triệu dòng nhân đôi hai chiều là ~13 triệu hàng và vài GB RAM.

    An toàn tràn số: khóa gộp = (ref * n_protein + protein) * n_endpoint + endpoint, chỉ
    dùng khi tích ba lực lượng còn nằm trong int64; không thì lùi về hàm của dự án.
    """
    sizes = [int(column.max()) + 1 if column.size else 1
             for column in (reference, protein, endpoint)]
    if sizes[0] * sizes[1] * sizes[2] >= 2 ** 62:
        return metric_module.query_groups([reference, protein, endpoint])
    key = (reference.astype(np.int64) * sizes[1] + protein) * sizes[2] + endpoint
    return np.unique(key, return_inverse=True)[1].astype(np.int64)


def verify_against_project(reference: np.ndarray, protein: np.ndarray,
                           endpoint: np.ndarray, sample: int = 200_000) -> None:
    """Khóa gộp phải gom nhóm GIỐNG HỆT metrics.query_groups, kiểm trên một mẫu con.

    Kiểm bằng số NHÓM và bằng quan hệ "cùng nhóm", chứ không so mã số: hai cách đánh số
    nhóm có thể khác nhau mà vẫn là cùng một cách gom.
    """
    size = min(sample, reference.size)
    if size == 0:
        return
    picked = np.random.default_rng(20261115).choice(reference.size, size=size,
                                                    replace=False)
    ours = composite_groups(reference[picked], protein[picked], endpoint[picked])
    theirs = metric_module.query_groups([reference[picked], protein[picked],
                                         endpoint[picked]])
    if np.unique(ours).size != np.unique(theirs).size:
        raise SystemExit("khóa gộp gom nhóm khác metrics.query_groups — dừng, không ghi gì")
    pairs = np.unique(np.stack([ours, theirs], axis=1), axis=0)
    if pairs.shape[0] != np.unique(ours).size:
        raise SystemExit("khóa gộp không ánh xạ 1-1 với metrics.query_groups — dừng")


def count_queries(reference: np.ndarray, protein: np.ndarray, endpoint: np.ndarray,
                  min_candidates: int) -> tuple[int, int]:
    """(số query, số query có >= min_candidates ứng viên) theo đúng định nghĩa của dự án."""
    groups = composite_groups(reference, protein, endpoint)
    sizes = np.bincount(groups)
    return int(sizes.size), int(np.count_nonzero(sizes >= min_candidates))


def partition_block(table: pa.Table, name: str, min_candidates: int) -> dict[str, object]:
    """Bốn cách đếm query cho MỘT partition, in cạnh nhau để so với 669.000 / 636.000."""
    kept = table.filter(pc.equal(table.column("partition"), name))
    if kept.num_rows == 0:
        return {"partition": name, "rows": 0}
    protein = codes(kept.column("protein_hash"))
    endpoint = codes(kept.column("endpoint"))
    # smiles_a và smiles_b phải dùng CHUNG một bảng mã: mã riêng từng cột thì cùng một
    # phân tử ở hai cột lại ra hai mã khác nhau, và số query sai hẳn. Nối hai cột rồi mã
    # một lần, sau đó cắt đôi.
    stacked = pa.chunked_array([flat(kept.column("smiles_a")),
                                flat(kept.column("smiles_b"))])
    shared = codes(stacked)
    first, second = shared[:kept.num_rows], shared[kept.num_rows:]

    verify_against_project(first, protein, endpoint)
    one_total, one_ranked = count_queries(first, protein, endpoint, min_candidates)
    both_reference = np.concatenate([first, second])
    both_protein = np.concatenate([protein, protein])
    both_endpoint = np.concatenate([endpoint, endpoint])
    two_total, two_ranked = count_queries(both_reference, both_protein, both_endpoint,
                                          min_candidates)
    return {
        "partition": name,
        "rows": int(kept.num_rows),
        "protein phân biệt": int(np.unique(protein).size),
        "pair_id phân biệt": int(np.unique(codes(kept.column("pair_id"))).size),
        "query · một chiều · mọi query": one_total,
        "query · một chiều · >= %d ứng viên" % min_candidates: one_ranked,
        "query · hai chiều · mọi query": two_total,
        "query · hai chiều · >= %d ứng viên" % min_candidates: two_ranked,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="",
                        help="file evaluator khac de doi chieu (vi du data/dataset_v3/"
                             "evaluator_training.parquet); mac dinh stage3.inputs.evaluator")
    parser.add_argument("--partitions", nargs="*", default=["train", "validation"],
                        help="partition nào được đếm; MẶC ĐỊNH KHÔNG CÓ test, cố ý")
    parser.add_argument("--min-candidates", type=int, default=0,
                        help="mặc định đọc stage3.metrics.min_candidates của config.json")
    parser.add_argument("--out", default="docs/BASELINE_HANDOFF.md")
    args = parser.parse_args()

    if "test" in args.partitions:
        raise SystemExit("KHÔNG đếm partition test: nó không được đụng tới ngoài lần chốt "
                         "cuối cùng. Bỏ 'test' khỏi --partitions.")

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    inputs = config["stage3"]["inputs"]
    settings = config["stage3"]["metrics"]
    min_candidates = args.min_candidates or int(settings["min_candidates"])

    evaluator = Path(args.dataset) if args.dataset else ROOT / inputs["evaluator"]
    if not evaluator.is_absolute():
        evaluator = ROOT / evaluator
    proteins = ROOT / inputs["proteins"]
    for path in (evaluator, proteins):
        if not path.is_file():
            raise SystemExit(f"thiếu {path}")

    print("Đang băm hai file đầu vào (vài chục giây)...")
    files = []
    for label, path in (("evaluator", evaluator), ("proteins", proteins)):
        digest, size = file_digest(path)
        rows = pq.ParquetFile(path).metadata.num_rows
        files.append({"tên": label, "đường dẫn": str(path.relative_to(ROOT)).replace("\\", "/"),
                      "sha256": digest, "byte": size, "dòng": rows})
        print(f"  {label:<10} {digest}  {size:,} byte  {rows:,} dòng")

    print("Đang đếm query...")
    table = pq.read_table(evaluator, columns=KEY_COLUMNS)
    blocks = [partition_block(table, name, min_candidates) for name in args.partitions]
    for block in blocks:
        print(f"  {block['partition']:<12} {block.get('rows', 0):>12,} dòng")

    lines = [
        "# Gói bàn giao cho nhóm baseline agent",
        "",
        f"Dựng tự động bằng `scripts/export_baseline_handoff.py`, ngày {date.today():%d/%m/%Y}.",
        "Việc 16 biên bản họp 09/11/2026 — điều kiện cứng: baseline agent sẽ được so với",
        "mô hình của paper, nên hai bên phải chạy trên dữ liệu TRÙNG TỪNG DÒNG.",
        "",
        "## 1. Hai file, và cách kiểm trùng từng dòng",
        "",
        "| file | đường dẫn | dòng | byte | sha256 |",
        "|---|---|---|---|---|",
    ]
    for entry in files:
        lines.append(f"| {entry['tên']} | `{entry['đường dẫn']}` | {entry['dòng']:,} | "
                     f"{entry['byte']:,} | `{entry['sha256']}` |")
    lines += [
        "",
        "Bên nhận chạy `sha256sum <file>` (Windows: `certutil -hashfile <file> SHA256`).",
        "Khớp thì chạy tiếp; KHÔNG khớp thì dừng và báo lại, đừng chạy rồi so kết quả —",
        "hai tập lệch nhau một ít dòng vẫn ra số trông hợp lý, và đó là kiểu sai khó phát",
        "hiện nhất.",
        "",
        "## 2. Stage 3 có lọc bớt dòng nào nữa không: KHÔNG",
        "",
        "Chỗ lọc dòng duy nhất của toàn Stage 3 là `read_columns`",
        "(`src/molopt/evaluator/data.py`), và nó chỉ lọc theo cột `partition`. Hai đường",
        "chia khác (`train_subset`, `partition_override`, `partition_override_pairs`) đều",
        "TẮT trong `config.json` mặc định; lần chạy nào bật thì ghi rõ trong manifest của",
        "lần chạy đó. Nên hai file trên một mình đã định nghĩa đủ tập dữ liệu: không có bộ",
        "lọc ngầm nào nằm ngoài chúng.",
        "",
        "Chia tập đọc thẳng từ cột `partition` có sẵn trong file. KHÔNG tự chia lại.",
        "KHÔNG đụng vào `partition == 'test'`.",
        "",
        "## 3. Một \"query\" là gì — chốt bằng số, không bằng lời",
        "",
        "Biên bản ghi \"669.000 vs 636.000 — đã đo là khác cách gom nhóm, không phải khác",
        "dữ liệu\" nhưng chưa ghi cách gom nào ra số nào. Bảng dưới đếm trên chính dữ liệu",
        "này; con số nào khớp thì tự nó chỉ ra hai bên đang nói cách gom nào.",
        "",
        "Định nghĩa của dự án (`metrics.query_groups`, gọi từ `train.py`):",
        "**một query = (phân tử tham chiếu, protein, endpoint)**. Hai thứ làm số đổi:",
        "",
        "* `expand_both_directions` — mỗi dòng (A, B) tính hai lần, A làm tham chiếu rồi B",
        f"  làm tham chiếu. Giá trị hiện tại: `{settings.get('expand_both_directions')}`.",
        f"* `min_candidates` = {min_candidates} — query ít ứng viên hơn ngần ấy bị bỏ khỏi",
        "  mẫu số của `pairwise_accuracy_macro` (`queries_ranked`), nhưng vẫn tồn tại",
        "  (`queries_total`).",
        "",
    ]
    for block in blocks:
        lines += [f"### partition `{block['partition']}`", ""]
        if not block.get("rows"):
            lines += ["(rỗng)", ""]
            continue
        lines += ["| cách đếm | số |", "|---|---|"]
        for key, value in block.items():
            if key in ("partition",):
                continue
            lines.append(f"| {key} | {value:,} |")
        lines.append("")
    lines += [
        "## 4. Thước đo phải khớp",
        "",
        "Số chính của paper là `pairwise_accuracy_macro`: tỉ lệ xếp đúng chiều của các cặp",
        "ứng viên TRONG một query, rồi lấy trung bình THEO QUERY (mỗi query một phiếu).",
        "Không phải micro (gộp mọi cặp rồi mới chia) — query dài sẽ chi phối micro.",
        "",
        f"Ngưỡng bỏ cặp quá gần nhau: `direction_tolerance` = {settings.get('direction_tolerance')}.",
        "Cặp có hai nhãn lệch nhau không quá ngần ấy bị loại khỏi MẪU SỐ. Bên kia không áp",
        "ngưỡng này thì con số không so được, dù dữ liệu trùng từng dòng.",
        "",
        "Lát cắt: `all` (mọi dòng) và `strict` (chỉ `unseen_pair = True`). Số báo cáo của",
        "paper là lát `strict`. Lát `all` KHÔNG dùng để so hai mô hình — đã đo được là nó",
        "đảo cả thứ tự xếp hạng.",
        "",
        "## 5. Việc của bên nhận",
        "",
        "1. Kiểm `sha256` hai file.",
        "2. Chia tập theo cột `partition`, không tự chia lại; không đụng `test`.",
        "3. Báo cáo `pairwise_accuracy_macro` lát `strict`, kèm mẫu số (`queries_ranked`).",
        "4. Chạy ít nhất 3 hạt giống và báo cả 3, không báo mỗi số tốt nhất. Sàn nhiễu hạt",
        "   giống đo được của dự án trên tập đầy đủ là 0,0021; chênh lệch nhỏ hơn ngần ấy",
        "   không kết luận được gì.",
        "",
    ]
    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nđã ghi {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
