"""Đo CƠ CHẾ của ba kết quả âm, thay vì chỉ báo "nó tệ hơn".

Ba kết quả âm cần giải thích (đúng chiều macro trên validation, lát cắt all):
    fusion projector 0,5455 / attention 0,5491  so với  concat 0,5586
    MoLFormer         0,5600                    so với  Morgan 0,5657 +/- 0,0014
    ESM2 650M         0,5614                    so với  ESM2 35M 0,5657 +/- 0,0014

Script này không train gì, chỉ đọc cache đã có và đếm tham số. Ba phép đo:

``ligand``
    Bài toán của ta là xếp hạng những phân tử KHÁC NHAU ĐÚNG MỘT PHÉP SỬA. Nếu biểu
    diễn ánh xạ A và B của cùng một cặp về gần như một điểm thì mô hình gần như không có
    gì để học. Phép đo: khoảng cách cosine trong-cặp (A với B của cùng cặp) chia cho
    khoảng cách cosine giữa-cặp (A với B của một cặp khác, lấy ngẫu nhiên). Tỉ lệ càng
    cao thì biểu diễn càng giữ được phép sửa cục bộ. Dùng cosine cho cả Morgan (nhị
    phân) lẫn MoLFormer (liên tục) để hai bên so được với nhau.

``protein``
    ESM2 gộp theo residue rồi lấy trung bình, nên model lớn hơn chưa chắc cho embedding
    phân biệt protein tốt hơn. Phép đo: khoảng cách cosine trung bình giữa các protein,
    và participation ratio của phổ hiệp phương sai - số chiều THỰC SỰ được dùng. Nếu
    650M có participation ratio không cao hơn 35M thì phần dung lượng thêm đã bị phép
    gộp trung bình xóa đi.

``parameters``
    projector và attention thêm tham số vào khối hợp nhất. Phép đo: số tham số train
    được của từng chế độ fusion, để nói được "thêm bao nhiêu tham số mà không đổi gì".

Cách chạy (local hoặc server, chỉ đọc, không cần GPU):
    python scripts/representation_diagnostics.py ligand
    python scripts/representation_diagnostics.py ligand --vectors embedding
    python scripts/representation_diagnostics.py protein
    python scripts/representation_diagnostics.py protein --cache-dir work/cache_esm650
    python scripts/representation_diagnostics.py parameters
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molopt.evaluator import features  # noqa: E402

BATCH_ROWS = 65536


def load_config() -> dict[str, Any]:
    """Đọc config.json của dự án."""
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def row_cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Khoảng cách cosine theo từng dòng của hai ma trận cùng kích thước.

    Vector toàn 0 (phân tử không có bit nào bật) cho khoảng cách 1, không chia cho 0.
    Cộng dồn bằng float64 trong einsum chứ không đổi kiểu cả ma trận: fingerprint 2048
    bit của 20.000 cặp ở float64 đã là 650 MB mỗi bên, đổi kiểu là hết RAM vô ích.
    """
    dots = np.einsum("ij,ij->i", left, right, dtype=np.float64)
    norms = (np.sqrt(np.einsum("ij,ij->i", left, left, dtype=np.float64))
             * np.sqrt(np.einsum("ij,ij->i", right, right, dtype=np.float64)))
    out = np.ones(len(left), dtype=np.float64)
    usable = norms > 0
    out[usable] = 1.0 - dots[usable] / norms[usable]
    return out


def partition_row_indices(path: Path, partition: str) -> np.ndarray:
    """Chỉ số dòng của một partition, đọc đúng một cột nên rất nhẹ."""
    table = pq.read_table(path, columns=["partition"])
    mask = pc.equal(table.column("partition"), pa.scalar(partition))
    flat = np.asarray(mask.combine_chunks().to_numpy(zero_copy_only=False))
    return np.flatnonzero(flat)


def take_pair_smiles(path: Path, wanted: np.ndarray) -> tuple[list[str], list[str]]:
    """Lấy smiles_a và smiles_b của đúng các dòng đã chọn, đọc theo batch.

    Vì sao không đọc cả cột: hai cột SMILES của 7,9 triệu dòng rất nặng, còn cách này
    chỉ giữ lại số dòng đã chọn.
    """
    chosen = np.sort(wanted)
    first: list[str] = []
    second: list[str] = []
    offset = 0
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(columns=["smiles_a", "smiles_b"],
                                     batch_size=BATCH_ROWS):
        end = offset + batch.num_rows
        inside = chosen[(chosen >= offset) & (chosen < end)] - offset
        if inside.size:
            picked = batch.take(pa.array(inside))
            first.extend(picked.column("smiles_a").to_pylist())
            second.extend(picked.column("smiles_b").to_pylist())
        offset = end
    return first, second


def vector_rows(keys: pa.Array, names: list[str]) -> np.ndarray:
    """Chỉ số dòng trong cache của từng SMILES; thiếu một cái là lỗi, không bỏ qua."""
    lookup = {key: index for index, key in enumerate(keys.to_pylist())}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} SMILES không có trong cache, "
                       f"ví dụ {missing[:3]}")
    return np.fromiter((lookup[name] for name in names), dtype=np.int64,
                       count=len(names))


def describe(values: np.ndarray) -> str:
    """Trung bình, độ lệch chuẩn và trung vị của một mảng, in gọn."""
    return (f"{values.mean():.4f} +/- {values.std(ddof=1):.4f} "
            f"(trung vị {np.median(values):.4f})")


def report_ligand(args: argparse.Namespace) -> None:
    """Biểu diễn phân tử có giữ được phép sửa giữa A và B hay không."""
    config = load_config()
    stage = config["stage3"]
    cache_dir = ROOT / (args.cache_dir or stage["paths"]["cache_dir"])
    name = ("molecule_fingerprints" if args.vectors == "fingerprint"
            else "molecule_embeddings")
    path = cache_dir / stage["files"][name]
    dataset = ROOT / stage["inputs"]["evaluator"]

    # Hai cache có ĐỊNH DẠNG KHÁC NHAU: fingerprint lưu bit đã đóng gói (read_packed
    # _cache + unpack_rows), embedding lưu vector float (read_vector_cache).
    if args.vectors == "fingerprint":
        bits = int(stage["fingerprint"]["bits"])
        keys, packed, metadata = features.read_packed_cache(path)
    else:
        bits = 0
        keys, vectors, metadata = features.read_vector_cache(path)

    indices = partition_row_indices(dataset, args.partition)
    rng = np.random.default_rng(args.seed)
    wanted = rng.choice(indices, size=min(args.rows, indices.size), replace=False)
    smiles_a, smiles_b = take_pair_smiles(dataset, wanted)
    rows_a = vector_rows(keys, smiles_a)
    rows_b = vector_rows(keys, smiles_b)

    if args.vectors == "fingerprint":
        left = features.unpack_rows(packed, rows_a, bits)
        right = features.unpack_rows(packed, rows_b, bits)
    else:
        left = vectors[rows_a]
        right = vectors[rows_b]
    within = row_cosine_distance(left, right)
    between = row_cosine_distance(left, right[rng.permutation(len(right))])

    print(f"biểu diễn phân tử : {args.vectors} | {path.relative_to(ROOT)}")
    print(f"model trong cache : {metadata.get('model', 'Morgan fingerprint (RDKit)')}")
    print(f"số chiều          : {left.shape[1]}")
    print(f"số cặp đã lấy mẫu : {len(within):,} (partition {args.partition}, "
          f"seed {args.seed})")
    print(f"cosine TRONG cặp  : {describe(within)}")
    print(f"cosine GIỮA cặp   : {describe(between)}")
    print(f"TỈ LỆ trong/giữa  : {within.mean() / between.mean():.4f}")
    for limit in (0.01, 0.05):
        share = 100.0 * float((within < limit).mean())
        print(f"  cặp có cosine < {limit:.2f}: {share:.2f} % (biểu diễn gần như "
              f"không phân biệt được A với B)")


def report_protein(args: argparse.Namespace) -> None:
    """Embedding protein lớn hơn có dùng thêm chiều nào thật không."""
    config = load_config()
    stage = config["stage3"]
    cache_dir = ROOT / (args.cache_dir or stage["paths"]["cache_dir"])
    path = cache_dir / stage["files"]["protein_embeddings"]
    keys, matrix, metadata = features.read_vector_cache(path)

    rng = np.random.default_rng(args.seed)
    size = min(args.rows, len(matrix))
    picked = matrix[rng.choice(len(matrix), size=size, replace=False)]
    left = picked[rng.permutation(size)]
    right = picked[rng.permutation(size)]
    distances = row_cosine_distance(left, right)

    centered = picked.astype(np.float64) - picked.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centered, compute_uv=False) ** 2
    total = float(spectrum.sum())
    participation = total ** 2 / float((spectrum ** 2).sum()) if total > 0 else 0.0

    print(f"embedding protein : {path.relative_to(ROOT)}")
    print(f"model trong cache : {metadata.get('model', 'chưa ghi')}")
    print(f"số protein        : {len(matrix):,} | số chiều {matrix.shape[1]}")
    print(f"max_residues      : {metadata.get('max_residues', 'chưa ghi')} | "
          f"long_sequence {metadata.get('long_sequence', 'chưa ghi')}")
    print(f"số protein lấy mẫu: {size:,} (seed {args.seed})")
    print(f"cosine giữa 2 protein ngẫu nhiên: {describe(distances)}")
    print(f"participation ratio             : {participation:.2f} / "
          f"{matrix.shape[1]} chiều "
          f"({100.0 * participation / matrix.shape[1]:.2f} % số chiều được dùng)")


def report_parameters(args: argparse.Namespace) -> None:
    """Mỗi chế độ fusion thêm bao nhiêu tham số train được."""
    from molopt.evaluator import models

    config = load_config()
    stage = config["stage3"]
    base = dict(stage["model"])
    protein_dim = int(stage["protein_encoder"]["embedding_dim"])
    molecule_bits = int(stage["fingerprint"]["bits"])
    fragment_bits = int(stage["fingerprint"]["fragment_bits"])

    print(f"molecule_bits {molecule_bits} | fragment_bits {fragment_bits} | "
          f"protein_dim {protein_dim} | endpoint {args.endpoints}")
    print(f"{'fusion':<12}{'head_block':<12}{'tham số':>14}{'so với concat':>16}")
    reference: dict[str, int] = {}
    for block in ("plain", "residual"):
        for fusion in ("concat", "projector", "attention"):
            settings = {**base, "fusion": fusion, "head_block": block}
            model = models.EvaluatorModel(settings, molecule_bits, fragment_bits,
                                          protein_dim, args.endpoints,
                                          models.HEAD_DIRECT, models.PROTEIN_FROZEN)
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if fusion == "concat":
                reference[block] = count
            print(f"{fusion:<12}{block:<12}{count:>14,}"
                  f"{count - reference[block]:>+16,}")


def main() -> int:
    """Đọc tham số dòng lệnh rồi chạy đúng một phép đo."""
    # Console Windows mặc định là cp1252, không in được chữ có dấu; không đặt lại thì
    # script chết vì UnicodeEncodeError chứ không phải vì phép đo sai.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Đo cơ chế của các kết quả âm.")
    parser.add_argument("what", choices=("ligand", "protein", "parameters"))
    parser.add_argument("--cache-dir", default=None,
                        help="thư mục cache khác stage3.paths.cache_dir, ví dụ "
                             "work/evaluator_cache_v4_esm650")
    parser.add_argument("--vectors", choices=("fingerprint", "embedding"),
                        default="fingerprint", help="ligand: Morgan hay MoLFormer")
    parser.add_argument("--rows", type=int, default=20000, help="số mẫu")
    parser.add_argument("--partition", default="train")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--endpoints", type=int, default=4,
                        help="parameters: số endpoint (Ki, Kd, IC50, EC50)")
    args = parser.parse_args()

    if args.what == "ligand":
        report_ligand(args)
    elif args.what == "protein":
        report_protein(args)
    else:
        report_parameters(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
