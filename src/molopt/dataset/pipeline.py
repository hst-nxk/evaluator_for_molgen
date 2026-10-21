"""Stage 2b: điều phối Observations -> dataset cặp MMP có Δp.

Luồng (mỗi bước ở một module riêng, file này chỉ nối chúng lại và lo I/O):
    lượt gom (pass 1): đọc hai bản sao dedup, lọc dòng đủ điều kiện, dựng assay_key và
        khóa phân tử, ghi ra shard theo (endpoint, bucket protein); đồng thời dựng bảng
        phân tử (SMILES đại diện) và bảng protein.
    bước 1  aggregate.collapse_replicates : gộp replicate trong assay.
    bước 2  mmp.find_pairs                : cặp MMP trong từng (assay, protein).
    bước 3  delta.assay_deltas            : Δp = p(B) − p(A) trong từng assay.
    bước 4  aggregate.aggregate_deltas    : gộp Δp giữa các assay, QC.
    dựng split  split.py                  : gom protein của dataset rồi chia 80/10/10.
    bước 5  run_export                    : hai file training + file evidence.

Vì sao chia shard theo bucket protein: mọi bước đều nhóm theo protein (bước 1 nhóm trong
một assay của một protein; cặp MMP chỉ trong một assay; bản ghi bước 4 khóa theo protein),
nên không có phụ thuộc giữa các bucket. Xử lý từng (endpoint, bucket) giữ bộ nhớ ở mức một
phần nhỏ của dữ liệu, chạy được trên máy 16 GB.

Mọi tham số nằm ở config.json mục stage2b. File này không sửa dữ liệu đầu vào.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import zlib
from collections import Counter
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from molopt.dataset import aggregate, delta, mmp, split

LOGGER = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
# Thư mục output của full được dựng dưới tên tạm rồi mới đổi tên, để một lần chạy bị ngắt
# không để lại thư mục trông như đã hoàn chỉnh.
PARTIAL_SUFFIX = ".partial"

# Vai trò cột cần đọc từ bản sao dedup (ánh xạ vai trò -> tên cột ở config stage2b.columns).
LOAD_ROLES = ("observation_id", "source", "assay_id", "assay_entry_id", "protein_hash",
              "protein_sequence", "protein_accession", "protein_name", "endpoint",
              "inchikey", "smiles", "cx_extension", "molecule_source_id", "qualifier",
              "p_activity", "protein_status", "assay_status", "keep", "activity_context")

# Shard: đúng những cột bước 1 cần, cộng observation_id để tra lại được dòng gốc.
SHARD_SCHEMA = pa.schema([
    ("observation_id", pa.string()),
    ("source", pa.string()),
    ("assay_key", pa.string()),
    ("protein_hash", pa.string()),
    ("molecule_key", pa.string()),
    ("source_id", pa.string()),
    ("p_activity", pa.float64()),
    ("independent", pa.bool_()),
])

# Hai chữ số hex đầu của protein_sequence_hash -> số bucket (xem bucket_of).
HEX_PAIRS = [f"{value:02x}" for value in range(256)]

# Nhãn các bước lọc, theo đúng thứ tự áp dụng (xem filter_steps); báo cáo in số dòng còn
# lại sau mỗi bước nên nhãn phải cố định ở một chỗ.
FILTER_LABELS = ("dedup_keep", "qualifier '='", "protein resolved", "assay resolved",
                 "có pActivity", "có InChIKey", "có SMILES")
FUNNEL_ALL = "tất cả"
FUNNEL_SMOKE = "mẫu smoke"


# --------------------------------------------------------------------------- định dạng


def fmt_int(value: int) -> str:
    """Số nguyên có dấu chấm phân cách hàng nghìn, vd. 5338213 -> '5.338.213'."""
    return f"{value:,}".replace(",", ".")


def fmt_float(value: float | None, digits: int = 2) -> str:
    """Số thực kiểu Việt Nam (dấu phẩy thập phân); None hoặc NaN -> '–'."""
    if value is None or (isinstance(value, float) and value != value):
        return "–"
    text = f"{value:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_pct(part: float, whole: float, digits: int = 1) -> str:
    """Tỷ lệ phần trăm part/whole, vd. '12,3 %'; whole = 0 -> '–'."""
    return "–" if not whole else fmt_float(100.0 * part / whole, digits) + " %"


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Bảng căn cột: cột đầu căn trái, các cột sau căn phải (cột số)."""
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render(cells: Sequence[str]) -> str:
        parts = [cells[0].ljust(widths[0])]
        parts += [cell.rjust(width) for cell, width in zip(cells[1:], widths[1:])]
        return "  ".join(parts)

    return [render(headers), "  ".join("-" * width for width in widths)] + [
        render(row) for row in rows]


def percentile_label(percentile: float) -> str:
    """Nhãn cột phân vị: 0 -> 'min', 50 -> 'median', 100 -> 'max', còn lại 'P<q>'."""
    return {0: "min", 50: "median", 100: "max"}.get(percentile, f"P{percentile:g}")


def log_lines(lines: Sequence[str]) -> None:
    """In từng dòng báo cáo qua logging (vào cả màn hình và file log của lần chạy)."""
    for line in lines:
        LOGGER.info("%s", line)


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 của một file, đọc theo khối nên không nạp cả file vào bộ nhớ."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- đường dẫn


@dataclass(frozen=True)
class Layout:
    """Đường dẫn trong một thư mục output của Stage 2b (full, hay một lần smoke)."""

    root: Path
    files: Mapping[str, str]

    def file(self, key: str) -> Path:
        """File đơn, vd. file('molecules') -> <root>/molecules.parquet."""
        return self.root / self.files[key]

    def bucket_file(self, key: str, endpoint: str, bucket: int) -> Path:
        """File của một (endpoint, bucket), vd. <root>/records/Ki/b03.parquet."""
        return self.root / self.files[key] / endpoint / f"b{bucket:02d}.parquet"

    def bucket_files(self, key: str, endpoint: str) -> list[Path]:
        """Mọi file bucket đã có của một endpoint, theo thứ tự bucket."""
        directory = self.root / self.files[key] / endpoint
        return sorted(directory.glob("b*.parquet")) if directory.is_dir() else []


def layouts(config: Mapping[str, Any], root: Path) -> tuple[Layout, Layout]:
    """(Layout của thư mục full, Layout của thư mục dataset) cho output chính thức."""
    settings = config["stage2b"]
    return (Layout(root / settings["paths"]["full_dir"], settings["files"]),
            Layout(root / settings["paths"]["dataset_dir"], settings["files"]))


# --------------------------------------------------------------------------- lọc dòng


def rename_roles(batch: pa.RecordBatch, columns: Mapping[str, str],
                 roles: Sequence[str]) -> pa.Table:
    """Batch đọc từ Parquet -> bảng có tên cột là tên vai trò (lấy theo tên, không theo vị trí)."""
    table = pa.Table.from_batches([batch])
    return pa.table({role: table.column(columns[role]) for role in roles})


def non_empty(values: pa.ChunkedArray | pa.Array) -> pa.ChunkedArray | pa.Array:
    """True ở ô có nội dung (không null và không rỗng)."""
    return pc.fill_null(pc.greater(pc.utf8_length(values), 0), False)


def filter_steps(table: pa.Table, exact_qualifiers: Sequence[str],
                 resolved: str) -> list[tuple[str, Any]]:
    """Các bước lọc dòng vào dataset, theo thứ tự, mỗi bước một mặt nạ bool.

    Giữ từng bước riêng để báo cáo in được số dòng còn lại sau mỗi bước (phễu lọc).
    Dòng ChEMBL có cờ potential_duplicate KHÔNG bị lọc ở đây: nó được giữ, chỉ không
    tính là phép đo độc lập (xem independent_flags).
    """
    masks = [
        table.column("keep"),
        pc.is_in(table.column("qualifier"),
                 value_set=pa.array(list(exact_qualifiers), pa.string())),
        pc.equal(table.column("protein_status"), resolved),
        pc.equal(table.column("assay_status"), resolved),
        pc.is_valid(table.column("p_activity")),
        non_empty(table.column("inchikey")),
        non_empty(table.column("smiles")),
    ]
    return list(zip(FILTER_LABELS, masks))


def combined_mask(steps: Sequence[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Mặt nạ cộng dồn sau mỗi bước (null coi là False)."""
    result, mask = [], None
    for label, step in steps:
        step = pc.fill_null(step, False)
        mask = step if mask is None else pc.and_(mask, step)
        result.append((label, mask))
    return result


def parse_context(text: str, observation_id: str) -> dict[str, Any]:
    """activity_context (JSON) của một dòng; lỗi nêu rõ observation nào và vì sao."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{observation_id}: activity_context không phải JSON hợp lệ "
                         f"({error})") from error


def independent_flags(table: pa.Table, settings: Mapping[str, Any]) -> pa.Array:
    """True ở dòng được tính là phép đo độc lập (không có cờ potential_duplicate).

    Chỉ parse JSON ở những dòng mà regex đã thấy khả năng có cờ bật; dòng còn lại chắc
    chắn không có cờ (giá trị 1 hay "1" đều khớp regex), nên không phải đọc JSON của
    hàng triệu dòng. Dòng đã lọc ra vẫn được parse đầy đủ rồi so bằng khóa/giá trị.
    """
    contexts = table.column("activity_context")
    candidates = pc.fill_null(
        pc.match_substring_regex(contexts, pattern=settings["potential_duplicate_regex"]), False)
    flags = np.ones(table.num_rows, dtype=bool)
    positions = np.flatnonzero(np.asarray(candidates.to_numpy(zero_copy_only=False), dtype=bool))
    if positions.size:
        take = pa.array(positions, type=pa.int64())
        texts = pc.take(contexts, take).to_pylist()
        ids = pc.take(table.column("observation_id"), take).to_pylist()
        key = settings["potential_duplicate_key"]
        wanted = str(settings["potential_duplicate_value"])
        for position, text, observation_id in zip(positions, texts, ids):
            value = parse_context(text, observation_id).get(key)
            if value is not None and str(value) == wanted:
                flags[position] = False
    return pa.array(flags, type=pa.bool_())


def assay_keys(table: pa.Table) -> pa.Array:
    """Cột assay_key = '<nguồn>:<ENTRYID hoặc rỗng>:<assay>'.

    Ở BindingDB, ASSAYID chỉ duy nhất trong một ENTRYID, nên khóa assay phải gồm cả hai;
    tiền tố nguồn giữ cho assay của hai nguồn không bao giờ bị coi là một (giống Stage 2a).
    """
    return pc.binary_join_element_wise(
        table.column("source"), pc.fill_null(table.column("assay_entry_id"), ""),
        pc.fill_null(table.column("assay_id"), ""), ":")


def bucket_of(hashes: pa.ChunkedArray | pa.Array, buckets: int) -> np.ndarray:
    """Số bucket của mỗi protein, từ hai chữ số hex đầu của protein_sequence_hash.

    Hash là SHA-256 dạng hex nên hai ký tự đầu phân bố đều; lấy phần dư cho `buckets`
    chia protein thành các phần gần bằng nhau và cố định giữa các lần chạy.
    """
    prefix = pc.utf8_slice_codeunits(hashes, 0, 2)
    position = pc.index_in(prefix, value_set=pa.array(HEX_PAIRS, pa.string()))
    codes = np.asarray(pc.fill_null(position, -1).to_numpy(zero_copy_only=False),
                       dtype=np.int64)
    if (codes < 0).any():
        bad = pc.take(hashes, pa.array(np.flatnonzero(codes < 0)[:1], pa.int64())).to_pylist()
        raise ValueError(f"protein_sequence_hash không phải chuỗi hex: {bad}")
    return codes % buckets


def endpoint_codes(values: pa.ChunkedArray | pa.Array, endpoints: Sequence[str]) -> np.ndarray:
    """Số hiệu endpoint của mỗi dòng theo config['endpoints']; không khớp -> −1."""
    position = pc.index_in(values, value_set=pa.array(list(endpoints), pa.string()))
    return np.asarray(pc.fill_null(position, -1).to_numpy(zero_copy_only=False), dtype=np.int64)


def smoke_mask(keys: pa.ChunkedArray | pa.Array, modulus: int) -> pa.Array:
    """Mẫu smoke: chỉ giữ assay có CRC32(assay_key) chia hết cho modulus.

    Lấy mẫu theo assay (không theo dòng) vì bước 1 và bước 2 chỉ có nghĩa khi cả assay
    còn nguyên. CRC32 cố định nên hai lần chạy smoke lấy đúng cùng tập assay.
    """
    selected = [zlib.crc32(key.encode("utf-8")) % modulus == 0 if key is not None else False
                for key in keys.to_pylist()]
    return pa.array(selected, type=pa.bool_())


def shard_rows(table: pa.Table, settings: Mapping[str, Any],
               pattern: str, suffix: str) -> pa.Table:
    """Các dòng đã lọc -> bảng theo SHARD_SCHEMA (thêm khóa phân tử và cờ độc lập)."""
    relative = mmp.relative_stereo_mask(table.column("cx_extension"), pattern)
    return pa.table({
        "observation_id": table.column("observation_id"),
        "source": table.column("source"),
        "assay_key": table.column("assay_key"),
        "protein_hash": table.column("protein_hash"),
        "molecule_key": mmp.identity_keys(table.column("inchikey"), relative, suffix),
        "source_id": table.column("molecule_source_id"),
        "p_activity": pc.cast(table.column("p_activity"), pa.float64()),
        "independent": independent_flags(table, settings),
    }, schema=SHARD_SCHEMA)


# --------------------------------------------------------------------------- gom dần


class GroupAccumulator:
    """Gom dần một bảng tổng hợp theo khóa, hợp nhất sau mỗi `merge_every` lô.

    Dùng cho bảng phân tử và bảng protein: cả hai nhỏ hơn dữ liệu nhiều lần nhưng vẫn
    phải dựng trong lúc đọc, nên không giữ mọi lô trong bộ nhớ mà hợp nhất định kỳ.
    Cột giá trị lấy min (nhỏ nhất theo thứ tự chuỗi) làm đại diện ổn định; cột 'rows'
    cộng dồn số dòng.
    """

    def __init__(self, key: str, values: Sequence[str], merge_every: int) -> None:
        self.key = key
        self.values = list(values)
        self.merge_every = merge_every
        self.parts: list[pa.Table] = []

    def _aggregate(self, table: pa.Table) -> pa.Table:
        grouped = table.group_by(self.key).aggregate(
            [(column, "min") for column in self.values] + [("rows", "sum")])
        names = [self.key] + [f"{column}_min" for column in self.values] + ["rows_sum"]
        return grouped.select(names).rename_columns([self.key] + self.values + ["rows"])

    def add(self, table: pa.Table) -> None:
        """Thêm một lô: bảng có cột khóa, các cột giá trị, và cột 'rows'."""
        self.parts.append(self._aggregate(table))
        if len(self.parts) >= self.merge_every:
            self.merge()

    def merge(self) -> None:
        """Hợp nhất các lô đang giữ thành một bảng duy nhất."""
        if len(self.parts) > 1:
            self.parts = [self._aggregate(pa.concat_tables(self.parts))]

    def table(self) -> pa.Table:
        """Bảng kết quả, sắp theo khóa (thứ tự cố định giữa các lần chạy)."""
        self.merge()
        if not self.parts:
            return pa.table({self.key: pa.array([], pa.string()),
                             **{column: pa.array([], pa.string()) for column in self.values},
                             "rows": pa.array([], pa.int64())})
        return self.parts[0].sort_by([(self.key, "ascending")])


class ShardWriter:
    """Ghi các dòng đã lọc thành file Parquet theo (endpoint, bucket), mở file khi cần."""

    def __init__(self, layout: Layout, key: str) -> None:
        self.layout = layout
        self.key = key
        self.writers: dict[tuple[str, int], pq.ParquetWriter] = {}
        self.rows: Counter = Counter()

    def write(self, endpoint: str, bucket: int, table: pa.Table) -> None:
        """Ghi thêm một phần dòng của (endpoint, bucket)."""
        handle = self.writers.get((endpoint, bucket))
        if handle is None:
            path = self.layout.bucket_file(self.key, endpoint, bucket)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = pq.ParquetWriter(path, SHARD_SCHEMA)
            self.writers[(endpoint, bucket)] = handle
        handle.write_table(table)
        self.rows[(endpoint, bucket)] += table.num_rows

    def close(self) -> None:
        """Đóng mọi file đang mở."""
        for handle in self.writers.values():
            handle.close()
        self.writers.clear()


# --------------------------------------------------------------------------- lượt gom


def collect_rows(config: Mapping[str, Any], root: Path, layout: Layout,
                 modulus: int | None) -> dict[str, Any]:
    """Lượt gom: đọc hai bản sao dedup, lọc, ghi shard, dựng bảng phân tử và protein.

    Nhận: config; thư mục gốc dự án; layout output; modulus lấy mẫu smoke (None = lấy hết).
    Trả về: thống kê (phễu lọc theo endpoint, số dòng mỗi nguồn, số shard, số phân tử,
    số protein).
    Đọc theo batch, mỗi batch ghi ngay ra shard, nên bộ nhớ không phụ thuộc kích thước
    dữ liệu. Dữ liệu đầu vào chỉ được đọc.
    """
    settings = config["stage2b"]
    columns, run = settings["columns"], settings["run"]
    endpoints = list(config["endpoints"])
    buckets = settings["delta"]["protein_buckets"]
    pattern = config["stage2a"]["dedup"]["relative_stereo_pattern"]
    suffix = settings["identity"]["relative_stereo_suffix"]
    names = [columns[role] for role in LOAD_ROLES]

    molecules = GroupAccumulator("molecule_key", ["smiles"], run["molecule_merge_batches"])
    proteins = GroupAccumulator("protein_hash",
                                ["protein_sequence", "protein_accession", "protein_name"],
                                run["molecule_merge_batches"])
    writer = ShardWriter(layout, "shards_dir")
    funnel = {label: np.zeros(len(endpoints), dtype=np.int64)
              for label in [FUNNEL_ALL, *FILTER_LABELS]
              + ([FUNNEL_SMOKE] if modulus else [])}
    source_rows: dict[str, int] = {}

    try:
        for source, relative_path in settings["inputs"].items():
            path = root / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"không thấy file đầu vào {path}")
            parquet = pq.ParquetFile(path)
            total, seen = parquet.metadata.num_rows, 0
            LOGGER.info("đọc %s (%s dòng)", path.name, fmt_int(total))
            for batch in parquet.iter_batches(batch_size=run["batch_rows"], columns=names):
                table = rename_roles(batch, columns, LOAD_ROLES)
                table = table.append_column("assay_key", assay_keys(table))
                codes = endpoint_codes(table.column("endpoint"), endpoints)
                steps = combined_mask(filter_steps(table, config["qualifiers"]["exact"],
                                                   settings["filter"]["resolved_status"]))
                funnel[FUNNEL_ALL] += np.bincount(codes[codes >= 0],
                                                  minlength=len(endpoints))
                for label, mask in steps:
                    flags = np.asarray(mask.to_numpy(zero_copy_only=False), dtype=bool)
                    funnel[label] += np.bincount(codes[flags & (codes >= 0)],
                                                 minlength=len(endpoints))
                kept = table.filter(steps[-1][1])
                kept_codes = codes[np.asarray(steps[-1][1].to_numpy(zero_copy_only=False),
                                              dtype=bool)]
                if modulus:
                    sample = smoke_mask(kept.column("assay_key"), modulus)
                    flags = np.asarray(sample.to_numpy(zero_copy_only=False), dtype=bool)
                    kept, kept_codes = kept.filter(sample), kept_codes[flags]
                    funnel[FUNNEL_SMOKE] += np.bincount(kept_codes[kept_codes >= 0],
                                                        minlength=len(endpoints))
                if kept.num_rows:
                    if (kept_codes < 0).any():
                        raise ValueError("endpoint của dòng không nằm trong config['endpoints']")
                    rows = shard_rows(kept, settings["filter"], pattern, suffix)
                    molecules.add(pa.table({
                        "molecule_key": rows.column("molecule_key"),
                        "smiles": kept.column("smiles"),
                        "rows": pa.array(np.ones(rows.num_rows, dtype=np.int64))}))
                    proteins.add(pa.table({
                        "protein_hash": kept.column("protein_hash"),
                        "protein_sequence": kept.column("protein_sequence"),
                        "protein_accession": kept.column("protein_accession"),
                        "protein_name": kept.column("protein_name"),
                        "rows": pa.array(np.ones(rows.num_rows, dtype=np.int64))}))
                    write_shards(writer, rows, kept_codes, endpoints, buckets)
                seen += batch.num_rows
                LOGGER.info("  %s: %s/%s dòng", source, fmt_int(seen), fmt_int(total))
            source_rows[source] = total
    finally:
        writer.close()

    molecule_table = molecules.table()
    molecule_table = molecule_table.append_column(
        "relative_stereo", pc.ends_with(molecule_table.column("molecule_key"), pattern=suffix))
    layout.file("molecules").parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(molecule_table, layout.file("molecules"))
    protein_table = proteins.table()
    protein_table = protein_table.append_column(
        "sequence_length", pc.utf8_length(protein_table.column("protein_sequence")))
    pq.write_table(protein_table, layout.file("proteins"))
    return {"source_rows": source_rows,
            "funnel": {label: counts.tolist() for label, counts in funnel.items()},
            "funnel_order": list(funnel),
            "endpoints": endpoints,
            "shard_rows": {f"{endpoint}/b{bucket:02d}": count
                           for (endpoint, bucket), count in sorted(writer.rows.items())},
            "molecules": molecule_table.num_rows,
            "proteins": protein_table.num_rows}


def write_shards(writer: ShardWriter, rows: pa.Table, codes: np.ndarray,
                 endpoints: Sequence[str], buckets: int) -> None:
    """Chia một lô dòng vào các shard (endpoint, bucket) rồi ghi.

    Sắp một lần theo mã (endpoint, bucket) rồi cắt thành đoạn liên tiếp, thay vì lọc lại
    bảng cho từng shard.
    """
    combined = codes * buckets + bucket_of(rows.column("protein_hash"), buckets)
    order = np.argsort(combined, kind="stable")
    ordered = rows.take(pa.array(order, type=pa.int64()))
    edges = np.searchsorted(combined[order], np.arange(len(endpoints) * buckets + 1))
    for code in range(len(endpoints) * buckets):
        first, last = int(edges[code]), int(edges[code + 1])
        if last > first:
            writer.write(endpoints[code // buckets], code % buckets,
                         ordered.slice(first, last - first))


# --------------------------------------------------------------------------- bước 1 → 4


def attach_smiles(groups: pa.Table, molecules: pa.Table) -> pa.Table:
    """Thêm cột smiles (SMILES đại diện của khóa phân tử) vào bảng bước 1."""
    keys = molecules.column("molecule_key").combine_chunks()
    values = molecules.column("smiles").combine_chunks()
    position = pc.index_in(groups.column("molecule_key"), value_set=keys)
    if pc.sum(pc.is_null(position)).as_py():
        missing = groups.filter(pc.is_null(position)).column("molecule_key").to_pylist()[:3]
        raise ValueError(f"khóa phân tử không có trong bảng phân tử: {missing}")
    return groups.append_column("smiles", pc.take(values, position))


def iter_tasks(groups: pa.Table, suffix: str, groups_per_task: int
               ) -> Iterator[list[tuple[int, list[mmp.GroupMolecule]]]]:
    """Chia bảng bước 1 thành các lô nhóm (assay, protein) để tìm cặp.

    Chỉ nhóm có từ hai phân tử trở lên mới có thể sinh cặp. Chỉ số dòng trong
    GroupMolecule là chỉ số tuyệt đối ở `groups`, nên bước 3 lấy lại được mọi cột.
    """
    bounds = aggregate.group_bounds(groups, ["assay_key", "protein_hash"])
    keys = groups.column("molecule_key").to_pylist()
    smiles = groups.column("smiles").to_pylist()
    task: list[tuple[int, list[mmp.GroupMolecule]]] = []
    for index in range(bounds.size - 1):
        first, last = int(bounds[index]), int(bounds[index + 1])
        if last - first < 2:
            continue
        task.append((index, [mmp.GroupMolecule(row, smiles[row],
                                               mmp.is_relative_key(keys[row], suffix))
                             for row in range(first, last)]))
        if len(task) >= groups_per_task:
            yield task
            task = []
    if task:
        yield task


def find_group_pairs(groups: pa.Table, settings: Mapping[str, Any], suffix: str,
                     pool: ProcessPoolExecutor | None
                     ) -> tuple[list[mmp.Pair], list[tuple[int, int]], Counter]:
    """Tìm cặp MMP cho mọi nhóm của một (endpoint, bucket).

    Nhận: bảng bước 1 đã có cột smiles; config stage2b.mmp; hậu tố lập thể tương đối;
    pool tiến trình con (None = chạy trong tiến trình hiện tại).
    Trả về: (cặp đã sắp theo (row_x, row_y), cặp mất vì quy tắc |r|, số đếm phụ).
    Sắp lại kết quả vì các tiến trình con trả về theo thứ tự hoàn thành, còn output phải
    giống nhau giữa các lần chạy.
    """
    pairs: list[mmp.Pair] = []
    lost: list[tuple[int, int]] = []
    counters: Counter = Counter()

    def consume(result: mmp.TaskResult) -> None:
        pairs.extend(pair for _, pair in result.pairs)
        lost.extend((row_x, row_y) for _, row_x, row_y in result.lost)
        counters["molecules"] += result.molecules
        counters["attachment_violations"] += result.attachment_violations
        counters["failed_smiles"] += result.failed_smiles

    tasks = iter_tasks(groups, suffix, settings["groups_per_task"])
    if pool is None:
        mmp.init_worker(fragment_settings(settings), settings["fragment_cache_size"])
        for task in tasks:
            consume(mmp.process_task(task))
    else:
        window = max(1, settings["workers"] * settings["pending_tasks_per_worker"])
        pending: set[Future] = set()
        for task in tasks:
            pending.add(pool.submit(mmp.process_task, task))
            if len(pending) >= window:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    consume(future.result())
        for future in as_completed(pending):
            consume(future.result())
    pairs.sort(key=lambda pair: (pair.row_x, pair.row_y))
    lost.sort()
    return pairs, lost, counters


def fragment_settings(settings: Mapping[str, Any]) -> mmp.FragmentSettings:
    """config stage2b.mmp -> FragmentSettings (bỏ các khóa chỉ về cách chạy)."""
    return mmp.FragmentSettings(
        min_cuts=settings["min_cuts"], max_cuts=settings["max_cuts"],
        max_cut_bonds=settings["max_cut_bonds"],
        min_core_fraction=settings["min_core_fraction"],
        max_variable_heavy_atoms=settings["max_variable_heavy_atoms"],
        attachment_points=settings["attachment_points"],
        attachment_degree=settings["attachment_degree"])


def dispersion_counts(records: pa.Table, levels: Sequence[float]) -> Counter:
    """Số bản ghi theo mức phân tán: đo được, ≤ mức, > mức, và không đo được (1 assay)."""
    values = np.asarray(records.column("dispersion").to_numpy(zero_copy_only=False),
                        dtype=np.float64)
    measured = ~np.isnan(values)
    counts: Counter = Counter({"measured": int(measured.sum()),
                               "single": int((~measured).sum())})
    for level in levels:
        counts[f"le_{level}"] = int((measured & (values <= level)).sum())
        counts[f"gt_{level}"] = int((measured & (values > level)).sum())
    return counts


def write_parquet(table: pa.Table, path: Path) -> None:
    """Ghi một bảng ra Parquet, tạo thư mục nếu chưa có."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def process_bucket(layout: Layout, endpoint: str, bucket: int, molecules: pa.Table,
                   config: Mapping[str, Any], pool: ProcessPoolExecutor | None
                   ) -> dict[str, Any]:
    """Chạy bước 1 → 4 cho một (endpoint, bucket) và ghi bốn file output của nó.

    Trả về: số đếm của bucket (dòng, nhóm, cặp, cặp mất, bản ghi, cờ QC, phân tán...).
    """
    settings = config["stage2b"]
    shard = layout.bucket_file("shards_dir", endpoint, bucket)
    if not shard.is_file():
        return {}
    started = time.perf_counter()
    rows = pq.read_table(shard)
    replicate = settings["replicate"]
    groups = aggregate.collapse_replicates(rows, replicate["merge_tolerance"],
                                           replicate["float_slack"], replicate["std_ddof"])
    write_parquet(groups.append_column(
        "endpoint", pa.array([endpoint] * groups.num_rows, pa.string())),
        layout.bucket_file("replicates_dir", endpoint, bucket))
    groups = attach_smiles(groups, molecules)

    suffix = settings["identity"]["relative_stereo_suffix"]
    pairs, lost, counters = find_group_pairs(groups, settings["mmp"], suffix, pool)
    deltas = delta.assay_deltas(groups, pairs, endpoint, settings["delta"]["pair_id_length"])
    write_parquet(deltas, layout.bucket_file("assay_deltas_dir", endpoint, bucket))
    write_parquet(delta.lost_pairs_table(groups, lost, endpoint),
                  layout.bucket_file("lost_pairs_dir", endpoint, bucket))

    thresholds = settings["qc"]["dispersion_thresholds"]
    if endpoint not in thresholds:
        raise KeyError(f"thiếu ngưỡng phân tán cho endpoint {endpoint} trong config")
    records = aggregate.aggregate_deltas(deltas, thresholds[endpoint],
                                         settings["qc"]["dispersion_ddof"])
    records = records.append_column("endpoint",
                                    pa.array([endpoint] * records.num_rows, pa.string()))
    write_parquet(records, layout.bucket_file("records_dir", endpoint, bucket))

    qc = Counter(records.column("qc_status").to_pylist())
    stats = {"rows": rows.num_rows, "groups": groups.num_rows,
             "multi_source_id_groups": int(pc.sum(groups.column("multi_source_id")).as_py() or 0),
             "assay_deltas": deltas.num_rows, "pairs_lost_relative": len(lost),
             "records": records.num_rows,
             "qc": {status: qc.get(status, 0) for status in aggregate.QC_STATUSES},
             "dispersion": dict(dispersion_counts(records, settings["qc"]["report_levels"])),
             "fragment_molecules": counters["molecules"],
             "attachment_violations": counters["attachment_violations"],
             "failed_smiles": counters["failed_smiles"],
             "seconds": round(time.perf_counter() - started, 1)}
    LOGGER.info("%s/b%02d: %s dòng -> %s nhóm -> %s Δp assay -> %s bản ghi (%s s)",
                endpoint, bucket, fmt_int(stats["rows"]), fmt_int(stats["groups"]),
                fmt_int(stats["assay_deltas"]), fmt_int(stats["records"]), stats["seconds"])
    return stats


def merge_stats(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cộng dồn số đếm của nhiều bucket thành số đếm của một endpoint."""
    total: dict[str, Any] = {"qc": Counter(), "dispersion": Counter()}
    for part in parts:
        for key, value in part.items():
            if key in ("qc", "dispersion"):
                total[key].update(value)
            else:
                total[key] = total.get(key, 0) + value
    total["qc"], total["dispersion"] = dict(total["qc"]), dict(total["dispersion"])
    return total


# --------------------------------------------------------------------------- báo cáo


def funnel_lines(collected: Mapping[str, Any]) -> list[str]:
    """Bảng phễu lọc: số dòng còn lại sau mỗi bước lọc, theo endpoint."""
    endpoints = collected["endpoints"]
    rows = [[label, *[fmt_int(count) for count in collected["funnel"][label]],
             fmt_int(sum(collected["funnel"][label]))]
            for label in collected["funnel_order"]]
    return ["Phễu lọc dòng vào dataset (số dòng còn lại sau mỗi bước):",
            *format_table(["bước", *endpoints, "tổng"], rows)]


def endpoint_lines(per_endpoint: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Bảng số liệu chính theo endpoint."""
    rows = []
    for endpoint, stats in per_endpoint.items():
        rows.append([endpoint, fmt_int(stats.get("rows", 0)), fmt_int(stats.get("groups", 0)),
                     fmt_int(stats.get("assay_deltas", 0)), fmt_int(stats.get("records", 0)),
                     fmt_int(stats.get("pairs_lost_relative", 0)),
                     fmt_int(stats.get("multi_source_id_groups", 0))])
    return ["Theo endpoint:",
            *format_table(["endpoint", "dòng", "nhóm bước 1", "Δp assay", "bản ghi",
                           "cặp mất |r|", "nhóm nhiều ID"], rows)]


def qc_lines(per_endpoint: Mapping[str, Mapping[str, Any]],
             levels: Sequence[float]) -> list[str]:
    """Bảng cờ QC và mức phân tán theo endpoint (đối chiếu Nelen et al. 2025)."""
    rows = []
    for endpoint, stats in per_endpoint.items():
        qc = stats.get("qc", {})
        spread = stats.get("dispersion", {})
        measured = spread.get("measured", 0)
        cells = [endpoint, fmt_int(qc.get(aggregate.QC_PASS, 0)),
                 fmt_int(qc.get(aggregate.QC_SINGLE, 0)),
                 fmt_int(qc.get(aggregate.QC_QUARANTINE, 0)), fmt_int(measured)]
        for level in levels:
            cells.append(fmt_pct(spread.get(f"le_{level}", 0), measured))
            cells.append(fmt_pct(spread.get(f"gt_{level}", 0), measured))
        rows.append(cells)
    headers = ["endpoint", "pass", "single_assay", "quarantine", "đo được phân tán"]
    for level in levels:
        headers += [f"≤ {fmt_float(level, 1)}", f"> {fmt_float(level, 1)}"]
    return ["Cờ QC và phân tán của Δp (tỷ lệ tính trên số bản ghi đo được phân tán):",
            *format_table(headers, rows)]


# --------------------------------------------------------------------------- chạy chính


def run_pipeline(config: Mapping[str, Any], root: Path, out_root: Path,
                 modulus: int | None) -> dict[str, Any]:
    """Chạy bước 1 → 4 vào một thư mục output (dùng cho cả smoke và full).

    Nhận: config; thư mục gốc dự án; thư mục output; modulus lấy mẫu smoke (None = full).
    Trả về: manifest (tham số, thời gian từng chặng, số đếm) và đã ghi nó ra file.
    """
    settings = config["stage2b"]
    layout = Layout(out_root, settings["files"])
    endpoints = list(config["endpoints"])
    buckets = settings["delta"]["protein_buckets"]
    out_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    collected = collect_rows(config, root, layout, modulus)
    collect_seconds = time.perf_counter() - started
    log_lines(funnel_lines(collected))

    molecules = pq.read_table(layout.file("molecules"), columns=["molecule_key", "smiles"])
    LOGGER.info("bảng phân tử: %s khóa; bảng protein: %s chuỗi",
                fmt_int(collected["molecules"]), fmt_int(collected["proteins"]))

    pairs_started = time.perf_counter()
    workers = settings["mmp"]["workers"]
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=mmp.init_worker,
                                   initargs=(fragment_settings(settings["mmp"]),
                                             settings["mmp"]["fragment_cache_size"]))
    per_endpoint: dict[str, Any] = {}
    try:
        for endpoint in endpoints:
            parts = [process_bucket(layout, endpoint, bucket, molecules, config, pool)
                     for bucket in range(buckets)]
            per_endpoint[endpoint] = merge_stats([part for part in parts if part])
    finally:
        if pool is not None:
            pool.shutdown()
    pairs_seconds = time.perf_counter() - pairs_started

    log_lines(endpoint_lines(per_endpoint))
    log_lines(qc_lines(per_endpoint, settings["qc"]["report_levels"]))
    manifest = {
        "stage": "stage2b_pairs",
        "mode": "smoke" if modulus else "full",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_root),
        "smoke_assay_modulus": modulus,
        "inputs": {source: {"path": str(root / path),
                            "rows": collected["source_rows"].get(source)}
                   for source, path in settings["inputs"].items()},
        "parameters": {key: settings[key] for key in
                       ("filter", "replicate", "identity", "mmp", "delta", "qc")},
        "collected": collected,
        "per_endpoint": per_endpoint,
        "totals": merge_stats(list(per_endpoint.values())),
        "seconds": {"gom_dong": round(collect_seconds, 1),
                    "buoc_1_den_4": round(pairs_seconds, 1),
                    "tong": round(time.perf_counter() - started, 1)},
    }
    manifest_path = layout.file("full_manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    LOGGER.info("manifest: %s", manifest_path)
    return manifest


def run_smoke(config: Mapping[str, Any], root: Path, stamp: str) -> dict[str, Any]:
    """Chạy thử trên một phần assay, ghi vào runs/dataset/smoke_<thời điểm>/.

    Ước lượng thời gian của full = thời gian smoke × modulus; đây là cận dưới thô, vì
    lượt gom vẫn phải đọc toàn bộ dữ liệu và chi phí tìm cặp tăng nhanh hơn số dòng khi
    assay lớn hơn.
    """
    settings = config["stage2b"]
    modulus = settings["run"]["smoke_assay_modulus"]
    out_root = root / settings["paths"]["runs_dir"] / f"smoke_{stamp}"
    manifest = run_pipeline(config, root, out_root, modulus)
    seconds = manifest["seconds"]
    log_lines(["Ước lượng cho full (thô, cận dưới):",
               *format_table(["chặng", "smoke (giây)", "ước lượng full (giờ)"],
                             [["gom dòng", fmt_float(seconds["gom_dong"], 0),
                               fmt_float(seconds["gom_dong"] / 3600, 2)],
                              ["bước 1 → 4", fmt_float(seconds["buoc_1_den_4"], 0),
                               fmt_float(seconds["buoc_1_den_4"] * modulus / 3600, 2)]])])
    return manifest


def run_full(config: Mapping[str, Any], root: Path, force: bool) -> dict[str, Any]:
    """Chạy toàn bộ vào data/dataset/full/; từ chối nếu đã có output trừ khi --force.

    Ghi vào thư mục tạm '<full_dir>.partial' rồi mới đổi tên, nên một lần chạy bị ngắt
    không để lại thư mục trông như đã hoàn chỉnh.
    """
    settings = config["stage2b"]
    full_dir = root / settings["paths"]["full_dir"]
    if full_dir.exists() and not force:
        raise FileExistsError(f"{full_dir} đã có; dùng --force để chạy lại từ đầu")
    partial = full_dir.with_name(full_dir.name + PARTIAL_SUFFIX)
    if partial.exists():
        shutil.rmtree(partial)
    manifest = run_pipeline(config, root, partial, None)
    if full_dir.exists():
        shutil.rmtree(full_dir)
    partial.rename(full_dir)
    manifest["output_dir"] = str(full_dir)
    (full_dir / settings["files"]["full_manifest"]).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("output: %s", full_dir)
    return manifest


# --------------------------------------------------------------------------- dựng split


def dataset_pair_counts(layout: Layout, endpoints: Sequence[str]) -> Counter:
    """Số cặp (bản ghi bước 4) của từng protein, cộng trên mọi endpoint và bucket."""
    columns = []
    for endpoint in endpoints:
        for path in layout.bucket_files("records_dir", endpoint):
            columns.append(pq.read_table(path, columns=["protein_hash"]).column("protein_hash"))
    return split.protein_pair_counts(columns)


def load_cluster_tsv(path: Path) -> dict[str, str]:
    """Đọc kết quả gom chuỗi dạng TSV hai cột: <đại diện>\\t<thành viên>.

    Đây là định dạng trung lập với công cụ (MMseqs2 createtsv xuất đúng dạng này; kết
    quả của công cụ khác phải đổi sang dạng này trước). Hàm không chạy công cụ nào.
    Trả về: {protein: tên nhóm}. Dòng thiếu cột hoặc protein xuất hiện ở hai nhóm là lỗi.
    """
    membership: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            parts = text.split("\t")
            if len(parts) < 2:
                raise ValueError(f"{path}:{number}: cần hai cột cách nhau bằng tab, thấy "
                                 f"{len(parts)}")
            representative, member = parts[0], parts[1]
            if member in membership and membership[member] != representative:
                raise ValueError(f"{path}:{number}: protein {member} nằm ở hai nhóm "
                                 f"({membership[member]} và {representative})")
            membership[member] = representative
    return membership


def run_split(config: Mapping[str, Any], root: Path, clusters: Path | None,
              tool: str | None, tool_version: str | None) -> dict[str, Any]:
    """Dựng split protein từ dataset đã có (bước 4), 80/10/10 theo số cặp.

    Nhận: config; thư mục gốc; file TSV kết quả gom chuỗi (None = chỉ ghi FASTA đầu vào);
    tên và phiên bản công cụ gom (ghi vào manifest, không tự suy ra).
    Trả về: {"fasta", "proteins", "pairs", "clustered", ...}.
    Luôn ghi FASTA của đúng những protein có mặt trong dataset cuối. Khi chưa có kết quả
    gom chuỗi thì dừng ở đó: việc gom chuỗi ở ngưỡng identity là việc của công cụ chuẩn
    (MMseqs2 / CD-HIT), không tự cài thuật toán alignment trong dự án này.
    """
    settings = config["stage2b"]
    full, dataset = layouts(config, root)
    endpoints = list(config["endpoints"])
    counts = dataset_pair_counts(full, endpoints)
    if not counts:
        raise FileNotFoundError(f"chưa có bản ghi bước 4 trong {full.root}; chạy --mode full")
    proteins = pq.read_table(full.file("proteins"),
                             columns=["protein_hash", "protein_sequence"])
    sequences = dict(zip(proteins.column("protein_hash").to_pylist(),
                         proteins.column("protein_sequence").to_pylist()))
    dataset.root.mkdir(parents=True, exist_ok=True)
    fasta = dataset.file("split_fasta")
    written = split.write_fasta(sequences, counts, fasta)
    digest = sha256_file(fasta)
    LOGGER.info("FASTA đầu vào của split: %s (%s chuỗi, %s cặp)", fasta, fmt_int(written),
                fmt_int(sum(counts.values())))
    LOGGER.info("SHA-256 FASTA: %s", digest)
    result: dict[str, Any] = {"fasta": str(fasta), "fasta_sha256": digest,
                              "proteins": written, "pairs": sum(counts.values()),
                              "clustered": False}
    if clusters is None:
        LOGGER.warning("chưa có kết quả gom chuỗi: truyền --clusters <file TSV hai cột "
                       "'đại diện<tab>thành viên'> cùng --cluster-tool và "
                       "--cluster-tool-version để dựng split")
        return result

    clustering = settings["split"]["clustering"]
    membership = load_cluster_tsv(clusters)
    table = split.build_split_table(counts, membership, settings["split"]["target_ratios"],
                                    settings["split"]["partitions"], settings["split"]["seed"])
    write_parquet(table, dataset.file("protein_split"))
    checks = split_checks(config, root, table, counts)
    manifest = {
        "stage": "stage2b_protein_split",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "built_from": {"records_dir": str(full.root / settings["files"]["records_dir"]),
                       "endpoints": endpoints},
        "input_fasta": {"path": str(fasta), "sha256": digest, "sequences": written},
        "clusters_file": {"path": str(clusters), "sha256": sha256_file(clusters)},
        "tool": tool, "tool_version": tool_version,
        "clustering": dict(clustering),
        "seed": settings["split"]["seed"],
        "target_ratios": settings["split"]["target_ratios"],
        "clusters": int(pc.count_distinct(table.column("protein_cluster_id")).as_py()),
        "proteins": table.num_rows,
        "pairs": sum(counts.values()),
        "pairs_by_partition": checks["pair_totals"]["pairs_by_partition"],
        "ratios": checks["ratios"],
        "checks": checks,
    }
    path = dataset.file("protein_split_manifest")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log_lines(split_lines(manifest))
    result.update({"clustered": True, "split_file": str(dataset.file("protein_split")),
                   "manifest": str(path), "clusters": manifest["clusters"]})
    return result


def split_checks(config: Mapping[str, Any], root: Path, table: pa.Table,
                 counts: Mapping[str, int]) -> dict[str, Any]:
    """Năm phép kiểm tra của split (mục 5 chỉ đối chiếu, không ảnh hưởng kết quả)."""
    settings = config["stage2b"]["split"]
    totals = split.check_pair_totals(table, counts)
    checks = {
        "clusters_single_partition": split.check_clusters_single_partition(table),
        "proteins_single_partition": split.check_proteins_single_partition(table),
        "pair_totals": totals,
        "ratios": split.ratio_report(totals["pairs_by_partition"], settings["target_ratios"]),
    }
    old_path = Path(settings["old_split_file"])
    if old_path.is_file():
        old = pq.read_table(old_path, columns=["target_sequence_hash", "partition"])
        checks["old_split"] = split.compare_old_split(table, old)
        checks["old_split"]["file"] = str(old_path)
    else:
        checks["old_split"] = {"file": str(old_path), "available": False}
    return checks


def split_lines(manifest: Mapping[str, Any]) -> list[str]:
    """Báo cáo split: số cặp và tỷ lệ theo partition, cộng đối chiếu với split cũ."""
    rows = []
    for partition, ratio in manifest["ratios"].items():
        pairs = manifest["pairs_by_partition"].get(partition, 0)
        rows.append([partition, fmt_int(pairs), fmt_pct(ratio["target"], 1.0),
                     fmt_pct(ratio["actual"], 1.0),
                     fmt_float(100.0 * ratio["deviation"], 2) + " điểm %"])
    lines = [f"Split protein: {fmt_int(manifest['proteins'])} protein, "
             f"{fmt_int(manifest['clusters'])} nhóm, {fmt_int(manifest['pairs'])} cặp.",
             *format_table(["partition", "cặp", "mục tiêu", "thực tế", "lệch"], rows)]
    old = manifest["checks"].get("old_split", {})
    if old.get("proteins_shared") is not None:
        lines.append(f"Đối chiếu split cũ: {fmt_int(old['proteins_shared'])} protein có ở cả "
                     f"hai, {fmt_pct(old['same_partition'], old['proteins_shared'])} cùng "
                     f"partition.")
    return lines


# --------------------------------------------------------------------------- bước 5


FLAG_COLUMNS = ["unseen_pair", "plausible_range", "short_protein"]
EVALUATOR_COLUMNS = ["pair_id", "endpoint", "protein_hash", "partition", "smiles_a", "smiles_b",
                     "constant_smiles", "variable_a", "variable_b", "delta_p_mean", "n_assays",
                     "qc_status", *FLAG_COLUMNS]
GENERATOR_COLUMNS = ["pair_id", "endpoint", "protein_hash", "partition", "smiles_weak",
                     "smiles_strong", "constant_smiles", "variable_weak", "variable_strong",
                     "gain", "n_assays", "qc_status", *FLAG_COLUMNS]
PAIR_CODE_SHIFT = 31


def pair_codes(table: pa.Table, codes: MutableMapping[str, int] | Mapping[str, int],
               grow: bool) -> np.ndarray:
    """Mã int64 cho cặp phân tử (A, B), để so cặp giữa các partition mà không giữ chuỗi.

    Nhận: bảng có molecule_a_key/molecule_b_key; từ điển khóa -> mã (dùng chung mọi bucket);
    grow = True thì thêm khóa mới vào từ điển, False thì khóa lạ trả về mã -1.
    Trả về: mảng int64 = (mã A << PAIR_CODE_SHIFT) | mã B, hoặc -1 nếu một phía là khóa lạ.
    Gói hai mã vào một int64 thay vì ghép chuỗi vì tập cặp lên tới hàng triệu phần tử.
    """
    out = np.empty(table.num_rows, dtype=np.int64)
    a_keys = table.column("molecule_a_key").to_pylist()
    b_keys = table.column("molecule_b_key").to_pylist()
    for index, (a, b) in enumerate(zip(a_keys, b_keys)):
        if grow:
            code_a = codes.setdefault(a, len(codes))
            code_b = codes.setdefault(b, len(codes))
        else:
            code_a = codes.get(a, -1)
            code_b = codes.get(b, -1)
        out[index] = -1 if code_a < 0 or code_b < 0 else (code_a << PAIR_CODE_SHIFT) | code_b
    if grow and len(codes) >= 1 << PAIR_CODE_SHIFT:
        raise ValueError(f"quá nhiều phân tử để gói mã cặp vào int64: {len(codes)}")
    return out


def train_pair_codes(layout: Layout, endpoints: Sequence[str],
                     assignment: pa.Table) -> tuple[dict[str, int], np.ndarray]:
    """Tập mã cặp (A, B) có mặt trong partition train, dùng để gắn cờ unseen_pair.

    Phải quét trước một lượt vì cùng một cặp phân tử có thể nằm ở bucket protein khác.
    Trả về: (từ điển khóa phân tử -> mã, mảng mã cặp của train đã sắp xếp, không trùng).
    """
    codes: dict[str, int] = {}
    chunks: list[np.ndarray] = []
    for endpoint in endpoints:
        for path in layout.bucket_files("records_dir", endpoint):
            table = pq.read_table(path, columns=["protein_hash", "molecule_a_key",
                                                 "molecule_b_key"])
            partition = partition_column(table, assignment)
            train = table.filter(pc.equal(partition, "train"))
            if train.num_rows:
                chunks.append(pair_codes(train, codes, grow=True))
    found = np.unique(np.concatenate(chunks)) if chunks else np.zeros(0, dtype=np.int64)
    LOGGER.info("cặp (A, B) phân biệt trong train: %s", fmt_int(len(found)))
    return codes, found


def unseen_pair_column(records: pa.Table, codes: Mapping[str, int],
                       train_codes: np.ndarray) -> pa.Array:
    """Cờ unseen_pair: cặp (A, B) của bản ghi không xuất hiện trong partition train.

    Dòng của train luôn False. Phân tử chưa từng gặp trong train có mã lạ nên cặp chắc chắn
    không thuộc train. Cờ chỉ mô tả, không lọc gì; dùng để định nghĩa tập đánh giá nghiêm
    ngặt "cặp chưa từng thấy" cho validation/test.
    """
    pairs = pair_codes(records, codes, grow=False)
    return pa.array(~np.isin(pairs, train_codes), pa.bool_())


def value_bounds(path: Path, pair_ids: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """Biên pActivity nhỏ nhất/lớn nhất trong mọi phép đo đóng góp cho từng bản ghi.

    Nhận: file Δp assay của đúng (endpoint, bucket) đó; pair_id của các bản ghi, đúng thứ tự.
    Trả về: hai mảng float64 (min, max) theo thứ tự bản ghi.
    """
    deltas = pq.read_table(path, columns=["pair_id", "p_a", "p_b"])
    table = pa.table({
        "pair_id": deltas.column("pair_id"),
        "lo": pc.min_element_wise(deltas.column("p_a"), deltas.column("p_b")),
        "hi": pc.max_element_wise(deltas.column("p_a"), deltas.column("p_b")),
    })
    grouped = table.group_by("pair_id").aggregate([("lo", "min"), ("hi", "max")])
    position = pc.index_in(pair_ids, value_set=grouped.column("pair_id").combine_chunks())
    if pc.sum(pc.is_null(position)).as_py():
        raise ValueError(f"{path}: có bản ghi không tìm thấy Δp assay nào theo pair_id")
    lo = pc.take(grouped.column("lo_min"), position)
    hi = pc.take(grouped.column("hi_max"), position)
    return (np.asarray(lo.to_numpy(zero_copy_only=False), dtype=np.float64),
            np.asarray(hi.to_numpy(zero_copy_only=False), dtype=np.float64))


def partition_column(records: pa.Table, assignment: pa.Table) -> pa.Array:
    """Partition của từng bản ghi, tra theo protein; protein thiếu trong split là lỗi."""
    keys = assignment.column("protein_sequence_hash").combine_chunks()
    values = assignment.column("partition").combine_chunks()
    position = pc.index_in(records.column("protein_hash"), value_set=keys)
    if pc.sum(pc.is_null(position)).as_py():
        missing = records.filter(pc.is_null(position)).column("protein_hash").to_pylist()[:3]
        raise ValueError(f"protein không có trong protein_split.parquet: {missing}; "
                         f"dựng lại split sau khi dataset đổi")
    return pc.take(values, position)


def oriented_generator_rows(records: pa.Table, min_gain: float) -> pa.Table:
    """Bản ghi hướng yếu -> mạnh với gain > min_gain (nghiêm ngặt).

    Δp mang dấu theo chiều A -> B. Δp > 0 nghĩa là B mạnh hơn, nên giữ nguyên; Δp < 0 thì
    đảo hai phía. gain = |Δp trung bình|.
    """
    mean = np.asarray(records.column("delta_p_mean").to_numpy(zero_copy_only=False),
                      dtype=np.float64)
    keep = np.abs(mean) > min_gain
    subset = records.filter(pa.array(keep, pa.bool_()))
    forward = pa.array(np.asarray(subset.column("delta_p_mean").to_numpy(
        zero_copy_only=False), dtype=np.float64) > 0, pa.bool_())
    columns = {
        "pair_id": subset.column("pair_id"), "endpoint": subset.column("endpoint"),
        "protein_hash": subset.column("protein_hash"), "partition": subset.column("partition"),
        "smiles_weak": pc.if_else(forward, subset.column("smiles_a"), subset.column("smiles_b")),
        "smiles_strong": pc.if_else(forward, subset.column("smiles_b"),
                                    subset.column("smiles_a")),
        "constant_smiles": subset.column("constant_smiles"),
        "variable_weak": pc.if_else(forward, subset.column("variable_a"),
                                    subset.column("variable_b")),
        "variable_strong": pc.if_else(forward, subset.column("variable_b"),
                                      subset.column("variable_a")),
        "gain": pc.abs(subset.column("delta_p_mean")),
        "n_assays": subset.column("n_assays"), "qc_status": subset.column("qc_status"),
        **{flag: subset.column(flag) for flag in FLAG_COLUMNS},
    }
    return pa.table(columns).select(GENERATOR_COLUMNS)


def run_export(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Bước 5: hai file training (evaluator, generator) và một file evidence.

    Phần training chỉ gồm bản ghi đủ điều kiện (qc_status thuộc export.eligible_qc); phần
    evidence giữ MỌI bản ghi, kể cả quarantine, nối với training bằng pair_id. Bộ lọc chỉ
    quyết định tính đủ điều kiện, không xóa evidence.
    Split được đọc từ protein_split.parquet; hàm này không tự dựng split.
    """
    settings = config["stage2b"]
    full, dataset = layouts(config, root)
    split_path = dataset.file("protein_split")
    if not split_path.is_file():
        raise FileNotFoundError(f"chưa có {split_path}; chạy --mode split trước")
    assignment = pq.read_table(split_path, columns=["protein_sequence_hash", "partition"])
    eligible = list(settings["export"]["eligible_qc"])
    min_gain = settings["export"]["generator_min_gain"]
    low, high = settings["export"]["plausible_p_range"]
    min_length = settings["export"]["min_protein_length"]
    proteins = pq.read_table(full.file("proteins"),
                             columns=["protein_hash", "sequence_length"])
    lengths = dict(zip(proteins.column("protein_hash").to_pylist(),
                       proteins.column("sequence_length").to_pylist()))
    endpoints = list(config["endpoints"])
    codes, train_codes = train_pair_codes(full, endpoints, assignment)
    started = time.perf_counter()

    writers: dict[str, pq.ParquetWriter] = {}
    counts: Counter = Counter()
    per_partition: dict[str, Counter] = {}
    try:
        for endpoint in endpoints:
            for path in full.bucket_files("records_dir", endpoint):
                records = pq.read_table(path)
                records = records.append_column("partition",
                                                partition_column(records, assignment))
                records = records.append_column("unseen_pair", unseen_pair_column(
                    records, codes, train_codes))
                deltas = full.bucket_file("assay_deltas_dir", endpoint, int(path.stem[1:]))
                lo, hi = value_bounds(deltas, records.column("pair_id"))
                records = records.append_column(
                    "plausible_range", pa.array((lo >= low) & (hi <= high), pa.bool_()))
                records = records.append_column("short_protein", pa.array(
                    [lengths[h] < min_length
                     for h in records.column("protein_hash").to_pylist()], pa.bool_()))
                for flag in FLAG_COLUMNS:
                    counts[f"{flag}_true"] += int(pc.sum(records.column(flag)).as_py() or 0)
                evidence = records
                training = records.filter(pc.is_in(records.column("qc_status"),
                                                   value_set=pa.array(eligible, pa.string())))
                evaluator = training.select(EVALUATOR_COLUMNS)
                generator = oriented_generator_rows(training, min_gain)
                for key, table in (("evidence_records", evidence), ("evaluator", evaluator),
                                   ("generator", generator)):
                    if key not in writers:
                        target = dataset.file(key)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        writers[key] = pq.ParquetWriter(target, table.schema)
                    writers[key].write_table(table)
                    counts[key] += table.num_rows
                for partition, number in Counter(
                        evaluator.column("partition").to_pylist()).items():
                    per_partition.setdefault(partition, Counter())[endpoint] += number
                counts[f"evidence_{endpoint}"] += evidence.num_rows
    finally:
        for handle in writers.values():
            handle.close()

    manifest = {
        "stage": "stage2b_export",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "records_dir": str(full.root / settings["files"]["records_dir"]),
        "split_file": str(split_path),
        "parameters": settings["export"],
        "rows": dict(counts),
        "flags": {flag: counts[f"{flag}_true"] for flag in FLAG_COLUMNS},
        "flag_note": ("Cờ chỉ mô tả, không lọc gì. unseen_pair: cặp (A, B) không xuất hiện "
                      "trong partition train (dòng train luôn False). plausible_range: mọi "
                      f"phép đo đóng góp có pActivity trong [{low}, {high}]. short_protein: "
                      f"chuỗi protein ngắn hơn {min_length} acid amin."),
        "evaluator_by_partition": {partition: dict(endpoints)
                                   for partition, endpoints in sorted(per_partition.items())},
        "seconds": round(time.perf_counter() - started, 1),
    }
    path = dataset.file("export_manifest")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [[key, fmt_int(counts[key])] for key in ("evidence_records", "evaluator", "generator")]
    flags = [[flag, fmt_int(counts[f"{flag}_true"]),
              fmt_pct(counts[f"{flag}_true"], counts["evidence_records"])]
             for flag in FLAG_COLUMNS]
    log_lines(["Bước 5 — xuất dataset:", *format_table(["file", "dòng"], rows), "",
               "Cờ mô tả trên evidence (không lọc gì):",
               *format_table(["cờ", "số dòng True", "tỷ lệ"], flags)])
    LOGGER.info("manifest: %s", path)
    return manifest


# --------------------------------------------------------------------------- verify


def check_conservation(layout: Layout, endpoints: Sequence[str]) -> dict[str, Any]:
    """Bảo toàn số cặp: mỗi Δp assay thuộc đúng một bản ghi, và ngược lại.

    So, cho từng (endpoint, bucket): số dòng Δp assay = tổng n_assays của các bản ghi;
    số bản ghi = số pair_id khác nhau trong Δp assay.
    """
    details, failures = {}, []
    for endpoint in endpoints:
        deltas_rows = records_rows = assay_sum = distinct = 0
        for path in layout.bucket_files("assay_deltas_dir", endpoint):
            bucket = path.stem
            deltas = pq.read_table(path, columns=["pair_id"])
            record_path = layout.bucket_file("records_dir", endpoint, int(bucket[1:]))
            records = pq.read_table(record_path, columns=["pair_id", "n_assays"])
            rows = deltas.num_rows
            unique = int(pc.count_distinct(deltas.column("pair_id")).as_py() or 0)
            total = int(pc.sum(records.column("n_assays")).as_py() or 0)
            if rows != total or unique != records.num_rows:
                failures.append({"endpoint": endpoint, "bucket": bucket,
                                 "assay_deltas": rows, "sum_n_assays": total,
                                 "distinct_pairs": unique, "records": records.num_rows})
            deltas_rows += rows
            records_rows += records.num_rows
            assay_sum += total
            distinct += unique
        details[endpoint] = {"assay_deltas": deltas_rows, "records": records_rows,
                             "sum_n_assays": assay_sum, "distinct_pairs": distinct}
    return {"passed": not failures, "per_endpoint": details, "failures": failures[:20]}


def list_offsets(column: pa.ChunkedArray) -> tuple[np.ndarray, pa.Array]:
    """(Ranh giới bắt đầu từ 0, mảng phẳng đúng phần được tham chiếu) của cột danh sách."""
    flat = column.combine_chunks()
    offsets = np.asarray(flat.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    if not offsets.size:
        return offsets, flat.values
    base, end = int(offsets[0]), int(offsets[-1])
    return offsets - base, flat.values.slice(base, end - base)


def check_record_stats(layout: Layout, config: Mapping[str, Any],
                       endpoints: Sequence[str]) -> dict[str, Any]:
    """Tính lại thống kê của bản ghi từ danh sách Δp lưu trong chính bản ghi.

    Kiểm tra mean, median, STD, min, max, phân tán và cờ QC của MỌI bản ghi, không lấy
    mẫu: đây là phép kiểm tra rẻ (chỉ đọc lại phần evidence đã lưu).
    """
    settings = config["stage2b"]
    tolerance = settings["run"]["verify_tolerance"]
    checked, mismatched, examples = 0, 0, []
    for endpoint in endpoints:
        threshold = settings["qc"]["dispersion_thresholds"][endpoint]
        for path in layout.bucket_files("records_dir", endpoint):
            table = pq.read_table(path, columns=[
                "pair_id", "n_assays", "delta_p_mean", "delta_p_median", "delta_p_std",
                "delta_p_min", "delta_p_max", "dispersion", "qc_status", "delta_p_values"])
            offsets, flat = list_offsets(table.column("delta_p_values"))
            values = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.float64)
            starts, sizes = offsets[:-1], np.diff(offsets)
            stored_sizes = np.asarray(table.column("n_assays").to_numpy(zero_copy_only=False),
                                      dtype=np.int64)
            mean = np.add.reduceat(values, starts) / sizes if sizes.size else np.empty(0)
            minimum = np.minimum.reduceat(values, starts) if sizes.size else np.empty(0)
            maximum = np.maximum.reduceat(values, starts) if sizes.size else np.empty(0)
            squares = (np.add.reduceat((values - np.repeat(mean, sizes)) ** 2, starts)
                       if sizes.size else np.empty(0))
            ddof = settings["qc"]["dispersion_ddof"]
            with np.errstate(invalid="ignore", divide="ignore"):
                std = np.where(sizes > ddof, np.sqrt(squares / np.maximum(sizes - ddof, 1)),
                               np.nan)
            spread = aggregate.dispersion(sizes, std, minimum, maximum)
            status = aggregate.qc_status(sizes, spread, threshold)
            median = aggregate.group_median(values, offsets) if sizes.size else np.empty(0)
            stored = {name: np.asarray(table.column(name).to_numpy(zero_copy_only=False),
                                       dtype=np.float64)
                      for name in ("delta_p_mean", "delta_p_median", "delta_p_std",
                                   "delta_p_min", "delta_p_max", "dispersion")}
            bad = (sizes != stored_sizes)
            for name, computed in (("delta_p_mean", mean), ("delta_p_median", median),
                                   ("delta_p_std", std), ("delta_p_min", minimum),
                                   ("delta_p_max", maximum), ("dispersion", spread)):
                bad |= ~close_or_both_nan(computed, stored[name], tolerance)
            bad |= status != np.asarray(table.column("qc_status").to_pylist(), dtype=object)
            checked += table.num_rows
            mismatched += int(bad.sum())
            for index in np.flatnonzero(bad)[:3]:
                examples.append({"file": path.name,
                                 "pair_id": table.column("pair_id")[int(index)].as_py()})
    return {"passed": mismatched == 0, "records_checked": checked,
            "mismatched": mismatched, "examples": examples[:20]}


def close_or_both_nan(left: np.ndarray, right: np.ndarray, tolerance: float) -> np.ndarray:
    """So hai mảng số thực: bằng nhau trong sai số, hoặc cùng là NaN (cùng 'không có')."""
    both_nan = np.isnan(left) & np.isnan(right)
    with np.errstate(invalid="ignore"):
        close = np.abs(left - right) <= tolerance
    return both_nan | (close & ~np.isnan(left) & ~np.isnan(right))


def check_keys(layout: Layout, config: Mapping[str, Any],
               endpoints: Sequence[str]) -> dict[str, Any]:
    """Không bản ghi nào trộn endpoint hay protein, và pair_id đúng công thức.

    Với từng (endpoint, bucket): cột endpoint đúng endpoint của thư mục; protein thuộc
    đúng bucket; khóa A < khóa B; pair_id tính lại từ (endpoint, protein, A, B) khớp giá
    trị đã lưu. Làm cho cả bảng Δp assay và bảng bản ghi.
    """
    settings = config["stage2b"]
    buckets = settings["delta"]["protein_buckets"]
    length = settings["delta"]["pair_id_length"]
    problems: list[dict[str, Any]] = []
    checked = 0
    for endpoint in endpoints:
        for key in ("assay_deltas_dir", "records_dir"):
            for path in layout.bucket_files(key, endpoint):
                bucket = int(path.stem[1:])
                table = pq.read_table(path, columns=["pair_id", "endpoint", "protein_hash",
                                                     "molecule_a_key", "molecule_b_key"])
                checked += table.num_rows
                wrong_endpoint = int(pc.sum(pc.not_equal(
                    table.column("endpoint"), pa.scalar(endpoint))).as_py() or 0)
                wrong_bucket = int((bucket_of(table.column("protein_hash"), buckets)
                                    != bucket).sum())
                keys_a = table.column("molecule_a_key").to_pylist()
                keys_b = table.column("molecule_b_key").to_pylist()
                proteins = table.column("protein_hash").to_pylist()
                stored = table.column("pair_id").to_pylist()
                wrong_order = sum(1 for a, b in zip(keys_a, keys_b) if a >= b)
                wrong_id = sum(1 for index, value in enumerate(stored)
                               if value != delta.pair_id(endpoint, proteins[index],
                                                         keys_a[index], keys_b[index], length))
                if wrong_endpoint or wrong_bucket or wrong_order or wrong_id:
                    problems.append({"file": f"{key}/{endpoint}/{path.name}",
                                     "wrong_endpoint": wrong_endpoint,
                                     "wrong_bucket": wrong_bucket,
                                     "wrong_order": wrong_order, "wrong_pair_id": wrong_id})
    return {"passed": not problems, "rows_checked": checked, "problems": problems[:20]}


def check_one_delta_per_assay(layout: Layout, endpoints: Sequence[str]) -> dict[str, Any]:
    """Mỗi assay góp đúng một Δp cho một cặp.

    Hai phía: bảng Δp assay không có (pair_id, assay_key) lặp; danh sách assay_keys của
    bản ghi không lặp và dài đúng n_assays.
    """
    problems: list[dict[str, Any]] = []
    pairs_checked = 0
    for endpoint in endpoints:
        for path in layout.bucket_files("assay_deltas_dir", endpoint):
            table = pq.read_table(path, columns=["pair_id", "assay_key"])
            grouped = table.group_by(["pair_id", "assay_key"]).aggregate([([], "count_all")])
            repeated = int(pc.sum(pc.greater(grouped.column("count_all"), 1)).as_py() or 0)
            if repeated:
                problems.append({"file": f"{endpoint}/{path.name}",
                                 "repeated_pair_assay": repeated})
        for path in layout.bucket_files("records_dir", endpoint):
            table = pq.read_table(path, columns=["pair_id", "n_assays", "assay_keys"])
            offsets, flat = list_offsets(table.column("assay_keys"))
            sizes = np.diff(offsets)
            stored = np.asarray(table.column("n_assays").to_numpy(zero_copy_only=False),
                                dtype=np.int64)
            wrong_length = int((sizes != stored).sum())
            keys = flat.to_pylist()
            duplicated = sum(1 for index in np.flatnonzero(sizes > 1)
                             if len(set(keys[offsets[index]:offsets[index + 1]]))
                             != int(sizes[index]))
            pairs_checked += table.num_rows
            if wrong_length or duplicated:
                problems.append({"file": f"{endpoint}/{path.name}",
                                 "wrong_assay_list_length": wrong_length,
                                 "duplicated_assay_in_record": duplicated})
    return {"passed": not problems, "records_checked": pairs_checked, "problems": problems[:20]}


def sample_assay_deltas(layout: Layout, endpoints: Sequence[str], sample_rows: int,
                        seed: int) -> pa.Table:
    """Lấy mẫu ngẫu nhiên (seed cố định) các dòng Δp assay trên mọi endpoint và bucket."""
    paths = [path for endpoint in endpoints
             for path in layout.bucket_files("assay_deltas_dir", endpoint)]
    sizes = np.array([pq.ParquetFile(path).metadata.num_rows for path in paths], dtype=np.int64)
    total = int(sizes.sum())
    if not total:
        return delta.ASSAY_DELTA_SCHEMA.empty_table()
    generator = np.random.default_rng(seed)
    chosen = np.sort(generator.choice(total, size=min(sample_rows, total), replace=False))
    edges = np.concatenate([[0], np.cumsum(sizes)])
    parts = []
    for index, path in enumerate(paths):
        local = chosen[(chosen >= edges[index]) & (chosen < edges[index + 1])] - edges[index]
        if local.size:
            parts.append(pq.read_table(path).take(pa.array(local, type=pa.int64())))
    return pa.concat_tables(parts) if parts else delta.ASSAY_DELTA_SCHEMA.empty_table()


def sampled_group_keys(sample: pa.Table) -> dict[tuple[str, str, str, str], int]:
    """{(endpoint, assay_key, protein, khóa phân tử): số lần xuất hiện} của hai phía mẫu."""
    keys: dict[tuple[str, str, str, str], int] = {}
    for side in ("molecule_a_key", "molecule_b_key"):
        for endpoint, assay, protein, molecule in zip(
                sample.column("endpoint").to_pylist(), sample.column("assay_key").to_pylist(),
                sample.column("protein_hash").to_pylist(), sample.column(side).to_pylist()):
            keys[(endpoint, assay, protein, molecule)] = keys.get(
                (endpoint, assay, protein, molecule), 0) + 1
    return keys


def check_against_observations(config: Mapping[str, Any], root: Path, layout: Layout,
                               endpoints: Sequence[str]) -> dict[str, Any]:
    """Tính lại Δp của một mẫu dòng, đi ngược về observations gốc.

    Cách làm: lấy mẫu dòng Δp assay; lấy khóa nhóm của hai phía; quét lại hai bản sao
    dedup với ĐÚNG bộ lọc của pipeline; gộp replicate lại từ đầu; so p(A), p(B), Δp,
    số dòng gốc và danh sách observation_id với giá trị đã lưu.
    Phép kiểm tra này độc lập với file trung gian: nó chỉ dùng dữ liệu đầu vào.
    """
    settings = config["stage2b"]
    run = settings["run"]
    sample = sample_assay_deltas(layout, endpoints, run["verify_sample_rows"],
                                 run["verify_seed"])
    if not sample.num_rows:
        return {"passed": True, "sampled_rows": 0, "note": "chưa có dòng Δp assay nào"}
    wanted = sampled_group_keys(sample)
    joined = pa.array(["|".join(key) for key in wanted], pa.string())
    columns = settings["columns"]
    names = [columns[role] for role in LOAD_ROLES]
    pattern = config["stage2a"]["dedup"]["relative_stereo_pattern"]
    suffix = settings["identity"]["relative_stereo_suffix"]
    parts: list[pa.Table] = []
    for source, relative_path in settings["inputs"].items():
        path = root / relative_path
        parquet = pq.ParquetFile(path)
        seen = 0
        for batch in parquet.iter_batches(batch_size=run["batch_rows"], columns=names):
            table = rename_roles(batch, columns, LOAD_ROLES)
            table = table.append_column("assay_key", assay_keys(table))
            steps = combined_mask(filter_steps(table, config["qualifiers"]["exact"],
                                               settings["filter"]["resolved_status"]))
            kept = table.filter(steps[-1][1])
            if kept.num_rows:
                rows = shard_rows(kept, settings["filter"], pattern, suffix)
                rows = rows.append_column("endpoint", kept.column("endpoint"))
                key = pc.binary_join_element_wise(
                    rows.column("endpoint"), rows.column("assay_key"),
                    rows.column("protein_hash"), rows.column("molecule_key"), "|")
                hit = rows.filter(pc.is_in(key, value_set=joined))
                if hit.num_rows:
                    parts.append(hit)
            seen += batch.num_rows
            LOGGER.info("  đối chiếu %s: %s/%s dòng", source, fmt_int(seen),
                        fmt_int(parquet.metadata.num_rows))
    if not parts:
        return {"passed": False, "sampled_rows": sample.num_rows,
                "note": "không tìm lại được dòng gốc nào của mẫu"}
    found = pa.concat_tables(parts)
    replicate = settings["replicate"]
    rebuilt: dict[tuple[str, str, str, str], tuple[float, int, tuple[str, ...]]] = {}
    for endpoint in endpoints:
        subset = found.filter(pc.equal(found.column("endpoint"), pa.scalar(endpoint)))
        if not subset.num_rows:
            continue
        groups = aggregate.collapse_replicates(subset.drop_columns(["endpoint"]),
                                               replicate["merge_tolerance"],
                                               replicate["float_slack"], replicate["std_ddof"])
        for row in groups.to_pylist():
            rebuilt[(endpoint, row["assay_key"], row["protein_hash"], row["molecule_key"])] = (
                row["p_mean"], row["n_raw"], tuple(sorted(row["observation_ids"])))
    return compare_sample(sample, rebuilt, settings["run"]["verify_tolerance"])


def compare_sample(sample: pa.Table, rebuilt: Mapping[tuple[str, str, str, str],
                                                      tuple[float, int, tuple[str, ...]]],
                   tolerance: float) -> dict[str, Any]:
    """So giá trị đã lưu của mẫu với giá trị tính lại từ observations gốc."""
    mismatched, missing, examples = 0, 0, []
    rows = sample.to_pylist()
    for row in rows:
        sides = []
        for side in ("a", "b"):
            key = (row["endpoint"], row["assay_key"], row["protein_hash"],
                   row[f"molecule_{side}_key"])
            sides.append(rebuilt.get(key))
        if any(side is None for side in sides):
            missing += 1
            examples.append({"pair_id": row["pair_id"], "lý do": "không thấy nhóm gốc"})
            continue
        (p_a, n_a, ids_a), (p_b, n_b, ids_b) = sides
        problems = []
        if abs(p_a - row["p_a"]) > tolerance or abs(p_b - row["p_b"]) > tolerance:
            problems.append("p_a/p_b")
        if abs((p_b - p_a) - row["delta_p"]) > tolerance:
            problems.append("delta_p")
        if ids_a != tuple(sorted(row["observation_ids_a"])) or \
                ids_b != tuple(sorted(row["observation_ids_b"])):
            problems.append("observation_ids")
        if problems:
            mismatched += 1
            examples.append({"pair_id": row["pair_id"], "lý do": ", ".join(problems)})
    return {"passed": mismatched == 0 and missing == 0, "sampled_rows": len(rows),
            "groups_rebuilt": len(rebuilt), "mismatched": mismatched,
            "groups_not_found": missing, "examples": examples[:20]}


def delta_distribution(layout: Layout, endpoints: Sequence[str],
                       percentiles: Sequence[float]) -> dict[str, Any]:
    """Phân bố Δp theo endpoint: phân vị của Δp trung bình mỗi bản ghi và của Δp từng assay."""
    result: dict[str, Any] = {}
    for endpoint in endpoints:
        means = [np.asarray(pq.read_table(path, columns=["delta_p_mean"]).column(
            "delta_p_mean").to_numpy(zero_copy_only=False), dtype=np.float64)
            for path in layout.bucket_files("records_dir", endpoint)]
        per_assay = [np.asarray(pq.read_table(path, columns=["delta_p"]).column(
            "delta_p").to_numpy(zero_copy_only=False), dtype=np.float64)
            for path in layout.bucket_files("assay_deltas_dir", endpoint)]
        entry: dict[str, Any] = {}
        for name, parts in (("records_mean", means), ("per_assay", per_assay)):
            values = np.concatenate(parts) if parts else np.empty(0)
            entry[name] = {"n": int(values.size), "mean": float(values.mean()) if values.size
                           else None,
                           "percentiles": {percentile_label(percentile):
                                           float(np.percentile(values, percentile))
                                           for percentile in percentiles} if values.size else {}}
        result[endpoint] = entry
    return result


def qc_summary(layout: Layout, config: Mapping[str, Any],
               endpoints: Sequence[str]) -> dict[str, Any]:
    """Số bản ghi theo cờ QC và theo mức phân tán, cho từng endpoint."""
    levels = config["stage2b"]["qc"]["report_levels"]
    summary = {}
    for endpoint in endpoints:
        qc: Counter = Counter()
        spread: Counter = Counter()
        for path in layout.bucket_files("records_dir", endpoint):
            table = pq.read_table(path, columns=["qc_status", "dispersion"])
            qc.update(table.column("qc_status").to_pylist())
            spread.update(dispersion_counts(table, levels))
        summary[endpoint] = {"qc": {status: qc.get(status, 0)
                                    for status in aggregate.QC_STATUSES},
                             "dispersion": dict(spread)}
    return summary


def check_export(config: Mapping[str, Any], root: Path, layout: Layout,
                 endpoints: Sequence[str]) -> dict[str, Any]:
    """Phần xuất khớp phần bản ghi: evidence đủ mọi bản ghi, training chỉ bản ghi đủ điều kiện."""
    settings = config["stage2b"]
    _, dataset = layouts(config, root)
    files = {key: dataset.file(key) for key in ("evidence_records", "evaluator", "generator")}
    if not all(path.is_file() for path in files.values()):
        return {"available": False, "passed": True,
                "note": "chưa chạy --mode export", "files": {k: str(v) for k, v in files.items()}}
    eligible = set(settings["export"]["eligible_qc"])
    min_gain = settings["export"]["generator_min_gain"]
    records = eligible_rows = 0
    for endpoint in endpoints:
        for path in layout.bucket_files("records_dir", endpoint):
            table = pq.read_table(path, columns=["qc_status"])
            records += table.num_rows
            eligible_rows += sum(1 for status in table.column("qc_status").to_pylist()
                                 if status in eligible)
    evidence = pq.ParquetFile(files["evidence_records"]).metadata.num_rows
    evaluator = pq.ParquetFile(files["evaluator"]).metadata.num_rows
    generator_table = pq.read_table(files["generator"], columns=["gain", "partition"])
    gains = np.asarray(generator_table.column("gain").to_numpy(zero_copy_only=False),
                       dtype=np.float64)
    below = int((gains <= min_gain).sum())
    return {"available": True,
            "passed": evidence == records and evaluator == eligible_rows and below == 0
            and generator_table.num_rows <= evaluator,
            "records": records, "evidence_rows": evidence, "eligible_records": eligible_rows,
            "evaluator_rows": evaluator, "generator_rows": generator_table.num_rows,
            "generator_rows_below_min_gain": below}


def verify_split(config: Mapping[str, Any], root: Path, layout: Layout,
                 endpoints: Sequence[str]) -> dict[str, Any]:
    """Năm phép kiểm tra của split, tính lại số cặp trực tiếp từ bản ghi bước 4."""
    _, dataset = layouts(config, root)
    path = dataset.file("protein_split")
    if not path.is_file():
        return {"available": False, "passed": True, "note": "chưa chạy --mode split",
                "file": str(path)}
    table = pq.read_table(path)
    counts = dataset_pair_counts(layout, endpoints)
    checks = split_checks(config, root, table, counts)
    passed = all(checks[name]["passed"] for name in
                 ("clusters_single_partition", "proteins_single_partition", "pair_totals"))
    return {"available": True, "passed": passed, "file": str(path), "checks": checks}


def run_verify(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Chạy mọi phép kiểm tra của Stage 2b trên output của full; trả về báo cáo.

    Các phép: bảo toàn số cặp; tính lại thống kê từ evidence; không trộn endpoint hay
    protein và pair_id đúng công thức; mỗi assay đúng một Δp; đối chiếu một mẫu ngược về
    observations gốc; phân bố Δp; số bản ghi theo cờ QC; kiểm tra split và phần xuất nếu
    đã có.
    """
    full, _ = layouts(config, root)
    endpoints = list(config["endpoints"])
    if not full.root.is_dir():
        raise FileNotFoundError(f"chưa có {full.root}; chạy --mode full trước")
    checks = {
        "bao_toan_cap": check_conservation(full, endpoints),
        "thong_ke_tu_evidence": check_record_stats(full, config, endpoints),
        "khoa_endpoint_protein": check_keys(full, config, endpoints),
        "mot_delta_moi_assay": check_one_delta_per_assay(full, endpoints),
        "doi_chieu_observations": check_against_observations(config, root, full, endpoints),
        "split": verify_split(config, root, full, endpoints),
        "export": check_export(config, root, full, endpoints),
    }
    report = {
        "stage": "stage2b_verify",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "full_dir": str(full.root),
        "checks": checks,
        "delta_distribution": delta_distribution(full, endpoints,
                                                 config["stage2b"]["run"]["delta_percentiles"]),
        "qc_summary": qc_summary(full, config, endpoints),
    }
    report["passed"] = all(check["passed"] for check in checks.values())
    log_lines(verify_lines(report, config))
    return report


def verify_lines(report: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """Báo cáo verify: PASS/FAIL từng phép, phân bố Δp, cờ QC."""
    rows = [[name, "PASS" if check["passed"] else "FAIL"]
            for name, check in report["checks"].items()]
    lines = ["Kết quả verify:", *format_table(["phép kiểm tra", "kết quả"], rows)]
    percentiles = config["stage2b"]["run"]["delta_percentiles"]
    headers = ["endpoint", "bản ghi", *[percentile_label(value) for value in percentiles]]
    table_rows = []
    for endpoint, entry in report["delta_distribution"].items():
        means = entry["records_mean"]
        table_rows.append([endpoint, fmt_int(means["n"]),
                           *[fmt_float(means["percentiles"].get(percentile_label(value)))
                             for value in percentiles]])
    lines += ["", "Phân bố Δp trung bình mỗi bản ghi:", *format_table(headers, table_rows)]
    lines += ["", *qc_lines(report["qc_summary"], config["stage2b"]["qc"]["report_levels"])]
    return lines
