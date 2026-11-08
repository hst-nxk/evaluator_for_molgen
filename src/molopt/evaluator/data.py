"""Stage 3: đọc dataset evaluator, lọc theo partition và theo lát cắt, đóng gói batch.

Dataset đầu vào (stage3.inputs.evaluator trong config) chỉ ĐỌC, không bao giờ ghi.

Module này không import torch: nó trả về mảng numpy gồm CHỈ SỐ vào các bảng cache
(fingerprint phân tử, fingerprint nhóm thế, embedding protein) chứ không trả về vector.
Nhờ vậy một batch chỉ là một mảng số nguyên, và việc tra vector để lại cho train.py.

Partition `test` bị chặn ở mức code: mọi hàm đọc dữ liệu đi qua `check_partition`, và chỉ
có đường duy nhất để đọc test là truyền allow_test=True — hiện không chỗ nào trong dự án
truyền giá trị đó.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"

# Cột cần đọc từ dataset. Giữ đúng tên của Stage 2b; không đổi tên ở đây để truy lại được.
PAIR_COLUMNS = ["pair_id", "endpoint", "protein_hash", "partition", "smiles_a", "smiles_b",
                "variable_a", "variable_b", "delta_p_mean", "n_assays", "n_documents",
                "n_distinct_values", "qc_status", "unseen_pair", "plausible_range",
                "source_quality_warning"]

# Khóa lát cắt mà config được phép dùng; khóa lạ là lỗi cấu hình, không bị bỏ qua im lặng.
SLICE_KEYS = ("unseen_pair", "plausible_range", "no_source_warning", "min_n_assays",
              "min_n_documents", "min_distinct_values", "abs_delta_p_above",
              "abs_delta_p_at_most")
# Hai khóa lát cắt theo độ lớn |Δp|, để tách riêng ca activity cliff. Định nghĩa cliff lấy
# từ MoleculeACE (van Tilborg, Alenicheva, Grisoni 2022, J. Chem. Inf. Model. 62(23):
# 5938-5951, DOI 10.1021/acs.jcim.2c01073; mã nguồn benchmark/cliffs.py): một cặp là cliff
# khi hai phân tử đủ giống nhau VÀ chênh hoạt tính hơn 10 lần. Dataset này toàn cặp MMP
# (cùng khung, khác đúng một nhóm thế) nên điều kiện giống nhau đã thỏa theo cách dựng dữ
# liệu — đúng nhánh mmp=True của find_cliffs. Còn lại là điều kiện hoạt tính: hơn 10 lần
# trên thang nồng độ = |Δp| > 1. `find_cliffs` của bài dùng dấu > NGẶT, nên
# abs_delta_p_above cũng ngặt, và abs_delta_p_at_most dùng <= để hai lát cắt phủ kín và
# không chồng nhau.
ABSOLUTE_DELTA_KEYS = ("abs_delta_p_above", "abs_delta_p_at_most")
# Khóa lát cắt dạng "ít nhất N" -> cột số nguyên của dataset mà nó so.
MINIMUM_KEYS = {"min_n_assays": "n_assays", "min_n_documents": "n_documents",
                "min_distinct_values": "n_distinct_values"}

# Cột khóa và cột giá trị của bảng đổi partition (xem load_partition_override).
OVERRIDE_COLUMNS = ("protein_hash", "partition")
# Bảng đổi partition đã đọc, theo đường dẫn tuyệt đối. Có cache vì read_columns được gọi
# lại nhiều lần trong một lần chạy (train, validation, từng lát cắt) và bảng này chỉ vài
# nghìn dòng nhưng nằm trên đĩa; đọc lại mỗi lần là tốn vô ích.
_OVERRIDE_CACHE: dict[Path, tuple[pa.Array, pa.Array]] = {}


def load_partition_override(config: Mapping[str, Any], root: Path
                            ) -> tuple[pa.Array, pa.Array] | None:
    """Đọc bảng đổi partition nếu config khai báo; không khai báo thì trả về None.

    Nhận: config, thư mục gốc. Trả về: (mảng protein_hash, mảng partition) hoặc None.

    Vì sao có đường này: thí nghiệm đổi ngưỡng giống nhau của protein khi chia tập chỉ
    đổi việc protein nào thuộc partition nào. Dataset, cặp phân tử, nhãn, cache
    fingerprint và cache embedding đều không đổi (cache đánh khóa theo phân tử và theo
    protein, không theo partition), nên dựng lại dataset chỉ để đổi một cột là vừa tốn
    vừa tạo thêm một phiên bản dữ liệu phải quản lý.
    """
    setting = str(config["stage3"]["inputs"].get("partition_override") or "").strip()
    if not setting:
        return None
    path = (root / setting).resolve()
    cached = _OVERRIDE_CACHE.get(path)
    if cached is not None:
        return cached
    if not path.is_file():
        raise FileNotFoundError(f"chưa có bảng đổi partition {path}; dựng bằng "
                                f"scripts/build_identity_split.py build")
    table = pq.read_table(path, columns=list(OVERRIDE_COLUMNS))
    keys = table.column("protein_hash").combine_chunks()
    values = table.column("partition").combine_chunks()
    if isinstance(keys, pa.ChunkedArray):
        keys = keys.chunk(0) if keys.num_chunks == 1 else pa.concat_arrays(keys.chunks)
    if isinstance(values, pa.ChunkedArray):
        values = values.chunk(0) if values.num_chunks == 1 else pa.concat_arrays(values.chunks)
    if len(set(keys.to_pylist())) != len(keys):
        raise ValueError(f"{path} có protein_hash trùng; mỗi protein chỉ được một partition")
    _OVERRIDE_CACHE[path] = (keys, values)
    return keys, values


def apply_partition_override(table: pa.Table, override: tuple[pa.Array, pa.Array]
                             ) -> pa.Table:
    """Thay cột 'partition' của bảng bằng partition tra từ 'protein_hash'.

    Protein không có trong bảng đổi là LỖI chứ không bị bỏ qua: thiếu một protein nghĩa
    là bảng đổi dựng từ lần export khác, và khi đó một phần dữ liệu sẽ chia theo cách cũ
    còn phần kia theo cách mới — kết quả sai mà không báo gì.
    """
    keys, values = override
    position = pc.index_in(table.column("protein_hash"), value_set=keys)
    missing = pc.sum(pc.is_null(position)).as_py() or 0
    if missing:
        unknown = pc.filter(table.column("protein_hash"), pc.is_null(position))
        raise ValueError(f"{missing} dòng có protein không nằm trong bảng đổi partition, "
                         f"ví dụ {unknown.to_pylist()[:3]}")
    replaced = pc.take(values, position)
    return table.set_column(table.schema.get_field_index("partition"), "partition", replaced)


@dataclass(frozen=True)
class PairArrays:
    """Một partition đã mã hóa thành chỉ số, sẵn sàng cho vòng train.

    molecule_a / molecule_b : chỉ số dòng trong bảng fingerprint phân tử
    fragment_a / fragment_b : chỉ số dòng trong bảng fingerprint nhóm thế
    protein                 : chỉ số dòng trong bảng embedding protein
    endpoint                : mã endpoint theo thứ tự config["endpoints"]
    target                  : Δp thật (nhãn)
    slices                  : {tên lát cắt: mặt nạ bool} để báo cáo chỉ số theo lát cắt
    """

    molecule_a: np.ndarray
    molecule_b: np.ndarray
    fragment_a: np.ndarray
    fragment_b: np.ndarray
    protein: np.ndarray
    endpoint: np.ndarray
    target: np.ndarray
    slices: Mapping[str, np.ndarray]

    def __len__(self) -> int:
        return int(self.target.size)


def check_partition(partition: str, allow_test: bool = False) -> str:
    """Chặn việc đọc partition test.

    Vì sao chặn ở mức code: test chỉ được dùng một lần, ở cuối, cho báo cáo cuối cùng.
    Mọi việc chọn mô hình, chọn tham số và early stopping chỉ dùng validation. Một lời
    nhắc trong tài liệu dễ bị bỏ qua hơn một ValueError.
    """
    if partition == TEST and not allow_test:
        raise ValueError("partition 'test' không được dùng để train, chọn tham số hay "
                         "early stopping; chỉ dùng 'train' và 'validation'")
    if partition not in (TRAIN, VALIDATION, TEST):
        raise ValueError(f"partition lạ: {partition!r}")
    return partition


def slice_mask(table: pa.Table, filters: Mapping[str, Any]) -> np.ndarray:
    """Mặt nạ bool của một lát cắt.

    Nhận: bảng cặp; các điều kiện của lát cắt (xem SLICE_KEYS).
    Trả về: mảng bool cùng số dòng với bảng.
    Lát cắt rỗng cho mặt nạ toàn True (tức là "toàn bộ partition").
    """
    mask = np.ones(table.num_rows, dtype=bool)
    for key, wanted in filters.items():
        if key not in SLICE_KEYS:
            raise ValueError(f"khóa lát cắt lạ: {key!r}; cho phép {SLICE_KEYS}")
        if key in ABSOLUTE_DELTA_KEYS:
            values = np.abs(np.asarray(
                table.column("delta_p_mean").to_numpy(zero_copy_only=False),
                dtype=np.float64))
            mask &= (values > float(wanted) if key == "abs_delta_p_above"
                     else values <= float(wanted))
        elif key == "no_source_warning":
            column = table.column("source_quality_warning")
            found = np.asarray(pc.is_null(column).to_numpy(zero_copy_only=False), dtype=bool)
            mask &= found if wanted else ~found
        elif key in MINIMUM_KEYS:
            values = np.asarray(table.column(MINIMUM_KEYS[key]).to_numpy(zero_copy_only=False))
            mask &= values >= int(wanted)
        else:
            values = np.asarray(
                pc.fill_null(table.column(key), False).to_numpy(zero_copy_only=False),
                dtype=bool)
            mask &= values if wanted else ~values
    return mask


def read_columns(config: Mapping[str, Any], root: Path, partition: str,
                 columns: Sequence[str], limit: int | None = None,
                 allow_test: bool = False) -> pa.Table:
    """Đọc một partition nhưng chỉ các cột được yêu cầu.

    Nhận: config, thư mục gốc, tên partition, danh sách cột, số dòng tối đa, allow_test.
    Trả về: bảng Arrow gồm đúng các cột đó (luôn kèm 'partition' để lọc được).
    Có hàm riêng vì baseline tầm thường chỉ cần hai cột (endpoint và nhãn) trên 8,4 triệu
    dòng train; đọc mọi cột cho việc đó là tốn vô ích.
    `limit` lấy các dòng ĐẦU TIÊN sau khi lọc partition, không lấy mẫu ngẫu nhiên: dataset
    được ghi theo thứ tự (endpoint, bucket protein) nên phần đầu vẫn đủ nhiều protein và
    nhiều cặp để kiểm tra pipeline, mà kết quả cố định giữa các lần chạy smoke.

    Đây là CHỖ DUY NHẤT trong Stage 3 lọc theo partition, nên bảng đổi partition của
    config cũng chỉ cần áp ở đây là mọi đường đọc (train, eval, errors, features) đều
    theo cùng một cách chia.
    """
    check_partition(partition, allow_test)
    path = root / config["stage3"]["inputs"]["evaluator"]
    if not path.is_file():
        raise FileNotFoundError(f"chưa có {path}; chạy Stage 2b --mode export trước")
    override = load_partition_override(config, root)
    wanted = list(columns)
    if "partition" not in wanted:
        wanted.append("partition")
    extra_key = override is not None and "protein_hash" not in wanted
    if extra_key:
        wanted.append("protein_hash")
    handle = pq.ParquetFile(path)
    chunks: list[pa.Table] = []
    rows = 0
    for batch in handle.iter_batches(batch_size=200_000, columns=wanted):
        table = pa.Table.from_batches([batch])
        if override is not None:
            table = apply_partition_override(table, override)
        table = table.filter(pc.equal(table.column("partition"), partition))
        if not table.num_rows:
            continue
        if limit is not None and rows + table.num_rows > limit:
            table = table.slice(0, limit - rows)
        chunks.append(table)
        rows += table.num_rows
        if limit is not None and rows >= limit:
            break
    if not chunks:
        raise ValueError(f"partition {partition!r} không có dòng nào trong {path}")
    result = pa.concat_tables(chunks)
    if extra_key:
        result = result.drop_columns(["protein_hash"])
    return result


def read_partition(config: Mapping[str, Any], root: Path, partition: str,
                   limit: int | None = None, allow_test: bool = False) -> pa.Table:
    """Đọc một partition của dataset evaluator, đủ PAIR_COLUMNS."""
    return read_columns(config, root, partition, PAIR_COLUMNS, limit, allow_test)


def code_column(values: pa.ChunkedArray | pa.Array, vocabulary: pa.Array,
                label: str) -> np.ndarray:
    """Chỉ số của từng giá trị trong `vocabulary`; giá trị thiếu là lỗi.

    Vì sao không bỏ qua giá trị thiếu: thiếu nghĩa là cache không khớp dataset (ví dụ
    cache dựng từ lần export trước), và huấn luyện trên dữ liệu khớp một nửa sẽ cho kết
    quả sai mà không báo gì.
    """
    position = pc.index_in(values, value_set=vocabulary)
    missing = pc.sum(pc.is_null(position)).as_py() or 0
    if missing:
        examples = pc.filter(values, pc.is_null(position)).to_pylist()[:3]
        raise ValueError(f"{missing} giá trị {label} không có trong cache, ví dụ {examples}; "
                         f"dựng lại cache bằng --mode prepare")
    return np.asarray(position.to_numpy(zero_copy_only=False), dtype=np.int64)


def encode_pairs(table: pa.Table, molecules: pa.Array, fragments: pa.Array,
                 proteins: pa.Array, endpoints: Sequence[str],
                 slices: Sequence[Mapping[str, Any]]) -> PairArrays:
    """Mã hóa bảng cặp thành chỉ số và nhãn.

    Nhận: bảng cặp; ba bảng từ vựng của cache (SMILES phân tử, SMILES nhóm thế, hash
    protein); danh sách endpoint theo thứ tự; định nghĩa các lát cắt.
    Trả về: PairArrays.
    """
    endpoint_vocabulary = pa.array(list(endpoints), pa.string())
    masks = {}
    for definition in slices:
        name = definition["name"]
        masks[name] = slice_mask(table, definition.get("filters", {}))
    return PairArrays(
        molecule_a=code_column(table.column("smiles_a"), molecules, "smiles_a"),
        molecule_b=code_column(table.column("smiles_b"), molecules, "smiles_b"),
        fragment_a=code_column(table.column("variable_a"), fragments, "variable_a"),
        fragment_b=code_column(table.column("variable_b"), fragments, "variable_b"),
        protein=code_column(table.column("protein_hash"), proteins, "protein_hash"),
        endpoint=code_column(table.column("endpoint"), endpoint_vocabulary, "endpoint"),
        target=np.asarray(table.column("delta_p_mean").to_numpy(zero_copy_only=False),
                          dtype=np.float32),
        slices=masks,
    )


def query_key_columns(table: pa.Table, molecules: pa.Array, proteins: pa.Array,
                      endpoints: Sequence[str]) -> dict[str, np.ndarray]:
    """Các cột khóa dùng để dựng query cho chỉ số ranking.

    Trả về: {'molecule_a', 'molecule_b', 'protein', 'endpoint'} đã mã hóa thành số nguyên.
    Việc nhân đôi hai chiều do metrics.expand_directions làm, không làm ở đây.
    """
    endpoint_vocabulary = pa.array(list(endpoints), pa.string())
    return {
        "molecule_a": code_column(table.column("smiles_a"), molecules, "smiles_a"),
        "molecule_b": code_column(table.column("smiles_b"), molecules, "smiles_b"),
        "protein": code_column(table.column("protein_hash"), proteins, "protein_hash"),
        "endpoint": code_column(table.column("endpoint"), endpoint_vocabulary, "endpoint"),
    }


def take_rows(arrays: PairArrays, rows: np.ndarray) -> PairArrays:
    """Lấy một tập con dòng của PairArrays, giữ nguyên các mặt nạ lát cắt tương ứng."""
    return PairArrays(
        molecule_a=arrays.molecule_a[rows], molecule_b=arrays.molecule_b[rows],
        fragment_a=arrays.fragment_a[rows], fragment_b=arrays.fragment_b[rows],
        protein=arrays.protein[rows], endpoint=arrays.endpoint[rows],
        target=arrays.target[rows],
        slices={name: mask[rows] for name, mask in arrays.slices.items()},
    )


def swap_sides(arrays: PairArrays) -> PairArrays:
    """Cùng dữ liệu nhưng đổi chỗ hai phía A và B; nhãn đổi dấu.

    Dùng để ĐO mức bất đối xứng của đầu ra `direct`: nếu mô hình đối xứng hoàn toàn thì
    f(A, B) = −f(B, A). Kiến trúc `twin` đối xứng do xây dựng, `direct` thì không, và dự
    án chọn đo mức lệch thay vì cưỡng chế nó (xem docs/STAGE3_EVALUATOR.md).
    Đây là đầu vào NHÂN TẠO cho mục đích chẩn đoán, không bao giờ dùng để train.
    """
    return PairArrays(
        molecule_a=arrays.molecule_b, molecule_b=arrays.molecule_a,
        fragment_a=arrays.fragment_b, fragment_b=arrays.fragment_a,
        protein=arrays.protein, endpoint=arrays.endpoint,
        target=-arrays.target, slices=arrays.slices,
    )


def identity_sides(arrays: PairArrays) -> PairArrays:
    """Cặp (X, X): phía B được thay bằng chính phía A; nhãn đúng bằng 0.

    Dùng để đo xem mô hình có cho Δp = 0 khi "phép sửa" không sửa gì. Cặp như vậy KHÔNG
    tồn tại trong dataset (cặp MMP luôn gồm hai phân tử khác nhau), nên phép đo này nói
    về khả năng ngoại suy ra ngoài phân bố huấn luyện, không nói về chất lượng khớp dữ
    liệu. Cũng là đầu vào nhân tạo, không dùng để train.
    """
    return PairArrays(
        molecule_a=arrays.molecule_a, molecule_b=arrays.molecule_a,
        fragment_a=arrays.fragment_a, fragment_b=arrays.fragment_a,
        protein=arrays.protein, endpoint=arrays.endpoint,
        target=np.zeros_like(arrays.target), slices=arrays.slices,
    )


def iter_batches(count: int, batch_size: int, rng: np.random.Generator | None = None
                 ) -> Iterator[np.ndarray]:
    """Chia 0..count-1 thành các batch chỉ số; có rng thì xáo trộn trước.

    Batch cuối ngắn hơn vẫn được dùng, không bỏ, để không mất dữ liệu ở mỗi epoch.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size phải >= 1, nhận {batch_size}")
    indices = np.arange(count, dtype=np.int64)
    if rng is not None:
        rng.shuffle(indices)
    for start in range(0, count, batch_size):
        yield indices[start:start + batch_size]


def iter_protein_batches(proteins: np.ndarray, batch_size: int,
                         rng: np.random.Generator | None = None,
                         max_proteins: int = 0) -> Iterator[np.ndarray]:
    """Chia batch sao cho MỖI BATCH CHỈ CHẠM ÍT PROTEIN; vẫn dùng hết mọi dòng một lần.

    Vì sao cần: ở cấu hình finetune, ESM2 nằm trong đồ thị tính toán và chạy một lượt cho
    mỗi protein PHÂN BIỆT trong batch, giữ lại activation cho lượt lùi. Lấy batch ngẫu
    nhiên thì 64 dòng chạm tới ~54 protein khác nhau (đo trên train v4), tức 54 lượt ESM2
    — ước tính ~68 GB, quá sức mọi card. Gom theo protein thì phần lớn batch chỉ có 1-2
    protein, nên bộ nhớ giảm khoảng một bậc mà SỐ BƯỚC MỖI EPOCH KHÔNG ĐỔI.

    Cách làm: xáo trộn dòng trong từng protein, xáo trộn thứ tự protein, nối lại rồi cắt
    thành các batch liên tiếp. Mỗi dòng xuất hiện đúng một lần mỗi epoch, như iter_batches.

    ĐÁNH ĐỔI, phải ghi vào paper: các dòng trong một batch không còn độc lập (phần lớn
    cùng một protein), nên gradient của mỗi bước nhiễu hơn so với lấy mẫu ngẫu nhiên.
    Đây là lý do hàm này CHỈ dùng cho finetune, còn frozen vẫn lấy mẫu ngẫu nhiên: ở
    frozen vector protein đã tính sẵn nên không có gì để tiết kiệm.

    `max_proteins` > 0 thì một batch KHÔNG BAO GIỜ chứa quá bấy nhiêu protein phân biệt:
    chạm mức đó là cắt batch sớm, dù chưa đủ `batch_size` dòng. Phần lớn batch chỉ có một
    protein nên việc cắt rất hiếm, nhưng nó biến mức dùng bộ nhớ từ giới hạn THỐNG KÊ
    thành giới hạn CHẮC CHẮN — cần thiết khi chia GPU với người khác, vì chỉ một batch
    hiếm chạm nhiều protein cũng đủ làm hết bộ nhớ của cả hai bên.

    Nhận: `proteins` là chỉ số protein của từng dòng (PairArrays.protein); cỡ batch; rng
    (None = không xáo trộn, để test lặp lại được); trần số protein mỗi batch (0 = không chặn).
    Trả về: lần lượt từng mảng chỉ số dòng.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size phải >= 1, nhận {batch_size}")
    order = np.argsort(proteins, kind="stable")
    sorted_proteins = proteins[order]
    # Ranh giới giữa các protein trong mảng đã sắp xếp.
    cuts = np.flatnonzero(np.diff(sorted_proteins)) + 1
    groups = np.split(order, cuts)
    if rng is not None:
        for group in groups:
            rng.shuffle(group)
        groups = [groups[i] for i in rng.permutation(len(groups))]
    if max_proteins < 1:
        flat = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
        for start in range(0, flat.size, batch_size):
            yield flat[start:start + batch_size]
        return
    held: list[np.ndarray] = []
    rows = 0
    seen = 0
    for group in groups:
        offset = 0
        # Protein này đã được tính vào batch đang dựng chưa. Một protein lớn có thể trải
        # qua nhiều batch, và ở MỖI batch nó phải được đếm lại một lần.
        counted = False
        while offset < group.size:
            if not counted and seen >= max_proteins:
                yield np.concatenate(held)
                held, rows, seen = [], 0, 0
            if not counted:
                seen += 1
                counted = True
            take = min(batch_size - rows, group.size - offset)
            held.append(group[offset:offset + take])
            rows += take
            offset += take
            if rows >= batch_size:
                yield np.concatenate(held)
                held, rows, seen = [], 0, 0
                counted = False
    if held:
        yield np.concatenate(held)


def distinct_strings(table: pa.Table, columns: Sequence[str]) -> pa.Array:
    """Tập giá trị chuỗi phân biệt trên nhiều cột, đã sắp xếp.

    Sắp xếp để thứ tự dòng trong cache không phụ thuộc thứ tự đọc, nhờ đó cache dựng lại
    hai lần cho cùng một file.
    """
    pieces = [table.column(name).combine_chunks() for name in columns]
    unique = pc.unique(pa.chunked_array(pieces).combine_chunks())
    return pc.take(unique, pc.array_sort_indices(unique))
