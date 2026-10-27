"""Tiện ích dùng chung cho các bước audit: đọc cột, mã hóa, gom nhóm, tóm tắt, ghi báo cáo.

Không hàm nào ở đây ghi vào data/: báo cáo và hình chỉ ghi vào thư mục runs của audit.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)
PERCENT = 100.0
NULL_CODE = -1


# --------------------------------------------------------------------------- đường dẫn


def runs_dir(config: Mapping[str, Any], root: Path) -> Path:
    """Thư mục ghi báo cáo audit (tạo nếu chưa có)."""
    path = root / config["audit"]["paths"]["runs_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_file(config: Mapping[str, Any], root: Path, key: str) -> Path:
    """Đường dẫn một file xuất của Stage 2b (khóa trong stage2b.files)."""
    stage = config["stage2b"]
    return root / stage["paths"]["dataset_dir"] / stage["files"][key]


def full_file(config: Mapping[str, Any], root: Path, key: str) -> Path:
    """Đường dẫn một file trung gian của Stage 2b trong thư mục full."""
    stage = config["stage2b"]
    return root / stage["paths"]["full_dir"] / stage["files"][key]


def observation_files(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    """Hai file observation sau dedup (đầu vào của Stage 2b), theo tên nguồn."""
    return {name: root / path for name, path in config["stage2b"]["inputs"].items()}


def read_observations(config: Mapping[str, Any], root: Path, roles: Sequence[str]) -> pa.Table:
    """Đọc các cột vai trò `roles` của cả hai nguồn, nối thành một bảng.

    Tên cột thật lấy từ stage2b.columns, bổ sung bằng audit.columns (cột Stage 2b không
    dùng, như source_curation và dedup_reason); bảng trả về dùng tên vai trò để code không
    phụ thuộc tên cột của Stage 1.
    """
    mapping = {**config["stage2b"]["columns"], **config["audit"]["columns"]}
    names = [mapping[role] for role in roles]
    tables = []
    for source, path in observation_files(config, root).items():
        table = pq.read_table(path, columns=names)
        LOGGER.info("đọc %s: %s dòng từ %s", source, f"{table.num_rows:,}", path)
        tables.append(table.rename_columns(list(roles)))
    return pa.concat_tables(tables).combine_chunks()


def usable_mask(table: pa.Table, config: Mapping[str, Any], require_keep: bool) -> np.ndarray:
    """Dòng đi vào Stage 2b: cùng bộ lọc với stage2b.filter.

    Điều kiện: qualifier thuộc qualifiers.exact, protein_status và assay_status bằng
    resolved_status, có InChIKey, SMILES và pActivity; thêm dedup_keep nếu require_keep.
    Bảng phải dùng tên cột vai trò (read_observations).
    """
    exact = pa.array(config["qualifiers"]["exact"])
    resolved = config["stage2b"]["filter"]["resolved_status"]
    parts = [
        pc.is_in(table.column("qualifier"), value_set=exact),
        pc.equal(table.column("protein_status"), resolved),
        pc.equal(table.column("assay_status"), resolved),
        pc.is_valid(table.column("inchikey")), pc.is_valid(table.column("smiles")),
        pc.is_valid(table.column("p_activity")),
    ]
    if require_keep:
        parts.append(table.column("keep"))
    mask = parts[0]
    for part in parts[1:]:
        mask = pc.and_(mask, part)
    return boolean(mask)


def boolean(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Cột/biểu thức boolean của pyarrow -> mảng numpy bool, null coi là False."""
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    return values.fill_null(False).to_numpy(zero_copy_only=False).astype(bool)


# --------------------------------------------------------------------------- mã hóa


def encode(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Mã hóa một cột thành số nguyên (cùng giá trị -> cùng mã); null -> NULL_CODE.

    Chuỗi được đổi sang large_string trước khi gộp mảnh, vì cột SMILES ghép hai phía có
    thể vượt giới hạn 2 GB của kiểu string thường.
    """
    if pa.types.is_string(values.type):
        values = values.cast(pa.large_string())
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    encoded = pc.dictionary_encode(values)
    indices = encoded.indices.fill_null(NULL_CODE)
    return indices.to_numpy(zero_copy_only=False).astype(np.int64)


def encode_rounded(values: pa.Array | pa.ChunkedArray, decimals: int) -> np.ndarray:
    """Làm tròn số thực tới `decimals` chữ số rồi đổi sang số nguyên (null bị bỏ qua).

    Giá trị của dòng null là tùy ý; dùng `valid_mask` để loại dòng null khỏi phép so.
    """
    array = values.to_numpy(zero_copy_only=False).astype(np.float64)
    scaled = np.round(np.nan_to_num(array, nan=0.0) * 10.0 ** decimals)
    return scaled.astype(np.int64)


def valid_mask(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """True ở dòng không null (và không NaN nếu là số thực)."""
    mask = pc.is_valid(values)
    if pa.types.is_floating(values.type):
        mask = pc.and_(mask, pc.invert(pc.is_nan(values)))
    if isinstance(mask, pa.ChunkedArray):
        mask = mask.combine_chunks()
    return mask.to_numpy(zero_copy_only=False).astype(bool)


def group_ids(columns: Sequence[np.ndarray]) -> np.ndarray:
    """Mã nhóm dày đặc (0, 1, 2, ...) cho các dòng có cùng giá trị ở MỌI cột.

    Nhận: các mảng số nguyên cùng độ dài. Trả về: mảng int64, cùng độ dài.
    Cách làm: sắp xếp theo mọi cột (np.lexsort), đánh dấu chỗ giá trị đổi, cộng dồn.
    Hai dòng cùng nhóm khi và chỉ khi chúng bằng nhau ở mọi cột.
    """
    if not columns:
        raise ValueError("group_ids cần ít nhất một cột")
    size = len(columns[0])
    if size == 0:
        return np.zeros(0, dtype=np.int64)
    # np.lexsort lấy cột CUỐI làm khóa chính, nên đảo thứ tự để cột đầu là khóa chính.
    order = np.lexsort(tuple(columns[::-1]))
    changed = np.zeros(size, dtype=bool)
    changed[0] = True
    for column in columns:
        ordered = column[order]
        changed[1:] |= ordered[1:] != ordered[:-1]
    ids = np.empty(size, dtype=np.int64)
    ids[order] = np.cumsum(changed) - 1
    return ids


def distinct_per_group(ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Số giá trị khác nhau của `values` trong từng nhóm `ids` (độ dài = số nhóm)."""
    if ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    pairs = group_ids([ids, values])
    group_of_pair = np.zeros(int(pairs.max()) + 1, dtype=np.int64)
    group_of_pair[pairs] = ids
    return np.bincount(group_of_pair, minlength=int(ids.max()) + 1)


# --------------------------------------------------------------------------- tóm tắt


def quantile_label(quantile: float) -> str:
    """Nhãn phân vị: 0,5 -> 'p50'; 0,999 -> 'p99.9' (làm tròn để tránh 94,999...)."""
    return f"p{round(quantile * PERCENT, 6):g}"


def quantile_summary(values: np.ndarray, quantiles: Sequence[float]) -> dict[str, Any]:
    """n, trung bình, độ lệch chuẩn và các phân vị của một mảng số (bỏ NaN)."""
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    result: dict[str, Any] = {"n": int(array.size)}
    if array.size == 0:
        result.update({"mean": None, "std": None})
        result.update({quantile_label(q): None for q in quantiles})
        return result
    result.update({"mean": float(array.mean()), "std": float(array.std())})
    for quantile, value in zip(quantiles, np.quantile(array, list(quantiles))):
        result[quantile_label(quantile)] = float(value)
    return result


def count_values(values: Iterable[Any]) -> dict[str, int]:
    """Đếm số lần xuất hiện của từng giá trị; khóa là chuỗi, sắp theo số lần giảm dần."""
    counts = Counter("null" if value is None else str(value) for value in values)
    return dict(counts.most_common())


def share(part: float, total: float) -> float | None:
    """Tỉ lệ phần trăm part/total; None khi total = 0."""
    return None if total == 0 else PERCENT * part / total


def bucket_counts(values: np.ndarray, edges: Sequence[float]) -> list[dict[str, Any]]:
    """Đếm theo khoảng [e_i, e_{i+1}), khoảng cuối là [e_last, +vô cùng).

    Giá trị nhỏ hơn mốc đầu được đếm riêng ở khoảng đầu tiên ('< e_0').
    """
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    rows: list[dict[str, Any]] = []
    below = int(np.count_nonzero(array < edges[0]))
    if below:
        rows.append({"range": f"< {edges[0]:g}", "count": below})
    for low, high in zip(edges, list(edges[1:]) + [np.inf]):
        count = int(np.count_nonzero((array >= low) & (array < high)))
        label = f"[{low:g}, {high:g})" if np.isfinite(high) else f">= {low:g}"
        rows.append({"range": label, "count": count, "percent": share(count, array.size)})
    return rows


# --------------------------------------------------------------------------- định dạng


def number(value: Any) -> str:
    """Số nguyên kiểu Việt (dấu chấm phân cách nghìn), số thực 4 chữ số; None -> '-'."""
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}".replace(",", ".")
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}".replace(".", ",")
    return str(value)


def percent(value: float | None) -> str:
    """Tỉ lệ phần trăm một chữ số thập phân; None -> '-'."""
    return "-" if value is None else f"{value:.1f} %".replace(".", ",")


def table_lines(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    """Bảng văn bản căn cột (cột đầu căn trái, các cột sau căn phải)."""
    cells = [[str(cell) for cell in header]] + [[str(cell) for cell in row] for row in rows]
    widths = [max(len(row[index]) for row in cells) for index in range(len(header))]
    lines = []
    for position, row in enumerate(cells):
        parts = [row[0].ljust(widths[0])]
        parts += [cell.rjust(width) for cell, width in zip(row[1:], widths[1:])]
        lines.append("  ".join(parts))
        if position == 0:
            lines.append("  ".join("-" * width for width in widths))
    return lines


def to_jsonable(value: Any) -> Any:
    """Đổi kiểu numpy về kiểu Python để ghi JSON."""
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    return value


def write_report(directory: Path, name: str, payload: Mapping[str, Any],
                 lines: Sequence[str]) -> tuple[Path, Path]:
    """Ghi báo cáo <name>.json (số liệu) và <name>.txt (bản đọc được)."""
    json_path = directory / f"{name}.json"
    text_path = directory / f"{name}.txt"
    json_path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False),
                         encoding="utf-8")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("báo cáo: %s", text_path)
    return json_path, text_path
