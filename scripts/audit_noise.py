"""Stage 2a, việc 2: audit độ nhiễu của observations (chỉ đọc, không lọc, không khuyến nghị).

Chạy từ thư mục gốc của repo, SAU khi đã chạy dedup_cross_source.py:
    python scripts\\audit_noise.py

Luồng dữ liệu:
    data/dedup/*_observations_dedup.parquet   (bản sao có cột dedup_*, chỉ đọc)
    -> với từng endpoint: đọc các dòng của endpoint đó ở cả hai nguồn
    -> tập audit: dedup_keep = True, qualifier '=', protein và assay resolved, có pActivity
    -> với từng phạm vi (chembl, bindingdb, all = gộp hai nguồn sau dedup):
         1. replicate trong cùng assay      2. số assay trên mỗi cặp phân tử–protein
         3. offset giữa các assay            4. phân bố pActivity
         5. top protein và STD replicate trung vị
         7. chẩn đoán replicate thật hay bản ghi trùng: phân bố STD và |Δ| theo khoảng,
            cờ potential_duplicate của ChEMBL, cặp assay cùng/khác publication
    -> 6. ChEMBL: giá trị gốc so với giá trị đã curation (đọc riêng file ChEMBL theo batch,
          chạy TRƯỚC mục 1–5 vì cùng lượt đọc đó lấy luôn cờ potential_duplicate)
    -> runs/stage2a/noise_audit_<thời điểm>.{log,txt,json}

Phạm vi 'bindingdb' là phần BindingDB còn lại sau dedup, tức là các phép đo không trùng
với ChEMBL.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Dùng lại cách chuẩn hóa DOI/patent của bước dedup, để "cùng publication" ở mục 7 có
# đúng nghĩa "chung tài liệu" như khi dedup (scripts/ nằm trên sys.path khi chạy script).
from dedup_cross_source import normalize_doi, normalize_patent

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
LOGGER = logging.getLogger("audit_noise")
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

CHEMBL = "chembl"
CHEMBL_PREFIX = CHEMBL + ":"
ALL_SCOPE = "all"
KEEP = "keep"
MIN_REPLICATES = 2
# Assay không có khóa tài liệu nào: không bao giờ được coi là "cùng publication".
PUBLICATION_MISSING = -1
# Các nhóm cặp assay ở mục 7: (cùng publication?, có cờ potential_duplicate ở ≥1 bên?).
OFFSET_CATEGORIES = tuple((same, flagged) for same in (True, False) for flagged in (True, False))
VIEW_SAME = "cùng publication"
VIEW_DIFFERENT = "khác publication"
VIEW_INDEPENDENT = "khác publication, không cờ"
VIEW_FLAGGED = "có cờ potential_duplicate"
VIEW_UNFLAGGED = "không cờ potential_duplicate"
OFFSET_VIEWS = {
    VIEW_SAME: ((True, True), (True, False)),
    VIEW_DIFFERENT: ((False, True), (False, False)),
    VIEW_INDEPENDENT: ((False, False),),
    VIEW_FLAGGED: ((True, True), (False, True)),
    VIEW_UNFLAGGED: ((True, False), (False, False)),
}

# Vai trò cột đọc cho các mục 1–5 và 7 (tên cột thật trong config stage2a.columns).
AUDIT_ROLES = (
    "observation_id", "source", "assay_id", "assay_entry_id", "protein_hash",
    "protein_accession", "protein_name", "inchikey", "endpoint", "qualifier", "p_activity",
    "protein_status", "assay_status", "pmid", "doi", "patent", "document_id",
)
# Vai trò cột đọc cho mục 6 (chỉ ChEMBL).
ORIGINAL_ROLES = (
    "observation_id", "endpoint", "qualifier", "p_activity", "protein_status",
    "assay_status", "activity_context", "value_source", "unit_source", "qualifier_source",
)
# Khóa trong activity_context (JSON) do Stage 1 ghi cho ChEMBL: bộ giá trị gốc.
CONTEXT_VALUE = "value"
CONTEXT_UNITS = "units"
CONTEXT_RELATION = "relation"


class OriginalComparison(NamedTuple):
    """Kết quả so một dòng ChEMBL: giá trị gốc với giá trị đã curation (standard_*)."""
    value_differs: bool
    unit_differs: bool
    relation_differs: bool
    original_missing: bool


# --------------------------------------------------------------------------- định dạng số


def fmt_int(value: int) -> str:
    """Số nguyên có dấu chấm phân cách hàng nghìn, vd. 5338213 -> '5.338.213'."""
    return f"{value:,}".replace(",", ".")


def fmt_float(value: float | None, digits: int = 2) -> str:
    """Số thực kiểu Việt Nam (dấu phẩy thập phân), vd. 1234.5 -> '1.234,50'; None -> '–'."""
    if value is None:
        return "–"
    text = f"{value:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_pct(part: int, whole: int, digits: int = 1) -> str:
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


# --------------------------------------------------------------------------- thống kê (thuần)


def values_of(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """Cột số Arrow -> mảng numpy float64, bỏ null."""
    return np.asarray(pc.drop_null(column).to_numpy(zero_copy_only=False), dtype=np.float64)


def summarize_values(values: np.ndarray, percentiles: Sequence[float]) -> dict[str, float | None]:
    """Các phân vị chính xác (nội suy tuyến tính) của một mảng; mảng rỗng -> None hết."""
    if values.size == 0:
        return {percentile_label(q): None for q in percentiles}
    results = np.percentile(values, list(percentiles))
    return {percentile_label(q): float(value) for q, value in zip(percentiles, results)}


def count_at_most(values: np.ndarray, thresholds: Sequence[float]) -> dict[str, int]:
    """Số phần tử <= từng ngưỡng (khóa là ngưỡng dạng chuỗi, cho JSON)."""
    return {str(threshold): int(np.count_nonzero(values <= threshold))
            for threshold in thresholds}


def bucket_counts(counts: np.ndarray, buckets: Sequence[int]) -> dict[str, int]:
    """Đếm theo giỏ: bằng đúng từng giá trị, riêng giỏ cuối là '>= giá trị'.

    Vd. buckets [1, 2, 3] -> {'1': ..., '2': ..., '≥3': ...}.
    """
    result = {str(bucket): int(np.count_nonzero(counts == bucket)) for bucket in buckets[:-1]}
    result[f"≥{buckets[-1]}"] = int(np.count_nonzero(counts >= buckets[-1]))
    return result


def as_numpy(column: pa.ChunkedArray | pa.Array, dtype: Any) -> np.ndarray:
    """Cột Arrow không có null -> mảng numpy kiểu `dtype`."""
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype)


def pair_indices(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mọi cặp (rows[i], rows[j]) với i < j; ít hơn 2 phần tử -> hai mảng rỗng.

    Vd. [4, 7, 9] -> ([4, 4, 7], [7, 9, 9]).
    """
    if rows.size < MIN_REPLICATES:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    first, second = np.triu_indices(rows.size, k=1)
    return rows[first], rows[second]


def bin_labels(edges: Sequence[float]) -> list[str]:
    """Nhãn các khoảng của mục 7: '= 0', '(0; e0)', '[e0; e1)', ..., '≥ e_cuối'."""
    def show(value: float) -> str:
        return f"{value:g}".replace(".", ",")

    labels = ["= 0", f"(0; {show(edges[0])})"]
    labels += [f"[{show(low)}; {show(high)})" for low, high in zip(edges, edges[1:])]
    return labels + [f"≥ {show(edges[-1])}"]


def bin_counts(values: np.ndarray, zero: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Đếm theo các khoảng của bin_labels: ô đầu là các phần tử có `zero` = True.

    Nhận: giá trị không âm, mặt nạ "bằng 0", các mốc tăng dần.
    Trả về: mảng len(edges) + 2 số đếm.
    Vì sao "bằng 0" là một mặt nạ riêng: STD bằng 0 được xác định bằng max − min của
    nhóm (chính xác), còn chính STD có thể lệch 0 một chút do làm tròn số thực.
    """
    rest = values[~zero]
    index = np.searchsorted(np.asarray(edges, dtype=np.float64), rest, side="right")
    counts = np.bincount(index, minlength=len(edges) + 1)
    return np.concatenate([[np.count_nonzero(zero)], counts]).astype(np.int64)


def publication_ids(pairs: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Gom assay thành publication: hai assay chung một khóa tài liệu thì cùng publication.

    Nhận: các cặp (assay_key, khóa tài liệu), vd. ('chembl::CHEMBL1', 'pmid:123').
    Trả về: {assay_key: số hiệu publication}, đánh số theo thứ tự tên để ổn định.
    Vì sao gom bắc cầu (union-find): một assay ChEMBL có PMID và DOI, một assay BindingDB
    chỉ có DOI; chung DOI là đủ để biết hai assay đến từ cùng một bài.
    """
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    owner: dict[str, str] = {}
    for assay, key in pairs:
        parent.setdefault(assay, assay)
        if key not in owner:
            owner[key] = assay
            continue
        first, second = find(assay), find(owner[key])
        if first != second:
            parent[max(first, second)] = min(first, second)
    roots: dict[str, int] = {}
    return {assay: roots.setdefault(find(assay), len(roots)) for assay in sorted(parent)}


def new_offset_stats(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Bộ đếm rỗng cho một nhóm cặp assay ở mục 7."""
    return {"pairs": 0,
            "bins": np.zeros(len(settings["diagnostic_bin_edges"]) + 2, dtype=np.int64),
            "above": np.zeros(len(settings["offset_thresholds"]), dtype=np.int64),
            "histogram": new_histogram(settings["offset_histogram_bin_width"],
                                       settings["offset_histogram_max"])}


def update_offset_stats(stats: Mapping[str, Any], differences: np.ndarray,
                        settings: Mapping[str, Any]) -> dict[str, Any]:
    """Bộ đếm mới = `stats` cộng thêm các |Δ| trong `differences`."""
    zero = differences <= settings["zero_tolerance"]
    return {
        "pairs": stats["pairs"] + int(differences.size),
        "bins": stats["bins"] + bin_counts(differences, zero, settings["diagnostic_bin_edges"]),
        "above": stats["above"] + np.array([np.count_nonzero(differences > threshold)
                                            for threshold in settings["offset_thresholds"]],
                                           dtype=np.int64),
        "histogram": add_to_histogram(stats["histogram"], differences,
                                      settings["offset_histogram_bin_width"]),
    }


def summarize_offset_view(parts: Sequence[Mapping[str, Any]],
                          settings: Mapping[str, Any]) -> dict[str, Any]:
    """Gộp bộ đếm của vài nhóm cặp assay thành số liệu báo cáo (dạng JSON được)."""
    combined = new_offset_stats(settings)
    for part in parts:
        combined = {key: combined[key] + part[key] for key in combined}
    labels = bin_labels(settings["diagnostic_bin_edges"])
    return {
        "pairs": int(combined["pairs"]),
        "bins": dict(zip(labels, (int(count) for count in combined["bins"]))),
        "above": {str(threshold): int(count) for threshold, count
                  in zip(settings["offset_thresholds"], combined["above"])},
        "abs_difference": histogram_percentiles(combined["histogram"],
                                                settings["offset_histogram_bin_width"],
                                                settings["category_percentiles"]),
    }


def new_histogram(bin_width: float, max_value: float) -> np.ndarray:
    """Histogram rỗng các ô rộng `bin_width` trên [0, max_value), cộng một ô tràn ở cuối."""
    return np.zeros(int(math.ceil(max_value / bin_width)) + 1, dtype=np.int64)


def add_to_histogram(counts: np.ndarray, values: np.ndarray, bin_width: float) -> np.ndarray:
    """Histogram mới = `counts` cộng các giá trị không âm `values` (giá trị quá lớn vào ô tràn)."""
    index = np.minimum((values / bin_width).astype(np.int64), counts.size - 1)
    return counts + np.bincount(index, minlength=counts.size)


def histogram_percentiles(counts: np.ndarray, bin_width: float,
                          percentiles: Sequence[float]) -> dict[str, float | None]:
    """Phân vị xấp xỉ từ histogram: tâm của ô chứa phần tử ở thứ hạng tương ứng.

    Nhận: số đếm mỗi ô, độ rộng ô, các phân vị. Trả về: {nhãn: giá trị} (sai số
    <= nửa độ rộng ô; None nếu histogram rỗng).
    Vì sao dùng histogram: số cặp assay tăng theo bình phương số assay của mỗi phân tử
    (một chất tham chiếu đo ở 500 assay cho 124.750 cặp), nên không giữ hết các hiệu
    trong bộ nhớ.
    """
    total = int(counts.sum())
    if total == 0:
        return {percentile_label(q): None for q in percentiles}
    cumulative = np.cumsum(counts)
    result = {}
    for q in percentiles:
        rank = q / 100.0 * (total - 1)
        index = int(np.searchsorted(cumulative, rank, side="right"))
        result[percentile_label(q)] = (index + 0.5) * bin_width
    return result


def most_common_labels(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str | None, str | None]]:
    """Với mỗi protein_hash: (accession, tên) xuất hiện nhiều nhất.

    Nhận: các dòng {protein_hash, protein_accession, protein_name, p_activity_count}.
    Vì sao cần: cùng một chuỗi protein có thể mang nhiều tên/accession ở các dòng khác
    nhau; nhãn hiển thị lấy cách ghi phổ biến nhất (hòa thì lấy accession, tên đứng
    trước theo thứ tự chữ, để kết quả không đổi giữa các lần chạy).
    """
    ordered = sorted(rows, key=lambda row: (-row["p_activity_count"],
                                            str(row["protein_accession"]),
                                            str(row["protein_name"])))
    labels: dict[str, tuple[str | None, str | None]] = {}
    for row in ordered:
        labels.setdefault(row["protein_hash"], (row["protein_accession"], row["protein_name"]))
    return labels


def numbers_equal(left: Any, right: Any, relative_tolerance: float) -> bool:
    """Hai giá trị có bằng nhau không: so như số nếu đọc được thành số, nếu không so chuỗi."""
    try:
        return math.isclose(float(left), float(right), rel_tol=relative_tolerance, abs_tol=0.0)
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()


def compare_original_standard(context: Mapping[str, Any], standard_value: str | None,
                              standard_unit: str | None, standard_relation: str | None,
                              relative_tolerance: float) -> OriginalComparison:
    """So bộ giá trị gốc của ChEMBL (value/units/relation) với bộ standard_* đã curation.

    Nhận: activity_context đã parse (chứa value, units, relation gốc), ba cột standard
    (Stage 1 lưu ở activity_value_source/unit_source/qualifier_source), dung sai.
    Trả về: OriginalComparison.
    Vì sao cần: ChEMBL đổi đơn vị và đôi khi sửa giá trị khi curation. Số dòng có thay
    đổi cho biết bao nhiêu giá trị là kết quả quy đổi của ChEMBL chứ không phải số tác
    giả công bố.
    """
    value = context.get(CONTEXT_VALUE)
    if value is None:
        return OriginalComparison(False, False, False, True)
    units = (context.get(CONTEXT_UNITS) or "").strip()
    relation = (context.get(CONTEXT_RELATION) or "").strip()
    return OriginalComparison(
        value_differs=standard_value is None or not numbers_equal(value, standard_value,
                                                                  relative_tolerance),
        unit_differs=units != (standard_unit or "").strip(),
        relation_differs=relation != (standard_relation or "").strip(),
        original_missing=False,
    )


# --------------------------------------------------------------------------- bảng Arrow (thuần)


def filter_steps(table: pa.Table | pa.RecordBatch, exact_qualifiers: Sequence[str],
                 resolved: str) -> list[tuple[str, pa.Array | pa.ChunkedArray]]:
    """Các bước lọc của tập audit, theo thứ tự, mỗi bước một mặt nạ bool.

    Nhận: bảng hoặc batch có cột theo tên vai trò và cột 'keep'; danh sách qualifier
    chính xác; giá trị status "đã xác định".
    Trả về: [(nhãn, mặt nạ)].
    Vì sao giữ từng bước riêng: báo cáo in số dòng còn lại sau mỗi bước, để thấy bộ
    lọc nào loại nhiều dữ liệu nhất.
    """
    return [
        ("dedup_keep", table.column(KEEP)),
        ("qualifier '='", pc.is_in(table.column("qualifier"),
                                   value_set=pa.array(list(exact_qualifiers), pa.string()))),
        ("protein resolved", pc.equal(table.column("protein_status"), resolved)),
        ("assay resolved", pc.equal(table.column("assay_status"), resolved)),
        ("có pActivity", pc.is_valid(table.column("p_activity"))),
    ]


def combined_mask(steps: Sequence[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Mặt nạ cộng dồn sau mỗi bước (null coi là False)."""
    result, mask = [], None
    for label, step in steps:
        step = pc.fill_null(step, False)
        mask = step if mask is None else pc.and_(mask, step)
        result.append((label, mask))
    return result


def apply_filters(
    table: pa.Table, steps: Sequence[tuple[str, Any]],
) -> tuple[pa.Table, list[tuple[str, int]]]:
    """Áp các bước lọc; trả về (bảng đã lọc, [(nhãn, số dòng còn lại)])."""
    cumulative = combined_mask(steps)
    funnel = [("tất cả", table.num_rows)]
    funnel += [(label, int(pc.sum(mask).as_py() or 0)) for label, mask in cumulative]
    return table.filter(cumulative[-1][1]), funnel


def add_assay_key(table: pa.Table) -> pa.Table:
    """Thêm cột assay_key = '<nguồn>:<ENTRYID hoặc rỗng>:<assay>'.

    Vì sao cần: ở BindingDB, ASSAYID chỉ duy nhất trong một ENTRYID (assay 1 của entry
    1025 khác assay 1 của entry 2000), nên khóa assay phải gồm cả hai. Tiền tố nguồn
    giữ cho assay của hai nguồn không bao giờ bị coi là một.
    """
    key = pc.binary_join_element_wise(table["source"], pc.fill_null(table["assay_entry_id"], ""),
                                      table["assay_id"], ":")
    return table.append_column("assay_key", key)


def publication_keys(table: pa.Table, dedup_settings: Mapping[str, Any]) -> pa.Table:
    """Các cặp (assay_key, khóa tài liệu) không lặp: 'pmid:…', 'doi:…', 'patent:…', 'doc:…'.

    Nhận: tập audit (có assay_key, source, pmid, doi, patent, document_id); config dedup
    (mẫu chuẩn hóa DOI/patent).
    Trả về: bảng hai cột assay_key, key.
    'doc:<nguồn>:<document_id>' là khóa dự phòng: tài liệu ChEMBL không có PMID/DOI (vd.
    bộ dữ liệu nộp trực tiếp) và entry BindingDB vẫn gom được các assay của chúng.
    """
    sources = [
        ("pmid", table["pmid"]),
        ("doi", normalize_doi(table["doi"], dedup_settings["doi_prefix_pattern"])),
        ("patent", normalize_patent(table["patent"], dedup_settings["patent_strip_pattern"],
                                    dedup_settings["patent_kind_code_pattern"],
                                    dedup_settings["patent_kind_code_replacement"])),
        ("doc", pc.binary_join_element_wise(table["source"], table["document_id"], ":")),
    ]
    parts = []
    for tag, values in sources:
        present = pc.fill_null(pc.greater(pc.utf8_length(values), 0), False)
        keys = pc.binary_join_element_wise(pa.scalar(tag), values, ":")
        parts.append(pa.table({"assay_key": table["assay_key"], "key": keys}).filter(present))
    return pa.concat_tables(parts).group_by(["assay_key", "key"]).aggregate([])


def add_publication(table: pa.Table, dedup_settings: Mapping[str, Any]
                    ) -> tuple[pa.Table, dict[str, int]]:
    """Thêm cột publication (số hiệu publication của assay) và trả về vài số kiểm tra.

    Trả về: (bảng thêm cột publication, {"assays", "publications", "largest_publication_assays"}).
    Số assay của publication lớn nhất được in ra để phát hiện gom nhầm (một khóa rác
    như DOI dùng chung sẽ nối hàng nghìn assay vào một publication).
    """
    keys = publication_keys(table, dedup_settings)
    ids = publication_ids(zip(keys["assay_key"].to_pylist(), keys["key"].to_pylist()))
    names = pa.array(list(ids), type=pa.string())
    numbers = pa.array(list(ids.values()), type=pa.int64())
    positions = pc.index_in(table["assay_key"], value_set=names)
    publication = pc.fill_null(pc.take(numbers, positions), PUBLICATION_MISSING)
    sizes = Counter(ids.values())
    stats = {"assays": len(ids), "publications": len(sizes),
             "largest_publication_assays": max(sizes.values(), default=0)}
    return table.append_column("publication", publication), stats


def replicate_groups(table: pa.Table, ddof: int) -> pa.Table:
    """Mỗi (assay, protein, phân tử) một dòng: số phép đo, pActivity trung bình, STD, ...

    Nhận: tập audit của một endpoint và một phạm vi (đã có assay_key, potential_duplicate,
    publication); ddof của STD.
    Trả về: bảng assay_key, protein_hash, inchikey, n, mean_p, std_p, spread_p (max − min),
    potential_duplicate (có ít nhất một dòng ChEMBL mang cờ), publication.
    Nhóm chỉ có một phép đo vẫn có mặt (cần cho mục 2 và 3); std_p của nhóm đó là null
    khi ddof = 1. Dòng không có InChIKey bị bỏ vì không xác định được "cùng phân tử".
    Endpoint không nằm trong khóa vì bảng đầu vào chỉ có một endpoint. Mọi dòng của một
    assay có cùng publication, nên lấy min là lấy đúng giá trị đó.
    """
    keyed = table.filter(pc.is_valid(table["inchikey"]))
    grouped = keyed.group_by(["assay_key", "protein_hash", "inchikey"]).aggregate([
        ("p_activity", "count"),
        ("p_activity", "mean"),
        ("p_activity", "stddev", pc.VarianceOptions(ddof=ddof)),
        ("p_activity", "min"),
        ("p_activity", "max"),
        ("potential_duplicate", "any"),
        ("publication", "min"),
    ])
    return pa.table({
        "assay_key": grouped["assay_key"],
        "protein_hash": grouped["protein_hash"],
        "inchikey": grouped["inchikey"],
        "n": grouped["p_activity_count"],
        "mean_p": grouped["p_activity_mean"],
        "std_p": grouped["p_activity_stddev"],
        "spread_p": pc.subtract(grouped["p_activity_max"], grouped["p_activity_min"]),
        "potential_duplicate": pc.fill_null(grouped["potential_duplicate_any"], False),
        "publication": pc.fill_null(grouped["publication_min"], PUBLICATION_MISSING),
    })


def with_replicates(groups: pa.Table) -> pa.Table:
    """Chỉ các nhóm có từ MIN_REPLICATES phép đo trở lên."""
    return groups.filter(pc.greater_equal(groups["n"], MIN_REPLICATES))


def replicate_summary(groups: pa.Table, percentiles: Sequence[float],
                      thresholds: Sequence[float]) -> dict[str, Any]:
    """Mục 1: số nhóm replicate, phân bố STD và số nhóm có STD <= từng ngưỡng."""
    replicated = with_replicates(groups)
    stds = values_of(replicated["std_p"])
    return {
        "groups": replicated.num_rows,
        "measurements": int(pc.sum(replicated["n"]).as_py() or 0),
        "std": summarize_values(stds, percentiles),
        "at_most": count_at_most(stds, thresholds),
    }


def replicate_diagnostics(groups: pa.Table, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Mục 7: phân bố STD replicate theo khoảng và cờ potential_duplicate của ChEMBL.

    Nhận: bảng replicate_groups, config audit.
    Trả về: số nhóm ≥ 2 phép đo, số nhóm mỗi khoảng (khoảng '= 0' xác định bằng
    max − min ≤ zero_tolerance), và với các nhóm ChEMBL: số nhóm STD ≈ 0 (< near_zero_std)
    và số nhóm còn lại, mỗi loại kèm số nhóm có cờ potential_duplicate.
    Vì sao tách STD ≈ 0: nhóm mà mọi giá trị giống hệt nhau gần như chắc là cùng một con
    số được ghi nhiều lần, không phải đo lặp thật; nếu cờ potential_duplicate tập trung ở
    đó thì đó là bằng chứng.
    """
    replicated = with_replicates(groups)
    stds = as_numpy(pc.fill_null(replicated["std_p"], 0.0), np.float64)
    zero = as_numpy(replicated["spread_p"], np.float64) <= settings["zero_tolerance"]
    counts = bin_counts(stds, zero, settings["diagnostic_bin_edges"])
    chembl = as_numpy(pc.starts_with(replicated["assay_key"], CHEMBL_PREFIX), bool)
    flagged = as_numpy(replicated["potential_duplicate"], bool)
    near = zero | (stds < settings["near_zero_std"])
    return {
        "groups": int(stds.size),
        "bins": dict(zip(bin_labels(settings["diagnostic_bin_edges"]),
                         (int(count) for count in counts))),
        "chembl_groups": int(np.count_nonzero(chembl)),
        "chembl_near_zero": int(np.count_nonzero(chembl & near)),
        "chembl_near_zero_flagged": int(np.count_nonzero(chembl & near & flagged)),
        "chembl_other": int(np.count_nonzero(chembl & ~near)),
        "chembl_other_flagged": int(np.count_nonzero(chembl & ~near & flagged)),
    }


def assay_count_distribution(groups: pa.Table, buckets: Sequence[int]) -> dict[str, Any]:
    """Mục 2: với mỗi cặp (protein, phân tử), số assay riêng biệt; đếm theo giỏ.

    Mỗi dòng của `groups` là một assay khác nhau của cặp đó, nên đếm dòng là đếm assay.
    """
    per_pair = groups.group_by(["protein_hash", "inchikey"]).aggregate([("n", "count")])
    counts = values_of(per_pair["n_count"]).astype(np.int64)
    return {"pairs": int(counts.size), "buckets": bucket_counts(counts, buckets)}


def assay_offsets(groups: pa.Table, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Mục 3: |pActivity assay A - pActivity assay B| của cùng phân tử trên cùng protein.

    Nhận: bảng replicate_groups và config audit.
    Trả về: số cặp phân tử–protein có >= 2 assay, số cặp assay, phân bố |Δ| và số cặp
    vượt từng ngưỡng.
    Quyết định:
    - Mỗi assay góp một giá trị là pActivity trung bình của các replicate trong assay đó,
      để nhiễu replicate (mục 1) không bị tính lại vào offset giữa assay.
    - Lấy trị tuyệt đối: dấu của hiệu chỉ phụ thuộc thứ tự đặt tên assay, không mang ý
      nghĩa.
    - min, max và số cặp vượt ngưỡng là chính xác; các phân vị ở giữa lấy từ histogram
      (sai số <= nửa offset_histogram_bin_width).
    - Mục 7: mỗi cặp assay còn được xếp vào nhóm (cùng/khác publication, có/không cờ
      potential_duplicate ở ít nhất một bên); mỗi nhóm có số đếm theo khoảng và phân vị.
    Cặp được dựng từ chỉ số dòng của `groups` (một danh sách duy nhất mỗi cặp phân
    tử–protein), nên |Δ|, publication và cờ của một cặp luôn lấy từ cùng hai dòng.
    """
    indexed = groups.append_column("row", pa.array(np.arange(groups.num_rows, dtype=np.int64)))
    lists = indexed.group_by(["protein_hash", "inchikey"]).aggregate([("row", "list")])
    multi = lists.filter(pc.greater_equal(pc.list_value_length(lists["row_list"]),
                                          MIN_REPLICATES))["row_list"]
    means = as_numpy(groups["mean_p"], np.float64)
    publications = as_numpy(groups["publication"], np.int64)
    flags = as_numpy(groups["potential_duplicate"], bool)
    categories = {category: new_offset_stats(settings) for category in OFFSET_CATEGORIES}
    minimum, maximum = math.inf, -math.inf
    chunk = settings["offset_lists_per_chunk"]
    for start in range(0, len(multi), chunk):
        pairs = [pair_indices(np.asarray(rows, dtype=np.int64))
                 for rows in multi.slice(start, chunk).to_pylist()]
        if not pairs:
            continue
        first = np.concatenate([pair[0] for pair in pairs])
        second = np.concatenate([pair[1] for pair in pairs])
        if first.size == 0:
            continue
        differences = np.abs(means[first] - means[second])
        same = ((publications[first] == publications[second])
                & (publications[first] != PUBLICATION_MISSING))
        flagged = flags[first] | flags[second]
        for (is_same, is_flagged), stats in categories.items():
            mask = (same == is_same) & (flagged == is_flagged)
            categories[(is_same, is_flagged)] = update_offset_stats(stats, differences[mask],
                                                                    settings)
        minimum = min(minimum, float(differences.min()))
        maximum = max(maximum, float(differences.max()))
    overall = new_offset_stats(settings)
    for stats in categories.values():
        overall = {key: overall[key] + stats[key] for key in overall}
    total = int(overall["pairs"])
    distribution = histogram_percentiles(overall["histogram"],
                                         settings["offset_histogram_bin_width"],
                                         settings["offset_percentiles"])
    if total:
        distribution[percentile_label(0)] = minimum
        distribution[percentile_label(100)] = maximum
    labels = bin_labels(settings["diagnostic_bin_edges"])
    return {
        "pairs_with_multiple_assays": len(multi),
        "assay_pairs": total,
        "abs_difference": distribution,
        "above": {str(threshold): int(count) for threshold, count
                  in zip(settings["offset_thresholds"], overall["above"])},
        "bins": dict(zip(labels, (int(count) for count in overall["bins"]))),
        "views": {name: summarize_offset_view([categories[key] for key in keys], settings)
                  for name, keys in OFFSET_VIEWS.items()},
    }


def p_activity_summary(table: pa.Table, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Mục 4: phân bố pActivity và số dòng nằm ngoài [p_activity_low, p_activity_high]."""
    values = values_of(table["p_activity"])
    return {
        "rows": int(values.size),
        "p_activity": summarize_values(values, settings["p_activity_percentiles"]),
        "below_low": int(np.count_nonzero(values < settings["p_activity_low"])),
        "above_high": int(np.count_nonzero(values > settings["p_activity_high"])),
    }


def top_proteins(table: pa.Table, groups: pa.Table, count: int) -> list[dict[str, Any]]:
    """Mục 5: `count` protein có nhiều dòng nhất, kèm STD replicate trung vị của từng protein.

    Protein được nhận diện bằng hash chuỗi (đúng construct đã đo), không bằng accession.
    STD trung vị chỉ tính trên các nhóm replicate (>= 2 phép đo) của protein đó.
    """
    sizes = table.group_by("protein_hash").aggregate([("p_activity", "count")])
    top = sizes.sort_by([("p_activity_count", "descending"),
                         ("protein_hash", "ascending")]).slice(0, count)
    hashes = top["protein_hash"].to_pylist()
    wanted = pa.array(hashes, type=pa.string())
    label_rows = (table.filter(pc.is_in(table["protein_hash"], value_set=wanted))
                  .group_by(["protein_hash", "protein_accession", "protein_name"])
                  .aggregate([("p_activity", "count")]).to_pylist())
    labels = most_common_labels(label_rows)
    replicated = with_replicates(groups)
    replicated = replicated.filter(pc.is_in(replicated["protein_hash"], value_set=wanted))
    stds: dict[str, list[float]] = defaultdict(list)
    for protein, std in zip(replicated["protein_hash"].to_pylist(),
                            replicated["std_p"].to_pylist()):
        if std is not None:
            stds[protein].append(std)
    result = []
    for rank, (protein, rows) in enumerate(zip(hashes, top["p_activity_count"].to_pylist()),
                                           start=1):
        accession, name = labels.get(protein, (None, None))
        result.append({
            "rank": rank, "protein_hash": protein, "accession": accession, "name": name,
            "rows": rows, "replicate_groups": len(stds[protein]),
            "median_replicate_std": statistics.median(stds[protein]) if stds[protein] else None,
        })
    return result


# --------------------------------------------------------------------------- đọc dữ liệu


def load_endpoint_rows(path: Path, endpoint: str, columns: Mapping[str, str],
                       keep_column: str) -> pa.Table:
    """Các cột cần cho mục 1–5 của mọi dòng thuộc một endpoint trong một file.

    Nhận: bản sao observations có cột dedup_*, endpoint, ánh xạ vai trò -> cột, tên cột keep.
    Trả về: bảng với tên cột là tên vai trò cộng 'keep'.
    Đọc theo từng endpoint để bộ nhớ chỉ chứa một endpoint mỗi lần.
    """
    names = [columns[role] for role in AUDIT_ROLES] + [keep_column]
    table = pq.read_table(path, columns=names, filters=[(columns["endpoint"], "==", endpoint)])
    return table.rename_columns([*AUDIT_ROLES, KEEP])


def parse_context(row: Mapping[str, Any]) -> dict[str, Any]:
    """activity_context (JSON) của một dòng ChEMBL; lỗi nêu rõ observation nào."""
    text = row["activity_context"]
    if text is None:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{row['observation_id']}: activity_context không phải JSON hợp lệ "
                         f"({error})") from error


def is_potential_duplicate(context: Mapping[str, Any], settings: Mapping[str, Any]) -> bool:
    """ChEMBL có đánh cờ potential_duplicate cho dòng này không (so dạng chuỗi, '1' == 1)."""
    value = context.get(settings["potential_duplicate_key"])
    return value is not None and str(value) == str(settings["potential_duplicate_value"])


def original_vs_standard(path: Path, columns: Mapping[str, str], keep_column: str,
                         exact_qualifiers: Sequence[str], settings: Mapping[str, Any]
                         ) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Mục 6: đếm dòng ChEMBL (trong tập audit) có giá trị gốc khác giá trị đã curation.

    Nhận: bản sao ChEMBL, ánh xạ cột, tên cột keep, qualifier chính xác, config audit.
    Trả về: ({endpoint: {"counts": {...}, "unit_conversions": [(gốc, standard, số dòng)]}},
    tập observation_id mà ChEMBL đánh cờ potential_duplicate).
    Đọc theo batch vì phải parse JSON từng dòng. Cùng lượt parse đó lấy luôn cờ
    potential_duplicate cho mục 7, để không phải đọc activity_context lần thứ hai.
    """
    names = [columns[role] for role in ORIGINAL_ROLES] + [keep_column]
    counters: dict[str, Counter] = defaultdict(Counter)
    conversions: dict[str, Counter] = defaultdict(Counter)
    flagged: set[str] = set()
    parquet = pq.ParquetFile(path)
    seen = 0
    for batch in parquet.iter_batches(batch_size=settings["batch_rows"], columns=names):
        batch = batch.rename_columns([*ORIGINAL_ROLES, KEEP])
        steps = filter_steps(batch, exact_qualifiers, settings["resolved_status"])
        for row in batch.filter(combined_mask(steps)[-1][1]).to_pylist():
            context = parse_context(row)
            if is_potential_duplicate(context, settings):
                flagged.add(row["observation_id"])
            result = compare_original_standard(context, row["value_source"],
                                               row["unit_source"], row["qualifier_source"],
                                               settings["original_value_relative_tolerance"])
            counter = counters[row["endpoint"]]
            counter["rows"] += 1
            for field in OriginalComparison._fields:
                counter[field] += int(getattr(result, field))
            counter["value_or_unit_differs"] += int(result.value_differs or result.unit_differs)
            if result.unit_differs:
                conversions[row["endpoint"]][(context.get(CONTEXT_UNITS),
                                              row["unit_source"])] += 1
        seen += batch.num_rows
        LOGGER.info("[mục 6] đã đọc %s/%s dòng ChEMBL", fmt_int(seen),
                    fmt_int(parquet.metadata.num_rows))
    top = settings["top_unit_conversions"]
    summary = {endpoint: {"counts": dict(counter),
                          "unit_conversions": [(original, standard, count) for
                                               (original, standard), count
                                               in conversions[endpoint].most_common(top)]}
               for endpoint, counter in counters.items()}
    LOGGER.info("[mục 6] %s dòng ChEMBL trong tập audit có cờ potential_duplicate",
                fmt_int(len(flagged)))
    return summary, flagged


# --------------------------------------------------------------------------- chạy


def audit_endpoint(paths: Mapping[str, Path], endpoint: str, config: Mapping[str, Any],
                   flagged: pa.Array) -> dict[str, Any]:
    """Mục 1–5 và 7 cho một endpoint: lọc từng nguồn, gộp, rồi tính cho từng phạm vi.

    `flagged`: observation_id mà ChEMBL đánh cờ potential_duplicate (từ mục 6).
    Publication được gom trên cả hai nguồn trước khi chia phạm vi, để cặp assay ChEMBL –
    BindingDB của cùng một bài cũng được nhận là "cùng publication" trong phạm vi 'all'.
    """
    stage = config["stage2a"]
    settings, columns = stage["audit"], stage["columns"]
    keep_column = stage["dedup_columns"]["keep"]
    exact = config["qualifiers"]["exact"]
    funnels, filtered = {}, []
    for source, path in paths.items():
        table = load_endpoint_rows(path, endpoint, columns, keep_column)
        rows, funnel = apply_filters(table, filter_steps(table, exact,
                                                         settings["resolved_status"]))
        funnel.append(("có InChIKey", int(pc.sum(pc.is_valid(rows["inchikey"])).as_py() or 0)))
        funnels[source] = funnel
        filtered.append(rows.drop_columns([KEEP, "protein_status", "assay_status"]))
    data = add_assay_key(pa.concat_tables(filtered))
    data = data.append_column("potential_duplicate",
                              pc.is_in(data["observation_id"], value_set=flagged))
    data, publications = add_publication(data, stage["dedup"])
    scopes = {}
    for scope in settings["scopes"]:
        subset = data if scope == ALL_SCOPE else data.filter(pc.equal(data["source"], scope))
        groups = replicate_groups(subset, settings["replicate_std_ddof"])
        scopes[scope] = {
            "replicates": replicate_summary(groups, settings["std_percentiles"],
                                            settings["replicate_std_thresholds"]),
            "assays_per_pair": assay_count_distribution(groups, settings["assay_count_buckets"]),
            "offsets": assay_offsets(groups, settings),
            "p_activity": p_activity_summary(subset, settings),
            "top_proteins": top_proteins(subset, groups, settings["top_proteins"]),
            "diagnostics": replicate_diagnostics(groups, settings),
        }
        LOGGER.info("[%s/%s] %s dòng, %s nhóm replicate", endpoint, scope,
                    fmt_int(subset.num_rows), fmt_int(scopes[scope]["replicates"]["groups"]))
    return {"funnel": funnels, "publications": publications, "scopes": scopes}


# --------------------------------------------------------------------------- báo cáo


def funnel_section(results: Mapping[str, Any]) -> list[str]:
    """Bảng số dòng còn lại sau từng bước lọc, theo endpoint và nguồn."""
    rows, headers = [], None
    for endpoint, result in results.items():
        for source, funnel in result["funnel"].items():
            headers = headers or ["endpoint / nguồn"] + [label for label, _ in funnel]
            rows.append([f"{endpoint} / {source}"] + [fmt_int(count) for _, count in funnel])
    return ["", "## Tập audit: số dòng còn lại sau từng bước lọc", ""] + format_table(
        headers or ["endpoint / nguồn"], rows) + [
        "Mỗi cột là số dòng còn lại sau bước đó (cộng dồn). 'có InChIKey' là số dòng dùng "
        "được cho mục 1–3 và 5."]


def replicate_section(results: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    """Mục 1: bảng replicate trong cùng assay."""
    labels = [percentile_label(q) for q in settings["std_percentiles"]]
    thresholds = settings["replicate_std_thresholds"]
    rows = []
    for endpoint, result in results.items():
        for scope, values in result["scopes"].items():
            summary = values["replicates"]
            rows.append([f"{endpoint} / {scope}", fmt_int(summary["groups"]),
                         fmt_int(summary["measurements"])]
                        + [fmt_float(summary["std"][label], 3) for label in labels]
                        + [fmt_pct(summary["at_most"][str(threshold)], summary["groups"])
                           for threshold in thresholds])
    headers = (["endpoint / phạm vi", "nhóm ≥2 phép đo", "phép đo"] + labels
               + [f"STD ≤ {fmt_float(threshold, 1)}" for threshold in thresholds])
    return ["", "## 1. Replicate trong cùng assay (STD pActivity, đơn vị log)",
            "Nhóm = (nguồn, assay, protein, endpoint, phân tử).", ""] + format_table(headers, rows)


def assay_count_section(results: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    """Mục 2: bảng số assay trên mỗi cặp phân tử–protein."""
    rows, bucket_labels = [], None
    for endpoint, result in results.items():
        for scope, values in result["scopes"].items():
            summary = values["assays_per_pair"]
            bucket_labels = list(summary["buckets"])
            cells = [f"{endpoint} / {scope}", fmt_int(summary["pairs"])]
            for label in bucket_labels:
                cells += [fmt_int(summary["buckets"][label]),
                          fmt_pct(summary["buckets"][label], summary["pairs"])]
            rows.append(cells)
    headers = ["endpoint / phạm vi", "cặp phân tử–protein"]
    for label in bucket_labels or []:
        headers += [f"{label} assay", "%"]
    return ["", "## 2. Số assay riêng biệt trên mỗi cặp (protein, endpoint, phân tử)", ""] + (
        format_table(headers, rows))


def offset_section(results: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    """Mục 3: bảng offset giữa các assay."""
    labels = [percentile_label(q) for q in settings["offset_percentiles"]]
    thresholds = settings["offset_thresholds"]
    rows = []
    for endpoint, result in results.items():
        for scope, values in result["scopes"].items():
            summary = values["offsets"]
            rows.append([f"{endpoint} / {scope}", fmt_int(summary["pairs_with_multiple_assays"]),
                         fmt_int(summary["assay_pairs"])]
                        + [fmt_float(summary["abs_difference"][label], 3) for label in labels]
                        + [fmt_pct(summary["above"][str(threshold)], summary["assay_pairs"])
                           for threshold in thresholds])
    headers = (["endpoint / phạm vi", "cặp có ≥2 assay", "cặp assay"] + labels
               + [f"> {fmt_float(threshold, 1)}" for threshold in thresholds])
    return ["", "## 3. Offset giữa các assay: |Δ pActivity| của cùng phân tử, cùng protein",
            "Mỗi assay góp pActivity trung bình của nó. min/max và tỷ lệ vượt ngưỡng chính xác; "
            f"phân vị ở giữa sai số ≤ {fmt_float(settings['offset_histogram_bin_width'] / 2, 4)}.",
            ""] + format_table(headers, rows)


def p_activity_section(results: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    """Mục 4: bảng phân bố pActivity."""
    labels = [percentile_label(q) for q in settings["p_activity_percentiles"]]
    rows = []
    for endpoint, result in results.items():
        for scope, values in result["scopes"].items():
            summary = values["p_activity"]
            rows.append([f"{endpoint} / {scope}", fmt_int(summary["rows"])]
                        + [fmt_float(summary["p_activity"][label], 2) for label in labels]
                        + [fmt_int(summary["below_low"]), fmt_int(summary["above_high"])])
    headers = (["endpoint / phạm vi", "dòng"] + labels
               + [f"< {fmt_float(settings['p_activity_low'], 0)}",
                  f"> {fmt_float(settings['p_activity_high'], 0)}"])
    return ["", "## 4. Phân bố pActivity", ""] + format_table(headers, rows)


def protein_section(results: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    """Mục 5: top protein theo số dòng, mỗi (endpoint, phạm vi) một bảng."""
    width = settings["protein_name_width"]
    lines = ["", f"## 5. Top {settings['top_proteins']} protein theo số dòng, kèm STD "
             "replicate trung vị"]
    for endpoint, result in results.items():
        for scope, values in result["scopes"].items():
            rows = [[str(item["rank"]), item["accession"] or "–",
                     (item["name"] or "–")[:width], fmt_int(item["rows"]),
                     fmt_int(item["replicate_groups"]),
                     fmt_float(item["median_replicate_std"], 3)]
                    for item in values["top_proteins"]]
            lines += ["", f"{endpoint} / {scope}"]
            lines += format_table(["#", "accession", "tên", "dòng", "nhóm replicate",
                                   "STD trung vị"], rows)
    return lines


def original_section(original: Mapping[str, Any]) -> list[str]:
    """Mục 6: bảng ChEMBL giá trị gốc so với giá trị đã curation."""
    rows = []
    for endpoint, values in original.items():
        counts = values["counts"]
        total = counts.get("rows", 0)
        rows.append([endpoint, fmt_int(total)] + [
            cell for field in ("value_differs", "unit_differs", "value_or_unit_differs",
                               "relation_differs")
            for cell in (fmt_int(counts.get(field, 0)), fmt_pct(counts.get(field, 0), total))
        ] + [fmt_int(counts.get("original_missing", 0))])
    lines = ["", "## 6. ChEMBL: giá trị gốc (value/units/relation) so với standard_* "
             "(trong tập audit)", ""]
    lines += format_table(["endpoint", "dòng", "value khác", "%", "units khác", "%",
                           "value hoặc units khác", "%", "relation khác", "%",
                           "không có value gốc"], rows)
    for endpoint, values in original.items():
        if values["unit_conversions"]:
            lines += ["", f"{endpoint}: các cặp (units gốc -> standard_units) phổ biến nhất"]
            lines += format_table(["units gốc", "standard_units", "dòng"],
                                  [[repr(original_unit), repr(standard_unit), fmt_int(count)]
                                   for original_unit, standard_unit, count
                                   in values["unit_conversions"]])
    return lines


def diagnostic_bins_section(results: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    """Mục 7.1 và 7.2: tỷ lệ nhóm replicate và cặp assay theo từng khoảng STD / |Δ|."""
    labels = bin_labels(settings["diagnostic_bin_edges"])
    std_rows, offset_rows = [], []
    for endpoint, result in results.items():
        for scope, values in result["scopes"].items():
            diagnostics, offsets = values["diagnostics"], values["offsets"]
            std_rows.append([f"{endpoint} / {scope}", fmt_int(diagnostics["groups"])]
                            + [fmt_pct(diagnostics["bins"][label], diagnostics["groups"])
                               for label in labels])
            offset_rows.append([f"{endpoint} / {scope}", fmt_int(offsets["assay_pairs"])]
                               + [fmt_pct(offsets["bins"][label], offsets["assay_pairs"])
                                  for label in labels])
    zero = f"{settings['zero_tolerance']:g}"
    return ["", "## 7. Chẩn đoán: replicate thật hay cùng một con số được ghi nhiều lần",
            "", "### 7.1. STD replicate theo khoảng (nhóm ≥ 2 phép đo trong cùng assay)",
            "'= 0': mọi giá trị trong nhóm giống hệt nhau (max − min ≤ "
            f"{zero}).", ""] + format_table(
        ["endpoint / phạm vi", "nhóm"] + labels, std_rows) + [
        "", "### 7.2. |Δ| giữa hai assay theo khoảng (cùng phân tử, cùng protein)",
        f"'= 0': |Δ| ≤ {zero}. Tỷ lệ chính xác (đếm từng cặp, không qua histogram).",
        ""] + format_table(["endpoint / phạm vi", "cặp assay"] + labels, offset_rows)


def potential_duplicate_section(results: Mapping[str, Any],
                                settings: Mapping[str, Any]) -> list[str]:
    """Mục 7.3: cờ potential_duplicate của ChEMBL trong nhóm replicate và trong cặp assay."""
    zero = bin_labels(settings["diagnostic_bin_edges"])[0]
    rows = []
    for endpoint, result in results.items():
        for scope, values in result["scopes"].items():
            # Chỉ ChEMBL có cờ này; phạm vi chỉ có BindingDB thì bỏ qua.
            if scope not in (CHEMBL, ALL_SCOPE):
                continue
            diagnostics = values["diagnostics"]
            flagged = values["offsets"]["views"][VIEW_FLAGGED]
            unflagged = values["offsets"]["views"][VIEW_UNFLAGGED]
            rows.append([
                f"{endpoint} / {scope}",
                fmt_int(diagnostics["chembl_near_zero"]),
                fmt_pct(diagnostics["chembl_near_zero_flagged"], diagnostics["chembl_near_zero"]),
                fmt_int(diagnostics["chembl_other"]),
                fmt_pct(diagnostics["chembl_other_flagged"], diagnostics["chembl_other"]),
                fmt_int(flagged["pairs"]),
                fmt_pct(flagged["pairs"], values["offsets"]["assay_pairs"]),
                fmt_pct(flagged["bins"][zero], flagged["pairs"]),
                fmt_pct(unflagged["bins"][zero], unflagged["pairs"]),
            ])
    return ["", "### 7.3. Cờ potential_duplicate của ChEMBL",
            "ChEMBL đặt cờ này khi giá trị có vẻ là trích dẫn lại một giá trị đã có trong một "
            "nguồn ChEMBL trước đó. Vì vậy cờ chủ yếu liên quan tới cặp assay khác tài liệu "
            "(cột bên phải), ít liên quan tới replicate trong cùng assay.", ""] + format_table(
        ["endpoint / phạm vi", "nhóm ChEMBL STD≈0", "% có cờ", "nhóm ChEMBL còn lại",
         "% có cờ", "cặp assay có cờ ≥1 bên", "% mọi cặp", "|Δ|=0 khi có cờ",
         "|Δ|=0 khi không cờ"], rows) + [
        "STD≈0: nhóm có STD < near_zero_std (kể cả nhóm '= 0'). Phạm vi bindingdb không có "
        "cờ này nên không in."]


def publication_views(scope: str) -> tuple[str, ...]:
    """Các nhóm cặp in ở mục 7.4 cho một phạm vi.

    'khác publication, không cờ' chỉ có nghĩa ở phạm vi chembl, nơi mọi dòng đều có thể
    mang cờ potential_duplicate.
    """
    if scope == CHEMBL:
        return VIEW_SAME, VIEW_DIFFERENT, VIEW_INDEPENDENT
    return VIEW_SAME, VIEW_DIFFERENT


def publication_section(results: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    """Mục 7.4: cặp assay cùng publication so với khác publication."""
    percentile_labels = [percentile_label(q) for q in settings["category_percentiles"]]
    threshold = str(settings["offset_thresholds"][0])
    zero, first_bin = bin_labels(settings["diagnostic_bin_edges"])[:2]
    rows = []
    for endpoint, result in results.items():
        for scope, values in result["scopes"].items():
            offsets = values["offsets"]
            for name in publication_views(scope):
                view = offsets["views"][name]
                below = view["bins"][zero] + view["bins"][first_bin]
                rows.append([f"{endpoint} / {scope} / {name}", fmt_int(view["pairs"]),
                             fmt_pct(view["pairs"], offsets["assay_pairs"]),
                             fmt_pct(view["bins"][zero], view["pairs"]),
                             fmt_pct(below, view["pairs"])]
                            + [fmt_float(view["abs_difference"][label], 3)
                               for label in percentile_labels]
                            + [fmt_pct(view["above"][threshold], view["pairs"])])
    checks = [[endpoint, fmt_int(result["publications"]["assays"]),
               fmt_int(result["publications"]["publications"]),
               fmt_int(result["publications"]["largest_publication_assays"])]
              for endpoint, result in results.items()]
    edge = fmt_float(settings["diagnostic_bin_edges"][0], 2)
    return ["", "### 7.4. Cặp assay cùng publication và khác publication",
            "Publication: assay được gom khi chung PMID, DOI, patent hoặc cùng tài liệu nguồn "
            "(document_id).",
            f"'{VIEW_INDEPENDENT}' chỉ in cho phạm vi {CHEMBL}: đó là ước lượng sạch nhất cho "
            "độ lệch giữa các phép đo độc lập trong ChEMBL. Cờ potential_duplicate chỉ có ở "
            "ChEMBL, nên ở phạm vi bindingdb dòng đó trùng với 'khác publication', còn ở phạm "
            f"vi {ALL_SCOPE} nó vẫn chứa các cặp BindingDB chưa lọc được (file JSON vẫn có đủ).",
            ""] + format_table(
        ["endpoint / phạm vi / nhóm cặp", "cặp assay", "% mọi cặp", "|Δ| = 0",
         f"|Δ| < {edge}"] + [f"|Δ| {label}" for label in percentile_labels]
        + [f"|Δ| > {fmt_float(settings['offset_thresholds'][0], 1)}"], rows) + [
        f"Phân vị lấy từ histogram (sai số ≤ "
        f"{fmt_float(settings['offset_histogram_bin_width'] / 2, 4)}); các tỷ lệ là chính xác.",
        "", "Kiểm tra cách gom publication (publication quá lớn là dấu hiệu gom nhầm):"] + (
        format_table(["endpoint", "assay", "publication", "assay của publication lớn nhất"],
                     checks))


def short_summary(results: Mapping[str, Any], original: Mapping[str, Any],
                  settings: Mapping[str, Any]) -> list[str]:
    """Vài dòng tóm tắt cho phạm vi gộp, in ở cuối để không phải mở JSON."""
    low, high = settings["replicate_std_thresholds"][0], settings["offset_thresholds"][-1]
    rows = []
    for endpoint, result in results.items():
        scope = result["scopes"].get(ALL_SCOPE) or next(iter(result["scopes"].values()))
        replicates, offsets = scope["replicates"], scope["offsets"]
        p_values = scope["p_activity"]
        rows.append([
            endpoint, fmt_int(p_values["rows"]), fmt_int(replicates["groups"]),
            fmt_float(replicates["std"]["median"], 3),
            fmt_pct(replicates["at_most"][str(low)], replicates["groups"]),
            fmt_int(offsets["pairs_with_multiple_assays"]),
            fmt_float(offsets["abs_difference"]["median"], 3),
            fmt_pct(offsets["above"][str(high)], offsets["assay_pairs"]),
            fmt_int(p_values["below_low"] + p_values["above_high"]),
            fmt_int(original.get(endpoint, {}).get("counts", {}).get("value_or_unit_differs", 0)),
        ])
    return ["", "=" * 72, f"TÓM TẮT (phạm vi '{ALL_SCOPE}' = hai nguồn gộp sau dedup)", ""] + (
        format_table(["endpoint", "dòng audit", "nhóm replicate", "STD trung vị",
                      f"STD ≤ {fmt_float(low, 1)}", "cặp ≥2 assay", "|Δ| trung vị",
                      f"|Δ| > {fmt_float(high, 1)}", "pActivity ngoài khoảng",
                      "ChEMBL gốc≠standard"], rows)) + ["=" * 72]


def load_config(path: Path) -> dict[str, Any]:
    """Đọc config.json (UTF-8)."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def setup_logging(log_path: Path) -> None:
    """Log ra màn hình và ra file (UTF-8)."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stderr),
                    logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def main() -> int:
    """Điểm vào: kiểm tra đầu vào, chạy mục 1–6, ghi báo cáo, in tóm tắt."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config(CONFIG_PATH)
    stage = config["stage2a"]
    settings = stage["audit"]
    keep_column = stage["dedup_columns"]["keep"]
    dedup_dir = ROOT / stage["paths"]["dedup_dir"]
    paths = {source: dedup_dir / name for source, name in stage["dedup"]["copy_files"].items()}

    # --- Đầu vào phải là bản sao do dedup_cross_source.py ghi (có cột dedup_keep).
    unknown_scopes = set(settings["scopes"]) - set(paths) - {ALL_SCOPE}
    if unknown_scopes:
        print(f"config stage2a.audit.scopes có phạm vi không tồn tại: {sorted(unknown_scopes)}")
        return 2
    for path in paths.values():
        if not path.is_file():
            print(f"Thiếu {path}. Chạy scripts\\dedup_cross_source.py trước.")
            return 2
        if keep_column not in pq.ParquetFile(path).schema_arrow.names:
            print(f"{path} không có cột {keep_column}. Chạy lại scripts\\dedup_cross_source.py.")
            return 2

    runs_dir = ROOT / stage["paths"]["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    stem = runs_dir / f"noise_audit_{datetime.now().strftime(TIMESTAMP_FORMAT)}"
    setup_logging(stem.with_suffix(".log"))
    started = time.monotonic()

    # --- Mục 6 đọc riêng file ChEMBL trước, lấy luôn cờ potential_duplicate cho mục 7;
    # rồi mục 1–5 và 7 theo từng endpoint.
    original, flagged = original_vs_standard(paths[CHEMBL], stage["columns"], keep_column,
                                             config["qualifiers"]["exact"], settings)
    flagged_ids = pa.array(sorted(flagged), type=pa.string())
    results = {}
    for endpoint in config["endpoints"]:
        LOGGER.info("[%s] bắt đầu; đã chạy %.0f giây", endpoint, time.monotonic() - started)
        results[endpoint] = audit_endpoint(paths, endpoint, config, flagged_ids)
    elapsed = time.monotonic() - started

    lines = ["# Stage 2a — Audit độ nhiễu (chỉ báo cáo, không lọc)",
             f"Thời gian chạy: {fmt_float(elapsed / 60, 1)} phút",
             "Phạm vi: chembl, bindingdb (phần còn lại sau dedup), all (hai nguồn gộp sau dedup)."]
    lines += funnel_section(results)
    lines += replicate_section(results, settings)
    lines += assay_count_section(results, settings)
    lines += offset_section(results, settings)
    lines += p_activity_section(results, settings)
    lines += protein_section(results, settings)
    lines += original_section(original)
    lines += diagnostic_bins_section(results, settings)
    lines += potential_duplicate_section(results, settings)
    lines += publication_section(results, settings)
    lines += short_summary(results, original, settings)
    stem.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stem.with_suffix(".json").write_text(
        json.dumps({"elapsed_seconds": elapsed, "endpoints": results,
                    "chembl_original_vs_standard": original},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n".join(short_summary(results, original, settings)))
    print(f"Báo cáo đầy đủ: {stem.with_suffix('.txt')}\nSố liệu: {stem.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
