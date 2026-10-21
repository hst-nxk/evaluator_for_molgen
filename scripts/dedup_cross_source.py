"""Stage 2a, việc 1: đánh dấu observation trùng giữa ChEMBL và BindingDB (không xóa dòng nào).

Chạy từ thư mục gốc của repo:
    python scripts\\dedup_cross_source.py            # dừng nếu output đã có
    python scripts\\dedup_cross_source.py --force    # ghi đè output cũ

Luồng dữ liệu:
    data/observations/*.parquet   (đầu vào Stage 1, chỉ đọc)
    -> lượt tổng quan: đếm giá trị source_curation, gom các dòng của ca kiểm chứng
    -> chia theo chữ cái đầu của InChIKey; mỗi phân vùng:
         ghép ứng viên trong cùng phân tử + endpoint + tài liệu (PMID/DOI/patent),
         chấm từng cặp theo protein và giá trị, gán mức căn cứ 1/2/3
    -> data/dedup/cross_source_duplicates.parquet   (một dòng cho mỗi cặp hoặc dòng lẻ)
    -> data/dedup/*_observations_dedup.parquet      (bản sao đủ 57 cột + 5 cột dedup_*)
    -> runs/stage2a/dedup_<thời điểm>.{log,txt,json} (log, báo cáo đọc được, số liệu)

Mức căn cứ (giảm dần độ chắc chắn):
    1  cột nguồn nhập liệu của một bên chỉ ra bên kia, và khớp publication + phân tử +
       protein + endpoint + giá trị                                  -> duplicate
    2  khớp publication + phân tử + protein + endpoint + giá trị
       (không có dấu hiệu nguồn nhập liệu)                           -> duplicate
    3  khớp một phần, hoặc dòng có dấu hiệu nguồn nhập liệu nhưng
       không tìm được bản tương ứng                                  -> uncertain
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
LOGGER = logging.getLogger("dedup_cross_source")
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
PARTIAL_SUFFIX = ".partial"

# Tên nguồn: đúng giá trị cột source_database do Stage 1 ghi.
CHEMBL = "chembl"
BINDINGDB = "bindingdb"
SOURCES = (CHEMBL, BINDINGDB)

# Mức căn cứ và trạng thái kết luận.
LEVEL_CURATION = 1
LEVEL_PUBLICATION = 2
LEVEL_PARTIAL = 3
LEVEL_LABELS = {
    LEVEL_CURATION: "1 — nguồn nhập liệu + publication",
    LEVEL_PUBLICATION: "2 — publication",
    LEVEL_PARTIAL: "3 — khớp một phần",
}
DUPLICATE = "duplicate"
UNCERTAIN = "uncertain"

# Kết quả so sánh từng tiêu chí của một cặp.
SEQUENCE = "sequence"
ACCESSION = "accession"
EXACT = "exact"
CLOSE = "close"
DIFFERS = "differs"
QUALIFIER_DIFFERS = "qualifier_differs"
MISSING = "missing"
NO_DOCUMENT = "none"

# Lý do ghi trong cột reason của file ánh xạ.
REASON_CURATION = "curation_source_and_publication"
REASON_PUBLICATION = "publication_match"
REASON_NO_COUNTERPART = "provenance_without_counterpart"
REASON_NO_INCHIKEY = "provenance_without_inchikey"

RECORD_PAIR = "pair"
RECORD_UNPAIRED = "unpaired"
GROUP_PREFIX = "dup:"

# Vai trò các cột cần đọc (tên cột thật lấy từ config stage2a.columns).
DEDUP_ROLES = (
    "observation_id", "source", "native_id", "curation", "pmid", "doi", "patent",
    "inchikey", "protein_hash", "protein_accession", "endpoint", "qualifier", "molar",
    "value_source", "unit_source", "assay_id", "assay_entry_id", "cx_extension",
)
# Loại tài liệu -> cột đã chuẩn hóa dùng để so (xem prepare_side).
DOCUMENT_FIELDS = {"pmid": "pmid", "doi": "doi_norm", "patent": "patent_norm"}
# Nhánh tài liệu của một cặp, để đánh giá riêng độ tin cậy của nhánh patent.
PATENT_KIND = "patent"
PUBLICATION_KINDS = frozenset({"pmid", "doi"})
BRANCH_PATENT = "chỉ patent"
BRANCH_PUBLICATION = "chỉ PMID/DOI"
BRANCH_BOTH = "patent + PMID/DOI"
BRANCH_NONE = "không chung tài liệu"
BRANCHES = (BRANCH_PUBLICATION, BRANCH_PATENT, BRANCH_BOTH, BRANCH_NONE)
# Trạng thái ghép của một dòng BindingDB có dấu hiệu nhập từ ChEMBL (báo cáo cờ |r|).
PAIRED = "có cặp"
PROVENANCE_STATES = (PAIRED, REASON_NO_COUNTERPART, REASON_NO_INCHIKEY)
# Khóa protein: chuỗi (hash SHA-256) hoặc accession UniProt.
PROTEIN_KEY_FIELDS = (("seq", "protein_hash"), ("acc", "protein_accession"))

# Các trường của mỗi bên trong file ánh xạ (tiền tố chembl_ / bindingdb_).
SIDE_FIELDS: list[tuple[str, pa.DataType]] = [
    ("observation_id", pa.string()),
    ("native_id", pa.string()),
    ("curation", pa.string()),
    ("provenance_flag", pa.bool_()),
    ("pmid", pa.string()),
    ("doi", pa.string()),
    ("patent", pa.string()),
    ("assay_id", pa.string()),
    ("assay_entry_id", pa.string()),
    ("protein_accession", pa.string()),
    ("protein_hash", pa.string()),
    ("qualifier", pa.string()),
    ("value_source", pa.string()),
    ("unit_source", pa.string()),
]
MAPPING_SCHEMA = pa.schema(
    [
        ("record_type", pa.string()),
        ("evidence_level", pa.int8()),
        ("dedup_status", pa.string()),
        ("reason", pa.string()),
        ("group_id", pa.string()),
        ("endpoint", pa.string()),
        ("molecule_inchikey", pa.string()),
        ("document_match", pa.string()),
        ("protein_match", pa.string()),
        ("value_match", pa.string()),
        ("p_activity_difference", pa.float64()),
    ]
    + [(f"{source}_{name}", dtype) for source in SOURCES for name, dtype in SIDE_FIELDS]
)
# Cờ trên từng observation (thành 5 cột dedup_* của bản sao; tên cột trong config).
FLAG_SCHEMA = pa.schema([
    ("observation_id", pa.string()),
    ("source", pa.string()),
    ("endpoint", pa.string()),
    ("group_id", pa.string()),
    ("level", pa.int8()),
    ("status", pa.string()),
    ("keep", pa.bool_()),
    ("reason", pa.string()),
])
FLAG_OUTPUT_KEYS = ("group_id", "level", "status", "keep", "reason")


class Partition(NamedTuple):
    """Một phân vùng dữ liệu theo ký tự đầu của InChIKey.

    kind: "letters" (ký tự thuộc `letters`), "other" (ký tự ngoài bảng chữ cái đã khai
    báo) hoặc "null" (dòng không có InChIKey).
    """
    label: str
    kind: str
    letters: frozenset[str]


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
    """Bảng căn cột: cột đầu căn trái, các cột sau căn phải (cột số).

    Nhận: tiêu đề và các dòng (mọi ô đã là chuỗi). Trả về: các dòng văn bản để in.
    """
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render(cells: Sequence[str]) -> str:
        parts = [cells[0].ljust(widths[0])]
        parts += [cell.rjust(width) for cell, width in zip(cells[1:], widths[1:])]
        return "  ".join(parts)

    return [render(headers), "  ".join("-" * width for width in widths)] + [
        render(row) for row in rows]


# --------------------------------------------------------------------------- chuẩn hóa (thuần)


def normalize_doi(values: pa.ChunkedArray | pa.Array, prefix_pattern: str) -> pa.ChunkedArray:
    """DOI về một dạng để so sánh: chữ thường, bỏ khoảng trắng, bỏ tiền tố URL/'doi:'.

    Nhận: cột DOI nguyên văn và regex tiền tố (config doi_prefix_pattern).
    Trả về: cột DOI đã chuẩn hóa (null giữ null).
    Vì sao cần: ChEMBL và BindingDB không ghi DOI theo cùng một kiểu
    ('10.1021/JM00001A001' và 'https://doi.org/10.1021/jm00001a001' là một bài).
    DOI không phân biệt hoa thường theo đặc tả, nên đưa về chữ thường là an toàn.
    """
    lowered = pc.utf8_trim_whitespace(pc.utf8_lower(values))
    return pc.replace_substring_regex(lowered, pattern=prefix_pattern, replacement="")


def normalize_patent(values: pa.ChunkedArray | pa.Array, strip_pattern: str,
                     kind_pattern: str, kind_replacement: str) -> pa.ChunkedArray:
    """Số patent về một dạng: chữ hoa, chỉ giữ chữ và số, bỏ mã loại văn bản ở cuối.

    Nhận: cột số patent nguyên văn và ba tham số regex trong config.
    Trả về: cột đã chuẩn hóa, vd. 'US-9718838-B2' -> 'US9718838'.
    Vì sao cần: dữ liệu patent không có PMID/DOI. Không so được số patent thì mọi phép
    đo lấy từ patent không bao giờ ghép được, kể cả khi một nguồn nhập từ nguồn kia.
    Mã loại (A1, B2, ...) bị bỏ vì hai nguồn có thể ghi hoặc không ghi phần này.
    """
    compact = pc.replace_substring_regex(pc.utf8_upper(values), pattern=strip_pattern,
                                         replacement="")
    return pc.replace_substring_regex(compact, pattern=kind_pattern,
                                      replacement=kind_replacement)


def provenance_mask(curation: pa.ChunkedArray | pa.Array,
                    markers: Sequence[str]) -> pa.ChunkedArray | pa.Array:
    """True ở dòng mà cột nguồn nhập liệu chứa một trong các dấu hiệu (không phân biệt hoa thường).

    Nhận: cột source_curation và danh sách dấu hiệu của nguồn đó (config provenance_markers).
    Trả về: cột bool, không có null (ô rỗng -> False).
    Vì sao cần: đây là căn cứ mức 1. Dòng BindingDB có Curation/DataSource chứa
    "chembl" là dòng BindingDB nhập lại từ ChEMBL, không phải phép đo độc lập.
    """
    if not markers:
        return pa.array([False] * len(curation), type=pa.bool_())
    result = None
    for marker in markers:
        hit = pc.match_substring(curation, pattern=marker, ignore_case=True)
        result = hit if result is None else pc.or_(result, hit)
    return pc.fill_null(result, False)


def value_keys(molar: pa.ChunkedArray | pa.Array, decimals: int) -> pa.ChunkedArray:
    """Khóa số nguyên của giá trị: round(-log10(mol/L) * 10^decimals).

    Nhận: cột nồng độ mol/L và số chữ số thập phân giữ lại.
    Trả về: cột int64 (null nếu không có nồng độ).
    Vì sao cần: dùng làm khóa join "cùng giá trị" mà không phải so số thực bằng dấu bằng.
    Kết luận cuối cùng vẫn do compare_values quyết định với dung sai trong config.
    """
    scaled = pc.multiply(pc.negate(pc.log10(molar)), 10 ** decimals)
    return pc.cast(pc.round(scaled), pa.int64())


# --------------------------------------------------------------------------- so sánh cặp (thuần)


def compare_documents(chembl_row: Mapping[str, Any], bindingdb_row: Mapping[str, Any],
                      document_keys: Sequence[str]) -> str:
    """Loại tài liệu mà hai dòng có chung, vd. 'pmid+doi'; không chung gì -> 'none'.

    Nhận: hai dòng đã qua prepare_side và danh sách loại tài liệu được dùng.
    Trả về: chuỗi các loại khớp nối bằng '+'.
    """
    matched = [kind for kind in document_keys
               if chembl_row[DOCUMENT_FIELDS[kind]]
               and chembl_row[DOCUMENT_FIELDS[kind]] == bindingdb_row[DOCUMENT_FIELDS[kind]]]
    return "+".join(matched) if matched else NO_DOCUMENT


def compare_proteins(chembl_hash: str | None, chembl_accession: str | None,
                     bindingdb_hash: str | None, bindingdb_accession: str | None) -> str:
    """Hai phép đo có cùng protein không: 'sequence', 'accession', 'missing' hoặc 'differs'.

    Nhận: hash chuỗi và accession của mỗi bên (None nếu Stage 1 không xác định được).
    Trả về: một trong bốn nhãn trên.
    Vì sao chấp nhận khớp accession dù chuỗi khác: hai nguồn có thể lưu hai construct
    khác nhau của cùng một protein cho cùng một phép đo (vd. P03367: 99 aa hoặc
    1.447 aa). Khi đã chung tài liệu, phân tử, endpoint và giá trị thì khác biệt đó là
    do cách ghi của nguồn, không phải hai thí nghiệm. Nhãn 'accession' được ghi lại
    riêng để có thể xem lại.
    """
    if chembl_hash and chembl_hash == bindingdb_hash:
        return SEQUENCE
    if chembl_accession and chembl_accession == bindingdb_accession:
        return ACCESSION
    if not (chembl_hash or chembl_accession) or not (bindingdb_hash or bindingdb_accession):
        return MISSING
    return DIFFERS


def compare_values(chembl_qualifier: str | None, chembl_molar: float | None,
                   bindingdb_qualifier: str | None, bindingdb_molar: float | None,
                   exact_relative_tolerance: float,
                   close_p_tolerance: float) -> tuple[str, float | None]:
    """So giá trị đã quy về mol/L; trả về (nhãn, |Δp|).

    Nhận: qualifier và nồng độ mol/L của mỗi bên, dung sai tương đối cho "bằng nhau",
    dung sai |Δp| cho "gần bằng".
    Trả về: nhãn ('exact', 'close', 'differs', 'qualifier_differs', 'missing') và
    |log10(c1) - log10(c2)| (None nếu thiếu giá trị).
    Vì sao có nhãn 'close': hai nguồn có thể làm tròn khác nhau (12,3 nM và 12 nM).
    Cặp 'close' không được coi là duplicate, chỉ ghi uncertain để người đọc tự quyết.
    """
    if chembl_molar is None or bindingdb_molar is None:
        return MISSING, None
    if chembl_molar <= 0 or bindingdb_molar <= 0:
        raise ValueError(f"nồng độ không dương: {chembl_molar!r}, {bindingdb_molar!r}")
    difference = abs(math.log10(chembl_molar) - math.log10(bindingdb_molar))
    if chembl_qualifier != bindingdb_qualifier:
        return QUALIFIER_DIFFERS, difference
    if math.isclose(chembl_molar, bindingdb_molar, rel_tol=exact_relative_tolerance,
                    abs_tol=0.0):
        return EXACT, difference
    if difference <= close_p_tolerance:
        return CLOSE, difference
    return DIFFERS, difference


def classify_pair(provenance: bool, protein_match: str,
                  value_match: str) -> tuple[int, str, str]:
    """Mức căn cứ, trạng thái và lý do của một cặp ứng viên (đã chung tài liệu).

    Nhận: có dấu hiệu nguồn nhập liệu không; kết quả so protein; kết quả so giá trị.
    Trả về: (mức 1/2/3, 'duplicate' | 'uncertain', lý do).
    Quyết định:
    - Giá trị phải bằng nhau (cùng qualifier). Chung bài báo và phân tử nhưng khác giá
      trị thì có thể là hai phép đo khác nhau trong cùng bài -> uncertain.
    - Protein phải khớp theo chuỗi hoặc accession. Thiếu protein ở một bên thì không
      phân biệt được một phép đo với các phép đo trên protein khác trong cùng bài
      (vd. một panel nhiều kinase cho cùng giá trị 10.000 nM) -> uncertain.
    - Khi đủ các điều kiện trên: có dấu hiệu nguồn nhập liệu -> mức 1, không có -> mức 2.
    """
    if value_match != EXACT:
        return LEVEL_PARTIAL, UNCERTAIN, f"value_{value_match}"
    if protein_match not in (SEQUENCE, ACCESSION):
        return LEVEL_PARTIAL, UNCERTAIN, f"protein_{protein_match}"
    if provenance:
        return LEVEL_CURATION, DUPLICATE, REASON_CURATION
    return LEVEL_PUBLICATION, DUPLICATE, REASON_PUBLICATION


def duplicate_groups(edges: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Gom các cạnh duplicate thành nhóm liên thông; trả về {observation_id: group_id}.

    Nhận: các cặp (observation ChEMBL, observation BindingDB) đã kết luận duplicate.
    Trả về: dict, group_id = 'dup:' + observation_id nhỏ nhất trong nhóm (ổn định giữa
    các lần chạy).
    Vì sao cần nhóm thay vì cặp: một phép đo ChEMBL có thể ứng với nhiều dòng BindingDB
    và ngược lại; mọi dòng nối với nhau qua các cạnh duplicate là cùng một nhóm.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            # Gốc luôn là node nhỏ nhất của nhóm, nên group_id không phụ thuộc thứ tự cạnh.
            parent[max(left_root, right_root)] = min(left_root, right_root)
    return {node: GROUP_PREFIX + find(node) for node in parent}


def drop_resolved_uncertain(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Bỏ các cặp uncertain mà cả hai đầu đều đã nằm trong một cặp duplicate.

    Nhận: các bản ghi cặp của một phân vùng. Trả về: (bản ghi giữ lại, số cặp bị bỏ).
    Vì sao cần: khi một bài báo đo cùng phân tử hai lần (giá trị A và B) và cả hai nguồn
    đều có A và B, ghép chéo sinh ra hai cặp "khác giá trị" A–B và B–A. Hai cặp này
    không mang thông tin gì thêm, vì cả bốn dòng đã được ghép đúng. Số cặp bị bỏ vẫn
    được in trong báo cáo.
    """
    resolved = {record[f"{source}_observation_id"] for record in records
                if record["dedup_status"] == DUPLICATE for source in SOURCES}
    kept = [record for record in records
            if record["dedup_status"] == DUPLICATE
            or not all(record[f"{source}_observation_id"] in resolved for source in SOURCES)]
    return kept, len(records) - len(kept)


def observation_flags(records: Iterable[Mapping[str, Any]],
                      keep_source: str) -> list[dict[str, Any]]:
    """Cờ dedup cho từng observation xuất hiện trong ít nhất một bản ghi.

    Nhận: bản ghi ánh xạ (đã có group_id), nguồn được giữ khi đếm (config keep_source).
    Trả về: danh sách dict theo FLAG_SCHEMA.
    Quy tắc:
    - status = duplicate nếu observation có ít nhất một cạnh duplicate, ngược lại uncertain.
    - level = mức mạnh nhất (số nhỏ nhất) trong các bản ghi có mặt observation.
    - keep = False chỉ khi duplicate và thuộc nguồn không được giữ. Phía được giữ luôn
      giữ mọi dòng của nó: dedup giữa hai nguồn không gộp các dòng trong cùng một nguồn.
    - reason: 'counterparts:<n>' cho duplicate; các lý do uncertain nối bằng ';'.
    """
    entries: dict[str, dict[str, Any]] = {}
    for record in records:
        for source in SOURCES:
            observation_id = record[f"{source}_observation_id"]
            if observation_id is None:
                continue
            entry = entries.setdefault(observation_id, {
                "observation_id": observation_id, "source": source,
                "endpoint": record["endpoint"], "group_id": None,
                "level": record["evidence_level"], "counterparts": set(), "reasons": set()})
            entry["level"] = min(entry["level"], record["evidence_level"])
            if record["dedup_status"] == DUPLICATE:
                other = BINDINGDB if source == CHEMBL else CHEMBL
                entry["group_id"] = record["group_id"]
                entry["counterparts"].add(record[f"{other}_observation_id"])
            else:
                entry["reasons"].add(record["reason"])
    flags = []
    for entry in entries.values():
        is_duplicate = entry["group_id"] is not None
        flags.append({
            "observation_id": entry["observation_id"],
            "source": entry["source"],
            "endpoint": entry["endpoint"],
            "group_id": entry["group_id"],
            "level": entry["level"],
            "status": DUPLICATE if is_duplicate else UNCERTAIN,
            "keep": not (is_duplicate and entry["source"] != keep_source),
            "reason": (f"counterparts:{len(entry['counterparts'])}" if is_duplicate
                       else ";".join(sorted(entry["reasons"]))),
        })
    return flags


def summarize_groups(records: Iterable[Mapping[str, Any]]) -> tuple[Counter, Counter]:
    """Đếm nhóm duplicate theo (mức, endpoint) và số nhóm có nhiều hơn một dòng ChEMBL.

    Nhận: bản ghi ánh xạ đã có group_id.
    Trả về: (Counter[(mức, endpoint)], Counter[mức] của nhóm có > 1 observation ChEMBL).
    Mức của nhóm là mức mạnh nhất trong các cạnh của nhóm.
    """
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"level": LEVEL_PARTIAL, "endpoint": None, CHEMBL: set()})
    for record in records:
        if record["dedup_status"] != DUPLICATE:
            continue
        group = groups[record["group_id"]]
        group["level"] = min(group["level"], record["evidence_level"])
        group["endpoint"] = record["endpoint"]
        group[CHEMBL].add(record[f"{CHEMBL}_observation_id"])
    by_level_endpoint = Counter((group["level"], group["endpoint"]) for group in groups.values())
    multi_chembl = Counter(group["level"] for group in groups.values() if len(group[CHEMBL]) > 1)
    return by_level_endpoint, multi_chembl


def document_branch(document_match: str | None) -> str:
    """Nhánh tài liệu của một cặp: chỉ PMID/DOI, chỉ patent, cả hai, hoặc không chung gì.

    Nhận: giá trị document_match của bản ghi ánh xạ, vd. 'pmid+doi', 'patent', 'none'.
    Trả về: một nhãn trong BRANCHES.
    Vì sao cần: dùng patent làm khóa tài liệu là giả định thêm vào, đề bài chỉ nói
    PMID/DOI. Tách riêng các cặp chỉ dựa vào patent để so tỷ lệ uncertain của chúng với
    nhánh PMID/DOI: nếu cao hơn hẳn thì nhánh patent đang ghép nhầm.
    """
    kinds = set() if document_match in (None, NO_DOCUMENT) else set(document_match.split("+"))
    has_patent = PATENT_KIND in kinds
    has_publication = bool(kinds & PUBLICATION_KINDS)
    if has_patent and has_publication:
        return BRANCH_BOTH
    if has_patent:
        return BRANCH_PATENT
    if has_publication:
        return BRANCH_PUBLICATION
    return BRANCH_NONE


def branch_counts(records: Iterable[Mapping[str, Any]]) -> Counter:
    """Đếm bản ghi cặp theo (nhánh tài liệu, mức, trạng thái, lý do); bỏ qua dòng lẻ."""
    return Counter((document_branch(record["document_match"]), record["evidence_level"],
                    record["dedup_status"], record["reason"])
                   for record in records if record["record_type"] == RECORD_PAIR)


def relative_stereo_mask(extensions: pa.ChunkedArray | pa.Array,
                         pattern: str) -> pa.ChunkedArray | pa.Array:
    """True ở dòng có cờ lập thể tương đối 'r' trong khối CXSMILES; ô rỗng -> False.

    Nhận: cột molecule_cx_extension, regex của cờ (config relative_stereo_pattern).
    Trả về: cột bool không có null.
    """
    return pc.fill_null(pc.match_substring_regex(extensions, pattern=pattern), False)


def provenance_stereo_counts(bindingdb: pa.Table, records: Iterable[Mapping[str, Any]],
                             keyed: bool, pattern: str) -> Counter:
    """Đếm dòng BindingDB có dấu hiệu nhập từ ChEMBL theo (trạng thái ghép, có cờ |r|).

    Nhận: bảng BindingDB của một phân vùng (đã qua prepare_side), bản ghi ánh xạ của
    phân vùng đó, phân vùng có InChIKey hay không, regex cờ 'r'.
    Trả về: Counter[(trạng thái trong PROVENANCE_STATES, True/False)].
    Vì sao cần: RDKit bỏ qua cờ |r|, nên InChIKey phía BindingDB có thể khác InChIKey của
    cùng chất phía ChEMBL và cặp bị sót. So tỷ lệ |r| giữa dòng có cặp và dòng không có
    cặp cho biết |r| có phải là một nguyên nhân gây sót hay không. Chỉ đếm, không đổi
    cách ghép.
    """
    paired = {record[f"{BINDINGDB}_observation_id"] for record in records
              if record["record_type"] == RECORD_PAIR}
    flagged = bindingdb.filter(bindingdb["provenance_flag"])
    relative = relative_stereo_mask(flagged["cx_extension"], pattern).to_pylist()
    unpaired_state = REASON_NO_COUNTERPART if keyed else REASON_NO_INCHIKEY
    counts: Counter = Counter()
    for observation_id, has_flag in zip(flagged["observation_id"].to_pylist(), relative):
        counts[(PAIRED if observation_id in paired else unpaired_state, has_flag)] += 1
    return counts


def branch_totals(branches: Mapping[tuple, int]) -> dict[str, dict[str, Any]]:
    """Tổng hợp branch_counts thành số cặp mỗi nhánh: tổng, mức 1, mức 2, uncertain, lý do.

    Nhận: Counter[(nhánh, mức, trạng thái, lý do)].
    Trả về: {nhánh: {"pairs", "level_1", "level_2", "uncertain", "uncertain_reasons"}}
    cho mọi nhánh trong BRANCHES (nhánh không có cặp -> toàn 0).
    """
    totals: dict[str, dict[str, Any]] = {
        branch: {"pairs": 0, "level_1": 0, "level_2": 0, "uncertain": 0,
                 "uncertain_reasons": Counter()}
        for branch in BRANCHES}
    for (branch, level, status, reason), count in branches.items():
        entry = totals[branch]
        entry["pairs"] += count
        if status == UNCERTAIN:
            entry["uncertain"] += count
            entry["uncertain_reasons"][reason] += count
        elif level == LEVEL_CURATION:
            entry["level_1"] += count
        elif level == LEVEL_PUBLICATION:
            entry["level_2"] += count
    return totals


def stereo_totals(stereo: Mapping[tuple[str, bool], int]) -> dict[str, dict[str, int]]:
    """Số dòng và số dòng có cờ |r| cho mỗi trạng thái ghép trong PROVENANCE_STATES."""
    return {state: {"rows": stereo.get((state, True), 0) + stereo.get((state, False), 0),
                    "with_r": stereo.get((state, True), 0)}
            for state in PROVENANCE_STATES}


def overlap_diagnosis(bindingdb_rows: Sequence[Mapping[str, Any]],
                      chembl_rows: Sequence[Mapping[str, Any]],
                      document_keys: Sequence[str]) -> list[tuple[str, int]]:
    """Hai tập dòng nhỏ có chung những gì: tài liệu, phân tử, protein, giá trị.

    Nhận: các dòng (đã qua prepare_side) của ca kiểm chứng ở mỗi nguồn.
    Trả về: danh sách (mô tả, số lượng) để in.
    Vì sao cần: khi ca kiểm chứng không bắt được, bảng này chỉ ra tiêu chí nào làm hỏng
    việc ghép (vd. không chung PMID, hay InChIKey khác nhau giữa hai nguồn).
    """
    def values(rows: Sequence[Mapping[str, Any]], *fields: str) -> set:
        return {tuple(row[field] for field in fields) for row in rows
                if all(row[field] is not None for field in fields)}

    def protein_keys(rows: Sequence[Mapping[str, Any]]) -> set:
        return {(tag, row[field]) for row in rows for tag, field in PROTEIN_KEY_FIELDS
                if row[field]}

    lines = [
        ("observation BindingDB trong khoảng", len(bindingdb_rows)),
        ("observation ChEMBL trong khoảng", len(chembl_rows)),
        ("dòng BindingDB có dấu hiệu nguồn nhập liệu",
         sum(1 for row in bindingdb_rows if row["provenance_flag"])),
    ]
    for kind in document_keys:
        field = DOCUMENT_FIELDS[kind]
        lines.append((f"{kind} chung", len(values(bindingdb_rows, field)
                                              & values(chembl_rows, field))))
    lines += [
        ("InChIKey chung", len(values(bindingdb_rows, "inchikey")
                               & values(chembl_rows, "inchikey"))),
        ("(InChIKey, endpoint) chung", len(values(bindingdb_rows, "inchikey", "endpoint")
                                           & values(chembl_rows, "inchikey", "endpoint"))),
        ("khóa protein chung (chuỗi hoặc accession)",
         len(protein_keys(bindingdb_rows) & protein_keys(chembl_rows))),
        ("(InChIKey, endpoint, qualifier, giá trị) chung",
         len(values(bindingdb_rows, "inchikey", "endpoint", "qualifier", "value_key")
             & values(chembl_rows, "inchikey", "endpoint", "qualifier", "value_key"))),
    ]
    return lines


def reservoir_add(sample: list[Any], seen: int, item: Any, size: int,
                  rng: random.Random) -> None:
    """Lấy mẫu ngẫu nhiên đều `size` phần tử từ một luồng (cập nhật `sample` tại chỗ).

    Nhận: mẫu hiện có, số phần tử đã thấy trước item này, item, kích thước mẫu, bộ sinh
    ngẫu nhiên có seed.
    Vì sao cần: chọn ví dụ cho báo cáo mà không phải giữ mọi cặp trong bộ nhớ. Seed cố
    định và thứ tự duyệt cố định nên lần chạy nào cũng ra cùng các ví dụ.
    """
    if len(sample) < size:
        sample.append(item)
        return
    slot = rng.randint(0, seen)
    if slot < size:
        sample[slot] = item


# --------------------------------------------------------------------------- đọc dữ liệu


def build_partitions(alphabet: str, letters_per_partition: int) -> list[Partition]:
    """Danh sách phân vùng: các nhóm chữ cái, rồi 'other', rồi 'null'.

    Nhận: bảng chữ cái của ký tự đầu InChIKey và số chữ cái mỗi phân vùng.
    Trả về: danh sách Partition phủ kín mọi dòng (không dòng nào bị bỏ sót).
    Vì sao chia phân vùng: hai dòng chỉ có thể trùng khi cùng InChIKey, nên xử lý từng
    phân vùng riêng cho kết quả giống hệt xử lý một lần, mà bộ nhớ chỉ cần cho một
    phần nhỏ dữ liệu. InChIKey là mã băm nên các chữ cái đầu phân bố khá đều.
    """
    partitions = [
        Partition(alphabet[start:start + letters_per_partition], "letters",
                  frozenset(alphabet[start:start + letters_per_partition]))
        for start in range(0, len(alphabet), letters_per_partition)]
    partitions.append(Partition("other", "other", frozenset(alphabet)))
    partitions.append(Partition("null", "null", frozenset()))
    return partitions


def partition_mask(keys: pa.Array, partition: Partition) -> pa.Array:
    """Mặt nạ các dòng thuộc phân vùng, tính trên cột InChIKey của một batch."""
    if partition.kind == "null":
        return pc.is_null(keys)
    first = pc.utf8_slice_codeunits(keys, 0, 1)
    inside = pc.is_in(first, value_set=pa.array(sorted(partition.letters), type=pa.string()))
    if partition.kind == "letters":
        return pc.fill_null(inside, False)
    return pc.fill_null(pc.and_(pc.is_valid(keys), pc.invert(inside)), False)


def read_partition(path: Path, columns: Mapping[str, str], partition: Partition,
                   batch_rows: int) -> pa.Table:
    """Đọc các cột cần cho dedup của những dòng thuộc một phân vùng.

    Nhận: file observations, ánh xạ vai trò -> tên cột, phân vùng, số dòng mỗi batch.
    Trả về: bảng Arrow với tên cột là tên vai trò (DEDUP_ROLES).
    Đọc theo batch và lọc ngay, nên bộ nhớ chỉ giữ phần thuộc phân vùng.
    """
    names = [columns[role] for role in DEDUP_ROLES]
    parquet = pq.ParquetFile(path)
    schema = pa.schema([parquet.schema_arrow.field(name) for name in names])
    pieces = []
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=names):
        pieces.append(batch.filter(partition_mask(batch.column(columns["inchikey"]),
                                                  partition)))
    return pa.Table.from_batches(pieces, schema=schema).rename_columns(list(DEDUP_ROLES))


def prepare_side(table: pa.Table, source: str, settings: Mapping[str, Any]) -> pa.Table:
    """Thêm các cột suy ra dùng để ghép: DOI/patent chuẩn hóa, cờ nguồn nhập liệu, khóa giá trị.

    Nhận: bảng của một nguồn (tên cột = vai trò), tên nguồn, config dedup.
    Trả về: bảng thêm doi_norm, patent_norm, provenance_flag, value_key, has_protein.
    """
    table = table.append_column("doi_norm", normalize_doi(table["doi"],
                                                          settings["doi_prefix_pattern"]))
    table = table.append_column("patent_norm", normalize_patent(
        table["patent"], settings["patent_strip_pattern"],
        settings["patent_kind_code_pattern"], settings["patent_kind_code_replacement"]))
    table = table.append_column("provenance_flag", provenance_mask(
        table["curation"], settings["provenance_markers"][source]))
    table = table.append_column("value_key", value_keys(table["molar"],
                                                        settings["value_key_decimals"]))
    return table.append_column("has_protein", pc.or_(pc.is_valid(table["protein_hash"]),
                                                     pc.is_valid(table["protein_accession"])))


def explode_keys(table: pa.Table, key_name: str,
                 sources: Sequence[tuple[str, str]]) -> pa.Table:
    """Một dòng cho mỗi giá trị khóa không rỗng, gắn nhãn loại: 'pmid:123', 'doi:10.1/x', ...

    Nhận: bảng, tên cột khóa mới, danh sách (nhãn, cột nguồn).
    Trả về: bảng dài hơn, thêm cột `key_name`.
    Vì sao cần: "chung tài liệu" nghĩa là chung PMID HOẶC chung DOI HOẶC chung patent.
    Trải mỗi loại thành một dòng thì một phép join bằng nhau bắt được mọi trường hợp.
    """
    parts = []
    for tag, column in sources:
        values = table[column]
        present = pc.fill_null(pc.greater(pc.utf8_length(values), 0), False)
        subset = table.filter(present)
        tagged = pc.binary_join_element_wise(pa.scalar(tag), subset[column], pa.scalar(":"))
        parts.append(subset.append_column(key_name, tagged))
    return pa.concat_tables(parts)


def candidate_pairs(chembl: pa.Table, bindingdb: pa.Table,
                    keys: Sequence[str]) -> set[tuple[str, str]]:
    """Các cặp (observation ChEMBL, observation BindingDB) bằng nhau trên mọi cột `keys`.

    Nhận: hai bảng đã có các cột khóa. Trả về: tập cặp observation_id, không lặp.
    Ô null không bao giờ khớp (quy ước join của Arrow).
    """
    left = chembl.select(["observation_id", *keys]).rename_columns(["c_id", *keys])
    right = bindingdb.select(["observation_id", *keys]).rename_columns(["b_id", *keys])
    joined = left.join(right, keys=list(keys), join_type="inner")
    unique = joined.select(["c_id", "b_id"]).group_by(["c_id", "b_id"]).aggregate([])
    return set(zip(unique["c_id"].to_pylist(), unique["b_id"].to_pylist()))


def find_candidates(chembl: pa.Table, bindingdb: pa.Table,
                    document_keys: Sequence[str]) -> set[tuple[str, str]]:
    """Mọi cặp ứng viên của một phân vùng: luôn chung phân tử, endpoint và tài liệu.

    Nhận: hai bảng đã qua prepare_side, danh sách loại tài liệu.
    Trả về: tập cặp observation_id.
    Hai phép join:
    1. Chung tài liệu + chung protein (chuỗi hoặc accession). Bắt các cặp duplicate và
       các cặp khác giá trị trong cùng bài, cùng protein.
    2. Chung tài liệu + cùng qualifier + cùng giá trị, khi một bên thiếu protein. Bắt
       các cặp "có thể trùng nhưng không kiểm được protein".
    Không join "chung tài liệu" trần (không kèm protein hay giá trị): với một panel
    hàng trăm protein trong cùng bài, phép join đó sinh ra hàng trăm nghìn cặp vô nghĩa.
    """
    document_sources = [(kind, DOCUMENT_FIELDS[kind]) for kind in document_keys]
    chembl_docs = explode_keys(chembl, "doc_key", document_sources)
    bindingdb_docs = explode_keys(bindingdb, "doc_key", document_sources)
    # --- Join 1: tài liệu + protein.
    protein_join_keys = ["inchikey", "endpoint", "doc_key", "protein_key"]
    pairs = candidate_pairs(explode_keys(chembl_docs, "protein_key", PROTEIN_KEY_FIELDS),
                            explode_keys(bindingdb_docs, "protein_key", PROTEIN_KEY_FIELDS),
                            protein_join_keys)
    # --- Join 2: tài liệu + giá trị, chỉ khi ít nhất một bên thiếu protein.
    value_join_keys = ["inchikey", "endpoint", "doc_key", "qualifier", "value_key"]
    chembl_missing = chembl_docs.filter(pc.invert(chembl_docs["has_protein"]))
    bindingdb_missing = bindingdb_docs.filter(pc.invert(bindingdb_docs["has_protein"]))
    pairs |= candidate_pairs(chembl_missing, bindingdb_docs, value_join_keys)
    pairs |= candidate_pairs(chembl_docs, bindingdb_missing, value_join_keys)
    return pairs


def rows_by_id(table: pa.Table, ids: set[str]) -> dict[str, dict[str, Any]]:
    """Các dòng có observation_id thuộc `ids`, dạng {observation_id: dict}."""
    if not ids:
        return {}
    wanted = pa.array(sorted(ids), type=pa.string())
    subset = table.filter(pc.is_in(table["observation_id"], value_set=wanted))
    return {row["observation_id"]: row for row in subset.to_pylist()}


# --------------------------------------------------------------------------- bản ghi ánh xạ


def side_fields(source: str, row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Các trường của một bên (tiền tố chembl_/bindingdb_); bên vắng mặt -> None hết."""
    return {f"{source}_{name}": (None if row is None else row[name]) for name, _ in SIDE_FIELDS}


def evaluate_pair(chembl_row: Mapping[str, Any], bindingdb_row: Mapping[str, Any],
                  settings: Mapping[str, Any]) -> dict[str, Any]:
    """Chấm một cặp ứng viên và trả về một bản ghi ánh xạ (chưa có group_id).

    Nhận: dòng ChEMBL, dòng BindingDB (đã qua prepare_side), config dedup.
    Trả về: dict theo MAPPING_SCHEMA.
    """
    try:
        value_match, difference = compare_values(
            chembl_row["qualifier"], chembl_row["molar"], bindingdb_row["qualifier"],
            bindingdb_row["molar"], settings["value_exact_relative_tolerance"],
            settings["value_close_p_tolerance"])
    except ValueError as error:
        raise ValueError(f"cặp {chembl_row['observation_id']} / "
                         f"{bindingdb_row['observation_id']}: {error}") from error
    protein_match = compare_proteins(chembl_row["protein_hash"], chembl_row["protein_accession"],
                                     bindingdb_row["protein_hash"],
                                     bindingdb_row["protein_accession"])
    provenance = bool(chembl_row["provenance_flag"] or bindingdb_row["provenance_flag"])
    level, status, reason = classify_pair(provenance, protein_match, value_match)
    return {
        "record_type": RECORD_PAIR,
        "evidence_level": level,
        "dedup_status": status,
        "reason": reason,
        "group_id": None,
        "endpoint": chembl_row["endpoint"],
        "molecule_inchikey": chembl_row["inchikey"],
        "document_match": compare_documents(chembl_row, bindingdb_row,
                                            settings["document_keys"]),
        "protein_match": protein_match,
        "value_match": value_match,
        "p_activity_difference": difference,
        **side_fields(CHEMBL, chembl_row),
        **side_fields(BINDINGDB, bindingdb_row),
    }


def unpaired_record(source: str, row: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """Bản ghi uncertain cho một dòng có dấu hiệu nguồn nhập liệu mà không có cặp nào."""
    return {
        "record_type": RECORD_UNPAIRED,
        "evidence_level": LEVEL_PARTIAL,
        "dedup_status": UNCERTAIN,
        "reason": reason,
        "group_id": None,
        "endpoint": row["endpoint"],
        "molecule_inchikey": row["inchikey"],
        "document_match": None,
        "protein_match": None,
        "value_match": None,
        "p_activity_difference": None,
        **side_fields(CHEMBL, row if source == CHEMBL else None),
        **side_fields(BINDINGDB, row if source == BINDINGDB else None),
    }


def unpaired_provenance_records(tables: Mapping[str, pa.Table], paired: set[str],
                                reason: str) -> list[dict[str, Any]]:
    """Bản ghi cho các dòng có dấu hiệu nguồn nhập liệu nhưng không nằm trong cặp nào."""
    records = []
    for source, table in tables.items():
        flagged = table.filter(table["provenance_flag"])
        for row in flagged.to_pylist():
            if row["observation_id"] not in paired:
                records.append(unpaired_record(source, row, reason))
    return records


def match_partition(chembl: pa.Table, bindingdb: pa.Table, keyed: bool,
                    settings: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Toàn bộ việc ghép của một phân vùng: ứng viên -> chấm điểm -> nhóm.

    Nhận: hai bảng đã qua prepare_side; keyed=False cho phân vùng không có InChIKey;
    config dedup.
    Trả về: (bản ghi ánh xạ đã có group_id, số cặp uncertain bị bỏ vì đã giải quyết).
    """
    records: list[dict[str, Any]] = []
    suppressed = 0
    if keyed:
        pairs = find_candidates(chembl, bindingdb, settings["document_keys"])
        chembl_rows = rows_by_id(chembl, {pair[0] for pair in pairs})
        bindingdb_rows = rows_by_id(bindingdb, {pair[1] for pair in pairs})
        records = [evaluate_pair(chembl_rows[c_id], bindingdb_rows[b_id], settings)
                   for c_id, b_id in sorted(pairs)]
        records, suppressed = drop_resolved_uncertain(records)
    # --- Dòng có dấu hiệu nguồn nhập liệu nhưng không có cặp: ghi lại, không đoán.
    paired = {record[f"{source}_observation_id"] for record in records for source in SOURCES}
    reason = REASON_NO_COUNTERPART if keyed else REASON_NO_INCHIKEY
    records += unpaired_provenance_records({CHEMBL: chembl, BINDINGDB: bindingdb}, paired,
                                           reason)
    # --- Gán group_id cho các cạnh duplicate.
    groups = duplicate_groups(
        (record[f"{CHEMBL}_observation_id"], record[f"{BINDINGDB}_observation_id"])
        for record in records if record["dedup_status"] == DUPLICATE)
    for record in records:
        if record["dedup_status"] == DUPLICATE:
            record["group_id"] = groups[record[f"{CHEMBL}_observation_id"]]
    return records, suppressed


# --------------------------------------------------------------------------- các lượt đọc


def id_range(bounds: Sequence[int]) -> set[str]:
    """Tập ID dạng chuỗi trong khoảng đóng [đầu, cuối] (ID phép đo của Stage 1 là chuỗi)."""
    start, end = bounds
    return {str(number) for number in range(start, end + 1)}


def overview_pass(path: Path, columns: Mapping[str, str], known_ids: set[str],
                  batch_rows: int) -> tuple[Counter, pa.Table]:
    """Một lượt đọc mỗi file: đếm giá trị source_curation và lấy các dòng của ca kiểm chứng.

    Nhận: file observations, ánh xạ cột, tập ID phép đo của ca kiểm chứng, cỡ batch.
    Trả về: (Counter giá trị source_curation, bảng các dòng có ID trong ca kiểm chứng).
    """
    names = [columns[role] for role in DEDUP_ROLES]
    wanted = pa.array(sorted(known_ids), type=pa.string())
    counts: Counter = Counter()
    pieces = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=names):
        for item in pc.value_counts(batch.column(columns["curation"])).to_pylist():
            counts[item["values"]] += item["counts"]
        pieces.append(batch.filter(pc.is_in(batch.column(columns["native_id"]),
                                            value_set=wanted)))
    schema = pa.schema([parquet.schema_arrow.field(name) for name in names])
    known = pa.Table.from_batches(pieces, schema=schema).rename_columns(list(DEDUP_ROLES))
    return counts, known


def write_flagged_copy(source_path: Path, output_path: Path, flags: pa.Table,
                       columns: Mapping[str, str], output_columns: Mapping[str, str],
                       batch_rows: int, label: str) -> int:
    """Ghi bản sao file observations, thêm 5 cột dedup_*; file gốc không bị đụng tới.

    Nhận: file gốc, file đích (.partial), bảng cờ của nguồn này, tên cột, cỡ batch, nhãn log.
    Trả về: số dòng đã ghi.
    Dòng không có cờ: dedup_keep = True, các cột còn lại null.
    """
    parquet = pq.ParquetFile(source_path)
    new_fields = [pa.field(output_columns[key], FLAG_SCHEMA.field(key).type)
                  for key in FLAG_OUTPUT_KEYS]
    schema = parquet.schema_arrow
    for field in new_fields:
        schema = schema.append(field)
    flag_ids = flags["observation_id"].combine_chunks()
    written = 0
    writer = pq.ParquetWriter(output_path, schema)
    try:
        for batch in parquet.iter_batches(batch_size=batch_rows):
            table = pa.Table.from_batches([batch])
            positions = pc.index_in(table[columns["observation_id"]], value_set=flag_ids)
            for key, field in zip(FLAG_OUTPUT_KEYS, new_fields):
                values = pc.take(flags[key], positions)
                if key == "keep":
                    values = pc.fill_null(values, True)
                table = table.append_column(field, values)
            writer.write_table(table)
            written += table.num_rows
        LOGGER.info("[%s] bản sao: %s dòng", label, fmt_int(written))
    finally:
        writer.close()
    return written


# --------------------------------------------------------------------------- báo cáo


def flag_counts(flags: pa.Table) -> list[dict[str, Any]]:
    """Số observation theo (nguồn, endpoint, status, level, keep), dạng list dict nhỏ."""
    grouped = flags.group_by(["source", "endpoint", "status", "level", "keep"]).aggregate(
        [("observation_id", "count")])
    return grouped.to_pylist()


def count_flags(rows: Sequence[Mapping[str, Any]], **conditions: Any) -> int:
    """Tổng số observation trong flag_counts thỏa mọi điều kiện cột = giá trị."""
    return sum(row["observation_id_count"] for row in rows
               if all(row[key] == value for key, value in conditions.items()))


def curation_section(curation_counts: Mapping[str, Counter], settings: Mapping[str, Any],
                     shown: int) -> list[str]:
    """Bảng các giá trị source_curation phổ biến nhất của mỗi nguồn, kèm cờ khớp dấu hiệu."""
    lines = ["", "## Giá trị cột nguồn nhập liệu (source_curation)"]
    for source in SOURCES:
        counts = curation_counts[source]
        markers = settings["provenance_markers"][source]
        rows = []
        for value, count in counts.most_common(shown):
            matched = value is not None and any(marker.lower() in value.lower()
                                                for marker in markers)
            rows.append([repr(value), fmt_int(count), "có" if matched else ""])
        lines += ["", f"{source} — dấu hiệu tìm: {markers}; "
                      f"{fmt_int(len(counts))} giá trị khác nhau"]
        lines += format_table(["giá trị", "số observation", "khớp dấu hiệu"], rows)
    return lines


def level_section(group_counts: Counter, multi_chembl: Counter,
                  counts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Bảng số nhóm duplicate và số observation theo mức căn cứ."""
    rows = []
    for level in (LEVEL_CURATION, LEVEL_PUBLICATION):
        groups = sum(count for (group_level, _), count in group_counts.items()
                     if group_level == level)
        rows.append([LEVEL_LABELS[level], fmt_int(groups), fmt_int(multi_chembl[level]),
                     fmt_int(count_flags(counts, status=DUPLICATE, level=level, source=CHEMBL)),
                     fmt_int(count_flags(counts, status=DUPLICATE, level=level,
                                         source=BINDINGDB))])
    return ["", "## Nhóm duplicate theo mức căn cứ", ""] + format_table(
        ["mức", "số nhóm", "nhóm có >1 dòng ChEMBL", "observation ChEMBL",
         "observation BindingDB"], rows) + [
        "Mức của một observation là mức mạnh nhất trong các cặp chứa nó.",
        "Nhóm có >1 dòng ChEMBL: một dòng BindingDB khớp nhiều dòng ChEMBL giống hệt nhau; "
        "mọi dòng ChEMBL đều được giữ."]


def uncertain_section(record_counts: Counter, suppressed: int) -> list[str]:
    """Bảng số bản ghi uncertain (mức 3) theo lý do."""
    rows = [[reason, fmt_int(count)] for (level, status, reason), count
            in sorted(record_counts.items(), key=lambda item: -item[1])
            if status == UNCERTAIN]
    return ["", "## Bản ghi uncertain (mức 3) theo lý do", ""] + format_table(
        ["lý do", "số bản ghi"], rows) + [
        f"Cặp uncertain bị bỏ vì cả hai đầu đã nằm trong cặp duplicate: {fmt_int(suppressed)}"]


def branch_section(totals: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Bảng cặp ứng viên theo nhánh tài liệu (chỉ PMID/DOI, chỉ patent, cả hai)."""
    rows, reason_rows = [], []
    for branch in BRANCHES:
        entry = totals[branch]
        if branch == BRANCH_NONE and not entry["pairs"]:
            continue
        rows.append([branch, fmt_int(entry["pairs"]), fmt_int(entry["level_1"]),
                     fmt_int(entry["level_2"]), fmt_int(entry["uncertain"]),
                     fmt_pct(entry["uncertain"], entry["pairs"])])
        for reason, count in sorted(entry["uncertain_reasons"].items(),
                                    key=lambda item: (-item[1], item[0])):
            reason_rows.append([branch, reason, fmt_int(count), fmt_pct(count, entry["pairs"])])
    lines = ["", "## Cặp ứng viên theo loại tài liệu chung", ""]
    lines += format_table(["nhánh", "số cặp", "duplicate mức 1", "duplicate mức 2",
                           "uncertain", "% uncertain"], rows)
    lines += ["", "Lý do uncertain theo nhánh (% tính trên số cặp của nhánh):"]
    lines += format_table(["nhánh", "lý do", "số cặp", "% cặp của nhánh"], reason_rows)
    return lines + [
        "Đếm theo cặp, không theo nhóm; không gồm cặp uncertain đã bị bỏ vì cả hai đầu "
        "đã là duplicate.",
        "Nếu nhánh 'chỉ patent' có % uncertain (nhất là value_differs, protein_*) cao hơn "
        "hẳn nhánh 'chỉ PMID/DOI', các cặp tìm qua patent đáng nghi."]


def stereo_section(totals: Mapping[str, Mapping[str, int]]) -> list[str]:
    """Bảng dòng BindingDB nhập từ ChEMBL theo trạng thái ghép và cờ |r|."""
    rows = [[state, fmt_int(entry["rows"]), fmt_int(entry["with_r"]),
             fmt_pct(entry["with_r"], entry["rows"])] for state, entry in totals.items()]
    lines = ["", "## Dòng BindingDB nhập từ ChEMBL, theo cờ lập thể tương đối |r|", ""]
    lines += format_table(["trạng thái ghép", "số dòng", "có |r|", "% có |r|"], rows)
    return lines + [
        "Chỉ gồm dòng BindingDB có dấu hiệu nhập từ ChEMBL (provenance_markers).",
        "Nếu % có |r| ở dòng không có cặp cao hơn hẳn dòng có cặp, |r| là một nguyên nhân "
        "làm dedup bỏ sót.",
        "Số dòng có |r| mà không có cặp là giới hạn trên của phần bỏ sót do |r|, không phải "
        "số chính xác: dòng không có cặp còn có thể do nguyên nhân khác (ChEMBL không có "
        "phép đo đó trong Stage 1, khác tài liệu, khác protein)."]


def affected_section(counts: Sequence[Mapping[str, Any]], totals: Mapping[str, int],
                     keep_source: str) -> list[str]:
    """Bảng số observation bị ảnh hưởng ở mỗi nguồn."""
    rows = []
    for source in SOURCES:
        total = totals[source]
        duplicate = count_flags(counts, source=source, status=DUPLICATE)
        dropped = count_flags(counts, source=source, status=DUPLICATE, keep=False)
        uncertain = count_flags(counts, source=source, status=UNCERTAIN)
        rows.append([source, fmt_int(total), fmt_int(duplicate), fmt_pct(duplicate, total),
                     fmt_int(dropped), fmt_int(uncertain), fmt_pct(uncertain, total)])
    return ["", "## Observation bị ảnh hưởng ở mỗi nguồn", ""] + format_table(
        ["nguồn", "tổng", "duplicate", "% tổng", "không đếm lại (keep=False)", "uncertain",
         "% tổng"], rows) + [
        f"Trong mỗi nhóm duplicate, dòng {keep_source} được giữ; dòng của nguồn kia có "
        "dedup_keep = False. Không dòng nào bị xóa."]


def endpoint_section(group_counts: Counter, counts: Sequence[Mapping[str, Any]],
                     endpoints: Sequence[str]) -> list[str]:
    """Bảng phân bố theo endpoint."""
    rows = []
    for endpoint in endpoints:
        rows.append([
            endpoint,
            fmt_int(group_counts[(LEVEL_CURATION, endpoint)]),
            fmt_int(group_counts[(LEVEL_PUBLICATION, endpoint)]),
            fmt_int(count_flags(counts, endpoint=endpoint, keep=False)),
            fmt_int(count_flags(counts, endpoint=endpoint, status=UNCERTAIN, source=CHEMBL)),
            fmt_int(count_flags(counts, endpoint=endpoint, status=UNCERTAIN, source=BINDINGDB)),
        ])
    return ["", "## Theo endpoint", ""] + format_table(
        ["endpoint", "nhóm mức 1", "nhóm mức 2", "observation keep=False",
         "uncertain ChEMBL", "uncertain BindingDB"], rows)


def describe_side(source: str, record: Mapping[str, Any]) -> str:
    """Một dòng mô tả một bên của bản ghi ánh xạ, đủ để tra lại trong raw."""
    def get(name: str) -> Any:
        return record[f"{source}_{name}"]

    assay = get("assay_id") if get("assay_entry_id") is None else (
        f"{get('assay_entry_id')}_{get('assay_id')}")
    return (f"{source:<9} id={get('native_id')}  assay={assay}  pmid={get('pmid')}  "
            f"doi={get('doi')}  patent={get('patent')}  curation={get('curation')!r}  "
            f"protein={get('protein_accession')}  giá trị={get('qualifier')} "
            f"{get('value_source')} {get('unit_source')}")


def examples_section(examples: Mapping[int, list[dict[str, Any]]],
                     reactant_set_column: str) -> list[str]:
    """Các ví dụ cụ thể kèm cách tra lại trong raw."""
    lines = ["", "## Ví dụ để tự tra lại trong raw"]
    number = 0
    for level in sorted(examples):
        for record in examples[level]:
            number += 1
            difference = record["p_activity_difference"]
            lines += [
                "",
                f"Ví dụ {number} — mức {LEVEL_LABELS[level]}, {record['dedup_status']}, "
                f"lý do: {record['reason']}",
                f"  endpoint={record['endpoint']}  InChIKey={record['molecule_inchikey']}  "
                f"tài liệu chung={record['document_match']}  protein={record['protein_match']}  "
                f"giá trị={record['value_match']}  |Δp|={fmt_float(difference, 4)}",
                "  " + describe_side(CHEMBL, record),
                "  " + describe_side(BINDINGDB, record),
                f"  Tra ChEMBL: SELECT * FROM activities WHERE activity_id = "
                f"{record[f'{CHEMBL}_native_id']};",
                f"  Tra BindingDB: BindingDB_All_*.tsv, cột '{reactant_set_column}' = "
                f"{record[f'{BINDINGDB}_native_id']}",
            ]
    if number == 0:
        lines.append("(không có cặp nào để làm ví dụ)")
    return lines


def known_case_section(known: Mapping[str, Any], known_records: Sequence[Mapping[str, Any]],
                       prepared: Mapping[str, pa.Table],
                       document_keys: Sequence[str]) -> tuple[list[str], dict[str, Any]]:
    """Kiểm chứng ca đã biết: nhóm rsid BindingDB có được ghép với nhóm activity ChEMBL không.

    Nhận: config known_case, các bản ghi ánh xạ chạm vào ca này, các dòng của ca này ở
    mỗi nguồn (đã qua prepare_side), danh sách loại tài liệu.
    Trả về: (các dòng báo cáo, số liệu cho JSON).
    """
    bindingdb_range = id_range(known["bindingdb_reactant_set_ids"])
    chembl_range = id_range(known["chembl_activity_ids"])
    bindingdb_rows = prepared[BINDINGDB].to_pylist()
    chembl_rows = prepared[CHEMBL].to_pylist()
    in_range = [record for record in known_records
                if record[f"{BINDINGDB}_native_id"] in bindingdb_range]
    status_counts = Counter((record["evidence_level"], record["dedup_status"])
                            for record in in_range)
    duplicates = [record for record in in_range if record["dedup_status"] == DUPLICATE]
    caught = {record[f"{BINDINGDB}_observation_id"] for record in duplicates
              if record[f"{CHEMBL}_native_id"] in chembl_range}
    duplicated = {record[f"{BINDINGDB}_observation_id"] for record in duplicates}
    outside_only = duplicated - caught
    outside = {record[f"{CHEMBL}_native_id"] for record in duplicates
               if record[f"{CHEMBL}_native_id"] not in chembl_range}
    total = len(bindingdb_rows)
    not_duplicate = total - len(duplicated)
    start_b, end_b = known["bindingdb_reactant_set_ids"]
    start_c, end_c = known["chembl_activity_ids"]
    # Không in một nhãn "bắt được" chung: 3/48 khớp trong khoảng và 33 khớp ngoài khoảng
    # là hai kết luận khác nhau (dedup đúng, nhưng giả thuyết về khoảng ID thì sai).
    lines = ["", "## Ca kiểm chứng",
             f"BindingDB reactant set {start_b}–{end_b}  so với  "
             f"ChEMBL activity {start_c}–{end_c}",
             f"KẾT QUẢ trên {fmt_int(total)} observation BindingDB trong khoảng:"]
    lines += format_table(["kết quả", "observation BindingDB", "%"], [
        ["duplicate với activity ChEMBL TRONG khoảng", fmt_int(len(caught)),
         fmt_pct(len(caught), total)],
        ["duplicate chỉ với activity ChEMBL NGOÀI khoảng", fmt_int(len(outside_only)),
         fmt_pct(len(outside_only), total)],
        ["không thành duplicate", fmt_int(not_duplicate), fmt_pct(not_duplicate, total)],
    ])
    lines += [f"Activity ChEMBL ngoài khoảng được ghép: {fmt_int(len(outside))} activity "
              "khác nhau.",
              "Giả thuyết 'hai khoảng ID là cùng các phép đo' chỉ đúng nếu phần lớn observation "
              "rơi vào dòng đầu; nếu phần lớn rơi vào dòng thứ hai thì dedup vẫn bắt được "
              "nhưng bản tương ứng nằm ở activity khác."]
    lines += ["", "Bản ghi ánh xạ có rsid trong khoảng, theo (mức, trạng thái):"]
    lines += format_table(["mức / trạng thái", "số bản ghi"],
                          [[f"{level} / {status}", fmt_int(count)]
                           for (level, status), count in sorted(status_counts.items())]
                          or [["(không có)", "0"]])
    diagnosis = overlap_diagnosis(bindingdb_rows, chembl_rows, document_keys)
    lines += ["", "Chẩn đoán (hai tập dòng có chung gì):"]
    lines += format_table(["tiêu chí", "số lượng"],
                          [[label, fmt_int(value)] for label, value in diagnosis])
    summary = {"caught_bindingdb_observations": len(caught),
               "duplicate_outside_range_only": len(outside_only),
               "not_duplicate": not_duplicate,
               "bindingdb_observations_in_range": total,
               "chembl_observations_in_range": len(chembl_rows),
               "chembl_activities_outside_range": sorted(outside),
               "records_by_level_status": {f"{level}/{status}": count
                                           for (level, status), count in status_counts.items()},
               "diagnosis": dict(diagnosis)}
    return lines, summary


# --------------------------------------------------------------------------- chạy


def load_config(path: Path) -> dict[str, Any]:
    """Đọc config.json (UTF-8)."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_settings(settings: Mapping[str, Any]) -> None:
    """Dừng sớm với thông báo rõ ràng nếu config dedup có giá trị không dùng được."""
    problems = []
    if settings["keep_source"] not in SOURCES:
        problems.append(f"keep_source phải là một trong {SOURCES}")
    unknown = set(settings["document_keys"]) - set(DOCUMENT_FIELDS)
    if unknown or not settings["document_keys"]:
        problems.append(f"document_keys phải khác rỗng và thuộc {sorted(DOCUMENT_FIELDS)}")
    if set(settings["provenance_markers"]) != set(SOURCES):
        problems.append(f"provenance_markers phải có đúng các khóa {SOURCES}")
    if settings["letters_per_partition"] < 1:
        problems.append("letters_per_partition phải >= 1")
    bad_levels = {int(level) for level in settings["examples_per_level"]} - set(LEVEL_LABELS)
    if bad_levels:
        problems.append(f"examples_per_level có mức không tồn tại: {sorted(bad_levels)}")
    try:
        re.compile(settings["relative_stereo_pattern"])
    except re.error as error:
        problems.append(f"relative_stereo_pattern không phải regex hợp lệ: {error}")
    if problems:
        raise ValueError("config stage2a.dedup không hợp lệ: " + "; ".join(problems))


def setup_logging(log_path: Path) -> None:
    """Log ra màn hình và ra file (UTF-8)."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stderr),
                    logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def run_dedup(inputs: Mapping[str, Path], partials: Mapping[str, Path],
              config: Mapping[str, Any]) -> dict[str, Any]:
    """Toàn bộ việc 1: lượt tổng quan, ghép theo phân vùng, ghi ánh xạ và hai bản sao.

    Nhận: file đầu vào mỗi nguồn, file .partial cho ánh xạ và hai bản sao, config.
    Trả về: dict số liệu dùng cho báo cáo.
    """
    stage = config["stage2a"]
    settings, columns = stage["dedup"], stage["columns"]
    batch_rows = settings["batch_rows"]
    started = time.monotonic()
    known = settings["known_case"]
    known_ids = {BINDINGDB: id_range(known["bindingdb_reactant_set_ids"]),
                 CHEMBL: id_range(known["chembl_activity_ids"])}

    # --- Lượt tổng quan: giá trị source_curation và các dòng của ca kiểm chứng.
    curation_counts, known_tables = {}, {}
    for source in SOURCES:
        LOGGER.info("[tổng quan] đọc %s", inputs[source])
        curation_counts[source], table = overview_pass(inputs[source], columns,
                                                       known_ids[source], batch_rows)
        known_tables[source] = prepare_side(table, source, settings)

    # --- Ghép theo phân vùng InChIKey; ghi ánh xạ ngay sau mỗi phân vùng.
    partitions = build_partitions(settings["partition_alphabet"],
                                  settings["letters_per_partition"])
    rng = random.Random(settings["example_seed"])
    quotas = {int(level): size for level, size in settings["examples_per_level"].items()}
    examples: dict[int, list[dict[str, Any]]] = {level: [] for level in quotas}
    seen_by_level: Counter = Counter()
    record_counts: Counter = Counter()
    group_counts: Counter = Counter()
    multi_chembl: Counter = Counter()
    branches: Counter = Counter()
    stereo: Counter = Counter()
    suppressed_total = 0
    flag_tables = []
    known_records = []
    writer = pq.ParquetWriter(partials["mapping"], MAPPING_SCHEMA)
    try:
        for number, partition in enumerate(partitions, start=1):
            tables = {source: prepare_side(read_partition(inputs[source], columns, partition,
                                                          batch_rows), source, settings)
                      for source in SOURCES}
            keyed = partition.kind != "null"
            try:
                records, suppressed = match_partition(tables[CHEMBL], tables[BINDINGDB],
                                                      keyed, settings)
            except Exception as error:
                raise RuntimeError(f"phân vùng {partition.label!r}: {error}") from error
            if records:
                writer.write_table(pa.Table.from_pylist(records, schema=MAPPING_SCHEMA))
            # Số liệu cho báo cáo.
            suppressed_total += suppressed
            record_counts.update((record["evidence_level"], record["dedup_status"],
                                  record["reason"]) for record in records)
            branches.update(branch_counts(records))
            stereo.update(provenance_stereo_counts(tables[BINDINGDB], records, keyed,
                                                   settings["relative_stereo_pattern"]))
            partition_groups, partition_multi = summarize_groups(records)
            group_counts.update(partition_groups)
            multi_chembl.update(partition_multi)
            flags = observation_flags(records, settings["keep_source"])
            flag_tables.append(pa.Table.from_pylist(flags, schema=FLAG_SCHEMA))
            for record in records:
                level = record["evidence_level"]
                if record["record_type"] == RECORD_PAIR and level in quotas:
                    reservoir_add(examples[level], seen_by_level[level], record,
                                  quotas[level], rng)
                    seen_by_level[level] += 1
                if (record[f"{BINDINGDB}_native_id"] in known_ids[BINDINGDB]
                        or record[f"{CHEMBL}_native_id"] in known_ids[CHEMBL]):
                    known_records.append(record)
            LOGGER.info("[phân vùng %d/%d %s] ChEMBL %s dòng, BindingDB %s dòng -> %s bản ghi "
                        "(%s duplicate); đã chạy %.0f giây",
                        number, len(partitions), partition.label,
                        fmt_int(tables[CHEMBL].num_rows), fmt_int(tables[BINDINGDB].num_rows),
                        fmt_int(len(records)),
                        fmt_int(sum(1 for record in records
                                    if record["dedup_status"] == DUPLICATE)),
                        time.monotonic() - started)
    finally:
        writer.close()

    # --- Bản sao observations có thêm cột dedup_*.
    flags = pa.concat_tables(flag_tables)
    totals = {}
    for source in SOURCES:
        source_flags = flags.filter(pc.equal(flags["source"], source))
        totals[source] = write_flagged_copy(inputs[source], partials[source], source_flags,
                                            columns, stage["dedup_columns"], batch_rows,
                                            source)
    return {
        "elapsed_seconds": time.monotonic() - started,
        "totals": totals,
        "curation_counts": curation_counts,
        "record_counts": record_counts,
        "group_counts": group_counts,
        "multi_chembl": multi_chembl,
        "branches": branches,
        "stereo": stereo,
        "suppressed": suppressed_total,
        "flag_counts": flag_counts(flags),
        "examples": examples,
        "known_records": known_records,
        "known_tables": known_tables,
    }


def build_report(result: Mapping[str, Any], config: Mapping[str, Any],
                 outputs: Mapping[str, Path]) -> tuple[list[str], dict[str, Any]]:
    """Báo cáo văn bản đầy đủ và dict số liệu cho file JSON."""
    settings = config["stage2a"]["dedup"]
    counts = result["flag_counts"]
    lines = ["# Stage 2a — Dedup ChEMBL ↔ BindingDB",
             f"Thời gian chạy: {fmt_float(result['elapsed_seconds'] / 60, 1)} phút"]
    lines += curation_section(result["curation_counts"], settings,
                              settings["curation_values_shown"])
    lines += level_section(result["group_counts"], result["multi_chembl"], counts)
    lines += uncertain_section(result["record_counts"], result["suppressed"])
    branches = branch_totals(result["branches"])
    stereo = stereo_totals(result["stereo"])
    lines += branch_section(branches)
    lines += stereo_section(stereo)
    lines += affected_section(counts, result["totals"], settings["keep_source"])
    lines += endpoint_section(result["group_counts"], counts, config["endpoints"])
    lines += examples_section(result["examples"],
                              config["bindingdb"]["all_columns"]["reactant_set_id"])
    known_lines, known_summary = known_case_section(
        settings["known_case"], result["known_records"], result["known_tables"],
        settings["document_keys"])
    lines += known_lines
    lines += ["", "## File output"] + [f"  {path}" for path in outputs.values()]
    summary = {
        "elapsed_seconds": result["elapsed_seconds"],
        "observations": result["totals"],
        "curation_counts": {source: dict(counter.most_common())
                            for source, counter in result["curation_counts"].items()},
        "groups_by_level_endpoint": {f"{level}/{endpoint}": count for (level, endpoint), count
                                     in result["group_counts"].items()},
        "groups_with_multiple_chembl_by_level": dict(result["multi_chembl"]),
        "records_by_level_status_reason": {f"{level}/{status}/{reason}": count
                                           for (level, status, reason), count
                                           in result["record_counts"].items()},
        "suppressed_uncertain_pairs": result["suppressed"],
        "pairs_by_document_branch": {
            branch: {**entry, "uncertain_reasons": dict(entry["uncertain_reasons"])}
            for branch, entry in branches.items()},
        "bindingdb_provenance_relative_stereo": stereo,
        "observation_flags": counts,
        "known_case": known_summary,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    return lines, summary


def short_summary(summary: Mapping[str, Any]) -> list[str]:
    """Vài dòng tóm tắt in ở cuối, để không phải mở JSON."""
    flags = summary["observation_flags"]
    known = summary["known_case"]
    lines = ["", "=" * 72, "TÓM TẮT"]
    for level in (LEVEL_CURATION, LEVEL_PUBLICATION):
        groups = sum(count for key, count in summary["groups_by_level_endpoint"].items()
                     if key.startswith(f"{level}/"))
        lines.append(f"  Nhóm duplicate mức {level}: {fmt_int(groups)}")
    for source in SOURCES:
        lines.append(
            f"  {source}: {fmt_int(summary['observations'][source])} observation, "
            f"duplicate {fmt_int(count_flags(flags, source=source, status=DUPLICATE))}, "
            f"keep=False {fmt_int(count_flags(flags, source=source, keep=False))}, "
            f"uncertain {fmt_int(count_flags(flags, source=source, status=UNCERTAIN))}")
    for branch in (BRANCH_PUBLICATION, BRANCH_PATENT, BRANCH_BOTH):
        entry = summary["pairs_by_document_branch"][branch]
        lines.append(f"  Cặp {branch}: {fmt_int(entry['pairs'])}, uncertain "
                     f"{fmt_pct(entry['uncertain'], entry['pairs'])}")
    unpaired = summary["bindingdb_provenance_relative_stereo"][REASON_NO_COUNTERPART]
    lines.append(f"  BindingDB nhập từ ChEMBL không có cặp: {fmt_int(unpaired['rows'])} dòng, "
                 f"có |r|: {fmt_int(unpaired['with_r'])}")
    lines.append(f"  Ca kiểm chứng ({fmt_int(known['bindingdb_observations_in_range'])} "
                 "observation BindingDB): duplicate trong khoảng "
                 f"{fmt_int(known['caught_bindingdb_observations'])}, chỉ ngoài khoảng "
                 f"{fmt_int(known['duplicate_outside_range_only'])}, không duplicate "
                 f"{fmt_int(known['not_duplicate'])}")
    lines.append("=" * 72)
    return lines


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Tham số dòng lệnh: chỉ có --force."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="ghi đè output đã có")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào: kiểm tra đường dẫn, chạy dedup, ghi output nguyên tử, in báo cáo."""
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config(CONFIG_PATH)
    stage = config["stage2a"]
    settings = stage["dedup"]
    validate_settings(settings)

    # --- Đường dẫn vào/ra.
    observations_dir = ROOT / config["paths"]["observations_dir"]
    inputs = {CHEMBL: observations_dir / config["output"]["chembl_file"],
              BINDINGDB: observations_dir / config["output"]["bindingdb_file"]}
    dedup_dir = ROOT / stage["paths"]["dedup_dir"]
    outputs = {"mapping": dedup_dir / settings["mapping_file"],
               CHEMBL: dedup_dir / settings["copy_files"][CHEMBL],
               BINDINGDB: dedup_dir / settings["copy_files"][BINDINGDB]}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        print(f"Thiếu file đầu vào (chạy Stage 1 trước): {', '.join(missing)}")
        return 2
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.force:
        print(f"Output đã có, không ghi đè (thêm --force để ghi lại): {', '.join(existing)}")
        return 2

    # --- Log và báo cáo của lần chạy.
    runs_dir = ROOT / stage["paths"]["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    dedup_dir.mkdir(parents=True, exist_ok=True)
    stem = runs_dir / f"dedup_{datetime.now().strftime(TIMESTAMP_FORMAT)}"
    setup_logging(stem.with_suffix(".log"))

    # --- Ghi vào .partial; chỉ đổi tên khi mọi file đã ghi xong.
    partials = {key: path.with_name(path.name + PARTIAL_SUFFIX) for key, path in outputs.items()}
    try:
        result = run_dedup(inputs, partials, config)
    except BaseException:
        for path in partials.values():
            path.unlink(missing_ok=True)
        raise
    for key, partial in partials.items():
        partial.replace(outputs[key])

    lines, summary = build_report(result, config, outputs)
    lines += short_summary(summary)
    stem.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                    default=str), encoding="utf-8")
    print("\n".join(lines))
    print(f"Báo cáo: {stem.with_suffix('.txt')}\nSố liệu: {stem.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
