"""Stage 2a: chẩn đoán thêm cho audit độ nhiễu (chỉ đọc, không lọc, không sửa dữ liệu).

Chạy từ thư mục gốc của repo, SAU khi đã chạy dedup_cross_source.py:
    python scripts\\diagnose_noise.py

Ba câu hỏi còn mở sau báo cáo audit 21/10/2026 (noise_audit_20261021_024216):
  A. Bản sao trong cùng một nguồn. Ở BindingDB, cặp assay khác publication có |Δ| = 0
     rất nhiều (Ki 48,9 %). Các cặp đó đến từ loại tài liệu nào, và có phải cả một bảng
     số liệu được chép sang tài liệu khác không? Câu trả lời quyết định có cần dedup
     trong cùng nguồn. ChEMBL (bỏ cặp có cờ potential_duplicate) chạy cùng cách làm
     đối chứng.
  B. Publication lớn nhất mỗi endpoint: một tài liệu thật hay do gom bắc cầu nhầm.
  C. Nhóm replicate có STD lớn: bao nhiêu nhóm gộp cả dạng có cờ lập thể tương đối |r|
     lẫn dạng không có cờ (InChIKey của Stage 1 bỏ cờ này nên hai dạng chung một khóa).

Tập dữ liệu, assay_key, publication, nhóm replicate và cặp assay đều dùng lại hàm của
audit_noise.py, nên số cặp ở đây khớp với mục 7.4 của báo cáo audit.
    -> runs/stage2a/noise_diagnose_<thời điểm>.{log,txt,json}
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# scripts/ nằm trên sys.path khi chạy script.
import audit_noise as audit
from dedup_cross_source import relative_stereo_mask

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
LOGGER = logging.getLogger("diagnose_noise")

CHEMBL = audit.CHEMBL
BINDINGDB = "bindingdb"
SOURCES = (CHEMBL, BINDINGDB)
# Vai trò cột đọc thêm so với audit: phần CX của SMILES (để biết cờ |r|), mã phân tử nguồn
# và số tâm lập thể chưa xác định cấu hình (Stage 1 tính bằng RDKit).
EXTRA_ROLES = ("cx_extension", "molecule_source_id", "unspecified_stereo_centers")
# Mục D: trạng thái tâm lập thể chưa xác định của phân tử trong nhóm.
STEREO_STATES = ("có tâm chưa xác định", "không có tâm chưa xác định", "không rõ")
ID_STATES = ("> 1 mã phân tử", "1 mã phân tử")
ROLES = audit.AUDIT_ROLES + EXTRA_ROLES
GROUP_KEYS = ["assay_key", "protein_hash", "inchikey"]

# Mục A: phạm vi -> có bỏ cặp mang cờ potential_duplicate không.
# BindingDB không có cờ này; ChEMBL bỏ cặp có cờ để làm đối chứng "dữ liệu độc lập".
COPY_SCOPES = {BINDINGDB: False, CHEMBL: True}
COPY_SCOPE_LABELS = {BINDINGDB: "bindingdb", CHEMBL: "chembl, bỏ cặp có cờ"}

# Loại tài liệu của một assay, mã hóa bằng bit: patent = 1, PMID/DOI = 2.
KIND_PATENT_BIT, KIND_ARTICLE_BIT = 1, 2
KIND_LABELS = {0: "không PMID/DOI/patent", 1: "patent", 2: "bài báo", 3: "patent + bài báo"}
KIND_COUNT = len(KIND_LABELS)
PATENT_PREFIX = "patent:"
ARTICLE_PREFIXES = ("pmid:", "doi:")
DOC_PREFIX = "doc:"

# Nhóm của một cặp assay ở mục A, ngoài các khoảng tỷ lệ |Δ| = 0 của cặp publication.
CATEGORY_MISSING = -2
CATEGORY_SMALL = -1


# --------------------------------------------------------------------------- hàm thuần


def bin_index(values: np.ndarray, zero: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Chỉ số khoảng của từng phần tử theo đúng cách chia của audit_noise.bin_counts.

    0 là '= 0' (theo mặt nạ `zero`), 1 là (0; e0), ..., len(edges) + 1 là '≥ e_cuối'.
    """
    index = np.searchsorted(np.asarray(edges, dtype=np.float64), values, side="right") + 1
    return np.where(zero, 0, index).astype(np.int64)


def fraction_labels(edges: Sequence[float]) -> list[str]:
    """Nhãn các khoảng tỷ lệ: '[0; e0)', '[e0; e1)', ..., '≥ e_cuối'."""
    def show(value: float) -> str:
        return f"{value:g}".replace(".", ",")

    bounds = [0.0, *edges]
    labels = [f"[{show(low)}; {show(high)})" for low, high in zip(bounds, bounds[1:])]
    return labels + [f"≥ {show(edges[-1])}"]


def assay_doc_kinds(keys: pa.Table) -> pa.Table:
    """Loại tài liệu của mỗi assay, suy ra từ các khóa publication của nó.

    Nhận: bảng (assay_key, key) của audit_noise.publication_keys.
    Trả về: bảng assay_key, kind (bit 1: có patent, bit 2: có PMID hoặc DOI).
    Lấy từ chính các khóa dùng để gom publication, để "loại tài liệu" và "publication"
    luôn nói về cùng một thứ.
    """
    keys_column = keys["key"]
    article = pc.or_(pc.starts_with(keys_column, ARTICLE_PREFIXES[0]),
                     pc.starts_with(keys_column, ARTICLE_PREFIXES[1]))
    flags = pa.table({"assay_key": keys["assay_key"],
                      "patent": pc.starts_with(keys_column, PATENT_PREFIX),
                      "article": article})
    grouped = flags.group_by("assay_key").aggregate([("patent", "any"), ("article", "any")])
    kind = (audit.as_numpy(grouped["patent_any"], bool).astype(np.int64) * KIND_PATENT_BIT
            + audit.as_numpy(grouped["article_any"], bool).astype(np.int64) * KIND_ARTICLE_BIT)
    return pa.table({"assay_key": grouped["assay_key"], "kind": pa.array(kind, pa.int64())})


def kind_codes(groups: pa.Table, kinds: pa.Table) -> np.ndarray:
    """Loại tài liệu của từng dòng `groups` (assay không có khóa nào -> 0)."""
    if kinds.num_rows == 0:
        return np.zeros(groups.num_rows, dtype=np.int64)
    positions = pc.index_in(groups["assay_key"], value_set=kinds["assay_key"].combine_chunks())
    return audit.as_numpy(pc.fill_null(pc.take(kinds["kind"], positions), 0), np.int64)


def assay_pairs(groups: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    """Mọi cặp dòng của `groups` có cùng (protein, phân tử): cùng cách dựng cặp với audit."""
    indexed = groups.append_column("row", pa.array(np.arange(groups.num_rows, dtype=np.int64)))
    lists = indexed.group_by(["protein_hash", "inchikey"]).aggregate([("row", "list")])
    pairs = [audit.pair_indices(np.asarray(rows, dtype=np.int64))
             for rows in lists["row_list"].to_pylist() if len(rows) >= audit.MIN_REPLICATES]
    if not pairs:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return (np.concatenate([pair[0] for pair in pairs]),
            np.concatenate([pair[1] for pair in pairs]))


def independent_mask(groups: pa.Table, first: np.ndarray, second: np.ndarray,
                     exclude_flagged: bool) -> np.ndarray:
    """Cặp khác publication (thiếu khóa ở một bên cũng tính là khác), có thể bỏ cặp có cờ.

    Đúng định nghĩa nhóm 'khác publication' / 'khác publication, không cờ' của audit 7.4.
    """
    publications = audit.as_numpy(groups["publication"], np.int64)
    same = ((publications[first] == publications[second])
            & (publications[first] != audit.PUBLICATION_MISSING))
    mask = ~same
    if exclude_flagged:
        flags = audit.as_numpy(groups["potential_duplicate"], bool)
        mask &= ~(flags[first] | flags[second])
    return mask


def copy_diagnosis(groups: pa.Table, first: np.ndarray, second: np.ndarray,
                   kinds: np.ndarray, zero_tolerance: float,
                   settings: Mapping[str, Any]) -> dict[str, Any]:
    """Mục A: các cặp assay khác publication có |Δ| = 0 đến từ đâu.

    Nhận: bảng replicate_groups của một phạm vi; các cặp đã chọn (chỉ số dòng); loại
    tài liệu từng dòng; ngưỡng "bằng 0"; config stage2a.diagnose.
    Trả về: dict gồm
    - pairs, zero_pairs: số cặp và số cặp |Δ| = 0;
    - categories: theo tỷ lệ |Δ| = 0 của cặp publication chứa cặp assay đó;
    - kinds: theo loại tài liệu hai bên;
    - copied_volume: số nhóm và phép đo dính vào cặp |Δ| = 0 ở khoảng tỷ lệ cao nhất;
    - top_publication_pairs: các cặp publication có nhiều cặp |Δ| = 0 nhất.
    Vì sao xét theo cặp publication: nếu hai tài liệu chép cùng một bảng số liệu thì gần
    như mọi phân tử–protein chung của chúng có |Δ| = 0. Trùng hợp ngẫu nhiên (hai lab
    cùng ra 10 µM) thì rải rác, tỷ lệ thấp. Tỷ lệ tính trên cặp publication có ít nhất
    copy_min_shared_pairs cặp assay chung; ít hơn thì không đủ để phân biệt.
    """
    edges = settings["copy_fraction_edges"]
    means = audit.as_numpy(groups["mean_p"], np.float64)
    publications = audit.as_numpy(groups["publication"], np.int64)
    sizes = audit.as_numpy(groups["n"], np.int64)
    zero = np.abs(means[first] - means[second]) <= zero_tolerance
    round_value = np.abs(means[first] - np.round(means[first])) <= settings["round_p_tolerance"]

    # --- Cặp publication: mã hóa (nhỏ, lớn) thành một số để đếm bằng np.unique.
    left, right = publications[first], publications[second]
    missing = (left == audit.PUBLICATION_MISSING) | (right == audit.PUBLICATION_MISSING)
    width = int(publications.max()) + 1 if publications.size else 1
    codes = np.minimum(left, right) * width + np.maximum(left, right)
    valid = ~missing
    unique, inverse, shared = np.unique(codes[valid], return_inverse=True, return_counts=True)
    zeros = np.bincount(inverse.ravel(), weights=zero[valid].astype(np.float64),
                        minlength=unique.size).astype(np.int64)
    fraction = zeros / np.maximum(shared, 1)
    publication_category = np.where(shared < settings["copy_min_shared_pairs"], CATEGORY_SMALL,
                                    np.searchsorted(np.asarray(edges, np.float64), fraction,
                                                    side="right"))
    category = np.full(first.size, CATEGORY_MISSING, dtype=np.int64)
    category[valid] = publication_category[inverse.ravel()]

    categories = []
    for code in [CATEGORY_MISSING, CATEGORY_SMALL, *range(len(edges) + 1)]:
        mask = category == code
        categories.append({
            "category": code,
            "publication_pairs": (None if code == CATEGORY_MISSING
                                  else int(np.count_nonzero(publication_category == code))),
            "pairs": int(np.count_nonzero(mask)),
            "zero_pairs": int(np.count_nonzero(mask & zero)),
            "zero_round_value": int(np.count_nonzero(mask & zero & round_value)),
        })

    # --- Loại tài liệu hai bên (không phân biệt thứ tự).
    copied = category == len(edges)
    kind_pair = (np.minimum(kinds[first], kinds[second]) * KIND_COUNT
                 + np.maximum(kinds[first], kinds[second]))
    kind_rows = []
    for code in np.unique(kind_pair).tolist():
        mask = kind_pair == code
        kind_rows.append({"kinds": [code // KIND_COUNT, code % KIND_COUNT],
                          "pairs": int(np.count_nonzero(mask)),
                          "zero_pairs": int(np.count_nonzero(mask & zero)),
                          "copied_zero_pairs": int(np.count_nonzero(mask & zero & copied))})

    # --- Khối lượng dính vào bản chép: nhóm ở hai đầu các cặp |Δ| = 0 của khoảng cao nhất.
    involved = np.unique(np.concatenate([first[copied & zero], second[copied & zero]]))
    copied_volume = {"groups": int(involved.size), "measurements": int(sizes[involved].sum()),
                     "scope_groups": int(groups.num_rows), "scope_measurements": int(sizes.sum())}

    # --- Ví dụ: cặp publication nhiều |Δ| = 0 nhất (hòa thì nhiều cặp chung hơn).
    order = np.lexsort((-shared, -zeros))
    top = [{"publications": [int(unique[i] // width), int(unique[i] % width)],
            "shared_pairs": int(shared[i]), "zero_pairs": int(zeros[i])}
           for i in order[:settings["top_publication_pairs"]] if zeros[i] > 0]
    return {"pairs": int(first.size), "zero_pairs": int(np.count_nonzero(zero)),
            "categories": categories, "kinds": kind_rows, "copied_volume": copied_volume,
            "top_publication_pairs": top}


def publication_key_counts(data: pa.Table, dedup_settings: Mapping[str, Any]
                           ) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Khóa tài liệu theo assay và theo publication.

    Nhận: tập dữ liệu đã có assay_key và publication; config dedup (chuẩn hóa DOI/patent).
    Trả về: (keys: assay_key, key; assay_publication: assay_key, publication;
    key_counts: publication, key, assay_key_count = số assay của publication mang khóa đó).
    """
    keys = audit.publication_keys(data, dedup_settings)
    assay_publication = data.group_by(["assay_key", "publication"]).aggregate([])
    joined = keys.join(assay_publication, "assay_key", join_type="inner")
    key_counts = joined.group_by(["publication", "key"]).aggregate([("assay_key", "count")])
    return keys, assay_publication, key_counts


def publication_labels(key_counts: pa.Table, publications: Sequence[int],
                       shown: int) -> dict[int, str]:
    """Nhãn dễ đọc của publication: vài khóa đầu, ưu tiên PMID/DOI/patent hơn khóa 'doc:'."""
    if not publications:
        return {}
    wanted = pa.array(sorted(set(publications)), type=pa.int64())
    rows = key_counts.filter(pc.is_in(key_counts["publication"], value_set=wanted)).to_pylist()
    items: dict[int, list[tuple[bool, int, str]]] = defaultdict(list)
    for row in rows:
        items[row["publication"]].append((row["key"].startswith(DOC_PREFIX),
                                          -row["assay_key_count"], row["key"]))
    labels = {}
    for publication, entries in items.items():
        keys = [key for *_, key in sorted(entries)]
        extra = len(keys) - shown
        labels[publication] = ", ".join(keys[:shown]) + (f" (+{extra} khóa)" if extra > 0 else "")
    return labels


def largest_publications(data: pa.Table, assay_publication: pa.Table, key_counts: pa.Table,
                         top_n: int, top_keys: int) -> list[dict[str, Any]]:
    """Mục B: `top_n` publication nhiều assay nhất và các khóa đã nối chúng lại.

    Trả về: mỗi publication một dict: số assay (tổng và theo nguồn), số protein, phân tử,
    phép đo, số khóa theo loại (pmid/doi/patent/doc) và các khóa phủ nhiều assay nhất.
    Cách đọc: một tài liệu thật có không quá 1 PMID và 1 DOI, và khóa phổ biến nhất phủ
    gần hết assay. Nhiều PMID/DOI trong một publication nghĩa là đã gom bắc cầu qua
    nhiều tài liệu.
    """
    valid = assay_publication.filter(pc.not_equal(assay_publication["publication"],
                                                  audit.PUBLICATION_MISSING))
    sizes = valid.group_by("publication").aggregate([("assay_key", "count")]).sort_by(
        [("assay_key_count", "descending"), ("publication", "ascending")]).slice(0, top_n)
    result = []
    for publication, assays in zip(sizes["publication"].to_pylist(),
                                   sizes["assay_key_count"].to_pylist()):
        members = valid.filter(pc.equal(valid["publication"], publication))["assay_key"]
        rows = data.filter(pc.equal(data["publication"], publication))
        keys = key_counts.filter(pc.equal(key_counts["publication"], publication)).sort_by(
            [("assay_key_count", "descending"), ("key", "ascending")])
        names = keys["key"].to_pylist()
        result.append({
            "publication": publication,
            "assays": assays,
            "assays_by_source": dict(Counter(key.split(":", 1)[0]
                                             for key in members.to_pylist())),
            "proteins": int(pc.count_distinct(rows["protein_hash"]).as_py()),
            "molecules": int(pc.count_distinct(rows["inchikey"]).as_py()),
            "measurements": rows.num_rows,
            "keys_by_tag": dict(Counter(name.split(":", 1)[0] for name in names)),
            "top_keys": list(zip(names[:top_keys],
                                 keys["assay_key_count"].to_pylist()[:top_keys])),
        })
    return result


def stereo_breakdown(subset: pa.Table, ddof: int,
                     settings: Mapping[str, Any]) -> dict[str, Any]:
    """Mục C: nhóm replicate (≥ 2 phép đo) theo khoảng STD, kèm cờ |r| và mã phân tử nguồn.

    Nhận: tập dữ liệu của một nguồn (có cột relative_stereo, molecule_source_id); ddof;
    config audit (diagnostic_bin_edges, zero_tolerance).
    Trả về: số nhóm và, với từng khoảng STD, số nhóm:
    - any_relative: có ít nhất một dòng mang cờ |r|;
    - mixed: có cả dòng mang |r| lẫn dòng không mang (dạng tương đối và dạng tuyệt đối
      chung một InChIKey vì Stage 1 bỏ cờ |r| khi tính khóa: đây là gộp do cách xử lý);
    - multiple_ids: nhiều hơn một mã phân tử nguồn (ChEMBL ID / BindingDB MonomerID);
    - multiple_ids_without_relative: nhiều mã nhưng không dòng nào mang |r| (nguồn có
      nhiều bản ghi cho cùng một cấu trúc, không liên quan |r|).
    Nhóm dựng giống audit_noise.replicate_groups, nên số nhóm khớp mục 7.1 của audit.
    """
    edges = settings["diagnostic_bin_edges"]
    keyed = subset.filter(pc.is_valid(subset["inchikey"]))
    grouped = keyed.group_by(GROUP_KEYS).aggregate([
        ("p_activity", "count"),
        ("p_activity", "stddev", pc.VarianceOptions(ddof=ddof)),
        ("p_activity", "min"),
        ("p_activity", "max"),
        ("relative_stereo", "any"),
        ("relative_stereo", "all"),
        ("molecule_source_id", "count_distinct"),
    ])
    multi = grouped.filter(pc.greater_equal(grouped["p_activity_count"], audit.MIN_REPLICATES))
    stds = audit.as_numpy(pc.fill_null(multi["p_activity_stddev"], 0.0), np.float64)
    spread = audit.as_numpy(pc.subtract(multi["p_activity_max"], multi["p_activity_min"]),
                            np.float64)
    index = bin_index(stds, spread <= settings["zero_tolerance"], edges)
    any_relative = audit.as_numpy(pc.fill_null(multi["relative_stereo_any"], False), bool)
    all_relative = audit.as_numpy(pc.fill_null(multi["relative_stereo_all"], False), bool)
    multiple = audit.as_numpy(multi["molecule_source_id_count_distinct"], np.int64) > 1
    mixed = any_relative & ~all_relative
    bins = {}
    for position, label in enumerate(audit.bin_labels(edges)):
        mask = index == position
        bins[label] = {
            "groups": int(np.count_nonzero(mask)),
            "any_relative": int(np.count_nonzero(mask & any_relative)),
            "mixed": int(np.count_nonzero(mask & mixed)),
            "multiple_ids": int(np.count_nonzero(mask & multiple)),
            "multiple_ids_without_relative": int(np.count_nonzero(mask & multiple
                                                                  & ~any_relative)),
        }
    return {"groups": int(stds.size), "mixed": int(np.count_nonzero(mixed)), "bins": bins}


def isomer_check(subset: pa.Table, ddof: int, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Mục D: nhóm nhiều mã phân tử nguồn có phải là các đồng phân lập thể vẽ thiếu lập thể.

    Nhận: tập dữ liệu của một nguồn (có molecule_source_id, unspecified_stereo_centers);
    ddof; config audit.
    Trả về: {trạng thái mã: {trạng thái lập thể: {"groups", "zero", "high"}}} với zero là
    số nhóm STD = 0, high là số nhóm ở khoảng STD cao nhất (≥ mốc cuối).
    Giả thuyết: patent tách riêng 'isomer 1' / 'isomer 2' (hai MonomerID) nhưng SMILES
    không ghi cấu hình, nên hai chất chung InChIKey mà giá trị khác nhau. Nếu đúng thì:
    (1) nhóm > 1 mã có tỷ lệ "có tâm chưa xác định" cao hơn hẳn nhóm 1 mã; (2) trong nhóm
    > 1 mã, STD lớn tập trung ở phần có tâm chưa xác định, còn phần không có tâm chưa xác
    định chủ yếu STD = 0 (hai bản ghi của cùng một chất).
    Stage 1 chỉ đếm tâm bất đối ở nguyên tử; đồng phân E/Z của liên kết đôi không được đếm.
    Nhóm có dòng không đếm được tâm lập thể (null) xếp vào 'không rõ'.
    """
    edges = settings["diagnostic_bin_edges"]
    keyed = subset.filter(pc.is_valid(subset["inchikey"]))
    flagged = keyed.append_column("stereo_unknown",
                                  pc.is_null(keyed["unspecified_stereo_centers"]))
    grouped = flagged.group_by(GROUP_KEYS).aggregate([
        ("p_activity", "count"),
        ("p_activity", "stddev", pc.VarianceOptions(ddof=ddof)),
        ("p_activity", "min"),
        ("p_activity", "max"),
        ("molecule_source_id", "count_distinct"),
        ("unspecified_stereo_centers", "max"),
        ("stereo_unknown", "any"),
    ])
    multi = grouped.filter(pc.greater_equal(grouped["p_activity_count"], audit.MIN_REPLICATES))
    stds = audit.as_numpy(pc.fill_null(multi["p_activity_stddev"], 0.0), np.float64)
    spread = audit.as_numpy(pc.subtract(multi["p_activity_max"], multi["p_activity_min"]),
                            np.float64)
    zero = spread <= settings["zero_tolerance"]
    high = bin_index(stds, zero, edges) == len(edges) + 1
    multiple = audit.as_numpy(multi["molecule_source_id_count_distinct"], np.int64) > 1
    unknown = audit.as_numpy(pc.fill_null(multi["stereo_unknown_any"], True), bool)
    unspecified = audit.as_numpy(
        pc.fill_null(multi["unspecified_stereo_centers_max"], 0), np.int64) > 0
    stereo_masks = dict(zip(STEREO_STATES, (unspecified & ~unknown, ~unspecified & ~unknown,
                                            unknown)))
    result: dict[str, Any] = {}
    for id_state, id_mask in zip(ID_STATES, (multiple, ~multiple)):
        result[id_state] = {
            state: {"groups": int(np.count_nonzero(id_mask & mask)),
                    "zero": int(np.count_nonzero(id_mask & mask & zero)),
                    "high": int(np.count_nonzero(id_mask & mask & high))}
            for state, mask in stereo_masks.items()}
    return result


def prepare(data: pa.Table, stage: Mapping[str, Any]) -> pa.Table:
    """Thêm cột relative_stereo (cờ |r|) và publication vào tập dữ liệu đã có assay_key."""
    data = data.append_column("relative_stereo", relative_stereo_mask(
        data["cx_extension"], stage["dedup"]["relative_stereo_pattern"]))
    data, _ = audit.add_publication(data, stage["dedup"])
    return data


def diagnose_data(data: pa.Table, config: Mapping[str, Any]) -> dict[str, Any]:
    """Mục A, B, C cho một endpoint, trên tập dữ liệu đã qua prepare()."""
    stage = config["stage2a"]
    audit_settings, settings = stage["audit"], stage["diagnose"]
    ddof = audit_settings["replicate_std_ddof"]
    keys, assay_publication, key_counts = publication_key_counts(data, stage["dedup"])
    kinds = assay_doc_kinds(keys)

    copies = {}
    for scope, exclude_flagged in COPY_SCOPES.items():
        groups = audit.replicate_groups(data.filter(pc.equal(data["source"], scope)), ddof)
        first, second = assay_pairs(groups)
        mask = independent_mask(groups, first, second, exclude_flagged)
        copies[scope] = copy_diagnosis(groups, first[mask], second[mask],
                                       kind_codes(groups, kinds),
                                       audit_settings["zero_tolerance"], settings)
    wanted = [publication for result in copies.values()
              for entry in result["top_publication_pairs"]
              for publication in entry["publications"]]
    labels = publication_labels(key_counts, wanted, settings["publication_keys_shown"])
    for result in copies.values():
        for entry in result["top_publication_pairs"]:
            entry["labels"] = [labels.get(publication, "–")
                               for publication in entry["publications"]]

    return {
        "copies": copies,
        "largest_publications": largest_publications(data, assay_publication, key_counts,
                                                     settings["top_publications"],
                                                     settings["top_keys"]),
        "stereo": {source: stereo_breakdown(data.filter(pc.equal(data["source"], source)),
                                            ddof, audit_settings)
                   for source in SOURCES},
        "isomers": {source: isomer_check(data.filter(pc.equal(data["source"], source)),
                                         ddof, audit_settings)
                    for source in SOURCES},
    }


# --------------------------------------------------------------------------- đọc dữ liệu


def load_endpoint(paths: Mapping[str, Path], endpoint: str, config: Mapping[str, Any],
                  flagged: pa.Array) -> pa.Table:
    """Tập audit của một endpoint (hai nguồn) đúng như audit_noise.audit_endpoint, thêm cột.

    Lọc giống hệt audit (dedup_keep, qualifier '=', protein/assay resolved, có pActivity);
    đọc thêm phần CX và mã phân tử nguồn cho mục C.
    """
    stage = config["stage2a"]
    columns, settings = stage["columns"], stage["audit"]
    keep_column = stage["dedup_columns"]["keep"]
    parts = []
    for path in paths.values():
        names = [columns[role] for role in ROLES] + [keep_column]
        table = pq.read_table(path, columns=names,
                              filters=[(columns["endpoint"], "==", endpoint)])
        table = table.rename_columns([*ROLES, audit.KEEP])
        rows, _ = audit.apply_filters(table, audit.filter_steps(
            table, config["qualifiers"]["exact"], settings["resolved_status"]))
        parts.append(rows.drop_columns([audit.KEEP, "protein_status", "assay_status"]))
    data = audit.add_assay_key(pa.concat_tables(parts))
    data = data.append_column("potential_duplicate",
                              pc.is_in(data["observation_id"], value_set=flagged))
    return prepare(data, stage)


# --------------------------------------------------------------------------- báo cáo


def category_label(code: int, settings: Mapping[str, Any]) -> str:
    """Nhãn nhóm của mục A."""
    if code == CATEGORY_MISSING:
        return "thiếu khóa tài liệu ở ≥ 1 bên"
    if code == CATEGORY_SMALL:
        return f"cặp publication chung < {settings['copy_min_shared_pairs']} cặp assay"
    return "tỷ lệ |Δ| = 0 " + fraction_labels(settings["copy_fraction_edges"])[code]


def copy_section(results: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    """Mục A: bảng theo tỷ lệ của cặp publication, theo loại tài liệu, khối lượng, ví dụ."""
    overview, category_rows, kind_rows, volume_rows, examples = [], [], [], [], []
    for endpoint, result in results.items():
        for scope, values in result["copies"].items():
            label = f"{endpoint} / {COPY_SCOPE_LABELS[scope]}"
            overview.append([label, audit.fmt_int(values["pairs"]),
                             audit.fmt_int(values["zero_pairs"]),
                             audit.fmt_pct(values["zero_pairs"], values["pairs"])])
            for entry in values["categories"]:
                publications = entry["publication_pairs"]
                category_rows.append([
                    f"{label} / {category_label(entry['category'], settings)}",
                    "–" if publications is None else audit.fmt_int(publications),
                    audit.fmt_int(entry["pairs"]), audit.fmt_int(entry["zero_pairs"]),
                    audit.fmt_pct(entry["zero_pairs"], entry["pairs"]),
                    audit.fmt_pct(entry["zero_pairs"], values["zero_pairs"]),
                    audit.fmt_pct(entry["zero_round_value"], entry["zero_pairs"])])
            for entry in values["kinds"]:
                kinds = " – ".join(KIND_LABELS[kind] for kind in entry["kinds"])
                kind_rows.append([f"{label} / {kinds}", audit.fmt_int(entry["pairs"]),
                                  audit.fmt_int(entry["zero_pairs"]),
                                  audit.fmt_pct(entry["zero_pairs"], entry["pairs"]),
                                  audit.fmt_int(entry["copied_zero_pairs"])])
            volume = values["copied_volume"]
            volume_rows.append([label, audit.fmt_int(volume["scope_groups"]),
                                audit.fmt_int(volume["scope_measurements"]),
                                audit.fmt_int(volume["groups"]),
                                audit.fmt_int(volume["measurements"]),
                                audit.fmt_pct(volume["measurements"],
                                              volume["scope_measurements"])])
            if scope == BINDINGDB and values["top_publication_pairs"]:
                examples.append(f"{endpoint}:")
                for number, entry in enumerate(values["top_publication_pairs"], start=1):
                    first, second = entry["labels"]
                    examples.append(
                        f"  {number}. [{first}] ↔ [{second}]: chung "
                        f"{audit.fmt_int(entry['shared_pairs'])} cặp assay, "
                        f"{audit.fmt_int(entry['zero_pairs'])} cặp |Δ| = 0 "
                        f"({audit.fmt_pct(entry['zero_pairs'], entry['shared_pairs'])})")
    top_label = fraction_labels(settings["copy_fraction_edges"])[-1]
    return [
        "", "## A. Cặp assay khác publication có |Δ| = 0: trùng hợp hay chép bảng số liệu",
        "Cặp assay: cùng phân tử, cùng protein, cùng nguồn, khác publication (định nghĩa như "
        "mục 7.4 của audit; ChEMBL bỏ thêm cặp có cờ potential_duplicate). Số cặp phải khớp "
        "dòng tương ứng ở mục 7.4.", ""] + audit.format_table(
        ["endpoint / phạm vi", "cặp assay", "|Δ| = 0", "%"], overview) + [
        "", "### A.1. Theo tỷ lệ |Δ| = 0 của cặp publication chứa cặp assay đó",
        "Hai tài liệu chép cùng một bảng thì gần như mọi cặp chung có |Δ| = 0 (tỷ lệ cao). "
        "Trùng hợp (vd. cùng ra đúng 10 µM) thì rải rác (tỷ lệ thấp) và thường là giá trị "
        "tròn.", ""] + audit.format_table(
        ["endpoint / phạm vi / nhóm", "cặp publication", "cặp assay", "|Δ| = 0", "% nhóm",
         "% mọi |Δ| = 0", "% |Δ| = 0 có p nguyên (10 nM, 1 µM, ...)"], category_rows) + [
        "", "### A.2. Theo loại tài liệu hai bên", ""] + audit.format_table(
        ["endpoint / phạm vi / loại tài liệu", "cặp assay", "|Δ| = 0", "%",
         f"|Δ| = 0 trong nhóm {top_label}"], kind_rows) + [
        "Loại tài liệu lấy từ khóa publication của assay: 'bài báo' = có PMID hoặc DOI.",
        "", f"### A.3. Khối lượng dính vào cặp |Δ| = 0 của nhóm tỷ lệ {top_label}",
        "Nhóm = (assay, protein, phân tử) ở một trong hai đầu của cặp. Đây là số liên quan, "
        "không phải số sẽ bị loại: dedup trong cùng nguồn chỉ bỏ một bên của mỗi cặp.",
        ""] + audit.format_table(
        ["endpoint / phạm vi", "nhóm trong phạm vi", "phép đo trong phạm vi", "nhóm liên quan",
         "phép đo liên quan", "% phép đo"], volume_rows) + [
        "", "### A.4. Cặp publication BindingDB có nhiều |Δ| = 0 nhất (để tra tay)",
        "Mỗi bên in tối đa vài khóa, ưu tiên PMID/DOI/patent; 'doc:bindingdb:<n>' là ENTRYID.",
        ""] + (examples or ["(không có)"])


def largest_section(results: Mapping[str, Any]) -> list[str]:
    """Mục B: publication lớn nhất mỗi endpoint."""
    rows, details = [], []
    for endpoint, result in results.items():
        for rank, entry in enumerate(result["largest_publications"], start=1):
            tags = entry["keys_by_tag"]
            top_share = (audit.fmt_pct(entry["top_keys"][0][1], entry["assays"])
                         if entry["top_keys"] else "–")
            rows.append([f"{endpoint} / {rank}", audit.fmt_int(entry["assays"]),
                         audit.fmt_int(entry["assays_by_source"].get(CHEMBL, 0)),
                         audit.fmt_int(entry["assays_by_source"].get(BINDINGDB, 0)),
                         audit.fmt_int(entry["proteins"]), audit.fmt_int(entry["molecules"]),
                         audit.fmt_int(entry["measurements"])]
                        + [audit.fmt_int(tags.get(tag, 0)) for tag in ("pmid", "doi", "patent",
                                                                        "doc")]
                        + [top_share])
            details.append(f"{endpoint} / {rank}: " + "; ".join(
                f"{key} ({audit.fmt_int(count)} assay)" for key, count in entry["top_keys"]))
    return ["", "## B. Publication nhiều assay nhất mỗi endpoint (kiểm tra gom bắc cầu)",
            "Một tài liệu thật: không quá 1 PMID và 1 DOI, khóa phổ biến nhất phủ gần hết "
            "assay. Nhiều PMID/DOI trong một publication nghĩa là đã nối nhiều tài liệu.",
            ""] + audit.format_table(
        ["endpoint / hạng", "assay", "assay ChEMBL", "assay BindingDB", "protein", "phân tử",
         "phép đo", "PMID", "DOI", "patent", "doc", "% assay khóa đầu phủ"], rows) + [
        "", "Các khóa phủ nhiều assay nhất:"] + details


def stereo_section(results: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    """Mục C: nhóm replicate theo khoảng STD, cờ |r| và số mã phân tử nguồn."""
    rows, summary = [], []
    high = audit.bin_labels(settings["diagnostic_bin_edges"])[-1]
    for endpoint, result in results.items():
        for source, values in result["stereo"].items():
            for label, entry in values["bins"].items():
                groups = entry["groups"]
                rows.append([f"{endpoint} / {source} / STD {label}", audit.fmt_int(groups),
                             audit.fmt_pct(entry["any_relative"], groups),
                             audit.fmt_pct(entry["mixed"], groups),
                             audit.fmt_pct(entry["multiple_ids"], groups),
                             audit.fmt_pct(entry["multiple_ids_without_relative"], groups)])
            top = values["bins"][high]
            summary.append([f"{endpoint} / {source}", audit.fmt_int(values["groups"]),
                            audit.fmt_pct(top["groups"], values["groups"]),
                            audit.fmt_int(top["mixed"]),
                            audit.fmt_pct(top["groups"] - top["mixed"],
                                          values["groups"] - values["mixed"])])
    return ["", "## C. Nhóm replicate theo STD: cờ |r| và số mã phân tử nguồn",
            "Nhóm ≥ 2 phép đo, dựng như mục 7.1 của audit (số nhóm phải khớp). 'trộn |r|': "
            "nhóm có cả dòng mang |r| lẫn dòng không mang, tức hai dạng lập thể bị gộp vì "
            "InChIKey của Stage 1 bỏ cờ |r|.", ""] + audit.format_table(
        ["endpoint / nguồn / khoảng", "nhóm", "% có |r|", "% trộn |r|", "% > 1 mã phân tử",
         "% > 1 mã, không |r|"], rows) + [
        "", f"Ảnh hưởng lên tỷ lệ nhóm STD {high}:", ""] + audit.format_table(
        ["endpoint / nguồn", "nhóm", f"% STD {high}", f"nhóm trộn |r| ở STD {high}",
         f"% STD {high} nếu bỏ nhóm trộn |r|"], summary)


def isomer_section(results: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    """Mục D: nhóm replicate theo số mã phân tử nguồn và tâm lập thể chưa xác định."""
    high = audit.bin_labels(settings["diagnostic_bin_edges"])[-1]
    rows = []
    for endpoint, result in results.items():
        for source, values in result["isomers"].items():
            for id_state, states in values.items():
                total = sum(entry["groups"] for entry in states.values())
                for state, entry in states.items():
                    groups = entry["groups"]
                    rows.append([f"{endpoint} / {source} / {id_state} / {state}",
                                 audit.fmt_int(groups), audit.fmt_pct(groups, total),
                                 audit.fmt_pct(entry["zero"], groups),
                                 audit.fmt_pct(entry["high"], groups)])
    return ["", "## D. Nhóm nhiều mã phân tử nguồn: có phải đồng phân lập thể vẽ thiếu lập thể",
            "Nhóm ≥ 2 phép đo như mục C. 'tâm chưa xác định': phân tử có ít nhất một tâm bất "
            "đối không ghi cấu hình (Stage 1 không đếm đồng phân E/Z). 'không rõ': có dòng "
            "Stage 1 không đếm được tâm lập thể.",
            "Giả thuyết đồng phân đúng nếu: (1) ở dòng '> 1 mã', cột '% trong trạng thái mã' của "
            "'có tâm chưa xác định' cao hơn hẳn dòng '1 mã'; (2) trong '> 1 mã', STD lớn tập "
            "trung ở 'có tâm chưa xác định', còn 'không có tâm chưa xác định' chủ yếu STD = 0.",
            ""] + audit.format_table(
        ["endpoint / nguồn / mã phân tử / lập thể", "nhóm", "% trong trạng thái mã",
         "% STD = 0", f"% STD {high}"], rows)


def setup_logging(log_path: Path) -> None:
    """Log ra màn hình và ra file (UTF-8), cho cả logger của script này và của audit."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handlers = (logging.StreamHandler(sys.stderr), logging.FileHandler(log_path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
    for logger in (LOGGER, audit.LOGGER):
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def build_report(results: Mapping[str, Any], config: Mapping[str, Any],
                 elapsed: float) -> list[str]:
    """Toàn bộ báo cáo văn bản."""
    stage = config["stage2a"]
    return (["# Stage 2a — Chẩn đoán thêm cho audit độ nhiễu (chỉ đọc, không lọc)",
             f"Thời gian chạy: {audit.fmt_float(elapsed / 60, 1)} phút"]
            + copy_section(results, stage["diagnose"])
            + largest_section(results)
            + stereo_section(results, stage["audit"])
            + isomer_section(results, stage["audit"]))


def main() -> int:
    """Điểm vào: kiểm tra đầu vào, chạy mục A–C cho từng endpoint, ghi báo cáo."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = audit.load_config(CONFIG_PATH)
    stage = config["stage2a"]
    keep_column = stage["dedup_columns"]["keep"]
    dedup_dir = ROOT / stage["paths"]["dedup_dir"]
    paths = {source: dedup_dir / name for source, name in stage["dedup"]["copy_files"].items()}
    for path in paths.values():
        if not path.is_file():
            print(f"Thiếu {path}. Chạy scripts\\dedup_cross_source.py trước.")
            return 2
        if keep_column not in pq.ParquetFile(path).schema_arrow.names:
            print(f"{path} không có cột {keep_column}. Chạy lại scripts\\dedup_cross_source.py.")
            return 2

    runs_dir = ROOT / stage["paths"]["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    stem = runs_dir / f"noise_diagnose_{datetime.now().strftime(audit.TIMESTAMP_FORMAT)}"
    setup_logging(stem.with_suffix(".log"))
    started = time.monotonic()

    # --- Cờ potential_duplicate: cùng hàm và cùng lượt đọc như audit (mục 6).
    _, flagged = audit.original_vs_standard(paths[CHEMBL], stage["columns"], keep_column,
                                            config["qualifiers"]["exact"], stage["audit"])
    flagged_ids = pa.array(sorted(flagged), type=pa.string())
    results = {}
    for endpoint in config["endpoints"]:
        LOGGER.info("[%s] bắt đầu; đã chạy %.0f giây", endpoint, time.monotonic() - started)
        results[endpoint] = diagnose_data(load_endpoint(paths, endpoint, config, flagged_ids),
                                          config)
    elapsed = time.monotonic() - started

    lines = build_report(results, config, elapsed)
    stem.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stem.with_suffix(".json").write_text(
        json.dumps({"elapsed_seconds": elapsed, "endpoints": results},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Báo cáo: {stem.with_suffix('.txt')}\nSố liệu: {stem.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
