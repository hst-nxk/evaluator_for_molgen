"""Audit trùng lặp: bản đã loại có đúng là bản sao không, và còn bản sao nào sót lại không.

Câu hỏi của buổi họp 26/10/2026:
    1. Các dòng BindingDB đã loại (dedup_keep = False) có thật sự là bản sao của ChEMBL?
    2. Trong các dòng còn giữ, có dòng nào cũng trùng mà quy tắc dedup không bắt được?
    3. Trong dataset cuối, các bản ghi có thật sự khác nhau hoàn toàn?

Cách làm: mỗi cột khóa được mã hóa thành số nguyên (numpy), rồi gom nhóm các dòng bằng
nhau ở MỌI cột của khóa. Khóa được nới lỏng dần theo audit.duplicates.key_levels:
    khóa dedup hiện dùng (có tài liệu) -> bỏ tài liệu -> bỏ cả protein.
Nhóm nào có cả dòng ChEMBL lẫn BindingDB ở mức nới lỏng là "nghi bản chép chưa bắt
được". Không dòng nào bị sửa hay xóa: script chỉ đếm và in ví dụ để tra ngược về nguồn.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from molopt.dataset import pipeline

from . import common

LOGGER = logging.getLogger(__name__)

READ_ROLES = (
    "observation_id", "source", "keep", "curation", "dedup_reason", "inchikey", "smiles",
    "endpoint", "protein_hash", "qualifier", "p_activity", "assay_id", "assay_entry_id",
    "protein_status", "assay_status", "document_pmid", "document_doi", "document_patent",
    "document_id",
)
KEY_COLUMNS = ("inchikey", "endpoint", "protein", "qualifier", "value", "document")


# --------------------------------------------------------------------------- dựng khóa


def blank_to_null(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Chuỗi rỗng coi như null (tài liệu ghi '' không phải là một tài liệu)."""
    return pc.if_else(pc.equal(values, ""), pa.scalar(None, values.type), values)


def prefixed(prefix: str, values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """'<prefix><giá trị>', null giữ null."""
    return pc.binary_join_element_wise(pa.scalar(prefix), blank_to_null(values), "")


def document_key(table: pa.Table, settings: Mapping[str, Any]) -> pa.ChunkedArray:
    """Một định danh tài liệu mỗi dòng, ưu tiên PMID > DOI > patent > mã tài liệu nguồn.

    Dùng cùng quy tắc chuẩn hóa DOI và patent với Stage 2b (pipeline.normalize_doi,
    pipeline.normalize_patent, tham số stage2a.dedup). Giới hạn: hai dòng cùng bài báo mà
    một bên chỉ ghi PMID, bên kia chỉ ghi DOI sẽ nhận hai khóa khác nhau; mức khóa có tài
    liệu vì thế chỉ là mốc so sánh, câu trả lời chính nằm ở các mức đã bỏ tài liệu.
    """
    doi = pipeline.normalize_doi(table.column("document_doi"),
                                 settings["doi_prefix_pattern"])
    patent = pipeline.normalize_patent(table.column("document_patent"),
                                       settings["patent_strip_pattern"],
                                       settings["patent_kind_code_pattern"],
                                       settings["patent_kind_code_replacement"])
    source_document = pc.binary_join_element_wise(
        pa.scalar("doc:"), table.column("source"), pa.scalar(":"),
        blank_to_null(table.column("document_id")), "")
    return pc.coalesce(prefixed("pmid:", table.column("document_pmid")),
                       prefixed("doi:", doi), prefixed("patent:", patent), source_document)


def assay_key(table: pa.Table) -> pa.ChunkedArray:
    """Khóa assay '<nguồn>:<entry>:<assay>' (ASSAYID của BindingDB chỉ duy nhất trong ENTRY)."""
    return pc.binary_join_element_wise(
        table.column("source"), table.column("assay_entry_id"), table.column("assay_id"),
        ":", null_handling="replace", null_replacement="")


def key_arrays(table: pa.Table, documents: pa.ChunkedArray, decimals: int
               ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Mã số nguyên và mặt nạ không-null của từng cột khóa (KEY_COLUMNS).

    Nhận: bảng observation (tên cột vai trò); khóa tài liệu (document_key); số chữ số
    thập phân giữ lại khi so giá trị pActivity.
    """
    sources = {
        "inchikey": table.column("inchikey"), "endpoint": table.column("endpoint"),
        "protein": table.column("protein_hash"), "qualifier": table.column("qualifier"),
        "document": documents,
    }
    codes = {name: common.encode(values) for name, values in sources.items()}
    valid = {name: common.valid_mask(values) for name, values in sources.items()}
    p_activity = table.column("p_activity")
    codes["value"] = common.encode_rounded(p_activity, decimals)
    valid["value"] = common.valid_mask(p_activity)
    return codes, valid


def level_rows(level: Mapping[str, Any], subset: np.ndarray,
               valid: Mapping[str, np.ndarray]) -> np.ndarray:
    """Chỉ số dòng thuộc `subset` và không null ở mọi cột của mức khóa."""
    mask = subset.copy()
    for column in level["columns"]:
        if column not in KEY_COLUMNS:
            raise KeyError(f"cột khóa không biết: {column}; có {KEY_COLUMNS}")
        mask &= valid[column]
    return np.flatnonzero(mask)


# --------------------------------------------------------------------------- phân loại nhóm


def classify_groups(ids: np.ndarray, is_chembl: np.ndarray, assays: np.ndarray,
                    documents: np.ndarray) -> dict[str, Any]:
    """Thành phần các nhóm có từ 2 dòng trở lên.

    Nhận: mã nhóm, cờ ChEMBL, mã assay và mã tài liệu của từng dòng (cùng độ dài).
    Trả về: số nhóm/dòng theo loại:
        cross_source   nhóm có cả ChEMBL và BindingDB (nghi bản chép giữa hai nguồn);
        <nguồn>_only   nhóm chỉ một nguồn, tách tiếp:
            same_assay           cùng một assay: replicate, Stage 2b bước 1 đã gộp;
            one_document_assays  nhiều assay của CÙNG một tài liệu: nghi chép trong bài;
            many_documents       nhiều tài liệu: có thể chép lại, hoặc trùng giá trị thật.
    `extra_rows` = số dòng trừ số nhóm, tức số dòng thừa nếu mỗi nhóm chỉ giữ một.
    """
    counts = np.bincount(ids)
    chembl = np.bincount(ids, weights=is_chembl.astype(np.float64)).astype(np.int64)
    other = counts - chembl
    distinct_assays = common.distinct_per_group(ids, assays)
    distinct_documents = common.distinct_per_group(ids, documents)
    duplicated = counts >= 2

    def summary(mask: np.ndarray) -> dict[str, int]:
        return {"groups": int(mask.sum()), "rows": int(counts[mask].sum()),
                "extra_rows": int((counts[mask] - 1).sum()),
                "chembl_rows": int(chembl[mask].sum()), "bindingdb_rows": int(other[mask].sum())}

    result: dict[str, Any] = {
        "rows": int(ids.size), "groups": int(counts.size),
        "duplicated": summary(duplicated),
        "cross_source": summary(duplicated & (chembl > 0) & (other > 0)),
    }
    for name, single in (("chembl_only", duplicated & (other == 0)),
                         ("bindingdb_only", duplicated & (chembl == 0))):
        result[name] = {
            "all": summary(single),
            "same_assay": summary(single & (distinct_assays == 1)),
            "one_document_assays": summary(single & (distinct_assays > 1)
                                           & (distinct_documents == 1)),
            "many_documents": summary(single & (distinct_assays > 1)
                                      & (distinct_documents > 1)),
        }
    return result


def twin_counts(ids: np.ndarray, is_chembl: np.ndarray, targets: np.ndarray,
                round_values: np.ndarray) -> dict[str, int]:
    """Trong các dòng `targets` (dòng BindingDB), bao nhiêu dòng có một dòng ChEMBL cùng nhóm.

    `round_values`: cờ giá trị pActivity tròn (vd. 5,00 = 10 µM). Giá trị tròn hay gặp ở
    ngưỡng sàng lọc, nên hai phép đo khác nhau trùng giá trị tròn có thể là trùng hợp
    thật chứ không phải bản chép; tách riêng để đọc đúng.
    """
    chembl_per_group = np.bincount(ids, weights=is_chembl.astype(np.float64))
    has_twin = chembl_per_group[ids] > 0
    hit = targets & has_twin
    return {"targets": int(targets.sum()), "with_chembl_twin": int(hit.sum()),
            "with_twin_round_value": int((hit & round_values).sum()),
            "without_twin": int((targets & ~has_twin).sum())}


# --------------------------------------------------------------------------- dataset cuối


def dataset_distinctness(path: Path) -> dict[str, int]:
    """Các bản ghi của dataset cuối có khác nhau hoàn toàn không.

    Đếm: pair_id lặp; bộ (A, B, protein, endpoint) lặp; và cặp ghi cả hai chiều
    (A, B) lẫn (B, A) cho cùng protein và endpoint — điều không được xảy ra vì chiều
    của cặp đã được cố định khi dựng dataset.
    """
    table = pq.read_table(path, columns=["pair_id", "smiles_a", "smiles_b", "protein_hash",
                                         "endpoint"])
    rows = table.num_rows
    pair_ids = common.encode(table.column("pair_id"))
    smiles = common.encode(pa.chunked_array(table.column("smiles_a").chunks
                                            + table.column("smiles_b").chunks))
    side_a, side_b = smiles[:rows], smiles[rows:]
    protein = common.encode(table.column("protein_hash"))
    endpoint = common.encode(table.column("endpoint"))
    forward = common.group_ids([side_a, side_b, protein, endpoint])
    both = common.group_ids([np.concatenate([side_a, side_b]),
                             np.concatenate([side_b, side_a]),
                             np.concatenate([protein, protein]),
                             np.concatenate([endpoint, endpoint])])
    reversed_present = np.bincount(both[rows:], minlength=int(both.max()) + 1) > 0
    return {
        "rows": rows,
        "duplicate_pair_id_rows": int(rows - np.unique(pair_ids).size),
        "duplicate_pair_rows": int(rows - int(forward.max()) - 1) if rows else 0,
        "both_directions_rows": int(reversed_present[both[:rows]].sum()),
        "same_molecule_both_sides": int(np.count_nonzero(side_a == side_b)),
    }


# --------------------------------------------------------------------------- ví dụ


def example_rows(table: pa.Table, rows: np.ndarray, documents: pa.ChunkedArray,
                 assays: pa.ChunkedArray) -> list[dict[str, Any]]:
    """Các cột cần để tra ngược về nguồn cho những dòng được chọn."""
    taken = table.take(pa.array(rows))
    document_values = documents.take(pa.array(rows)).to_pylist()
    assay_values = assays.take(pa.array(rows)).to_pylist()
    result = []
    for index, row in enumerate(taken.to_pylist()):
        result.append({
            "observation_id": row["observation_id"], "source": row["source"],
            "curation": row["curation"], "keep": row["keep"], "endpoint": row["endpoint"],
            "inchikey": row["inchikey"], "protein_hash": row["protein_hash"],
            "qualifier": row["qualifier"], "p_activity": row["p_activity"],
            "assay": assay_values[index], "document": document_values[index],
            "dedup_reason": row["dedup_reason"],
        })
    return result


def sample_groups(ids: np.ndarray, mask: np.ndarray, count: int,
                  rng: np.random.Generator) -> list[np.ndarray]:
    """Chọn ngẫu nhiên (có seed) tối đa `count` nhóm thỏa `mask`; trả về chỉ số dòng."""
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        return []
    chosen = rng.choice(candidates, size=min(count, candidates.size), replace=False)
    order = np.argsort(ids, kind="stable")
    starts = np.searchsorted(ids[order], chosen, side="left")
    ends = np.searchsorted(ids[order], chosen, side="right")
    return [order[start:end] for start, end in zip(starts, ends)]


# --------------------------------------------------------------------------- chạy


def run(config: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], list[str]]:
    """Chạy cả ba phần audit trùng lặp. Trả về (số liệu, các dòng báo cáo)."""
    settings = config["audit"]["duplicates"]
    chembl_name = config["stage2a"]["dedup"]["keep_source"]
    table = common.read_observations(config, root, READ_ROLES)
    documents = document_key(table, config["stage2a"]["dedup"])
    codes, valid = key_arrays(table, documents, int(settings["value_decimals"]))
    assays = assay_key(table)
    assay_codes = common.encode(assays)
    is_chembl = common.boolean(pc.equal(table.column("source"), chembl_name))
    keep = common.boolean(table.column("keep"))
    reason = common.boolean(pc.equal(table.column("dedup_reason"),
                                     settings["provenance_reason"]))
    p_values = table.column("p_activity").to_numpy(zero_copy_only=False).astype(np.float64)
    round_values = np.abs(p_values - np.round(p_values)) <= float(settings["round_p_tolerance"])
    usable_kept = common.usable_mask(table, config, require_keep=True)
    usable_any = common.usable_mask(table, config, require_keep=False)
    rng = np.random.default_rng(int(settings["seed"]))
    examples_per_level = int(settings["examples_per_level"])
    LOGGER.info("dòng đọc được %s; dòng dùng được và được giữ %s", f"{table.num_rows:,}",
                f"{int(usable_kept.sum()):,}")

    payload: dict[str, Any] = {
        "rows_read": table.num_rows, "usable_kept_rows": int(usable_kept.sum()),
        "usable_any_rows": int(usable_any.sum()),
        "removed_rows": int((~keep).sum()), "settings": settings,
        "kept_groups": {}, "twins": {}, "examples": {},
    }
    targets = {
        "removed": ~keep & ~is_chembl,
        "provenance_without_counterpart": keep & ~is_chembl & reason & usable_any,
        "kept_other": keep & ~is_chembl & ~reason & usable_any,
    }
    for level in settings["key_levels"]:
        name = level["name"]
        rows = level_rows(level, usable_kept, valid)
        ids = common.group_ids([codes[column][rows] for column in level["columns"]])
        payload["kept_groups"][name] = classify_groups(
            ids, is_chembl[rows], assay_codes[rows], codes["document"][rows])
        payload["kept_groups"][name]["rows_skipped_null"] = int(usable_kept.sum() - rows.size)
        cross = (np.bincount(ids, weights=is_chembl[rows].astype(np.float64)) > 0) & \
            (np.bincount(ids, weights=(~is_chembl[rows]).astype(np.float64)) > 0)
        payload["examples"][name] = [
            example_rows(table, rows[group], documents, assays)
            for group in sample_groups(ids, cross, examples_per_level, rng)]

        everything = np.ones(table.num_rows, dtype=bool)
        twin_rows = level_rows(level, everything, valid)
        twin_ids = common.group_ids([codes[column][twin_rows]
                                     for column in level["columns"]])
        payload["twins"][name] = {
            target: twin_counts(twin_ids, is_chembl[twin_rows], mask[twin_rows],
                                round_values[twin_rows])
            | {"rows_skipped_null": int(mask.sum() - mask[twin_rows].sum())}
            for target, mask in targets.items()}
        LOGGER.info("xong mức khóa %s", name)

    loosest = settings["key_levels"][-1]
    loose_rows = level_rows(loosest, np.ones(table.num_rows, dtype=bool), valid)
    loose_ids = common.group_ids([codes[column][loose_rows] for column in loosest["columns"]])
    no_twin = targets["removed"][loose_rows] & ~(
        np.bincount(loose_ids, weights=is_chembl[loose_rows].astype(np.float64))[loose_ids] > 0)
    picked = rng.choice(np.flatnonzero(no_twin),
                        size=min(examples_per_level, int(no_twin.sum())), replace=False) \
        if no_twin.any() else np.zeros(0, dtype=np.int64)
    payload["examples"]["removed_without_twin"] = example_rows(
        table, loose_rows[picked], documents, assays)

    payload["dataset"] = dataset_distinctness(common.dataset_file(config, root, "evaluator"))
    return payload, report_lines(payload, settings)


# --------------------------------------------------------------------------- báo cáo


def report_lines(payload: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    """Bản báo cáo đọc được."""
    lines = ["# Audit trùng lặp (chỉ đọc, không sửa dữ liệu)", "",
             f"Dòng observation đọc: {common.number(payload['rows_read'])}",
             f"Dòng dùng được và được giữ (vào Stage 2b): "
             f"{common.number(payload['usable_kept_rows'])}",
             f"Dòng đã loại (dedup_keep = False): {common.number(payload['removed_rows'])}",
             f"Giá trị so khớp sau khi làm tròn {settings['value_decimals']} chữ số pActivity.",
             "", "Các mức khóa:"]
    for level in settings["key_levels"]:
        lines.append(f"  {level['name']}: {level['label']}")

    lines += ["", "## 1. Bản đã loại có đúng là bản sao không",
              "Mỗi dòng BindingDB đã loại phải có một dòng ChEMBL bằng hệt ở các cột khóa.", ""]
    lines += twin_table(payload, "removed")
    lines += ["", "## 2. Bản sao còn sót trong các dòng được giữ", "",
              "2a. Dòng BindingDB ghi nguồn 'ChEMBL' nhưng dedup không tìm được bản tương ứng:"]
    lines += twin_table(payload, "provenance_without_counterpart")
    lines += ["", "2b. Dòng BindingDB còn lại (không ghi nguồn ChEMBL), dùng được:"]
    lines += twin_table(payload, "kept_other")
    lines += ["", "Có bản ChEMBL ở mức bỏ tài liệu = cùng phân tử, protein, endpoint, giá trị",
              "nhưng khác (hoặc thiếu) tài liệu: nghi bản chép mà dedup không bắt được.",
              "Cột 'giá trị tròn' tách riêng phần có thể chỉ là trùng hợp (vd. 10 µM).", ""]

    lines += ["2c. Nhóm trùng trong các dòng được giữ, theo mức khóa:"]
    header = ["mức", "nhóm trùng", "dòng thừa", "hai nguồn", "dòng thừa 2 nguồn",
              "ChEMBL: cùng assay", "ChEMBL: 1 tài liệu", "ChEMBL: nhiều tài liệu",
              "BDB: cùng assay", "BDB: 1 tài liệu", "BDB: nhiều tài liệu"]
    rows = []
    for name, block in payload["kept_groups"].items():
        rows.append([name, common.number(block["duplicated"]["groups"]),
                     common.number(block["duplicated"]["extra_rows"]),
                     common.number(block["cross_source"]["groups"]),
                     common.number(block["cross_source"]["extra_rows"])]
                    + [common.number(block[source][kind]["extra_rows"])
                       for source in ("chembl_only", "bindingdb_only")
                       for kind in ("same_assay", "one_document_assays", "many_documents")])
    lines += common.table_lines(header, rows)
    lines += ["Các cột theo nguồn là số DÒNG THỪA. 'cùng assay' là replicate, Stage 2b bước 1",
              "đã gộp; '1 tài liệu' là nhiều assay của cùng một bài báo/patent cho đúng một giá",
              "trị; 'nhiều tài liệu' là nhiều bài cùng báo một giá trị."]

    dataset = payload["dataset"]
    lines += ["", "## 3. Dataset cuối (evaluator_training) có khác nhau hoàn toàn không", ""]
    lines += common.table_lines(["kiểm tra", "số dòng"], [
        ["tổng số dòng", common.number(dataset["rows"])],
        ["pair_id lặp", common.number(dataset["duplicate_pair_id_rows"])],
        ["(A, B, protein, endpoint) lặp", common.number(dataset["duplicate_pair_rows"])],
        ["có cả chiều (B, A)", common.number(dataset["both_directions_rows"])],
        ["A trùng B", common.number(dataset["same_molecule_both_sides"])],
    ])
    lines += ["Cả bốn số kiểm tra phải bằng 0.", "",
              "## Ví dụ để tra ngược (chi tiết trong file .json, mục examples)"]
    for name, groups in payload["examples"].items():
        lines.append(f"  {name}: {len(groups)} ví dụ")
    return lines


def twin_table(payload: Mapping[str, Any], target: str) -> list[str]:
    """Bảng số dòng có/không có bản ChEMBL bằng hệt, theo từng mức khóa."""
    rows = []
    for name, block in payload["twins"].items():
        item = block[target]
        rows.append([name, common.number(item["targets"]),
                     common.number(item["rows_skipped_null"]),
                     common.number(item["with_chembl_twin"]),
                     common.percent(common.share(item["with_chembl_twin"], item["targets"])),
                     common.number(item["with_twin_round_value"]),
                     common.number(item["without_twin"])])
    return common.table_lines(["mức", "số dòng", "thiếu cột khóa", "có bản ChEMBL", "%",
                               "trong đó giá trị tròn", "không có"], rows)
