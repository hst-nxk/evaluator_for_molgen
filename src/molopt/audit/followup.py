"""Audit bổ sung 27/10/2026: ba vấn đề còn tồn trước khi dựng dataset_v3 (chỉ đọc).

Ba câu hỏi, mỗi câu một phần của báo cáo:
    1. relative  : các dòng evaluator trùng (SMILES A, SMILES B, protein, endpoint) có phải
                   do khóa phân tử mang hậu tố lập thể tương đối (stage2b.identity.
                   relative_stereo_suffix) không, và hai bản có cùng nhãn Δp không.
    2. peptide   : phân tử khớp mẫu peptide (audit.molecules.peptide_smarts) còn bao nhiêu
                   sau ngưỡng MW (stage2b.export.large_molecule_weight); mỗi phân tử có bao
                   nhiêu mẫu gốc amino acid; danh sách SMILES để xem bằng mắt có phải thuốc
                   phân tử nhỏ dạng peptidomimetic không.
    3. in_source : nhóm trùng trong CÙNG một nguồn, cùng tài liệu, khác assay (khóa
                   audit.duplicates.key_levels[0]): có phải cả assay bị nhập hai lần, tức
                   một assay nằm trọn trong assay kia, không.
    4. lipinski  : (thêm 29/10/2026 theo gợi ý của sếp) bao nhiêu phân tử và cặp nằm trong
                   quy tắc 5 của Lipinski; chỉ đếm, không lọc.
Không dòng nào bị sửa hay xóa; chỉ đếm và in ví dụ để tra ngược.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import common, duplicates, molecules

LOGGER = logging.getLogger(__name__)

RELATIVE_COLUMNS = ("pair_id", "smiles_a", "smiles_b", "protein_hash", "endpoint",
                    "molecule_a_key", "molecule_b_key", "delta_p_mean", "partition")
PEPTIDE_COLUMNS = ("smiles_a", "smiles_b", "endpoint", "partition")
# Bốn đại lượng của quy tắc 5 (thứ tự cột của lipinski_properties) và khóa ngưỡng tương
# ứng trong config audit.followup.lipinski.
LIPINSKI_METRICS = ("mw", "logp", "donors", "acceptors")
LIPINSKI_LIMIT_KEYS = {"mw": "max_weight", "logp": "max_logp", "donors": "max_donors",
                       "acceptors": "max_acceptors"}


# --------------------------------------------------------------------------- tiện ích


def strip_suffix(keys: pa.ChunkedArray, suffix: str) -> pa.ChunkedArray:
    """Khóa phân tử bỏ hậu tố lập thể tương đối ở cuối (nếu có)."""
    return pc.replace_substring_regex(keys, pattern=re.escape(suffix) + "$", replacement="")


def group_range(ids: np.ndarray, values: np.ndarray, groups: int) -> np.ndarray:
    """max − min của `values` trong từng nhóm `ids` (độ dài = groups; NaN nếu nhóm rỗng)."""
    high = np.full(groups, -np.inf)
    low = np.full(groups, np.inf)
    np.maximum.at(high, ids, values)
    np.minimum.at(low, ids, values)
    spread = high - low
    spread[~np.isfinite(spread)] = np.nan
    return spread


def closed_buckets(values: np.ndarray, edges: Sequence[float]) -> list[dict[str, Any]]:
    """Đếm theo khoảng đóng bên phải: ≤ e_0, (e_0, e_1], ..., > e_last (bỏ NaN)."""
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    rows = [{"range": f"≤ {edges[0]:g}", "count": int(np.count_nonzero(array <= edges[0]))}]
    for low, high in zip(edges, edges[1:]):
        count = int(np.count_nonzero((array > low) & (array <= high)))
        rows.append({"range": f"({low:g}, {high:g}]", "count": count})
    rows.append({"range": f"> {edges[-1]:g}",
                 "count": int(np.count_nonzero(array > edges[-1]))})
    return rows


def bucket_table(rows: Sequence[Mapping[str, Any]], total: int) -> list[str]:
    """Bảng (khoảng, số, %) từ kết quả closed_buckets / common.bucket_counts."""
    return common.table_lines(["khoảng", "số", "%"],
                              [[row["range"], common.number(row["count"]),
                                common.percent(common.share(row["count"], total))]
                               for row in rows])


# --------------------------------------------------------------------------- 1. -REL


def relative_duplicates(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Phần 1: dòng evaluator trùng (A, B, protein, endpoint) và hậu tố -REL.

    Đọc evidence_records (có khóa phân tử) và chỉ giữ các bản ghi có trong evaluator
    (khớp pair_id), để đếm đúng trên tập mà audit duplicates đã báo. Một nhóm trùng được
    coi là "do -REL" khi mọi dòng trong nhóm có cùng khóa phân tử sau khi bỏ hậu tố.
    """
    stage = config["stage2b"]
    suffix = stage["identity"]["relative_stereo_suffix"]
    literature = config["audit"]["literature"]
    tolerance = float(stage["qc"]["collapse_tolerance"]) + float(
        stage["replicate"]["float_slack"])
    edges = [tolerance, float(literature["within_error"]), float(literature["meaningful"])]
    examples = int(config["audit"]["followup"]["examples"])

    evidence = pq.read_table(common.dataset_file(config, root, "evidence_records"),
                             columns=list(RELATIVE_COLUMNS))
    evaluator_ids = pq.read_table(common.dataset_file(config, root, "evaluator"),
                                  columns=["pair_id"]).column("pair_id").combine_chunks()
    in_evaluator = pc.is_in(evidence.column("pair_id"), value_set=evaluator_ids)
    table = evidence.filter(in_evaluator).combine_chunks()
    rows = table.num_rows
    LOGGER.info("phần 1: %s bản ghi evidence, %s có trong evaluator", f"{evidence.num_rows:,}",
                f"{rows:,}")

    smiles = common.encode(pa.chunked_array(table.column("smiles_a").chunks
                                            + table.column("smiles_b").chunks))
    ids = common.group_ids([smiles[:rows], smiles[rows:],
                            common.encode(table.column("protein_hash")),
                            common.encode(table.column("endpoint"))])
    groups = int(ids.max()) + 1 if rows else 0
    sizes = np.bincount(ids, minlength=groups)
    duplicated_group = sizes >= 2
    duplicated_row = duplicated_group[ids]

    relative_a = common.boolean(pc.ends_with(table.column("molecule_a_key"), pattern=suffix))
    relative_b = common.boolean(pc.ends_with(table.column("molecule_b_key"), pattern=suffix))
    base_a = common.encode(strip_suffix(table.column("molecule_a_key"), suffix))
    base_b = common.encode(strip_suffix(table.column("molecule_b_key"), suffix))
    same_base = (common.distinct_per_group(ids, base_a) == 1) & (
        common.distinct_per_group(ids, base_b) == 1)
    has_relative = np.bincount(ids, weights=(relative_a | relative_b).astype(np.float64),
                               minlength=groups) > 0
    explained = duplicated_group & same_base & has_relative
    partitions = common.distinct_per_group(ids, common.encode(table.column("partition")))

    delta = table.column("delta_p_mean").to_numpy(zero_copy_only=False).astype(np.float64)
    picked = np.flatnonzero(duplicated_row)
    dense = np.unique(ids[picked], return_inverse=True)[1]
    spread = group_range(dense, delta[picked], int(dense.max()) + 1 if picked.size else 0)

    pattern = Counter(zip(relative_a[picked].tolist(), relative_b[picked].tolist()))
    names = {(False, False): "A thường, B thường", (True, True): "A -REL, B -REL",
             (True, False): "A -REL, B thường", (False, True): "A thường, B -REL"}

    unexplained = np.flatnonzero(duplicated_group & ~explained)
    chosen = list(np.flatnonzero(explained)[:examples]) + list(unexplained[:examples])
    order = np.argsort(ids, kind="stable")
    starts = np.searchsorted(ids[order], chosen, side="left")
    ends = np.searchsorted(ids[order], chosen, side="right")
    sample = []
    for group, start, end in zip(chosen, starts, ends):
        members = order[start:end]
        sample.append({
            "explained": bool(explained[group]),
            "rows": [{"pair_id": table.column("pair_id")[int(row)].as_py(),
                      "molecule_a_key": table.column("molecule_a_key")[int(row)].as_py(),
                      "molecule_b_key": table.column("molecule_b_key")[int(row)].as_py(),
                      "smiles_a": table.column("smiles_a")[int(row)].as_py(),
                      "smiles_b": table.column("smiles_b")[int(row)].as_py(),
                      "delta_p_mean": float(delta[int(row)])} for row in members]})

    return {
        "evaluator_rows": rows,
        "duplicate_groups": int(duplicated_group.sum()),
        "duplicate_rows": int(duplicated_row.sum()),
        "extra_rows": int(duplicated_row.sum() - duplicated_group.sum()),
        "group_sizes": common.count_values(sizes[duplicated_group].tolist()),
        "explained_groups": int(explained.sum()),
        "unexplained_groups": int(unexplained.size),
        "groups_in_many_partitions": int(np.count_nonzero(duplicated_group & (partitions > 1))),
        "key_pattern_rows": {names[key]: count for key, count in pattern.items()},
        "delta_spread_edges": edges,
        "delta_spread": closed_buckets(spread, edges),
        "rows_with_relative_key": int(np.count_nonzero(relative_a | relative_b)),
        "examples": sample,
    }


def relative_lines(part: Mapping[str, Any]) -> list[str]:
    """Phần báo cáo đọc được của mục 1."""
    rows = part["evaluator_rows"]
    lines = ["## 1. Dòng evaluator trùng (A, B, protein, endpoint) và hậu tố -REL", "",
             f"Dòng evaluator: {common.number(rows)}",
             f"Dòng có ít nhất một khóa -REL: {common.number(part['rows_with_relative_key'])}"
             f" ({common.percent(common.share(part['rows_with_relative_key'], rows))})",
             f"Nhóm trùng: {common.number(part['duplicate_groups'])}; dòng thừa: "
             f"{common.number(part['extra_rows'])} (audit 27/10 báo 39.022)",
             f"Nhóm giải thích được bằng -REL (cùng khóa sau khi bỏ hậu tố): "
             f"{common.number(part['explained_groups'])}",
             f"Nhóm KHÔNG giải thích được: {common.number(part['unexplained_groups'])}",
             f"Nhóm nằm ở nhiều partition (phải bằng 0): "
             f"{common.number(part['groups_in_many_partitions'])}",
             "", "Cỡ nhóm (số dòng mỗi nhóm -> số nhóm):"]
    lines += [f"  {size}: {common.number(count)}" for size, count in part["group_sizes"].items()]
    lines += ["", "Dạng khóa của các dòng trong nhóm trùng:"]
    lines += [f"  {name}: {common.number(count)}"
              for name, count in part["key_pattern_rows"].items()]
    lines += ["", "Chênh lệch nhãn trong nhóm (max − min của delta_p_mean), số nhóm:"]
    lines += bucket_table(part["delta_spread"], part["duplicate_groups"])
    lines += ["Mốc: gộp bản sao (stage2b.qc.collapse_tolerance) và 0,3 / 1,0 (audit.literature)."]
    return lines


# --------------------------------------------------------------------------- 2. peptide


def residue_counts(smiles: Sequence[str], smarts: str) -> np.ndarray:
    """Số lần khớp mẫu gốc amino acid (N–Cα–C=O) trong từng SMILES (-1 nếu không đọc được).

    Đây là con số MÔ TẢ để xem phân bố, không phải định nghĩa peptide.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise ValueError(f"residue_smarts không hợp lệ: {smarts}")
    counts = np.full(len(smiles), -1, dtype=np.int64)
    for index, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text) if text else None
        if molecule is not None:
            counts[index] = len(molecule.GetSubstructMatches(pattern))
    return counts


def peptide_check(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Phần 2: phân tử dạng peptide còn lại sau ngưỡng MW, và số gốc amino acid của chúng."""
    settings = config["audit"]["followup"]
    molecule_settings = config["audit"]["molecules"]
    limit = float(config["stage2b"]["export"]["large_molecule_weight"])
    table = pq.read_table(common.dataset_file(config, root, "evaluator"),
                          columns=list(PEPTIDE_COLUMNS))
    unique = molecules.unique_strings(table.column("smiles_a"), table.column("smiles_b"))
    LOGGER.info("phần 2: tính chỉ số cho %s phân tử", f"{len(unique):,}")
    described = molecules.describe_all(unique.to_pylist(), molecule_settings)
    weight = described[:, molecules.METRICS.index("mw")]
    heavy = described[:, molecules.METRICS.index("heavy_atoms")]
    peptide = described[:, molecules.METRICS.index("peptide_like")] > 0
    small = weight < limit

    index_a = molecules.positions(table.column("smiles_a"), unique)
    index_b = molecules.positions(table.column("smiles_b"), unique)
    if (index_a < 0).any() or (index_b < 0).any():
        raise ValueError("có SMILES rỗng trong evaluator; dữ liệu không đúng như Stage 2b")
    pair_peptide = peptide[index_a] | peptide[index_b]
    pair_kept = small[index_a] & small[index_b]
    remaining = pair_peptide & pair_kept

    subset = np.flatnonzero(peptide & small)
    residues = residue_counts([unique[int(i)].as_py() for i in subset],
                              settings["residue_smarts"])
    pair_count = np.bincount(index_a, minlength=len(unique)) + np.bincount(
        index_b, minlength=len(unique))
    top = subset[np.argsort(-pair_count[subset], kind="stable")[:int(settings["top_molecules"])]]
    rng = np.random.default_rng(int(config["audit"]["duplicates"]["seed"]))
    random_pick = rng.choice(subset, size=min(int(settings["top_molecules"]), subset.size),
                             replace=False) if subset.size else subset
    residue_of = dict(zip(subset.tolist(), residues.tolist()))

    def describe_rows(chosen: np.ndarray) -> list[dict[str, Any]]:
        return [{"smiles": unique[int(i)].as_py(), "mw": float(weight[i]),
                 "heavy_atoms": float(heavy[i]), "residues": residue_of[int(i)],
                 "pairs": int(pair_count[i])} for i in chosen]

    endpoints = table.column("endpoint").to_numpy(zero_copy_only=False)
    partitions = table.column("partition").to_numpy(zero_copy_only=False)
    return {
        "pairs": table.num_rows, "molecules": len(unique),
        "unparsed_molecules": int(np.isnan(weight).sum()),
        "weight_limit": limit,
        "peptide_molecules": int(peptide.sum()),
        "peptide_molecules_below_limit": int(subset.size),
        "peptide_pairs": int(pair_peptide.sum()),
        "peptide_pairs_after_weight_filter": int(remaining.sum()),
        "pairs_after_weight_filter": int(pair_kept.sum()),
        "by_endpoint": common.count_values(endpoints[remaining].tolist()),
        "by_partition": common.count_values(partitions[remaining].tolist()),
        "residue_bins": common.bucket_counts(residues.astype(np.float64),
                                             list(settings["residue_bins"])),
        "residue_unparsed": int(np.count_nonzero(residues < 0)),
        "weight_of_remaining": common.quantile_summary(weight[subset],
                                                       config["audit"]["quantiles"]),
        "top_by_pairs": describe_rows(top),
        "random_sample": describe_rows(random_pick),
    }


def peptide_lines(part: Mapping[str, Any]) -> list[str]:
    """Phần báo cáo đọc được của mục 2."""
    after = part["pairs_after_weight_filter"]
    lines = ["## 2. Phân tử dạng peptide còn lại sau ngưỡng MW", "",
             f"Cặp evaluator: {common.number(part['pairs'])}; phân tử: "
             f"{common.number(part['molecules'])} (không đọc được: "
             f"{common.number(part['unparsed_molecules'])})",
             f"Ngưỡng MW (stage2b.export.large_molecule_weight): {part['weight_limit']:g} Da",
             f"Phân tử khớp mẫu peptide: {common.number(part['peptide_molecules'])}; "
             f"trong đó MW < ngưỡng: {common.number(part['peptide_molecules_below_limit'])}",
             f"Cặp có phân tử dạng peptide: {common.number(part['peptide_pairs'])}",
             f"Cặp còn lại sau lọc MW: {common.number(after)}; trong đó có phân tử dạng "
             f"peptide: {common.number(part['peptide_pairs_after_weight_filter'])} "
             f"({common.percent(common.share(part['peptide_pairs_after_weight_filter'], after))})",
             "", "Theo endpoint: " + ", ".join(f"{key} {common.number(value)}"
                                              for key, value in part["by_endpoint"].items()),
             "Theo partition: " + ", ".join(f"{key} {common.number(value)}"
                                           for key, value in part["by_partition"].items()),
             "", "Số mẫu gốc amino acid (N–Cα–C=O) của phân tử dạng peptide có MW < ngưỡng:"]
    lines += bucket_table(part["residue_bins"], part["peptide_molecules_below_limit"])
    lines += [f"Không đọc được: {common.number(part['residue_unparsed'])}", "",
              "Phân tử dạng peptide, MW < ngưỡng, nhiều cặp nhất (xem bằng mắt):"]
    lines += common.table_lines(
        ["SMILES", "MW", "nguyên tử nặng", "gốc", "cặp"],
        [[row["smiles"], f"{row['mw']:.0f}", f"{row['heavy_atoms']:.0f}", row["residues"],
          common.number(row["pairs"])] for row in part["top_by_pairs"]])
    lines += ["", "Mẫu ngẫu nhiên (cùng điều kiện), để ước lượng tỉ lệ bắt nhầm:"]
    lines += common.table_lines(
        ["SMILES", "MW", "nguyên tử nặng", "gốc", "cặp"],
        [[row["smiles"], f"{row['mw']:.0f}", f"{row['heavy_atoms']:.0f}", row["residues"],
          common.number(row["pairs"])] for row in part["random_sample"]])
    return lines


# --------------------------------------------------------------------------- 3. trong nguồn


def assay_descriptions(config: Mapping[str, Any], root: Path,
                       observation_ids: Sequence[str]) -> dict[str, str | None]:
    """Mô tả assay (cột audit.columns.assay_description) của các observation được chọn."""
    mapping = {**config["stage2b"]["columns"], **config["audit"]["columns"]}
    id_column, text_column = mapping["observation_id"], mapping["assay_description"]
    wanted = pa.array(sorted(set(observation_ids)), pa.string())
    found: dict[str, str | None] = {}
    for path in common.observation_files(config, root).values():
        part = pq.read_table(path, columns=[id_column, text_column],
                             filters=pc.is_in(pc.field(id_column), value_set=wanted))
        found.update(zip(part.column(id_column).to_pylist(),
                         part.column(text_column).to_pylist()))
    return found


def in_source_copies(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Phần 3: nhóm trùng trong cùng một nguồn, cùng tài liệu, khác assay.

    Với mỗi cặp assay cùng xuất hiện trong ít nhất một nhóm trùng: shared = số nhóm
    chung; overlap = shared / số nhóm khóa của assay nhỏ hơn. overlap = 1 nghĩa là mọi
    phép đo của assay nhỏ đều có bản giống hệt (cùng phân tử, protein, endpoint,
    qualifier, giá trị) ở assay kia: dấu hiệu assay bị nhập hai lần.
    """
    dup_settings = config["audit"]["duplicates"]
    settings = config["audit"]["followup"]
    table = common.read_observations(config, root, duplicates.READ_ROLES)
    documents = duplicates.document_key(table, config["stage2a"]["dedup"])
    codes, valid = duplicates.key_arrays(table, documents, int(dup_settings["value_decimals"]))
    assays = duplicates.assay_key(table)
    assay_codes = common.encode(assays)
    source_codes = common.encode(table.column("source"))
    usable = common.usable_mask(table, config, require_keep=True)
    level = dup_settings["key_levels"][0]
    rows = duplicates.level_rows(level, usable, valid)
    ids = common.group_ids([codes[column][rows] for column in level["columns"]])
    LOGGER.info("phần 3: %s dòng dùng được, khóa %s", f"{rows.size:,}", level["name"])

    one_source = common.distinct_per_group(ids, source_codes[rows]) == 1
    many_assays = common.distinct_per_group(ids, assay_codes[rows]) >= 2
    target = one_source & many_assays
    assay_groups = common.distinct_per_group(assay_codes[rows], ids)

    chosen = np.flatnonzero(target[ids])
    unique_pairs = np.unique(np.stack([ids[chosen], assay_codes[rows][chosen]], axis=1), axis=0)
    shared: Counter = Counter()
    bounds = np.flatnonzero(np.diff(unique_pairs[:, 0])) + 1
    for block in np.split(unique_pairs[:, 1], bounds):
        for left, right in combinations(sorted(block.tolist()), 2):
            shared[(left, right)] += 1

    # Một dòng đại diện cho mỗi assay (dòng nào cũng được: mô tả assay giống nhau).
    representative = np.full(int(assay_codes.max()) + 1, -1, dtype=np.int64)
    representative[assay_codes[rows]] = rows
    source_column = table.column("source")
    edges = list(settings["overlap_bins"])
    pair_rows = []
    for (left, right), count in shared.items():
        smaller = min(int(assay_groups[left]), int(assay_groups[right]))
        pair_rows.append((left, right, count, smaller, count / smaller,
                          source_column[int(representative[left])].as_py()))
    overlaps = np.array([row[4] for row in pair_rows], dtype=np.float64)
    full = [row for row in pair_rows if row[2] == row[3]]

    observation_ids = table.column("observation_id")
    texts = assay_descriptions(config, root, [
        observation_ids[int(representative[code])].as_py()
        for row in pair_rows for code in row[:2]])

    def text_of(code: int) -> str | None:
        return texts.get(observation_ids[int(representative[code])].as_py())

    def same_text(left: int, right: int) -> bool:
        """Hai mô tả giống hệt; thiếu mô tả ở một bên thì không tính là giống."""
        first, second = text_of(left), text_of(right)
        return first is not None and first == second

    text_match = Counter((row[5], row[2] == row[3], same_text(row[0], row[1]))
                         for row in pair_rows)
    examples = int(settings["examples"])
    width = int(settings["description_chars"])
    sample = []
    for left, right, count, smaller, _, source in sorted(full, key=lambda r: -r[2])[:examples]:
        sample.append({
            "source": source, "shared_groups": count,
            "assays": [{"assay_key": assays[int(representative[code])].as_py(),
                        "groups": int(assay_groups[code]),
                        "document": documents[int(representative[code])].as_py(),
                        "description": (text_of(code) or "")[:width]}
                       for code in (left, right)]})
    return {
        "usable_rows": int(rows.size), "level": level["name"],
        "target_groups": int(target.sum()),
        "target_extra_rows": int(np.count_nonzero(target[ids]) - target.sum()),
        "assay_pairs": len(pair_rows),
        "overlap_bins": common.bucket_counts(overlaps, edges),
        "full_containment_pairs": len(full),
        "full_by_source": common.count_values(row[5] for row in full),
        "description_match": [{"source": key[0], "full": key[1], "same_text": key[2],
                               "pairs": value} for key, value in sorted(text_match.items())],
        "examples": sample,
    }


def in_source_lines(part: Mapping[str, Any]) -> list[str]:
    """Phần báo cáo đọc được của mục 3."""
    lines = ["## 3. Trùng trong cùng một nguồn, cùng tài liệu, khác assay", "",
             f"Dòng dùng được và được giữ: {common.number(part['usable_rows'])}; khóa "
             f"{part['level']}",
             f"Nhóm trùng một nguồn, ≥ 2 assay: {common.number(part['target_groups'])}; "
             f"dòng thừa: {common.number(part['target_extra_rows'])}",
             f"Cặp assay dính nhau: {common.number(part['assay_pairs'])}; trong đó assay nhỏ "
             f"nằm TRỌN trong assay kia: {common.number(part['full_containment_pairs'])}",
             "Nằm trọn, theo nguồn: " + ", ".join(
                 f"{key} {common.number(value)}" for key, value in part["full_by_source"].items()),
             "", "overlap = số nhóm chung / số nhóm khóa của assay nhỏ hơn (1 = nằm trọn):"]
    lines += bucket_table(part["overlap_bins"], part["assay_pairs"])
    lines += ["", "Mô tả assay có giống hệt nhau không (theo nguồn, nằm trọn hay không):"]
    lines += common.table_lines(
        ["nguồn", "nằm trọn", "mô tả giống hệt", "cặp assay"],
        [[row["source"], row["full"], row["same_text"], common.number(row["pairs"])]
         for row in part["description_match"]])
    return lines


# --------------------------------------------------------------------------- 4. Lipinski


def lipinski_properties(smiles: Sequence[str]) -> np.ndarray:
    """Bốn đại lượng của Lipinski cho từng SMILES, cột theo LIPINSKI_METRICS.

    MW: Descriptors.MolWt; logP: Crippen.MolLogP (Wildman & Crippen 1999), KHÔNG phải
    CLogP hay MLogP mà bài Lipinski dùng; HBD/HBA: CalcNumLipinskiHBD/HBA của RDKit (đếm
    NH + OH và N + O, đúng cách đếm của bài). SMILES không đọc được -> NaN ở mọi cột.
    Hàm ở mức module để chạy được trong tiến trình con.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    result = np.full((len(smiles), len(LIPINSKI_METRICS)), np.nan, dtype=np.float64)
    for index, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text) if text else None
        if molecule is None:
            continue
        result[index] = (Descriptors.MolWt(molecule), Crippen.MolLogP(molecule),
                         rdMolDescriptors.CalcNumLipinskiHBD(molecule),
                         rdMolDescriptors.CalcNumLipinskiHBA(molecule))
    return result


def lipinski_all(smiles: list[str], settings: Mapping[str, Any]) -> np.ndarray:
    """lipinski_properties cho cả danh sách, chia khối và chạy song song.

    settings: audit.molecules (dùng chung workers và chunk_size với audit kích thước).
    """
    size = int(settings["chunk_size"])
    chunks = [smiles[start:start + size] for start in range(0, len(smiles), size)]
    workers = int(settings["workers"])
    if workers <= 1 or len(chunks) <= 1:
        parts = [lipinski_properties(chunk) for chunk in chunks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(lipinski_properties, chunks))
    if not parts:
        return np.zeros((0, len(LIPINSKI_METRICS)), dtype=np.float64)
    return np.vstack(parts)


def lipinski_exceeded(properties: np.ndarray, limits: Mapping[str, Any]) -> np.ndarray:
    """Ma trận bool: phân tử vượt từng tiêu chí (cột theo LIPINSKI_METRICS).

    "Vượt" theo đúng chữ của bài: lớn hơn ngưỡng (MW > 500, logP > 5, HBD > 5, HBA > 10).
    """
    bounds = np.array([float(limits[LIPINSKI_LIMIT_KEYS[name]]) for name in LIPINSKI_METRICS])
    with np.errstate(invalid="ignore"):
        return properties > bounds


def violation_counts(properties: np.ndarray, limits: Mapping[str, Any]) -> np.ndarray:
    """Số tiêu chí Lipinski bị vượt (0-4) của từng phân tử; -1 nếu không đọc được."""
    counts = lipinski_exceeded(properties, limits).sum(axis=1).astype(np.int64)
    counts[np.isnan(properties).any(axis=1)] = -1
    return counts


def pair_violations(violations: np.ndarray, index_a: np.ndarray,
                    index_b: np.ndarray) -> np.ndarray:
    """Số tiêu chí bị vượt của phía tệ hơn trong mỗi cặp; -1 nếu một phía không đọc được."""
    left, right = violations[index_a], violations[index_b]
    worst = np.maximum(left, right)
    worst[(left < 0) | (right < 0)] = -1
    return worst


def ordered_counts(values: np.ndarray) -> dict[str, int]:
    """Đếm theo giá trị nguyên, sắp theo giá trị (không theo tần suất)."""
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(keys, counts)}


def lipinski_check(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Phần 4: bao nhiêu phân tử và cặp evaluator nằm trong quy tắc 5 của Lipinski.

    Chỉ đếm, không lọc. Báo số tiêu chí bị vượt (0-4) của từng phân tử, của phía tệ hơn
    trong mỗi cặp, số phân tử vượt từng tiêu chí, và phân bố MW theo các mốc
    audit.followup.lipinski.weight_edges.
    """
    limits = config["audit"]["followup"]["lipinski"]
    table = pq.read_table(common.dataset_file(config, root, "evaluator"),
                          columns=list(PEPTIDE_COLUMNS))
    unique = molecules.unique_strings(table.column("smiles_a"), table.column("smiles_b"))
    LOGGER.info("phần 4: tính Lipinski cho %s phân tử", f"{len(unique):,}")
    properties = lipinski_all(unique.to_pylist(), config["audit"]["molecules"])
    violations = violation_counts(properties, limits)
    exceeded = lipinski_exceeded(properties, limits)

    index_a = molecules.positions(table.column("smiles_a"), unique)
    index_b = molecules.positions(table.column("smiles_b"), unique)
    if (index_a < 0).any() or (index_b < 0).any():
        raise ValueError("có SMILES rỗng trong evaluator; dữ liệu không đúng như Stage 2b")
    worst = pair_violations(violations, index_a, index_b)
    weight = properties[:, LIPINSKI_METRICS.index("mw")]
    pair_weight = np.fmax(weight[index_a], weight[index_b])
    edges = [float(edge) for edge in limits["weight_edges"]]
    partitions = table.column("partition").to_numpy(zero_copy_only=False)

    by_partition = {}
    for partition in sorted(set(partitions.tolist())):
        chosen = worst[partitions == partition]
        by_partition[str(partition)] = {
            "pairs": int(chosen.size), "zero": int(np.count_nonzero(chosen == 0)),
            "at_most_one": int(np.count_nonzero((chosen >= 0) & (chosen <= 1)))}
    return {
        "molecules": len(unique),
        "unparsed_molecules": int(np.count_nonzero(violations < 0)),
        "molecule_violations": ordered_counts(violations),
        "molecules_exceeding": {name: int(np.count_nonzero(exceeded[:, column]))
                                for column, name in enumerate(LIPINSKI_METRICS)},
        "pairs": table.num_rows,
        "pair_worst_violations": ordered_counts(worst),
        "by_partition": by_partition,
        "weight_edges": edges,
        "molecule_weight": closed_buckets(weight, edges),
        "pair_weight": closed_buckets(pair_weight, edges),
        "limits": {name: limits[LIPINSKI_LIMIT_KEYS[name]] for name in LIPINSKI_METRICS},
    }


def lipinski_lines(part: Mapping[str, Any]) -> list[str]:
    """Phần báo cáo đọc được của mục 4."""
    molecules_total, pairs = part["molecules"], part["pairs"]
    limits = part["limits"]

    def violation_table(counts: Mapping[str, int], total: int) -> list[str]:
        return common.table_lines(
            ["số tiêu chí vượt", "số", "%"],
            [["không đọc được" if key == "-1" else key, common.number(value),
              common.percent(common.share(value, total))] for key, value in counts.items()])

    lines = ["## 4. Quy tắc 5 của Lipinski (chỉ đếm, không lọc)", "",
             f"Ngưỡng: MW > {limits['mw']:g}, logP > {limits['logp']:g}, "
             f"HBD > {limits['donors']:g}, HBA > {limits['acceptors']:g} "
             "(Lipinski et al. 2001). logP là Crippen của RDKit, không phải CLogP/MLogP.",
             f"Phân tử: {common.number(molecules_total)}; cặp evaluator: {common.number(pairs)}",
             "", "Phân tử theo số tiêu chí bị vượt:"]
    lines += violation_table(part["molecule_violations"], molecules_total)
    lines += ["", "Phân tử vượt từng tiêu chí:"]
    lines += common.table_lines(
        ["tiêu chí", "số", "%"],
        [[name, common.number(value), common.percent(common.share(value, molecules_total))]
         for name, value in part["molecules_exceeding"].items()])
    lines += ["", "Cặp theo số tiêu chí bị vượt của phía tệ hơn:"]
    lines += violation_table(part["pair_worst_violations"], pairs)
    lines += ["", "Theo partition (cặp có cả hai phía vượt 0 tiêu chí / tối đa 1 tiêu chí):"]
    lines += common.table_lines(
        ["partition", "cặp", "0 tiêu chí", "%", "≤ 1 tiêu chí", "%"],
        [[name, common.number(row["pairs"]), common.number(row["zero"]),
          common.percent(common.share(row["zero"], row["pairs"])),
          common.number(row["at_most_one"]),
          common.percent(common.share(row["at_most_one"], row["pairs"]))]
         for name, row in part["by_partition"].items()])
    lines += ["", "MW của phân tử:"]
    lines += bucket_table(part["molecule_weight"], molecules_total)
    lines += ["", "MW lớn hơn trong hai phía của cặp:"]
    lines += bucket_table(part["pair_weight"], pairs)
    return lines


# --------------------------------------------------------------------------- chạy


PARTS = {
    "relative": (relative_duplicates, relative_lines),
    "peptide": (peptide_check, peptide_lines),
    "in_source": (in_source_copies, in_source_lines),
    "lipinski": (lipinski_check, lipinski_lines),
}


def run(config: Mapping[str, Any], root: Path,
        parts: Sequence[str] = tuple(PARTS)) -> tuple[dict[str, Any], list[str]]:
    """Chạy các phần được chọn (mặc định mọi phần). Trả về (số liệu, các dòng báo cáo).

    Chạy riêng từng phần khi máy ít RAM: in_source đọc mọi observation (nặng nhất),
    relative đọc evidence_records, peptide và lipinski chỉ đọc 4 cột của evaluator.
    """
    unknown = [part for part in parts if part not in PARTS]
    if unknown:
        raise KeyError(f"phần không biết: {unknown}; có {list(PARTS)}")
    payload: dict[str, Any] = {}
    lines = ["# Audit bổ sung trước dataset_v3 (chỉ đọc, không sửa dữ liệu)", ""]
    for part in parts:
        compute, describe = PARTS[part]
        payload[part] = compute(config, root)
        lines += describe(payload[part]) + [""]
    lines += ["Ví dụ chi tiết (khóa, SMILES, mô tả assay) nằm trong file .json cùng tên."]
    return payload, lines
