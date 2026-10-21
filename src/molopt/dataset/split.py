"""Stage 2b: dựng split protein trên dataset cuối (sau bước 4).

Thứ tự: gộp replicate → MMP → Δp trong assay → gộp Δp giữa assay → DỰNG SPLIT → xuất.
Split dựng trên protein thực sự có trong dataset cuối (bản ghi bước 4), vì protein không
tạo được cặp nào không cần chia và chỉ làm lệch tỷ lệ.

Các phần ở đây không phụ thuộc công cụ gom chuỗi:
    - đếm số cặp (bản ghi bước 4) của mỗi protein;
    - ghi FASTA đầu vào cho công cụ gom;
    - xếp nhóm protein vào partition theo số cặp (nhóm lớn trước, seed phá thế hòa);
    - bảng protein_split, manifest, và các kiểm tra của verify.
Bước gom chuỗi (protein -> nhóm ở ngưỡng identity) do MMseqs2 chạy ngoài dự án, không
viết lại ở đây: đó là việc của công cụ chuẩn. Tham số đã chốt ngày 21/10/2026 nằm trong
config `stage2b.split.clustering` (identity 50 % trên số cột alignment, connected
component, coverage 0,8 trên chuỗi ngắn hơn); xem docs/STAGE2B_DATASET.md mục 8.1. Module
này chỉ nhận lại bảng nhóm dạng TSV hai cột.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

SPLIT_SCHEMA = pa.schema([
    ("protein_sequence_hash", pa.string()),
    ("protein_cluster_id", pa.string()),
    ("partition", pa.string()),
    ("cluster_proteins", pa.int64()),
    ("cluster_pairs", pa.int64()),
    ("pairs", pa.int64()),
])
CLUSTER_ID_PREFIX = "pcl_"
CLUSTER_ID_LENGTH = 20


def protein_pair_counts(protein_hashes: Iterable[pa.ChunkedArray | pa.Array]) -> Counter:
    """Số bản ghi bước 4 (số cặp) của mỗi protein, cộng dồn qua nhiều bảng."""
    counts: Counter = Counter()
    for column in protein_hashes:
        table = pa.table({"protein_hash": column}).group_by("protein_hash").aggregate(
            [("protein_hash", "count")])
        counts.update(dict(zip(table["protein_hash"].to_pylist(),
                               table["protein_hash_count"].to_pylist())))
    return counts


def write_fasta(sequences: Mapping[str, str], hashes: Iterable[str], path: Path,
                line_width: int = 60) -> int:
    """Ghi FASTA, tiêu đề là protein_sequence_hash; trả về số chuỗi đã ghi.

    Nhận: {hash: chuỗi}, các hash cần ghi (theo thứ tự tăng dần), đường dẫn.
    Thiếu chuỗi cho một hash là lỗi: protein trong dataset luôn phải có chuỗi.
    """
    written = 0
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for protein in sorted(hashes):
            sequence = sequences.get(protein)
            if not sequence:
                raise ValueError(f"protein {protein} có trong dataset nhưng không có chuỗi")
            handle.write(f">{protein}\n")
            for start in range(0, len(sequence), line_width):
                handle.write(sequence[start:start + line_width] + "\n")
            written += 1
    return written


def cluster_id(members: Iterable[str]) -> str:
    """ID ổn định của một nhóm, chỉ phụ thuộc tập protein trong nhóm (không phụ thuộc
    công cụ hay protein đại diện mà công cụ chọn)."""
    digest = hashlib.sha256("|".join(sorted(members)).encode("ascii")).hexdigest()
    return CLUSTER_ID_PREFIX + digest[:CLUSTER_ID_LENGTH]


def assign_partitions(cluster_pairs: Mapping[str, int], ratios: Mapping[str, float],
                      partitions: Sequence[str], seed: int) -> dict[str, str]:
    """Xếp từng nhóm vào một partition sao cho số cặp gần tỷ lệ mục tiêu.

    Nhận: {nhóm: số cặp}; {partition: tỷ lệ}; thứ tự partition (dùng khi hòa); seed.
    Trả về: {nhóm: partition}.
    Cách xếp (tham lam, nhóm lớn trước): nhóm được duyệt theo số cặp giảm dần, hòa thì
    theo thứ tự xáo bằng seed; mỗi nhóm vào partition còn thiếu nhiều nhất so với mục
    tiêu (mục tiêu = tỷ lệ × tổng số cặp). Nhóm lớn đặt trước vì đặt sau thì không còn
    chỗ để bù, tỷ lệ cuối sẽ lệch.
    """
    total = sum(cluster_pairs.values())
    targets = {partition: ratios[partition] * total for partition in partitions}
    filled = dict.fromkeys(partitions, 0)
    rng = random.Random(seed)
    names = sorted(cluster_pairs)
    rng.shuffle(names)
    tie_order = {name: position for position, name in enumerate(names)}
    ordered = sorted(cluster_pairs, key=lambda name: (-cluster_pairs[name], tie_order[name]))
    assignment = {}
    for name in ordered:
        best = max(partitions, key=lambda partition: (targets[partition] - filled[partition],
                                                      -partitions.index(partition)))
        assignment[name] = best
        filled[best] += cluster_pairs[name]
    return assignment


def build_split_table(protein_pairs: Mapping[str, int], membership: Mapping[str, str],
                      ratios: Mapping[str, float], partitions: Sequence[str],
                      seed: int) -> pa.Table:
    """Bảng protein_split: một dòng mỗi protein.

    Nhận: {protein: số cặp}; {protein: nhóm do công cụ gom trả về (tên bất kỳ)};
    tỷ lệ; thứ tự partition; seed.
    Trả về: bảng theo SPLIT_SCHEMA; protein_cluster_id là cluster_id() của tập thành viên.
    Protein có cặp mà không có nhóm là lỗi (công cụ gom bỏ sót chuỗi).
    """
    missing = sorted(set(protein_pairs) - set(membership))
    if missing:
        raise ValueError(f"{len(missing)} protein có cặp nhưng không có nhóm, vd. {missing[:3]}")
    members: dict[str, list[str]] = defaultdict(list)
    for protein in protein_pairs:
        members[membership[protein]].append(protein)
    ids = {name: cluster_id(proteins) for name, proteins in members.items()}
    sizes = {ids[name]: sum(protein_pairs[p] for p in proteins)
             for name, proteins in members.items()}
    assignment = assign_partitions(sizes, ratios, partitions, seed)
    rows = sorted(protein_pairs)
    clusters = [ids[membership[protein]] for protein in rows]
    return pa.table({
        "protein_sequence_hash": rows,
        "protein_cluster_id": clusters,
        "partition": [assignment[cluster] for cluster in clusters],
        "cluster_proteins": [len(members[membership[protein]]) for protein in rows],
        "cluster_pairs": [sizes[cluster] for cluster in clusters],
        "pairs": [protein_pairs[protein] for protein in rows],
    }, schema=SPLIT_SCHEMA)


# --------------------------------------------------------------------------- kiểm tra


def check_clusters_single_partition(split: pa.Table) -> dict[str, Any]:
    """Kiểm tra 1: không nhóm nào nằm ở hai partition."""
    grouped = split.group_by("protein_cluster_id").aggregate([("partition", "count_distinct")])
    bad = grouped.filter(pc.greater(grouped["partition_count_distinct"], 1))
    return {"passed": bad.num_rows == 0, "clusters_in_multiple_partitions": bad.num_rows,
            "examples": bad["protein_cluster_id"].to_pylist()[:5]}


def check_proteins_single_partition(split: pa.Table) -> dict[str, Any]:
    """Kiểm tra 2: không chuỗi protein nào xuất hiện ở hai partition (hay hai dòng)."""
    grouped = split.group_by("protein_sequence_hash").aggregate(
        [("partition", "count_distinct"), ("partition", "count")])
    multi_partition = pc.sum(pc.greater(grouped["partition_count_distinct"], 1)).as_py() or 0
    duplicated = pc.sum(pc.greater(grouped["partition_count"], 1)).as_py() or 0
    return {"passed": multi_partition == 0 and duplicated == 0,
            "proteins_in_multiple_partitions": multi_partition,
            "proteins_with_multiple_rows": duplicated}


def check_pair_totals(split: pa.Table, dataset_pairs: Mapping[str, int]) -> dict[str, Any]:
    """Kiểm tra 3: tổng số cặp theo partition bằng tổng số cặp của dataset.

    dataset_pairs: {protein: số bản ghi} đếm lại trực tiếp từ dataset, không lấy từ split.
    """
    in_split = dict(zip(split["protein_sequence_hash"].to_pylist(), split["pairs"].to_pylist()))
    missing = sorted(set(dataset_pairs) - set(in_split))
    extra = sorted(set(in_split) - set(dataset_pairs))
    mismatched = sorted(p for p in set(in_split) & set(dataset_pairs)
                        if in_split[p] != dataset_pairs[p])
    by_partition = Counter()
    for protein, partition in zip(split["protein_sequence_hash"].to_pylist(),
                                  split["partition"].to_pylist()):
        by_partition[partition] += dataset_pairs.get(protein, 0)
    total = sum(dataset_pairs.values())
    return {"passed": not missing and not extra and not mismatched
            and sum(by_partition.values()) == total,
            "dataset_pairs": total, "pairs_by_partition": dict(by_partition),
            "proteins_missing_from_split": len(missing), "proteins_not_in_dataset": len(extra),
            "proteins_with_wrong_pair_count": len(mismatched)}


def ratio_report(pairs_by_partition: Mapping[str, int], ratios: Mapping[str, float]
                 ) -> dict[str, dict[str, float]]:
    """Kiểm tra 4: tỷ lệ thực tế theo số cặp so với mục tiêu, và độ lệch (điểm %)."""
    total = sum(pairs_by_partition.values())
    return {partition: {"target": target,
                        "actual": pairs_by_partition.get(partition, 0) / total if total else 0.0,
                        "deviation": (pairs_by_partition.get(partition, 0) / total - target
                                      if total else 0.0)}
            for partition, target in ratios.items()}


def compare_old_split(split: pa.Table, old: pa.Table) -> dict[str, Any]:
    """Kiểm tra 5 (chỉ báo cáo): protein có ở cả hai split, bao nhiêu % cùng partition.

    old: bảng của split cũ với hai cột target_sequence_hash, partition.
    """
    previous = dict(zip(old["target_sequence_hash"].to_pylist(), old["partition"].to_pylist()))
    shared = same = 0
    crosstab: Counter = Counter()
    for protein, partition in zip(split["protein_sequence_hash"].to_pylist(),
                                  split["partition"].to_pylist()):
        if protein in previous:
            shared += 1
            same += int(previous[protein] == partition)
            crosstab[f"{previous[protein]} -> {partition}"] += 1
    return {"proteins_new": split.num_rows, "proteins_old": old.num_rows,
            "proteins_shared": shared, "same_partition": same,
            "same_partition_share": same / shared if shared else None,
            "old_to_new": dict(sorted(crosstab.items()))}
