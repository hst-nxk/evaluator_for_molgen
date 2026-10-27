"""Audit kích thước phân tử: phân tử lớn/peptide và tác động của ngưỡng small molecule.

Câu hỏi của buổi họp 26/10/2026: bài toán chỉ làm small molecule, nên phân tử lớn và dạng
peptide là nhiễu. Trước khi chọn ngưỡng lọc, cần biết: phân bố kích thước thực tế, và với
mỗi ngưỡng ứng viên thì mất bao nhiêu cặp ở từng partition. Script KHÔNG lọc gì; ngưỡng
trong audit.molecules.cutoffs chỉ là các mức để báo cáo.

Đo trên evaluator_training (các cặp đi vào train/validation/test):
    - mỗi phân tử khác nhau: nguyên tử nặng, tổng nguyên tử kể cả H, MW, liên kết quay,
      dạng peptide (audit.molecules.peptide_smarts);
    - mỗi nhóm thế (variable_a/variable_b): nguyên tử nặng, và tỉ lệ so với phân tử;
    - một cặp bị loại ở một ngưỡng khi A HOẶC B vượt ngưỡng.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import common, plots

LOGGER = logging.getLogger(__name__)
METRICS = ("heavy_atoms", "total_atoms", "mw", "rotatable_bonds", "peptide_like")
READ_COLUMNS = ["smiles_a", "smiles_b", "variable_a", "variable_b", "partition"]


def describe(smiles: Sequence[str], peptide_smarts: str) -> np.ndarray:
    """Các chỉ số kích thước của từng SMILES (một hàng mỗi SMILES, cột theo METRICS).

    SMILES không đọc được cho NaN ở mọi cột. Nguyên tử giả '*' của nhóm thế có số hiệu 0,
    nên không được tính là nguyên tử nặng và không góp vào MW.
    Hàm ở mức module để chạy được trong tiến trình con (ProcessPoolExecutor).
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    pattern = Chem.MolFromSmarts(peptide_smarts)
    if pattern is None:
        raise ValueError(f"peptide_smarts không hợp lệ: {peptide_smarts}")
    result = np.full((len(smiles), len(METRICS)), np.nan, dtype=np.float64)
    for index, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text) if text else None
        if molecule is None:
            continue
        hydrogens = sum(atom.GetTotalNumHs() for atom in molecule.GetAtoms())
        result[index] = (
            molecule.GetNumHeavyAtoms(),
            molecule.GetNumHeavyAtoms() + hydrogens,
            Descriptors.MolWt(molecule),
            rdMolDescriptors.CalcNumRotatableBonds(molecule),
            float(molecule.HasSubstructMatch(pattern)),
        )
    return result


def describe_all(smiles: list[str], settings: Mapping[str, Any]) -> np.ndarray:
    """describe() cho cả danh sách, chia khối và chạy song song."""
    size = int(settings["chunk_size"])
    chunks = [smiles[start:start + size] for start in range(0, len(smiles), size)]
    workers = int(settings["workers"])
    if workers <= 1 or len(chunks) <= 1:
        parts = [describe(chunk, settings["peptide_smarts"]) for chunk in chunks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(describe, chunks,
                                  [settings["peptide_smarts"]] * len(chunks)))
    if not parts:
        return np.zeros((0, len(METRICS)), dtype=np.float64)
    return np.vstack(parts)


def unique_strings(*columns: pa.ChunkedArray) -> pa.Array:
    """Các chuỗi khác nhau (không null) của nhiều cột gộp lại."""
    merged = pa.chunked_array([chunk for column in columns for chunk in column.chunks])
    return pc.drop_null(pc.unique(merged.cast(pa.large_string())))


def positions(values: pa.ChunkedArray, unique: pa.Array) -> np.ndarray:
    """Vị trí của từng giá trị trong `unique` (-1 nếu null)."""
    index = pc.index_in(values.cast(pa.large_string()), value_set=unique)
    return index.fill_null(common.NULL_CODE).to_numpy(zero_copy_only=False).astype(np.int64)


def row_metric(table: np.ndarray, index: np.ndarray, column: int) -> np.ndarray:
    """Giá trị một chỉ số cho từng dòng theo vị trí (NaN nếu vị trí -1)."""
    values = np.full(index.size, np.nan, dtype=np.float64)
    found = index >= 0
    values[found] = table[index[found], column]
    return values


def cutoff_impact(pair_values: np.ndarray, partitions: np.ndarray, molecule_values: np.ndarray,
                  cutoffs: Sequence[float], names: Sequence[str]) -> list[dict[str, Any]]:
    """Với mỗi ngưỡng: số phân tử vượt ngưỡng, số cặp bị loại (tổng và theo partition).

    `pair_values` là giá trị lớn hơn của A và B ở mỗi cặp; cặp có NaN (không đọc được
    SMILES) không bị tính là vượt ngưỡng và được đếm riêng ở báo cáo.
    """
    rows = []
    for cutoff in cutoffs:
        over = pair_values > cutoff
        row: dict[str, Any] = {
            "cutoff": cutoff,
            "molecules_over": int(np.count_nonzero(molecule_values > cutoff)),
            "pairs_removed": int(over.sum()),
            "pairs_removed_percent": common.share(int(over.sum()), pair_values.size),
        }
        for name in names:
            in_partition = partitions == name
            removed = int(np.count_nonzero(over & in_partition))
            row[name] = {"removed": removed,
                         "percent": common.share(removed, int(in_partition.sum()))}
        rows.append(row)
    return rows


def run(config: Mapping[str, Any], root: Path, figures: Path,
        output_name: str) -> tuple[dict[str, Any], list[str]]:
    """Đo kích thước phân tử/nhóm thế và tác động của các ngưỡng ứng viên."""
    settings = config["audit"]["molecules"]
    quantiles = config["audit"]["quantiles"]
    partitions_names = list(config["stage2b"]["split"]["partitions"])
    table = pq.read_table(common.dataset_file(config, root, "evaluator"), columns=READ_COLUMNS)
    LOGGER.info("đọc %s cặp", f"{table.num_rows:,}")

    molecules = unique_strings(table.column("smiles_a"), table.column("smiles_b"))
    fragments = unique_strings(table.column("variable_a"), table.column("variable_b"))
    LOGGER.info("tính chỉ số cho %s phân tử và %s nhóm thế", f"{len(molecules):,}",
                f"{len(fragments):,}")
    molecule_table = describe_all(molecules.to_pylist(), settings)
    fragment_table = describe_all(fragments.to_pylist(), settings)

    descriptor_path = common.runs_dir(config, root) / f"{output_name}_descriptors.parquet"
    pq.write_table(pa.table({"smiles": molecules, **{
        metric: molecule_table[:, column] for column, metric in enumerate(METRICS)}}),
        descriptor_path)

    index_a = positions(table.column("smiles_a"), molecules)
    index_b = positions(table.column("smiles_b"), molecules)
    fragment_a = positions(table.column("variable_a"), fragments)
    fragment_b = positions(table.column("variable_b"), fragments)
    partitions = table.column("partition").to_numpy(zero_copy_only=False)
    heavy_column = METRICS.index("heavy_atoms")
    peptide_column = METRICS.index("peptide_like")

    payload: dict[str, Any] = {
        "pairs": table.num_rows, "molecules": len(molecules), "fragments": len(fragments),
        "molecules_unparsed": int(np.isnan(molecule_table[:, heavy_column]).sum()),
        "fragments_unparsed": int(np.isnan(fragment_table[:, heavy_column]).sum()),
        "settings": settings, "descriptor_file": str(descriptor_path),
        "molecule_summary": {}, "pair_max_summary": {}, "cutoffs": {},
    }
    for column, metric in enumerate(METRICS):
        payload["molecule_summary"][metric] = common.quantile_summary(
            molecule_table[:, column], quantiles)
        pair_max = np.fmax(row_metric(molecule_table, index_a, column),
                           row_metric(molecule_table, index_b, column))
        payload["pair_max_summary"][metric] = common.quantile_summary(pair_max, quantiles)
        if metric in settings["cutoffs"]:
            payload["cutoffs"][metric] = cutoff_impact(
                pair_max, partitions, molecule_table[:, column],
                list(settings["cutoffs"][metric]), partitions_names)

    peptide_pairs = np.fmax(row_metric(molecule_table, index_a, peptide_column),
                            row_metric(molecule_table, index_b, peptide_column)) > 0
    payload["peptide"] = {
        "molecules": int(np.nansum(molecule_table[:, peptide_column])),
        "pairs": int(peptide_pairs.sum()),
        "pairs_percent": common.share(int(peptide_pairs.sum()), table.num_rows),
        "by_partition": {name: int(np.count_nonzero(peptide_pairs & (partitions == name)))
                         for name in partitions_names},
    }

    fragment_heavy_a = row_metric(fragment_table, fragment_a, heavy_column)
    fragment_heavy_b = row_metric(fragment_table, fragment_b, heavy_column)
    heavy_a = row_metric(molecule_table, index_a, heavy_column)
    heavy_b = row_metric(molecule_table, index_b, heavy_column)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.concatenate([fragment_heavy_a / heavy_a, fragment_heavy_b / heavy_b])
    payload["fragment_summary"] = {
        "heavy_atoms": common.quantile_summary(fragment_table[:, heavy_column], quantiles),
        "heavy_atoms_per_side": common.quantile_summary(
            np.concatenate([fragment_heavy_a, fragment_heavy_b]), quantiles),
        "fraction_of_molecule": common.quantile_summary(ratio, quantiles),
        "heavy_atom_counts": common.bucket_counts(
            np.concatenate([fragment_heavy_a, fragment_heavy_b]),
            list(settings["fragment_bucket_edges"])),
    }

    payload["figures"] = [str(path) for path in draw(figures, settings, molecule_table,
                                                     fragment_heavy_a, fragment_heavy_b,
                                                     ratio)]
    return payload, report_lines(payload)


def draw(figures: Path, settings: Mapping[str, Any], molecule_table: np.ndarray,
         fragment_heavy_a: np.ndarray, fragment_heavy_b: np.ndarray,
         ratio: np.ndarray) -> list[Path]:
    """Hình phân bố kích thước phân tử (kèm các ngưỡng ứng viên) và kích thước nhóm thế.

    Khoảng và số cột của từng hình lấy từ audit.molecules.histograms; trục y log vì phần
    đuôi phân tử lớn rất mỏng so với phần chính.
    """
    ranges = settings["histograms"]
    paths = []
    for metric in ("heavy_atoms", "total_atoms", "mw"):
        span = ranges[metric]
        paths.append(plots.histograms(
            figures / f"molecule_{metric}.png",
            f"{metric}: mỗi phân tử khác nhau tính một lần; vạch đứt là ngưỡng ứng viên",
            [(metric, {"phân tử": molecule_table[:, METRICS.index(metric)]})],
            span["min"], span["max"], int(span["bins"]), metric,
            vlines={metric: list(settings["cutoffs"].get(metric, []))}, log_y=True))
    span = ranges["fragment_heavy_atoms"]
    paths.append(plots.histograms(
        figures / "fragment_heavy_atoms.png", "Nguyên tử nặng của nhóm thế (mỗi phía mỗi cặp)",
        [("nhóm thế", {"phía A": fragment_heavy_a, "phía B": fragment_heavy_b})],
        span["min"], span["max"], int(span["bins"]), "số nguyên tử nặng"))
    span = ranges["fragment_fraction"]
    paths.append(plots.histograms(
        figures / "fragment_fraction.png", "Nhóm thế / phân tử, theo số nguyên tử nặng",
        [("tỉ lệ", {"cả hai phía": ratio})], span["min"], span["max"], int(span["bins"]),
        "tỉ lệ"))
    return paths


def report_lines(payload: Mapping[str, Any]) -> list[str]:
    """Bản báo cáo đọc được."""
    lines = ["# Audit kích thước phân tử (evaluator_training, không lọc gì)", "",
             f"Cặp: {common.number(payload['pairs'])}; phân tử khác nhau: "
             f"{common.number(payload['molecules'])}; nhóm thế khác nhau: "
             f"{common.number(payload['fragments'])}",
             f"SMILES không đọc được: phân tử {common.number(payload['molecules_unparsed'])}, "
             f"nhóm thế {common.number(payload['fragments_unparsed'])}", "",
             "## 1. Phân bố (mỗi phân tử tính một lần)"]
    header = ["chỉ số", "n", "p1", "p50", "p95", "p99", "p99.9", "max"]
    rows = []
    for metric, summary in payload["molecule_summary"].items():
        rows.append([metric, common.number(summary["n"])]
                    + [common.number(summary.get(key)) for key in
                       ("p1", "p50", "p95", "p99", "p99.9", "p100")])
    lines += common.table_lines(header, rows)
    lines += ["peptide_like: tỉ lệ phân tử khớp mẫu khung peptide (cột mean)."]

    peptide = payload["peptide"]
    lines += ["", "## 2. Dạng peptide",
              f"Phân tử dạng peptide: {common.number(peptide['molecules'])}",
              f"Cặp có A hoặc B dạng peptide: {common.number(peptide['pairs'])} "
              f"({common.percent(peptide['pairs_percent'])})",
              "Theo partition: " + ", ".join(f"{name} {common.number(count)}"
                                             for name, count in peptide["by_partition"].items())]

    lines += ["", "## 3. Nếu lọc theo ngưỡng thì mất bao nhiêu cặp",
              "Cặp bị loại khi A HOẶC B vượt ngưỡng.", ""]
    for metric, block in payload["cutoffs"].items():
        names = [key for key in block[0] if isinstance(block[0][key], dict)] if block else []
        lines.append(f"{metric}:")
        lines += common.table_lines(
            ["ngưỡng", "phân tử vượt", "cặp bị loại", "%", *names],
            [[f"{row['cutoff']:g}", common.number(row["molecules_over"]),
              common.number(row["pairs_removed"]), common.percent(row["pairs_removed_percent"]),
              *(f"{common.number(row[name]['removed'])} "
                f"({common.percent(row[name]['percent'])})" for name in names)]
             for row in block])
        lines.append("")

    fragment = payload["fragment_summary"]
    lines += ["## 4. Kích thước nhóm thế (phần thay đổi của phép sửa)"]
    rows = []
    for label, key in (("nguyên tử nặng, mỗi nhóm thế khác nhau", "heavy_atoms"),
                       ("nguyên tử nặng, mỗi phía của mỗi cặp", "heavy_atoms_per_side"),
                       ("nhóm thế / phân tử", "fraction_of_molecule")):
        summary = fragment[key]
        rows.append([label, common.number(summary["n"])]
                    + [common.number(summary.get(item)) for item in
                       ("p1", "p50", "p95", "p99", "p100")])
    lines += common.table_lines(["chỉ số", "n", "p1", "p50", "p95", "p99", "max"], rows)
    lines.append("Số nguyên tử nặng của nhóm thế (mỗi phía của mỗi cặp):")
    lines += common.table_lines(
        ["khoảng", "số", "%"],
        [[row["range"], common.number(row["count"]), common.percent(row.get("percent"))]
         for row in fragment["heavy_atom_counts"]])
    lines += ["", f"Chỉ số từng phân tử: {payload['descriptor_file']}",
              "Hình: " + ", ".join(payload["figures"])]
    return lines
