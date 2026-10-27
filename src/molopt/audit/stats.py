"""Thống kê tổng quan dữ liệu và danh sách điểm bất thường, kèm hình minh họa.

Yêu cầu của buổi họp 26/10/2026:
    - thống kê lại số lượng, độ đo, chỉ số của toàn bộ dữ liệu để xem có gì bất thường;
    - bài toán là regression nên phải vẽ phân bố của bốn nhãn (Ki, Kd, IC50, EC50);
    - độ phủ protein, cluster protein, kích thước phân tử (phần phân tử ở molecules.py);
    - điểm bất thường chỉ được ĐÁNH DẤU, không xóa, để truy vết được về nguồn.

Ba tầng dữ liệu:
    observations   dòng dùng được và được giữ sau dedup (đầu vào Stage 2b)
    evaluator      evaluator_training (cặp và nhãn Δp đi vào train/validation/test)
    split          protein_split (protein, cluster, partition)
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

from . import common, plots
from .duplicates import assay_key

LOGGER = logging.getLogger(__name__)
OBSERVATION_ROLES = ("source", "keep", "inchikey", "smiles", "endpoint", "protein_hash",
                     "qualifier", "p_activity", "assay_id", "assay_entry_id",
                     "protein_status", "assay_status")
EVALUATOR_COLUMNS = ["endpoint", "partition", "protein_hash", "delta_p_mean", "n_assays",
                     "n_documents", "qc_status", "min_target_confidence",
                     "p_a_range", "max_replicate_std", "unseen_pair", "plausible_range",
                     "short_protein", "inactive_comment", "nonstandard_sequence",
                     "source_quality_warning"]
FLAG_COLUMNS = ("unseen_pair", "short_protein", "inactive_comment", "nonstandard_sequence")


def value_counts(values: pa.ChunkedArray) -> dict[str, int]:
    """Đếm từng giá trị của một cột pyarrow (null ghi là 'null'), sắp giảm dần."""
    counts = pc.value_counts(values.combine_chunks()).to_pylist()
    pairs = [("null" if item["values"] is None else str(item["values"]), item["counts"])
             for item in counts]
    return dict(sorted(pairs, key=lambda pair: -pair[1]))


def per_group_counts(codes: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Số giá trị khác nhau của `codes` trong mỗi nhóm `groups` (bỏ nhóm không có dòng)."""
    if codes.size == 0:
        return np.zeros(0, dtype=np.int64)
    counts = common.distinct_per_group(common.group_ids([groups]), codes)
    return counts[counts > 0]


# --------------------------------------------------------------------------- observations


def observation_stats(config: Mapping[str, Any], root: Path,
                      quantiles: list[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Thống kê dòng observation dùng được; trả về (số liệu, dữ liệu để vẽ)."""
    table = common.read_observations(config, root, OBSERVATION_ROLES)
    usable = common.usable_mask(table, config, require_keep=True)
    sources = table.column("source").to_numpy(zero_copy_only=False)
    endpoints = table.column("endpoint").to_numpy(zero_copy_only=False)
    p_values = table.column("p_activity").to_numpy(zero_copy_only=False).astype(np.float64)
    assays = common.encode(assay_key(table))
    molecules = common.encode(table.column("inchikey"))
    proteins = common.encode(table.column("protein_hash"))
    low, high = config["stage2b"]["export"]["plausible_p_range"]

    result: dict[str, Any] = {"rows_read": table.num_rows, "usable_rows": int(usable.sum()),
                              "by_source_endpoint": {}, "p_activity": {}, "anomalies": {}}
    for source in sorted(set(sources.tolist())):
        result["by_source_endpoint"][source] = {
            endpoint: int(np.count_nonzero(usable & (sources == source)
                                           & (endpoints == endpoint)))
            for endpoint in config["endpoints"]}
    figure_series = {}
    for endpoint in config["endpoints"]:
        mask = usable & (endpoints == endpoint)
        result["p_activity"][endpoint] = common.quantile_summary(p_values[mask], quantiles)
        figure_series[endpoint] = {source: p_values[mask & (sources == source)]
                                   for source in sorted(set(sources.tolist()))}
    usable_p = p_values[usable]
    result["anomalies"] = {
        "p < 0 (nồng độ trên 1 mol/L, vô lý về vật lý)": int(np.count_nonzero(usable_p < 0)),
        f"p < {low:g} (khoảng plausible_p_range của pipeline, chưa có nguồn)": int(
            np.count_nonzero(usable_p < low)),
        f"p > {high:g} (khoảng plausible_p_range của pipeline, chưa có nguồn)": int(
            np.count_nonzero(usable_p > high)),
    }
    rows = np.flatnonzero(usable)
    result["distinct"] = {"molecules": int(np.unique(molecules[rows]).size),
                          "proteins": int(np.unique(proteins[rows]).size),
                          "assays": int(np.unique(assays[rows]).size)}
    result["per_assay_rows"] = common.quantile_summary(
        np.bincount(common.group_ids([assays[rows]])), quantiles)
    result["per_assay_molecules"] = common.quantile_summary(
        per_group_counts(molecules[rows], assays[rows]), quantiles)
    result["per_protein_assays"] = common.quantile_summary(
        per_group_counts(assays[rows], proteins[rows]), quantiles)
    result["per_protein_molecules"] = common.quantile_summary(
        per_group_counts(molecules[rows], proteins[rows]), quantiles)
    return result, figure_series


# --------------------------------------------------------------------------- evaluator


def evaluator_stats(config: Mapping[str, Any], root: Path, settings: Mapping[str, Any],
                    quantiles: list[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Thống kê evaluator_training; trả về (số liệu, dữ liệu để vẽ)."""
    table = pq.read_table(common.dataset_file(config, root, "evaluator"),
                          columns=EVALUATOR_COLUMNS)
    numeric = ("endpoint", "partition", "protein_hash", "delta_p_mean", "n_assays",
               "n_documents", "p_a_range", "max_replicate_std")
    data = {name: table.column(name).to_numpy(zero_copy_only=False) for name in numeric}
    warning = common.boolean(pc.is_valid(table.column("source_quality_warning")))
    plausible = common.boolean(table.column("plausible_range"))
    delta = data["delta_p_mean"].astype(np.float64)
    partitions = list(config["stage2b"]["split"]["partitions"])
    rows = table.num_rows
    levels = config["audit"]["literature"]
    tolerance = float(config["stage2b"]["qc"]["collapse_tolerance"])
    within, meaningful = float(levels["within_error"]), float(levels["meaningful"])
    unit_copy = float(levels["unit_copy"])
    edges = [0.0, tolerance, within, meaningful, unit_copy]

    result: dict[str, Any] = {"rows": rows, "counts": {}, "proteins": {}, "delta_p": {},
                              "abs_delta_buckets": {}}
    for partition in partitions:
        in_partition = data["partition"] == partition
        result["counts"][partition] = {
            endpoint: int(np.count_nonzero(in_partition & (data["endpoint"] == endpoint)))
            for endpoint in config["endpoints"]}
        result["proteins"][partition] = int(np.unique(data["protein_hash"][in_partition]).size)
    figure_series = {}
    for endpoint in config["endpoints"]:
        mask = data["endpoint"] == endpoint
        result["delta_p"][endpoint] = {
            partition: common.quantile_summary(delta[mask & (data["partition"] == partition)],
                                               quantiles) for partition in partitions}
        result["abs_delta_buckets"][endpoint] = common.bucket_counts(
            np.abs(delta[mask]), edges)
        figure_series[endpoint] = {partition: delta[mask & (data["partition"] == partition)]
                                   for partition in partitions}

    result["n_assays"] = common.bucket_counts(data["n_assays"].astype(np.float64),
                                              list(settings["count_edges"]))
    result["n_documents"] = common.bucket_counts(data["n_documents"].astype(np.float64),
                                                 list(settings["count_edges"]))
    result["qc_status"] = value_counts(table.column("qc_status"))
    result["min_target_confidence"] = value_counts(table.column("min_target_confidence"))
    result["flags"] = {name: int(common.boolean(table.column(name)).sum())
                       for name in FLAG_COLUMNS}
    result["flags"]["plausible_range = False"] = int(np.count_nonzero(~plausible))
    result["flags"]["source_quality_warning"] = int(warning.sum())

    p_a_range = data["p_a_range"].astype(np.float64)
    replicate_std = data["max_replicate_std"].astype(np.float64)
    result["anomalies"] = {
        f"|Δp| > {unit_copy:g} (vượt mức chép sai đơn vị)": int(np.count_nonzero(
            np.abs(delta) > unit_copy + tolerance)),
        f"p(A) giữa các assay lệch ≈ {unit_copy:g} (chép sai đơn vị nM/µM)": int(
            np.count_nonzero(np.abs(p_a_range - unit_copy) <= tolerance)),
        f"max_replicate_std > {within:g}": int(np.count_nonzero(replicate_std > within)),
        f"max_replicate_std > {meaningful:g}": int(np.count_nonzero(
            replicate_std > meaningful)),
        "pActivity ngoài khoảng hợp lý (plausible_range = False)":
            result["flags"]["plausible_range = False"],
        "nguồn tự gắn cảnh báo chất lượng": int(warning.sum()),
        "protein ngắn (short_protein)": result["flags"]["short_protein"],
        "chuỗi protein có ký tự lạ (nonstandard_sequence)":
            result["flags"]["nonstandard_sequence"],
        "ghi chú 'inactive' của nguồn": result["flags"]["inactive_comment"],
    }
    result["anomalies_percent"] = {name: common.share(count, rows)
                                   for name, count in result["anomalies"].items()}
    return result, figure_series


# --------------------------------------------------------------------------- split


def split_stats(config: Mapping[str, Any], root: Path,
                quantiles: list[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Protein, cluster và số cặp theo partition; phân bố cỡ cluster."""
    table = pq.read_table(common.dataset_file(config, root, "protein_split"))
    partition = table.column("partition").to_numpy(zero_copy_only=False)
    cluster = table.column("protein_cluster_id").to_numpy(zero_copy_only=False)
    cluster_proteins = table.column("cluster_proteins").to_numpy(zero_copy_only=False)
    cluster_pairs = table.column("cluster_pairs").to_numpy(zero_copy_only=False)
    pairs = table.column("pairs").to_numpy(zero_copy_only=False)
    result: dict[str, Any] = {"proteins": table.num_rows,
                              "clusters": int(np.unique(cluster).size), "partitions": {}}
    figure_series: dict[str, Any] = {"pairs_per_protein": {}, "cluster_proteins": {}}
    for name in config["stage2b"]["split"]["partitions"]:
        mask = partition == name
        _, first = np.unique(cluster[mask], return_index=True)
        result["partitions"][name] = {
            "proteins": int(mask.sum()), "clusters": int(first.size),
            "pairs": int(pairs[mask].sum()),
            "pairs_per_protein": common.quantile_summary(pairs[mask], quantiles),
            "proteins_per_cluster": common.quantile_summary(
                cluster_proteins[mask][first], quantiles),
            "pairs_per_cluster": common.quantile_summary(cluster_pairs[mask][first], quantiles),
        }
        figure_series["pairs_per_protein"][name] = np.log10(np.maximum(pairs[mask], 1))
        figure_series["cluster_proteins"][name] = np.log10(
            np.maximum(cluster_proteins[mask][first], 1))
    return result, figure_series


# --------------------------------------------------------------------------- chạy


def run(config: Mapping[str, Any], root: Path,
        figures: Path) -> tuple[dict[str, Any], list[str]]:
    """Chạy ba tầng thống kê và vẽ hình."""
    settings = config["audit"]["stats"]
    quantiles = list(config["audit"]["quantiles"])
    observations, p_series = observation_stats(config, root, quantiles)
    LOGGER.info("xong thống kê observation")
    evaluator, delta_series = evaluator_stats(config, root, settings, quantiles)
    LOGGER.info("xong thống kê evaluator_training")
    split, split_series = split_stats(config, root, quantiles)
    LOGGER.info("xong thống kê split")

    histograms = settings["histograms"]
    figure_paths = [
        plots.histograms(figures / "delta_p_by_endpoint.png",
                         "Phân bố nhãn Δp theo endpoint và partition",
                         list(delta_series.items()), histograms["delta_p"]["min"],
                         histograms["delta_p"]["max"], int(histograms["delta_p"]["bins"]),
                         "Δp = p(B) − p(A)"),
        plots.histograms(figures / "delta_p_by_endpoint_log.png",
                         "Phân bố nhãn Δp (trục y log, thấy rõ phần đuôi)",
                         list(delta_series.items()), histograms["delta_p"]["min"],
                         histograms["delta_p"]["max"], int(histograms["delta_p"]["bins"]),
                         "Δp = p(B) − p(A)", log_y=True),
        plots.histograms(figures / "p_activity_by_endpoint.png",
                         "pActivity của observation dùng được, theo nguồn",
                         list(p_series.items()), histograms["p_activity"]["min"],
                         histograms["p_activity"]["max"], int(histograms["p_activity"]["bins"]),
                         "pActivity",
                         vlines={name: list(config["stage2b"]["export"]["plausible_p_range"])
                                 for name in config["endpoints"]}),
        plots.histograms(figures / "split_sizes.png",
                         "Số cặp mỗi protein và số protein mỗi cluster (log10)",
                         [("log10(số cặp mỗi protein)", split_series["pairs_per_protein"]),
                          ("log10(số protein mỗi cluster)", split_series["cluster_proteins"])],
                         histograms["log_counts"]["min"], histograms["log_counts"]["max"],
                         int(histograms["log_counts"]["bins"]), "log10"),
        plots.bars(figures / "pairs_by_endpoint.png", "Số cặp theo endpoint",
                   list(config["endpoints"]),
                   [sum(evaluator["counts"][partition][endpoint]
                        for partition in evaluator["counts"])
                    for endpoint in config["endpoints"]], "số cặp"),
        plots.bars(figures / "qc_status.png", "Trạng thái QC trong evaluator_training",
                   list(evaluator["qc_status"]), list(evaluator["qc_status"].values()),
                   "số cặp", log_y=True),
    ]
    payload = {"observations": observations, "evaluator": evaluator, "split": split,
               "settings": settings, "figures": [str(path) for path in figure_paths]}
    return payload, report_lines(payload, config)


# --------------------------------------------------------------------------- báo cáo


def summary_row(label: str, summary: Mapping[str, Any]) -> list[str]:
    """Một dòng bảng phân vị: n, p1, p25, p50, p75, p99, max."""
    return [label, common.number(summary["n"])] + [
        common.number(summary.get(key)) for key in ("p1", "p25", "p50", "p75", "p99", "p100")]


SUMMARY_HEADER = ["nhóm", "n", "p1", "p25", "p50", "p75", "p99", "max"]


def report_lines(payload: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """Bản báo cáo đọc được."""
    observations = payload["observations"]
    evaluator = payload["evaluator"]
    split = payload["split"]
    endpoints = list(config["endpoints"])
    lines = ["# Thống kê dữ liệu và điểm bất thường (chỉ đọc, chỉ đánh dấu)", "",
             "## 1. Observation dùng được (sau dedup, đầu vào Stage 2b)",
             f"Dòng đọc: {common.number(observations['rows_read'])}; dùng được: "
             f"{common.number(observations['usable_rows'])}",
             "Khác nhau: " + ", ".join(f"{key} {common.number(value)}"
                                       for key, value in observations["distinct"].items()), ""]
    lines += common.table_lines(
        ["nguồn", *endpoints],
        [[source, *(common.number(counts[name]) for name in endpoints)]
         for source, counts in observations["by_source_endpoint"].items()])
    lines += ["", "pActivity theo endpoint:"]
    lines += common.table_lines(SUMMARY_HEADER, [summary_row(name, summary) for name, summary
                                                 in observations["p_activity"].items()])
    lines += ["", "Mật độ dữ liệu:"]
    lines += common.table_lines(SUMMARY_HEADER, [
        summary_row("dòng mỗi assay", observations["per_assay_rows"]),
        summary_row("phân tử mỗi assay", observations["per_assay_molecules"]),
        summary_row("assay mỗi protein", observations["per_protein_assays"]),
        summary_row("phân tử mỗi protein", observations["per_protein_molecules"])])
    lines += ["", "Điểm bất thường (observation):"]
    lines += common.table_lines(["kiểm tra", "số dòng"], [
        [name, common.number(count)] for name, count in observations["anomalies"].items()])

    lines += ["", "## 2. evaluator_training (nhãn Δp)",
              f"Tổng: {common.number(evaluator['rows'])} cặp", ""]
    lines += common.table_lines(
        ["partition", *endpoints, "protein"],
        [[name, *(common.number(counts[endpoint]) for endpoint in endpoints),
          common.number(evaluator["proteins"][name])]
         for name, counts in evaluator["counts"].items()])
    for endpoint in endpoints:
        lines += ["", f"Δp {endpoint}:"]
        lines += common.table_lines(SUMMARY_HEADER, [
            summary_row(name, summary)
            for name, summary in evaluator["delta_p"][endpoint].items()])
        lines += common.table_lines(
            ["|Δp|", "số cặp", "%"],
            [[row["range"], common.number(row["count"]), common.percent(row.get("percent"))]
             for row in evaluator["abs_delta_buckets"][endpoint]])
    for key, label in (("n_assays", "Số assay mỗi cặp"), ("n_documents", "Số tài liệu mỗi cặp")):
        lines += ["", f"{label}:"]
        lines += common.table_lines(
            ["khoảng", "số cặp", "%"],
            [[row["range"], common.number(row["count"]), common.percent(row.get("percent"))]
             for row in evaluator[key]])
    lines += ["", "Trạng thái QC: " + ", ".join(
        f"{name} {common.number(count)}" for name, count in evaluator["qc_status"].items()),
              "min_target_confidence: " + ", ".join(
        f"{name} {common.number(count)}"
        for name, count in evaluator["min_target_confidence"].items()),
              "", "Điểm bất thường (đã gắn cờ trong dataset, KHÔNG xóa):"]
    lines += common.table_lines(["kiểm tra", "số cặp", "%"], [
        [name, common.number(count), common.percent(evaluator["anomalies_percent"][name])]
        for name, count in evaluator["anomalies"].items()])

    lines += ["", "## 3. Split theo protein",
              f"Protein: {common.number(split['proteins'])}; cluster: "
              f"{common.number(split['clusters'])}", ""]
    lines += common.table_lines(
        ["partition", "protein", "cluster", "cặp"],
        [[name, common.number(block["proteins"]), common.number(block["clusters"]),
          common.number(block["pairs"])] for name, block in split["partitions"].items()])
    for key, label in (("pairs_per_protein", "Số cặp mỗi protein"),
                       ("proteins_per_cluster", "Số protein mỗi cluster"),
                       ("pairs_per_cluster", "Số cặp mỗi cluster")):
        lines += ["", f"{label}:"]
        lines += common.table_lines(SUMMARY_HEADER, [
            summary_row(name, block[key]) for name, block in split["partitions"].items()])
    lines += ["", "Hình: " + ", ".join(payload["figures"])]
    return lines
