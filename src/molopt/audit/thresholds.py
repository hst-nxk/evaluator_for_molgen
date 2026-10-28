"""Audit ngưỡng QC: Δp giữa các assay giống nhau, lệch nhau bao nhiêu, theo mức của tài liệu.

Câu hỏi của buổi họp 26/10/2026:
    - "Nhiều trường hợp Δp gần như y hệt giữa các assay" là bao nhiêu trường hợp?
    - Bao nhiêu bản ghi vượt ngưỡng, với ngưỡng lấy từ tài liệu chứ không tự đặt?

Mức so sánh (audit.literature), cùng cách đếm với Landrum & Riniker 2024 (JCIM 64, 1560;
DOI 10.1021/acs.jcim.4c00049), bài so cùng một hợp chất đo ở hai assay:
    within_error   0,3 log  "expected window for experimental error" (khoảng 2 lần)
    meaningful     1,0 log  ngưỡng "truly meaningful activity difference" (bài tự ghi là
                            giới hạn quy ước)
    unit_copy      3,0 log  cặp lệch đúng 3 log là cùng một số bị chép sai đơn vị
                            (nM/µM); bài loại các cặp này cùng với cặp bằng nhau tuyệt đối
Bài so p của MỘT hợp chất; ở đây so Δp của một CẶP hợp chất, nên con số không so thẳng
được với tỉ lệ trong bài. Sai lệch hệ thống của assay triệt tiêu trong Δp, nên tỉ lệ ở
đây được kỳ vọng thấp hơn.

Đếm ở hai mức, đọc evidence_records (mọi bản ghi, kể cả bản ghi bị cách ly), KHÔNG lọc:
    cặp assay   mỗi cặp hai assay trong một bản ghi tính một lần (cách đếm của bài báo)
    bản ghi     khoảng Δp = max − min của bản ghi: vượt t nghĩa là có ít nhất một cặp
                assay lệch hơn t
"Giống hệt" = lệch ≤ stage2b.qc.collapse_tolerance (0,01), đúng mức Stage 2b đã dùng để
gộp hai Δp làm một.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import common, plots

LOGGER = logging.getLogger(__name__)
COLUMNS = ["endpoint", "n_assays", "n_distinct_values", "n_documents", "delta_p_min",
           "delta_p_max", "dispersion", "qc_status", "delta_p_values"]
LEVEL_KEYS = ("within_error", "meaningful", "unit_copy")


def load(path: Path) -> tuple[dict[str, np.ndarray], pa.ChunkedArray]:
    """Cột vô hướng của evidence_records (numpy) và cột danh sách Δp từng assay."""
    table = pq.read_table(path, columns=COLUMNS)
    LOGGER.info("đọc %s bản ghi từ %s", f"{table.num_rows:,}", path)
    data = {name: table.column(name).to_numpy(zero_copy_only=False)
            for name in COLUMNS if name != "delta_p_values"}
    for name in ("delta_p_min", "delta_p_max", "dispersion"):
        data[name] = data[name].astype(np.float64)
    data["range"] = data["delta_p_max"] - data["delta_p_min"]
    return data, table.column("delta_p_values")


# --------------------------------------------------------------------------- cặp assay


def flatten_lists(values: pa.ChunkedArray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Giá trị phẳng và mã bản ghi (0..len(rows)-1) của các danh sách ở dòng `rows`."""
    taken = values.take(pa.array(rows)).combine_chunks()
    lengths = pc.list_value_length(taken).fill_null(0).to_numpy(zero_copy_only=False)
    flat = pc.list_flatten(taken).to_numpy(zero_copy_only=False).astype(np.float64)
    owners = np.repeat(np.arange(rows.size, dtype=np.int64), lengths.astype(np.int64))
    return flat, owners


def pairs_within(flat: np.ndarray, owners: np.ndarray, distance: float) -> np.ndarray:
    """Số cặp (i < j) cùng bản ghi có |v_i − v_j| ≤ distance, theo từng bản ghi.

    Cách làm: đặt mỗi bản ghi vào một đoạn riêng trên trục số (khoảng cách giữa các đoạn
    lớn hơn mọi hiệu có thể), sắp xếp, rồi với mỗi phần tử đếm số phần tử đứng sau nó mà
    không vượt quá giá trị + distance (np.searchsorted). Không cần vòng lặp Python.
    """
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64)
    spread = float(np.nanmax(np.abs(flat))) if np.isfinite(flat).any() else 0.0
    spacing = 2.0 * spread + abs(distance) + 1.0
    keys = owners.astype(np.float64) * spacing + flat
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    upper = np.searchsorted(ordered, ordered + distance, side="right")
    within = upper - np.arange(ordered.size) - 1
    return np.bincount(owners[order], weights=within,
                       minlength=int(owners.max()) + 1).astype(np.int64)


def pair_level(flat: np.ndarray, owners: np.ndarray, levels: Mapping[str, float],
               tolerance: float) -> dict[str, Any]:
    """Tỉ lệ cặp assay giống hệt, lệch > 0,3, > 1,0 và lệch ≈ 3,0 (chép sai đơn vị)."""
    if flat.size == 0:
        return {"assay_pairs": 0}
    counts = np.bincount(owners)
    total = int((counts * (counts - 1) // 2).sum())
    identical = int(pairs_within(flat, owners, tolerance).sum())
    result: dict[str, Any] = {
        "assay_pairs": total, "identical": identical,
        "identical_percent": common.share(identical, total),
    }
    for key in ("within_error", "meaningful"):
        over = total - int(pairs_within(flat, owners, float(levels[key])).sum())
        result[f"over_{key}"] = over
        result[f"over_{key}_percent"] = common.share(over, total)
    copy = float(levels["unit_copy"])
    unit = int(pairs_within(flat, owners, copy + tolerance).sum()
               - pairs_within(flat, owners, copy - tolerance).sum())
    result["unit_copy"] = unit
    result["unit_copy_percent"] = common.share(unit, total)
    return result


# --------------------------------------------------------------------------- bản ghi


def record_level(data: Mapping[str, np.ndarray], mask: np.ndarray,
                 levels: Mapping[str, float]) -> dict[str, Any]:
    """Bản ghi ≥ 2 assay chia theo khoảng Δp (max − min) tại các mức của tài liệu."""
    multi = mask & (data["n_assays"] >= 2)
    spread = data["range"]
    within, meaningful = float(levels["within_error"]), float(levels["meaningful"])
    groups = {
        "identical": multi & (data["n_distinct_values"] == 1),
        "near_identical": multi & (data["n_distinct_values"] >= 2) & (spread <= within),
        "between": multi & (spread > within) & (spread <= meaningful),
        "over_meaningful": multi & (spread > meaningful),
    }
    total = int(multi.sum())
    result: dict[str, Any] = {"multi_assay": total}
    for name, selected in groups.items():
        result[name] = int(selected.sum())
        result[f"{name}_percent"] = common.share(int(selected.sum()), total)
    result["range_nan"] = int(np.count_nonzero(multi & np.isnan(spread)))
    return result


def identical_summary(data: Mapping[str, np.ndarray], mask: np.ndarray,
                      assay_edges: list[int]) -> dict[str, Any]:
    """Bản ghi từ ≥ 2 assay mà chỉ có MỘT giá trị Δp, tách theo số tài liệu và số assay."""
    multi = mask & (data["n_assays"] >= 2)
    identical = multi & (data["n_distinct_values"] == 1)
    one_document = identical & (data["n_documents"] <= 1)
    many_documents = identical & (data["n_documents"] >= 2)
    by_assays = []
    for low, high in zip(assay_edges, assay_edges[1:] + [None]):
        bucket = multi & (data["n_assays"] >= low)
        if high is not None:
            bucket &= data["n_assays"] < high
        same = bucket & identical
        label = f"{low}-{high - 1}" if high is not None else f">= {low}"
        by_assays.append({"n_assays": label, "records": int(bucket.sum()),
                          "identical": int(same.sum()),
                          "percent": common.share(int(same.sum()), int(bucket.sum()))})
    return {
        "multi_assay": int(multi.sum()), "identical": int(identical.sum()),
        "identical_percent": common.share(int(identical.sum()), int(multi.sum())),
        "identical_one_document": int(one_document.sum()),
        "identical_many_documents": int(many_documents.sum()),
        "by_n_assays": by_assays,
    }


def current_quarantine(data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    """Số bản ghi đang bị cách ly bởi ngưỡng hiện tại của pipeline (để đối chiếu)."""
    multi_value = mask & (data["n_distinct_values"] >= 2)
    quarantined = int(np.count_nonzero(mask & (data["qc_status"] == "quarantine")))
    return {"multi_value": int(multi_value.sum()), "quarantined": quarantined,
            "percent": common.share(quarantined, int(multi_value.sum()))}


# --------------------------------------------------------------------------- chạy


def endpoint_block(data: Mapping[str, np.ndarray], values: pa.ChunkedArray, mask: np.ndarray,
                   levels: Mapping[str, float], tolerance: float,
                   settings: Mapping[str, Any], quantiles: Sequence[float]) -> dict[str, Any]:
    """Mọi số liệu của một endpoint, cho tất cả bản ghi và cho bản ghi ≥ 2 tài liệu."""
    multi = mask & (data["n_assays"] >= 2)
    independent = multi & (data["n_documents"] >= 2)
    block: dict[str, Any] = {
        "records": int(mask.sum()),
        "qc_status": common.count_values(data["qc_status"][mask]),
        "identical": identical_summary(data, mask, list(settings["assay_buckets"])),
        "records_all": record_level(data, mask, levels),
        "records_independent": record_level(data, independent, levels),
        "current_quarantine": current_quarantine(data, mask),
        "range_quantiles": common.quantile_summary(data["range"][multi], quantiles),
    }
    for name, selected in (("pairs_all", multi), ("pairs_independent", independent)):
        flat, owners = flatten_lists(values, np.flatnonzero(selected))
        block[name] = pair_level(flat, owners, levels, tolerance)
    return block


def run(config: Mapping[str, Any], root: Path,
        figures: Path) -> tuple[dict[str, Any], list[str]]:
    """Đếm Δp giống hệt và Δp vượt các mức của tài liệu, theo từng endpoint."""
    settings = config["audit"]["thresholds"]
    literature = config["audit"]["literature"]
    levels = {key: float(literature[key]) for key in LEVEL_KEYS}
    tolerance = float(config["stage2b"]["qc"]["collapse_tolerance"])
    quantiles = config["audit"]["quantiles"]
    data, values = load(common.dataset_file(config, root, "evidence_records"))
    payload: dict[str, Any] = {
        "records": int(data["endpoint"].size), "levels": levels, "tolerance": tolerance,
        "source": literature["source"],
        "current_thresholds": config["stage2b"]["qc"]["dispersion_thresholds"],
        "endpoints": {}}
    panels = []
    for endpoint in config["endpoints"]:
        mask = data["endpoint"] == endpoint
        payload["endpoints"][endpoint] = endpoint_block(data, values, mask, levels, tolerance,
                                                        settings, quantiles)
        multi = mask & (data["n_assays"] >= 2)
        panels.append((endpoint, {
            "≥ 2 assay": data["range"][multi],
            "≥ 2 tài liệu": data["range"][multi & (data["n_documents"] >= 2)]}))
        LOGGER.info("xong %s", endpoint)
    histogram = settings["histogram"]
    payload["figures"] = [str(plots.histograms(
        figures / "delta_p_range_by_endpoint.png",
        "Khoảng Δp giữa các assay (max − min); vạch: 0,3 / 1,0 / 3,0 log (Landrum 2024)",
        panels, histogram["min"], histogram["max"], int(histogram["bins"]),
        "max − min của Δp", vlines={name: list(levels.values())
                                    for name in config["endpoints"]}, log_y=True))]
    return payload, report_lines(payload)


# --------------------------------------------------------------------------- báo cáo


def report_lines(payload: Mapping[str, Any]) -> list[str]:
    """Bản báo cáo đọc được."""
    levels = payload["levels"]
    within, meaningful = levels["within_error"], levels["meaningful"]
    lines = ["# Δp giữa các assay: giống hệt, lệch, vượt mức (đọc evidence_records)", "",
             f"Tổng bản ghi: {common.number(payload['records'])}",
             f"Mức so sánh: {within:g} / {meaningful:g} / {levels['unit_copy']:g} log. "
             f"Nguồn: {payload['source']}",
             f"Giống hệt = lệch ≤ {payload['tolerance']:g} (ngưỡng gộp của Stage 2b).", "",
             "## 1. Bản ghi ≥ 2 assay, theo khoảng Δp (max − min)"]
    header = ["endpoint", "≥ 2 assay", "giống hệt", f"≤ {within:g}",
              f"({within:g}, {meaningful:g}]", f"> {meaningful:g}"]
    for key, title in (("records_all", "Tất cả"),
                       ("records_independent", "Chỉ bản ghi có ≥ 2 tài liệu")):
        lines.append(f"{title}:")
        rows = []
        for name, block in payload["endpoints"].items():
            part = block[key]
            rows.append([name, common.number(part["multi_assay"])] + [
                f"{common.number(part[group])} ({common.percent(part[f'{group}_percent'])})"
                for group in ("identical", "near_identical", "between", "over_meaningful")])
        lines += common.table_lines(header, rows)
        lines.append("")
    lines += ["'giống hệt': mọi assay cho cùng một Δp; '≤ 0,3': khác nhau nhưng trong sai số",
              "thí nghiệm; '> 1,0': ít nhất hai assay lệch nhau hơn 1 log.", "",
              "## 2. Theo từng cặp assay (cách đếm của Landrum & Riniker 2024)"]
    for key, title in (("pairs_all", "Tất cả"), ("pairs_independent", "≥ 2 tài liệu")):
        lines.append(f"{title}:")
        rows = []
        for name, block in payload["endpoints"].items():
            part = block[key]
            if not part.get("assay_pairs"):
                rows.append([name, "0", "-", "-", "-", "-"])
                continue
            rows.append([name, common.number(part["assay_pairs"]),
                         common.percent(part["identical_percent"]),
                         common.percent(part["over_within_error_percent"]),
                         common.percent(part["over_meaningful_percent"]),
                         f"{common.number(part['unit_copy'])}"])
        lines += common.table_lines(
            ["endpoint", "cặp assay", "giống hệt", f"> {within:g}", f"> {meaningful:g}",
             f"≈ {levels['unit_copy']:g} (sai đơn vị)"], rows)
        lines.append("")
    lines += ["Bài báo (p của một hợp chất, KHÔNG phải Δp): IC50 65 % > 0,3 và 27 % > 1,0",
              "khi curation tối thiểu; 48 % và 13 % khi curation tối đa. Chỉ để tham khảo.", "",
              "## 3. Bản ghi chỉ có một giá trị Δp, theo số tài liệu và số assay"]
    lines += common.table_lines(
        ["endpoint", "≥ 2 assay", "chỉ 1 giá trị", "%", "trong 1 tài liệu", "qua ≥ 2 tài liệu"],
        [[name, common.number(block["identical"]["multi_assay"]),
          common.number(block["identical"]["identical"]),
          common.percent(block["identical"]["identical_percent"]),
          common.number(block["identical"]["identical_one_document"]),
          common.number(block["identical"]["identical_many_documents"])]
         for name, block in payload["endpoints"].items()])
    lines.append("")
    for name, block in payload["endpoints"].items():
        lines.append(f"{name}, theo số assay:")
        lines += common.table_lines(
            ["n_assays", "bản ghi", "chỉ 1 giá trị", "%"],
            [[row["n_assays"], common.number(row["records"]), common.number(row["identical"]),
              common.percent(row["percent"])] for row in block["identical"]["by_n_assays"]])
        lines.append("")
    lines += ["## 4. Đối chiếu: ngưỡng cách ly của pipeline (stage2b.qc.dispersion_thresholds)",
              f"Ngưỡng: {payload['current_thresholds']}"]
    lines += common.table_lines(
        ["endpoint", "bản ghi ≥ 2 giá trị", "đang bị cách ly", "%"],
        [[name, common.number(block["current_quarantine"]["multi_value"]),
          common.number(block["current_quarantine"]["quarantined"]),
          common.percent(block["current_quarantine"]["percent"])]
         for name, block in payload["endpoints"].items()])
    lines += ["", "Hình: " + ", ".join(payload["figures"])]
    return lines
