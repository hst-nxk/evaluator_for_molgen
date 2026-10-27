"""Vẽ hình cho báo cáo audit (matplotlib, chế độ không cửa sổ).

Histogram được tính bằng numpy trước rồi mới vẽ, nên vẽ vài triệu điểm vẫn nhanh.
Giá trị ngoài [low, high] được dồn vào cột ngoài cùng để không bị mất khỏi hình; nhãn
trục ghi rõ điều đó.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

COLUMNS_PER_ROW = 2
PANEL_WIDTH = 6.0
PANEL_HEIGHT = 3.6
DPI = 150
LINE_STYLE = "--"


def pyplot():
    """Nạp matplotlib ở chế độ Agg (không cần màn hình, chạy được trên server)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def histograms(path: Path, title: str,
               panels: Sequence[tuple[str, Mapping[str, np.ndarray]]],
               low: float, high: float, bins: int, xlabel: str,
               vlines: Mapping[str, Sequence[float]] | None = None,
               log_y: bool = False) -> Path:
    """Mỗi panel một histogram, mỗi chuỗi trong panel một đường (tỉ lệ theo chuỗi).

    Nhận: đường dẫn PNG; tiêu đề; danh sách (tên panel, {tên chuỗi: giá trị}); khoảng và
    số cột; nhãn trục x; vạch dọc theo tên panel; trục y log hay không.
    Trục y là tỉ lệ dòng của chuỗi rơi vào mỗi cột, nên các chuỗi khác cỡ so được với nhau.
    """
    plt = pyplot()
    rows = max(1, int(np.ceil(len(panels) / COLUMNS_PER_ROW)))
    columns = min(COLUMNS_PER_ROW, max(1, len(panels)))
    figure, axes = plt.subplots(rows, columns, figsize=(PANEL_WIDTH * columns,
                                                       PANEL_HEIGHT * rows), squeeze=False)
    edges = np.linspace(low, high, bins + 1)
    for axis, (name, series) in zip(axes.flat, panels):
        for label, values in series.items():
            array = np.asarray(values, dtype=np.float64)
            array = np.clip(array[~np.isnan(array)], low, high)
            if array.size == 0:
                continue
            counts, _ = np.histogram(array, bins=edges)
            axis.stairs(counts / array.size, edges, label=f"{label} (n={array.size:,})")
        for value in (vlines or {}).get(name, []):
            axis.axvline(value, linestyle=LINE_STYLE, linewidth=1)
        axis.set_title(name)
        axis.set_xlabel(f"{xlabel} (ngoài [{low:g}, {high:g}] dồn vào cột biên)")
        axis.set_ylabel("tỉ lệ dòng")
        if log_y:
            axis.set_yscale("log")
        axis.legend(fontsize="small")
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return path


def bars(path: Path, title: str, labels: Sequence[str], values: Sequence[float],
         ylabel: str, log_y: bool = False) -> Path:
    """Biểu đồ cột đơn giản (vd. số cặp theo endpoint, theo trạng thái QC)."""
    plt = pyplot()
    figure, axis = plt.subplots(figsize=(PANEL_WIDTH * COLUMNS_PER_ROW, PANEL_HEIGHT))
    axis.bar(list(labels), list(values))
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if log_y:
        axis.set_yscale("log")
    axis.tick_params(axis="x", labelrotation=30)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return path
