"""Hình kết quả chính của tuần 10-16/11/2026: đúng chiều theo QUY MÔ DỮ LIỆU TRAIN.

    python scripts/plot_data_scaling.py
    python scripts/plot_data_scaling.py --out runs/figures/data_scaling.png

VÌ SAO HÌNH NÀY. Hiệu MoLFormer − Morgan trên lát `strict` là +0,0062 ở 100.000 dòng và
+0,0124 ở 6.627.559 dòng. Đọc theo cột thì thấy chuyện quan trọng hơn hiệu ở một quy mô:
gấp 66 lần dữ liệu giúp MoLFormer +0,0041 và KHÔNG giúp Morgan (−0,0021). Morgan bão hòa ở
quy mô 100k. Một con số +0,0124 đứng một mình không nói được điều đó.

HAI PANEL, cố ý:
    trái  — lát `strict` (424.473 dòng, cặp (A,B) chưa từng có trong train). Đây là số đi
            vào paper.
    phải  — lát `all` (toàn bộ 493.852 dòng, trong đó 14,05 % dòng trùng cặp với train).
            Giữ panel này để thấy ở quy mô 6,6 triệu dòng hai lát ĐẢO THỨ TỰ: Morgan vượt
            lên chỉ vì nó học thuộc được các cặp trùng. Ở quy mô 100k thì hai panel gần
            giống nhau, vì chưa học thuộc được gì.

SỐ LẤY TỪ ĐÂU. Đúng một đường với `scripts/summarize_runs.py`:
`validation_metrics[<lát>]["ranking"]["pairwise_accuracy_macro"]` của manifest, tức chỉ số
tại MỐC TỐT NHẤT đã chọn, không phải mốc cuối. Script này KHÔNG tính lại gì từ dữ liệu thô,
nên số trên hình trùng tuyệt đối với bảng.

ĐIỀU KIỆN LỌC (đổi thì sửa ở CRITERIA bên dưới): head `direct`, chọn checkpoint theo lát
`strict`, metric chọn là pairwise_accuracy_macro, cấu hình thắng 7x102 residual, không
chuẩn hóa nhãn, không partition_override. Mỗi ô phải có ĐÚNG 3 hạt giống; thiếu hoặc thừa
thì script DỪNG và in ra nó tìm thấy gì, chứ không vẽ một hình nửa vời.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (20261023, 20261024, 20261025)
SLICES = ("strict", "all")

# Mốc đã ghi trong nhật ký ngày 12/11/2026 (mục "phần ba"). Chỉ dùng để ĐỐI CHIẾU và cảnh
# báo nếu lệch, KHÔNG dùng để vẽ — hình luôn vẽ số đọc từ manifest.
RECORDED = {("fingerprint", "subset"): 0.5499, ("fingerprint", "full"): 0.5478,
            ("embedding", "subset"): 0.5561, ("embedding", "full"): 0.5602}

ARMS = (("fingerprint", "Morgan fingerprint (2048 bit)", "tab:orange", "s"),
        ("embedding", "MoLFormer-XL embedding", "tab:blue", "o"))


def keep(manifest: dict[str, Any]) -> bool:
    """Lần chạy này có thuộc hình không? Xem ĐIỀU KIỆN LỌC trong docstring."""
    overrides = manifest.get("overrides") or {}
    if manifest.get("head") != "direct":
        return False
    if manifest.get("selection_slice") != "strict":
        return False
    if manifest.get("selection_metric") != "pairwise_accuracy_macro":
        return False
    if manifest.get("partition_override"):
        return False
    if "label_normalization" in overrides:
        return False
    if manifest.get("seed") not in SEEDS:
        return False
    model = manifest.get("model") or {}
    # `fusion` phải là concat: các lần chạy attention/projector cũng là direct + 7x102 +
    # chọn theo strict, không lọc ra thì chúng rơi vào đúng ô Morgan và làm ô đó thành 6
    # hạt giống.
    if model.get("fusion") != "concat":
        return False
    if model.get("head_block") != "residual":
        return False
    if (model.get("residual_blocks"), model.get("residual_width")) != (7, 102):
        return False
    return True


def macro(manifest: dict[str, Any], slice_name: str) -> float | None:
    """Đúng chiều macro tại mốc tốt nhất, đúng đường mà summarize_runs.py dùng."""
    block = (manifest.get("validation_metrics") or {}).get(slice_name) or {}
    value = ((block.get("ranking") or {}).get("pairwise_accuracy_macro"))
    return None if value is None else float(value)


def collect(runs_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Gom các lần chạy thành 4 ô: (biểu diễn, quy mô) -> số đo của 3 hạt giống."""
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(runs_dir.glob("train_*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not keep(manifest):
            continue
        overrides = manifest.get("overrides") or {}
        features = ("embedding" if overrides.get("model.ligand_features") == "embedding"
                    else "fingerprint")
        scale = "subset" if manifest.get("train_subset") else "full"
        rows = int(((manifest.get("rows") or {}).get("train")) or 0)
        values = {name: macro(manifest, name) for name in SLICES}
        if any(value is None for value in values.values()):
            continue
        cell = cells.setdefault((features, scale), {"rows": rows, "seeds": {}})
        if cell["rows"] != rows:
            raise SystemExit(f"ô ({features}, {scale}) có hai số dòng train khác nhau: "
                             f"{cell['rows']} và {rows}; lọc chưa đủ chặt")
        cell["seeds"][int(manifest["seed"])] = {"values": values, "manifest": path.name,
                                                "best": (manifest.get("best_epoch"),
                                                         manifest.get("best_step"))}
    return cells


def summarise(cells: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Trung bình và độ lệch chuẩn mẫu của từng ô; dừng nếu ô nào không đủ 3 hạt."""
    problems = []
    for features, _, _, _ in ARMS:
        for scale in ("subset", "full"):
            cell = cells.get((features, scale))
            count = 0 if cell is None else len(cell["seeds"])
            if count != 3:
                problems.append(f"({features}, {scale}): {count} hạt giống")
    if problems:
        print("THIẾU DỮ LIỆU, không vẽ. Mỗi ô cần đúng 3 hạt giống:", file=sys.stderr)
        for line in problems:
            print("   " + line, file=sys.stderr)
        for key in sorted(cells):
            for seed in sorted(cells[key]["seeds"]):
                print(f"   có: {key} hạt {seed} <- "
                      f"{cells[key]['seeds'][seed]['manifest']}", file=sys.stderr)
        raise SystemExit(1)

    report: dict[str, Any] = {"slices": list(SLICES), "cells": {}}
    for features, _, _, _ in ARMS:
        for scale in ("subset", "full"):
            cell = cells[(features, scale)]
            entry: dict[str, Any] = {"rows": cell["rows"], "seeds": {}}
            for name in SLICES:
                points = [cell["seeds"][seed]["values"][name] for seed in sorted(SEEDS)]
                entry[name] = {"mean": statistics.fmean(points),
                               "sd": statistics.stdev(points),
                               "per_seed": points}
            for seed in sorted(SEEDS):
                entry["seeds"][str(seed)] = {
                    "manifest": cell["seeds"][seed]["manifest"],
                    "best_epoch": cell["seeds"][seed]["best"][0],
                    "best_step": cell["seeds"][seed]["best"][1]}
            report["cells"][f"{features}_{scale}"] = entry
    return report


def show_table(report: dict[str, Any]) -> None:
    """In bảng ra stdout để đối chiếu bằng mắt với summarize_runs.py."""
    for name in SLICES:
        print(f"\nlát {name} — đúng chiều macro (3 hạt giống 20261023/24/25)")
        print(f"{'biểu diễn':<32}{'quy mô':>12}{'trung bình':>12}{'sd':>9}"
              f"{'từng hạt':>26}")
        print("-" * 91)
        for features, title, _, _ in ARMS:
            for scale in ("subset", "full"):
                cell = report["cells"][f"{features}_{scale}"]
                block = cell[name]
                seeds = " ".join(f"{value:.4f}" for value in block["per_seed"])
                print(f"{title:<32}{cell['rows']:>12,}{block['mean']:>12.4f}"
                      f"{block['sd']:>9.4f}{seeds:>26}")
        for scale in ("subset", "full"):
            morgan = report["cells"][f"fingerprint_{scale}"][name]["mean"]
            molformer = report["cells"][f"embedding_{scale}"][name]["mean"]
            print(f"{'  hiệu MoLFormer − Morgan':<32}"
                  f"{report['cells'][f'fingerprint_{scale}']['rows']:>12,}"
                  f"{molformer - morgan:>+12.4f}")
    print("\nthay đổi khi gấp 66 lần dữ liệu (lát strict):")
    for features, title, _, _ in ARMS:
        subset = report["cells"][f"{features}_subset"]["strict"]["mean"]
        full = report["cells"][f"{features}_full"]["strict"]["mean"]
        print(f"   {title:<32}{full - subset:>+9.4f}")

    print("\nđối chiếu với mốc đã ghi trong nhật ký 12/11 (lát strict):")
    for (features, scale), recorded in RECORDED.items():
        measured = report["cells"][f"{features}_{scale}"]["strict"]["mean"]
        flag = "" if abs(measured - recorded) < 0.00005 else "   <-- LỆCH, kiểm lại"
        print(f"   {features:<12}{scale:<8}đo {measured:.4f} | nhật ký {recorded:.4f}{flag}")


def draw(report: dict[str, Any], target: Path) -> None:
    """Hai panel: lát strict (số của paper) và lát all (số bị nhiễm học thuộc)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Nhãn tiếng Anh: hình này đi vào bài nộp ICML.
    titles = {"strict": "Unseen molecule pairs (strict, 424,473 rows)",
              "all": "All validation rows (493,852; 14.05% pairs seen in train)"}
    # Trục y phóng vào khoảng ĐO ĐƯỢC. Nếu vẽ cả mức ngẫu nhiên 0,500 thì bốn đường bị nén
    # vào một phần năm phía trên và không nhìn ra độ dốc — mà độ dốc mới là nội dung của
    # hình. Bù lại, mức ngẫu nhiên được ghi thành chú thích dưới hình để không ai hiểu nhầm
    # rằng trục bắt đầu từ 0,5.
    values = [report["cells"][f"{features}_{scale}"][name]["mean"]
              + sign * report["cells"][f"{features}_{scale}"][name]["sd"]
              for features, _, _, _ in ARMS for scale in ("subset", "full")
              for name in SLICES for sign in (-1, 1)]
    low, high = min(values), max(values)
    margin = max((high - low) * 0.18, 0.0015)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    for index, name in enumerate(SLICES):
        axis = axes[index]
        for features, title, colour, marker in ARMS:
            xs, ys, errors = [], [], []
            for scale in ("subset", "full"):
                cell = report["cells"][f"{features}_{scale}"]
                xs.append(cell["rows"])
                ys.append(cell[name]["mean"])
                errors.append(cell[name]["sd"])
            axis.errorbar(xs, ys, yerr=errors, label=title, color=colour, marker=marker,
                          markersize=7, linewidth=1.8, capsize=4)
            for position, (x, y) in enumerate(zip(xs, ys)):
                axis.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                              xytext=(0, 10), fontsize=8, color=colour,
                              ha="left" if position == 0 else "right")
        axis.set_xscale("log")
        axis.set_xlim(60_000, 1.1e7)
        axis.set_ylim(low - margin, high + margin)
        axis.set_xticks([100_000, 6_627_559])
        axis.set_xticklabels(["100k", "6.63M"])
        axis.minorticks_off()
        axis.set_xlabel("training rows (log scale)")
        axis.set_title(titles[name], fontsize=10)
        axis.grid(alpha=0.3, linestyle=":")
    axes[0].set_ylabel("pairwise direction accuracy (macro over queries)")
    axes[0].legend(loc="lower left", fontsize=9)
    figure.suptitle("Morgan fingerprints saturate with data; MoLFormer embeddings do not",
                    fontsize=12)
    figure.text(0.5, 0.012,
                "Chance level is 0.500; the y-axis is zoomed to the measured range. "
                "Error bars: standard deviation over 3 seeds (20261023/24/25). "
                "Checkpoints selected on the strict slice.",
                ha="center", fontsize=8, color="dimgrey")
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200)
    plt.close(figure)
    print(f"\nđã ghi {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="",
                        help="mặc định đọc stage3.paths.runs_dir trong config.json")
    parser.add_argument("--out", default="runs/figures/data_scaling.png")
    parser.add_argument("--no-figure", action="store_true",
                        help="chỉ in bảng, không cần matplotlib")
    args = parser.parse_args()
    # Console Windows mặc định không phải UTF-8; không có hai dòng này thì mọi chữ có dấu
    # in ra thành ế. error_analysis.py làm đúng việc này cho stdout.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if args.runs_dir:
        runs_dir = Path(args.runs_dir)
    else:
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        runs_dir = ROOT / config["stage3"]["paths"]["runs_dir"]
    if not runs_dir.is_dir():
        print(f"không thấy thư mục {runs_dir}", file=sys.stderr)
        return 1

    report = summarise(collect(runs_dir))
    show_table(report)
    target = Path(args.out)
    if not args.no_figure:
        draw(report, target)
    target.with_suffix(".json").parent.mkdir(parents=True, exist_ok=True)
    target.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"       {target.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
