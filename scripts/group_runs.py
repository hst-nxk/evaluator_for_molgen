"""Gộp các lần chạy CHỈ KHÁC NHAU Ở HẠT GIỐNG rồi in trung bình và độ lệch chuẩn.

    python scripts/group_runs.py --slice strict
    python scripts/group_runs.py --slice strict --min-seeds 3
    python scripts/group_runs.py --slice all --metric top_1
    python scripts/group_runs.py --slice strict --grep ligand_features=embedding

VÌ SAO CẦN. `summarize_runs.py` in TỪNG lần chạy, làm tròn 4 chữ số. Cộng tay ba dòng đó
rồi chia ba thì lệch 0,0001 so với số thật — đã mắc đúng lỗi đó ngày 12/11 (ghi 0,5602
trong khi trung bình thật là 0,5603, và sd 0,0002 trong khi thật là 0,0001). File này đọc
thẳng manifest ở độ chính xác đầy đủ, nên mọi con số đưa vào nhật ký hay paper lấy từ đây.

CÁCH NHÓM: hai lần chạy vào cùng một nhóm khi và chỉ khi chúng giống nhau ở MỌI thứ trừ
`seed` — so bằng các khóa của manifest (cấu hình, đầu ra, toàn bộ khối `model`, lr, wd,
loss, chuẩn hóa nhãn, tập con, partition_override, lát chọn checkpoint, lọc edit...), chứ
không so bằng chuỗi nhãn. Nhờ vậy hai lần chạy trông giống nhau trên bảng nhưng khác một
tham số ẩn sẽ KHÔNG bị gộp nhầm.

ĐỘ LỆCH CHUẨN là sd MẪU (chia n−1), cùng quy ước với mọi con số đã báo cáo của dự án.
Nhóm chỉ có 1 hạt giống thì cột sd để trống, không in 0.

LƯU Ý VỀ LÁT: với lát `all` thì chỉ số lấy ở MỐC TỐT NHẤT trong `history` (đúng như
summarize_runs.py), còn các lát khác lấy ở `validation_metrics[<lát>]`. Hai đường này là
cùng một checkpoint.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from summarize_runs import best_block, label  # noqa: E402

# Các khóa manifest định nghĩa "cùng một thí nghiệm". `seed` CỐ Ý không có ở đây: đó là
# thứ duy nhất được phép khác nhau trong một nhóm.
IDENTITY_KEYS = ("configuration", "head", "protein_mode", "model", "batch_size",
                 "learning_rate", "weight_decay", "loss", "huber_delta", "label_scaler",
                 "augment_reverse", "train_subset", "partition_override",
                 "selection_metric", "selection_slice", "edit_filter",
                 "branch_learning_rate_scale", "sharpness_penalty", "accumulate_steps",
                 "group_by_protein")


def identity(manifest: dict[str, Any]) -> str:
    """Khóa nhóm: mọi thứ trừ hạt giống.

    Gồm HAI phần, cố ý:
      - nhãn (đã bỏ `seed`): bắt các tham số dòng lệnh. CẦN phần này vì khối `model` của
        manifest KHÔNG có `ligand_features` (`models.architecture_settings` lọc bỏ), nên
        một lần chạy Morgan và một lần chạy MoLFormer có khối `model` GIỐNG HỆT nhau —
        thiếu nhãn thì hai biểu diễn bị gộp làm một và trung bình ra số vô nghĩa.
      - các khóa cấu trúc của manifest: bắt những khác biệt không hiện trong nhãn.
    Nhãn bỏ qua `gpu_memory_limit_mib`, `eval_every_steps`, `accumulate_steps` và vài
    khóa hạ tầng khác (xem `skip` trong summarize_runs.label), nên đợt đêm 13/11 giới hạn
    VRAM 4500 MiB vẫn gộp được với các đợt trước đặt 7500 — đúng ý muốn, vì mức giới hạn
    không đổi kết quả. `eval_batch_size` thì KHÔNG được bỏ qua: nó nằm trong nhãn, nên các
    lần chạy cũ không đặt khóa đó sẽ thành nhóm riêng. Đó cũng đúng, vì chúng là đợt khác.
    """
    structural = json.dumps({key: manifest.get(key) for key in IDENTITY_KEYS},
                            sort_keys=True, ensure_ascii=False, default=str)
    return name_of(manifest) + "\n" + structural


def name_of(manifest: dict[str, Any]) -> str:
    """Nhãn để in: nhãn của summarize_runs.py, bỏ phần seed=..."""
    parts = [part for part in label(manifest).split(" ") if not part.startswith("seed=")]
    return " ".join(parts)


def value_of(manifest: dict[str, Any], slice_name: str, metric: str) -> float | None:
    """Chỉ số tại mốc tốt nhất, đúng đường mà summarize_runs.py dùng."""
    block = best_block(manifest, slice_name) or {}
    ranking = block.get("ranking") or {}
    if metric in ranking:
        return None if ranking[metric] is None else float(ranking[metric])
    regression = block.get("regression") or {}
    if metric in regression:
        return None if regression[metric] is None else float(regression[metric])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slice", dest="slice_name", default="strict")
    parser.add_argument("--metric", default="pairwise_accuracy_macro",
                        help="khóa trong khối ranking hoặc regression của lát đó")
    parser.add_argument("--min-seeds", type=int, default=2,
                        help="bỏ qua nhóm có ít hạt giống hơn mức này (mặc định 2)")
    parser.add_argument("--grep", default="",
                        help="chỉ in nhóm có chuỗi này trong nhãn")
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--width", type=int, default=110, help="độ rộng cột nhãn")
    args = parser.parse_args()
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

    groups: dict[str, dict[str, Any]] = {}
    for path in sorted(runs_dir.glob("train_*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        value = value_of(manifest, args.slice_name, args.metric)
        if value is None:
            continue
        group = groups.setdefault(identity(manifest),
                                  {"name": name_of(manifest), "seeds": {}})
        seed = int(manifest.get("seed") or 0)
        # Cùng hạt giống chạy lại hai lần: giữ lần MỚI NHẤT, và nói ra.
        if seed in group["seeds"]:
            print(f"ghi chú: {group['name']} có hai lần chạy hạt {seed}; lấy lần mới hơn",
                  file=sys.stderr)
        group["seeds"][seed] = value

    rows = []
    for group in groups.values():
        if len(group["seeds"]) < args.min_seeds:
            continue
        if args.grep and args.grep not in group["name"]:
            continue
        points = [group["seeds"][seed] for seed in sorted(group["seeds"])]
        rows.append((statistics.fmean(points),
                     statistics.stdev(points) if len(points) > 1 else None,
                     len(points), group["name"], sorted(group["seeds"]), points))
    rows.sort(key=lambda row: row[0], reverse=True)

    if not rows:
        print(f"không có nhóm nào đủ {args.min_seeds} hạt giống"
              + (f" khớp '{args.grep}'" if args.grep else ""))
        return 0

    # Phần phân biệt các dòng nằm ở CUỐI nhãn (ligand_features, train_subset, fusion...),
    # còn đầu nhãn thì mọi dòng giống nhau. Cắt từ bên phải là cắt mất đúng phần cần đọc,
    # nên bỏ hẳn tiền tố chung và in nó một lần.
    token_lists = [name.split(" ") for _, _, _, name, _, _ in rows]
    shared: list[str] = []
    if len(token_lists) > 1:
        for tokens in zip(*token_lists):
            if len(set(tokens)) != 1:
                break
            shared.append(tokens[0])
    prefix = len(shared)

    width = args.width
    print(f"lát {args.slice_name} | {args.metric} | nhóm = giống nhau mọi thứ trừ hạt giống"
          f" | sd mẫu (n-1) | {runs_dir}")
    if shared:
        print(f"mọi dòng dưới đây đều có: {' '.join(shared)}")
    print(f"{'trung bình':>11}{'sd':>9}{'n':>4}  {'từng hạt':<26}khác nhau ở")
    print("-" * (50 + width))
    for mean, sd, count, name, seeds, points in rows:
        rest = " ".join(name.split(" ")[prefix:]) or "(không còn khác gì)"
        each = " ".join(f"{value:.4f}" for value in points)
        shown = rest if len(rest) <= width else "…" + rest[-(width - 1):]
        sd_text = "" if sd is None else f"{sd:.4f}"
        print(f"{mean:>11.4f}{sd_text:>9}{count:>4}  {each:<26}{shown}")
    print()
    print("Thứ tự hạt giống của cột 'từng hạt' (giống nhau ở mọi nhóm nếu chỉ in một dòng):")
    orders = {" ".join(str(seed) for seed in seeds) for _, _, _, _, seeds, _ in rows}
    for order in sorted(orders):
        print(f"   {order}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
