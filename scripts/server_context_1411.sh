#!/usr/bin/env bash
# Việc 9 lát (b) họp 09/11/2026: "cùng Context P, khác cặp (A, B)".
# KHÔNG cần GPU, không train lại gì: chỉ đọc các thư mục errors_* đã có.
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   bash scripts/server_context_1411.sh 2>&1 | tee runs/context_1411.log
#
# Chạy ngay ở tiền cảnh, vài phút là xong — không cần setsid.
#
# VIỆC 9 CÓ HAI LÁT, biên bản giao cả hai:
#   (a) cùng cặp (A, B), khác Context P  -> `error_analysis.py flip`, đã chạy 11/11:
#       nhóm nhãn ĐỔI DẤU giữa các P chỉ đúng chiều 0,5055 (đoán bừa), nhóm GIỮ DẤU
#       0,5908; sd dự đoán giữa các P là 0,0114 trong khi sd nhãn là 0,6566.
#   (b) cùng Context P, khác cặp (A, B)  -> lệnh `context`, chạy ở đây lần đầu.
# Lát (a) hỏi mô hình có ĐỌC protein không (câu trả lời: gần như không). Lát (b) hỏi
# chiều ngược lại: trong CÙNG một protein, mô hình có phân biệt được cặp này với cặp kia
# không. Hai lát cộng lại mới nói được mô hình đang biến thiên theo chiều nào.
#
# --worst 200 ghi ra worst_rows.csv: 200 dòng TRÁI DẤU lệch nặng nhất, để đọc tay như
# biên bản yêu cầu ("đọc trực tiếp các ca sai để tìm quy luật"), không chỉ nhìn số tổng.
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
WORST=200

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd "$script_dir/.." && pwd -P)"
expected="$(cd "$PROJECT_ROOT" 2>/dev/null && pwd -P)"
if [[ -z "$expected" || "$root" != "$expected" ]]; then
    echo "Script đang ở $root, không phải $PROJECT_ROOT. Dừng." >&2
    exit 1
fi
cd "$root" || exit 1

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
    echo "Env đang là '${CONDA_DEFAULT_ENV:-không có}', cần '$ENV_NAME'. Dừng." >&2
    exit 1
fi

mapfile -t folders < <(ls -dt runs/evaluator/errors_*_validation_*/ 2>/dev/null)
if [[ ${#folders[@]} -eq 0 ]]; then
    echo "Không thấy thư mục runs/evaluator/errors_*_validation_*." >&2
    echo "Dựng trước bằng: bash scripts/server_errors_gap_1311.sh <gpu>" >&2
    exit 1
fi

echo "=== [$(date '+%F %T')] việc 9 lát (b): cùng Context P, khác cặp (A, B)"
echo "=== ${#folders[@]} thư mục validation tìm thấy (mới nhất trước)"

failed=0
for folder in "${folders[@]}"; do
    folder="${folder%/}"
    echo
    echo "################################################################################"
    echo "=== $folder"
    # Mỗi dump đi kèm report.json ghi checkpoint đã dùng. In ra để biết bảng dưới đây là
    # của MÔ HÌNH NÀO — hai dump chỉ khác nhau ở cái tên dài, rất dễ đọc nhầm sang nhau.
    python - "$folder" <<'PY'
import json
import sys
from pathlib import Path

report = Path(sys.argv[1]) / "report.json"
if not report.is_file():
    print("   (không có report.json, không biết checkpoint nào)")
    raise SystemExit(0)
data = json.loads(report.read_text(encoding="utf-8"))
checkpoint = data.get("checkpoint") or data.get("checkpoint_path") or "?"
print(f"   checkpoint : {Path(str(checkpoint)).name}")
for key in ("partition", "rows"):
    if key in data:
        print(f"   {key:<11}: {data[key]}")
PY
    python scripts/error_analysis.py context --rows "$folder" --worst "$WORST" \
        || { failed=$((failed + 1)); echo "=== LỖI ở $folder"; }
done

echo
echo "=== [$(date '+%F %T')] xong | số thư mục lỗi: $failed"
echo "Mỗi thư mục giờ có thêm: context_pairs.json và worst_rows.csv"
echo "Kéo worst_rows.csv về máy mà đọc tay:"
echo "  scp 'brev-h100:$PROJECT_ROOT/runs/evaluator/errors_*_validation_*/worst_rows.csv' ."
echo "XONG_CONTEXT_1411"
