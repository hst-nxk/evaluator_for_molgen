#!/usr/bin/env bash
# Toàn bộ thí nghiệm GPU của họp 09/11/2026, chạy tuần tự trên subset nhỏ.
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_meeting_0911.sh 1 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/meeting_0911.log
#
# Tham số duy nhất là SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/meeting_0911.pid)
#
# ĐIỀU KIỆN TRƯỚC KHI CHẠY. Phải có bảng subset, dựng ở LOCAL rồi copy lên, hoặc dựng
# ngay trên server bằng:
#   python scripts/build_train_subset.py build --clusters 200 --rows-per-cluster 500
# Script này kiểm sự tồn tại của file và dừng ngay nếu thiếu.
#
# VÌ SAO TỪNG THÍ NGHIỆM, và nó trả lời câu nào trong biên bản 09/11/2026:
#
#   0. mốc        — direct trên subset, KHÔNG augment. Mọi so sánh phía sau đều so với
#                   lần này, nên nó phải chạy trước và không được bỏ.
#   1. augment    — direct + augment đảo chiều (Task 2, dòng 103/113/243). Anh Tiến dự
#                   đoán ở dòng 141: "chắc chắn nó sẽ cải thiện được ít nhiều".
#   2. mse        — direct, đổi loss sang MSE. KHÔNG có trong biên bản: cả nhóm đang tưởng
#                   objective hiện tại là MSE, nhưng config ghi huber với delta 1,0 và
#                   không script nào ghi đè. Lần chạy này cho biết chênh lệch thật là bao
#                   nhiêu, để bảng của việc 7 nói đúng objective.
#   3. linear     — baseline Linear không bias thay cả tầng Fusion (Task 3, dòng 173/246).
#   4. ridge      — cùng head linear nhưng weight_decay 0,1, tức Ridge. AdamW dùng weight
#                   decay tách rời nên với một lớp Linear duy nhất đó đúng là phạt L2.
#   5. twin+edit  — ô thứ tư của bảng ablation: F_A đi với A, F_B đi với B (Task 4, dòng
#                   163/248). Phản đối xứng do cấu tạo.
#   6. antisym    — hàm lẻ trên vector hiệu, nhân cổng theo ngữ cảnh (Task 4, dòng 135 của
#                   anh Long + dòng 123 của anh Tiến về ReLU phá tính lẻ). Cũng phản đối
#                   xứng do cấu tạo, nhưng còn học được tương tác trong cặp.
#   7. antisym-lt — cùng head trên, nhưng antisymmetric_hidden rỗng: một lớp Linear
#                   không bias trên vector hiệu, tức mô hình tuyến tính CHẮC CHẮN giữ
#                   được tính phản đối xứng. Đây là trần dưới về dung lượng của head 6:
#                   nếu 6 không hơn 7 thì phần phi tuyến của hàm lẻ không đóng góp gì.
#   8. no-expand  — mốc ở thí nghiệm 0 nhưng tắt expand_both_directions. Cũng không có
#                   trong biên bản: phép nhân đôi hai chiều hiện tại đổi dấu PREDICTION đã
#                   lưu, mà tỉ lệ đúng chiều kiểm sign(p) == sign(t) nên dòng nhân đôi là
#                   dư thừa về mặt logic — nó chỉ đổi trọng số macro. Lần chạy này cho biết
#                   con số thật khi không nhân đôi.
#
# MỌI THÍ NGHIỆM DÙNG CHUNG: cùng subset, cùng seed, cùng cấu hình 7x102 đang là tốt nhất,
# cùng selection_metric=pairwise_accuracy_macro. Chỉ MỘT thứ đổi mỗi lần, nếu không thì
# chênh lệch không quy được cho nguyên nhân nào.
#
# Cache: dùng work/evaluator_cache_v4 có sẵn, KHÔNG dựng lại gì. Subset và augment không
# chạm tới cache: cache đánh khoá theo phân tử và theo protein, không theo dòng.
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="meeting_0911"
SUBSET="work/splits/train_subset_c200_r500.parquet"
if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: số GPU. Ví dụ: bash $0 1" >&2
    exit 1
fi
GPU_INDEX="$1"

# Cấu hình nền, giữ nguyên của lần chạy tốt nhất tuần 03-08/11/2026.
BASE="--set gpu_memory_limit_mib=7500 --set eval_batch_size=2048"
BASE+=" --set selection_metric=pairwise_accuracy_macro"
BASE+=" --set model.head_block=residual --set model.dropout=0.5"
BASE+=" --set model.residual_blocks=7 --set model.residual_width=102"
BASE+=" --set model.residual_hidden_factor=1.849"
BASE+=" --set model.residual_hidden_dropout=0.3936"
BASE+=" --set model.residual_dropout=0.0"
BASE+=" --set learning_rate=9.86e-05 --set weight_decay=0.0"
BASE+=" --set inputs.train_subset=$SUBSET"

# Head phản đối xứng và head linear KHÔNG dùng được khối residual, nên hai nhóm thí
# nghiệm có cấu hình nền khác nhau. Giữ nguyên learning rate và dropout để phần đổi là
# đúng kiến trúc.
PLAIN="--set gpu_memory_limit_mib=7500 --set eval_batch_size=2048"
PLAIN+=" --set selection_metric=pairwise_accuracy_macro"
PLAIN+=" --set model.dropout=0.5"
PLAIN+=" --set learning_rate=9.86e-05 --set weight_decay=0.0"
PLAIN+=" --set inputs.train_subset=$SUBSET"

# --- Chốt thư mục: script phải nằm đúng trong PROJECT_ROOT -------------------------------
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

if [[ ! -f "$root/$SUBSET" ]]; then
    echo "Thiếu $SUBSET. work/ không đi theo git; scp từ máy local lên:" >&2
    echo "  scp work/splits/train_subset_c200_r500.* brev-h100:$PROJECT_ROOT/work/splits/" >&2
    exit 1
fi

if [[ ! -f "$root/work/evaluator_cache_v4/protein_embeddings.parquet" ]]; then
    echo "Thiếu cache embedding protein. Dựng trước bằng:" >&2
    echo "  python run_evaluator.py --mode prepare" >&2
    exit 1
fi

cache="$root/.cache"
mkdir -p runs "$cache"
log="$root/runs/$LOG_NAME.log"
exec >>"$log" 2>&1

exec 9>"$cache/$LOG_NAME.lock"
if ! flock -n 9; then
    echo "=== [$(date '+%F %T')] Đang có một lần chạy khác. Dừng."
    exit 1
fi
echo "$$" >"$cache/$LOG_NAME.pid"
trap 'echo "=== [$(date "+%F %T")] bị dừng bởi tín hiệu"; exit 1' TERM INT

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

echo
echo "################################################################################"
echo "=== [$(date '+%F %T')] bắt đầu | GPU $GPU_INDEX | PID $$ | $root"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
echo "=== subset: $SUBSET"
python - <<'PYEOF'
import json
from pathlib import Path
meta = Path("work/splits/train_subset_c200_r500.json")
if meta.is_file():
    report = json.loads(meta.read_text(encoding="utf-8"))
    print(f"=== subset: {report['rows']:,} dòng | {report['clusters_with_rows']} cụm | "
          f"trần {report['rows_per_cluster_cap']} dòng/cụm | "
          f"cụm lớn nhất chiếm {100 * report['largest_cluster_share']:.2f} %")
PYEOF
echo "=== MỐC ĐỂ SO: trên TOÀN BỘ train, đúng chiều macro = 0,5670 / 0,5643 / 0,5658"
echo "=== (3 hạt giống, trung bình 0,5657 +/- 0,0014). Subset nhỏ hơn nên con số tuyệt đối"
echo "=== KHÔNG so trực tiếp với mốc đó được; chỉ so các lần trong chính script này."

# Mỗi phần tử: "nhãn|đối số của run_evaluator.py"
declare -a RUNS=(
"0 mốc: direct trên subset, không augment|--mode train --config frozen --head direct $BASE"
"1 augment đảo chiều (Task 2)|--mode train --config frozen --head direct $BASE --set augment_reverse=true"
"2 đổi loss sang MSE (objective thật đang là huber)|--mode train --config frozen --head direct $BASE --set loss=mse"
"3 baseline Linear không bias (Task 3)|--mode train --config frozen --head direct $PLAIN --set model.head_block=linear"
"4 baseline Ridge (Linear + weight_decay 0,1)|--mode train --config frozen --head direct $PLAIN --set model.head_block=linear --set weight_decay=0.1"
"5 twin có nhánh edit (Task 4, ô thứ tư)|--mode train --config frozen --head twin $BASE --set model.twin_edit=true"
"6 antisymmetric: hàm lẻ x cổng ngữ cảnh (Task 4)|--mode train --config frozen --head antisymmetric $PLAIN"
"7 antisymmetric tuyến tính (một lớp Linear không bias trên vector hiệu)|--mode train --config frozen --head antisymmetric $PLAIN --set model.antisymmetric_hidden=[]"
"8 mốc nhưng TẮT expand_both_directions|--mode train --config frozen --head direct $BASE --set metrics.expand_both_directions=false"
)

failed=0
for index in "${!RUNS[@]}"; do
    label="${RUNS[$index]%%|*}"
    arguments="${RUNS[$index]#*|}"
    echo
    echo "--------------------------------------------------------------------------------"
    echo "=== [$(date '+%F %T')] $((index + 1))/${#RUNS[@]}: $label"
    echo "=== python run_evaluator.py $arguments"
    started=$SECONDS
    python run_evaluator.py $arguments
    status=$?
    minutes=$(((SECONDS - started) / 60))
    if [[ $status -eq 0 ]]; then
        echo "=== [$(date '+%F %T')] $((index + 1)) xong sau $minutes phút"
    else
        echo "=== [$(date '+%F %T')] $((index + 1)) LỖI mã $status sau $minutes phút"
        failed=$((failed + 1))
    fi
done

echo
echo "=== [$(date '+%F %T')] xong tất cả | số lần lỗi: $failed"
echo "Đọc kết quả: python scripts/summarize_runs.py --ranking"
echo "Với các lần phản đối xứng, kiểm thêm khóa asymmetry trong manifest: antisymmetry_max"
echo "phải <= 1e-8 và meets_target phải là true. Nếu KHÔNG đạt thì là lỗi kiến trúc, không"
echo "phải chuyện train chưa hội tụ."
echo "XONG_MEETING_0911"
