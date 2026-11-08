#!/usr/bin/env bash
# Chấm ba checkpoint trên TRAIN để biết phần nào làm khoảng cách train-valid giãn ra.
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_eval_train_gap.sh 1 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/eval_train_gap.log
#
# Tham số duy nhất là SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/eval_train_gap.pid)
#
# VÌ SAO. Họp 02/11/2026 giao: "train 0,75 vs valid 0,55 lệch 20 điểm, đổi cách chia thử
# lại". Đến nay train chỉ được chấm HAI lần, trên hai cấu hình khác nhau ít nhất sáu chỗ
# (head, dropout, learning_rate, selection_metric, seed, chế độ đo), nên con số
# 0,7481 (02/11) và 0,8682 (07/11) KHÔNG quy được cho thay đổi nào. Thêm nữa lần 02/11
# chạy hết 3 phút 20 giây, trong khi validation 494 nghìn dòng đã mất ~60-80 giây, nên rất
# có thể nó chỉ chấm một phần train — tức hai số đó có thể không so được với nhau.
#
# Ba checkpoint dưới đây CÙNG seed 20261023, CÙNG selection_metric, CÙNG dropout 0,5, chỉ
# khác phần head. Chấm cả ba bằng cùng một lệnh thì ba số train so được với nhau, và so
# được với validation đã biết:
#     frozen_direct_20261106_144423  head phẳng        3.710.273 tham số  valid 0,5607
#     frozen_direct_20261106_151113  residual 2x512    4.632.897 tham số  valid 0,5616
#     frozen_direct_20261106_184804  residual 7x102    3.478.022 tham số  valid 0,5670
#
# Lưu ý khi đọc kết quả: cấu hình THẮNG lại là cấu hình NHỎ NHẤT (3.478.022, ít hơn head
# phẳng 6 %). Nên nếu train tăng theo thứ tự đó thì nguyên nhân KHÔNG phải là thêm dung
# lượng, mà là dạng kiến trúc hoặc cách train (learning rate 9,86e-05, chạy tới epoch 3).
# Câu "thêm dung lượng nên khớp train chặt hơn" chỉ đúng cho projector và attention
# (+482.368 và +2.063.936 tham số, cùng kém hơn concat), không đúng cho cấu hình thắng.
#
# Không train lại gì, chỉ nạp checkpoint và chấm. Model settings nằm trong chính file .pt.
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="eval_train_gap"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: số GPU. Ví dụ: bash $0 1" >&2
    exit 1
fi
GPU_INDEX="$1"

CHECKPOINTS=(
    "frozen_direct_20261106_144423"
    "frozen_direct_20261106_151113"
    "frozen_direct_20261106_184804"
)
NAMES=(
    "head phẳng | 3.710.273 tham số | valid 0,5607"
    "residual 2x512 | 4.632.897 tham số | valid 0,5616"
    "residual 7x102 | 3.478.022 tham số | valid 0,5670"
)

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

for name in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "$root/runs/evaluator/$name.pt" ]]; then
        echo "Thiếu checkpoint runs/evaluator/$name.pt. Dừng." >&2
        exit 1
    fi
done

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
echo "=== Đọc dòng 'n = ...' của mỗi lần chấm: ba lần PHẢI cùng số dòng mới so được."

failed=0
for index in "${!CHECKPOINTS[@]}"; do
    echo
    echo "--------------------------------------------------------------------------------"
    echo "=== [$(date '+%F %T')] $((index + 1))/${#CHECKPOINTS[@]}: ${NAMES[$index]}"
    echo "=== checkpoint: ${CHECKPOINTS[$index]}.pt"
    started=$SECONDS
    python run_evaluator.py --mode eval --partition train \
            --checkpoint "runs/evaluator/${CHECKPOINTS[$index]}.pt" \
            --set gpu_memory_limit_mib=7500 --set eval_batch_size=2048
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
echo "Mốc cũ để so: train 0,7481 / valid 0,5569 (02/11, cấu hình KHÁC, có thể chỉ chấm"
echo "một phần train) và train 0,8682 / valid 0,5643 (07/11, --mode errors)."
echo "XONG_EVAL_TRAIN_GAP"
