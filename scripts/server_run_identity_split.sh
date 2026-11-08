#!/usr/bin/env bash
# Hai lần train + hai lần chấm train trả lời câu hỏi còn lại: trần 0,566 có phải do
# protein chưa từng thấy hay không?
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_identity_split.sh 1 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/identity_split.log
#
# Tham số duy nhất là SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/identity_split.pid)
#
# VÌ SAO. Tuần 03-08/11/2026 đã loại bảy nguyên nhân của trần 0,566: kiến trúc khối hợp
# nhất, dung lượng mô hình, ràng buộc đối xứng, số lượng dòng train, nhánh phép sửa,
# biểu diễn phân tử, dung lượng mã hoá protein. Lát cắt theo chất lượng nhãn cho thấy
# nhãn sạch chỉ nâng validation thêm ~0,04 mà khoảng cách train-validation KHÔNG co lại
# (0,3012 -> 0,3070 -> 0,3112). Còn đúng một nguyên nhân chưa thử: validation toàn protein
# thuộc họ mà train chưa từng thấy.
#
# Phép thử: giữ nguyên mọi thứ, chỉ đổi NGƯỠNG GIỐNG NHAU dùng để gom protein thành họ.
#     ngưỡng 30 %  ->  1.510 họ   ->  họ to, validation xa train hơn
#     ngưỡng 50 %  ->  2.415 họ   <- cách chia đang dùng (ĐÃ CÓ KẾT QUẢ, không chạy lại)
#     ngưỡng 80 %  ->  3.290 họ   ->  họ nhỏ, validation gần train hơn
# Nếu điểm validation đi lên theo ngưỡng thì nguyên nhân được xác nhận; nếu ba điểm bằng
# nhau trong khoảng nhiễu hạt giống (0,0014) thì giả thuyết "do protein lạ" cũng bị loại
# và phải tìm chỗ khác.
#
# VÌ SAO KHÔNG PHẢI BA LẦN TRAIN. Mốc 50 % không cần chạy lại: ngày 08/11/2026 đã kiểm
#     python scripts/build_identity_split.py verify --identity 0.5 \
#             --clusters runs/dataset/mmseqs_v4b/p50_cluster.tsv
# và đường đổi partition tái tạo ĐÚNG TỪNG DÒNG cách chia đang có (5.428/5.428 protein,
# 0 protein đổi chỗ). Nên các lần chạy cũ (train 0,8682 / validation 0,5670, hạt giống
# 20261023) chính là mốc 50 %, đi qua đúng cùng một đường mã.
#
# VÌ SAO KHÔNG DỰNG DATASET v5. Cách chia chỉ phụ thuộc protein. Cặp phân tử, nhãn, cache
# fingerprint và cache embedding không đổi (hai cache đánh khoá theo phân tử và theo
# protein, không theo partition). Thứ duy nhất đổi là cột `partition`, và Stage 3 chỉ đọc
# cột đó ở một chỗ duy nhất (data.read_columns). Bảng đổi nạp qua
# `--set inputs.partition_override=...`; dataset_v4 không bị chạm.
#
# SỐ DÒNG ba điều kiện gần bằng nhau nên so được (đo ngày 08/11/2026 trên local):
#     50 %  train 6.627.559  validation 493.852  (2.686 / 1.249 protein)
#     30 %  train 6.640.992  validation 494.022  (3.213 /   965 protein)
#     80 %  train 6.626.286  validation 486.006  (2.488 / 1.389 protein)
#
# MỖI DÒNG CHẤM CẢ TRAIN, vì câu hỏi là khoảng cách train-validation chứ không chỉ là
# điểm validation. Chấm train mất 8-9 phút mỗi lần, đã đo ngày 08/11.
#
# Test KHÔNG bị đụng tới: data.check_partition vẫn chặn, bảng đổi không mở đường nào.
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="identity_split"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: số GPU. Ví dụ: bash $0 1" >&2
    exit 1
fi
GPU_INDEX="$1"

TAGS=("id30" "id80")
NAMES=("ngưỡng 30 % | 1.510 họ | validation XA train hơn"
       "ngưỡng 80 % | 3.290 họ | validation GẦN train hơn")

# Cấu hình thắng của tuần này, giữ y hệt mọi lần chạy khác để chỉ cách chia là biến.
BEST_SET="--set gpu_memory_limit_mib=7500 --set eval_batch_size=2048"
BEST_SET+=" --set selection_metric=pairwise_accuracy_macro"
BEST_SET+=" --set model.head_block=residual --set model.dropout=0.5"
BEST_SET+=" --set model.residual_blocks=7 --set model.residual_width=102"
BEST_SET+=" --set model.residual_hidden_factor=1.849"
BEST_SET+=" --set model.residual_hidden_dropout=0.3936"
BEST_SET+=" --set model.residual_dropout=0.0"
BEST_SET+=" --set learning_rate=9.86e-05 --set weight_decay=0.0"
BEST_SET+=" --set seed=20261023"

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

for tag in "${TAGS[@]}"; do
    if [[ ! -f "$root/splits/partition_override_$tag.parquet" ]]; then
        echo "Thiếu splits/partition_override_$tag.parquet. Pull lại rồi chạy." >&2
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
echo "=== Mốc 50 % đã có, KHÔNG chạy lại: train 0,8682 / validation 0,5670 (hạt giống"
echo "=== 20261023). Nhiễu hạt giống của mốc đó là 0,0014 — chênh nhỏ hơn mức này thì"
echo "=== KHÔNG được gọi là tốt hơn hay kém hơn."

failed=0
for index in "${!TAGS[@]}"; do
    tag="${TAGS[$index]}"
    override="--set inputs.partition_override=splits/partition_override_$tag.parquet"
    step_log="$root/runs/${LOG_NAME}_${tag}_train.txt"

    echo
    echo "--------------------------------------------------------------------------------"
    echo "=== [$(date '+%F %T')] $((index + 1))/${#TAGS[@]} TRAIN: ${NAMES[$index]}"
    started=$SECONDS
    python run_evaluator.py --mode train --config frozen --head direct \
            $BEST_SET $override 2>&1 | tee "$step_log"
    status=${PIPESTATUS[0]}
    minutes=$(((SECONDS - started) / 60))
    if [[ $status -ne 0 ]]; then
        echo "=== [$(date '+%F %T')] train $tag LỖI mã $status sau $minutes phút"
        failed=$((failed + 1))
        continue
    fi
    echo "=== [$(date '+%F %T')] train $tag xong sau $minutes phút"

    # Lấy checkpoint từ chính log của lần train này, không đoán theo thời gian sửa file:
    # trên GPU dùng chung có thể có lần chạy khác ghi vào runs/evaluator cùng lúc.
    checkpoint="$(grep -o 'checkpoint tốt nhất.*-> .*\.pt' "$step_log" | tail -n 1 \
                  | sed 's/.*-> //')"
    if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
        echo "=== [$(date '+%F %T')] không đọc được checkpoint của $tag từ $step_log"
        failed=$((failed + 1))
        continue
    fi

    echo "=== [$(date '+%F %T')] CHẤM TRAIN $tag | checkpoint: $checkpoint"
    started=$SECONDS
    python run_evaluator.py --mode eval --partition train \
            --checkpoint "$checkpoint" $override \
            --set gpu_memory_limit_mib=7500 --set eval_batch_size=2048
    status=$?
    minutes=$(((SECONDS - started) / 60))
    if [[ $status -eq 0 ]]; then
        echo "=== [$(date '+%F %T')] chấm train $tag xong sau $minutes phút"
    else
        echo "=== [$(date '+%F %T')] chấm train $tag LỖI mã $status sau $minutes phút"
        failed=$((failed + 1))
    fi
done

echo
echo "=== [$(date '+%F %T')] xong tất cả | số lần lỗi: $failed"
echo "Đọc kết quả: python scripts/summarize_runs.py --ranking (cột đúng chiều macro,"
echo "lát 'all'), và dòng 'n = ...' của mỗi lần chấm train phải cùng cỡ mới so được."
echo "XONG_IDENTITY_SPLIT"
