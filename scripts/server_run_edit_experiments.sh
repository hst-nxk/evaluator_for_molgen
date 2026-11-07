#!/usr/bin/env bash
# Bốn lần train trả lời MỘT câu: nhánh edit đang học quy luật hay đang tra bảng?
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_edit_experiments.sh 7 > /dev/null 2>&1 &
#   tail -f runs/edit_experiments.log
#
# Tham số duy nhất là SỐ GPU (bắt buộc, không tự dò): GPU trên máy này dùng chung với
# người khác, nên script không được tự chọn.
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/edit_experiments.pid)
#
# VÌ SAO BỐN DÒNG NÀY. Phân tích ca sai trên train ngày 07/11/2026
# (runs/evaluator/errors_frozen_direct_train_20261107_081720) cho bảng MAE theo số dòng
# train dùng chung một phép sửa, đi NGƯỢC chiều mọi nhóm khác:
#     phép sửa gặp 1 lần   : 1.676.782 dòng, MAE 0,1440, đúng dấu 0,9881
#     phép sửa gặp 66+ lần : 1.327.275 dòng, MAE 0,3216, đúng dấu 0,8790
# Mọi nhóm khác đi đúng chiều mật độ (protein_rows 0,3045 -> 0,1582; molecule_a_rows
# 0,3678 -> 0,1329). Chỉ phép sửa là ngược, tức nhánh edit được dùng như bảng tra.
#
#   dòng 1  bỏ phép sửa chỉ gặp một lần khỏi train (25 % dòng)
#   dòng 2  ĐỐI CHỨNG: bỏ ngẫu nhiên đúng cùng số dòng. Thiếu dòng này thì "kém đi"
#           không phân biệt được với "ít dữ liệu hơn thì kém hơn".
#   dòng 3  hạ learning rate của riêng nhánh edit còn 1/10
#   dòng 4  hạ còn 3/10
# Dòng 3 và 4 theo Wang W, Tran D, Feiszli M. What Makes Training Multi-Modal
# Classification Networks Hard? CVPR 2020, arXiv:1905.12681 — nguyên nhân thứ hai họ nêu
# là các mô thức quá khớp và tổng quát hóa với TỐC ĐỘ KHÁC NHAU. Hệ số 0,1 và 0,3 là tham
# số quét, CHƯA CÓ NGUỒN: bài báo không cho con số nào.
#
# KHÔNG có dòng đối chứng "cấu hình gốc": cấu hình này với seed 20261023 đã chạy xong
# ngày 06/11, đúng chiều 0,5670. Bốn dòng dưới dùng đúng cấu hình và đúng seed đó nên so
# thẳng được, không cần chạy lại.
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="edit_experiments"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: số GPU. Ví dụ: bash $0 7" >&2
    exit 1
fi
GPU_INDEX="$1"

# Cấu hình tốt nhất của đợt quét 06/11 (trial seed 20261023, đúng chiều 0,5670).
# Giữ nguyên từng con số: đổi một cái là mất mốc so sánh.
BEST_SET="--set gpu_memory_limit_mib=7500 --set eval_batch_size=2048"
BEST_SET+=" --set selection_metric=pairwise_accuracy_macro"
BEST_SET+=" --set model.head_block=residual --set model.dropout=0.5"
BEST_SET+=" --set model.residual_blocks=7 --set model.residual_width=102"
BEST_SET+=" --set model.residual_hidden_factor=1.849"
BEST_SET+=" --set model.residual_hidden_dropout=0.3936"
BEST_SET+=" --set model.residual_dropout=0.0"
BEST_SET+=" --set learning_rate=9.86e-05 --set weight_decay=0.0"
BEST_SET+=" --set seed=20261023"

RUNS=(
    "--set min_edit_rows=2"
    "--set min_edit_rows=2 --set min_edit_rows_control=true"
    "--set branch_learning_rate_scale.edit_encoder=0.1"
    "--set branch_learning_rate_scale.edit_encoder=0.3"
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

cache="$root/.cache"
mkdir -p runs "$cache"
log="$root/runs/$LOG_NAME.log"
exec >>"$log" 2>&1

# Không cho chạy hai lần cùng lúc: hai lần train cùng GPU sẽ vượt trần VRAM.
exec 9>"$cache/$LOG_NAME.lock"
if ! flock -n 9; then
    echo "=== [$(date '+%F %T')] Đang có một lần chạy khác. Dừng."
    exit 1
fi
echo "$$" >"$cache/$LOG_NAME.pid"
trap 'echo "=== [$(date "+%F %T")] bị dừng bởi tín hiệu"; exit 1' TERM INT

# Ghim GPU và chặn số luồng: máy dùng chung, không được chiếm hết CPU của người khác.
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

echo
echo "################################################################################"
echo "=== [$(date '+%F %T')] bắt đầu | GPU $GPU_INDEX | PID $$ | $root"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true

failed=0
for index in "${!RUNS[@]}"; do
    extra="${RUNS[$index]}"
    echo
    echo "--------------------------------------------------------------------------------"
    echo "=== [$(date '+%F %T')] dòng $((index + 1))/${#RUNS[@]}: $extra"
    started=$SECONDS
    # shellcheck disable=SC2086
    python run_evaluator.py --mode train --config frozen --head direct \
            $BEST_SET $extra
    status=$?
    minutes=$(((SECONDS - started) / 60))
    if [[ $status -eq 0 ]]; then
        echo "=== [$(date '+%F %T')] dòng $((index + 1)) xong sau $minutes phút"
    else
        echo "=== [$(date '+%F %T')] dòng $((index + 1)) LỖI mã $status sau $minutes phút"
        failed=$((failed + 1))
    fi
done

echo
echo "=== [$(date '+%F %T')] xong tất cả | số dòng lỗi: $failed"
echo "Đọc kết quả: python scripts/summarize_runs.py --ranking"
echo "XONG_EDIT_EXPERIMENTS"
