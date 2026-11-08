#!/usr/bin/env bash
# Hai lần train trả lời MỘT câu: ESM2 650M kém hơn 35M thật, hay chỉ vì cổng 128 chiều?
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_protein_width.sh 7 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/protein_width.log
#
# Tham số duy nhất là SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/protein_width.pid)
#
# VÌ SAO. Ngày 07/11/2026 đã chạy ESM2 650M đóng băng, mọi thứ khác giữ nguyên:
# đúng chiều macro 0,5614, so với 0,5657 +/- 0,0014 (3 seed) của 35M. Tức TỆ HƠN 0,0043.
# Nhưng `model.protein_out = 128`, nên embedding 480 chiều của 35M bị nén xuống 128 còn
# 1280 chiều của 650M cũng bị nén xuống đúng 128 — chặt hơn 10 lần. Vậy lần chạy đó chưa
# phân biệt được "650M không tốt hơn" với "cổng 128 chiều bóp mất phần hơn".
#
#   dòng 1  650M với protein_out = 256
#   dòng 2  35M  với protein_out = 256  <- ĐỐI CHỨNG, bắt buộc
#
# Thiếu dòng 2 thì nếu dòng 1 tốt hơn 0,5614 cũng không biết công của 650M hay công của
# cổng rộng hơn. Chỉ khi dòng 1 hơn dòng 2 rõ rệt thì mới nói được điều gì về 650M.
#
# protein_out = 256 chọn bằng ligand_out (256) đang có trong config, KHÔNG có nguồn từ
# bài báo nào — đây là tham số quét, không phải mặc định mới.
#
# Cache: dòng 1 đọc work/evaluator_cache_v4_esm650 (đã dựng xong 07/11), dòng 2 đọc cache
# mặc định work/evaluator_cache_v4. Không dựng lại gì. train.Caches từ chối cache dựng
# bằng model khác config (train.py:364-367), nên không thể lẫn hai cache.
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="protein_width"
CACHE_650M="work/evaluator_cache_v4_esm650"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: số GPU. Ví dụ: bash $0 7" >&2
    exit 1
fi
GPU_INDEX="$1"

# Đầu residual đã quét được 06/11; giữ nguyên để head không phải biến số.
BEST_SET="--set gpu_memory_limit_mib=7500 --set eval_batch_size=2048"
BEST_SET+=" --set selection_metric=pairwise_accuracy_macro"
BEST_SET+=" --set model.head_block=residual --set model.dropout=0.5"
BEST_SET+=" --set model.residual_blocks=7 --set model.residual_width=102"
BEST_SET+=" --set model.residual_hidden_factor=1.849"
BEST_SET+=" --set model.residual_hidden_dropout=0.3936"
BEST_SET+=" --set model.residual_dropout=0.0"
BEST_SET+=" --set learning_rate=9.86e-05 --set weight_decay=0.0"
BEST_SET+=" --set seed=20261023 --set model.protein_out=256"

M650="--set protein_encoder.model=facebook/esm2_t33_650M_UR50D"
M650+=" --set protein_encoder.embedding_dim=1280"
M650+=" --set paths.cache_dir=$CACHE_650M"

RUNS=("$M650" "")
NAMES=("650M protein_out=256" "35M protein_out=256 (ĐỐI CHỨNG)")

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

if [[ ! -d "$root/$CACHE_650M" ]]; then
    echo "Chưa có $CACHE_650M; chạy scripts/server_run_esm650.sh trước. Dừng." >&2
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

failed=0
for index in "${!RUNS[@]}"; do
    echo
    echo "--------------------------------------------------------------------------------"
    echo "=== [$(date '+%F %T')] dòng $((index + 1))/${#RUNS[@]}: ${NAMES[$index]}"
    started=$SECONDS
    # shellcheck disable=SC2086
    python run_evaluator.py --mode train --config frozen --head direct \
            $BEST_SET ${RUNS[$index]}
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
echo "Mốc: 35M protein_out=128 được 0,5657 +/- 0,0014 (3 seed); 650M protein_out=128"
echo "được 0,5614 (1 seed)."
echo "Đọc kết quả: python scripts/summarize_runs.py --ranking"
echo "XONG_PROTEIN_WIDTH"
