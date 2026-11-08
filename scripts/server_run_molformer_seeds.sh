#!/usr/bin/env bash
# Hai hạt giống nữa cho MoLFormer: bịt lỗ duy nhất còn lại trong bảng kết quả tuần này.
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_molformer_seeds.sh 1 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/molformer_seeds.log
#
# Tham số duy nhất là SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/molformer_seeds.pid)
#
# VÌ SAO. Ngày 07/11/2026 chạy MoLFormer thay Morgan, mọi thứ khác giữ nguyên: đúng chiều
# macro 0,5600 so với 0,5670 của hạt giống 20261023 dùng Morgan. Kém 0,0070 ~ 5 lần độ
# lệch chuẩn hạt giống (0,0014), nghe thì chắc, NHƯNG đó là so một lần chạy với một lần
# chạy. Morgan có 3 hạt giống (0,5670 / 0,5643 / 0,5658), MoLFormer mới có 1.
#
# Và đây là kết quả âm DUY NHẤT của tuần chưa có cơ chế. Giả thuyết ban đầu — "MoLFormer
# làm nhoè phép sửa giữa A và B" — đã tự đo và BỊ LOẠI: tỉ lệ khoảng cách cosine trong
# cặp so với giữa cặp là 0,2659 với MoLFormer và 0,2443 với Morgan, tức MoLFormer không
# kém hơn ở điểm đó (scripts/representation_diagnostics.py ligand).
#
# Nên trước khi đi tìm cơ chế, phải chắc hiện tượng có thật. Hai hạt giống này trả lời
# đúng câu đó: nếu trung bình 3 hạt giống của MoLFormer vẫn thấp hơn Morgan quá 0,0014
# thì hiện tượng có thật và đáng đi tìm cơ chế; nếu ba lần trải rộng và chồng lên Morgan
# thì 0,5600 chỉ là một hạt giống xấu, và phải sửa lại cách báo cáo.
#
# Cache: dùng work/evaluator_cache_v4 có sẵn, KHÔNG dựng lại gì. Cache embedding phân tử
# (molecule_embeddings.parquet) đã dựng ngày 07/11 bằng
#   python run_evaluator.py --mode prepare --molecule-embeddings
# Nhánh phép sửa vẫn dùng fingerprint ở cả hai chế độ, cố ý: như vậy ablation chỉ đổi
# đúng một thứ là cách biểu diễn phân tử.
#
# Trọng số: ibm/MoLFormer-XL-both-10pct, revision ghim ở tag compat-v4 (commit 7b12d946),
# xem config.json mục stage3.molecule_encoder.
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="molformer_seeds"
CACHE_FILE="work/evaluator_cache_v4/molecule_embeddings.parquet"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: số GPU. Ví dụ: bash $0 1" >&2
    exit 1
fi
GPU_INDEX="$1"

SEEDS=(20261024 20261025)
# Morgan ở đúng hai hạt giống này cho 0,5643 và 0,5658 — so thẳng từng cặp được.
NAMES=("hạt giống 20261024 | Morgan cùng hạt giống: 0,5643"
       "hạt giống 20261025 | Morgan cùng hạt giống: 0,5658")

BEST_SET="--set gpu_memory_limit_mib=7500 --set eval_batch_size=2048"
BEST_SET+=" --set selection_metric=pairwise_accuracy_macro"
BEST_SET+=" --set model.ligand_features=embedding"
BEST_SET+=" --set model.head_block=residual --set model.dropout=0.5"
BEST_SET+=" --set model.residual_blocks=7 --set model.residual_width=102"
BEST_SET+=" --set model.residual_hidden_factor=1.849"
BEST_SET+=" --set model.residual_hidden_dropout=0.3936"
BEST_SET+=" --set model.residual_dropout=0.0"
BEST_SET+=" --set learning_rate=9.86e-05 --set weight_decay=0.0"

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

if [[ ! -f "$root/$CACHE_FILE" ]]; then
    echo "Thiếu $CACHE_FILE. Dựng trước bằng:" >&2
    echo "  python run_evaluator.py --mode prepare --molecule-embeddings" >&2
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
echo "=== Đã có: MoLFormer hạt giống 20261023 = 0,5600. Morgan 3 hạt giống ="
echo "=== 0,5670 / 0,5643 / 0,5658, trung bình 0,5657 +/- 0,0014."
echo "=== Chênh nhỏ hơn 0,0014 thì KHÔNG được gọi là kém hơn."

failed=0
for index in "${!SEEDS[@]}"; do
    echo
    echo "--------------------------------------------------------------------------------"
    echo "=== [$(date '+%F %T')] $((index + 1))/${#SEEDS[@]}: ${NAMES[$index]}"
    started=$SECONDS
    python run_evaluator.py --mode train --config frozen --head direct \
            $BEST_SET --set seed="${SEEDS[$index]}"
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
echo "Đọc kết quả: python scripts/summarize_runs.py --ranking, lọc các lần có"
echo "ligand_features=embedding; lấy trung bình 3 hạt giống rồi mới so với Morgan."
echo "XONG_MOLFORMER_SEEDS"
