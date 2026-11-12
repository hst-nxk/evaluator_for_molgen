#!/usr/bin/env bash
# Đợt chạy đêm 13/11/2026: bốn phần, 12 lần chạy, MỌI lần chọn checkpoint theo lát `strict`.
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_night_1311.sh 5 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/night_1311.log
#
# Tham số duy nhất là CHỈ SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/night_1311.pid)
#
# MỐC SO (3 hạt giống, đo trên lát `strict`, chọn checkpoint theo lát `strict`):
#   direct + residual, Morgan      0,5478 +/- 0,0002
#   direct + residual, MoLFormer   0,5602 +/- 0,0010
#   antisymmetric, Morgan          0,5477 +/- 0,0012
#   antisymmetric + MoLFormer      0,5586 +/- 0,0010
#
# PHẦN 1 + 2 (~2,5 giờ): ĐỐI CHỨNG KHỚP cho việc 5 họp 09/11 (yêu cầu của anh Tiến về tính
#   phản đối xứng). Các lần chạy `antisymmetric` KHÔNG có khối residual, vì head đó BỎ QUA
#   `head_block`: trong src/molopt/evaluator/models.py nhánh residual nằm ở `elif` ngay sau
#   `if head == HEAD_ANTISYMMETRIC`. Vậy nên:
#     - thêm --set model.head_block=residual vào lần chạy antisymmetric là VÔ NGHĨA, mô hình
#       không đổi gì, ra đúng số cũ (đã suýt chạy nhầm như thế, xem nhật ký 13/11);
#     - so `antisymmetric` (head phẳng) với `direct` + 7x102 là còn MỘT BIẾN CHƯA KHOÁ.
#   Hai phần này chạy `direct` với head PHẲNG, cùng lr/wd/dropout, để hiệu "ép phản đối
#   xứng" đọc được sạch. Nếu hiệu vẫn quanh 0 thì kết luận "ép phản đối xứng miễn phí" đứng
#   vững hoàn toàn, và paper nhận được kiến trúc thoả f(A,B) = -f(B,A) đúng 0,000e+00.
#
# PHẦN 3 (~1,5 giờ): fusion=attention, 3 hạt. Phép đo 1 hạt ngày 11/11 cho +0,0026 trên lát
#   `strict`; đây là biến thể fusion DUY NHẤT còn đáng đo (projector đã đo ba lần, đều là số
#   không hoặc âm trên `strict`). Mốc so: 0,5478.
#
# PHẦN 4 (~1,5 giờ): bỏ nhánh edit, trên nền MoLFormer. Số `direct_no_edit` đang có (0,5379)
#   là nền Morgan và chọn theo lát `all`; bảng ablation của paper phải đo trên mô hình cuối
#   chứ không phải mô hình cũ. Mốc so: 0,5602.
#
# ĐỌC LOG: mỗi mốc đo in hai dòng ranking; dòng "lát strict" là dòng dùng để chọn và là số
# của đợt này. KHÔNG dùng partition test ở bất kỳ đâu (run_evaluator.py chặn sẵn).
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="night_1311"
SEEDS=(20261023 20261024 20261025)

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: chỉ số GPU. Ví dụ: bash $0 5" >&2
    exit 1
fi
GPU_INDEX="$1"

# gpu_memory_limit_mib hạ xuống 4500 (các đợt trước là 7500): tối 13/11 GPU 5 chỉ còn
# ~5.600 MiB trống vì máy dùng chung. Nhu cầu thật của job này ~800 MiB.
COMMON="--set gpu_memory_limit_mib=4500 --set eval_batch_size=2048"
COMMON+=" --set selection_metric=pairwise_accuracy_macro --set selection_slice=strict"
COMMON+=" --set model.dropout=0.5 --set learning_rate=9.86e-05 --set weight_decay=0.0"

RESIDUAL="--set model.head_block=residual --set model.residual_blocks=7"
RESIDUAL+=" --set model.residual_width=102 --set model.residual_hidden_factor=1.849"
RESIDUAL+=" --set model.residual_hidden_dropout=0.3936 --set model.residual_dropout=0.0"

MOLFORMER="--set model.ligand_features=embedding"

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

for name in protein_embeddings molecule_embeddings; do
    if [[ ! -f "$root/work/evaluator_cache_v4/$name.parquet" ]]; then
        echo "Thiếu work/evaluator_cache_v4/$name.parquet. Dừng." >&2
        exit 1
    fi
done

cache="$root/.cache"
mkdir -p runs "$cache"
log="$root/runs/$LOG_NAME.log"
exec >>"$log" 2>&1

exec 9>"$cache/$LOG_NAME.lock"
if ! flock -n 9; then
    echo "=== [$(date '+%F %T')] Đợt này đang chạy rồi. Dừng."
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
echo "=== [$(date '+%F %T')] đợt đêm 13/11 bắt đầu | GPU $GPU_INDEX | PID $$"
echo "=== $root"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
echo "=== Đọc số ở DÒNG 'lát strict'. Mốc: Morgan 0,5478 | MoLFormer 0,5602 |"
echo "===   antisymmetric 0,5477 | antisymmetric+MoLFormer 0,5586."

failed=0
total=0

run_one() {
    local label="$1"
    shift
    total=$((total + 1))
    echo
    echo "--------------------------------------------------------------------------------"
    echo "=== [$(date '+%F %T')] $total: $label"
    echo "=== python run_evaluator.py $*"
    local started=$SECONDS
    local status=0
    python run_evaluator.py "$@" || status=$?
    # Bắt mã thoát vào biến TRƯỚC khi echo: $(date) trong cùng dòng echo chạy trước và
    # ghi đè $?, nên in $? trực tiếp sẽ luôn ra 0.
    local minutes=$(((SECONDS - started) / 60))
    if [[ $status -eq 0 ]]; then
        echo "=== [$(date '+%F %T')] $total xong sau $minutes phút"
    else
        echo "=== [$(date '+%F %T')] $total LỖI mã $status sau $minutes phút"
        failed=$((failed + 1))
    fi
}

echo
echo "================================================================================"
echo "=== PHẦN 1: direct head PHẲNG, Morgan — đối chứng khớp cho antisymmetric 0,5477"

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | direct head phẳng | Morgan" \
        --mode train --config frozen --head direct $COMMON --set seed="$seed"
done

echo
echo "================================================================================"
echo "=== PHẦN 2: direct head PHẲNG, MoLFormer — đối chứng cho antisymmetric+MoLFormer 0,5586"

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | direct head phẳng | MoLFormer" \
        --mode train --config frozen --head direct $COMMON --set seed="$seed" $MOLFORMER
done

echo
echo "================================================================================"
echo "=== PHẦN 3: fusion=attention + residual, Morgan — mốc so 0,5478"

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | fusion=attention" \
        --mode train --config frozen --head direct $COMMON $RESIDUAL \
        --set model.fusion=attention --set seed="$seed"
done

echo
echo "================================================================================"
echo "=== PHẦN 4: bỏ nhánh edit trên nền MoLFormer — mốc so 0,5602"

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | direct_no_edit | MoLFormer" \
        --mode train --config frozen --head direct_no_edit $COMMON $RESIDUAL \
        --set seed="$seed" $MOLFORMER
done

echo
echo "=== [$(date '+%F %T')] xong tất cả | $total lần chạy | số lần lỗi: $failed"
echo "Đọc kết quả:"
echo "  python scripts/summarize_runs.py --ranking --slice strict | tee runs/summary_strict_1411.log"
echo "  python scripts/plot_data_scaling.py"
echo "XONG_NIGHT_1311"
