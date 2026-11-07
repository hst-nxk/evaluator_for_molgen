#!/usr/bin/env bash
# Mở băng ESM2: can thiệp duy nhất nhắm vào nghi phạm duy nhất còn lại.
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_finetune.sh 3 > /dev/null 2>&1 &
#   tail -f runs/finetune.log
#
# Tham số đầu là SỐ GPU (bắt buộc, không tự dò). Mọi tham số sau đó được chuyển thẳng
# cho run_evaluator.py, ví dụ: bash scripts/server_run_finetune.sh 0 --set batch_size=128
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/finetune.pid)
#
# VÌ SAO LẦN CHẠY NÀY. Đến 07/11/2026 đã loại trừ được năm nguyên nhân, mỗi cái có số
# liệu trong repo: kiến trúc head/fusion (10 biến thể trong dải 0,0215), biểu diễn phân
# tử (MoLFormer 0,5600 < Morgan 0,5657), ràng buộc đối xứng (twin 0,5624), số lượng dòng
# train (bỏ 25 % chỉ mất 0,0023), và động lực học nhánh edit (lọc dữ liệu lẫn hạ learning
# rate đều âm). Nghi phạm còn lại là biểu diễn protein, và phân tích ca sai trên train
# (runs/evaluator/errors_frozen_direct_train_20261107_081720) chỉ thẳng vào đó: MAE theo
# protein_rows đi từ 0,3045 (protein ít dữ liệu) xuống 0,1582 (protein nhiều dữ liệu),
# chênh lệch lớn nhất trong mọi nhóm đã phân tầng.
#
# Cấu hình `finetune` đã có code đầy đủ và đã smoke ngày 01/11/2026
# (runs/evaluator/smoke_finetune_direct_20261101_*.json) nhưng CHƯA BAO GIỜ chạy thật:
# mọi lần chạy trong bảng xếp hạng đều là `frozen`.
#
# KHÔNG PHẢI ABLATION MỘT BIẾN. stage3.train.finetune_overrides buộc phải đổi batch_size
# 512 -> 256, eval_batch_size 2048 -> 512, và cắt chuỗi ở max_residues = 512, vì ESM2 nằm
# trong đồ thị tính toán nên bộ nhớ tỉ lệ với số protein PHÂN BIỆT trong batch. Vậy so
# sánh với 0,5657 của `frozen` là "mỗi bên ở thiết lập tốt nhất của nó", không phải đổi
# đúng một biến. Phải nói rõ điều này khi báo cáo.
#
# TRẦN VRAM HẠ XUỐNG 7000 MiB (08/11/2026). Tám GPU của máy này đang bị người khác dùng
# gần hết: chỗ trống lớn nhất đo được là 8.468 MiB, phần còn lại 1-6 GB. Vì vậy
# max_proteins_per_batch hạ từ 6 xuống 3: finetune_overrides_note đo được ~1,25 GB mỗi
# protein phân biệt, nên trần 3 protein cần ~3,8 GB thay vì ~7,5 GB. Theo chính ghi chú
# đó thì 98 % batch chỉ có đúng một protein, nên mức trần này chỉ chạm vào phần đuôi 1-2 %
# số batch. Khi GPU rảnh hơn thì nâng lại bằng:
#   bash scripts/server_run_finetune.sh <gpu> \
#        --set finetune_overrides.batch_size=256 \
#        --set finetune_overrides.max_proteins_per_batch=6 \
#        --set gpu_memory_limit_mib=20000
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="finetune"

if [[ $# -lt 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Tham số đầu phải là số GPU. Ví dụ: bash $0 0" >&2
    exit 1
fi
GPU_INDEX="$1"
shift
EXTRA=("$@")

# Đầu residual đã quét được ngày 06/11; giữ nguyên để phần head không phải biến số.
# Trần VRAM và max_proteins_per_batch hạ xuống cho vừa GPU đang bị chiếm (xem đầu file).
BEST_SET="--set gpu_memory_limit_mib=7000"
BEST_SET+=" --set finetune_overrides.max_proteins_per_batch=3"
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

cache="$root/.cache"
mkdir -p runs "$cache"
log="$root/runs/$LOG_NAME.log"
exec >>"$log" 2>&1

# Không cho chạy hai lần cùng lúc: lần này chiếm tới ~20 GB VRAM.
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
echo "=== [$(date '+%F %T')] bắt đầu finetune | GPU $GPU_INDEX | PID $$ | $root"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
echo "=== Có số đo validation mỗi 2000 bước, không cần đợi chạy xong."
if [[ ${#EXTRA[@]} -gt 0 ]]; then
    echo "=== tham số thêm: ${EXTRA[*]}"
fi

started=$SECONDS
# shellcheck disable=SC2086
python run_evaluator.py --mode train --config finetune --head direct \
        $BEST_SET "${EXTRA[@]+"${EXTRA[@]}"}"
status=$?
minutes=$(((SECONDS - started) / 60))
if [[ $status -eq 0 ]]; then
    echo "=== [$(date '+%F %T')] xong sau $minutes phút"
else
    echo "=== [$(date '+%F %T')] LỖI mã $status sau $minutes phút"
    echo "Nếu là hết VRAM: thêm --set finetune_overrides.batch_size=128, hoặc"
    echo "--set finetune_overrides.max_proteins_per_batch=2, hoặc chọn GPU trống hơn."
fi
echo "Đọc kết quả: python scripts/summarize_runs.py --ranking"
echo "XONG_FINETUNE"
