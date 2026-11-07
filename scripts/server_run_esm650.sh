#!/usr/bin/env bash
# ESM2 650M thay cho 35M, VẪN ĐÓNG BĂNG. Nhắm vào nghi phạm cuối: biểu diễn protein.
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_esm650.sh 7 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/esm650.log
#
# Tham số đầu là SỐ GPU (bắt buộc). Tham số sau đó chuyển thẳng cho bước train.
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/esm650.pid)
#
# VÌ SAO LẦN CHẠY NÀY, VÀ VÌ SAO KHÔNG PHẢI FINETUNE. Đến 08/11/2026 đã loại trừ được
# năm nguyên nhân, mỗi cái có số liệu trong repo: kiến trúc head/fusion (10 biến thể
# trong dải 0,0215), biểu diễn phân tử (MoLFormer 0,5600 < Morgan 0,5657), ràng buộc đối
# xứng (twin 0,5624), số lượng dòng train (bỏ 25 % chỉ mất 0,0023), và động lực học nhánh
# edit (lọc dữ liệu lẫn hạ learning rate đều âm). Nghi phạm còn lại là biểu diễn protein:
# phân tích ca sai trên train cho MAE theo protein_rows đi từ 0,3045 (protein ít dữ liệu)
# xuống 0,1582 (protein nhiều dữ liệu), chênh lớn nhất trong mọi nhóm đã phân tầng.
#
# Cách hiển nhiên là mở băng ESM2 (`--config finetune`), nhưng ĐO ĐƯỢC ngày 08/11/2026
# trên data/dataset_v4/full/proteins.parquet (10.573 protein, 1.745.644 dòng) thì
# finetune ép long_sequence='truncate' với max_residues=512 (train.py:1066 và :1302),
# trong khi frozen dùng 'windows' ở 1022 nên không mất gì:
#     protein dài hơn 256  : 93,2 % số dòng
#     protein dài hơn 512  : 47,6 % số dòng
#     protein dài hơn 1022 : 18,7 % số dòng
# Tức finetune cắt mất chuỗi của gần một nửa số dòng, nên kết quả âm sẽ KHÔNG phân biệt
# được "mở băng không giúp" với "bỏ mất phần đuôi protein". Lần chạy này giữ nguyên
# 'windows' ở 1022 và chỉ đổi đúng một biến: ESM2 35M -> 650M.
#
# Lần chạy ra số gì thì cũng đọc được:
#   - tốt hơn 0,5657 rõ rệt  -> biểu diễn protein đúng là trần, hướng đi cho tuần sau.
#   - không khác            -> loại trừ luôn nghi phạm thứ sáu; trần nằm ở dữ liệu/nhãn,
#                              dẫn thẳng sang việc 9 (split 30 %/80 %).
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="esm650"
CACHE_DIR="work/evaluator_cache_v4_esm650"

if [[ $# -lt 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Tham số đầu phải là số GPU. Ví dụ: bash $0 7" >&2
    exit 1
fi
GPU_INDEX="$1"
shift
EXTRA=("$@")

# Bắt buộc đổi cả paths.cache_dir: cache embedding chỉ có MỘT file protein_embeddings
# .parquet, nên dùng chung thư mục sẽ đè mất cache 35M đang dùng cho mốc 0,5657.
# batch_sequences hạ 8 -> 4: 650M nặng hơn 35M khoảng 19 lần về số tham số.
PROTEIN_SET="--set protein_encoder.model=facebook/esm2_t33_650M_UR50D"
PROTEIN_SET+=" --set protein_encoder.embedding_dim=1280"
PROTEIN_SET+=" --set protein_encoder.batch_sequences=4"
PROTEIN_SET+=" --set paths.cache_dir=$CACHE_DIR"

# Đầu residual đã quét được ngày 06/11; giữ nguyên để head không phải biến số.
BEST_SET="--set gpu_memory_limit_mib=12000 --set eval_batch_size=2048"
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
echo "=== [$(date '+%F %T')] bắt đầu ESM2 650M | GPU $GPU_INDEX | PID $$ | $root"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
if [[ ${#EXTRA[@]} -gt 0 ]]; then
    echo "=== tham số thêm cho bước train: ${EXTRA[*]}"
fi

# --- Bước 1: dựng cache trong thư mục RIÊNG ----------------------------------------------
# run_prepare dựng lại cả fingerprint (không có cờ bỏ qua), nên bước này lâu hơn chỉ
# embed protein. Đổi lại: thư mục cache độc lập hoàn toàn, cache 35M không bị chạm.
echo
echo "=== [$(date '+%F %T')] bước 1/2: dựng cache vào $CACHE_DIR"
started=$SECONDS
# shellcheck disable=SC2086
python run_evaluator.py --mode prepare $PROTEIN_SET --set gpu_memory_limit_mib=12000
status=$?
echo "=== [$(date '+%F %T')] bước 1/2 mã $status sau $(((SECONDS - started) / 60)) phút"
if [[ $status -ne 0 ]]; then
    echo "Dừng: không có cache thì không train được."
    echo "Nếu là hết VRAM: thêm --set protein_encoder.batch_sequences=2, hoặc GPU trống hơn."
    echo "XONG_ESM650"
    exit 1
fi

# --- Bước 2: train, chỉ khác 35M ở đúng một biến ------------------------------------------
echo
echo "=== [$(date '+%F %T')] bước 2/2: train frozen direct với embedding 650M"
echo "=== Có số đo validation mỗi 2000 bước, không cần đợi chạy xong."
started=$SECONDS
# shellcheck disable=SC2086
python run_evaluator.py --mode train --config frozen --head direct \
        $PROTEIN_SET $BEST_SET "${EXTRA[@]+"${EXTRA[@]}"}"
status=$?
minutes=$(((SECONDS - started) / 60))
if [[ $status -eq 0 ]]; then
    echo "=== [$(date '+%F %T')] bước 2/2 xong sau $minutes phút"
else
    echo "=== [$(date '+%F %T')] bước 2/2 LỖI mã $status sau $minutes phút"
fi
echo "So với mốc: ESM2 35M cùng cấu hình được 0,5657 +/- 0,0014 (3 seed)."
echo "Đọc kết quả: python scripts/summarize_runs.py --ranking"
echo "XONG_ESM650"
