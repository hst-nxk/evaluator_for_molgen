#!/usr/bin/env bash
# Chạy lại hai mốc với checkpoint được chọn theo lát `strict` (12/11/2026).
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_strict_1211.sh 3 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/strict_1211.log
#
# Tham số duy nhất là CHỈ SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/strict_1211.pid)
#
# VÌ SAO ĐỢT NÀY. Tới 12/11/2026, MỌI checkpoint trong runs/evaluator đều được chọn và
# dừng sớm theo chỉ số tính trên TOÀN BỘ validation (lát 'all'), trong đó 14,05 % dòng có
# cặp (A, B) đã xuất hiện trong train. Trên 66.628 dòng trùng cặp đó, dự đoán tương quan
# 0,8045 với nhãn TRAIN của chính cặp ấy (MAE 0,2896) nhưng chỉ 0,4477 với nhãn THẬT của
# validation (MAE 0,4615) — tức mô hình tra lại nhãn đã học. Chọn checkpoint theo lát
# 'all' là chọn đúng checkpoint học thuộc giỏi nhất.
#
# Và hai lát không chỉ lệch nhau một hằng số, chúng ĐẢO thứ tự xếp hạng (đo 12/11, 3 hạt
# giống mỗi bên, runs/summary_1111.log + runs/summary_1111_strict.log):
#
#                        lát all            lát strict
#   Morgan               0,5657 +/- 0,0014  0,5446 +/- 0,0021
#   MoLFormer            0,5610 +/- 0,0009  0,5597 +/- 0,0015
#   hiệu                 -0,0047            +0,0151
#
# Nhưng cả hai cột trên đều là checkpoint CHỌN theo lát 'all'. Đợt này trả lời đúng một
# câu: khi chọn checkpoint theo lát 'strict' thì con số trung thực đi tới đâu, và kết luận
# "MoLFormer tốt hơn Morgan" có giữ không. Nếu giữ, đây là kết quả chính của paper về
# biểu diễn phân tử. Nếu không giữ, phải biết trước khi viết.
#
# XEN KẼ THEO HẠT GIỐNG, cố ý. Chạy hết 3 hạt Morgan rồi mới tới MoLFormer thì nếu đợt bị
# cắt giữa đường sẽ có 3 Morgan và 0 MoLFormer — không so được gì. Xen kẽ thì sau mỗi cặp
# đã có một phép so hoàn chỉnh, và sau hai cặp đã đủ nói hướng.
#
# MỐC ĐỂ SO (đều là checkpoint chọn theo lát 'all'):
#   Morgan      lát all 0,5657 +/- 0,0014 | lát strict 0,5446 +/- 0,0021
#   MoLFormer   lát all 0,5610 +/- 0,0009 | lát strict 0,5597 +/- 0,0015
# Số của đợt này phải đọc Ở CỘT strict và so với 0,5446 / 0,5597.
#
# ĐỌC LOG. Mỗi mốc đo in HAI dòng ranking: dòng "validation" là lát all (giữ nguyên để so
# với lần chạy cũ) và dòng "lát strict" là lát dùng để chọn. Dừng sớm đi theo dòng thứ hai.
#
# KHÔNG dùng partition test ở bất kỳ đâu trong script này.
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="strict_1211"
SEEDS=(20261023 20261024 20261025)

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: chỉ số GPU. Ví dụ: bash $0 3" >&2
    exit 1
fi
GPU_INDEX="$1"

# Cấu hình nền: giống hệt các đợt trước để số gộp được vào cùng một bảng. Thứ DUY NHẤT
# thêm vào là selection_slice=strict.
COMMON="--set gpu_memory_limit_mib=7500 --set eval_batch_size=2048"
COMMON+=" --set selection_metric=pairwise_accuracy_macro"
COMMON+=" --set selection_slice=strict"
COMMON+=" --set model.dropout=0.5"
COMMON+=" --set learning_rate=9.86e-05 --set weight_decay=0.0"

RESIDUAL="--set model.head_block=residual"
RESIDUAL+=" --set model.residual_blocks=7 --set model.residual_width=102"
RESIDUAL+=" --set model.residual_hidden_factor=1.849"
RESIDUAL+=" --set model.residual_hidden_dropout=0.3936"
RESIDUAL+=" --set model.residual_dropout=0.0"

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

if [[ ! -f "$root/work/evaluator_cache_v4/protein_embeddings.parquet" ]]; then
    echo "Thiếu cache embedding protein. Dựng trước bằng:" >&2
    echo "  python run_evaluator.py --mode prepare" >&2
    exit 1
fi

# MoLFormer cần cache embedding phân tử, đã dựng 07/11/2026. Kiểm trước để không chết ở
# giữa đợt sau khi đã tốn một lần train Morgan.
if [[ ! -f "$root/work/evaluator_cache_v4/molecule_embeddings.parquet" ]]; then
    echo "Thiếu cache embedding phân tử (MoLFormer). Dựng trước bằng:" >&2
    echo "  python run_evaluator.py --mode prepare --molecule-embeddings" >&2
    exit 1
fi

# Lát 'strict' phải có thật trong config, không thì mọi lần train sẽ chết ngay.
if ! python -c "
import json, sys
slices = json.load(open('config.json', encoding='utf-8'))['stage3']['metrics']['slices']
sys.exit(0 if any(s['name'] == 'strict' for s in slices) else 1)
"; then
    echo "config.json không khai báo lát 'strict' trong stage3.metrics.slices. Dừng." >&2
    exit 1
fi

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
echo "=== [$(date '+%F %T')] đợt chọn-theo-lát-strict bắt đầu | GPU $GPU_INDEX | PID $$"
echo "=== $root"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
echo "=== MỐC (checkpoint chọn theo lát all): Morgan strict 0,5446 +/- 0,0021 |"
echo "===   MoLFormer strict 0,5597 +/- 0,0015. Đọc số đợt này ở CỘT strict."

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
echo "=== PHẦN 1: hai mốc, 3 hạt giống, xen kẽ theo hạt giống (~4 giờ)"
echo "=== Mỗi cặp Morgan/MoLFormer cùng hạt giống là một phép so hoàn chỉnh."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | Morgan fingerprint (mốc)" \
        --mode train --config frozen --head direct $COMMON $RESIDUAL --set seed="$seed"
    run_one "hạt giống $seed | MoLFormer thay Morgan" \
        --mode train --config frozen --head direct $COMMON $RESIDUAL --set seed="$seed" \
        --set model.ligand_features=embedding
done

echo
echo "================================================================================"
echo "=== PHẦN 2: hai hạt giống nữa cho antisymmetric trên toàn bộ train (~45 phút)"
echo "=== CHỌN THEO LÁT all, cố ý: để gộp được với lần chạy 11/11 (0,5607 lát all /"
echo "=== 0,5428 lát strict) thành một trung bình 3 hạt. Hiện mới có 1 hạt nên chưa"
echo "=== kết luận được antisymmetric hơn hay kém direct."

# Dựng lại tường minh thay vì thay chuỗi trong $COMMON: thay chuỗi mà sai một ký tự thì
# cả hai lần chạy này âm thầm chọn theo lát strict và không gộp được với lần 11/11.
ALL_SLICE="--set gpu_memory_limit_mib=7500 --set eval_batch_size=2048"
ALL_SLICE+=" --set selection_metric=pairwise_accuracy_macro"
ALL_SLICE+=" --set selection_slice=all"
ALL_SLICE+=" --set model.dropout=0.5"
ALL_SLICE+=" --set learning_rate=9.86e-05 --set weight_decay=0.0"

for seed in 20261024 20261025; do
    run_one "hạt giống $seed | antisymmetric ĐẦY ĐỦ (chọn theo lát all)" \
        --mode train --config frozen --head antisymmetric $ALL_SLICE --set seed="$seed"
done

echo
echo "=== [$(date '+%F %T')] xong tất cả | $total lần chạy | số lần lỗi: $failed"
echo "Đọc kết quả:"
echo "  python scripts/summarize_runs.py --ranking --slice strict | tee runs/summary_strict_1211.log"
echo "  python scripts/summarize_runs.py --ranking | tee runs/summary_all_1211.log"
echo "Nhãn các lần chạy đợt này có 'selection_slice=strict' nên phân biệt được với đợt cũ."
echo "XONG_STRICT_1211"
