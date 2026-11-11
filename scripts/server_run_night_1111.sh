#!/usr/bin/env bash
# Đợt chạy đêm 11/11/2026: thêm hạt giống cho 9 thí nghiệm của họp 09/11, rồi hai lần
# train ĐẦY ĐỦ cho hai kiến trúc phản đối xứng.
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_night_1111.sh 3 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/night_1111.log
#
# Tham số duy nhất là CHỈ SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/night_1111.pid)
#
# TỰ ĐỘNG CHỜ script ban ngày. Script này giữ cùng một file lock với
# `server_run_meeting_0911.sh`, nhưng ở chế độ CHỜ chứ không bỏ cuộc, nên bật nó ngay bây
# giờ là được: nó ngủ cho tới khi đợt ban ngày nhả lock rồi mới bắt đầu. Nhờ vậy không cần
# thức để bấm lệnh thứ hai, và hai đợt không bao giờ tranh cùng một GPU.
#
# VÌ SAO THỨ TỰ NÀY, giá trị giảm dần. Nếu sáng ra vẫn còn đang chạy phần cuối thì những
# phần trước đã có kết quả đầy đủ trong log và trong manifest:
#
#   Phần 1-2: 9 thí nghiệm × 2 hạt giống nữa (20261024, 20261025).
#       BẮT BUỘC, không phải cho đủ giờ. Subset chỉ 100.000 dòng, ít hơn toàn bộ train 66
#       lần, nên nhiễu hạt giống ở đây gần như chắc chắn lớn hơn ±0,0014 đo được trên toàn
#       bộ train. Với MỘT hạt giống, chênh 0,005 giữa hai thí nghiệm không phân biệt được
#       với nhiễu, tức cả bảng 9 thí nghiệm không kết luận được gì. Đây đúng là cái bẫy đã
#       làm chọn sai dropout 0,6 vì MAE hồi 05/11/2026.
#       Chi phí: 8 phút × 18 = ~2 giờ 25 phút.
#
#   Phần 3-4: train ĐẦY ĐỦ (6,6 triệu dòng) cho `antisymmetric` và `twin` + `twin_edit`.
#       Cần bất kể subset nói gì, vì báo cáo phải trả lời được "đã ép phản đối xứng chưa,
#       và nó mất bao nhiêu trên dữ liệu thật". Hai kiến trúc này là Task 4 của họp.
#       `antisymmetric` đi trước vì nó là cái duy nhất vừa phản đối xứng đúng theo cấu tạo
#       vừa còn học được tương tác trong cặp (twin chấm từng phân tử rời nhau nên không).
#       Chi phí: ~2,5-3 giờ mỗi lần.
#
# MỐC ĐỂ SO. Thí nghiệm 0 của đợt ban ngày, cùng subset, hạt giống 20261023: đúng chiều
# macro 0,5484 (epoch 3). Toàn bộ train, cùng head direct: 0,5657 ± 0,0014 — KHÔNG so trực
# tiếp với số subset được.
#
# KHÔNG dùng partition test ở bất kỳ đâu trong script này.
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="night_1111"
SHARED_LOCK="meeting_0911"
SUBSET="work/splits/train_subset_c200_r500.parquet"
EXTRA_SEEDS=(20261024 20261025)

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: chỉ số GPU. Ví dụ: bash $0 3" >&2
    exit 1
fi
GPU_INDEX="$1"

# Cấu hình nền, giống hệt đợt ban ngày để hai đợt gộp được vào một bảng.
COMMON="--set gpu_memory_limit_mib=7500 --set eval_batch_size=2048"
COMMON+=" --set selection_metric=pairwise_accuracy_macro"
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

# Lock riêng: chặn chạy hai lần đợt đêm.
exec 9>"$cache/$LOG_NAME.lock"
if ! flock -n 9; then
    echo "=== [$(date '+%F %T')] Đợt đêm đang chạy rồi. Dừng."
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
echo "=== [$(date '+%F %T')] đợt đêm bắt đầu | GPU $GPU_INDEX | PID $$ | $root"

# Lock dùng chung với đợt ban ngày, chế độ CHỜ. Nếu đợt kia đã xong thì lấy được ngay.
echo "=== [$(date '+%F %T')] chờ đợt ban ngày ($SHARED_LOCK) nhả GPU, tối đa 2 giờ..."
exec 8>"$cache/$SHARED_LOCK.lock"
# Có giới hạn chờ: nếu đợt ban ngày treo thì cả đêm sẽ mất trắng. Chạy chồng cũng không
# sao về bộ nhớ — đợt ban ngày đo được VRAM cao nhất PyTorch giữ chỉ 250 MiB trên GPU
# 80 GB — nên thà chạy chồng còn hơn không chạy gì.
if flock -w 7200 8; then
    echo "=== [$(date '+%F %T')] đã tới lượt, bắt đầu chạy."
else
    echo "=== [$(date '+%F %T')] CHỜ 2 GIỜ KHÔNG ĐƯỢC: đợt ban ngày có thể đang treo."
    echo "=== Chạy tiếp dù sao, chấp nhận dùng chung GPU. Kiểm lại log đợt ban ngày."
fi
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
echo "=== MỐC ĐỂ SO (subset, hạt giống 20261023): đúng chiều macro 0,5484"
echo "=== Toàn bộ train, head direct: 0,5657 +/- 0,0014 — KHÔNG so trực tiếp với subset."

failed=0
total=0

# Chạy một lần train, in nhãn và thời gian. Không dừng cả đợt khi một lần lỗi: một kiến
# trúc chết không được phép làm mất các lần còn lại.
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
    # Phải bắt mã thoát vào biến TRƯỚC khi gọi echo: $(date) trong cùng dòng echo chạy
    # trước và ghi đè $?, nên in $? trực tiếp sẽ luôn ra 0.
    local minutes=$(((SECONDS - started) / 60))
    if [[ $status -eq 0 ]]; then
        echo "=== [$(date '+%F %T')] $total xong sau $minutes phút"
    else
        echo "=== [$(date '+%F %T')] $total LỖI mã $status sau $minutes phút"
        failed=$((failed + 1))
    fi
}

# Chín thí nghiệm của họp 09/11, trên subset, với một hạt giống cho trước. Giữ ĐÚNG thứ tự
# và đúng tham số của đợt ban ngày; chỉ thêm --set seed.
run_nine_on_subset() {
    local seed="$1"
    local base="$COMMON $RESIDUAL --set inputs.train_subset=$SUBSET --set seed=$seed"
    local plain="$COMMON --set inputs.train_subset=$SUBSET --set seed=$seed"

    run_one "hạt giống $seed | 0 mốc: direct, không augment" \
        --mode train --config frozen --head direct $base
    run_one "hạt giống $seed | 1 augment đảo chiều" \
        --mode train --config frozen --head direct $base --set augment_reverse=true
    run_one "hạt giống $seed | 2 đổi loss sang MSE" \
        --mode train --config frozen --head direct $base --set loss=mse
    run_one "hạt giống $seed | 3 baseline Linear không bias" \
        --mode train --config frozen --head direct $plain --set model.head_block=linear
    run_one "hạt giống $seed | 4 baseline Ridge (Linear + wd 0,1)" \
        --mode train --config frozen --head direct $plain \
        --set model.head_block=linear --set weight_decay=0.1
    run_one "hạt giống $seed | 5 twin có nhánh edit" \
        --mode train --config frozen --head twin $base --set model.twin_edit=true
    run_one "hạt giống $seed | 6 antisymmetric: hàm lẻ x cổng ngữ cảnh" \
        --mode train --config frozen --head antisymmetric $plain
    run_one "hạt giống $seed | 7 antisymmetric tuyến tính" \
        --mode train --config frozen --head antisymmetric $plain \
        "--set" "model.antisymmetric_hidden=[]"
    run_one "hạt giống $seed | 8 mốc nhưng TẮT expand_both_directions" \
        --mode train --config frozen --head direct $base \
        --set metrics.expand_both_directions=false
}

echo
echo "================================================================================"
echo "=== PHẦN 1-2: thêm hai hạt giống cho 9 thí nghiệm trên subset (~2,5 giờ)"
echo "=== Không có phần này thì bảng 9 thí nghiệm không phân biệt được với nhiễu."
for seed in "${EXTRA_SEEDS[@]}"; do
    run_nine_on_subset "$seed"
done

echo
echo "================================================================================"
echo "=== PHẦN 3-4: train ĐẦY ĐỦ 6,6 triệu dòng cho hai kiến trúc phản đối xứng"
echo "=== Mỗi lần ~2,5-3 giờ. Đây là con số đi vào báo cáo, so trực tiếp được với 0,5657."
echo "=== Kiểm trong manifest: asymmetry.antisymmetry_max phải <= 1e-8 và meets_target"
echo "=== phải là true. Nếu KHÔNG đạt thì là lỗi kiến trúc, không phải chưa hội tụ."

run_one "ĐẦY ĐỦ | antisymmetric (hàm lẻ + cổng ngữ cảnh)" \
    --mode train --config frozen --head antisymmetric $COMMON --set seed=20261023
run_one "ĐẦY ĐỦ | twin + nhánh edit" \
    --mode train --config frozen --head twin $COMMON $RESIDUAL --set seed=20261023 \
    --set model.twin_edit=true

echo
echo "=== [$(date '+%F %T')] xong tất cả | $total lần chạy | số lần lỗi: $failed"
echo "Đọc kết quả cả hai đợt: python scripts/summarize_runs.py --ranking"
echo "XONG_NIGHT_1111"
