#!/usr/bin/env bash
# Đợt chạy đêm 12/11/2026: bốn câu hỏi còn mở sau khi MoLFormer thắng trên lát `strict`.
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_night_1211.sh 5 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/night_1211.log
#
# Tham số duy nhất là CHỈ SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/night_1211.pid)
#
# NỀN TẢNG. Sáng 12/11 đo xong: chọn checkpoint theo lát `strict` thì MoLFormer hơn Morgan
# +0,0124 (0,5602 +/- 0,0010 so với 0,5478 +/- 0,0002, 3 hạt giống mỗi bên,
# runs/summary_strict_1211.log). Đó là kết quả chính của tuần. Mọi mốc so sánh trong đợt này
# đều là hai số đó, và MỌI lần chạy ở đây đều chọn checkpoint theo lát `strict`.
#
# BỐN CÂU HỎI, xếp theo giá trị chia cho chi phí — phần rẻ và nhiều thông tin nhất chạy
# TRƯỚC, để nếu đợt bị cắt giữa đường thì phần mất là phần ít quan trọng nhất.
#
# PHẦN 1 (~50 phút): MoLFormer có hơn Morgan vì BIỂU DIỄN TỐT HƠN, hay chỉ vì KHÓ HỌC THUỘC
#   HƠN? Hai thứ đó khác nhau và paper phải nói đúng một trong hai. Trên subset 100.000 dòng
#   không mô hình nào học thuộc được cặp validation (đã chứng minh: ở quy mô subset điểm lát
#   `strict` CAO HƠN lát `all` +0,0017, tức không có phần thưởng học thuộc nào). Nếu MoLFormer
#   vẫn hơn Morgan ở đó thì là biểu diễn tốt hơn thật. Nếu hết hơn thì +0,0124 chỉ là chuyện
#   Morgan học thuộc giỏi hơn, và phải viết lại cách diễn giải.
#   Mốc Morgan subset cũ (0,5499 +/- 0,0007) chọn theo lát `all`, KHÔNG dùng làm mốc ở đây;
#   phần này chạy lại cả hai nhánh theo cùng một quy tắc chọn.
#
# PHẦN 2 (~1,5 giờ): antisymmetric chọn theo lát `strict`. Hiện có 3 hạt chọn theo `all`:
#   `all` 0,5609 +/- 0,0021 so với direct 0,5657 +/- 0,0014 (t ~ 3,3, kém hơn có bằng chứng);
#   `strict` 0,5403 +/- 0,0028 so với 0,5446 +/- 0,0021 (t ~ 2,1, CHƯA đủ chứng minh).
#   Trên subset thì antisymmetric lại HƠN +0,0034 (t ~ 3,0). Phần này đo đúng ô còn trống:
#   đủ dữ liệu VÀ chọn checkpoint trung thực. Mốc so: 0,5478 +/- 0,0002.
#
# PHẦN 3 (~2 giờ): hai món tốt nhất có cộng dồn không — MoLFormer + antisymmetric. Mốc so:
#   MoLFormer một mình 0,5602 +/- 0,0010.
#
# PHẦN 4 (~1 giờ): `label_normalization=scale` (việc 14 họp 09/11, code đã viết từ 11/11 mà
#   CHƯA CHẠY LẦN NÀO). Chạy trên nền MoLFormer vì đó là nền mới, không phải nền Morgan cũ.
#   Mốc so: 0,5602 +/- 0,0010. Dùng 'scale' chứ không phải 'standard' vì train.py chặn mọi
#   phép chuẩn hóa có phép dịch (shift làm nhãn z và −z không còn đối xứng).
#
# XEN KẼ THEO HẠT GIỐNG ở phần 1, cố ý: chạy hết 3 hạt Morgan rồi mới tới MoLFormer thì nếu
# bị cắt giữa đường sẽ có 3 Morgan và 0 MoLFormer, không so được gì. Phần 2, 3, 4 đều so với
# mốc đã đo xong nên không cần xen kẽ.
#
# ĐỌC LOG. Mỗi mốc đo in HAI dòng ranking: dòng "validation" là lát `all` (giữ để so với lần
# chạy cũ) và dòng "lát strict" là lát dùng để chọn. Dừng sớm đi theo dòng thứ hai. Số của
# đợt này phải đọc Ở DÒNG "lát strict".
#
# KHÔNG dùng partition test ở bất kỳ đâu trong script này.
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="night_1211"
SEEDS=(20261023 20261024 20261025)
SUBSET_FILE="work/splits/train_subset_c200_r500.parquet"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: chỉ số GPU. Ví dụ: bash $0 5" >&2
    exit 1
fi
GPU_INDEX="$1"

# Cấu hình nền: giống hệt các đợt trước để số gộp được vào cùng một bảng.
COMMON="--set gpu_memory_limit_mib=7500 --set eval_batch_size=2048"
COMMON+=" --set selection_metric=pairwise_accuracy_macro"
COMMON+=" --set selection_slice=strict"
COMMON+=" --set model.dropout=0.5"
COMMON+=" --set learning_rate=9.86e-05 --set weight_decay=0.0"

# Cấu hình thắng của đợt quét 27 cấu hình. CHỈ dùng cho head `direct`: các lần chạy
# antisymmetric từ trước tới nay đều KHÔNG có mấy cờ này, thêm vào thì số mới không gộp
# được với 3 hạt đã đo.
RESIDUAL="--set model.head_block=residual"
RESIDUAL+=" --set model.residual_blocks=7 --set model.residual_width=102"
RESIDUAL+=" --set model.residual_hidden_factor=1.849"
RESIDUAL+=" --set model.residual_hidden_dropout=0.3936"
RESIDUAL+=" --set model.residual_dropout=0.0"

# Tên khóa đã kiểm 12/11 bằng run_evaluator.apply_overrides: `train_subset` đi vào khối
# `inputs`, KHÔNG phải khóa trần; mọi thứ thuộc kiến trúc đi qua tiền tố `model.`.
SUBSET="--set inputs.train_subset=$SUBSET_FILE"
MOLFORMER="--set model.ligand_features=embedding"

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

# Phần 1 cần đúng file subset mà bảng 9 thí nghiệm đã dùng, không phải một file khác.
if [[ ! -f "$root/$SUBSET_FILE" ]]; then
    echo "Thiếu $SUBSET_FILE (phần 1 cần đúng file subset của bảng cũ). Dừng." >&2
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
echo "=== [$(date '+%F %T')] đợt đêm 12/11 bắt đầu | GPU $GPU_INDEX | PID $$"
echo "=== $root"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
echo "=== MỐC (đều chọn theo lát strict, 3 hạt): Morgan 0,5478 +/- 0,0002 |"
echo "===   MoLFormer 0,5602 +/- 0,0010. Đọc số đợt này ở DÒNG 'lát strict'."

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
echo "=== PHẦN 1: biểu diễn tốt hơn, hay chỉ khó học thuộc hơn? (~50 phút)"
echo "=== subset 100.000 dòng, nơi KHÔNG mô hình nào học thuộc được cặp validation."
echo "=== Hai nhánh cùng một quy tắc chọn, xen kẽ theo hạt giống."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | subset | Morgan fingerprint" \
        --mode train --config frozen --head direct $COMMON $RESIDUAL $SUBSET \
        --set seed="$seed"
    run_one "hạt giống $seed | subset | MoLFormer thay Morgan" \
        --mode train --config frozen --head direct $COMMON $RESIDUAL $SUBSET \
        --set seed="$seed" $MOLFORMER
done

echo
echo "================================================================================"
echo "=== PHẦN 2: antisymmetric chọn theo lát strict, toàn bộ train (~1,5 giờ)"
echo "=== KHÔNG thêm cờ residual: giữ đúng kiến trúc của 3 hạt đã đo để gộp bảng được."
echo "=== Mốc so: direct Morgan 0,5478 +/- 0,0002."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | antisymmetric ĐẦY ĐỦ (chọn theo lát strict)" \
        --mode train --config frozen --head antisymmetric $COMMON --set seed="$seed"
done

echo
echo "================================================================================"
echo "=== PHẦN 3: hai món tốt nhất có cộng dồn không (~2 giờ)"
echo "=== MoLFormer + antisymmetric. Mốc so: MoLFormer một mình 0,5602 +/- 0,0010."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | antisymmetric + MoLFormer" \
        --mode train --config frozen --head antisymmetric $COMMON --set seed="$seed" \
        $MOLFORMER
done

echo
echo "================================================================================"
echo "=== PHẦN 4: label_normalization=scale, chưa chạy lần nào (~1 giờ)"
echo "=== Trên nền MoLFormer. Mốc so: 0,5602 +/- 0,0010."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | MoLFormer | label_normalization=scale" \
        --mode train --config frozen --head direct $COMMON $RESIDUAL \
        --set seed="$seed" $MOLFORMER --set label_normalization=scale
done

echo
echo "=== [$(date '+%F %T')] xong tất cả | $total lần chạy | số lần lỗi: $failed"
echo "Đọc kết quả:"
echo "  python scripts/summarize_runs.py --ranking --slice strict | tee runs/summary_strict_1311.log"
echo "  python scripts/summarize_runs.py --ranking | tee runs/summary_all_1311.log"
echo "Nhãn các lần chạy đợt này có 'selection_slice=strict' nên phân biệt được với đợt cũ."
echo "XONG_NIGHT_1211"
