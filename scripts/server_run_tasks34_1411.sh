#!/usr/bin/env bash
# Đợt 14/11/2026: VIỆC 3 và VIỆC 4 của họp 09/11 — 12 lần chạy, tất cả trên TẬP CON
# 100.000 dòng, Morgan, chọn checkpoint theo lát `strict`.
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_tasks34_1411.sh 5 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/tasks34_1411.log
#
# Tham số duy nhất là CHỈ SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/tasks34_1411.pid)
#
# ------------------------------------------------------------------------------------
# HAI VIỆC ĐANG TRẢ LỜI
#
# VIỆC 3 (phần 2 và 3): "thêm đặc trưng hiệu A − B vào input, theo PBCNet 2.0: lấy hai
#   vector trừ nhau rồi nhân tương tác". Nút mới: model.pair_features =
#   difference | difference_product. Hiệu và tích lấy trên vector ĐÃ MÃ HÓA.
#   Phải nói thẳng khi báo cáo: MLP xấp xỉ được hiệu của hai vector, nên cái thêm vào
#   KHÔNG phải thông tin mới mà là một ưu tiên quy nạp — đặt sẵn phép toán bài toán cần.
#   Vì vậy kết quả bằng 0 là một khả năng thật, và bằng 0 cũng là câu trả lời.
#   CHƯA ĐỐI CHIẾU với bài PBCNet 2.0 gốc: việc 6 (đọc lại paper) chưa làm. Code cài
#   đúng mô tả trong biên bản, không khẳng định trùng thiết kế của bài.
#
# VIỆC 4 (phần 4): "ép đối xứng bằng hàm loss, phải so trực tiếp với việc 2 để chứng
#   minh có khác augmentation hay không". Nút mới: antisymmetry_penalty = w, mất mát cộng
#   thêm w * trung bình((f(A,B) + f(B,A))^2), lượt ngược chạy mô hình trên đầu vào ĐÃ ĐẢO.
#   Ba đường cùng một mục tiêu, khác chỗ đặt ràng buộc:
#       việc 2 — ở DỮ LIỆU   (augment đảo chiều)   -> đã đo: 0,5500 +/- 0,0055
#       việc 4 — ở MẤT MÁT   (đợt này)
#       việc 5 — ở CẤU TẠO   (head antisymmetric)  -> đã đo: 0,5533 +/- 0,0018
#   Cả ba đo trên CÙNG tập con và CÙNG ba hạt giống, nên lần này so được thẳng hàng.
#
# ------------------------------------------------------------------------------------
# MỐC SO (tập con 100.000 dòng, lát `strict`, chọn checkpoint theo `strict`, 3 hạt giống;
# nguồn: bảng 9 thí nghiệm mục 12/11/2026 trong nhật ký):
#   direct + residual 7x102, Morgan      0,5499 +/- 0,0007   <- mốc chính
#   + augment đảo chiều (việc 2)         0,5500 +/- 0,0055
#   antisymmetric (việc 5)               0,5533 +/- 0,0018
#   Linear không bias (việc 11)          0,5437 +/- 0,0001
# Mốc nhiễu hạt giống của dự án: 0,0021. Ngưỡng chứng minh: 3/3 hạt cùng dấu VÀ t >= 3.
#
# VÌ SAO PHẦN 1 TỒN TẠI. Đợt này thêm hai khóa mới vào config.json (model.pair_features,
# antisymmetry_penalty). Khóa mới làm khối `model` trong manifest khác đi, nên
# scripts/group_runs.py xếp các lần chạy mới vào nhóm RIÊNG so với các lần chạy cũ —
# không gộp chung được với mốc 0,5499 ở trên. Phần 1 chạy lại đúng cấu hình mốc bằng code
# mới để hiệu của phần 2, 3, 4 đọc được trong cùng một đợt, không phải so chéo hai đợt.
# Nếu phần 1 KHÔNG ra 0,5499 +/- 0,0007 thì dừng lại: nghĩa là code mới đã đổi hành vi mặc
# định, và mọi số của đợt này không đọc được. (Test đã ghim số tham số không đổi:
# 3.478.022 cho pair_features=none, nhưng số đo mới là bằng chứng cuối cùng.)
#
# ĐỌC LOG: mỗi mốc đo in hai dòng ranking; dòng "lát strict" là dòng dùng để chọn và là
# số của đợt này. Phần 4 in thêm mỗi epoch một dòng "phạt phản đối xứng trung bình" —
# đó là số ĐO LÚC TRAIN (còn dropout) nên KHÔNG dùng để báo cáo tính phản đối xứng; số
# báo cáo là dòng bất đối xứng ở cuối mỗi lần chạy, đo ở chế độ eval.
# KHÔNG dùng partition test ở bất kỳ đâu (run_evaluator.py chặn sẵn).
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="tasks34_1411"
SEEDS=(20261023 20261024 20261025)
SUBSET_FILE="work/splits/train_subset_c200_r500.parquet"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: chỉ số GPU. Ví dụ: bash $0 5" >&2
    exit 1
fi
GPU_INDEX="$1"

COMMON="--set gpu_memory_limit_mib=4500 --set eval_batch_size=2048"
COMMON+=" --set selection_metric=pairwise_accuracy_macro --set selection_slice=strict"
COMMON+=" --set model.dropout=0.5 --set learning_rate=9.86e-05 --set weight_decay=0.0"
COMMON+=" --set inputs.train_subset=$SUBSET_FILE"

RESIDUAL="--set model.head_block=residual --set model.residual_blocks=7"
RESIDUAL+=" --set model.residual_width=102 --set model.residual_hidden_factor=1.849"
RESIDUAL+=" --set model.residual_hidden_dropout=0.3936 --set model.residual_dropout=0.0"

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

if [[ ! -f "$root/$SUBSET_FILE" ]]; then
    echo "Thiếu $SUBSET_FILE. Dựng bằng:" >&2
    echo "  python scripts/build_train_subset.py build --clusters 200 --rows-per-cluster 500" >&2
    exit 1
fi
for name in protein_embeddings molecule_embeddings; do
    if [[ ! -f "$root/work/evaluator_cache_v4/$name.parquet" ]]; then
        echo "Thiếu work/evaluator_cache_v4/$name.parquet. Dừng." >&2
        exit 1
    fi
done

# Hai khóa mới phải CÓ trong config.json, nếu không `--set` sẽ báo lỗi ở lần chạy đầu của
# phần 2 — tức sau khi đã đốt xong phần 1. Kiểm trước, mất một giây.
python - <<'PY' || exit 1
import json
import sys

config = json.load(open("config.json", encoding="utf-8"))
model = config["stage3"]["model"]
train = config["stage3"]["train"]
problems = []
if "pair_features" not in model:
    problems.append("thiếu stage3.model.pair_features")
elif model["pair_features"] != "none":
    problems.append(f"stage3.model.pair_features = {model['pair_features']!r}, phải là 'none'")
if "antisymmetry_penalty" not in train:
    problems.append("thiếu stage3.train.antisymmetry_penalty")
elif float(train["antisymmetry_penalty"]) != 0.0:
    problems.append(f"antisymmetry_penalty = {train['antisymmetry_penalty']}, phải là 0")
if problems:
    print("config.json chưa sẵn sàng: " + "; ".join(problems), file=sys.stderr)
    print("Kéo code mới về (git pull) rồi chạy lại.", file=sys.stderr)
    raise SystemExit(1)
PY

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
echo "=== [$(date '+%F %T')] đợt việc 3 + việc 4 bắt đầu | GPU $GPU_INDEX | PID $$"
echo "=== $root"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
echo "=== Tập con 100.000 dòng, Morgan, lát strict. Mốc chính: 0,5499 +/- 0,0007."
echo "=== Phần 1 PHẢI ra lại mốc đó; nếu không thì dừng đọc các phần sau."

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
echo "=== PHẦN 1: ĐỐI CHỨNG KHỚP bằng code mới — phải ra lại 0,5499 +/- 0,0007"

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | đối chứng (pair_features=none, phạt=0)" \
        --mode train --config frozen --head direct $COMMON $RESIDUAL --set seed="$seed"
done

echo
echo "================================================================================"
echo "=== PHẦN 2: VIỆC 3a — thêm token hiệu enc(A) − enc(B)"

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | pair_features=difference" \
        --mode train --config frozen --head direct $COMMON $RESIDUAL \
        --set model.pair_features=difference --set seed="$seed"
done

echo
echo "================================================================================"
echo "=== PHẦN 3: VIỆC 3b — thêm cả hiệu lẫn tích enc(A) * enc(B)"

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | pair_features=difference_product" \
        --mode train --config frozen --head direct $COMMON $RESIDUAL \
        --set model.pair_features=difference_product --set seed="$seed"
done

echo
echo "================================================================================"
echo "=== PHẦN 4: VIỆC 4 — ép phản đối xứng bằng HÀM MẤT MÁT (w = 1,0)"
echo "=== So thẳng với việc 2 (0,5500 +/- 0,0055) và việc 5 (0,5533 +/- 0,0018)."
echo "=== Mỗi batch hai lượt forward nên phần này chậm khoảng gấp đôi ba phần trên."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | antisymmetry_penalty=1.0" \
        --mode train --config frozen --head direct $COMMON $RESIDUAL \
        --set antisymmetry_penalty=1.0 --set seed="$seed"
done

echo
echo "=== [$(date '+%F %T')] xong tất cả | $total lần chạy | số lần lỗi: $failed"
echo "Đọc kết quả:"
echo "  python scripts/group_runs.py --slice strict --min-seeds 3 | tee runs/groups_tasks34.log"
echo "  python scripts/summarize_runs.py --ranking --slice strict | tee runs/summary_tasks34.log"
echo "Đọc mức vi phạm phản đối xứng của phần 4:"
echo "  grep -n 'phạt phản đối xứng\\|bất đối xứng' runs/$LOG_NAME.log"
echo "XONG_TASKS34_1411"
