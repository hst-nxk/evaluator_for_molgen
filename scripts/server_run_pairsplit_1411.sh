#!/usr/bin/env bash
# Đợt 14/11/2026 (đợt 4): VIỆC 13 — CHIA THEO CẶP, KHÔNG CHIA THEO PROTEIN.
# 9 lần chạy, cùng tập con 100.000 dòng, chọn checkpoint theo lát `strict`,
# ba hạt giống 20261023/24/25.
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   python scripts/build_pair_split.py --out work/splits/pair_split_1411.parquet
#   setsid nohup bash scripts/server_run_pairsplit_1411.sh 1 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/pairsplit_1411.log
#
# Tham số duy nhất là CHỈ SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/pairsplit_1411.pid)
#
# ------------------------------------------------------------------------------------
# VIỆC 13 BIÊN BẢN 09/11/2026, nguyên văn:
#   "Thu split theo tieu chi nguoc lai. Cho phep TRUNG protein P, nhung cap (A,B) o
#    validation KHONG duoc trung train - do kha nang tong quat hoa tren cau truc hoa hoc."
#
# Và nó TRÙNG với kịch bản 2 của việc 15:
#   "Kich ban 2 (de hon): protein da co du lieu lich su, du doan cho cap chat (A,B) moi."
# Nên một đợt này trả lời cả hai việc, và là số để quyết định có hạ kịch bản hay không.
#
# ------------------------------------------------------------------------------------
# PHẢI ĐỌC TRƯỚC KHI SO SỐ — GIỚI HẠN CỦA ĐỢT NÀY
#
# Tập validation ở đây KHÁC hẳn tập validation của mọi số đã đo: nó là ~25 % số dòng của
# chính tập con 100k, chia theo cặp phân tử; còn các số cũ đo trên 424.473 dòng của những
# protein chưa từng thấy. Vì vậy:
#
#   * KHÔNG được trừ thẳng 0,55xx của đợt này cho 0,5561 của đợt cũ rồi gọi là "chênh".
#     Hai con số đo trên hai tập khác nhau, bằng hai bài toán khác nhau.
#   * Câu đọc được: (a) so với sàn 0,5000 của đoán ngẫu nhiên, kịch bản 2 cao hơn bao
#     nhiêu; (b) TRONG đợt này, MoLFormer so với Morgan — câu hỏi biểu diễn hỏi lại dưới
#     cách chia dễ hơn; (c) phạt phản đối xứng còn sống dưới cách chia này không.
#   * Nếu kịch bản 2 cũng chỉ quanh 0,56 thì cái trần KHÔNG nằm ở chỗ protein mới, và
#     việc 15 không nên hạ kịch bản — hạ cũng không khá hơn. Đó là kết luận mạnh nhất mà
#     đợt này có thể cho, và nó là kết luận ÂM.
#
# Bảng chia do scripts/build_pair_split.py dựng, và nó in ra ba con số phải nhìn trước
# khi tin đợt này: số cặp trùng hai bên (phải bằng 0), tỉ lệ protein của validation có
# mặt trong train (càng cao càng đúng ý việc 13), và số dòng mỗi bên.
#
# BA NHÓM, xếp theo thứ tự quan trọng giảm dần (kill giữa đợt chỉ mất nhóm cuối):
#   1. MoLFormer, cách chia theo cặp          -> con số chính của việc 13 và việc 15.
#   2. Morgan,    cách chia theo cặp          -> câu hỏi biểu diễn dưới cách chia mới.
#   3. MoLFormer + phạt phản đối xứng w=1,0   -> hiệu đã chứng minh ở cách chia cũ
#                                                (+0,0026, t = 6,41) có sống ở đây không.
#
# Sàn nhiễu hạt giống 0,0021. Ngưỡng chứng minh của dự án: 3/3 hạt cùng dấu VÀ t >= 3.
# KHÔNG dùng partition test ở bất kỳ đâu (run_evaluator.py chặn sẵn).
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="pairsplit_1411"
SEEDS=(20261023 20261024 20261025)
SPLIT_FILE="work/splits/pair_split_1411.parquet"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: chỉ số GPU. Ví dụ: bash $0 1" >&2
    exit 1
fi
GPU_INDEX="$1"

# Chú ý: KHÔNG có --set inputs.train_subset ở đây. Bảng chia theo cặp đã chỉ gồm đúng
# các dòng của tập con 100k, và data.py từ chối bật hai đường chia cùng lúc.
COMMON="--set gpu_memory_limit_mib=4500 --set eval_batch_size=2048"
COMMON+=" --set selection_metric=pairwise_accuracy_macro --set selection_slice=strict"
COMMON+=" --set model.dropout=0.5 --set learning_rate=9.86e-05 --set weight_decay=0.0"
COMMON+=" --set inputs.partition_override_pairs=$SPLIT_FILE"

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

if [[ ! -f "$root/$SPLIT_FILE" ]]; then
    echo "Thiếu $SPLIT_FILE. Dựng trước bằng:" >&2
    echo "  python scripts/build_pair_split.py --out $SPLIT_FILE" >&2
    exit 1
fi
for name in protein_embeddings molecule_embeddings; do
    if [[ ! -f "$root/work/evaluator_cache_v4/$name.parquet" ]]; then
        echo "Thiếu work/evaluator_cache_v4/$name.parquet. Dừng." >&2
        echo "  python run_evaluator.py --mode prepare --molecule-embeddings" >&2
        exit 1
    fi
done

# Kiểm bảng chia TRƯỚC khi đốt 3 tiếng GPU: đúng ba cột, không cặp nào trùng hai bên,
# hai bên đều có dòng, và protein của validation phải CÓ mặt trong train — nếu không thì
# đây không phải việc 13 mà vẫn là cách chia cũ dưới tên mới.
PYTHONIOENCODING=utf-8 python - <<'PY' || exit 1
import json
import sys

import pyarrow.parquet as pq

sys.path.insert(0, "src")

split = pq.read_table("work/splits/pair_split_1411.parquet")
problems = []
missing = {"pair_id", "partition", "unseen_pair"} - set(split.column_names)
if missing:
    problems.append(f"bảng chia thiếu cột {sorted(missing)}")
else:
    partitions = split.column("partition").to_pylist()
    counts = {name: partitions.count(name) for name in set(partitions)}
    if counts.get("train", 0) == 0 or counts.get("validation", 0) == 0:
        problems.append(f"một bên rỗng: {counts}")
    unseen = split.column("unseen_pair").to_pylist()
    wrong = sum(1 for part, flag in zip(partitions, unseen)
                if flag != (part == "validation"))
    if wrong:
        problems.append(f"{wrong} dòng có unseen_pair không khớp partition")
    print(f"bảng chia: {counts}")

config = json.load(open("config.json", encoding="utf-8"))
inputs = config["stage3"]["inputs"]
if "partition_override_pairs" not in inputs:
    problems.append("config.json chưa có khóa stage3.inputs.partition_override_pairs")
for name in ("partition_override", "train_subset"):
    if str(inputs.get(name) or "").strip():
        problems.append(f"{name} đang bật trong config.json; không được bật cùng lúc")
if problems:
    print("chưa sẵn sàng: " + "; ".join(problems), file=sys.stderr)
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
echo "=== [$(date '+%F %T')] đợt việc 13 (chia theo cặp) bắt đầu | GPU $GPU_INDEX | PID $$"
echo "=== $root"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
echo "=== Bảng chia: $SPLIT_FILE"
echo "=== Trùng protein ĐƯỢC PHÉP; trùng cặp (A, B) thì KHÔNG. Đây là kịch bản 2."
echo "=== KHÔNG trừ thẳng số đợt này với số các đợt cũ: khác tập validation."

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
echo "=== NHÓM 1: MoLFormer trên cách chia theo cặp. SỐ CHÍNH của việc 13 + việc 15."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | MoLFormer | chia theo cặp" \
        --mode train --config frozen --head direct \
        $COMMON $RESIDUAL $MOLFORMER --set seed=$seed
done

echo
echo "================================================================================"
echo "=== NHÓM 2: Morgan trên cùng cách chia. Câu hỏi biểu diễn hỏi lại ở kịch bản 2."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | Morgan | chia theo cặp" \
        --mode train --config frozen --head direct \
        $COMMON $RESIDUAL --set seed=$seed
done

echo
echo "================================================================================"
echo "=== NHÓM 3: MoLFormer + phạt phản đối xứng w=1,0. Hiệu đã chứng minh ở cách chia"
echo "=== cũ (+0,0026, t = 6,41) có sống dưới cách chia này không."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | MoLFormer + phạt 1,0 | chia theo cặp" \
        --mode train --config frozen --head direct \
        $COMMON $RESIDUAL $MOLFORMER --set antisymmetry_penalty=1.0 --set seed=$seed
done

echo
echo "################################################################################"
echo "=== [$(date '+%F %T')] XONG_PAIRSPLIT: $total lần chạy, $failed lỗi."
echo "=== Đọc kết quả:"
echo "===   python scripts/group_runs.py --slice strict --min-seeds 3 --grep pair_split"
echo "=== Nhắc lại: số đợt này KHÔNG trừ thẳng với số các đợt trước."
rm -f "$cache/$LOG_NAME.pid"
exit $((failed > 0))
