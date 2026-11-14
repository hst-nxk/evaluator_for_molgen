#!/usr/bin/env bash
# Đợt 14/11/2026 (đợt 3): BÓC HAI NHÁNH PHỤ. 12 lần chạy, tập con 100.000 dòng, MoLFormer,
# chọn checkpoint theo lát `strict`, ba hạt giống 20261023/24/25.
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_stripped_1411.sh 4 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/stripped_1411.log
#
# Tham số duy nhất là CHỈ SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/stripped_1411.pid)
#
# ------------------------------------------------------------------------------------
# VÌ SAO CÓ ĐỢT NÀY: HAI SỐ ĐO CHỈ VỀ CÙNG MỘT CHỖ
#
# 1. Bỏ hẳn nhánh PROTEIN không làm xấu thước nào (đo 14/11, Morgan, tập con 100k):
#       đúng chiều  0,5505 +/- 0,0005  so  0,5499 +/- 0,0007   (t =  1,06)
#       MAE         0,6007 +/- 0,0006  so  0,6010 +/- 0,0015   (t = -0,32)
#       RMSE        0,8212 +/- 0,0003  so  0,8216 +/- 0,0011   (t = -0,62)
#    Khớp với việc 9: sd dự đoán giữa các Context P chỉ 0,0114 trong khi sd nhãn 0,6566.
#
# 2. Bỏ nhánh EDIT không làm xấu xếp hạng (MoLFormer, FULL 6,6M dòng):
#       direct_no_edit 0,5598 +/- 0,0004  so  direct 0,5603 +/- 0,0009  (t = -0,64)
#    MAE của nó còn là số tốt nhất toàn bảng (0,5963 +/- 0,0007) nhưng sd của nhóm đối
#    chứng lớn (0,0040) nên t chỉ -1,96: GỢI Ý tốt hơn, CHƯA chứng minh.
#
# Hai nhánh phụ, mỗi nhánh bỏ riêng đều không mất gì. Câu chưa ai trả lời: bỏ CẢ HAI thì
# sao? Nếu vẫn không mất gì thì đó là câu mạnh nhất của phần kiến trúc trong paper —
# trần kết quả không nằm ở kiến trúc mà ở chỗ bài toán chỉ còn tín hiệu của riêng cặp
# phân tử, và mô hình cuối có thể nhỏ hơn hẳn mà không kém.
#
# ------------------------------------------------------------------------------------
# BỐN NHÓM, xếp theo thứ tự quan trọng giảm dần (kill giữa đợt chỉ mất nhóm cuối):
#   1. no_protein + direct_no_edit            -> bóc CẢ HAI. Câu hỏi chính.
#   2. no_protein + direct_no_edit + phạt 1,0 -> ứng viên cho mô hình cuối của paper.
#   3. direct_no_edit (giữ protein)           -> ablation đơn ở 100k, hiện chỉ có ở 6,6M.
#   4. no_protein (giữ edit), MoLFormer       -> ablation đơn, hiện chỉ có ở Morgan.
#
# MỐC SO (cùng tập con 100k, cùng MoLFormer, cùng ba hạt giống, lát strict):
#   MoLFormer đủ nhánh (đối chứng)      0,5561 +/- 0,0006
#   MoLFormer + phạt w=1,0              0,5588 +/- 0,0004   (t = 6,41 so với trên)
# Đối chứng này đã chạy BA lần ở ba đợt khác nhau và ra số trùng từng hạt
# (0,5558 / 0,5558 / 0,5568), nên đợt này KHÔNG chạy lại nó — đỡ 30 phút GPU.
# Sàn nhiễu hạt giống 0,0021. Ngưỡng chứng minh: 3/3 hạt cùng dấu VÀ t >= 3.
#
# CÁCH ĐỌC KẾT QUẢ ÂM: ở đợt này "không khác gì" là kết quả MONG ĐỢI và là kết quả TỐT —
# nó nói mô hình nhỏ hơn thì ngang bằng. Nhưng không được đổi "t < 3" thành "bằng nhau":
# ba hạt giống chỉ đủ loại trừ hiệu lớn hơn khoảng 0,002, không chứng minh được hiệu 0.
# Câu viết được: "không đo được khác biệt ở mức ba hạt giống".
#
# SỐ THAM SỐ: train.py ghi `parameters` vào manifest và in ra log. Đó là con số dùng cho
# câu "bỏ bao nhiêu phần trăm tham số", đọc từ log chứ không tính tay.
#   grep -n "tham số" runs/stripped_1411.log
#
# KHÔNG dùng partition test ở bất kỳ đâu (run_evaluator.py chặn sẵn).
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="stripped_1411"
SEEDS=(20261023 20261024 20261025)
SUBSET_FILE="work/splits/train_subset_c200_r500.parquet"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: chỉ số GPU. Ví dụ: bash $0 4" >&2
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

if [[ ! -f "$root/$SUBSET_FILE" ]]; then
    echo "Thiếu $SUBSET_FILE. Dựng bằng:" >&2
    echo "  python scripts/build_train_subset.py build --clusters 200 --rows-per-cluster 500" >&2
    exit 1
fi
for name in protein_embeddings molecule_embeddings; do
    if [[ ! -f "$root/work/evaluator_cache_v4/$name.parquet" ]]; then
        echo "Thiếu work/evaluator_cache_v4/$name.parquet. Dừng." >&2
        echo "  python run_evaluator.py --mode prepare --molecule-embeddings" >&2
        exit 1
    fi
done

# `no_protein` phải có trong danh sách cấu hình, và head `direct_no_edit` phải tồn tại.
# Kiểm trước, mất một giây, đỡ chạy nửa đợt mới báo lỗi tên cấu hình.
python - <<'PY' || exit 1
import json
import sys

sys.path.insert(0, "src")
from molopt.evaluator import models

config = json.load(open("config.json", encoding="utf-8"))
problems = []
allowed = config["stage3"]["train"].get("configurations") or []
if "no_protein" not in allowed:
    problems.append(f"'no_protein' không có trong stage3.train.configurations = {allowed}")
if "direct_no_edit" not in models.JOINT_HEADS:
    problems.append(f"head 'direct_no_edit' không có trong JOINT_HEADS = {models.JOINT_HEADS}")
if float(config["stage3"]["train"]["antisymmetry_penalty"]) != 0.0:
    problems.append("antisymmetry_penalty mặc định không còn là 0")
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
echo "=== [$(date '+%F %T')] đợt bóc nhánh phụ bắt đầu | GPU $GPU_INDEX | PID $$"
echo "=== $root"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
echo "=== Tập con 100.000 dòng, MoLFormer, lát strict, 3 hạt giống."
echo "=== Mốc: MoLFormer đủ nhánh 0,5561 +/- 0,0006 | + phạt 0,5588 +/- 0,0004."
echo "=== Câu hỏi: bỏ CẢ nhánh protein lẫn nhánh edit thì có mất gì không?"
echo "=== 'Không khác gì' ở đây là kết quả tốt, nhưng phải viết là 'không đo được khác"
echo "=== biệt ở mức ba hạt giống', không được viết 'bằng nhau'."

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
echo "=== NHÓM 1: BÓC CẢ HAI — no_protein + direct_no_edit. CÂU HỎI CHÍNH."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | no_protein + direct_no_edit (bóc cả hai nhánh phụ)" \
        --mode train --config no_protein --head direct_no_edit $COMMON $RESIDUAL \
        $MOLFORMER --set seed="$seed"
done

echo
echo "================================================================================"
echo "=== NHÓM 2: mô hình bóc gọn + phạt phản đối xứng w=1,0 — ỨNG VIÊN MÔ HÌNH CUỐI"
echo "=== Phạt chạy hai lượt forward mỗi batch nên nhóm này chậm hơn nhóm 1."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | no_protein + direct_no_edit + phạt 1.0" \
        --mode train --config no_protein --head direct_no_edit $COMMON $RESIDUAL \
        $MOLFORMER --set antisymmetry_penalty=1.0 --set seed="$seed"
done

echo
echo "================================================================================"
echo "=== NHÓM 3: chỉ bỏ nhánh EDIT, giữ protein (ablation đơn ở 100k)"
echo "=== Hiện chỉ có số ở FULL 6,6M dòng (0,5598 +/- 0,0004), không so thẳng được."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | direct_no_edit, giữ nhánh protein" \
        --mode train --config frozen --head direct_no_edit $COMMON $RESIDUAL \
        $MOLFORMER --set seed="$seed"
done

echo
echo "================================================================================"
echo "=== NHÓM 4: chỉ bỏ nhánh PROTEIN, giữ edit, trên MoLFormer (ablation đơn)"
echo "=== Hiện chỉ có số trên Morgan (0,5505 +/- 0,0005)."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | no_protein, giữ nhánh edit, MoLFormer" \
        --mode train --config no_protein --head direct $COMMON $RESIDUAL \
        $MOLFORMER --set seed="$seed"
done

echo
echo "=== [$(date '+%F %T')] xong tất cả | $total lần chạy | số lần lỗi: $failed"
echo "Đọc kết quả:"
echo "  for m in pairwise_accuracy_macro mae rmse; do echo \"## \$m\"; \\"
echo "    python scripts/group_runs.py --slice strict --min-seeds 3 --metric \$m; done \\"
echo "    | tee runs/groups_stripped.log"
echo "Số tham số của từng cấu hình (cho câu 'bỏ bao nhiêu phần trăm tham số'):"
echo "  grep -n 'tham số' runs/$LOG_NAME.log"
echo "XONG_STRIPPED_1411"
