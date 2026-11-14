#!/usr/bin/env bash
# Cơ chế đứng sau kết quả lớn nhất dự án từng đo: vì sao Morgan thắng MoLFormer +0,1186
# trên cách chia theo CẶP (đợt 4, 14/11/2026).
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_errors_pairsplit_1411.sh 1 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/errors_pairsplit_1411.log
#
# Tham số duy nhất là CHỈ SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/errors_pairsplit_1411.pid)
#
# KHÔNG train gì. Chỉ nạp checkpoint đã có rồi chấm từng dòng (`--mode errors`). Tập của
# cách chia này nhỏ (train 75.000 dòng, validation 24.946 dòng) nên cả đợt ~10 phút.
#
# SỐ NỀN TẢNG (đợt 4, lát strict, 3 hạt giống, runs/pairsplit_1411.log):
#
#   Morgan                0,7769  sd 0,0070   0,7778 / 0,7696 / 0,7834
#   MoLFormer             0,6583  sd 0,0061   0,6605 / 0,6514 / 0,6630
#   MoLFormer + phạt 1.0  0,6301  sd 0,0044   0,6308 / 0,6342 / 0,6254
#
# Morgan hơn +0,1186; chênh từng hạt +0,1173 / +0,1182 / +0,1204, tức 3/3 cùng dấu, t ~ 129.
# MAE nói cùng chiều (0,5132 so 0,5654). Trên cách chia theo PROTEIN thì ngược lại:
# MoLFormer hơn +0,0068 (t = 6,4). Đổi dấu, độ lớn gấp 17 lần.
#
# GIẢ THUYẾT ĐANG KIỂM: khi protein P đã có trong train, bài toán chuyển dần về TRA CỨU,
# và Morgan mã hóa cấu trúc CHÍNH XÁC nên khớp lại được đúng phân tử đã thấy cùng protein
# đó, còn embedding MoLFormer trơn nên mất lợi thế này.
#
# VÌ SAO KHÔNG DÙNG `flip`: `flip` đòi cùng một cặp (A, B) xuất hiện với >= 2 Context P, mà
# validation của đợt 4 có 24.751 cặp phân biệt trên 24.946 dòng — gần như không nhóm nào đủ
# điều kiện, nên nó sẽ trả về gần 0 nhóm. Phép đo đúng là `lookup`: chia dòng validation
# theo số lần phân tử A, B đã xuất hiện TRONG TRAIN, rồi so đúng chiều theo thùng.
#
# BẰNG CHỨNG NẰM Ở HIỆU HAI MÔ HÌNH, KHÔNG Ở MỘT MÔ HÌNH: thùng tần suất cao có thể dễ hơn
# vì lý do khác (phân tử phổ biến thường là phân tử đã tối ưu, Δp nhỏ, v.v.). Chỉ khi lợi
# thế của Morgan DỒN vào thùng cao và TẮT ở thùng 0 thì cơ chế tra cứu được xác nhận. Lợi
# thế rải đều mọi thùng thì KHÔNG phải tra cứu, và phải nói ra như vậy.
#
# CẢNH BÁO ĐÃ KIỂM, hai cờ bắt buộc, thiếu là số sai mà KHÔNG báo lỗi:
#
#   1. `--set inputs.partition_override_pairs=...`: thiếu cờ này thì Stage 3 đọc cột
#      partition gốc, tức quay về cách chia theo protein, và dump sẽ là của tập khác hẳn.
#   2. `--set model.ligand_features=embedding` cho nhánh MoLFormer: checkpoint lưu
#      `model_settings` nên kiến trúc dựng lại được từ file .pt, NHƯNG `ligand_features`
#      đọc từ config đang sống (src/molopt/evaluator/train.py). Thiếu cờ đó thì trọng số là
#      MoLFormer mà mảng đầu vào là Morgan 2048 bit.
#
# Script tự kiểm cả hai bằng cách đọc manifest của checkpoint, chứ không tin vào tên file.
#
# CHỈ LẤY HẠT GIỐNG 20261023 cho cả hai nhánh: câu hỏi ở đây là CƠ CHẾ, không phải độ lớn,
# nên một hạt là đủ và phép so hai mô hình không lẫn nhiễu hạt giống. Nhánh phạt 1.0 bỏ qua
# ở đợt này — nó không nằm trong giả thuyết tra cứu.
#
# KHÔNG dùng partition test ở bất kỳ đâu (run_evaluator.py chặn sẵn).
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="errors_pairsplit_1411"
SEED=20261023
SPLIT_FILE="work/splits/pair_split_1411.parquet"
MIN_FREE_GIB=20

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: chỉ số GPU. Ví dụ: bash $0 1" >&2
    exit 1
fi
GPU_INDEX="$1"

# 6000 chu khong 7500: chieu 15/11 ca 8 GPU deu bi chiem 73-81 GB / 81,5 GB, cho
# trong nhieu nhat chi ~8,4 GB. Xin 7500 la OOM tren GPU dang dong nhat.
COMMON="--set gpu_memory_limit_mib=6000 --set eval_batch_size=2048 \
--set inputs.partition_override_pairs=$SPLIT_FILE"
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

if [[ ! -f "$root/$SPLIT_FILE" ]]; then
    echo "Thiếu bảng chia theo cặp $SPLIT_FILE. Dựng trước bằng:" >&2
    echo "  python scripts/build_pair_split.py --out $SPLIT_FILE" >&2
    exit 1
fi

if [[ ! -f "$root/work/evaluator_cache_v4/molecule_embeddings.parquet" ]]; then
    echo "Thiếu cache embedding phân tử (MoLFormer). Dựng trước bằng:" >&2
    echo "  python run_evaluator.py --mode prepare --molecule-embeddings" >&2
    exit 1
fi

# Đĩa /data đã ở 97 % chiều 14/11 và tụt ~33 G trong mười phút vì job của người khác.
# Lần phóng đợt 3 đã chết 4 lần chạy vì `OSError: [Errno 28] No space left on device`,
# rồi torch đọc lại checkpoint ghi cụt và chết tiếp. Thà dừng trước khi ghi gì.
free_gib="$(df -BG --output=avail "$root" | tail -1 | tr -dc '0-9')"
if [[ -z "$free_gib" || "$free_gib" -lt "$MIN_FREE_GIB" ]]; then
    echo "Chỉ còn ${free_gib:-?} GiB trên đĩa, cần >= $MIN_FREE_GIB GiB. Dừng." >&2
    echo "Đợt 3 đã mất 4 lần chạy vì đĩa đầy giữa lúc ghi checkpoint." >&2
    exit 1
fi

cache="$root/.cache"
mkdir -p runs runs/figures "$cache"
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
echo "=== [$(date '+%F %T')] cơ chế đợt 4: tra cứu hay tổng quát hóa | GPU $GPU_INDEX | PID $$"
echo "=== $root"
echo "=== bảng chia: $SPLIT_FILE"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
echo "=== chỗ trống trên đĩa:"
df -h "$root" | tail -1

# --- Tìm hai checkpoint của ĐỢT 4: cùng hạt giống, cùng bảng chia theo cặp ---------------
# Đặt SAU khi đã chuyển stdout vào log, để nếu không tìm được thì lý do nằm trong log chứ
# không bay vào /dev/null của nohup.
echo
echo "=== Tìm checkpoint hạt giống $SEED, chia theo cặp, head direct, chọn theo lát strict"
picked="$cache/$LOG_NAME.checkpoints"
if ! SEED="$SEED" SPLIT_FILE="$SPLIT_FILE" python - >"$picked" <<'PY'
import glob
import json
import os
import sys

seed = int(os.environ["SEED"])
split = os.environ["SPLIT_FILE"]
found = {}
for path in sorted(glob.glob("runs/evaluator/train_*.json")):
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        continue
    overrides = manifest.get("overrides") or {}
    # Khóa chốt: ĐÚNG bảng chia theo cặp. Manifest ghi trường này từ 14/11/2026
    # (src/molopt/evaluator/train.py); manifest cũ hơn không có trường -> bị loại, đúng ý.
    if (manifest.get("partition_override_pairs") or "") != split:
        continue
    if manifest.get("selection_slice") != "strict":
        continue
    if manifest.get("head") != "direct":
        continue
    if manifest.get("seed") != seed:
        continue
    # Bỏ nhánh phạt phản đối xứng: không nằm trong giả thuyết tra cứu.
    if any(key.endswith("antisymmetry_penalty") for key in overrides):
        continue
    checkpoint = manifest.get("checkpoint") or ""
    if not checkpoint:
        continue
    kind = ("molformer" if overrides.get("model.ligand_features") == "embedding"
            else "morgan")
    found.setdefault(kind, []).append((checkpoint, path))

problem = [kind for kind in ("morgan", "molformer") if len(found.get(kind, [])) != 1]
if problem:
    for kind in ("morgan", "molformer"):
        for checkpoint, path in found.get(kind, []):
            sys.stderr.write("  %s <- %s\n" % (kind, path))
    sys.stderr.write("Can dung MOT checkpoint moi loai, thieu/thua o: %s\n"
                     % ", ".join(problem))
    raise SystemExit(1)

for kind in ("morgan", "molformer"):
    checkpoint, path = found[kind][0]
    sys.stderr.write("  %-9s %s\n     manifest %s\n" % (kind, checkpoint, path))
    print(checkpoint)
PY
then
    echo "=== Không chốt được checkpoint. Dừng, không chạy gì."
    exit 1
fi

MORGAN_CKPT="$(sed -n 1p "$picked")"
MOLFORMER_CKPT="$(sed -n 2p "$picked")"
for file in "$MORGAN_CKPT" "$MOLFORMER_CKPT"; do
    if [[ ! -f "$file" ]]; then
        echo "=== Không thấy file checkpoint: $file. Dừng."
        exit 1
    fi
done

failed=0
total=0

latest_dir() {
    local partition="$1"
    ls -dt runs/evaluator/errors_*_"$partition"_*/ 2>/dev/null | head -1 | sed 's:/*$::'
}

run_errors() {
    local label="$1" checkpoint="$2" partition="$3"
    shift 3
    total=$((total + 1))
    echo
    echo "--------------------------------------------------------------------------------"
    echo "=== [$(date '+%F %T')] $total: $label"
    echo "=== python run_evaluator.py --mode errors --checkpoint $checkpoint" \
        "--partition $partition $COMMON $*"
    local started=$SECONDS
    local status=0
    python run_evaluator.py --mode errors --checkpoint "$checkpoint" \
        --partition "$partition" $COMMON "$@" || status=$?
    # Bắt mã thoát vào biến TRƯỚC khi echo: $(date) trong cùng dòng echo chạy trước và
    # ghi đè $?, nên in $? trực tiếp sẽ luôn ra 0.
    local minutes=$(((SECONDS - started) / 60))
    if [[ $status -eq 0 ]]; then
        echo "=== [$(date '+%F %T')] $total xong sau $minutes phút"
    else
        echo "=== [$(date '+%F %T')] $total LỖI mã $status sau $minutes phút"
        failed=$((failed + 1))
    fi
    return $status
}

MORGAN_TRAIN=""
MORGAN_VALID=""
MOLFORMER_TRAIN=""
MOLFORMER_VALID=""

# Morgan xong hẳn mới sang MoLFormer: đợt bị cắt giữa đường thì phần đã xong vẫn là một
# CẶP train+validation đầy đủ của một mô hình.
if run_errors "Morgan · validation" "$MORGAN_CKPT" validation; then
    MORGAN_VALID="$(latest_dir validation)"
fi
if run_errors "Morgan · train" "$MORGAN_CKPT" train; then
    MORGAN_TRAIN="$(latest_dir train)"
fi
if run_errors "MoLFormer · validation" "$MOLFORMER_CKPT" validation $MOLFORMER; then
    MOLFORMER_VALID="$(latest_dir validation)"
fi
if run_errors "MoLFormer · train" "$MOLFORMER_CKPT" train $MOLFORMER; then
    MOLFORMER_TRAIN="$(latest_dir train)"
fi

echo
echo "=== Thư mục dump"
echo "=== Morgan     train $MORGAN_TRAIN"
echo "=== Morgan     valid $MORGAN_VALID"
echo "=== MoLFormer  train $MOLFORMER_TRAIN"
echo "=== MoLFormer  valid $MOLFORMER_VALID"

# --- Bảng cơ chế -------------------------------------------------------------------------
# Lát `all` cố ý: trên cách chia theo cặp, apply_pair_partition_override ghi
# unseen_pair = True cho MỌI dòng validation, nên strict TRÙNG all ở đây. Dùng `all` để
# khỏi phụ thuộc vào việc cột đó có được ghi đè đúng hay không.
analysis() {
    local name="$1"
    shift
    echo
    echo "================================================================================"
    echo "=== [$(date '+%F %T')] $name"
    echo "=== python scripts/error_analysis.py $*"
    python scripts/error_analysis.py "$@" || failed=$((failed + 1))
}

for pair in "Morgan:$MORGAN_TRAIN:$MORGAN_VALID" \
            "MoLFormer:$MOLFORMER_TRAIN:$MOLFORMER_VALID"; do
    name="${pair%%:*}"
    rest="${pair#*:}"
    train_dir="${rest%%:*}"
    valid_dir="${rest#*:}"
    if [[ -z "$train_dir" || -z "$valid_dir" ]]; then
        echo
        echo "=== Thiếu thư mục dump cho $name. Bỏ qua phần phân tích của nó."
        failed=$((failed + 1))
        continue
    fi
    lower="$(echo "$name" | tr '[:upper:]' '[:lower:]')"
    analysis "lookup · $name" lookup --train "$train_dir" --validation "$valid_dir" \
        --slice all --label "$name" --out "runs/figures/lookup_${lower}_1411.json"
    analysis "gap · $name" gap --train "$train_dir" --validation "$valid_dir" \
        --slice all --out "runs/figures/gap_pairsplit_${lower}_1411.json"
    analysis "classes · $name" classes --train "$train_dir" --validation "$valid_dir" \
        --slice all --out "runs/figures/classes_pairsplit_${lower}_1411.png"
done

echo
echo "################################################################################"
if [[ $failed -eq 0 ]]; then
    echo "=== [$(date '+%F %T')] XONG_ERRORS_PAIRSPLIT: $total lượt dump, 0 lỗi."
else
    echo "=== [$(date '+%F %T')] XONG_ERRORS_PAIRSPLIT: $total lượt dump, $failed LỖI."
fi
echo "=== Đọc: so cột 'đúng chiều' theo thùng giữa hai file"
echo "===   runs/figures/lookup_morgan_1411.json"
echo "===   runs/figures/lookup_molformer_1411.json"
echo "=== Lợi thế Morgan DỒN vào thùng tần suất cao và TẮT ở thùng 0 -> cơ chế tra cứu."
echo "=== Lợi thế rải đều mọi thùng -> KHÔNG phải tra cứu, phải nói ra như vậy."
df -h "$root" | tail -1
exit $((failed > 0))
