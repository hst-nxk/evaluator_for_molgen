#!/usr/bin/env bash
# Việc 6 họp 09/11/2026, phần còn thiếu: khoảng cách train–validation của checkpoint được
# chọn theo lát `strict`, cho CẢ HAI biểu diễn phân tử.
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_errors_gap_1311.sh 5 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/errors_gap_1311.log
#
# Tham số duy nhất là CHỈ SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/errors_gap_1311.pid)
#
# KHÔNG train gì. Chỉ nạp checkpoint đã có rồi chấm từng dòng (`--mode errors`), nên tổng
# thời gian ~30 phút: mỗi lượt validation (493.852 dòng) ~1 phút, mỗi lượt train
# (6.627.559 dòng) ~13 phút.
#
# NỀN TẢNG — số đã đo sáng 12/11 trên checkpoint chọn theo lát `all`
# (frozen_direct_20261106_184804.pt, hạt giống 20261023, 7x102 residual, Morgan, e3/b10000;
# runs/gap_1211.log):
#
#               train     validation
#   MAE        0,2223        0,5947
#   R2         0,8218       -0,0013
#   Spearman   0,8965        0,1529
#   đúng chiều 0,9489        0,5714     <- theo DÒNG (micro), không so được với cột
#   spread      0,829         0,354        `đúng chiều` macro của summarize_runs.py
#
# Checkpoint đó học thuộc tập train gần như hoàn toàn và chuyển giao gần như bằng 0. Nó là
# đúng cái mô hình đã sinh ra con số 0,5657 trên lát `all` — tức baseline cũ của paper.
#
# CÂU HỎI CỦA ĐỢT NÀY, hai câu, cùng một bảng trả lời:
#
#   1. Quy tắc chọn checkpoint mới mua được chính xác cái gì? Checkpoint chọn theo `strict`
#      dừng ở `e0/b6000` thay vì `e3/b10000`, tức thấy dữ liệu ít hơn ~7 lần. Nếu khoảng
#      cách train–validation của nó NHỎ hơn hẳn thì +0,0032 mà Morgan lấy lại được trên lát
#      `strict` có cơ chế: ta dừng TRƯỚC khi mô hình học thuộc.
#
#   2. Morgan có học thuộc nhiều hơn MoLFormer không? Hôm nay đã suy ra gián tiếp từ khoảng
#      cách strict−all: Morgan -0,0072, MoLFormer -0,0022, tức gấp ~3 lần. Đó là SUY RA, ghi
#      trong nhật ký là CHƯA ĐO. Bảng train–validation theo từng biểu diễn là phép đo TRỰC
#      TIẾP cho cùng cơ chế, không cần train thêm gì. Nếu Morgan chênh rộng hơn MoLFormer
#      thì cơ chế được xác nhận bằng hai con đường độc lập.
#
# DÙNG HẠT GIỐNG 20261023 cho cả hai nhánh, cố ý: bản dump 06/11 ở trên cũng là hạt 20261023,
# nên phép so "chọn theo all" với "chọn theo strict" không lẫn nhiễu hạt giống.
#
# THỨ TỰ: mỗi mô hình chạy validation TRƯỚC rồi train, và Morgan xong hẳn mới sang MoLFormer.
# Nếu đợt bị cắt giữa đường thì phần đã xong vẫn là một CẶP đầy đủ, vẫn ra được một bảng.
#
# CẢNH BÁO ĐÃ KIỂM: checkpoint lưu `model_settings` nên kiến trúc dựng lại từ file .pt, NHƯNG
# `ligand_features` thì đọc từ config đang sống (src/molopt/evaluator/train.py dòng 340), nên
# nhánh MoLFormer BẮT BUỘC có `--set model.ligand_features=embedding`. Thiếu cờ đó thì trọng
# số là MoLFormer mà mảng đầu vào là Morgan 2048 bit.
#
# `scripts/error_analysis.py gap` KHÔNG tự kiểm hai thư mục có cùng checkpoint hay không, nên
# script này kiểm hộ trước khi gọi, và bỏ qua gap/hist của cặp nào lệch.
#
# KHÔNG dùng partition test ở bất kỳ đâu trong script này (run_evaluator.py chặn sẵn).
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="errors_gap_1311"
SEED=20261023

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Cần đúng một tham số: chỉ số GPU. Ví dụ: bash $0 5" >&2
    exit 1
fi
GPU_INDEX="$1"

COMMON="--set gpu_memory_limit_mib=7500 --set eval_batch_size=2048"
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

if [[ ! -f "$root/work/evaluator_cache_v4/molecule_embeddings.parquet" ]]; then
    echo "Thiếu cache embedding phân tử (MoLFormer). Dựng trước bằng:" >&2
    echo "  python run_evaluator.py --mode prepare --molecule-embeddings" >&2
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
echo "=== [$(date '+%F %T')] việc 6: khoảng cách train–validation | GPU $GPU_INDEX | PID $$"
echo "=== $root"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
echo "=== Mỗi lượt train ghi rows.parquet của 6,6 triệu dòng; chỗ trống trên đĩa:"
df -h "$root" | tail -1

# --- Tìm hai checkpoint: cùng hạt giống, chọn theo lát strict, full train ----------------
# Đặt SAU khi đã chuyển stdout vào log, để nếu không tìm được thì lý do nằm trong log chứ
# không bay vào /dev/null của nohup.
echo
echo "=== Tìm checkpoint hạt giống $SEED, chọn theo lát strict, full train, không chuẩn hóa nhãn"
picked="$cache/$LOG_NAME.checkpoints"
if ! SEED="$SEED" python - >"$picked" <<'PY'
import glob
import json
import os
import sys

seed = int(os.environ["SEED"])
found = {}
for path in sorted(glob.glob("runs/evaluator/train_*.json")):
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        continue
    overrides = manifest.get("overrides") or {}
    if manifest.get("selection_slice") != "strict":
        continue
    if manifest.get("head") != "direct":
        continue
    if manifest.get("seed") != seed:
        continue
    if manifest.get("train_subset"):
        continue
    if manifest.get("partition_override"):
        continue
    if "label_normalization" in overrides:
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

# Thư mục mới nhất mà `--mode errors` vừa ghi cho một partition. Tên thư mục do train.py
# đặt (errors_<cấu hình>_<đầu ra>_<partition>_<stamp>) nên không đoán trước được stamp.
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

# Kiểm hai thư mục có cùng checkpoint và đúng checkpoint mong đợi, rồi mới gap + hist.
pair_report() {
    local name="$1" checkpoint="$2" train_dir="$3" validation_dir="$4"
    echo
    echo "================================================================================"
    echo "=== [$(date '+%F %T')] bảng $name"
    echo "=== train      $train_dir"
    echo "=== validation $validation_dir"
    if [[ -z "$train_dir" || -z "$validation_dir" ]]; then
        echo "=== Thiếu thư mục dump. Bỏ qua bảng này."
        failed=$((failed + 1))
        return 1
    fi
    if ! python - "$checkpoint" "$train_dir" "$validation_dir" <<'PY'
import json
import os
import sys

want, train_dir, validation_dir = sys.argv[1], sys.argv[2], sys.argv[3]


def checkpoint_of(folder: str) -> str:
    with open(os.path.join(folder, "report.json"), encoding="utf-8") as handle:
        return str(json.load(handle).get("checkpoint") or "")


train, validation = checkpoint_of(train_dir), checkpoint_of(validation_dir)
print("  train      checkpoint:", train)
print("  validation checkpoint:", validation)
print("  mong doi             :", want)
# report.json luu duong dan NHU DA TRUYEN (tuong doi), manifest luu duong dan tuyet doi,
# nen so theo ten file.
ok = bool(train) and train == validation and os.path.basename(train) == os.path.basename(want)
print("  ->", "KHOP" if ok else "*** LECH - KHONG DUNG BANG NAY ***")
raise SystemExit(0 if ok else 1)
PY
    then
        echo "=== Lệch checkpoint: KHÔNG chạy gap/hist cho $name."
        failed=$((failed + 1))
        return 1
    fi
    python scripts/error_analysis.py gap --train "$train_dir" \
        --validation "$validation_dir" --out "runs/gap_${name}_1311.json" || \
        failed=$((failed + 1))
    python scripts/error_analysis.py hist --train "$train_dir" \
        --validation "$validation_dir" --out "runs/figures/dist_${name}_1311.png" || \
        failed=$((failed + 1))
}

echo
echo "================================================================================"
echo "=== NHÁNH 1/2: Morgan fingerprint, checkpoint chọn theo lát strict"
echo "=== $MORGAN_CKPT"

morgan_validation=""
morgan_train=""
if run_errors "Morgan | validation" "$MORGAN_CKPT" validation; then
    morgan_validation="$(latest_dir validation)"
fi
if run_errors "Morgan | train (6,6 triệu dòng, ~13 phút)" "$MORGAN_CKPT" train; then
    morgan_train="$(latest_dir train)"
fi
pair_report morgan_strict "$MORGAN_CKPT" "$morgan_train" "$morgan_validation" || true

echo
echo "================================================================================"
echo "=== NHÁNH 2/2: MoLFormer, checkpoint chọn theo lát strict"
echo "=== $MOLFORMER_CKPT"
echo "=== BẮT BUỘC có --set model.ligand_features=embedding (train.py dòng 340)"

molformer_validation=""
molformer_train=""
if run_errors "MoLFormer | validation" "$MOLFORMER_CKPT" validation $MOLFORMER; then
    molformer_validation="$(latest_dir validation)"
fi
if run_errors "MoLFormer | train (6,6 triệu dòng, ~13 phút)" "$MOLFORMER_CKPT" train \
    $MOLFORMER; then
    molformer_train="$(latest_dir train)"
fi
pair_report molformer_strict "$MOLFORMER_CKPT" "$molformer_train" \
    "$molformer_validation" || true

echo
echo "=== [$(date '+%F %T')] xong tất cả | $total lượt chấm | số lần lỗi: $failed"
echo "Đã ghi:"
echo "  runs/gap_morgan_strict_1311.json      runs/figures/dist_morgan_strict_1311.png"
echo "  runs/gap_molformer_strict_1311.json   runs/figures/dist_molformer_strict_1311.png"
echo
echo "So ba bảng với nhau (cùng hạt giống 20261023, cùng partition):"
echo "  chọn theo all, Morgan      -> runs/gap_1211.log      (MAE 0,2223 / 0,5947)"
echo "  chọn theo strict, Morgan   -> bảng morgan_strict ở trên"
echo "  chọn theo strict, MoLFormer-> bảng molformer_strict ở trên"
echo "Cột cần đọc: chênh của mae, r2, spearman và spread_ratio."
echo "XONG_ERRORS_GAP_1311"
