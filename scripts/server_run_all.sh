#!/usr/bin/env bash
# Chạy toàn bộ thí nghiệm Stage 3 trên server, không cần người trông.
#
# Cách chạy (trong terminal server):
#   cd /mnt/disk4/khainx && conda activate nxk && cd evaluator_for_molgen
#   setsid nohup bash scripts/server_run_all.sh > /dev/null 2>&1 &
#   tail -f runs/server_run_all.log    # xem tiến độ; Ctrl+C chỉ thoát xem, không dừng chạy
#
# Chỉ đánh giá lại checkpoint TỐT NHẤT của mọi lần train đã có (không train):
#   setsid nohup bash scripts/server_run_all.sh eval > /dev/null 2>&1 &
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/server_run_all.pid)
#
# Cam kết của script:
#   - Chỉ chạy khi chính nó nằm đúng trong PROJECT_ROOT; mọi thứ ghi ra đều nằm trong đó:
#     HOME, cache Hugging Face/torch/triton/CUDA và file tạm đều chuyển vào .cache/.
#   - Chỉ dùng conda env ENV_NAME; sai env thì dừng ngay.
#   - Chỉ dùng 1 GPU: GPU còn trống nhiều bộ nhớ nhất lúc bắt đầu.
#   - VRAM bị chặn trên bởi stage3.train.gpu_memory_limit_mib trong config.json (9 GiB).
#   - Không dùng partition test: code Stage 3 chỉ đọc train và validation.
#   - Không chạy finetune: theo ước tính (chưa đo) nó cần quá trần VRAM ở batch hiện tại;
#     cần đo riêng trước khi đưa vào.
#
# Thứ tự: kiểm tra môi trường -> pytest -> baseline -> smoke -> train. Kiểm tra môi trường
# hoặc pytest lỗi thì dừng hẳn. Từ baseline trở đi, một bước lỗi chỉ được ghi lại, các bước
# sau vẫn chạy; riêng smoke của cấu hình nào lỗi thì bỏ train của cấu hình đó.
set -u -o pipefail

MODE="${1:-all}"
if [[ "$MODE" != "all" && "$MODE" != "eval" ]]; then
    echo "Chế độ '$MODE' không hợp lệ: dùng 'all' (mặc định) hoặc 'eval'." >&2
    exit 1
fi
PROJECT_ROOT="/mnt/disk4/khainx/evaluator_for_molgen"
# Chỗ bạn thường đứng khi gõ `conda activate nxk`; chỉ dùng khi script phải tự kích hoạt env.
CONDA_START_DIR="/mnt/disk4/khainx"
ENV_NAME="nxk"
LOG_NAME="server_run_all"
# GPU phải còn trống ít nhất bằng trần VRAM (9 GiB) cộng phần dự phòng cho CUDA context.
MIN_FREE_GPU_MIB=10240
# Chế độ eval chỉ dự đoán: VRAM cao nhất ĐO ĐƯỢC ngày 25/10/2026 của 4 lần train frozen /
# no_protein là 286-336 MiB, đã gồm cả bước dự đoán validation (batch 4096) như eval.
# Ngưỡng này gấp khoảng 6 lần số đo đó.
MIN_FREE_GPU_MIB_EVAL=2048
SECONDS_PER_MINUTE=60
SMOKE_CONFIGS=("no_protein" "frozen")
# Thứ tự train theo docs/STAGE3_EVALUATOR.md, chưa gồm finetune.
TRAIN_RUNS=(
    "frozen direct"
    "no_protein direct"
    "frozen direct_no_edit"
    "frozen twin"
)

# --- 1. Chốt thư mục: script phải nằm đúng trong PROJECT_ROOT -----------------------------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd "$script_dir/.." && pwd -P)"
expected="$(cd "$PROJECT_ROOT" 2>/dev/null && pwd -P)"
if [[ -z "$expected" || "$root" != "$expected" ]]; then
    echo "Script đang ở $root, không phải $PROJECT_ROOT. Dừng." >&2
    exit 1
fi
cd "$root" || exit 1

cache="$root/.cache"
mkdir -p runs "$cache"
log="$root/runs/$LOG_NAME.log"
exec >>"$log" 2>&1

# Không cho chạy hai lần cùng lúc.
exec 9>"$cache/$LOG_NAME.lock"
if ! flock -n 9; then
    echo "=== [$(date '+%F %T')] Đang có một lần chạy khác. Dừng."
    exit 1
fi
echo "$$" >"$cache/$LOG_NAME.pid"
trap 'echo "=== [$(date "+%F %T")] script bị dừng bởi tín hiệu"; exit 1' TERM INT

echo
echo "################################################################################"
echo "=== [$(date '+%F %T')] BẮT ĐẦU server_run_all $MODE (PID $$) trong $root"

fail() {
    echo "=== [$(date '+%F %T')] DỪNG: $*"
    exit 1
}

# --- 2. Chốt conda env ----------------------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" && -n "${CONDA_EXE:-}" ]]; then
    # Chỉ xảy ra khi chạy từ shell chưa kích hoạt env: tự kích hoạt giống cách bạn vẫn làm.
    set +u
    cd "$CONDA_START_DIR" || fail "không vào được $CONDA_START_DIR"
    eval "$("$CONDA_EXE" shell.bash hook)"
    conda activate "$ENV_NAME"
    cd "$root" || fail "không quay lại được $root"
    set -u
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
    fail "env đang là '${CONDA_DEFAULT_ENV:-không có}', cần '$ENV_NAME'"
fi
python_path="$(command -v python)"
if [[ "$python_path" != "$CONDA_PREFIX"/* ]]; then
    fail "python ($python_path) không thuộc env $ENV_NAME ($CONDA_PREFIX)"
fi
echo "env: $CONDA_DEFAULT_ENV | python: $python_path"

# --- 3. Giữ mọi file ghi ra trong project -----------------------------------------------
# Làm SAU khi kích hoạt env, vì conda cần HOME thật để đọc cấu hình.
export HOME="$cache/home"
export XDG_CACHE_HOME="$cache/xdg_cache"
export XDG_CONFIG_HOME="$cache/xdg_config"
export HF_HOME="$cache/huggingface"
export TORCH_HOME="$cache/torch"
export TRITON_CACHE_DIR="$cache/triton"
export CUDA_CACHE_PATH="$cache/nv"
export MPLCONFIGDIR="$cache/matplotlib"
export TMPDIR="$cache/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$HF_HOME" "$TORCH_HOME" \
    "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TMPDIR"

# --- 4. Chốt dữ liệu --------------------------------------------------------------------
for required in \
    data/dataset_v3/evaluator_training.parquet \
    data/dataset_v3/full/proteins.parquet \
    work/evaluator_cache_v3/cache_manifest.json \
    work/evaluator_cache_v3/fragment_fingerprints.parquet \
    work/evaluator_cache_v3/molecule_fingerprints.parquet \
    work/evaluator_cache_v3/protein_embeddings.parquet; do
    [[ -f "$required" ]] || fail "thiếu $required"
done
echo "dữ liệu: đủ 6 file"

# --- 5. Chọn 1 GPU ----------------------------------------------------------------------
nvidia-smi || fail "không gọi được nvidia-smi"
gpu_line="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -n -r | head -n 1)"
gpu_index="$(echo "${gpu_line%%,*}" | tr -d ' ')"
gpu_free="$(echo "${gpu_line#*,}" | tr -d ' ')"
if [[ -z "$gpu_index" || -z "$gpu_free" ]]; then
    fail "không đọc được bộ nhớ GPU"
fi
min_free="$MIN_FREE_GPU_MIB"
[[ "$MODE" == "eval" ]] && min_free="$MIN_FREE_GPU_MIB_EVAL"
if (( gpu_free < min_free )); then
    fail "GPU trống nhất (số $gpu_index) chỉ còn $gpu_free MiB < $min_free MiB"
fi
export CUDA_VISIBLE_DEVICES="$gpu_index"
echo "dùng GPU số $gpu_index (còn trống $gpu_free MiB lúc bắt đầu)"

# --- 6. Chạy từng bước ------------------------------------------------------------------
summary=()

run_step() {
    # Nhận: nhãn của bước, rồi lệnh cần chạy. Trả về mã thoát của lệnh.
    local label="$1"
    shift
    local started=$SECONDS
    echo
    echo "=== [$(date '+%F %T')] BẮT ĐẦU: $label"
    echo "    lệnh: $*"
    "$@"
    local code=$?
    local minutes=$(( (SECONDS - started) / SECONDS_PER_MINUTE ))
    local status="OK"
    (( code == 0 )) || status="LỖI (mã $code)"
    echo "=== [$(date '+%F %T')] XONG: $label -> $status sau $minutes phút"
    summary+=("$label: $status, $minutes phút")
    return "$code"
}

print_summary() {
    echo
    echo "=== [$(date '+%F %T')] TÓM TẮT"
    local line
    for line in "${summary[@]}"; do
        echo "    $line"
    done
    echo "    Kết quả từng lần chạy: runs/evaluator/ (file .json, .log, .pt)"
}

if [[ "$MODE" == "eval" ]]; then
    # Checkpoint lấy từ manifest của từng lần train (không lấy theo tên file .pt, vì smoke
    # và train đặt tên checkpoint giống nhau).
    shopt -s nullglob
    manifests=(runs/evaluator/train_*.json)
    (( ${#manifests[@]} > 0 )) || fail "chưa có manifest train nào trong runs/evaluator"
    for manifest in "${manifests[@]}"; do
        checkpoint="$(python -c 'import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"] or "")' "$manifest")"
        if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
            summary+=("eval $manifest: BỎ QUA (không có checkpoint)")
            continue
        fi
        run_step "eval $(basename "$checkpoint")" \
            python run_evaluator.py --mode eval --checkpoint "$checkpoint"
    done
    print_summary
    echo "=== [$(date '+%F %T')] HẾT"
    rm -f "$cache/$LOG_NAME.pid"
    exit 0
fi

run_step "kiểm tra thư viện và GPU" python - <<'PY' || fail "môi trường chưa đúng"
import sys

import rdkit
import torch
import transformers

print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| rdkit", rdkit.__version__)
if not torch.cuda.is_available():
    sys.exit("torch không thấy GPU")
print("GPU:", torch.cuda.get_device_name(0))
PY

if ! run_step "pytest" python -m pytest -q; then
    print_summary
    fail "pytest lỗi, không train"
fi

run_step "baseline" python run_evaluator.py --mode baseline

declare -A smoke_ok=()
for configuration in "${SMOKE_CONFIGS[@]}"; do
    if run_step "smoke $configuration" \
        python run_evaluator.py --mode smoke --config "$configuration"; then
        smoke_ok[$configuration]=1
    fi
done

for entry in "${TRAIN_RUNS[@]}"; do
    read -r configuration head <<<"$entry"
    label="train $configuration $head"
    if [[ " ${SMOKE_CONFIGS[*]} " == *" $configuration "* && -z "${smoke_ok[$configuration]:-}" ]]
    then
        summary+=("$label: BỎ QUA (smoke $configuration lỗi)")
        continue
    fi
    run_step "$label" \
        python run_evaluator.py --mode train --config "$configuration" --head "$head"
done

print_summary
echo "=== [$(date '+%F %T')] HẾT"
rm -f "$cache/$LOG_NAME.pid"
