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
#   - VRAM bị chặn trên bởi stage3.train.gpu_memory_limit_mib trong config.json (9 GiB);
#     riêng các lần finetune tự nâng trần bằng --set gpu_memory_limit_mib.
#   - Không dùng partition test: code Stage 3 chỉ đọc train và validation.
#   - Finetune chạy SAU CÙNG: nó đắt nhất và trần VRAM của nó chưa đo được lần nào, nên
#     nếu nó hỏng thì mọi kết quả frozen / no_protein đã xong và đã ghi ra đĩa từ trước.
#
# Thứ tự: kiểm tra môi trường -> pytest -> prepare -> baseline -> smoke -> train. Kiểm tra
# môi trường, pytest hoặc prepare lỗi thì dừng hẳn. Từ baseline trở đi, một bước lỗi chỉ
# được ghi lại, các bước sau vẫn chạy; riêng smoke của cấu hình nào lỗi thì bỏ train của
# cấu hình đó.
#
# CẦN CÓ SẴN TRÊN SERVER: data/dataset_v4/ (2,4 GB, bị gitignore nên git pull KHÔNG mang
# theo). Thiếu nó thì bước prepare hỏng ngay và script dừng. Chép bằng rsync từ máy cá nhân.
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
SMOKE_CONFIGS=("no_protein" "frozen" "finetune")
# Thứ tự train theo docs/STAGE3_EVALUATOR.md, chưa gồm finetune.
# Mỗi dòng: "<cấu hình> <đầu ra> <sharpness on|off>". Cột thứ ba bật số hạng phạt chuẩn
# gradient đầu ra (IGNITE, arXiv:2503.04242); thiếu cột thì coi là off.
# Hai dòng cuối là ablation có / không số hạng phạt, giữ nguyên mọi tham số khác, nên
# so sánh được trực tiếp với dòng đầu. Chạy sau cùng vì mỗi batch cần 3 lượt forward.
# Hai lần finetune chạy sau cùng và cần tham số riêng:
#   gpu_memory_limit_mib 20480 : trần 9 GiB của config là để chia GPU với người khác;
#       finetune giữ ESM2 trong đồ thị nên cần nhiều hơn. 20 GiB vẫn chừa ~4 GiB trên
#       card 24 GB. Vòng train gom batch theo protein nên mức dùng thường xuyên chỉ
#       ~1,4 GiB; trần này là để đỡ những batch hiếm chạm nhiều protein cùng lúc.
#   eval_every_steps 4000      : batch 256 cho 25.889 bước mỗi epoch, nên 4000 bước
#       là khoảng 6 lần đo validation mỗi epoch, bằng nhịp của frozen.
FINETUNE_SET="--set gpu_memory_limit_mib=20480 --set eval_every_steps=4000"

# Mỗi dòng: "<cấu hình> <đầu ra> <sharpness on|off> [--set khóa=giá_trị ...]"
#
# ĐỢT 2 (01/11/2026). Đợt 1 gồm 19 dòng ablation kiến trúc + quét chống overfit; kết
# quả của nó (đọc bằng scripts/summarize_runs.py) quyết định đợt này:
#   - Tốt nhất đợt 1: frozen direct model.dropout=0.5 -> MAE 0,5852, Spearman 0,1343,
#     R2 0,0173, thắng ở epoch 0 bước 10000. Baseline luôn đoán 0: MAE 0,5879.
#   - dropout là nút DUY NHẤT ăn rõ, và 0,5 tốt hơn 0,3 -> chưa tới đỉnh, nên đợt này
#     đi tiếp 0,6 và 0,7. Chỉ khi MAE xấu đi mới biết đã quá đỉnh.
#   - weight_decay đứng một mình đều làm xấu (0,03/0,1/0,3 -> 0,6006/0,6019/0,5979 so
#     với 0,5976 khi để trần) -> đợt này BỎ weight_decay.
#   - IGNITE xấu hơn đối chứng cả hai lần (0,6018 / 0,5976 và 0,5880 / 0,5864)
#     -> đợt này BỎ sharpness, mọi dòng đều off.
#   - Gộp cả ba nút (huber_delta=0.3 + weight_decay=0.1 + dropout=0.3) ra 0,5864,
#     TỆ hơn chỉ dropout=0.5 một mình -> bỏ cách gộp đó,
#     chỉ còn thử ghép dropout với huber_delta vì huber_delta có nguồn.
# Hai dòng cuối tách một CONFOUND của đợt 1: đợt đó đổi cả accumulate_steps (1->8)
# VÀ eval_every_steps (4000->500), nên không biết phần thắng là nhờ cộng dồn gradient
# hay chỉ nhờ đo validation dày hơn. Mốc thắng của các lần cộng dồn là bước 3000, mà
# với nhịp 4000 thì bước đó CHƯA TỪNG được đo. Hai dòng này giữ nhịp đo 500 và để
# accumulate_steps = 1, nên hiệu so với đợt 1 là tác dụng của nhịp đo một mình:
#   - dòng frozen (rẻ, ~5 phút) so với 0,5852 của dropout=0.5 nhịp 2000.
#   - dòng finetune (~18 phút) so với 0,5874 của lần cộng dồn nhịp 500.
# huber_delta 0.3: dưới 0,3 log là sai số thí nghiệm (Landrum & Riniker 2024,
# DOI 10.1021/acs.jcim.4c00049) nên phạt bình phương, trên mức đó phạt tuyến tính.
# Là nút DUY NHẤT ở đây có nguồn, nên giữ lại để thử ghép với dropout.
# model.dropout 0.6 / 0.7: CHƯA CÓ NGUỒN, nối tiếp thang 0,1 -> 0,3 -> 0,5 đợt 1.
# eval_every_steps 1000 / 500: CHƯA CÓ NGUỒN, chỉ dùng để tách confound nói trên.
TRAIN_RUNS=(
    "frozen direct off --set model.dropout=0.6"
    "frozen direct off --set model.dropout=0.7"
    "frozen direct off --set model.dropout=0.5 --set huber_delta=0.3"
    "frozen direct off --set model.dropout=0.6 --set huber_delta=0.3"
    "frozen direct off --set model.dropout=0.5 --set eval_every_steps=1000"
    "finetune direct off $FINETUNE_SET --set eval_every_steps=500"
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
# Chỉ hai file ĐẦU VÀO là bắt buộc phải có sẵn: chúng nằm trong data/, bị gitignore, nên
# phải chép tay sang server. Bốn file cache trong work/ KHÔNG kiểm ở đây vì chính bước
# prepare dựng ra chúng — đòi chúng tại đây thì script chết trước khi kịp chạy prepare.
for required in \
    data/dataset_v4/evaluator_training.parquet \
    data/dataset_v4/full/proteins.parquet; do
    [[ -f "$required" ]] || fail "thiếu $required (chép tay sang server, data/ bị gitignore)"
done
echo "dữ liệu đầu vào: đủ 2 file; cache sẽ dựng ở bước prepare"

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

# Cache fingerprint + embedding protein nằm trong work/, mà work/ bị gitignore: sau khi
# `git pull` về server thì cache CHƯA CÓ (hoặc còn là cache của dataset cũ). Dựng lại ở
# đây để một lệnh là đủ. Hỏng bước này thì mọi lần train sau đều hỏng, nên dừng hẳn.
if ! run_step "prepare (cache fingerprint + ESM2)" python run_evaluator.py --mode prepare
then
    print_summary
    fail "prepare lỗi, không train"
fi

run_step "baseline" python run_evaluator.py --mode baseline

declare -A smoke_ok=()
for configuration in "${SMOKE_CONFIGS[@]}"; do
    # Smoke của finetune phải chạy với ĐÚNG trần VRAM mà lần train của nó sẽ dùng: giữ
    # trần 9 GiB ở đây thì smoke hết bộ nhớ, script coi là finetune hỏng và bỏ luôn cả
    # hai lần train finetune — mất cả đêm vì một con số không liên quan.
    smoke_extra=""
    if [[ "$configuration" == "finetune" ]]; then
        smoke_extra="$FINETUNE_SET"
    fi
    # shellcheck disable=SC2086
    if run_step "smoke $configuration" \
        python run_evaluator.py --mode smoke --config "$configuration" $smoke_extra; then
        smoke_ok[$configuration]=1
    fi
done

for entry in "${TRAIN_RUNS[@]}"; do
    read -r configuration head penalty extra <<<"$entry"
    penalty="${penalty:-off}"
    label="train $configuration $head sharpness=$penalty${extra:+ $extra}"
    if [[ " ${SMOKE_CONFIGS[*]} " == *" $configuration "* && -z "${smoke_ok[$configuration]:-}" ]]
    then
        summary+=("$label: BỎ QUA (smoke $configuration lỗi)")
        continue
    fi
    # $extra KHÔNG được bọc nháy: nó phải tách thành nhiều tham số --set riêng biệt.
    # shellcheck disable=SC2086
    run_step "$label" \
        python run_evaluator.py --mode train --config "$configuration" --head "$head" \
        --sharpness "$penalty" $extra
done

print_summary
echo "=== [$(date '+%F %T')] HẾT"
rm -f "$cache/$LOG_NAME.pid"
