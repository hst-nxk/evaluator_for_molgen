#!/usr/bin/env bash
# Đợt 14/11/2026 (đợt 2, chạy qua đêm): mang VIỆC 4 sang MoLFormer, bỏ hẳn nhánh protein
# (hệ quả của việc 9), tách phần đóng góp của twin_edit (việc 12), và kiểm độ bền của
# trọng số phạt.
# 18 lần chạy, tất cả trên TẬP CON 100.000 dòng, chọn checkpoint theo lát `strict`.
# Ước lượng 3,5 tiếng nếu GPU rảnh; trên GPU đang chia với job khác thì 5-6 tiếng.
#
# Năm phần, xếp theo thứ tự QUAN TRỌNG GIẢM DẦN, nên kill giữa đêm chỉ mất phần cuối:
#   1. đối chứng MoLFormer                     3 lần  -> phải ra lại 0,5561 +/- 0,0006
#   2. MoLFormer + phạt w = 1,0                3 lần  -> việc 4 có vào paper hay không
#   3. no_protein (bỏ nhánh protein)           3 lần  -> hệ quả trực tiếp của việc 9
#   4. head twin trơn (không twin_edit)        3 lần  -> tách phần đóng góp, việc 12
#   5. phạt w = 0,3 và w = 3,0 trên Morgan     6 lần  -> độ bền của trọng số
#
# Cách chạy (trên server Brev, trong shell đã `conda activate nxk`):
#   cd /data/khainx/evaluator_for_molgen
#   setsid nohup bash scripts/server_run_penalty_molformer_1411.sh 4 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/penalty_molformer_1411.log
#
# Tham số duy nhất là CHỈ SỐ GPU (bắt buộc, không tự dò).
#
# Dừng giữa chừng:
#   kill -- -$(cat .cache/penalty_molformer_1411.pid)
#
# ------------------------------------------------------------------------------------
# VÌ SAO CÓ ĐỢT NÀY
#
# Đợt 1 (scripts/server_run_tasks34_1411.sh) đã đo xong trên Morgan, tập con 100k,
# ba hạt giống 20261023/24/25, lát `strict`:
#     đối chứng                                0,5499 +/- 0,0007
#     việc 3a pair_features=difference         0,5500 +/- 0,0022   (t = 0,08)
#     việc 3b thêm cả tích                     0,5509 +/- 0,0016   (t = 0,95)
#     việc 4 antisymmetry_penalty=1.0          0,5541 +/- 0,0002   (t = 10,48, 3/3 hạt)
#     việc 5 head antisymmetric                0,5533 +/- 0,0018   (t = 3,04, 3/3 hạt)
#     việc 2 augment đảo chiều                 0,5500 +/- 0,0055   (tính tay, xem ghi chú)
# Mốc nhiễu hạt giống của dự án 0,0021. Ngưỡng chứng minh: 3/3 hạt cùng dấu VÀ t >= 3.
# => Việc 3 KHÔNG có tác dụng. Việc 4 có, và là hiệu lớn nhất đo được từ một thay đổi
#    không đụng vào cách biểu diễn đầu vào.
#
# NHƯNG số đó mới là Morgan. Trên CÙNG tập con 100k, MoLFormer đạt 0,5561 +/- 0,0006,
# tức cách biểu diễn cho +0,0062 còn phạt cho +0,0041. Hai đường có thể trùng nhau: phạt
# chỉ đang bù một khuyết điểm mà MoLFormer không có. Đêm 13/11 đã có tiền lệ đúng kiểu
# này — các khoản lợi từ kiến trúc biến mất khi đổi sang MoLFormer. Vì vậy:
#
#   PHẦN 1 + 2 là câu hỏi quyết định: phạt còn tác dụng trên MoLFormer không?
#     - Nếu CÒN (3/3 hạt, t >= 3 so với phần 1) -> đây là kết quả đi vào paper cho việc 4.
#     - Nếu MẤT -> phải viết rõ trong báo cáo: phạt chỉ là cách vá biểu diễn Morgan,
#       không phải phát hiện về tính phản đối xứng. Kết quả âm này vẫn phải báo.
#
#   PHẦN 3 là hệ quả của việc 9, đo đêm 13/11: mô hình gần như không đọc protein (lát (a)
#     sd dự đoán giữa các Context P 0,0114 so với sd nhãn 0,6566; lát (b) đúng chiều trong
#     một protein 0,5605-0,5690, đúng bằng mức trên toàn validation). Bỏ hẳn nhánh protein
#     mà điểm không giảm -> bằng chứng mạnh nhất cho phần phân tích lỗi. Điểm GIẢM rõ ->
#     protein có đóng góp và phải viết lại cách diễn giải hai lát của việc 9.
#
#   PHẦN 4 tách phần đóng góp của twin_edit (việc 12). Bảng 13/11 có twin + twin_edit=True
#     0,5517 +/- 0,0005 (t = 3,71), nhưng nhóm đó đổi HAI thứ cùng lúc so với đối chứng:
#     head twin và twin_edit. Thiếu đúng nhóm twin trơn 3 hạt trên cùng tập con.
#
#   PHẦN 5 hỏi hiệu của phạt có phụ thuộc trọng số không? w = 0,3 và w = 3,0 trên Morgan.
#     Nếu cả ba mức đều cho hiệu dương thì kết luận bền; nếu chỉ w = 1,0 ăn thì phải nói
#     đó là một mức đã dò, không phải một tính chất.
#     Phần 5 là 6 lần chạy, xếp CUỐI nên kill giữa đêm chỉ mất phần này.
#
# ------------------------------------------------------------------------------------
# LƯU Ý ĐỌC SỐ
#   - Phần 1 là ĐỐI CHỨNG KHỚP cho MoLFormer. Mốc cũ 0,5561 +/- 0,0006 đo bằng code
#     trước khi có hai khóa mới; phần 1 chạy lại bằng code mới để hiệu của phần 2 đọc
#     được trong cùng một đợt. Nếu phần 1 lệch khỏi 0,5561 +/- 0,0006 thì dừng đọc.
#   - Phạt chạy HAI lượt forward mỗi batch, nên phần 2 chậm hơn phần 1 rõ rệt.
#   - Dòng "phạt phản đối xứng trung bình" in mỗi epoch là số đo LÚC TRAIN (còn dropout),
#     KHÔNG dùng để báo cáo tính phản đối xứng. Số báo cáo là dòng bất đối xứng ở cuối
#     mỗi lần chạy, đo ở chế độ eval.
#   - KHÔNG dùng partition test ở bất kỳ đâu (run_evaluator.py chặn sẵn).
#   - Con số việc 2 (0,5500 +/- 0,0055) hiện vẫn là số TÍNH TAY từ ba dòng đã làm tròn
#     4 chữ số của summarize_runs.py: ba lần chạy đó bị group_runs.py tách nhóm vì thứ tự
#     cờ --set khác nhau. `name_of` đã sửa (13/11) để sắp xếp lại các mảnh nhãn; chạy lại
#     group_runs.py sau đợt này sẽ cho số đầy đủ độ chính xác, và số đó mới được đưa vào
#     báo cáo.
set -u -o pipefail

PROJECT_ROOT="/data/khainx/evaluator_for_molgen"
ENV_NAME="nxk"
LOG_NAME="penalty_molformer_1411"
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

# Thiếu cờ này thì nhánh phân tử vẫn là Morgan và cả đợt vô nghĩa (train.py dòng 340 đọc
# ligand_features từ config đang sống, không từ model_settings của checkpoint).
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

# Khóa phạt phải CÓ trong config.json, nếu không `--set` báo lỗi ở phần 2 — tức sau khi
# đã đốt xong phần 1. Kiểm trước, mất một giây.
python - <<'PY' || exit 1
import json
import sys

config = json.load(open("config.json", encoding="utf-8"))
train = config["stage3"]["train"]
model = config["stage3"]["model"]
problems = []
if "antisymmetry_penalty" not in train:
    problems.append("thiếu stage3.train.antisymmetry_penalty")
elif float(train["antisymmetry_penalty"]) != 0.0:
    problems.append(f"antisymmetry_penalty = {train['antisymmetry_penalty']}, phải là 0")
if model.get("ligand_features") != "fingerprint":
    problems.append(f"model.ligand_features = {model.get('ligand_features')!r}, phải là"
                    " 'fingerprint' de phan 1 la doi chung dung")
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
echo "=== [$(date '+%F %T')] đợt phạt x MoLFormer bắt đầu | GPU $GPU_INDEX | PID $$"
echo "=== $root"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
echo "=== Tập con 100.000 dòng, lát strict, 3 hạt giống."
echo "=== Mốc MoLFormer 100k: 0,5561 +/- 0,0006. Phần 1 PHẢI ra lại mốc đó."
echo "=== Câu hỏi của đợt: phạt phản đối xứng (+0,0041 trên Morgan) còn sống trên"
echo "=== MoLFormer không? Mất thì phải báo kết quả âm, không bỏ im."

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
echo "=== PHẦN 1: ĐỐI CHỨNG MoLFormer, không phạt — phải ra lại 0,5561 +/- 0,0006"

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | MoLFormer, phạt=0 (đối chứng)" \
        --mode train --config frozen --head direct $COMMON $RESIDUAL $MOLFORMER \
        --set seed="$seed"
done

echo
echo "================================================================================"
echo "=== PHẦN 2: MoLFormer + phạt phản đối xứng w = 1,0 — CÂU HỎI CHÍNH CỦA ĐỢT"
echo "=== Hai lượt forward mỗi batch nên phần này chậm hơn phần 1."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | MoLFormer + antisymmetry_penalty=1.0" \
        --mode train --config frozen --head direct $COMMON $RESIDUAL $MOLFORMER \
        --set antisymmetry_penalty=1.0 --set seed="$seed"
done

echo
echo "================================================================================"
echo "=== PHẦN 3: BỎ HẲN NHÁNH PROTEIN (--config no_protein) — hệ quả của việc 9"
echo "=== Việc 9 đo được: mô hình gần như không đọc protein. Lát (a) sd dự đoán giữa các"
echo "=== Context P chỉ 0,0114 trong khi sd nhãn 0,6566; lát (b) đúng chiều TRONG một"
echo "=== protein 0,5605-0,5690, đúng bằng mức trên toàn validation. Nếu bỏ hẳn nhánh"
echo "=== protein mà điểm không giảm thì đó là bằng chứng mạnh nhất, và là số phải đưa"
echo "=== vào phần phân tích lỗi của báo cáo. Nếu điểm GIẢM rõ thì ngược lại: protein có"
echo "=== đóng góp, và hai lát của việc 9 phải viết lại cách diễn giải."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | no_protein (bỏ nhánh protein)" \
        --mode train --config no_protein --head direct $COMMON $RESIDUAL \
        --set seed="$seed"
done

echo
echo "================================================================================"
echo "=== PHẦN 4: head twin TRƠN (không twin_edit) — tách phần đóng góp, việc 12"
echo "=== Bảng 13/11 có nhóm twin + twin_edit=True: 0,5517 +/- 0,0005 (t = 3,71 so với"
echo "=== đối chứng 0,5499 +/- 0,0007). Nhưng nhóm đó đổi HAI thứ cùng lúc: head twin và"
echo "=== twin_edit. Trên tập con 100k không có nhóm twin trơn đủ 3 hạt để tách. Phần này"
echo "=== dựng đúng nhóm còn thiếu đó."

for seed in "${SEEDS[@]}"; do
    run_one "hạt giống $seed | head twin, twin_edit tắt" \
        --mode train --config frozen --head twin $COMMON $RESIDUAL --set seed="$seed"
done

echo
echo "================================================================================"
echo "=== PHẦN 5: độ bền của trọng số phạt trên Morgan (w = 0,3 và w = 3,0)"
echo "=== So với w = 1,0 đã đo: 0,5541 +/- 0,0002, và đối chứng 0,5499 +/- 0,0007."
echo "=== Phần này 6 lần chạy, xếp CUỐI: kill được mà không mất phần nào ở trên."

for weight in 0.3 3.0; do
    for seed in "${SEEDS[@]}"; do
        run_one "hạt giống $seed | Morgan + antisymmetry_penalty=$weight" \
            --mode train --config frozen --head direct $COMMON $RESIDUAL \
            --set antisymmetry_penalty="$weight" --set seed="$seed"
    done
done

echo
echo "=== [$(date '+%F %T')] xong tất cả | $total lần chạy | số lần lỗi: $failed"
echo "Đọc kết quả:"
echo "  python scripts/group_runs.py --slice strict --min-seeds 3 | tee runs/groups_penalty.log"
echo "Nếu có dòng CẢNH BÁO về khối metrics thì nhóm đó trộn hai thước đo, đừng đọc."
echo "  python scripts/summarize_runs.py --ranking --slice strict | tee runs/summary_penalty.log"
echo "XONG_PENALTY_MOLFORMER_1411"
