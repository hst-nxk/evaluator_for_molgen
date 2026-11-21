#!/usr/bin/env bash
# Train the t9c1 reference architecture on the full native protein-cluster split.
# Launch from the evaluator environment on a server GPU selected by its owner:
#   setsid nohup bash scripts/server_run_full_no_protein_2111.sh <GPU> >/dev/null 2>&1 &
# Follow runs/full_no_protein_2111_gpu<GPU>.log.
#
# THE CONTROL. t9c1 - the same architecture, the same flags, the same seed 20261023, on
# SUB100 instead of the full train partition - measured, on the SAME validation rows:
#
#   val/strict 0.55057 | val/all 0.54915 | train/all 0.62558 | gap (train-val, all) 0.07643
#
# The validation set is shared because work/splits/train_subset_c200_r500.parquet subsets
# the TRAIN partition only ("validation giu nguyen" in its own manifest, 1.51 % of
# 6,627,559 train rows), so this run and t9c1 differ in the training rows and in nothing
# else.
#
# SIZE, NOT TIME. At batch size 512 the full partition gives about 12,945 batches per
# epoch against SUB100's 196. That factor of 66 describes batches per epoch and says
# nothing about wall-clock: the number of validation passes and where early stopping lands
# both move the total. For an ETA, take the real time to the FIRST validation mark in this
# log and scale from that, rather than from any figure written here in advance.
#
# ONE SEED - WHAT THAT DOES AND DOES NOT ALLOW. The FULL - SUB100 difference at this seed
# is measured directly, on the same validation rows, so it is a real paired difference and
# not an estimate. What is unknown is whether that difference holds across seeds. The
# 0.00046 standard deviation quoted elsewhere describes the three SUB100 runs; use it as
# context for how much one SUB100 number moves, and do NOT turn 0.001 into a threshold
# that the FULL - SUB100 difference has to clear.
#
# WHAT THIS RUN IS AND IS NOT. It keeps t9c1's reference hyper-parameters in order to
# measure what enlarging the training set does UNDER ONE TRAINING RULE. Note that the same
# epoch ceiling still gives FULL far more update steps. It therefore does not establish the
# best configuration on FULL, and it does not test whether dropout 0.5 - load-bearing at
# 100,000 rows, per the 2x2 of 2026-11-21 - is too strong at this scale.

set -Eeuo pipefail
if [[ $# != 1 || ! $1 =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "Usage: bash scripts/server_run_full_no_protein_2111.sh <GPU index>" >&2
  exit 2
fi
gpu=$1
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p runs
source scripts/gpu_batch_lock.sh
claim_gpu "$gpu" "$PWD/.cache" full_no_protein_2111 || exit $?
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1

log="runs/full_no_protein_2111_gpu${gpu}.log"
stamp=$(date +%Y%m%d%H%M%S)
if [[ -s "$log" ]]; then
  mv -- "$log" "$log.$stamp-$$"
fi
exec >>"$log" 2>&1
trap 'echo "FAILED (exit $?) at line $LINENO: $BASH_COMMAND" >&2' ERR
# claim_gpu is silent on success, and it runs BEFORE this redirection on purpose: a run
# that is refused the lock must not write into the log of the batch already holding it.
# So the lock is recorded here instead, once it is ours and the old log has been moved.
echo "=== $(date +%F' '%T) GPU $gpu claimed by pid $$ ==="

# Keep the historical picker and table readers' <task>-<pid> convention.
task="t12full$stamp"
shared=(
  --set inputs.train_subset= --set inputs.partition_override= --set inputs.partition_override_pairs=
  --set device=cuda --set "gpu_memory_limit_mib=${MEMORY_LIMIT:-4500}" --set eval_batch_size=2048
  --set seed=20261023 --set selection_metric=pairwise_accuracy_macro --set selection_slice=strict
)
reference=(
  --set model.ligand_features=fingerprint --set model.fusion=concat
  --set model.head_block=residual --set model.residual_blocks=7 --set model.residual_width=102
  --set model.residual_hidden_factor=1.849 --set model.residual_hidden_dropout=0.3936
  --set model.residual_dropout=0.0 --set model.dropout=0.5
  --set learning_rate=9.86e-05 --set weight_decay=0.0
  --set min_edit_rows=1 --set augment_reverse=false --set antisymmetry_penalty=0.0
  --set alignment_weight=0.0
)

run() {
  echo "+ $*"
  "$@"
}

# The cap applies to intermediate measurements only. The final eval covers every train row.
run python run_evaluator.py --mode train --config no_protein --head direct --run-id "$task-$$" \
  "${shared[@]}" "${reference[@]}" --set measure_train_metrics=true --set measure_train_max_rows=200000
checkpoint=$(python scripts/pick_train_checkpoint.py runs/evaluator "$task-$$")
run python run_evaluator.py --mode eval --partition train --checkpoint "$checkpoint" \
  --run-id "$task-eval-$$" "${shared[@]}"

run python scripts/figures_for_batch.py runs/evaluator "$log" --slice all --train-slice all
run python scripts/diagnostics_tables.py runs/evaluator "$log" --slice all --train-slice all
run python scripts/diagnostics_tables.py runs/evaluator "$log" --slice strict --train-slice all
echo "FINISHED: $checkpoint"
