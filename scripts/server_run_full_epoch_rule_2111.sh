#!/usr/bin/env bash
# FULL no_protein under the EPOCH rule: the comparison t12full could not make.
# Launch from the evaluator environment on a server GPU selected by its owner:
#   setsid nohup bash scripts/server_run_full_epoch_rule_2111.sh <GPU> >/dev/null 2>&1 &
# Follow runs/full_epoch_rule_2111_gpu<GPU>.log.
#
# WHY A SECOND FULL RUN. t12full (2026-11-21) selected its checkpoint at epoch 0 step 2000
# and stopped at step 12,000, 93 % of its first epoch, scoring val/strict 0.54545 against
# t9c1's 0.55057. It stopped because of the measurement SCHEDULE, not because of the data:
#
#   train.py:1671  patience = early_stopping_patience_evals (5) when eval_every_steps > 0,
#                  else early_stopping_patience (2)
#   train.py:1249  `step` is reset at the start of every epoch
#   train.py:1872  consider(epoch, None, loss) runs at the end of every COMPLETED epoch
#
# One `since_best` counter serves both paths. On SUB100, 196 steps per epoch never reach
# the 2000-step mark, so consider() is called once per epoch and patience 5 means FIVE
# EPOCHS without improvement - which is what t9c1 got. On FULL, 12,945 steps per epoch mean
# five intra-epoch marks were enough to exhaust the same counter inside epoch 0. The two
# runs were therefore scored under different rules.
#
# THE RULE THIS RUN USES. eval_every_steps=0 turns the intra-epoch marks off, so both runs
# select at the end of an epoch and get ONE chance per pass over the training set;
# early_stopping_patience=5 is then set explicitly, because the eval_every=0 branch reads
# that setting and its default is 2, not 5. Everything else is t9c1's reference flag set.
#
#   THE CONTROL IS t9c1, UNCHANGED AND NOT RERUN: on SUB100 the intra-epoch marks never
#   fired, so t9c1 already ran under exactly this rule. val/strict 0.55057 | val/all
#   0.54915 | train/all 0.62558 | gap 0.07643, same seed 20261023, same validation rows.
#
# WHAT IT ANSWERS AND WHAT IT DOES NOT. It answers: at the same epoch budget and the same
# selection rule, what does enlarging the training set 66-fold do to the ranking score? It
# does NOT answer whether the schedule difference alone explains t12full's -0.00512 - that
# would need t12full's schedule applied to SUB100, which its 196-step epochs cannot carry.
# And it does not look for a peak before step 2000: see the second run below, whose marks
# are deliberately kept out of the comparison.
#
# COST. Up to 10 epochs at about 26 minutes per epoch, measured from t12full's own log
# (2000 steps + one validation pass + one 200,000-row train measurement took 243 s, and an
# epoch is 12,945 steps), so up to about 4 h 20 for the main run, less if it stops earlier.
# The observation run adds roughly one more epoch.

set -Eeuo pipefail
if [[ $# != 1 || ! $1 =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "Usage: bash scripts/server_run_full_epoch_rule_2111.sh <GPU index>" >&2
  exit 2
fi
gpu=$1
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p runs
source scripts/gpu_batch_lock.sh
claim_gpu "$gpu" "$PWD/.cache" full_epoch_rule_2111 || exit $?
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1

log="runs/full_epoch_rule_2111_gpu${gpu}.log"
stamp=$(date +%Y%m%d%H%M%S)
if [[ -s "$log" ]]; then
  mv -- "$log" "$log.$stamp-$$"
fi
exec >>"$log" 2>&1
trap 'echo "FAILED (exit $?) at line $LINENO: $BASH_COMMAND" >&2' ERR
# claim_gpu is silent on success and runs BEFORE this redirection on purpose: a run that is
# refused the lock must not write into the log of the batch already holding it.
echo "=== $(date +%F' '%T) GPU $gpu claimed by pid $$ ==="

task="t13full$stamp"
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
# The epoch rule. patience=5 is written out because the eval_every=0 branch of train.py:1671
# reads early_stopping_patience, whose default is 2 - leaving it out would change the rule.
epoch_rule=(--set eval_every_steps=0 --set early_stopping_patience=5)

run() {
  echo "+ $*"
  "$@"
}

# ---- the comparison itself -------------------------------------------------------------
# The cap applies to the end-of-epoch measurements only; the final eval covers every train
# row. The 200,000-row sample is NOT comparable to t9c1's train/all, which was measured on
# the whole 100,000-row training set: a sparse sample of 6,627,559 rows gives sparser
# ranking queries. The eval below is the comparable train-side number.
run python run_evaluator.py --mode train --config no_protein --head direct --run-id "$task-$$" \
  "${shared[@]}" "${reference[@]}" "${epoch_rule[@]}" \
  --set measure_train_metrics=true --set measure_train_max_rows=200000
checkpoint=$(python scripts/pick_train_checkpoint.py runs/evaluator "$task-$$")
run python run_evaluator.py --mode eval --partition train --checkpoint "$checkpoint" \
  --run-id "$task-eval-$$" "${shared[@]}"

# ---- observation only, NOT part of the comparison --------------------------------------
# t12full's best point was its FIRST mark, so the real optimum may lie before step 2000 and
# nobody has measured there. This run looks, with marks every 500 steps over ONE epoch. Its
# own selection and stopping are therefore on a different schedule from the run above, so
# its score must never be quoted beside t9c1 - only its CURVE is being read. The train-side
# measurement is off here, because the curve does not need it and it doubles the cost.
run python run_evaluator.py --mode train --config no_protein --head direct \
  --run-id "$task-curve-$$" "${shared[@]}" "${reference[@]}" \
  --set eval_every_steps=500 --set epochs=1 --set early_stopping_patience_evals=100 \
  --set measure_train_metrics=false

run python scripts/figures_for_batch.py runs/evaluator "$log" --slice all --train-slice all
run python scripts/diagnostics_tables.py runs/evaluator "$log" --slice all --train-slice all
run python scripts/diagnostics_tables.py runs/evaluator "$log" --slice strict --train-slice all
echo "FINISHED: $checkpoint"
