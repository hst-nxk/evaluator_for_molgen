#!/usr/bin/env bash
# FULL no_protein at two more seeds, under the epoch rule - the spread t13full lacks.
#   setsid nohup bash scripts/server_run_full_seeds_2211.sh <GPU> >/dev/null 2>&1 &
#   tail -F runs/full_seeds_2211_gpu<GPU>.log
#
# WHY. t13full measured FULL against t9c1 at seed 20261023 only: val/strict 0.54394 against
# 0.55057 (-0.00663) and val/all 0.56266 against 0.54915 (+0.01351). A single paired
# difference has no spread, so neither number can be defended. SUB100 already has three
# seeds, so two more FULL runs give three paired differences:
#
#   seed 20261023   t9c1  val/strict 0.55057      (t13full: 0.54394, measured)
#   seed 20261024   t9c2  val/strict 0.54994      (this batch, arm 1)
#   seed 20261025   t9c3  val/strict 0.55084      (this batch, arm 2)
#
# The project's bar: 3/3 seeds of one sign AND t >= 3, noise floor 0.0021. On val/all the
# measured difference is +0.0135, about six times that floor, so the question there is
# whether the sign holds; on val/strict it is -0.0066, three times the floor, same question.
#
# THE RULE IS t13full's, UNCHANGED: eval_every_steps=0 and early_stopping_patience=5, so
# selection happens at the end of an epoch and both sides get one chance per pass over the
# training set. train.py:1671 reads early_stopping_patience on this branch and its default
# is 2, so the 5 is written out. Nothing else differs from t13full but the seed.
#
# COST. t13full stopped after 7 epochs at 12.1 min per epoch, about 1 h 25 plus the
# train-partition eval, so roughly 1 h 45 per seed and about 3 h 30 for the batch. A run
# that improves for longer takes up to 10 epochs, i.e. at most about 2 h 15 each.

set -u -o pipefail

if [[ $# != 1 || ! $1 =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "a GPU index is required, for example: bash scripts/server_run_full_seeds_2211.sh 3" >&2
  exit 2
fi
GPU="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/runs/full_seeds_2211_gpu$GPU.log"
FAILED=0
LAST_STATUS=1
mkdir -p "$ROOT/runs" "$ROOT/.cache"
export PYTHONUNBUFFERED=1

source "$ROOT/scripts/gpu_batch_lock.sh"
claim_gpu "$GPU" "$ROOT/.cache" "full_seeds_2211" || exit $?
# The cap is not a reservation: on 2026-11-21 this batch was given 4.39 GiB on a card with
# 20 MiB actually free and died at its first backward pass, 30 s and two W&B runs later.
# Checked AFTER the lock, so two batches cannot both read "enough free" and then start.
require_free_vram "$GPU" "${MEMORY_LIMIT:-4500}" || exit $?
export CUDA_VISIBLE_DEVICES="$GPU"

# One log holds one batch: the figure and table scripts refuse a log with two pids
# (check_diagnostics_manifests.pid_from_log). The old log is kept, with the pid in its name,
# and the move is checked - carrying on would append to a file we just said was saved.
if [[ -s "$LOG" ]]; then
  KEPT="$LOG.$(date +%Y%m%d_%H%M%S)-$$"
  if ! mv "$LOG" "$KEPT"; then
    echo "could not move the previous log aside ($LOG -> $KEPT); not writing over it" >&2
    exit 4
  fi
  echo "the previous log of this GPU was kept as $KEPT" | tee -a "$LOG"
fi

say() { echo "" | tee -a "$LOG"; echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }

run() {
  echo "+ $*" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  LAST_STATUS=$?
  if [[ $LAST_STATUS -ne 0 ]]; then
    FAILED=$((FAILED + 1))
    echo "!! FAILED (code $LAST_STATUS) at: $*" | tee -a "$LOG"
  fi
  return 0
}

SHARED="--set inputs.train_subset= --set inputs.partition_override="
SHARED+=" --set inputs.partition_override_pairs= --set device=cuda"
SHARED+=" --set gpu_memory_limit_mib=${MEMORY_LIMIT:-4500} --set eval_batch_size=2048"
SHARED+=" --set selection_metric=pairwise_accuracy_macro --set selection_slice=strict"
REFERENCE="--set model.ligand_features=fingerprint --set model.fusion=concat"
REFERENCE+=" --set model.head_block=residual --set model.residual_blocks=7"
REFERENCE+=" --set model.residual_width=102 --set model.residual_hidden_factor=1.849"
REFERENCE+=" --set model.residual_hidden_dropout=0.3936 --set model.residual_dropout=0.0"
REFERENCE+=" --set model.dropout=0.5 --set learning_rate=9.86e-05 --set weight_decay=0.0"
REFERENCE+=" --set min_edit_rows=1 --set augment_reverse=false"
REFERENCE+=" --set antisymmetry_penalty=0.0 --set alignment_weight=0.0"
EPOCH_RULE="--set eval_every_steps=0 --set early_stopping_patience=5"

SEEDS=(20261024 20261025)
stamp=$(date +%Y%m%d%H%M%S)

say "FULL no_protein, epoch rule, seeds ${SEEDS[*]}, pid $$"
echo "control per seed: t9c2 val/strict 0.54994 (20261024), t9c3 0.55084 (20261025)" \
  | tee -a "$LOG"

for position in 0 1; do
  seed="${SEEDS[$position]}"
  label="t14full$stamp$((position + 2))"
  say "$label - training (no_protein, FULL, seed $seed)"
  run python run_evaluator.py --mode train --config no_protein --head direct \
    --run-id "$label-$$" $SHARED $REFERENCE $EPOCH_RULE --set "seed=$seed" \
    --set measure_train_metrics=true --set measure_train_max_rows=200000
  trained=$LAST_STATUS
  checkpoint=""
  if [[ $trained -eq 0 ]]; then
    checkpoint="$(python "$ROOT/scripts/pick_train_checkpoint.py" "$ROOT/runs/evaluator" \
      "$label-$$" 2>> "$LOG" || true)"
    if [[ -z "$checkpoint" ]]; then
      # Trained and returned 0, yet nothing scorable: the train half of this seed is lost,
      # and that is a failure of the batch even though the command succeeded.
      FAILED=$((FAILED + 1))
      echo "!! $label trained but produced NO usable checkpoint" | tee -a "$LOG"
    fi
  else
    echo "!! $label has no checkpoint because its training run failed" | tee -a "$LOG"
  fi
  if [[ -n "$checkpoint" ]]; then
    # The 200,000-row measurement taken during training is NOT comparable to t9c's
    # train/all: on 6,627,559 rows a sparse sample inflates the ranking macro (0.77484
    # against 0.62818 for the same t12full checkpoint). This full pass is the comparable one.
    say "$label - the SAME checkpoint scored on the whole TRAIN partition"
    run python run_evaluator.py --mode eval --partition train --checkpoint "$checkpoint" \
      --run-id "$label-eval-$$" $SHARED --set "seed=$seed"
  fi
done

say "figures and tables"
run python scripts/figures_for_batch.py runs/evaluator "$LOG" --slice all --train-slice all
run python scripts/diagnostics_tables.py runs/evaluator "$LOG" --slice all --train-slice all
run python scripts/diagnostics_tables.py runs/evaluator "$LOG" --slice strict --train-slice all

say "FINISHED with $FAILED failed command(s)"
echo "read the three paired differences with t9c1 0.55057 / t9c2 0.54994 / t9c3 0.55084" \
  | tee -a "$LOG"
[[ $FAILED -ne 0 ]] && exit 1
exit 0
