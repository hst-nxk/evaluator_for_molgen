#!/usr/bin/env bash
#
# WHICH dropout holds SUB100 back - the encoders' or the head's? A 2x2 at the extremes.
#
#   setsid nohup bash scripts/server_run_dropout_2111.sh 6 > /dev/null 2>&1 &
#   tail -F runs/dropout_2111_gpu6.log
#
# WHY. On SUB100, one checkpoint and one split, read at the SAME slice on both sides:
# train/all pairwise macro 0.6376 against validation/all 0.5482, a gap of 0.0894 (MAE
# 0.5662 against 0.5887). That reading is from t9a, the frozen (with-ESM2) arm of the
# 2026-11-20 batch; train/strict holds 0 rows on this split, so `all` is the only slice
# both partitions have. The gap is real and ITS CAUSE IS UNKNOWN. t2d cannot settle it: it
# turned dropout off AND changed the budget and the stopping rule, so it moved three
# things at once.
#
# WHAT `model.dropout` ACTUALLY DOES - the mistake this script's first version made. With
# `residual_hidden_dropout` and `residual_dropout` both passed explicitly (as the whole
# reference flag set does), models.py:577 gives the head those two values and
# `model.dropout` never reaches it. Building the model at both ends of a model.dropout
# sweep shows it:
#
#   model.dropout=0.5 -> ligand_encoder [0.5], edit_encoder [0.5],
#                        predictor [0.3936, 0.0] x 7 blocks
#   model.dropout=0.0 -> predictor [0.3936, 0.0] x 7 blocks
#
# So `model.dropout` is the ENCODERS' dropout, and the head's regularisation is
# `residual_hidden_dropout` (0.3936, in all seven blocks). A sweep over model.dropout
# alone, called a head sweep, would have answered a question nobody asked. Caught in
# review on 2026-11-21 before any GPU was spent.
#
# THE DESIGN. Two factors, each at its reference value and at zero, which is a 2x2 and not
# a line:
#
#   arm a   encoders 0.5   head 0.3936    the reference - must reproduce t9c
#   arm b   encoders 0.5   head 0.0       the head's dropout alone
#   arm c   encoders 0.0   head 0.3936    the encoders' dropout alone
#   arm d   encoders 0.0   head 0.0       both, for the interaction
#
# Three seeds each: twelve runs. The extremes are deliberate: the question is whether
# regularisation explains a gap of 0.0894 at all, and the largest contrast gives that the
# best chance of showing. If an arm moves, the follow-up batch is a line through that one
# factor to find where it is best; if nothing moves, the gap belongs to something neither
# knob touches and the next experiment is a different one entirely. Everything else is the
# reference setting, including the EPOCH BUDGET (config.json's, not overridden) and the
# checkpoint rule - t2d's failure mode is the one being avoided.
#
# THE BASELINE THIS BATCH IS JUDGED AGAINST is t9c's, read off its own manifests on the
# CPU on 2026-11-21 (the six manifests of batch 1073224, each eval/train half confirmed to
# carry the checkpoint of its own train run):
#
#   seed        train/all   val/all   gap       val/strict
#   20261023    0.62558     0.54915   +0.07643  0.55057
#   20261024    0.61543     0.54877   +0.06665  0.54994
#   20261025    0.62803     0.54968   +0.07835  0.55084
#   mean        0.62301     0.54920   +0.07381  0.55045
#
#   MAE at `all`: 0.57223 train, 0.58824 validation.
#
# The gap to beat is therefore 0.07381, NOT the 0.0894 quoted above - that one belongs to
# t9a, which has the ESM2 branch this batch does not, so t9a is NOT this batch's control.
#
# HOW TO READ IT.
#   1. Arm a against the three validation/strict values above, SEED BY SEED. If it does not
#      reproduce them, something else changed and no other arm means anything.
#   2. Per seed, from val/all and from the gap:
#        b - a            the head's dropout
#        c - a            the encoders' dropout
#        d - c - b + a    the interaction
#      Seed by seed, because three paired differences carry their own spread and a
#      difference of two means does not.
#   3. Read val/all AND the gap, not the gap alone: a gap also narrows when the TRAIN score
#      falls, which is the model getting worse rather than generalising. Validation rising
#      = that dropout was throttling the model. Train rising with validation flat =
#      memorisation, i.e. the gap is generalisation and not capacity.
#
# WHY no_protein. The 2026-11-20 batch found the ESM2 branch changes nothing on this split
# (+0.0005, signs mixed), so dropping it removes a branch from the comparison and makes
# every run cheaper.
#
# COST. Twelve training runs on SUB100 (100,000 train rows), each followed by a second
# pass that scores the SAME checkpoint on the train partition. About 16 minutes per run on
# 2026-11-20, so roughly 3 hours 15 minutes.

set -u -o pipefail

if [[ $# -lt 1 ]]; then
  echo "a GPU index is required, for example: bash scripts/server_run_dropout_2111.sh 6" >&2
  exit 2
fi
GPU="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG="$ROOT/runs/dropout_2111_gpu$GPU.log"
SUB_SPLIT="work/splits/train_subset_c200_r500.parquet"
FAILED=0

mkdir -p "$ROOT/runs" "$ROOT/.cache"
export PYTHONUNBUFFERED=1

say() { echo "" | tee -a "$LOG"; echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }

run() {
  echo "+ $*" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  local status=$?
  LAST_STATUS=$status
  if [[ $status -ne 0 ]]; then
    FAILED=$((FAILED + 1))
    echo "!! FAILED (code $status) at: $*" | tee -a "$LOG"
  fi
  return 0
}

source "$ROOT/scripts/gpu_batch_lock.sh"
claim_gpu "$GPU" "$ROOT/.cache" "dropout_2111" || exit $?
export CUDA_VISIBLE_DEVICES="$GPU"

# ONE LOG HOLDS ONE BATCH. The log name is fixed per GPU and every write appends, so a
# second run on the same GPU would leave two pids in one file - and both the figure script
# and the table script refuse a log with two pids (check_diagnostics_manifests.pid_from_log)
# rather than attribute a task to the wrong batch. So the previous log is moved aside here:
# after the lock is taken, before the first line is written, and kept rather than deleted.
if [[ -s "$LOG" ]]; then
  # The pid is in the kept name as well as the second: two batches started inside the same
  # second would otherwise overwrite each other's kept log. And `mv` is CHECKED - if it
  # fails (a full disk, a read-only runs/), carrying on would append this batch's lines to
  # the old file after announcing it had been saved, and both readers would then refuse the
  # log for holding two pids. There is nothing to salvage from that, so it stops here,
  # before any GPU time is spent.
  KEPT="$LOG.$(date +%Y%m%d_%H%M%S)-$$"
  if ! mv "$LOG" "$KEPT"; then
    echo "could not move the previous log aside ($LOG -> $KEPT); not writing over it" >&2
    exit 4
  fi
  echo "the previous log of this GPU was kept as $KEPT" | tee -a "$LOG"
fi

if [[ ! -f "$SUB_SPLIT" ]]; then
  echo "missing $SUB_SPLIT; this batch has nothing to run" | tee -a "$LOG"
  exit 3
fi

# THE REFERENCE FLAG SET, minus the two flags this batch varies. Neither `model.dropout`
# nor `model.residual_hidden_dropout` is here, on purpose: both are set per arm below, and
# a flag left in this shared string would be set twice, with the later one silently
# winning. `residual_dropout` IS here because it stays 0.0 in every arm.
SETTINGS="--set selection_metric=pairwise_accuracy_macro --set selection_slice=strict"
SETTINGS+=" --set gpu_memory_limit_mib=${MEMORY_LIMIT:-4500} --set eval_batch_size=2048"
SETTINGS+=" --set learning_rate=9.86e-05 --set weight_decay=0.0"
RESIDUAL="--set model.head_block=residual --set model.residual_blocks=7"
RESIDUAL+=" --set model.residual_width=102 --set model.residual_hidden_factor=1.849"
RESIDUAL+=" --set model.residual_dropout=0.0"
# measure_train_metrics: a forward pass over the train rows at every measurement, so the
# learning curve carries two lines of the SAME metric. The reading this batch exists to
# explain is a train-versus-validation gap, so the curve of both sides is the point.
MEASURE="--set measure_train_metrics=true"

SEEDS=(20261023 20261024 20261025)
# arm -> (encoder dropout, head hidden dropout). Arm a is the reference and reproduces t9c.
ARMS=(a b c d)
ENCODER=(0.5 0.5 0.0 0.0)
HEAD=(0.3936 0.0 0.3936 0.0)
RUN_ID=""
LAST_STATUS=1

new_run_id() { RUN_ID="$1-$$"; }

checkpoint_of_last_train() {
  if [[ ${LAST_STATUS:-1} -ne 0 ]]; then
    return 0
  fi
  # `|| true` keeps one dead picker from ending the batch, but it also swallows the exit
  # code, so the CALLER counts an empty answer as a failure (see train_and_score). A train
  # run that succeeded and then produced no scorable checkpoint costs the gap of that seed,
  # which is the number this whole batch exists to measure, so "0 failed commands" must
  # not be printed over it.
  python "$ROOT/scripts/pick_train_checkpoint.py" "$ROOT/runs/evaluator" "$RUN_ID" \
    2>> "$LOG" || true
}

# One training run, then the SAME checkpoint scored on the train partition. The gap has to
# come from one checkpoint on one split, and the selected checkpoint is usually not the end
# of the last epoch, so the train side cannot be read off the training loss.
train_and_score() {
  local label="$1"
  shift
  say "$label - training (no_protein, SUB100)"
  new_run_id "$label"
  run python run_evaluator.py --mode train --run-id "$RUN_ID" --config no_protein \
    --head direct $SETTINGS $RESIDUAL "$@"
  local checkpoint
  checkpoint="$(checkpoint_of_last_train)"
  if [[ -z "$checkpoint" && ${LAST_STATUS:-1} -eq 0 ]]; then
    # The training run itself reported success, so this is the picker or a missing file:
    # one of the twelve gaps cannot be computed, and that IS a failure of the batch. The
    # other arms are independent, so they go on.
    FAILED=$((FAILED + 1))
    echo "!! $label trained but produced NO usable checkpoint; its train half is skipped" \
      | tee -a "$LOG"
    return 0
  fi
  if [[ -z "$checkpoint" ]]; then
    # Its training run failed above and was counted there; not counted twice.
    echo "!! $label has no checkpoint because its training run failed" \
      | tee -a "$LOG"
    return 0
  fi
  say "$label - the SAME checkpoint scored on TRAIN"
  new_run_id "$label-eval"
  run python run_evaluator.py --mode eval --run-id "$RUN_ID" --checkpoint "$checkpoint" \
    --partition train $SETTINGS "$@"
}

say "dropout 2x2 on SUB100, 4 arms x 3 seeds, pid $$"
echo "encoders (model.dropout) x head (model.residual_hidden_dropout), each at its" \
  | tee -a "$LOG"
echo "reference value and at 0; arm a is the reference and must reproduce t9c" \
  | tee -a "$LOG"

for index in 0 1 2 3; do
  arm="${ARMS[$index]}"
  encoder="${ENCODER[$index]}"
  head="${HEAD[$index]}"
  say "arm $arm - encoders $encoder, head $head, three seeds"
  for position in 0 1 2; do
    train_and_score "t10$arm$((position + 1))" \
      --set "inputs.train_subset=$SUB_SPLIT" \
      --set "seed=${SEEDS[$position]}" \
      --set "model.dropout=$encoder" \
      --set "model.residual_hidden_dropout=$head" $MEASURE
  done
done

say "figures and tables"
# Two readings, on purpose. `all`/`all` is the gap: train/strict holds 0 rows on this
# split, so a strict train column would be empty and a gap read across two slices is not a
# gap. `strict` is how t9c was reported, and arm a has to reproduce that number, so the
# comparable table is printed as well.
run python scripts/figures_for_batch.py runs/evaluator "$LOG"
run python scripts/diagnostics_tables.py runs/evaluator "$LOG" --slice all --train-slice all
run python scripts/diagnostics_tables.py runs/evaluator "$LOG"

say "FINISHED with $FAILED failed command(s)"
echo "read with:" | tee -a "$LOG"
echo "  grep -n 'FAILED\\|SKIPPED\\|NO figure' $LOG" | tee -a "$LOG"
echo "  sed -n '/Every run of this batch/,\$p' $LOG" | tee -a "$LOG"
[[ $FAILED -ne 0 ]] && exit 1
exit 0
