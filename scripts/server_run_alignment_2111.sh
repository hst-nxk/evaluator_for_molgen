#!/usr/bin/env bash
#
# Task 12 of the 2026-11-09 meeting: does a CLIP-style alignment step help?
#
#   setsid nohup bash scripts/server_run_alignment_2111.sh 2 > /dev/null 2>&1 &
#   tail -F runs/alignment_2111_gpu2.log
#
# WHAT THE MEETING ASKED. Minute 12: "F_A is the part cut out of A so it has to travel WITH
# A, F_B with B - they should not just be concatenated loosely", with the suggested route
# "a representation-space alignment step, CLIP-style, before predicting". The term is
# implemented in train.alignment_term: in the shared space, each molecule's projected token
# is pulled towards the projected token of ITS OWN fragment and pushed away from the other
# rows' fragments, symmetrically, at temperature 0.07 (Radford et al. 2021, arXiv:2103.00020).
#
# The positives are a MASK, not a diagonal. On this very split, with batch 512, 31.2 % of
# rows share their substituent with another row of the same batch (31,219 of 100,000 on the
# A side, 31,267 on the B side), so a diagonal loss would punish matching a molecule with
# the same fragment standing in another row - the pairing the hypothesis is about. Several
# cells per row may be positive, each row divided by its own count (train.positive_mask).
#
# THE CONTROL IS INSIDE THIS BATCH, and it has to be. The term needs a shared space, so
# every run here uses `fusion=projector`, which is NOT what the reference runs (t3a, t3c,
# t5r*, t9*) used - those are `fusion=concat`. A difference against them would therefore
# measure the projector and the alignment at once. Arm a is `fusion=projector` with the
# term OFF; arms b and c differ from it in exactly one number, the weight.
#
#   arm a   alignment_weight 0      the control
#   arm b   alignment_weight 0.1    a term worth about a tenth of the data loss
#   arm c   alignment_weight 1.0    a term as large as the data loss
#
# Two weights and not one: at 0.1 the alignment is a nudge, at 1.0 it competes with the
# prediction. If both move the validation score the same way, the direction is real; if
# only 1.0 moves it and downwards, the term is simply taking the capacity away from the
# task, which is also an answer.
#
# HOW TO READ IT. Per seed, b - a and c - a on validation/strict (the project's selection
# slice) - three paired differences, because a difference of two means carries no spread.
# The project's bar: 3/3 seeds of one sign AND t >= 3, with a noise floor of 0.0021.
#
# Then, from each manifest, `alignment_loss_last_epoch` beside
# `alignment_contribution_last_epoch` (the weight times the term, i.e. what was actually
# added to the loss) and the train loss. The ONLY value that can be read exactly is
# ln(batch): the term equals it when every similarity in the batch is equal. It does not
# run backwards - a number near ln(batch) does not prove a flat representation, and one
# above it does not prove the pairs are systematically reversed. What the numbers do settle
# is whether the term was heavy enough to matter: a run whose contribution is a rounding
# error beside the data loss did not test the hypothesis, whatever its validation did.
#
# WHAT IT CANNOT ANSWER. The term aligns a molecule with its own fragment; the protein
# branch has no per-row partner, so nothing here aligns the protein. And the weights are
# untuned - three points on one axis, not a search.
#
# COST. Nine training runs on SUB100 (100,000 train rows), each followed by the same
# checkpoint scored on the train partition. The alignment term adds no forward pass, so a
# run costs what a t9 run cost: about 16 minutes, so roughly 2 hours 30 minutes in all.

set -u -o pipefail

if [[ $# -lt 1 ]]; then
  echo "a GPU index is required, for example: bash scripts/server_run_alignment_2111.sh 2" >&2
  exit 2
fi
GPU="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG="$ROOT/runs/alignment_2111_gpu$GPU.log"
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
claim_gpu "$GPU" "$ROOT/.cache" "alignment_2111" || exit $?
export CUDA_VISIBLE_DEVICES="$GPU"

# ONE LOG HOLDS ONE BATCH: both the figure script and the table script refuse a log with
# two pids (check_diagnostics_manifests.pid_from_log). The previous log is kept, not
# deleted, with the pid in its name so two starts in one second keep both; and `mv` is
# checked, because carrying on would append to the old file after saying it was saved.
if [[ -s "$LOG" ]]; then
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

# The reference flag set, plus the two flags this batch needs and minus none. The
# alignment weight is NOT here: it is set per arm, and a flag in two places is set twice.
SETTINGS="--set selection_metric=pairwise_accuracy_macro --set selection_slice=strict"
SETTINGS+=" --set gpu_memory_limit_mib=${MEMORY_LIMIT:-4500} --set eval_batch_size=2048"
SETTINGS+=" --set model.dropout=0.5 --set learning_rate=9.86e-05 --set weight_decay=0.0"
# fusion=projector is a CONDITION of the term, not a variable: one shared width for every
# branch. fusion_dim is config.json's 256, written out so the batch is self-describing.
SETTINGS+=" --set model.fusion=projector --set model.fusion_dim=256"
RESIDUAL="--set model.head_block=residual --set model.residual_blocks=7"
RESIDUAL+=" --set model.residual_width=102 --set model.residual_hidden_factor=1.849"
RESIDUAL+=" --set model.residual_hidden_dropout=0.3936 --set model.residual_dropout=0.0"
MEASURE="--set measure_train_metrics=true"

SEEDS=(20261023 20261024 20261025)
ARMS=(a b c)
WEIGHTS=(0.0 0.1 1.0)
RUN_ID=""
LAST_STATUS=1

new_run_id() { RUN_ID="$1-$$"; }

checkpoint_of_last_train() {
  if [[ ${LAST_STATUS:-1} -ne 0 ]]; then
    return 0
  fi
  python "$ROOT/scripts/pick_train_checkpoint.py" "$ROOT/runs/evaluator" "$RUN_ID" \
    2>> "$LOG" || true
}

train_and_score() {
  local label="$1"
  shift
  say "$label - training (frozen, SUB100, fusion=projector)"
  new_run_id "$label"
  run python run_evaluator.py --mode train --run-id "$RUN_ID" --config frozen \
    --head direct $SETTINGS $RESIDUAL "$@"
  local checkpoint
  checkpoint="$(checkpoint_of_last_train)"
  if [[ -z "$checkpoint" && ${LAST_STATUS:-1} -eq 0 ]]; then
    # Trained, then no scorable checkpoint: one of the nine train sides is lost, and that
    # is a failure of the batch even though the training command returned 0.
    FAILED=$((FAILED + 1))
    echo "!! $label trained but produced NO usable checkpoint; its train half is skipped" \
      | tee -a "$LOG"
    return 0
  fi
  if [[ -z "$checkpoint" ]]; then
    echo "!! $label has no checkpoint because its training run failed" | tee -a "$LOG"
    return 0
  fi
  say "$label - the SAME checkpoint scored on TRAIN"
  new_run_id "$label-eval"
  run python run_evaluator.py --mode eval --run-id "$RUN_ID" --checkpoint "$checkpoint" \
    --partition train $SETTINGS "$@"
}

say "CLIP-style alignment on SUB100, 3 arms x 3 seeds, pid $$"
echo "one variable: alignment_weight in ${WEIGHTS[*]}; arm a (weight 0) is the control," \
  | tee -a "$LOG"
echo "and it is fusion=projector like every other arm, NOT the concat reference runs" \
  | tee -a "$LOG"

for index in 0 1 2; do
  arm="${ARMS[$index]}"
  weight="${WEIGHTS[$index]}"
  say "arm $arm - alignment_weight $weight, three seeds"
  for position in 0 1 2; do
    train_and_score "t11$arm$((position + 1))" \
      --set "inputs.train_subset=$SUB_SPLIT" \
      --set "seed=${SEEDS[$position]}" \
      --set "alignment_weight=$weight" $MEASURE
  done
done

say "figures and tables"
run python scripts/figures_for_batch.py runs/evaluator "$LOG"
run python scripts/diagnostics_tables.py runs/evaluator "$LOG"
run python scripts/diagnostics_tables.py runs/evaluator "$LOG" --slice all --train-slice all

say "FINISHED with $FAILED failed command(s)"
echo "read with:" | tee -a "$LOG"
echo "  grep -n 'FAILED\\|SKIPPED\\|alignment (InfoNCE)' $LOG" | tee -a "$LOG"
echo "  sed -n '/Every run of this batch/,\$p' $LOG" | tee -a "$LOG"
[[ $FAILED -ne 0 ]] && exit 1
exit 0
