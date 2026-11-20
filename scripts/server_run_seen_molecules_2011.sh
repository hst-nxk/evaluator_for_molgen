#!/usr/bin/env bash
#
# Is the PAIR75 score carried by rows whose molecules were already in training?
#
#   setsid nohup bash scripts/server_run_seen_molecules_2011.sh 1 t3a-<batch pid> \
#       > /dev/null 2>&1 &
#   tail -F runs/seen_molecules_2011_gpu1.log
#
# WHY. On PAIR75 the MLP reaches pairwise_accuracy_macro 0.7778 while the classical
# baselines on the SAME split reach 0.63-0.68 and everything on SUB100 collapses towards
# 0.53. PAIR75 is keyed on the PAIR, so no validation pair was in training - but a single
# MOLECULE of a validation pair may have been, inside another pair. If the score lives on
# those rows then 0.7778 is not a generalisation number. This runs the two row dumps the
# analysis needs (train and validation, same checkpoint, same split) and then the analysis.
#
# WHAT IT NEEDS. Either the RUN ID of the training task (`t3a-<batch pid>`), which this
# script resolves to a checkpoint through scripts/pick_train_checkpoint.py, or the path of
# a .pt file directly. The picker takes two POSITIONAL arguments and no flags:
#   python scripts/pick_train_checkpoint.py runs/evaluator t3a-4174688
# (`--task t3a --pid 4174688` is not its interface and is refused - that wrong form was in
# this header until 2026-11-20.)
#
# The split flag is passed to BOTH dumps: the partitions have to be the ones the checkpoint
# was trained against, or "seen in train" means a different train set. The protein shuffle
# is deliberately NOT passed - `--mode errors` refuses a checkpoint whose recorded
# protein_permute_seed differs from the command's (train.py), so a t3b checkpoint cannot be
# analysed by accident with this script.
#
# COST. Two forward passes over PAIR75 (75,000 train rows, 24,946 validation rows), no
# training: a few minutes.

set -u -o pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: bash scripts/server_run_seen_molecules_2011.sh <GPU index>"\
       "<run id, e.g. t3a-4174688, or a checkpoint .pt>" >&2
  exit 2
fi
GPU="$1"
TARGET="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG="$ROOT/runs/seen_molecules_2011_gpu$GPU.log"
PAIR_SPLIT="work/splits/pair_split_1411.parquet"
FAILED=0

mkdir -p "$ROOT/runs" "$ROOT/.cache"
source "$ROOT/scripts/gpu_batch_lock.sh"
claim_gpu "$GPU" "$ROOT/.cache" "seen_molecules_2011" || exit $?
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

# A run id is resolved here rather than by hand: the picker reads the task's own manifest,
# so the checkpoint cannot be a file from another task that happens to be lying around.
if [[ "$TARGET" == *.pt ]]; then
  CHECKPOINT="$TARGET"
else
  CHECKPOINT="$(python scripts/pick_train_checkpoint.py runs/evaluator "$TARGET")" || {
    echo "could not resolve the run id '$TARGET' to a checkpoint" | tee -a "$LOG"
    exit 3
  }
  echo "run id $TARGET -> $CHECKPOINT" | tee -a "$LOG"
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "no such checkpoint: $CHECKPOINT" | tee -a "$LOG"
  exit 3
fi
if [[ ! -f "$PAIR_SPLIT" ]]; then
  echo "missing $PAIR_SPLIT; build it first with scripts/build_pair_split.py" | tee -a "$LOG"
  exit 3
fi

SETTINGS="--set selection_metric=pairwise_accuracy_macro --set selection_slice=strict"
SETTINGS+=" --set gpu_memory_limit_mib=4500 --set eval_batch_size=2048"
SETTINGS+=" --set inputs.partition_override_pairs=$PAIR_SPLIT"

say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }

run() {
  echo "+ $*" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  local status=$?
  if [[ $status -ne 0 ]]; then
    FAILED=$((FAILED + 1))
    echo "!! FAILED (code $status) at: $*" | tee -a "$LOG"
  fi
  return 0
}

for partition in train validation; do
  say "row dump, $partition, $(basename "$CHECKPOINT")"
  run python run_evaluator.py --mode errors --run-id "t6$partition-$$" \
    --checkpoint "$CHECKPOINT" --partition "$partition" $SETTINGS
done

if [[ $FAILED -ne 0 ]]; then
  say "FINISHED WITH $FAILED FAILED COMMAND(S) - no analysis was run"
  exit 1
fi

# The dumps are found by the run id this script minted, so a dump of some older run cannot
# be analysed by mistake.
TRAIN_DUMP="$(ls -d runs/evaluator/errors_*_train_*t6train-$$ 2> /dev/null | head -1)"
VALIDATION_DUMP="$(ls -d runs/evaluator/errors_*_validation_*t6validation-$$ 2> /dev/null | head -1)"
if [[ -z "$TRAIN_DUMP" || -z "$VALIDATION_DUMP" ]]; then
  say "the dumps of this run are not where they were expected; nothing was analysed"
  echo "train='$TRAIN_DUMP' validation='$VALIDATION_DUMP'" | tee -a "$LOG"
  exit 1
fi

say "analysis - molecules seen in training"
# `--slice all` and not strict: on a pair-keyed split every validation pair is unseen, so
# the two slices hold the same rows (24,946 of 24,946 in the 2026-11-19 batch) and `all`
# says so plainly instead of implying a filter that filters nothing.
run python scripts/seen_molecules.py --train "$TRAIN_DUMP" \
  --validation "$VALIDATION_DUMP" --slice all --label "PAIR75 t3a" \
  --out "runs/evaluator/seen_molecules_t3a-$$.json"

if [[ $FAILED -ne 0 ]]; then
  say "THE ANALYSIS REFUSED ITS OWN OUTPUT - read the log; the breakdown must not be quoted"
  exit 1
fi
say "DONE"
echo "report: runs/evaluator/seen_molecules_t3a-$$.json" | tee -a "$LOG"
