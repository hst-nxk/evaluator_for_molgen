#!/usr/bin/env bash
#
# Do the classical baselines lose their score on new molecules the way the MLP does?
#
#   setsid nohup bash scripts/server_run_classical_seen_2011.sh \
#       runs/evaluator/classical_frozen_20261120_082629_t1c-4174688.json \
#       runs/evaluator/errors_frozen_direct_train_20261120_124613_t6train-3831959 \
#       > /dev/null 2>&1 &
#   tail -F runs/classical_seen_2011.log
#
# WHY. The breakdown of 2026-11-20 showed where the MLP's PAIR75 score lives: direction
# accuracy 0.8092 on rows whose two molecules both appeared in training, 0.5677 on rows
# where neither did, with the label magnitudes of the two groups the same. The MLP's macro
# for the unseen group is 0.6673, which sits inside the range the classical baselines
# reach overall (0.6267-0.6801) - but those are macros over DIFFERENT query sets (215
# queries against 3,892), so the two numbers do not compare and the advantage of +0.0977
# cannot be said to survive or to vanish. This scores the baselines on the same three
# groups, which is the comparison that settles it.
#
# WHAT IT NEEDS. The classical manifest of the PAIR75 batch, and the TRAIN row dump of the
# MLP run on the same split. The train dump is only used for the set of molecules seen in
# training, which is a property of the split and not of the model, so one dump serves
# every baseline as well as the MLP.
#
# NO GPU. Every model here is scikit-learn or xgboost on CPU. The feature matrix is the
# memory high-water mark, the same as in the original classical run, so this is the one
# thing to watch. It takes no GPU lock, so it can run beside a training batch.
#
# Each refit is checked against the metrics its manifest recorded BEFORE any dump is
# written (scripts/classical_row_dump.py); a model whose refit does not reproduce them is
# refused and does not reach the breakdown.

set -u -o pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: bash scripts/server_run_classical_seen_2011.sh"\
       "<classical manifest.json> <train row dump directory>" >&2
  exit 2
fi
MANIFEST="$1"
TRAIN_DUMP="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG="$ROOT/runs/classical_seen_2011.log"
FAILED=0
mkdir -p "$ROOT/runs"
export PYTHONUNBUFFERED=1

if [[ ! -f "$MANIFEST" ]]; then
  echo "no such manifest: $MANIFEST" | tee -a "$LOG"
  exit 3
fi
if [[ ! -f "$TRAIN_DUMP/rows.parquet" ]]; then
  echo "no rows.parquet in $TRAIN_DUMP; that has to be a --mode errors train dump" \
    | tee -a "$LOG"
  exit 3
fi

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

STAMP="classical_seen_$$"
say "refitting the baselines of $(basename "$MANIFEST") and dumping their validation rows"
run python scripts/classical_row_dump.py "$MANIFEST" --models all

# The dumps this run wrote, newest first: the script stamps them with the time it started,
# so the newest folders of each model are the ones it just produced.
for model in linear ridge lasso xgboost; do
  DUMP="$(ls -dt runs/evaluator/errors_classical_${model}_*_validation_* 2> /dev/null \
    | head -1)"
  if [[ -z "$DUMP" ]]; then
    echo "no dump for $model; it was refused or never fitted - see above" | tee -a "$LOG"
    continue
  fi
  say "breakdown - $model"
  run python scripts/seen_molecules.py --train "$TRAIN_DUMP" --validation "$DUMP" \
    --slice all --label "PAIR75 $model" \
    --out "runs/evaluator/seen_molecules_${model}_$$.json"
done

if [[ $FAILED -ne 0 ]]; then
  say "FINISHED WITH $FAILED FAILED COMMAND(S) - read the log before quoting anything"
  exit 1
fi
say "DONE"
echo "compare each model's 0/1/2 table with the MLP's: 0.6673 / 0.7103 / 0.8311" \
  | tee -a "$LOG"
